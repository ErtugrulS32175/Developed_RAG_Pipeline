"""The local large-v3 speech service: one closed model profile, closed
errors, nothing spoken or secret in the logs.

Nothing here loads faster-whisper, touches a GPU or opens a socket. The
model reaches the app through the same factory seam production uses, so
these tests prove WHAT the service asks of a model (large-v3 / cuda /
float16, language="tr", vad_filter=True) without ever owning one. The
real hardware run is manual evidence and lives outside this suite.
"""
import asyncio
from contextlib import asynccontextmanager
import dataclasses
import importlib.util
import inspect
import logging
import os
import socket
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from services import speech_service as speech

ROOT = Path(__file__).resolve().parent.parent
KEY = "test-speech-key-" + "x" * 24
AUTH = {"Authorization": f"Bearer {KEY}"}
URL = "/v1/audio/transcriptions"


# ---------------------------------------------------------------------
# guards and fakes
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A unit test that reaches the network or the Hub is not a unit test.

    Name resolution and outbound connects are refused. The one connect
    allowed is to loopback, because asyncio's Windows event loop builds
    its self-pipe from a loopback ``socketpair`` and would otherwise not
    start at all -- that is the loop, not the network."""
    real_connect = socket.socket.connect

    def refuse(*args, **kwargs):
        raise AssertionError("test sirasinda ag erisimi denendi")

    def loopback_only(sock, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else ""
        if host in ("127.0.0.1", "::1"):
            return real_connect(sock, address, *args, **kwargs)
        refuse()
    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


class FakeSegment:
    def __init__(self, text):
        self.text = text


class RecordingGenerator:
    """A segment generator that knows whether it was drained."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.yielded = 0
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.yielded >= len(self._texts):
            self.exhausted = True
            raise StopIteration
        text = self._texts[self.yielded]
        self.yielded += 1
        return FakeSegment(text)


class FakeModel:
    def __init__(self, texts=(" Bir", " iki", " uc"), *, block=None,
                 fail=None):
        self.texts = texts
        self.block = block          # threading.Event the worker waits on
        self.fail = fail
        self.calls = []
        self.generators = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self.block is not None:
            self.block.wait(timeout=10)
        if self.fail is not None:
            raise self.fail
        generator = RecordingGenerator(self.texts)
        self.generators.append(generator)
        return generator, SimpleNamespace(duration=1.0)


class Factory:
    def __init__(self, model=None, error=None):
        self.model = model if model is not None else FakeModel()
        self.error = error
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.model


def fake_probe(seconds=1.0, error=None):
    calls = []

    def probe(path, limit):
        calls.append((path, limit))
        if error is not None:
            raise error
        if seconds > limit:
            raise speech.AudioTooLong()
        return seconds
    probe.calls = calls
    return probe


def make_app(monkeypatch, tmp_path, *, factory=None, probe=None,
             limits=None):
    monkeypatch.setenv("SPEECH_API_KEY", KEY)
    settings = speech.load_settings(os.environ)
    return speech.create_app(
        settings,
        model_factory=factory if factory is not None else Factory(),
        probe=probe if probe is not None else fake_probe(),
        limits=limits if limits is not None else speech.DEFAULT_LIMITS,
        upload_dir=tmp_path,
    )


def upload(client, *, data=b"RIFF-kurgu-ses-baytlari", name="dikte.webm",
           content_type="audio/webm", fields=None, headers=AUTH):
    return client.post(
        URL, files={"file": (name, data, content_type)},
        data=fields or {}, headers=headers)


def _closed_error(response, code):
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"] == {
        "version": 1,
        "code": code,
        "request_id": response.headers["X-Request-ID"],
    }
    assert response.status_code == speech.ERROR_STATUS_BY_CODE[code]
    return body


# ---------------------------------------------------------------------
# the closed profile
# ---------------------------------------------------------------------

def test_the_profile_and_limits_are_pinned():
    assert speech.MODEL_NAME == "large-v3"
    assert speech.DEVICE == "cuda"
    assert speech.COMPUTE_TYPE == "float16"
    assert speech.LANGUAGE == "tr"
    assert speech.VAD_FILTER is True
    assert speech.DEFAULT_BIND_HOST == "0.0.0.0"
    assert speech.DEFAULT_PORT == 8012
    assert speech.MIN_API_KEY_CHARS == 32
    # first local dictation profile; E5 quota policy takes these over
    assert speech.DEFAULT_LIMITS == speech.Limits(
        max_upload_bytes=25 * 1024 * 1024,
        multipart_overhead_bytes=64 * 1024,
        max_audio_seconds=120.0,
        transcribe_timeout_seconds=90.0,
        queue_wait_seconds=20.0,
        max_waiting=2,
        upload_chunk_bytes=64 * 1024,
    )
    # mp4 is what Safari's MediaRecorder names an AAC dictation
    assert speech.ALLOWED_EXTENSIONS == frozenset(
        {"wav", "webm", "mp3", "mp4", "m4a", "ogg", "flac"})
    assert dict(speech.ERROR_STATUS_BY_CODE) == {
        "invalid_request": 400,
        "invalid_audio": 400,
        "authentication_required": 401,
        "resource_not_found": 404,
        "method_not_allowed": 405,
        "payload_too_large": 413,
        "audio_too_long": 413,
        "unsupported_media_type": 415,
        "validation_failed": 422,
        "internal_error": 500,
        "service_unavailable": 503,
        "speech_busy": 503,
        "speech_timeout": 504,
    }


def test_the_production_factory_carries_exactly_the_closed_profile(
        monkeypatch):
    calls = []

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=WhisperModel))
    model = speech.production_model_factory()
    assert isinstance(model, WhisperModel)
    assert calls == [(("large-v3",), {"device": "cuda",
                                      "compute_type": "float16"})]
    default = inspect.signature(speech.create_app).parameters["model_factory"]
    assert default.default is speech.production_model_factory


def test_the_model_is_loaded_once_for_the_life_of_the_service(
        monkeypatch, tmp_path):
    factory = Factory()
    app = make_app(monkeypatch, tmp_path, factory=factory)
    with TestClient(app) as client:
        assert factory.calls == 1
        assert upload(client).status_code == 200
        assert upload(client).status_code == 200
        assert client.get("/readyz").status_code == 200
    assert factory.calls == 1


def test_transcribe_receives_exactly_language_tr_and_vad_filter(
        monkeypatch, tmp_path):
    model = FakeModel()
    app = make_app(monkeypatch, tmp_path, factory=Factory(model))
    with TestClient(app) as client:
        assert upload(client, name="ozel-ad.webm").status_code == 200
    (audio, kwargs), = model.calls
    assert kwargs == {"language": "tr", "vad_filter": True}
    assert isinstance(audio, str)
    assert Path(audio).parent == tmp_path
    assert audio.endswith(speech.UPLOAD_SUFFIX)
    assert "ozel-ad" not in audio       # the user's name is never a path


def test_a_successful_response_carries_only_text(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = upload(client)
    assert response.status_code == 200
    assert response.json() == {"text": "Bir iki uc"}
    assert len(response.headers["X-Request-ID"]) == 8


def test_the_segment_generator_is_drained_in_order(monkeypatch, tmp_path):
    model = FakeModel(texts=(" bir", " iki", " uc", " dort", " bes"))
    app = make_app(monkeypatch, tmp_path, factory=Factory(model))
    with TestClient(app) as client:
        response = upload(client)
    assert response.json() == {"text": "bir iki uc dort bes"}
    generator, = model.generators
    assert generator.exhausted is True
    assert generator.yielded == 5


def test_transcripts_are_joined_without_adding_or_normalising_words(
        monkeypatch, tmp_path):
    model = FakeModel(texts=(" Sayin", " yetkili,", "  47 000 adet."))
    app = make_app(monkeypatch, tmp_path, factory=Factory(model))
    with TestClient(app) as client:
        response = upload(client)
    assert response.json() == {"text": "Sayin yetkili,  47 000 adet."}


# ---------------------------------------------------------------------
# service identity
# ---------------------------------------------------------------------

def test_the_key_is_required_and_bounded_at_startup(monkeypatch):
    monkeypatch.delenv("SPEECH_API_KEY", raising=False)
    with pytest.raises(speech.SpeechConfigError):
        speech.load_settings(os.environ)
    monkeypatch.setenv("SPEECH_API_KEY", "")
    with pytest.raises(speech.SpeechConfigError):
        speech.load_settings(os.environ)
    monkeypatch.setenv("SPEECH_API_KEY", "kisa")
    with pytest.raises(speech.SpeechConfigError):
        speech.load_settings(os.environ)
    monkeypatch.setenv("SPEECH_API_KEY", KEY)
    settings = speech.load_settings(os.environ)
    assert (settings.host, settings.port) == ("0.0.0.0", 8012)
    assert KEY not in repr(settings)
    assert KEY not in str(settings)


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer "},
    {"Authorization": "Bearer"},
    {"Authorization": "Bearer yanlis-anahtar-" + "y" * 24},
    {"Authorization": "Bearer " + KEY[:-1]},
    {"Authorization": "Basic " + KEY},
    {"Authorization": KEY},
], ids=["yok", "bos", "sadece-sema", "yanlis", "kisaltilmis", "basic",
        "semasiz"])
def test_a_missing_empty_or_wrong_key_is_refused_closed(
        monkeypatch, tmp_path, headers):
    model = FakeModel()
    probe = fake_probe()
    app = make_app(monkeypatch, tmp_path, factory=Factory(model), probe=probe)
    with TestClient(app) as client:
        response = upload(client, headers=headers)
    _closed_error(response, "authentication_required")
    assert model.calls == []
    assert probe.calls == []
    assert list(tmp_path.iterdir()) == []


def test_the_comparison_is_timing_safe(monkeypatch):
    seen = []
    import hmac

    def compare(a, b):
        seen.append((a, b))
        return False
    monkeypatch.setattr(hmac, "compare_digest", compare)
    assert speech.key_matches("sunulan", "beklenen") is False
    assert seen == [(b"sunulan", b"beklenen")]


# ---------------------------------------------------------------------
# request validation, before anything expensive
# ---------------------------------------------------------------------

def test_a_model_other_than_large_v3_is_refused(monkeypatch, tmp_path):
    model = FakeModel()
    app = make_app(monkeypatch, tmp_path, factory=Factory(model))
    with TestClient(app) as client:
        for wrong in ("whisper-1", "large-v3-turbo", "Large-V3", ""):
            response = upload(client, fields={"model": wrong})
            _closed_error(response, "invalid_request")
        assert upload(client, fields={"model": "large-v3"}).status_code == 200
    assert len(model.calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_a_language_other_than_tr_is_refused(monkeypatch, tmp_path):
    model = FakeModel()
    app = make_app(monkeypatch, tmp_path, factory=Factory(model))
    with TestClient(app) as client:
        for wrong in ("en", "TR", "tr-TR", ""):
            response = upload(client, fields={"language": wrong})
            _closed_error(response, "invalid_request")
        assert upload(client, fields={"language": "tr"}).status_code == 200
    assert len(model.calls) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name,content_type", [
    ("notlar.txt", "text/plain"),
    ("dikte.webm", "text/plain"),
    ("dikte.exe", "audio/webm"),
    ("dikte", "audio/webm"),
    ("dikte.mp4", "video/mp4"),
    ("dikte.webm", "application/json"),
], ids=["txt", "webm-adli-text", "exe", "uzantisiz", "mp4", "json-tipli"])
def test_an_unsupported_format_is_refused_before_staging(
        monkeypatch, tmp_path, name, content_type):
    model = FakeModel()
    probe = fake_probe()
    app = make_app(monkeypatch, tmp_path, factory=Factory(model), probe=probe)
    with TestClient(app) as client:
        response = upload(client, name=name, content_type=content_type)
    _closed_error(response, "unsupported_media_type")
    assert model.calls == []
    assert probe.calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name,content_type", [
    ("dikte.wav", "audio/wav"),
    ("dikte.wav", "audio/x-wav"),
    ("dikte.webm", "audio/webm"),
    ("dikte.webm", "video/webm"),
    ("dikte.mp3", "audio/mpeg"),
    ("dikte.m4a", "audio/mp4"),
    ("dikte.m4a", "audio/x-m4a"),
    ("dikte.mp4", "audio/mp4"),
    ("dikte.ogg", "audio/ogg"),
    ("dikte.flac", "audio/flac"),
    # OpenWebUI v0.11.0 streams the part through aiohttp without a type,
    # which arrives as octet-stream; the extension is the signal then
    ("kayit.webm", "application/octet-stream"),
    ("kayit.mp4", "application/octet-stream"),
    ("KAYIT.WAV", "application/octet-stream"),
])
def test_every_documented_format_is_accepted(
        monkeypatch, tmp_path, name, content_type):
    app = make_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = upload(client, name=name, content_type=content_type)
    assert response.status_code == 200, response.text


def test_an_empty_upload_is_refused_before_the_model(monkeypatch, tmp_path):
    model = FakeModel()
    probe = fake_probe()
    app = make_app(monkeypatch, tmp_path, factory=Factory(model), probe=probe)
    with TestClient(app) as client:
        response = upload(client, data=b"")
    _closed_error(response, "invalid_audio")
    assert model.calls == []
    assert probe.calls == []
    assert list(tmp_path.iterdir()) == []


def test_a_missing_file_field_is_a_closed_validation_error(
        monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post(URL, data={"model": "large-v3"}, headers=AUTH)
    _closed_error(response, "validation_failed")


# one byte over the file ceiling (caught while staging) and one byte over
# the whole-body ceiling (caught in the gate before any parsing)
OVER_FILE = speech.DEFAULT_LIMITS.max_upload_bytes + 1
OVER_BODY = (speech.DEFAULT_LIMITS.max_upload_bytes
             + speech.DEFAULT_LIMITS.multipart_overhead_bytes + 1)


@pytest.mark.parametrize("size", [OVER_FILE, OVER_BODY],
                         ids=["dosya-siniri", "govde-siniri"])
def test_a_byte_limit_breach_with_content_length_is_refused_before_the_model(
        monkeypatch, tmp_path, size):
    model = FakeModel()
    probe = fake_probe()
    app = make_app(monkeypatch, tmp_path, factory=Factory(model), probe=probe)
    with TestClient(app) as client:
        response = upload(client, data=b"x" * size)
    _closed_error(response, "payload_too_large")
    assert model.calls == []
    assert probe.calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("size", [OVER_FILE, OVER_BODY],
                         ids=["dosya-siniri", "govde-siniri"])
def test_a_byte_limit_breach_without_content_length_is_refused_while_streaming(
        monkeypatch, tmp_path, size):
    """OpenWebUI streams the part through aiohttp, so there is no
    Content-Length to pre-check: the body has to be bounded as it arrives."""
    model = FakeModel()
    probe = fake_probe()
    app = make_app(monkeypatch, tmp_path, factory=Factory(model), probe=probe)
    boundary = "speech-sinir-testi"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="dikte.webm"'
        "\r\nContent-Type: audio/webm\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    payload = b"x" * size

    def chunks():
        yield head
        step = 1 << 20
        for start in range(0, len(payload), step):
            yield payload[start:start + step]
        yield tail

    with TestClient(app) as client:
        response = client.post(
            URL, content=chunks(),
            headers={**AUTH,
                     "Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert "content-length" not in {k.lower() for k in response.request.headers}
    _closed_error(response, "payload_too_large")
    assert model.calls == []
    assert probe.calls == []
    assert list(tmp_path.iterdir()) == []


def test_a_duration_breach_is_refused_before_the_model(monkeypatch, tmp_path):
    model = FakeModel()
    probe = fake_probe(seconds=120.5)
    app = make_app(monkeypatch, tmp_path, factory=Factory(model), probe=probe)
    with TestClient(app) as client:
        response = upload(client)
    _closed_error(response, "audio_too_long")
    assert model.calls == []
    (path, limit), = probe.calls
    assert limit == speech.DEFAULT_LIMITS.max_audio_seconds == 120.0
    assert not Path(path).exists()
    assert list(tmp_path.iterdir()) == []


def test_undecodable_audio_is_a_closed_400(monkeypatch, tmp_path):
    model = FakeModel()
    probe = fake_probe(error=speech.InvalidAudio())
    app = make_app(monkeypatch, tmp_path, factory=Factory(model), probe=probe)
    with TestClient(app) as client:
        response = upload(client)
    _closed_error(response, "invalid_audio")
    assert model.calls == []
    assert list(tmp_path.iterdir()) == []


def _fake_av(monkeypatch, *, duration, frames, audio=True, open_error=None,
             decode_error=None):
    """A stand-in for PyAV shaped like the parts the probe touches. A
    browser's streamed WebM states no duration, which is the path every
    OpenWebUI dictation takes, so it is pinned here without real bytes."""
    state = {"decoded": 0, "closed": False, "level": "unset"}

    class Frame:
        def __init__(self, seconds):
            self.sample_rate = 16000
            self.samples = int(seconds * 16000)

    class Container:
        streams = [SimpleNamespace(type="audio")] if audio else [
            SimpleNamespace(type="video")]

        def __init__(self):
            self.duration = duration

        def decode(self, stream):
            for seconds in frames:
                if decode_error is not None and state["decoded"] == 1:
                    raise decode_error
                state["decoded"] += 1
                yield Frame(seconds)

        def close(self):
            state["closed"] = True

    def open_(path, mode="r"):
        if open_error is not None:
            raise open_error
        return Container()

    fake = types.SimpleNamespace(
        open=open_,
        logging=types.SimpleNamespace(
            set_level=lambda level: state.__setitem__("level", level)))
    monkeypatch.setitem(sys.modules, "av", fake)
    return state


def test_the_probe_decodes_boundedly_when_the_container_states_no_duration(
        monkeypatch):
    # header known: answered without decoding a frame
    state = _fake_av(monkeypatch, duration=2_500_000, frames=[1.0] * 10)
    assert speech.probe_audio_seconds("x", 120.0) == 2.5
    assert state["decoded"] == 0 and state["closed"] is True
    assert state["level"] is None          # ffmpeg diagnostics silenced
    state = _fake_av(monkeypatch, duration=121_000_000, frames=[])
    with pytest.raises(speech.AudioTooLong):
        speech.probe_audio_seconds("x", 120.0)
    # header unknown: decoded, and the decode STOPS at the limit
    state = _fake_av(monkeypatch, duration=None, frames=[1.0] * 3600)
    with pytest.raises(speech.AudioTooLong):
        speech.probe_audio_seconds("x", 2.5)
    assert state["decoded"] == 3 and state["closed"] is True
    state = _fake_av(monkeypatch, duration=None, frames=[0.5, 0.5, 0.25])
    assert speech.probe_audio_seconds("x", 120.0) == 1.25
    assert state["decoded"] == 3
    # not audio, undecodable, unopenable: closed refusals, vendor text dropped
    _fake_av(monkeypatch, duration=None, frames=[1.0], audio=False)
    with pytest.raises(speech.InvalidAudio):
        speech.probe_audio_seconds("x", 120.0)
    _fake_av(monkeypatch, duration=None, frames=[1.0, 1.0],
             decode_error=RuntimeError("C:/gizli/yol.upload bozuk"))
    with pytest.raises(speech.InvalidAudio) as caught:
        speech.probe_audio_seconds("x", 120.0)
    assert "gizli" not in repr(caught.value) and caught.value.__cause__ is None
    _fake_av(monkeypatch, duration=None, frames=[],
             open_error=OSError("C:/gizli/yol.upload"))
    with pytest.raises(speech.InvalidAudio) as caught:
        speech.probe_audio_seconds("x", 120.0)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_the_real_probe_measures_a_wav_and_rejects_garbage(tmp_path):
    """Runs only where the speech runtime is installed; the unit suite
    proves the seam, this proves the production probe behind it."""
    pytest.importorskip("av")
    import wave
    path = tmp_path / "sessiz.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * 2)      # two seconds
    seconds = speech.probe_audio_seconds(str(path), 120.0)
    assert 1.9 <= seconds <= 2.1
    with pytest.raises(speech.AudioTooLong):
        speech.probe_audio_seconds(str(path), 1.0)
    garbage = tmp_path / "bozuk.webm"
    garbage.write_bytes(b"bu bir ses dosyasi degil " * 40)
    with pytest.raises(speech.InvalidAudio):
        speech.probe_audio_seconds(str(garbage), 120.0)


# ---------------------------------------------------------------------
# nothing leaks
# ---------------------------------------------------------------------

def test_nothing_secret_or_spoken_reaches_the_logs(
        monkeypatch, tmp_path, caplog, capsys):
    audio_sentinel = b"SES-NOBETCI-" + b"q" * 20
    transcript_sentinel = "TRANSKRIPT-NOBETCI-" + "w" * 20
    name_sentinel = "DOSYA-ADI-NOBETCI"
    error_sentinel = "HATA-METNI-NOBETCI"
    model = FakeModel(texts=(transcript_sentinel,))
    app = make_app(monkeypatch, tmp_path, factory=Factory(model),
                   probe=fake_probe())
    caplog.set_level(logging.DEBUG)
    with TestClient(app, raise_server_exceptions=False) as client:
        ok = upload(client, data=audio_sentinel, name=f"{name_sentinel}.webm")
        assert ok.json() == {"text": transcript_sentinel}
        refused = upload(client, data=audio_sentinel,
                         name=f"{name_sentinel}.webm",
                         headers={"Authorization": "Bearer " + KEY[:-3]})
        assert refused.status_code == 401
        model.fail = RuntimeError(error_sentinel)
        failed = upload(client, data=audio_sentinel,
                        name=f"{name_sentinel}.webm")
        _closed_error(failed, "internal_error")
        model.fail = None
        bad = make_app(monkeypatch, tmp_path, factory=Factory(FakeModel()),
                       probe=fake_probe(error=speech.InvalidAudio()))
        with TestClient(bad) as bad_client:
            invalid = upload(bad_client, data=audio_sentinel,
                             name=f"{name_sentinel}.webm")
            _closed_error(invalid, "invalid_audio")
    captured = capsys.readouterr()
    haystacks = [caplog.text, captured.out, captured.err,
                 refused.text, failed.text, invalid.text]
    for needle in (KEY, audio_sentinel.decode(), transcript_sentinel,
                   name_sentinel, error_sentinel):
        for haystack in haystacks:
            assert needle not in haystack
    assert caplog.records, "servis hicbir olay kaydi uretmedi"
    assert list(tmp_path.iterdir()) == []


def test_a_model_load_failure_closes_readiness_without_content(
        monkeypatch, tmp_path, caplog):
    secret_path = "C:/gizli/model/yolu"
    factory = Factory(error=RuntimeError(f"cannot open {secret_path}"))
    model = factory.model
    app = make_app(monkeypatch, tmp_path, factory=factory)
    caplog.set_level(logging.DEBUG)
    with TestClient(app) as client:
        ready = client.get("/readyz")
        _closed_error(ready, "service_unavailable")
        assert client.get("/healthz").json() == {"status": "ok"}
        response = upload(client)
        _closed_error(response, "service_unavailable")
    assert factory.calls == 1
    assert model.calls == []
    assert "gizli" not in caplog.text
    assert "gizli" not in ready.text
    assert "RuntimeError" in caplog.text     # the type is the whole story
    assert list(tmp_path.iterdir()) == []


def test_healthz_says_only_that_the_process_is_alive(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path)
    client = TestClient(app)                 # no lifespan: nothing loaded
    assert client.get("/healthz").json() == {"status": "ok"}
    _closed_error(client.get("/readyz"), "service_unavailable")


def test_readyz_publishes_the_closed_profile(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "language": "tr",
    }


def test_unknown_routes_and_methods_answer_in_the_same_closed_shape(
        monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        _closed_error(client.get("/v1/models"), "resource_not_found")
        _closed_error(client.get(URL), "method_not_allowed")


# ---------------------------------------------------------------------
# the temporary file and the single GPU slot
# ---------------------------------------------------------------------

def test_the_temporary_file_is_removed_on_success_and_failure(
        monkeypatch, tmp_path):
    probe = fake_probe()
    app = make_app(monkeypatch, tmp_path, probe=probe)
    with TestClient(app) as client:
        assert upload(client).status_code == 200
        (path, _), = probe.calls
        assert Path(path).parent == tmp_path
        assert not Path(path).exists()
        failing = make_app(monkeypatch, tmp_path,
                           factory=Factory(FakeModel(fail=RuntimeError("x"))))
        with TestClient(failing, raise_server_exceptions=False) as bad:
            _closed_error(upload(bad), "internal_error")
    assert list(tmp_path.iterdir()) == []


async def _wait_until(condition, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("beklenen durum olusmadi")
        await asyncio.sleep(0.01)


async def _post(client):
    return await client.post(
        URL, files={"file": ("dikte.webm", b"RIFF-kurgu", "audio/webm")},
        headers=AUTH)


@asynccontextmanager
async def _running(app):
    """Drive the ASGI lifespan protocol by hand: one loop for the app, the
    client and the assertions, so concurrency is real and deterministic."""
    inbox = asyncio.Queue()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def receive():
        return await inbox.get()

    async def send(message):
        if message["type"] == "lifespan.startup.complete":
            started.set()
        elif message["type"] == "lifespan.startup.failed":
            raise AssertionError("lifespan startup failed")
        elif message["type"] == "lifespan.shutdown.complete":
            stopped.set()

    task = asyncio.create_task(app(
        {"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}},
        receive, send))
    await inbox.put({"type": "lifespan.startup"})
    await started.wait()
    try:
        yield
    finally:
        await inbox.put({"type": "lifespan.shutdown"})
        await stopped.wait()
        await task


def _scenario(app, body):
    async def run():
        async with _running(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                    transport=transport, base_url="http://speech") as client:
                await body(client)
    asyncio.run(run())


def test_a_cancelled_waiter_removes_its_own_file(monkeypatch, tmp_path):
    release = threading.Event()
    model = FakeModel(block=release)
    limits = dataclasses.replace(speech.DEFAULT_LIMITS,
                                 queue_wait_seconds=5.0, max_waiting=2)
    app = make_app(monkeypatch, tmp_path, factory=Factory(model),
                   limits=limits)
    engine = app.state.engine

    async def body(client):
        first = asyncio.create_task(_post(client))
        await _wait_until(lambda: len(model.calls) == 1)
        second = asyncio.create_task(_post(client))
        await _wait_until(lambda: engine.waiting == 1)
        assert len(list(tmp_path.iterdir())) == 2
        second.cancel()
        await asyncio.gather(second, return_exceptions=True)
        await _wait_until(lambda: len(list(tmp_path.iterdir())) == 1)
        assert engine.waiting == 0
        release.set()
        assert (await first).status_code == 200
        await _wait_until(lambda: list(tmp_path.iterdir()) == [])

    _scenario(app, body)
    assert len(model.calls) == 1


def test_a_timed_out_job_still_releases_the_slot_and_its_file(
        monkeypatch, tmp_path):
    release = threading.Event()
    model = FakeModel(block=release)
    limits = dataclasses.replace(speech.DEFAULT_LIMITS,
                                 transcribe_timeout_seconds=0.2,
                                 queue_wait_seconds=0.1, max_waiting=1)
    app = make_app(monkeypatch, tmp_path, factory=Factory(model),
                   limits=limits)

    async def body(client):
        timed_out = await _post(client)
        _closed_error(timed_out, "speech_timeout")
        assert len(list(tmp_path.iterdir())) == 1   # the job still runs
        busy = await _post(client)
        _closed_error(busy, "speech_busy")
        release.set()
        await _wait_until(lambda: list(tmp_path.iterdir()) == [])
        after = await _post(client)
        assert after.status_code == 200

    _scenario(app, body)
    assert len(model.calls) == 2


def test_a_second_concurrent_job_waits_boundedly_and_never_piles_up(
        monkeypatch, tmp_path):
    release = threading.Event()
    model = FakeModel(block=release)
    limits = dataclasses.replace(speech.DEFAULT_LIMITS,
                                 queue_wait_seconds=0.3, max_waiting=1)
    app = make_app(monkeypatch, tmp_path, factory=Factory(model),
                   limits=limits)
    engine = app.state.engine

    async def body(client):
        first = asyncio.create_task(_post(client))
        await _wait_until(lambda: len(model.calls) == 1)
        waiter = asyncio.create_task(_post(client))
        await _wait_until(lambda: engine.waiting == 1)
        overflow = await _post(client)          # queue is full: refused now
        _closed_error(overflow, "speech_busy")
        assert not waiter.done()
        _closed_error(await waiter, "speech_busy")   # bounded wait expired
        assert len(model.calls) == 1            # nothing ran concurrently
        release.set()
        assert (await first).status_code == 200
        assert (await _post(client)).status_code == 200
        await _wait_until(lambda: list(tmp_path.iterdir()) == [])

    _scenario(app, body)
    assert len(model.calls) == 2


# ---------------------------------------------------------------------
# contract surface and deployment
# ---------------------------------------------------------------------

def test_openapi_response_and_error_schemas_are_closed(monkeypatch, tmp_path):
    spec = make_app(monkeypatch, tmp_path).openapi()
    components = spec["components"]["schemas"]
    assert "HTTPValidationError" not in components
    assert "ValidationError" not in components
    for name, schema in components.items():
        if name.startswith("Body_"):
            continue        # the multipart REQUEST shape FastAPI derives
        assert schema.get("additionalProperties") is False, name
    assert set(spec["paths"]) == {"/healthz", "/readyz", URL}
    for path, methods in spec["paths"].items():
        for method, operation in methods.items():
            for status, response in operation["responses"].items():
                ref = response["content"]["application/json"]["schema"]["$ref"]
                assert ref.startswith("#/components/schemas/"), (path, status)
    codes = components["ErrorEnvelope"]["properties"]["code"]["enum"]
    assert set(codes) == set(speech.ERROR_STATUS_BY_CODE)
    ready = components["ReadyResponse"]["properties"]
    assert ready["model"]["const"] == "large-v3"
    assert ready["device"]["const"] == "cuda"
    assert ready["compute_type"]["const"] == "float16"
    assert ready["language"]["const"] == "tr"
    assert set(components["TranscriptionResponse"]["properties"]) == {"text"}


def test_no_speech_runtime_is_imported_by_the_unit_suite(monkeypatch, tmp_path):
    before = set(sys.modules)
    app = make_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert upload(client).status_code == 200
    newly_imported = set(sys.modules) - before
    assert not ({"faster_whisper", "ctranslate2", "av"} & newly_imported)
    assert not any(name.startswith("huggingface_hub") for name in newly_imported)


def test_main_serves_one_worker_on_the_documented_bind(monkeypatch):
    recorded = {}

    def run(app, **kwargs):
        recorded["app"] = app
        recorded.update(kwargs)
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=run))
    monkeypatch.setenv("SPEECH_API_KEY", KEY)
    monkeypatch.setenv("SPEECH_PORT", "8012")
    assert speech.main() == 0
    assert recorded["host"] == "0.0.0.0"
    assert recorded["port"] == 8012
    assert recorded["workers"] == 1
    assert recorded["app"].state.engine is not None


def test_the_module_entrypoint_resolves_and_fails_closed_without_a_key():
    assert importlib.util.find_spec("services.speech_service") is not None
    env = {k: v for k, v in os.environ.items() if k != "SPEECH_API_KEY"}
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-m", "services.speech_service"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
    assert "SPEECH_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_speech_override_adds_only_stt_variables_to_open_webui():
    override = (ROOT / "docker-compose.speech.yml").read_text(encoding="utf-8")
    base = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    lines = [line for line in override.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert lines[:3] == ["services:", "  open-webui:", "    environment:"]
    body = lines[3:]
    assert body == [
        "      AUDIO_STT_ENGINE: openai",
        "      AUDIO_STT_OPENAI_API_BASE_URL: http://host.docker.internal:8012/v1",
        "      AUDIO_STT_OPENAI_API_KEY: ${SPEECH_API_KEY:?set SPEECH_API_KEY}",
        "      AUDIO_STT_MODEL: large-v3",
        "      AUDIO_STT_OPENAI_API_REQUEST_FORMAT: multipart",
        "      AUDIO_STT_SUPPORTED_CONTENT_TYPES: "
        + speech.OPENWEBUI_SUPPORTED_CONTENT_TYPES,
    ]
    for forbidden in ("image:", "ports:", "volumes:", "command:",
                      "profiles:", "extra_hosts:"):
        assert forbidden not in override
    assert "AUDIO_STT_" not in base
    assert "docker-compose.speech.yml" not in base
    assert "\t" not in override


def test_the_request_body_documents_exactly_the_multipart_fields(
        monkeypatch, tmp_path):
    """The two Form parameters exist for this document: OpenWebUI sends
    `model` (and sometimes `language`) beside `file`, and the contract
    says so. Validation reads the raw form, so this pin is what keeps the
    parameters from being dead weight."""
    spec = make_app(monkeypatch, tmp_path).openapi()
    body = spec["paths"][URL]["post"]["requestBody"]
    assert body["required"] is True
    ref = body["content"]["multipart/form-data"]["schema"]["$ref"]
    schema = spec["components"]["schemas"][ref.rsplit("/", 1)[1]]
    assert set(schema["properties"]) == {"file", "model", "language"}
    assert schema["required"] == ["file"]
    # OpenAPI 3.1: a binary part is a string with a content media type
    assert schema["properties"]["file"]["type"] == "string"
    assert (schema["properties"]["file"]["contentMediaType"]
            == "application/octet-stream")
    for optional in ("model", "language"):
        assert {"type": "string"} in schema["properties"][optional]["anyOf"]


def test_the_env_example_declares_the_speech_key_without_a_value():
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    assert values["SPEECH_API_KEY"] == ""
    compose = (ROOT / "docker-compose.speech.yml").read_text(encoding="utf-8")
    assert "${SPEECH_API_KEY:?" in compose


def test_the_speech_requirements_stay_out_of_the_main_tree():
    speech_requirements = (ROOT / "requirements-speech.txt").read_text(
        encoding="utf-8")
    main_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "faster-whisper" in speech_requirements
    for heavy in ("faster-whisper", "ctranslate2", "whisperx", "pyannote",
                  "redis", "celery", "kafka"):
        assert heavy not in main_requirements.lower()
    for absent in ("whisperx", "pyannote", "redis", "celery", "kafka",
                   "opentelemetry", "boto3", "minio"):
        assert absent not in speech_requirements.lower()


def test_the_readme_documents_the_gpu_operation_order():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Local speech dictation in OpenWebUI", 1)[1]
    section = section.split("\n### ", 1)[0]
    for fact in (
            "python -m services.speech_service",
            "docker-compose.speech.yml",
            "SPEECH_API_KEY",
            "large-v3",
            "float16",
            "1. Do not run the embed and reranker containers",
            "2. Stop the speech service",
            "3. Start the embed and reranker",
            "25 MiB",
            "120 seconds",
            "E5"):
        assert fact in section, fact
