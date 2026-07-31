import base64
import json
import time

import pytest

from pipeline.api import owui_chat


def _img_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


def _data_url(payload=b"fake-png-bytes", mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(payload).decode()


# --- message parsing ---

def test_message_text_reads_plain_string_turn():
    assert owui_chat.message_text("merhaba") == "merhaba"


def test_message_text_joins_text_parts_and_ignores_images():
    content = [{"type": "text", "text": "tabloyu"}, _img_part(_data_url()),
               {"type": "text", "text": "cikar"}]
    assert owui_chat.message_text(content) == "tabloyu\ncikar"


def test_message_image_urls_empty_for_plain_string():
    assert owui_chat.message_image_urls("merhaba") == []


def test_latest_image_url_found_in_earlier_turn():
    """A follow-up question carries no image, but the conversation still has one."""
    messages = [
        {"role": "user", "content": [_img_part("data:image/png;base64,AAA")]},
        {"role": "assistant", "content": "tablo..."},
        {"role": "user", "content": "peki toplami ne"},
    ]
    assert owui_chat.latest_image_url(messages) == "data:image/png;base64,AAA"


def test_latest_image_url_prefers_the_most_recent_image():
    messages = [
        {"role": "user", "content": [_img_part("data:image/png;base64,OLD")]},
        {"role": "user", "content": [_img_part("data:image/png;base64,NEW")]},
    ]
    assert owui_chat.latest_image_url(messages) == "data:image/png;base64,NEW"


def test_latest_image_url_none_when_no_image_anywhere():
    assert owui_chat.latest_image_url([{"role": "user", "content": "selam"}]) is None


def test_latest_image_url_accepts_objects_with_content_attribute():
    class Msg:
        def __init__(self, content):
            self.content = content

    assert owui_chat.latest_image_url([Msg([_img_part("data:image/png;base64,X")])]) \
        == "data:image/png;base64,X"


# --- data url decoding ---

def test_decode_data_url_returns_bytes_and_extension():
    raw, ext = owui_chat.decode_data_url(_data_url(b"hello", "image/jpeg"))
    assert raw == b"hello"
    assert ext == "jpg"


def test_decode_data_url_unknown_mime_falls_back_to_png():
    _, ext = owui_chat.decode_data_url(_data_url(b"x", "image/heic"))
    assert ext == "png"


def test_decode_data_url_rejects_plain_http_url():
    assert owui_chat.decode_data_url("https://example.com/a.png") is None


# --- extraction + caching ---

def test_extract_tables_caches_by_image_content(tmp_path, monkeypatch):
    """The same image must not pay for a second multi-minute model run."""
    calls = []

    def fake_consensus(path):
        calls.append(path)
        return [{"headers": ["A"], "rows": [["1"]], "confidence": 1.0}]

    monkeypatch.setattr(owui_chat, "run_consensus", fake_consensus)
    monkeypatch.setattr(owui_chat, "export_result_xlsx", lambda r, p: None)
    monkeypatch.setattr(owui_chat, "UPLOAD_DIR", tmp_path / "up")
    monkeypatch.setattr(owui_chat, "EXPORT_DIR", tmp_path / "out")
    monkeypatch.setattr(owui_chat, "_CACHE", {})

    url = _data_url(b"same-image")
    first = owui_chat.extract_tables(url)
    second = owui_chat.extract_tables(url)

    assert len(calls) == 1
    assert first is second


def test_extract_tables_survives_a_failing_export(tmp_path, monkeypatch):
    def boom(result, path):
        raise RuntimeError("openpyxl patladi")

    monkeypatch.setattr(owui_chat, "run_consensus",
                        lambda p: [{"headers": ["A"], "rows": [["1"]], "confidence": 1.0}])
    monkeypatch.setattr(owui_chat, "export_result_xlsx", boom)
    monkeypatch.setattr(owui_chat, "UPLOAD_DIR", tmp_path / "up")
    monkeypatch.setattr(owui_chat, "EXPORT_DIR", tmp_path / "out")
    monkeypatch.setattr(owui_chat, "_CACHE", {})

    entry = owui_chat.extract_tables(_data_url(b"broken-export"))
    assert entry["files"] == [None]
    assert entry["results"][0]["headers"] == ["A"]


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(owui_chat, "_CACHE", {})
    monkeypatch.setattr(owui_chat, "_CACHE_MAX", 3)
    for i in range(5):
        owui_chat._cache_put(f"d{i}", {"results": [], "files": []})
    assert len(owui_chat._CACHE) == 3


# --- rendering ---

def _entry(needs_review=False, disagreements=(), name="abc-0.xlsx"):
    return {
        "results": [{
            "headers": ["Kod", "Tutar"],
            "rows": [["1", "10,00"]],
            "confidence": 0.97,
            "needs_review": needs_review,
            "disagreements": list(disagreements),
        }],
        "files": [name],
    }


def test_render_includes_table_confidence_and_download_link():
    md = owui_chat.render_tables(_entry())
    assert "| Kod | Tutar |" in md
    assert "0.97" in md
    assert "/files/abc-0.xlsx" in md
    assert owui_chat.PUBLIC_BASE_URL in md


def test_render_reports_disagreement_count_when_review_needed():
    md = owui_chat.render_tables(_entry(needs_review=True, disagreements=[1, 2, 3]))
    assert "3 hücrede" in md


def test_render_warns_without_count_when_no_disagreement_listed():
    md = owui_chat.render_tables(_entry(needs_review=True))
    assert "gözden geçirilmeli" in md


def test_render_omits_link_when_export_failed():
    md = owui_chat.render_tables(_entry(name=None))
    assert "/files/" not in md


# --- reply routing ---

def test_tables_reply_asks_for_an_image_when_none_attached():
    assert owui_chat.tables_reply([{"role": "user", "content": "selam"}]) \
        == owui_chat.NO_IMAGE_MSG


def test_tables_reply_rejects_non_data_url_image(monkeypatch):
    messages = [{"role": "user", "content": [_img_part("https://example.com/a.png")]}]
    assert owui_chat.tables_reply(messages) == owui_chat.BAD_IMAGE_MSG


def test_tables_reply_reports_when_no_table_found(monkeypatch):
    monkeypatch.setattr(owui_chat, "extract_tables", lambda url: {"results": [], "files": []})
    messages = [{"role": "user", "content": [_img_part(_data_url())]}]
    assert owui_chat.tables_reply(messages) == owui_chat.NO_TABLE_MSG


def test_tables_reply_propagates_extraction_failure(monkeypatch):
    def boom(url):
        raise RuntimeError("servis kapali")

    monkeypatch.setattr(owui_chat, "extract_tables", boom)
    messages = [{"role": "user", "content": [_img_part(_data_url())]}]
    with pytest.raises(RuntimeError):
        owui_chat.tables_reply(messages)


# --- SSE ---

def _payloads(chunks):
    return [json.loads(c[len("data: "):]) for c in chunks if not c.startswith("data: [DONE]")]


def test_sse_chunk_is_openai_shaped():
    payload = json.loads(owui_chat.sse_chunk("id1", "m", delta="merhaba")[len("data: "):])
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "merhaba"
    assert payload["choices"][0]["finish_reason"] is None


def test_stream_text_emits_answer_then_stop_then_done():
    chunks = list(owui_chat.stream_text("cevap", "ragtest-rag"))
    assert chunks[-1] == "data: [DONE]\n\n"
    payloads = _payloads(chunks)
    assert payloads[0]["choices"][0]["delta"]["content"] == "cevap"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_tables_reports_missing_image_and_terminates():
    chunks = list(owui_chat.stream_tables([{"role": "user", "content": "selam"}], "m"))
    assert owui_chat.NO_IMAGE_MSG in _payloads(chunks)[0]["choices"][0]["delta"]["content"]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_stream_tables_does_not_wait_a_tick_for_a_fast_result(monkeypatch):
    """A cache hit finishes in milliseconds -- polling has to be finer than the
    tick interval or every follow-up turn would stall for a full tick."""
    monkeypatch.setattr(owui_chat, "STREAM_TICK_SECONDS", 30.0)
    monkeypatch.setattr(owui_chat, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(owui_chat, "extract_tables", lambda url: _entry())

    messages = [{"role": "user", "content": [_img_part(_data_url())]}]
    started = time.monotonic()
    chunks = list(owui_chat.stream_tables(messages, "m"))
    assert time.monotonic() - started < 5.0
    assert "| Kod | Tutar |" in "".join(
        p["choices"][0]["delta"].get("content", "") for p in _payloads(chunks))


def test_stream_tables_renders_result_after_progress(monkeypatch):
    monkeypatch.setattr(owui_chat, "STREAM_TICK_SECONDS", 0.01)
    monkeypatch.setattr(owui_chat, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(owui_chat, "extract_tables", lambda url: _entry())

    messages = [{"role": "user", "content": [_img_part(_data_url())]}]
    chunks = list(owui_chat.stream_tables(messages, "m"))
    text = "".join(p["choices"][0]["delta"].get("content", "") for p in _payloads(chunks))

    assert "çalıştırılıyor" in text
    assert "| Kod | Tutar |" in text
    assert chunks[-1] == "data: [DONE]\n\n"


def test_stream_tables_surfaces_failure_as_text_not_a_crash(monkeypatch):
    """Headers are already sent by then, so the error has to arrive as content."""
    monkeypatch.setattr(owui_chat, "STREAM_TICK_SECONDS", 0.01)
    monkeypatch.setattr(owui_chat, "_POLL_SECONDS", 0.01)

    def boom(url):
        raise RuntimeError("servis kapali")

    monkeypatch.setattr(owui_chat, "extract_tables", boom)
    messages = [{"role": "user", "content": [_img_part(_data_url())]}]
    chunks = list(owui_chat.stream_tables(messages, "m"))
    text = "".join(p["choices"][0]["delta"].get("content", "") for p in _payloads(chunks))

    assert "servis kapali" in text
    assert chunks[-1] == "data: [DONE]\n\n"
