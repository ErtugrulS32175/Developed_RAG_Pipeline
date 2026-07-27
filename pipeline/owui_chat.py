"""OpenWebUI chat adapter: turns OpenAI-style chat messages into table-extraction
runs and back into markdown/SSE.

Kept separate from api.py because that module opens a Postgres connection at
import time -- this one has no database dependency, so the message parsing,
caching and rendering here stay unit-testable without a live DB.
"""
import base64
import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.table_pipeline import run_consensus
from pipeline.table_export import table_to_markdown, export_result_xlsx

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "./output/owui"))

# Base URL for download links. OpenWebUI reaches the API from inside its
# container (host.docker.internal), but the Excel link is clicked in the user's
# browser on the host -- those are different origins, so the link can't reuse
# whatever URL OpenWebUI itself was configured with.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
# How often the streaming reply emits a progress tick while the models run.
STREAM_TICK_SECONDS = float(os.getenv("STREAM_TICK_SECONDS", "5"))
# Checked far more often than a tick is emitted, so a cache hit (which finishes
# in milliseconds) isn't held back by the tick interval.
_POLL_SECONDS = 0.25

DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)
MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}
# Only ever serve files we named ourselves -- keeps a crafted request from
# walking out of EXPORT_DIR.
EXPORT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+\.xlsx$")

NO_IMAGE_MSG = ("Tablo çıkarımı için bir tablo görüntüsü yükleyin: sohbet "
                "kutusundaki + ile bir PNG/JPG ekleyip gönderin.")
BAD_IMAGE_MSG = "Görüntü çözülemedi (yalnızca gömülü base64 görüntüler destekleniyor)."
NO_TABLE_MSG = "Görüntüde tablo bulunamadı."

# Extraction is minutes of GPU work, so results are cached by image content:
# a follow-up turn in the same chat re-finds the same image and answers instantly.
_CACHE = {}
_CACHE_MAX = 32


def _content(msg):
    """Message content, whether the caller passed a pydantic model or a dict."""
    return msg.get("content") if isinstance(msg, dict) else msg.content


def message_text(content) -> str:
    """The user's typed text, whether the turn is a plain string or a
    multimodal content-part list."""
    if isinstance(content, str):
        return content
    return "\n".join(
        p.get("text", "") for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ).strip()


def message_image_urls(content) -> list:
    """Every image_url attached to a multimodal turn, in order."""
    if isinstance(content, str):
        return []
    urls = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if url:
                urls.append(url)
    return urls


def latest_image_url(messages):
    """The most recent image attached anywhere in the conversation. Looking only
    at the last turn would break every follow-up question, since OpenWebUI
    re-sends the image only on the turn it was attached to."""
    for msg in reversed(messages):
        urls = message_image_urls(_content(msg))
        if urls:
            return urls[-1]
    return None


def decode_data_url(data_url: str):
    """(bytes, extension) for a base64 data: URL -- what OpenWebUI embeds for an
    attached image. None for a plain http(s) url, which we don't fetch."""
    m = DATA_URL_RE.match(data_url)
    if not m:
        return None
    return base64.b64decode(m.group("data")), MIME_EXT.get(m.group("mime"), "png")


def _cache_put(digest, entry):
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[digest] = entry


def extract_tables(data_url: str):
    """Run the two-VLM consensus on one attached image and write an xlsx per
    detected table. Returns {"results", "files"} (files parallel to results, an
    entry is None if its export failed), or None if the url wasn't decodable."""
    decoded = decode_data_url(data_url)
    if decoded is None:
        return None
    raw, ext = decoded

    # Content hash doubles as the cache key and the on-disk name, so the same
    # image uploaded twice is neither re-extracted nor re-saved.
    digest = hashlib.sha256(raw).hexdigest()[:16]
    if digest in _CACHE:
        return _CACHE[digest]

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    src = UPLOAD_DIR / f"owui-{digest}.{ext}"
    if not src.exists():
        src.write_bytes(raw)

    results = run_consensus(str(src))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for i, r in enumerate(results):
        name = f"{digest}-{i}.xlsx"
        try:
            export_result_xlsx(r, str(EXPORT_DIR / name))
            files.append(name)
        except Exception as e:
            # A failed export must not cost the user the extracted table itself.
            print(f"[API] xlsx yazilamadi ({name}): {e}")
            files.append(None)

    entry = {"results": results, "files": files}
    _cache_put(digest, entry)
    return entry


def render_tables(entry) -> str:
    """Render consensus results as markdown OpenWebUI can display inline, each
    with its confidence, review warning and Excel download link."""
    blocks = []
    for i, (r, name) in enumerate(zip(entry["results"], entry["files"]), 1):
        md = table_to_markdown(r["headers"], r["rows"])
        note = f"_Güven: {r['confidence']:.2f}_"
        if r.get("needs_review"):
            n = len(r.get("disagreements", []))
            note += (f" — ⚠️ {n} hücrede modeller uyuşmuyor, gözden geçirin"
                     if n else " — ⚠️ gözden geçirilmeli")
        if name:
            note += f"\n\n[📥 Excel indir (uyuşmazlıklar işaretli)]({PUBLIC_BASE_URL}/files/{name})"
        blocks.append(f"**Tablo {i}**\n\n{md}\n\n{note}")
    return "\n\n---\n\n".join(blocks)


def tables_reply(messages) -> str:
    """Full non-streaming reply for the table model. Propagates extraction
    failures so the caller can turn them into an HTTP error."""
    url = latest_image_url(messages)
    if url is None:
        return NO_IMAGE_MSG
    entry = extract_tables(url)
    if entry is None:
        return BAD_IMAGE_MSG
    if not entry["results"]:
        return NO_TABLE_MSG
    return render_tables(entry)


def sse_chunk(chat_id, model, delta=None, finish=None) -> str:
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": delta} if delta is not None else {},
            "finish_reason": finish,
        }],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_tables(messages, model):
    """Progress-reporting SSE for the table model. Two VLMs on one image take
    minutes; without ticks an unbroken spinner is indistinguishable from a hang.
    The work runs in a thread so the generator stays free to emit them."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
    url = latest_image_url(messages)
    if url is None:
        yield sse_chunk(chat_id, model, delta=NO_IMAGE_MSG)
    else:
        yield sse_chunk(chat_id, model, delta="⏳ Görüntü alındı, iki model çalıştırılıyor")
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(extract_tables, url)
            since_tick = 0.0
            while not fut.done():
                time.sleep(_POLL_SECONDS)
                since_tick += _POLL_SECONDS
                if since_tick >= STREAM_TICK_SECONDS:
                    since_tick = 0.0
                    yield sse_chunk(chat_id, model, delta=".")
            try:
                entry = fut.result()
                text = (BAD_IMAGE_MSG if entry is None
                        else NO_TABLE_MSG if not entry["results"]
                        else render_tables(entry))
            except Exception as e:
                text = f"⚠️ Tablo çıkarımı başarısız: {e}"
        yield sse_chunk(chat_id, model, delta="\n\n" + text)
    yield sse_chunk(chat_id, model, finish="stop")
    yield "data: [DONE]\n\n"


def stream_text(answer, model):
    """The RAG path has no mid-flight stages worth reporting, so it emits one
    chunk -- enough that OpenWebUI's stream toggle works for both models."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
    yield sse_chunk(chat_id, model, delta=answer)
    yield sse_chunk(chat_id, model, finish="stop")
    yield "data: [DONE]\n\n"
