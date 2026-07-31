"""MinerU2.5 table backend (1.2B document-parsing VLM, opendatalab).

Reads the whole image as one table ("Table Recognition") and returns the same
{tables:[{headers,rows}]} contract as the other backends.

Carries one behaviour the other backends do not: a second read at a larger
scale when the first output looks like it lost repeated rows. See RETRY_SCALE.

Trial backend (router.TABLE_BACKEND=mineru), wired alongside the production
consensus pair rather than into it.
"""
import dataclasses
import io
import os
from collections import Counter

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from pipeline.extraction.table_export import parse_html_tables

app = FastAPI()
MODEL_ID = os.getenv("MINERU_MODEL", "opendatalab/MinerU2.5-Pro-2605-1.2B")
# Both the 2509 and 2605-Pro checkpoints declare architectures=[
# "Qwen2VLForConditionalGeneration"] in their config.json (the model card's
# AutoModelForMultimodalLM mention does not match the weights, and that class
# does not exist in transformers 4.x). Overridable for a future checkpoint.
MODEL_CLASS = os.getenv("MINERU_MODEL_CLASS", "Qwen2VLForConditionalGeneration")
# table  = one pass, whole image treated as a table ("Table Recognition" prompt).
# layout = the library's two_step_extract (layout detection, then per-region
#          recognition) for a mixed document page.
# Default is `table` because this backend is fed table crops, and because the
# layout stage does not work on them: given a page that is entirely a table the
# model answers the layout prompt with table content instead of bounding boxes,
# so parse_layout_output yields zero blocks and nothing reaches stage two
# ("Layout output does not match expected format" in the library's own log).
MODE = os.getenv("MINERU_MODE", "table").lower()
# The library's default table sampling sets no_repeat_ngram_size=100, forbidding
# the model from ever emitting the same 100-token span twice. A financial table
# whose records are genuinely identical then loses the repeats. Off by default;
# set >0 to restore the library's behaviour.
NO_REPEAT_NGRAM = int(os.getenv("MINERU_NO_REPEAT_NGRAM", "0"))

# --- Repeated-row safety net ------------------------------------------------
# Densely packed rows that are IDENTICAL have no visual cue separating them, so
# the model merges them and a few of the repeats are simply missing -- every
# value correct, only the repetition count lost. That failure is silent: number
# verification sees nothing wrong, and a second *model* at the same scale can
# make the same mistake. Reading the same image larger does resolve them.
#
# A scale sweep on a low-resolution sample showed the response is NOT monotonic:
# 1.0x drops records, a small enlargement recovers them, an intermediate scale
# collapses the reading completely (hundreds of junk rows), and from ~1.75x
# upward the output is stable and identical at every larger scale. 2.0x sits in
# the middle of that plateau, clear of the unstable point below it, and larger
# scales only cost tokens (image area grows quadratically).
#
# It is deliberately NOT applied to every image: a flat table that was already
# perfect at 1x lost a cell at every scale >= 1.25x, gaining nothing. So the
# retry is spent only where the risk is visible in the first reading.
RETRY_SCALE = float(os.getenv("MINERU_RETRY_SCALE", "2.0"))
# How many times a row must repeat before the first reading is treated as
# suspect. Distinct rows (a date or index column varying per row) never trip it.
REPEAT_TRIGGER = int(os.getenv("MINERU_REPEAT_TRIGGER", "3"))
_STATE = {}


def _load():
    if not _STATE:
        import transformers
        from mineru_vl_utils import MinerUClient

        cls = getattr(transformers, MODEL_CLASS)
        model = cls.from_pretrained(MODEL_ID, dtype="auto", device_map="auto").eval()
        processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        client = MinerUClient(
            backend="transformers", model=model, processor=processor)
        table_params = client.sampling_params.get("table")
        if table_params is not None:
            client.sampling_params["table"] = dataclasses.replace(
                table_params, no_repeat_ngram_size=NO_REPEAT_NGRAM or None)
        _STATE["client"] = client
    return _STATE["client"]


def _attr(block, name):
    """Blocks come back as objects, but tolerate a dict shape too."""
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _read(image, client):
    """One model pass -> (tables, raw table fragments, block-type counts)."""
    from mineru_vl_utils.post_process import convert_otsl_to_html

    with torch.inference_mode():
        if MODE == "layout":
            blocks = client.two_step_extract(image)
            raw_blocks = [_attr(b, "content") for b in blocks
                          if _attr(b, "type") == "table"]
            found = {}
            for b in blocks:
                found[_attr(b, "type")] = found.get(_attr(b, "type"), 0) + 1
        else:
            out = client.content_extract(image, type="table")
            raw_blocks = [str(out)] if out else []
            found = {"table": 1 if raw_blocks else 0}

    # This checkpoint can emit OTSL (<fcel>/<ucel>/<nl>) rather than the HTML the
    # docs describe. Their converter is a no-op on HTML, so it covers both.
    raw_blocks = [c for c in raw_blocks if c]
    html_blocks = [convert_otsl_to_html(c) for c in raw_blocks]

    # One fragment per detected table; parse each separately so two tables on a
    # page stay two tables instead of being concatenated into one grid.
    tables = []
    for html in html_blocks:
        tables.extend(parse_html_tables(html))
    return tables, raw_blocks, html_blocks, found


def _max_repeat(tables):
    """How many times the most-repeated non-empty data row occurs. Counts
    duplicates anywhere, not just adjacent ones: a form that splits each record
    over two physical lines interleaves them, so identical records are not
    neighbours."""
    best = 0
    for table in tables:
        rows = [tuple(str(c).strip() for c in row) for row in table.get("rows", [])]
        rows = [r for r in rows if any(r)]
        if rows:
            best = max(best, max(Counter(rows).values()))
    return best


def _row_counts(tables):
    return [len(t.get("rows", [])) for t in tables]


@app.post("/table")
async def extract_table(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    client = _load()

    tables, raw_blocks, html_blocks, found = _read(image, client)

    repeat = _max_repeat(tables)
    scale_check = {"triggered": False, "max_repeat": repeat}
    if RETRY_SCALE > 1 and repeat >= REPEAT_TRIGGER:
        w, h = image.size
        bigger = image.resize((round(w * RETRY_SCALE), round(h * RETRY_SCALE)),
                              Image.LANCZOS)
        r_tables, r_raw, r_html, r_found = _read(bigger, client)
        first_counts, retry_counts = _row_counts(tables), _row_counts(r_tables)
        agree = first_counts == retry_counts
        scale_check = {
            "triggered": True,
            "max_repeat": repeat,
            "scale": RETRY_SCALE,
            "first_rows": first_counts,
            "retry_rows": retry_counts,
            "agree": agree,
        }
        print("[MINERU] %dx tekrarlanan satir -> %.2fx yeniden okundu: %s vs %s%s"
              % (repeat, RETRY_SCALE, first_counts, retry_counts,
                 "" if agree else "  <-- UYUSMAZLIK, insan kontrolu"))
        # The retry exists because the first reading is the one under suspicion,
        # so its result wins. Row counts that disagree are reported, never
        # silently reconciled -- "more rows" is not a safe tiebreak, since an
        # unstable scale can produce far MORE rows than the table really has.
        tables, raw_blocks, html_blocks, found = r_tables, r_raw, r_html, r_found

    dbg = os.getenv("MINERU_DEBUG_DIR")
    if dbg:
        with open(os.path.join(dbg, "mineru_last_raw.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(raw_blocks) or "(tablo blogu yok)")
    return {"tables": tables, "raw": "\n\n".join(html_blocks),
            "block_types": found, "scale_check": scale_check}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": bool(_STATE)}
