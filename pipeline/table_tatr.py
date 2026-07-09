"""Deterministic scanned-table -> {headers, rows} extraction, no VLM.

Consolidates the four engines proven on sample1.jpeg into one path:
  * TATR detection        -> find the table on the page, crop it
  * TATR structure (v1.1) -> column boundaries + the column-header band
  * PaddleOCR word boxes  -> numeric cells (pixel-faithful, best on digits)
  * EasyOCR (Turkish)     -> text cells: headers + name/text columns (reads
                             i / I / s / g / c / o / u correctly, no master list)
  * master_match (opt.)   -> snap text columns to a known list + flag misses

Rows are recovered by clustering the PaddleOCR word boxes on their y-center
(TATR's per-row boxes are vertically unreliable on scans); columns come from
TATR. This module is pure vision/assembly -- it does NOT run PaddleOCR itself
(that lives in paddle_env behind paddle_service). Feed it the table crop and
the PaddleOCR word boxes for that crop; it runs TATR + EasyOCR internally.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import (AutoImageProcessor, TableTransformerForObjectDetection,
                          TableTransformerConfig)

from pipeline import tatr_extract as te
from pipeline import master_match as mm
from pipeline.text_normalize import normalize_tr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DET_REPO = "microsoft/table-transformer-detection"
STRUCT_REPO = "microsoft/table-transformer-structure-recognition-v1.1-all"
STRUCT_SIZE = {"shortest_edge": 800, "longest_edge": 1000}
ID2NAME = {0: "table", 1: "table column", 2: "table row", 3: "table column header",
           4: "table projected row header", 5: "table spanning cell", 6: "no object"}

# lazy singletons -- these models/readers are slow to build, load once per process
_MODELS = {}
_EASYOCR = None


def _load_tatr(repo):
    """Load a TATR model, patching config.json's dilation=null (transformers 5.x's
    strict dataclass rejects None; these checkpoints are dilation=False)."""
    d = json.load(open(hf_hub_download(repo, "config.json")))
    if d.get("dilation") is None:
        d["dilation"] = False
    cfg = TableTransformerConfig.from_dict(d)
    return TableTransformerForObjectDetection.from_pretrained(repo, config=cfg).to(DEVICE).eval()


def _model(repo):
    if repo not in _MODELS:
        _MODELS[repo] = (AutoImageProcessor.from_pretrained(repo), _load_tatr(repo))
    return _MODELS[repo]


def _reader():
    global _EASYOCR
    if _EASYOCR is None:
        import easyocr  # imported lazily: heavy, and only the text path needs it
        _EASYOCR = easyocr.Reader(["tr"], gpu=(DEVICE == "cuda"))
    return _EASYOCR


def detect_and_crop(image, pad=20, threshold=0.6):
    """Find the highest-scoring table on the page and return (crop, crop_box).
    crop_box is (x0,y0,x1,y1) in the original image so callers can map back."""
    proc, model = _model(DET_REPO)
    with torch.no_grad():
        out = model(**proc(images=image, return_tensors="pt").to(DEVICE))
    det = proc.post_process_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor([image.size[::-1]]))[0]
    tables = []
    for s, l, b in zip(det["scores"], det["labels"], det["boxes"]):
        if "table" in model.config.id2label[l.item()]:
            tables.append((s.item(), b.tolist()))
    if not tables:
        return None, None
    tables.sort(reverse=True)
    bx = tables[0][1]
    box = (max(0, int(bx[0] - pad)), max(0, int(bx[1] - pad)),
           min(image.width, int(bx[2] + pad)), min(image.height, int(bx[3] + pad)))
    return image.crop(box), box


def _structure(table_img):
    """TATR structure recognition on the crop -> (columns sorted L->R, header
    y-range). Rows are intentionally dropped; we rebuild them from OCR."""
    proc, model = _model(STRUCT_REPO)
    with torch.no_grad():
        sout = model(**proc(images=table_img, return_tensors="pt", size=STRUCT_SIZE).to(DEVICE))
    objects = te.outputs_to_objects(
        {"pred_logits": sout.logits, "pred_boxes": sout.pred_boxes}, table_img.size, ID2NAME)
    cols = sorted([o for o in objects if o["label"] == "table column" and o["score"] >= 0.5],
                  key=lambda o: o["bbox"][0])
    hdrs = [o for o in objects if o["label"] == "table column header" and o["score"] >= 0.5]
    hdr_y = (min(o["bbox"][1] for o in hdrs), max(o["bbox"][3] for o in hdrs)) if hdrs else None
    return cols, hdr_y


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _read_region(table_img, box, pad=4, scale=3):
    """EasyOCR a rectangular region of the crop, top-to-bottom/left-to-right,
    normalized. Used for header cells (well-separated, one per column)."""
    arr_w, arr_h = table_img.size
    x0 = max(0, int(box[0] - pad)); y0 = max(0, int(box[1] - pad))
    x1 = min(arr_w, int(box[2] + pad)); y1 = min(arr_h, int(box[3] + pad))
    crop = table_img.crop((x0, y0, x1, y1))
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    dets = _reader().readtext(np.array(crop), detail=1, paragraph=False)
    dets.sort(key=lambda d: (round(d[0][0][1] / 10), d[0][0][0]))
    return normalize_tr(" ".join(d[1] for d in dets).strip())


def _clean_text(s):
    import re
    return re.sub(r"^[^0-9A-Za-zÇĞİÖŞÜçğıöşü]+", "", normalize_tr(s).strip())


def _read_text_column(table_img, col, y0, y1, inset=2, scale=2, row_gap=22):
    """Read a whole text column as one strip (EasyOCR handles a natural
    multi-line column better than tight per-cell crops), then split into rows by
    clustering detections on y. Returns [(y_center_in_crop, text), ...]."""
    nx0 = col["bbox"][0] + inset
    nx1 = col["bbox"][2] - inset
    ny0 = max(0, int(y0 - 3)); ny1 = min(table_img.height, int(y1 + 3))
    strip = table_img.crop((nx0, ny0, nx1, ny1))
    strip = strip.resize((strip.width * scale, strip.height * scale), Image.LANCZOS)
    dets = _reader().readtext(np.array(strip), detail=1, paragraph=False)
    # (y_center, x_center, text) in scaled strip coordinates
    items = sorted((sum(p[1] for p in b) / 4, sum(p[0] for p in b) / 4, t) for b, t, c in dets)
    rows, cur, last = [], [], None
    for cyc, cxc, t in items:
        if last is not None and cyc - last > row_gap * scale:
            rows.append(cur); cur = []
        cur.append((cyc, cxc, t)); last = cyc
    if cur:
        rows.append(cur)
    out = []
    for r in rows:
        text = _clean_text(" ".join(t for _, _, t in sorted(r, key=lambda z: z[1])))
        row_cy = ny0 + (sum(cyc for cyc, _, _ in r) / len(r)) / scale
        out.append((row_cy, text))
    return out


def assemble(table_img, ocr_words, *, text_cols=(1,), master_names=None, row_gap=12):
    """Build {headers, rows} from a table crop + its PaddleOCR word boxes.

    ocr_words: [{"text": str, "box": [x0,y0,x1,y1]}, ...] in crop pixels.
    text_cols: column indices whose *data* cells are read by EasyOCR-tr instead
               of PaddleOCR (names, Turkish text). Headers are always EasyOCR.
    master_names: optional iterable -> snap text_cols to it and flag non-matches.
    """
    cols, hdr_y = _structure(table_img)
    ncol = len(cols)
    if ncol == 0:
        return {"headers": [], "rows": [], "flags": ["TATR sutun bulamadi"]}

    def col_of(w):
        x0, _, x1, _ = w["box"]
        return max(range(ncol), key=lambda i: _overlap(x0, x1, cols[i]["bbox"][0], cols[i]["bbox"][2]))

    def cy(w):
        return (w["box"][1] + w["box"][3]) / 2

    # ROWS: cluster PaddleOCR words by y-center
    words = sorted(ocr_words, key=cy)
    clusters, cur, last = [], [], None
    for w in words:
        c = cy(w)
        if last is not None and c - last > row_gap:
            clusters.append(cur); cur = []
        cur.append(w); last = c
    if cur:
        clusters.append(cur)

    def cluster_cy(cl):
        return sum(cy(w) for w in cl) / len(cl)

    def is_header(cl):
        return hdr_y is not None and hdr_y[0] <= cluster_cy(cl) <= hdr_y[1]

    def row_cells(cl):
        cells = ["" for _ in range(ncol)]
        buckets = {}
        for w in cl:
            buckets.setdefault(col_of(w), []).append((w["box"][0], w["text"]))
        for ci, ws in buckets.items():
            cells[ci] = normalize_tr(" ".join(t for _, t in sorted(ws)).strip())
        return cells

    data_cl = sorted([cl for cl in clusters if not is_header(cl)], key=cluster_cy)
    rows = [row_cells(cl) for cl in data_cl]

    # HEADERS: EasyOCR each column over the header band (falls back to the
    # PaddleOCR header cluster if TATR didn't localize a header band)
    headers = ["" for _ in range(ncol)]
    if hdr_y is not None:
        for i, c in enumerate(cols):
            headers[i] = _read_region(table_img, (c["bbox"][0], hdr_y[0], c["bbox"][2], hdr_y[1]))
    else:
        header_cl = sorted([cl for cl in clusters if cluster_cy(cl) < (cluster_cy(data_cl[0]) if data_cl else 1e9)],
                           key=cluster_cy)
        for cl in header_cl:
            for i, v in enumerate(row_cells(cl)):
                if v:
                    headers[i] = (headers[i] + " " + v).strip()

    # TEXT COLUMNS: override PaddleOCR digits-recognizer text with EasyOCR-tr.
    # When EasyOCR splits the strip into exactly as many rows as we have data
    # rows, map by order (the y-centers line up 1:1 and this avoids off-by-one
    # drift near tight rows); otherwise fall back to nearest-y matching.
    if data_cl and rows:
        row_cys = [cluster_cy(cl) for cl in data_cl]
        y0 = min(w["box"][1] for cl in data_cl for w in cl)
        y1 = max(w["box"][3] for cl in data_cl for w in cl)
        for ci in text_cols:
            if ci >= ncol:
                continue
            text_rows = [(cy_, t) for cy_, t in _read_text_column(table_img, cols[ci], y0, y1) if t]
            if len(text_rows) == len(rows):
                for j, (_, text) in enumerate(text_rows):
                    rows[j][ci] = text
            else:
                for row_cy, text in text_rows:
                    j = min(range(len(row_cys)), key=lambda k: abs(row_cys[k] - row_cy))
                    rows[j][ci] = text

    # MASTER MATCH (optional): snap text columns + collect review flags
    flags = []
    if master_names:
        index = mm.build_index(master_names)
        for ci in text_cols:
            if ci >= ncol:
                continue
            for r, row in enumerate(rows):
                corrected, matched = mm.correct_value(row[ci], index)
                row[ci] = corrected
                if not matched:
                    flags.append(f"satir {r + 1} sutun {ci}: master listede yok ({corrected!r})")

    return {"headers": headers, "rows": rows, "flags": flags}


def extract_from_crop(table_img, ocr_words, **kw):
    """Convenience: assemble from an already-detected crop + its word boxes."""
    return assemble(table_img, ocr_words, **kw)
