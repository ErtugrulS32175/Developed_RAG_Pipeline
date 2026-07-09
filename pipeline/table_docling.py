"""Combine Granite-Docling's structure+numbers with EasyOCR-tr Turkish text.

Granite-Docling returns a perfect grid and correct numbers, but mangles Turkish
diacritics (ı/ş/ğ/ç/ö/ü → dropped or replaced). This re-reads the text columns
(and headers) with EasyOCR-tr and maps them onto Granite's rows by order. Granite
emits no per-cell boxes in OTSL mode, so column geometry comes from PaddleOCR word
boxes (per-word, they separate adjacent columns cleanly, unlike EasyOCR's
full-crop detections which merge neighbours). Cuts fall on the gutter midpoint so
a band never catches a neighbouring column's edge digit.

Validated on sample1: headers exact, names near-exact without a master list (only a name→
a name, ş→s residual) — matching the standalone TATR path. Pair with master_match
for the last polish. Reuses the EasyOCR helpers in pipeline.table_tatr.
"""
import re

from pipeline import master_match as mm
from pipeline import table_tatr as tt


def table_bbox_from_doctags(doctags, width, height, loc_scale=500):
    """Table pixel bbox from the first <otsl><loc..> quadruple (DocTags loc is a
    0-`loc_scale` grid). Returns (x0,y0,x1,y1) or None."""
    m = re.search(r"<otsl><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>", doctags)
    if not m:
        return None
    a, b, c, d = (int(x) for x in m.groups())
    return (a * width / loc_scale, b * height / loc_scale,
            c * width / loc_scale, d * height / loc_scale)


def column_bands(word_boxes, ncol, crop_w):
    """Split the crop into `ncol` x-bands using PaddleOCR word boxes: cut at the
    midpoint of the (ncol-1) largest x-center gaps. Returns [(x0,x1)]*ncol."""
    xcen = sorted(((b[0] + b[2]) / 2, b[0], b[2]) for b in word_boxes)
    if len(xcen) < ncol:
        return None
    centers = [c for c, _, _ in xcen]
    gap_idx = sorted(range(1, len(centers)), key=lambda i: centers[i] - centers[i - 1],
                     reverse=True)[:ncol - 1]
    cut_x = sorted((xcen[i - 1][2] + xcen[i][1]) / 2 for i in gap_idx)
    edges = [0.0] + cut_x + [float(crop_w)]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


def _read_header_row(crop, bands, hy, scale=3):
    """Read the whole header row as one EasyOCR strip and assign each detected
    word to the band its x-center falls in -- avoids splitting a header word
    (e.g. 'Fiyat') across two columns the way a tight per-band crop does."""
    import numpy as np
    x_left, x_right = bands[0][0], bands[-1][1]
    strip = crop.crop((max(0, int(x_left)), max(0, int(hy[0] - 4)),
                       min(crop.width, int(x_right)), min(crop.height, int(hy[1] + 4))))
    strip = strip.resize((strip.width * scale, strip.height * scale), tt.Image.LANCZOS)
    dets = tt._reader().readtext(np.array(strip), detail=1, paragraph=False)
    cells = ["" for _ in bands]
    for b, t, c in dets:
        xc = x_left + ((b[0][0] + b[2][0]) / 2) / scale  # x-center back in crop coords
        for i, (bx0, bx1) in enumerate(bands):
            if bx0 <= xc < bx1:
                cells[i] = (cells[i] + " " + t).strip()
                break
    return [tt.normalize_tr(c) for c in cells]


def _header_y_range(word_boxes, gap=12):
    """y-range of the topmost row cluster of word boxes (the header row)."""
    ys = sorted(((b[1] + b[3]) / 2, b[1], b[3]) for b in word_boxes)
    cur, last = [], None
    for cy, y0, y1 in ys:
        if last is not None and cy - last > gap:
            break
        cur.append((y0, y1)); last = cy
    if not cur:
        return None
    return min(y0 for y0, _ in cur), max(y1 for _, y1 in cur)


def correct(crop, headers, rows, word_boxes, *, text_cols=(1,), fix_headers=True,
            master_names=None):
    """Overwrite Granite's mangled Turkish text with EasyOCR-tr. `rows` are the
    data rows only (header already split off). If `master_names` is given, snap
    text columns to it (folds the last ş→s style residual) and flag non-matches
    for review. Returns (headers, rows, flags)."""
    ncol = len(headers)
    bands = column_bands(word_boxes, ncol, crop.width)
    if not bands:
        return headers, rows, ["sutun geometrisi cikarilamadi (yetersiz OCR kutusu)"]

    flags = []
    if fix_headers:
        hy = _header_y_range(word_boxes)
        if hy is not None:
            for i, val in enumerate(_read_header_row(crop, bands, hy)):
                if val:
                    headers[i] = val

    ndata = len(rows)
    for ci in text_cols:
        if ci >= len(bands):
            continue
        col = {"bbox": [bands[ci][0], 0, bands[ci][1], 0]}
        trows = [(cy, t) for cy, t in tt._read_text_column(crop, col, 0, crop.height) if t]
        if len(trows) == ndata + 1:  # drop the header text-row read alongside data
            trows = trows[1:]
        if len(trows) == ndata:
            for j, (_, t) in enumerate(trows):
                rows[j][ci] = t
        else:
            flags.append(f"sutun {ci}: {len(trows)} metin satiri != {ndata} veri satiri, atlandi")

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
    return headers, rows, flags
