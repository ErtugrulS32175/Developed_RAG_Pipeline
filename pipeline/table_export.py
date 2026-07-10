import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from openpyxl import Workbook

from pipeline.text_normalize import has_residual_marks


class _HTMLTableExtractor(HTMLParser):
    """Pull <table>s out of a VLM's markdown/HTML output into rows of cell text.
    Tolerant of the messy HTML doc-parsing VLMs (PaddleOCR-VL, HunyuanOCR) emit:
    ignores unknown tags, treats <th>/<td> as cells, <tr> as rows. Cell spans
    (colspan/rowspan) are NOT expanded -- flagged upstream as a structure issue."""

    def __init__(self):
        super().__init__()
        self.tables, self._rows, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._rows.append([])
        elif tag in ("td", "th") and self._rows:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "table" and self._rows is not None:
            if any(self._rows):
                self.tables.append([r for r in self._rows if r])
            self._rows = None


def parse_html_tables(text):
    """VLM markdown/HTML -> [{headers, rows}]. First row becomes the header, the
    rest are data (VLMs put column labels in the first <tr>). Returns [] if no
    <table> is present."""
    p = _HTMLTableExtractor()
    try:
        p.feed(text or "")
    except Exception:
        pass
    out = []
    for grid in p.tables:
        if not grid:
            continue
        headers = grid[0]
        rows = grid[1:] if len(grid) > 1 else []
        out.append({"headers": headers, "rows": rows})
    return out


def _extract_balanced(text, start):
    """Return the balanced bracket substring starting at text[start] (a '[' or
    '{'), string-aware so brackets inside quoted values don't count."""
    open_ch = text[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_table_json(raw):
    """Parse a VLM's table output into {headers, rows}, tolerant of the mess
    VLMs produce: ``` fences and botched trailing brackets (e.g. ']}]' instead
    of ']]}') that make a strict json.loads return nothing. Strategy: strip
    fences, try strict parse, else pull the headers array and each row array
    out individually, skipping stray braces. Lives client-side on purpose so
    parser fixes don't need a redeploy of the GPU table service."""
    if not raw:
        return {"headers": [], "rows": []}
    text = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", raw.strip())).strip()

    try:
        data = json.loads(text)
        return {"headers": data.get("headers", []), "rows": data.get("rows", [])}
    except (json.JSONDecodeError, AttributeError):
        pass

    headers = []
    hm = re.search(r'"headers"\s*:\s*\[', text)
    if hm:
        h = _extract_balanced(text, hm.end() - 1)
        if h:
            try:
                headers = json.loads(h)
            except json.JSONDecodeError:
                headers = []

    rows = []
    rm = re.search(r'"rows"\s*:\s*\[', text)
    if rm:
        i, depth = rm.end(), 1
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "[":
                row = _extract_balanced(text, i)
                if row is None:
                    break
                try:
                    rows.append(json.loads(row))
                except json.JSONDecodeError:
                    pass
                i += len(row)
                continue
            if c == "]":
                depth -= 1
            i += 1
    return {"headers": headers, "rows": rows}


# Fold Turkish letters to an ASCII base so the OCR cross-check tolerates the
# expected ı/i, ş/s, ğ/g ... disagreement between two OCR engines: text cells
# come from EasyOCR-tr ("Sirket-A") but the page cross-check text comes from the
# latin PaddleOCR recognizer ("Sirket-A"). Without this, correct Turkish names get
# falsely flagged as hallucination candidates.
_TR_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ç": "c", "Ç": "c", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
})


def _squash(s) -> str:
    """Lowercase, fold Turkish diacritics, and strip whitespace/punctuation for
    tolerant matching, so neither formatting differences (1.000,50 vs 1000.50,
    spacing) nor cross-engine Turkish-char disagreement trigger false
    hallucination flags in the OCR cross-check."""
    return re.sub(r"[\s\W_]+", "", str(s).translate(_TR_FOLD).lower())


def validate_table(headers, rows, ocr_text=None):
    """Return (confidence, issues). Detects problems, never corrects them --
    low-confidence tables are flagged for a human, not dropped. When ocr_text
    (the same page's OCR output) is given, cross-check every cell against it:
    values that never appear in the OCR text are hallucination candidates
    (invented cells / dropped letters the normalize layer can't catch)."""
    if not headers or not rows:
        return 0.0, ["bos tablo (header veya satir yok)"]

    issues = []
    width = len(headers)

    bad_width = sum(1 for row in rows if len(row) != width)
    if bad_width:
        issues.append(f"{bad_width} satir header genisligiyle uyusmuyor (sutun sayisi tutmuyor)")

    empty_rows = sum(1 for row in rows if all(str(c).strip() == "" for c in row))
    if empty_rows:
        issues.append(f"{empty_rows} tamamen bos satir")

    unmatched = []
    if ocr_text:
        ocr_norm = _squash(ocr_text)
        for row in rows:
            for cell in row:
                token = str(cell).strip()
                if token and _squash(token) not in ocr_norm:
                    unmatched.append(token)
        if unmatched:
            sample = ", ".join(unmatched[:5])
            issues.append(f"{len(unmatched)} hucre OCR metninde yok (uydurma adayi): {sample}")

    flat = " ".join(str(c) for row in rows for c in row)
    if has_residual_marks(flat) or has_residual_marks(" ".join(str(h) for h in headers)):
        issues.append("cozulmemis Turkce karakter isareti kaldi (normalize eksik)")

    confidence = sum(1 for row in rows if len(row) == width) / len(rows)
    n_cells = sum(len(row) for row in rows) or 1
    if unmatched:
        confidence *= max(0.0, 1 - len(unmatched) / n_cells)
    return round(confidence, 2), issues


def estimate_table_confidence(headers, rows) -> float:
    """Cheap deterministic proxy for extraction quality -- not a model
    confidence. Gemma's table output is generative JSON, not a detection
    model, so there's no calibrated per-cell score to report; this just
    checks the shape came back consistent (every row matches the header
    width), which is exactly the kind of structural regression PaddleOCR's
    table model was dropped for (see the table-module-status notes)."""
    if not headers or not rows:
        return 0.0
    width = len(headers)
    consistent = sum(1 for row in rows if len(row) == width)
    return round(consistent / len(rows), 2)


def table_to_markdown(headers, rows, *, filename=None, page=None, table_id=None, confidence=None) -> str:
    if not headers:
        return ""
    lines = []
    if filename is not None or table_id is not None:
        lines.append(f"Belge: {filename}")
        lines.append(f"Sayfa: {page}")
        lines.append(f"Tablo: {table_id}")
        if confidence is not None:
            lines.append(f"Güven: {confidence:.2f}")
        lines.append("")
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def save_table_json(table_id, page, headers, rows, confidence, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "table_id": table_id,
        "page": page,
        "headers": headers,
        "rows": rows,
        "confidence": confidence,
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_table_xlsx(headers, rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    if headers:
        ws.append(headers)
    for row in rows:
        ws.append(row)
    wb.save(path)


def save_table_csv(headers, rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(rows)
