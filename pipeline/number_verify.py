"""Deterministic number-fidelity verification for VLM-extracted tables.

VLMs generalize across table formats but can silently alter digits (drop a
thousands separator, swap 6->8, invent a value). For financial tables where every
cell matters, this layer cross-checks each NUMERIC cell in the VLM output against
a faithful, deterministic OCR reading of the same image (classic PaddleOCR, which
is pixel-faithful on digits). A numeric cell whose digits never appear in the OCR
reading is a hallucination candidate -> flagged (never dropped). Produces a
number-fidelity score that feeds the pipeline's confidence + human-review routing.
"""
import re

# a single number: digits with internal thousands/decimal separators (NO spaces
# -- whitespace separates distinct numbers, so "12,34 56,78" -> two tokens)
_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")


def is_numeric(cell) -> bool:
    """True for cells that are essentially a number (digits plus the usual
    financial punctuation), so we only fidelity-check numbers, not text."""
    s = str(cell).strip()
    if not re.search(r"\d", s):
        return False
    return bool(re.fullmatch(r"[-+()%.,\s\d]+", s))


def _digits(s) -> str:
    """Digits only -- drop thousands/decimal separators, sign, spaces -- so
    1.234,56 / 1234,56 / 1,234.56 all compare equal. We verify the DIGITS were
    read faithfully, not the formatting (normalization handles formatting)."""
    return re.sub(r"\D", "", str(s))


def numeric_token_set(ocr_text) -> set:
    """All numeric digit-strings present in a deterministic OCR reading."""
    keys = set()
    for m in _NUM_RE.finditer(ocr_text or ""):
        k = _digits(m.group())
        if k:
            keys.add(k)
    return keys


def verify(headers, rows, ocr_text):
    """Cross-check numeric cells against the deterministic OCR reading.

    Returns (fidelity, flags):
      fidelity = matched_numeric / total_numeric  (1.0 when there are no numbers)
      flags    = [(row_idx, col_idx, value), ...] for numeric cells whose digits
                 never appear in the OCR text (hallucination candidates).
    """
    ocr_keys = numeric_token_set(ocr_text)
    total = matched = 0
    flags = []
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            if not is_numeric(cell):
                continue
            total += 1
            key = _digits(cell)
            if key and key in ocr_keys:
                matched += 1
            else:
                flags.append((ri, ci, str(cell).strip()))
    fidelity = 1.0 if total == 0 else round(matched / total, 3)
    return fidelity, flags


def flags_to_messages(flags, headers=None):
    """Human-readable review notes for flagged numeric cells."""
    msgs = []
    for ri, ci, val in flags:
        col = headers[ci] if headers and ci < len(headers) else f"sutun {ci}"
        msgs.append(f"satir {ri + 1} / {col}: '{val}' deterministik OCR'da yok (hane uydurma adayi)")
    return msgs
