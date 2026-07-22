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
# -- whitespace separates distinct numbers, so "179,53 564,26" -> two tokens)
_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")


def is_numeric(cell) -> bool:
    """True for cells that are essentially a number (digits plus the usual
    financial punctuation), so we only fidelity-check numbers, not text."""
    s = str(cell).strip()
    if not re.search(r"\d", s):
        return False
    return bool(re.fullmatch(r"[-+()%.,\s\d]+", s))


# a monetary value: digits (with optional . thousands) and a , decimal part
_FINANCIAL_RE = re.compile(r"[-+(]?\d[\d.]*,\d{1,2}\)?%?")


def is_financial(cell) -> bool:
    """True only for a monetary/decimal value -- a number with a comma decimal
    part (12,34 / 1.234,56 / 0,00 / -7,89). Fidelity-checking is scoped to these:
    they are the values that matter and that a VLM might silently alter. Bare
    integers used as row indices / years / months, and ISO year-month dates, are
    NOT financial -- and the deterministic OCR often can't read them from narrow
    columns -- so verifying them only produces false hallucination flags."""
    return bool(_FINANCIAL_RE.fullmatch(str(cell).strip()))


def _digits(s) -> str:
    """Digits only -- drop thousands/decimal separators, sign, spaces -- so
    1.373,66 / 1373,66 / 1,373.66 all compare equal. We verify the DIGITS were
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
    """Cross-check FINANCIAL cells against the deterministic OCR reading.

    Only monetary/decimal cells (is_financial) are checked -- row indices, years,
    months and dates are skipped, since the deterministic OCR frequently can't
    read those narrow/small columns and would false-flag correct values.

    Returns (fidelity, flags):
      fidelity = matched / total financial cells  (1.0 when there are none)
      flags    = [(row_idx, col_idx, value), ...] for financial cells whose digits
                 never appear in the OCR text (hallucination candidates).
    """
    ocr_keys = numeric_token_set(ocr_text)
    total = matched = 0
    flags = []
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            if not is_financial(cell):
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
