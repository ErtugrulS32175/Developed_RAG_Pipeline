"""Mechanical repair of the two Turkish-character error classes OCR/VLM models
make (documented in the OCRTurk benchmark error analysis). This layer only
*corrects* deterministic glyph errors; it never touches content errors like
dropped letters or swallowed spaces -- those are caught (not fixed) by the
validation/cross-check layer and left for a human. See [[ocrturk_benchmark]].
"""

import re
import unicodedata

# Spacing modifier symbols the models emit as a SEPARATE character (typically
# BEFORE the base letter) instead of the composed Turkish glyph, e.g.
# "DOGAL"->"DO<breve>GAL", "BESERI"->"BE<cedilla>SER<dot>I". NFC cannot compose
# these (they are spacing modifiers in the wrong position), so map the sequence
# explicitly. Combining-mark forms (mark AFTER the letter) are handled by NFC
# first and don't need entries here, but we include a couple defensively.
_MODIFIER_SEQ = {
    "˘g": "ğ", "˘G": "Ğ",   # spacing breve + g
    "¸s": "ş", "¸S": "Ş",   # spacing cedilla + s
    "¸c": "ç", "¸C": "Ç",   # spacing cedilla + c
    "˙I": "İ",                    # spacing dot-above + I  (e.g. "GSYIH")
    "˙i": "i",
}

# Wrong-character substitutions NFC and the sequence map don't cover: the model
# emitted a genuinely different glyph. Safe to map back because these do not
# occur in Turkish text. "GSYİH"->"GSYÌH".
_TR_FIX = {"Ì": "İ", "ì": "i"}

# Marks that should NOT survive normalization in clean Turkish text; if any
# remain, a glyph wasn't recomposed and the value is suspect (review flag).
_RESIDUAL = re.compile(r"[̀-ͯ˘˙¸]")


def normalize_tr(text):
    """Recompose decomposed Turkish characters (NFC), fix spacing-modifier
    sequences and wrong-glyph substitutions. Idempotent and safe on already
    clean text. Returns the input unchanged if it is empty/None."""
    if not text:
        return text
    s = unicodedata.normalize("NFC", str(text))
    for wrong, right in _MODIFIER_SEQ.items():
        if wrong in s:
            s = s.replace(wrong, right)
    for wrong, right in _TR_FIX.items():
        if wrong in s:
            s = s.replace(wrong, right)
    return s


_NUM_RE = re.compile(r"-?\d[\d.,]*")


def normalize_number(text):
    """Canonicalize a numeric cell to Turkish format (comma decimal, dot
    thousands) by SWAPPING separators only -- never regrouping -- so a plain
    integer like a 4-digit year or code is left untouched. Non-numeric text
    (an ISO date, a percentage, anything with no separator) passes through
    unchanged.

    Fixes the common VLM slip of emitting an English decimal ('12.34') for a
    Turkish table ('12,34'), and a fully English number ('9,876.54' ->
    '9.876,54'). A dot before a 3-digit group is read as a thousands separator and
    kept as-is."""
    s = str(text).strip()
    if not _NUM_RE.fullmatch(s) or ("." not in s and "," not in s):
        return text
    neg, body = s.startswith("-"), s.lstrip("-")
    dot, comma = body.rfind("."), body.rfind(",")
    if dot >= 0 and comma >= 0:
        if dot > comma:                     # English (dot=decimal, comma=thousands) -> swap
            body = body.replace(",", "\0").replace(".", ",").replace("\0", ".")
        # else already Turkish -> leave
    elif dot >= 0:
        parts = body.split(".")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            body = body.replace(".", ",")   # a lone decimal dot -> comma
        # else 3-digit group / multiple dots = thousands -> leave
    # comma-only is already Turkish decimal (or rare English thousands) -> leave
    return ("-" if neg else "") + body


def has_residual_marks(text):
    """True if, after normalization, stray accent/modifier marks remain -- a
    signal that normalize_tr hit a sequence it doesn't know about."""
    if not text:
        return False
    return bool(_RESIDUAL.search(unicodedata.normalize("NFC", str(text))))
