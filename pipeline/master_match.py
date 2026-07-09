"""Snap OCR'd values to a known master list, tolerant of the Turkish-character
confusions latin OCR makes (ı/i, ş/s, ğ/g, ç/c, ö/o, ü/u). Both a *correction*
(OCR "Yildiz" -> canonical "Yıldız") and a *validation* signal (no match =>
flag for review). Deterministic: only snaps on an unambiguous match, never
invents. Feed it a customer/product master list for the real pipeline.
"""
import unicodedata

# Fold Turkish special chars (both cases) to an ASCII base so an OCR mis-read of
# a diacritic still matches. Applied before lowercasing the rest.
_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "I": "i",
    "ş": "s", "Ş": "s",
    "ğ": "g", "Ğ": "g",
    "ç": "c", "Ç": "c",
    "ö": "o", "Ö": "o",
    "ü": "u", "Ü": "u",
})


def fold(s) -> str:
    """OCR-confusion-insensitive match key."""
    return unicodedata.normalize("NFC", str(s)).translate(_FOLD).lower().strip()


def build_index(master):
    """master: iterable of canonical strings -> {folded_key: canonical}. Later
    duplicates on the same key win; keep the master clean if that matters."""
    return {fold(m): m for m in master if str(m).strip()}


def snap_word(word, index):
    """(canonical, matched) -- exact fold-match snaps to the canonical spelling."""
    hit = index.get(fold(word))
    return (hit, True) if hit is not None else (word, False)


def correct_value(value, index, per_word=True):
    """Snap a cell value to master entries. per_word matches each token (good for
    'Ad Soyad'); otherwise matches the whole cell. Returns (corrected, all_matched)
    where all_matched is False if any token had no master match (=> review)."""
    value = str(value)
    if not per_word:
        return snap_word(value.strip(), index)
    tokens = value.split()
    if not tokens:
        return value, True
    out, ok = [], True
    for tok in tokens:
        canon, matched = snap_word(tok, index)
        out.append(canon)
        ok = ok and matched
    return " ".join(out), ok
