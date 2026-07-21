"""Known-form header templates: recognize a table as a known form and stamp its
CORRECT header block on top of the model's (possibly garbled) extraction.

A template is one JSON file per form, holding that form's canonical header:

    {
      "name": "some_form",
      "header_rows":  [[...row0...], [...row1...]],   # display grid, top-left
                                                      # holds text, blanks under
                                                      # a span
      "header_merges": [[r, c, rowspan, colspan], ...]
    }

Templates are USER DATA and live under a gitignored directory -- never shipped in
the repo (a template encodes real document headers). Matching is text-based and
Turkish-fold tolerant so OCR garble ("GRUP~B" vs "GroupB") still lines up with
the right form; stamping requires the extraction's column count to equal the
template's, otherwise the caller flags it for a human instead of forcing a wrong
alignment.
"""
import json
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.table_export import _squash, flatten_header

DEFAULT_TEMPLATE_DIR = Path("data/header_templates")


def load_templates(directory=DEFAULT_TEMPLATE_DIR):
    """Load form templates from a gitignored directory of JSON files. Returns []
    when the directory is absent (templates are user data, not shipped)."""
    d = Path(directory)
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if t.get("header_rows"):
            t.setdefault("name", p.stem)
            out.append(t)
    return out


def _tokens(header_rows):
    """Every non-empty header cell, Turkish-folded, for fuzzy comparison."""
    return [s for row in header_rows for s in (_squash(c) for c in row) if s]


def _best_sim(token, others):
    """Best fuzzy similarity (0..1) of `token` against any of `others`."""
    return max((SequenceMatcher(None, token, o).ratio() for o in others), default=0.0)


def match_template(header_rows, templates, *, token_thresh=0.6, min_score=0.5):
    """Identify which known form a (garbled) header block belongs to. Scores each
    template by the fraction of its header tokens that fuzzily appear in the
    incoming header (Turkish-folded, so OCR noise/diacritics don't block a
    match). Returns (template, score) for the best match at/above `min_score`,
    else (None, best_score)."""
    incoming = _tokens(header_rows)
    if not incoming:
        return None, 0.0
    best, best_score = None, 0.0
    for t in templates:
        cand = _tokens(t.get("header_rows", []))
        if not cand:
            continue
        matched = sum(1 for w in cand if _best_sim(w, incoming) >= token_thresh)
        score = matched / len(cand)
        if score > best_score:
            best, best_score = t, score
    if best is not None and best_score >= min_score:
        return best, round(best_score, 2)
    return None, round(best_score, 2)


def resolve_header(table, templates):
    """Base-layer header resolution for one extracted table.

    If the table has a grouped (multi-row) header, try to recognize the form and
    stamp its canonical header on top; otherwise pass the table through untouched.
    Returns (table, info) where info = {template, undefined_form, match_score}:

      * recognized + stamped   -> corrected table,  undefined_form=False
      * recognized but columns don't line up (can't safely stamp) -> original
        table, undefined_form=True (flag the header for a human)
      * not recognized (grouped header, no template match) -> original table,
        undefined_form=True
      * flat header (no grouped structure) -> passes through, undefined_form=False

    This is the always-works safety net: a grouped header we can't map to a known
    form is never silently trusted -- it's marked for human header review.
    """
    header_rows = table.get("header_rows")
    if not header_rows:
        return table, {"template": None, "undefined_form": False, "match_score": 0.0}
    tpl, score = match_template(header_rows, templates)
    if tpl is not None:
        stamped = apply_template(table, tpl)
        if stamped is not None:
            return stamped, {"template": tpl["name"], "undefined_form": False,
                             "match_score": score}
        # recognized the form but the extraction's column count disagrees with
        # the template -> don't force a wrong alignment, hand it to a human
        return table, {"template": tpl["name"], "undefined_form": True,
                       "match_score": score}
    return table, {"template": None, "undefined_form": True, "match_score": score}


def apply_template(parsed, template):
    """Stamp a matched template's canonical header onto a parsed table: swap in
    the template's correct header_rows/header_merges (fixing garbled text AND any
    wonky spans the model produced) while keeping the data rows untouched.

    Requires the data width to equal the template width for a clean positional
    swap; on a width mismatch returns None so the caller can flag the form as
    recognized-but-misaligned for human review rather than forcing bad columns.
    """
    tpl_rows = template.get("header_rows") or []
    if not tpl_rows:
        return None
    width = max(len(r) for r in tpl_rows)
    data = parsed.get("rows", [])
    data_width = max((len(r) for r in data), default=0)
    if data_width != width:
        return None
    merges = [tuple(m) for m in template.get("header_merges", [])]
    return {
        **parsed,
        "headers": flatten_header(tpl_rows, merges),
        "header_rows": tpl_rows,
        "header_merges": merges,
        "template": template.get("name"),
    }
