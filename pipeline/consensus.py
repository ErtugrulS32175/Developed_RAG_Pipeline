"""Dual-model consensus: run one image through two independent small VLM backends
and reconcile their tables cell-by-cell. Agreement -> auto-accept (two independent
~0.99 readers concurring is near-certain); disagreement -> flag that cell for a
human with both candidates. Structural mismatch (different shapes) -> whole table
to review, since the models can't even be aligned.

The two backends have DIFFERENT architectures (layout-detect + unwarp + VL vs.
end-to-end OCR VLM), so their error modes are largely independent -- the case a
cross-check catches. It does NOT catch correlated errors (both misread a cell the
same way), so the numeric cross-check (number_verify) stays as a third, independent
signal for financial cells. Pure functions here (no I/O) so they unit-test offline.
"""
from pipeline.table_export import _squash


def _eq(x, y) -> bool:
    """Cells agree if they match after Turkish-folding + format-stripping, so
    ı/i and 1.000,50 vs 1000.50 don't count as disagreement (that noise would
    otherwise flood human review)."""
    return _squash(x) == _squash(y)


def reconcile(primary, secondary, prim_name="A", sec_name="B"):
    """Reconcile two {headers, rows} readings of the same table. `primary` supplies
    the displayed value when the two agree. Returns a merged table plus a per-cell
    disagreement list (each with both candidates) and an agreement ratio."""
    ph, pr = primary.get("headers", []), primary.get("rows", [])
    sh, sr = secondary.get("headers", []), secondary.get("rows", [])
    shape_p, shape_s = (len(pr), len(ph)), (len(sr), len(sh))
    disagreements = []

    if shape_p != shape_s:
        # Can't align cell-wise -- structural disagreement is itself the signal.
        return {
            "shape_match": False,
            "shape_primary": shape_p, "shape_secondary": shape_s,
            "headers": ph, "rows": pr,           # show primary's reading
            "disagreements": [{"kind": "shape", prim_name: shape_p, sec_name: shape_s}],
            "agreement": 0.0,
            "review_mask_headers": [True] * len(ph),
            "review_mask_rows": [[True] * len(r) for r in pr],
        }

    headers, hmask = [], []
    for j, (x, y) in enumerate(zip(ph, sh)):
        ok = _eq(x, y)
        headers.append(x)
        hmask.append(not ok)
        if not ok:
            disagreements.append({"kind": "header", "pos": j, prim_name: x, sec_name: y})

    rows, rmask = [], []
    for i, (rp, rs) in enumerate(zip(pr, sr)):
        mrow, mmask = [], []
        for j, (x, y) in enumerate(zip(rp, rs)):
            ok = _eq(x, y)
            mrow.append(x)
            mmask.append(not ok)
            if not ok:
                disagreements.append({"kind": "cell", "pos": (i, j), prim_name: x, sec_name: y})
        rows.append(mrow)
        rmask.append(mmask)

    total = len(headers) + sum(len(r) for r in rows)
    agreement = round((total - len(disagreements)) / total, 4) if total else 0.0
    return {
        "shape_match": True,
        "shape_primary": shape_p, "shape_secondary": shape_s,
        "headers": headers, "rows": rows,
        "disagreements": disagreements,
        "agreement": agreement,
        "review_mask_headers": hmask,
        "review_mask_rows": rmask,
    }
