"""Offline two-model A/B + consensus report from two pod_eval result files.

The pod only runs inference and dumps raw output (eval.pod_eval, once per adapter).
Everything below is offline (no GPU): re-parse each model's raw, reconcile the two
readings, score each vs GT, and write a consensus .xlsx with disagreements
highlighted. Parser/reconcile fixes never need the GPU re-rented.

    python -m eval.consensus_report output/eval/vl.json output/eval/hy.json
        [--primary paddleocr_vl] [--xlsx-dir output/eval]

The two files are eval.pod_eval outputs (they carry raw + per-model scores). Images
are matched by stem; GT (data/gt/<stem>.json) is used for scoring if present.
"""
import argparse
import json
import os

from eval import table_eval
from pipeline.consensus import reconcile
from pipeline.table_export import export_result_xlsx, parse_html_tables
from pipeline.text_normalize import normalize_tr


def _by_stem(path):
    return {r["image"]: r for r in json.load(open(path, encoding="utf-8"))}


def _table_from_row(row):
    """Re-parse the model's raw text (so the latest parser applies), normalize
    Turkish, and return the first table."""
    tabs = parse_html_tables(row.get("raw") or "")
    t = tabs[0] if tabs else {"headers": [], "rows": []}
    return {
        "headers": [normalize_tr(h) for h in t.get("headers", [])],
        "rows": [[normalize_tr(c) for c in r] for r in t.get("rows", [])],
    }


def _score_str(pred, stem):
    gt_path = os.path.join("data/gt", f"{stem}.json")
    if not os.path.exists(gt_path):
        return "(GT yok)"
    sc = table_eval.score(pred, json.load(open(gt_path, encoding="utf-8")))
    ca = "N/A" if sc["cell_acc"] is None else f"{sc['cell_acc']:.4f}"
    return f"TEDS={sc['teds']:.4f} num_fid={sc['number_fid']:.4f} cell_acc={ca}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primary_file", help="pod_eval json for the primary backend")
    ap.add_argument("secondary_file", help="pod_eval json for the secondary backend")
    ap.add_argument("--primary", default="paddleocr_vl")
    ap.add_argument("--secondary", default="hunyuan")
    ap.add_argument("--xlsx-dir", default="output/eval")
    a = ap.parse_args()

    prim, sec = _by_stem(a.primary_file), _by_stem(a.secondary_file)
    stems = [s for s in prim if s in sec]
    os.makedirs(a.xlsx_dir, exist_ok=True)
    print(f"ortak gorsel: {len(stems)}  (primary={a.primary}, secondary={a.secondary})\n")

    for stem in stems:
        pt = _table_from_row(prim[stem])
        st = _table_from_row(sec[stem])
        rec = reconcile(pt, st, a.primary, a.secondary)

        print(f"== {stem} ==")
        print(f"  {a.primary:<13} {_score_str(pt, stem)}  sekil={(len(pt['rows']), len(pt['headers']))}")
        print(f"  {a.secondary:<13} {_score_str(st, stem)}  sekil={(len(st['rows']), len(st['headers']))}")
        if rec["shape_match"]:
            print(f"  uyum={rec['agreement']}  ayrisan_hucre={len(rec['disagreements'])}")
        else:
            print(f"  SEKIL AYRISMASI: {rec['shape_primary']} vs {rec['shape_secondary']} "
                  f"-> tum tablo review")

        result = {
            "mode": "consensus",
            "backends": [a.primary, a.secondary],
            "headers": rec["headers"], "rows": rec["rows"],
            "agreement": rec["agreement"], "shape_match": rec["shape_match"],
            "structural_confidence": None, "number_fidelity": None,
            "confidence": rec["agreement"],
            "needs_review": bool(rec["disagreements"]) or not rec["shape_match"],
            "issues": [], "disagreements": rec["disagreements"],
        }
        out = os.path.join(a.xlsx_dir, f"{stem}_consensus.xlsx")
        export_result_xlsx(result, out)
        print(f"  -> {out}\n")


if __name__ == "__main__":
    main()
