"""A/B eval runner: run each backend on each labeled image, score vs ground truth.

    python -m eval.run_eval                      # all GTs x default backends
    python -m eval.run_eval --backends tatr docling gemma
    python -m eval.run_eval --images sample1

Prints a comparison table (TEDS / number-fidelity / cell-accuracy) and writes the
full results to output/eval/results.json. Backends are reached exactly as in
production (router.TABLE_BACKENDS URLs), so pointing a URL at a RunPod tunnel runs
the same harness against a cloud VLM with no code change.
"""
import argparse
import glob
import json
import os
import time

from eval import table_eval
from pipeline import table_pipeline

GT_DIR = "data/gt"
OUT = "output/eval"
# ground-truth stem -> source image (a GT json's "source" field wins if present)
DEFAULT_BACKENDS = ["tatr", "docling"]


def _load_gts(only=None):
    gts = {}
    for path in sorted(glob.glob(os.path.join(GT_DIR, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if only and stem not in only:
            continue
        gt = json.load(open(path, encoding="utf-8"))
        img = gt.get("source")
        if not img or not os.path.exists(img):
            print(f"[SKIP] {stem}: kaynak gorsel yok ({img})")
            continue
        gts[stem] = (img, gt)
    return gts


def run(backends, images=None, raw_images=None):
    # GT-backed jobs (scored) + raw image paths (predictions dumped, not scored --
    # for harvesting a RunPod session's VLM outputs on images not yet labeled, so
    # we can score them offline later without renting the GPU again).
    jobs = [(stem, img, gt) for stem, (img, gt) in _load_gts(images).items()]
    for p in (raw_images or []):
        jobs.append((os.path.splitext(os.path.basename(p))[0], p, None))
    if not jobs:
        print("Is yok: data/gt/*.json bos ve --raw verilmedi.")
        return []
    results = []
    for stem, img, gt in jobs:
        if not os.path.exists(img):
            print(f"[SKIP] {stem}: gorsel yok ({img})"); continue
        for be in backends:
            row = {"image": stem, "backend": be}
            try:
                t0 = time.time()
                tables = table_pipeline.run(img, backend=be)
                row["secs"] = round(time.time() - t0, 1)
                pred = tables[0] if tables else {"headers": [], "rows": []}
                if gt is not None:
                    row.update(table_eval.score(pred, gt))
                else:
                    row["scored"] = False  # no GT yet -- dumped for offline scoring
                row["pipeline_conf"] = pred.get("confidence")
                row["needs_review"] = pred.get("needs_review")
                # keep the prediction so a bad score is debuggable without re-running
                row["pred"] = {"headers": pred.get("headers", []), "rows": pred.get("rows", [])}
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            results.append(row)
            _print_row(row)
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n-> {os.path.join(OUT, 'results.json')}")
    return results


def _fmt(v):
    return "  N/A" if v is None else f"{v:.4f}"


def _print_row(r):
    if "error" in r:
        print(f"  {r['image']:<10} {r['backend']:<8}  HATA: {r['error']}")
        return
    if r.get("scored") is False:
        p = r["pred"]
        print(f"  {r['image']:<10} {r['backend']:<8}  (GT yok, kaydedildi)  "
              f"pred=({len(p['rows'])}x{len(p['headers'])})  conf={r.get('pipeline_conf')}  {r.get('secs','?')}s")
        return
    ca = "  N/A " if r["cell_acc"] is None else f"{r['cell_acc']:.4f}"
    print(f"  {r['image']:<10} {r['backend']:<8}  TEDS={r['teds']:.4f}  "
          f"num_fid={r['number_fid']:.4f}  cell_acc={ca}  "
          f"sekil={'=' if r['shape_match'] else 'x'} "
          f"pred{r['pred_shape']} gt{r['gt_shape']}  {r.get('secs','?')}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    ap.add_argument("--images", nargs="+", default=None,
                    help="GT stem adlari; vermezsen hepsi")
    ap.add_argument("--raw", nargs="+", default=None,
                    help="GT'siz gorsel yollari: koss + tahmini kaydet (skorlama yok), "
                         "GT sonra hazir olunca offline skorlanir")
    a = ap.parse_args()
    print(f"backends={a.backends}  images={a.images or 'hepsi'}  raw={a.raw or '-'}\n")
    run(a.backends, a.images, a.raw)
