"""Self-contained ON-POD A/B eval for the Gemma table backend (rented-GPU friendly).

Runs ENTIRELY on the RunPod box against the local vLLM adapter (:8101):
  * no SSH tunnel, no local PaddleOCR, no pipeline/router import (those drag in
    docling / pypdfium2 / paddle -- none of which belong on the GPU box);
  * POSTs each image straight to the adapter and scores vs ground truth;
  * ALWAYS persists the raw model output for every image, so if parsing or
    scoring is off we fix it OFFLINE and re-score -- we never re-rent the GPU
    just to re-inspect an answer.

Because the expensive resource is the GPU and it is done the moment vLLM has
answered the N images, this does all images in one shot and writes everything to
output/eval/pod_results.json (raw text + every parsed table + scores + timing).

    gemma_env/bin/python -m eval.pod_eval
    ADAPTER_URL=http://127.0.0.1:8101/table gemma_env/bin/python -m eval.pod_eval

Images that have a GT are scored (TEDS / number_fid / cell_acc); images without a
GT yet are dumped (raw + parsed) for offline scoring later -- pass those via the
POD_EVAL_RAW env var so no dataset filenames live in the repo.
"""
import glob
import json
import os
import time

import requests

from eval import table_eval

ADAPTER_URL = os.getenv("ADAPTER_URL", "http://127.0.0.1:8101/table")
GT_DIR = "data/gt"
OUT_DIR = "output/eval"
# Override per backend so a two-model A/B (run once per adapter) doesn't clobber
# itself: POD_EVAL_OUT=output/eval/vl.json / .../hy.json
OUT = os.getenv("POD_EVAL_OUT", os.path.join(OUT_DIR, "pod_results.json"))
# GT-less images to dump (raw + parsed) for later offline scoring. Kept OUT of the
# repo: set POD_EVAL_RAW="data/a.jpeg,data/b.jpeg" (comma-separated) at run time.
RAW_IMAGES = [p.strip() for p in os.getenv("POD_EVAL_RAW", "").split(",") if p.strip()]
TIMEOUT = float(os.getenv("EVAL_TIMEOUT", "900"))


def _predict(img):
    """POST one image to the adapter. Returns (tables, raw_text, secs)."""
    t0 = time.time()
    with open(img, "rb") as f:
        r = requests.post(
            ADAPTER_URL,
            files={"file": (os.path.basename(img), f, "application/octet-stream")},
            timeout=TIMEOUT,
        )
    r.raise_for_status()
    d = r.json()
    return d.get("tables") or [], d.get("raw", ""), round(time.time() - t0, 1)


def _load_gt_jobs(only=None):
    """(stem, image_path, gt_dict) for every data/gt/*.json whose source exists."""
    jobs = []
    for path in sorted(glob.glob(os.path.join(GT_DIR, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if only and stem not in only:
            continue
        gt = json.load(open(path, encoding="utf-8"))
        img = gt.get("source")
        if not img or not os.path.exists(img):
            print(f"[ATLA] {stem}: kaynak gorsel yok ({img})")
            continue
        jobs.append((stem, img, gt))
    return jobs


def run():
    jobs = _load_gt_jobs()
    for p in RAW_IMAGES:
        if os.path.exists(p):
            jobs.append((os.path.splitext(os.path.basename(p))[0], p, None))
        else:
            print(f"[ATLA] GT'siz gorsel yok: {p}")

    results = []
    for stem, img, gt in jobs:
        row = {"image": stem, "source": img}
        try:
            tables, raw, secs = _predict(img)
            row["secs"] = secs
            row["raw"] = raw                       # ALWAYS kept -> offline re-parse
            row["n_tables"] = len(tables)
            row["tables"] = tables                 # every parsed table, not just [0]
            pred = tables[0] if tables else {"headers": [], "rows": []}
            row["pred_shape"] = [len(pred.get("rows", [])), len(pred.get("headers", []))]
            if gt is not None:
                row["scored"] = True
                row.update(table_eval.score(pred, gt))
            else:
                row["scored"] = False
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        results.append(row)
        _print_row(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {OUT}  (ham cikti dahil; parse/skor offline duzeltilebilir)")
    return results


def _print_row(r):
    if "error" in r:
        print(f"  {r['image']:<10}  HATA: {r['error']}")
        return
    raw_n = len(r.get("raw", "") or "")
    base = (f"  {r['image']:<10}  {r['n_tables']} tablo  pred{tuple(r['pred_shape'])}  "
            f"raw={raw_n}ch  {r.get('secs','?')}s")
    if r.get("scored"):
        ca = "  N/A " if r["cell_acc"] is None else f"{r['cell_acc']:.4f}"
        print(base + f"  |  TEDS={r['teds']:.4f}  num_fid={r['number_fid']:.4f}  "
                     f"cell_acc={ca}  sekil={'=' if r['shape_match'] else 'x'} gt{tuple(r['gt_shape'])}")
    else:
        print(base + "  |  (GT yok, kaydedildi)")


if __name__ == "__main__":
    print(f"adapter={ADAPTER_URL}\n")
    run()
