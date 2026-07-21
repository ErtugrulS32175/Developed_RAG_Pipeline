"""Backend-agnostic scanned-table -> structured-table pipeline (production flow).

    scanned image
        |
    1) PREPROCESS        deskew + contrast + upscale        (pipeline/image_preprocess)
        |
    2) TABLE BACKEND     PaddleOCR-VL | HunyuanOCR | Gemma | Docling | TATR
        |                (pluggable via router.TABLE_BACKENDS; VLM = structure,
        |                 generalizes across varied/complex table formats)
    3) NUMBER VERIFY     cross-check numeric cells vs deterministic OCR reading
        |                (pipeline/number_verify -> catches VLM digit hallucination)
    4) NORMALIZE + VALIDATE   Turkish chars + structural checks   (table_export)
        |
    5) CONFIDENCE + HUMAN-IN-THE-LOOP   weakest-link score, low -> review, none dropped
        |
    {headers, rows, confidence, issues, needs_review}  ->  Excel / DB

The VLM gives structure (consistent across formats), the deterministic number
check guards financial fidelity, and the confidence layer routes the uncertain
tail to a human. Swap the table engine with TABLE_BACKEND -- every downstream
stage is engine-agnostic. Preprocessing is owned HERE (run backend services with
PREPROCESS=0) so the VLM and the verification OCR read the exact same image.
"""
import os
import sys
import tempfile

from PIL import Image

from pipeline import consensus
from pipeline import header_templates
from pipeline import image_preprocess as ip
from pipeline import number_verify
from pipeline import router
from pipeline.text_normalize import normalize_tr
from pipeline.table_export import export_result_xlsx, validate_table

# Two small VLMs, cross-checked: same image through both, agree -> auto-accept,
# disagree -> that cell to human review (see pipeline/consensus.py).
CONSENSUS_BACKENDS = tuple(
    os.getenv("CONSENSUS_BACKENDS", "paddleocr_vl,hunyuan").split(","))

REVIEW_THRESHOLD = float(os.getenv("TABLE_REVIEW_THRESHOLD", "0.9"))
# Preprocessing profile per backend: the deterministic engine (small models) gets
# deskew+contrast to resolve faint/skewed cells; VLM backends handle resolution
# internally and an enhanced image can HURT them (Granite misclassifies a large
# table as a picture -> empty), so VLMs get the raw image by default.
_DETERMINISTIC = {"tatr"}
# Upscale is OFF by default (scale=1.0): it breaks TATR structure detection --
# even 1.25x merges a sample table's narrow first column into the second (cell
# accuracy drops). Upscale was only ever needed to spread rows for the
# y-clustering step on tiny/dense scans, NOT for TATR; set PREPROCESS_SCALE>1
# there. deskew+denoise+CLAHE stay on (geometry-safe, lift faint text).
PREPROCESS_SCALE = float(os.getenv("PREPROCESS_SCALE", "1.0"))


def _finalize(table, ocr_text, backend, review_threshold, templates=()):
    """Stages 3-5 on one raw {headers, rows} from a backend. If the table has a
    grouped header, first try to recognize the form and stamp its canonical
    header (base-layer safety net); an unrecognized grouped header is flagged for
    human header review rather than trusted."""
    table, hdr = header_templates.resolve_header(table, templates)
    headers = [normalize_tr(h) for h in table.get("headers", [])]
    rows = [[normalize_tr(c) for c in row] for row in table.get("rows", [])]

    # validate_table does STRUCTURAL checks only here (width, empty rows, residual
    # Turkish marks) -- NOT the general OCR cross-check, which false-positives on
    # text cells (name ordering) and is redundant with the precise numeric check.
    struct_conf, issues = validate_table(headers, rows)
    num_fidelity, num_flags = number_verify.verify(headers, rows, ocr_text)
    issues = list(issues) + number_verify.flags_to_messages(num_flags, headers)

    # weakest-link: a table is only as trustworthy as its shakiest signal. For
    # financial data we'd rather over-flag than pass a bad number silently.
    confidence = round(min(struct_conf, num_fidelity), 3)
    needs_review = confidence < review_threshold or bool(issues)

    result = {
        "backend": backend,
        "headers": headers,
        "rows": rows,
        "confidence": confidence,
        "structural_confidence": struct_conf,
        "number_fidelity": num_fidelity,
    }
    # keep the two-level structure so the exporter can rebuild a merged header
    if table.get("header_rows"):
        result["header_rows"] = [[normalize_tr(c) for c in r] for r in table["header_rows"]]
        result["header_merges"] = table.get("header_merges", [])
    if hdr["template"]:
        result["template"] = hdr["template"]
    if hdr["undefined_form"]:
        issues = list(issues) + ["tanimlanmamis form - basliklari kontrol edin"]
        result["review_all_headers"] = True
        needs_review = True

    result["issues"] = issues
    result["needs_review"] = needs_review
    return result


def run(image_path, backend=None, preprocess=None, review_threshold=REVIEW_THRESHOLD):
    """Run the full flow on one image. `backend` overrides TABLE_BACKEND.
    `preprocess=None` auto-selects by backend (deterministic=on, VLM=off).
    Returns a list of finalized table dicts (one per detected table)."""
    backend = (backend or router.TABLE_BACKEND).lower()
    if preprocess is None:
        preprocess = backend in _DETERMINISTIC
    work = image_path
    tmp = None
    if preprocess:
        enhanced = ip.enhance(Image.open(image_path).convert("RGB"), scale=PREPROCESS_SCALE)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        enhanced.save(tmp.name)
        work = tmp.name
    try:
        # deterministic reading of the SAME image the backend sees (number verify
        # + OCR cross-check both rely on this being pixel-faithful on digits)
        ocr_text = router.ocr_via_paddle(work)
        raw_tables = router.tables_from_image(work, backend)
        templates = header_templates.load_templates()
        return [_finalize(t, ocr_text, backend, review_threshold, templates)
                for t in raw_tables]
    finally:
        if tmp is not None:
            os.unlink(tmp.name)


def _normalize_table(table):
    """normalize_tr every cell so the two backends are compared on clean Turkish
    text (residual-mark differences shouldn't register as disagreement). A
    grouped header's structure (header_rows/header_merges) is carried through for
    template matching + faithful merged export."""
    out = {
        "headers": [normalize_tr(h) for h in table.get("headers", [])],
        "rows": [[normalize_tr(c) for c in row] for row in table.get("rows", [])],
    }
    if table.get("header_rows"):
        out["header_rows"] = [[normalize_tr(c) for c in r] for r in table["header_rows"]]
        out["header_merges"] = table.get("header_merges", [])
    return out


def _finalize_consensus(rec, ocr_text, backends, review_threshold, templates=()):
    """Stages 3-5 on a reconciled (two-backend) table. Confidence is the weakest
    of {structural, numeric fidelity, model agreement}; ANY disagreement or shape
    mismatch forces review -- nothing is auto-accepted where the models differ.
    A grouped header is run through the form-template stage first (recognized ->
    stamp canonical header; unrecognized -> flag for human header review)."""
    table = {"headers": rec["headers"], "rows": rec["rows"]}
    if rec.get("header_rows"):
        table["header_rows"] = rec["header_rows"]
        table["header_merges"] = rec.get("header_merges", [])
    table, hdr = header_templates.resolve_header(table, templates)
    headers, rows = table["headers"], table["rows"]

    struct_conf, issues = validate_table(headers, rows)
    num_fidelity, num_flags = number_verify.verify(headers, rows, ocr_text)
    issues = list(issues) + number_verify.flags_to_messages(num_flags, headers)

    if not rec["shape_match"]:
        issues.append(f"modeller farkli sekil verdi: {rec['shape_primary']} vs "
                      f"{rec['shape_secondary']} (yapisal ayrisma)")
    elif rec["disagreements"]:
        issues.append(f"{len(rec['disagreements'])} hucrede modeller ayristi "
                      f"(insan gozden gecirmeli)")

    confidence = round(min(struct_conf, num_fidelity, rec["agreement"]), 3)
    needs_review = (confidence < review_threshold or bool(issues)
                    or not rec["shape_match"])

    # a recognized (stamped) header is trusted, so its per-column disagreements
    # are moot -- drop them so the correct header isn't flagged
    disagreements = rec["disagreements"]
    if hdr["template"] and not hdr["undefined_form"]:
        disagreements = [d for d in disagreements if d.get("kind") != "header"]

    result = {
        "mode": "consensus",
        "backends": list(backends),
        "headers": headers,
        "rows": rows,
        "confidence": confidence,
        "agreement": rec["agreement"],
        "structural_confidence": struct_conf,
        "number_fidelity": num_fidelity,
        "shape_match": rec["shape_match"],
        "disagreements": disagreements,
    }
    if table.get("header_rows"):
        result["header_rows"] = table["header_rows"]
        result["header_merges"] = table.get("header_merges", [])
    if hdr["template"]:
        result["template"] = hdr["template"]
    if hdr["undefined_form"]:
        issues = list(issues) + ["tanimlanmamis form - basliklari kontrol edin"]
        result["review_all_headers"] = True
        needs_review = True

    result["issues"] = issues
    result["needs_review"] = needs_review
    return result


def run_consensus(image_path, backends=CONSENSUS_BACKENDS, preprocess=False,
                  review_threshold=REVIEW_THRESHOLD):
    """Cross-check two backends on one image. backends[0] is primary (its value
    wins where they agree). Both are VLMs -> no preprocessing by default. Returns
    a list of consensus table dicts (one per detected table)."""
    prim_be, sec_be = backends[0], backends[1]
    work = image_path
    tmp = None
    if preprocess:
        enhanced = ip.enhance(Image.open(image_path).convert("RGB"), scale=PREPROCESS_SCALE)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        enhanced.save(tmp.name)
        work = tmp.name
    try:
        ocr_text = router.ocr_via_paddle(work)
        prim = [_normalize_table(t) for t in router.tables_from_image(work, prim_be)]
        sec = [_normalize_table(t) for t in router.tables_from_image(work, sec_be)]
        templates = header_templates.load_templates()
        results = []
        for i in range(max(len(prim), len(sec))):
            a = prim[i] if i < len(prim) else {"headers": [], "rows": []}
            b = sec[i] if i < len(sec) else {"headers": [], "rows": []}
            rec = consensus.reconcile(a, b, prim_be, sec_be)
            # carry a grouped header (whichever model produced one) so the
            # template stage can recognize the form
            src = a if a.get("header_rows") else (b if b.get("header_rows") else None)
            if src is not None:
                rec["header_rows"] = src["header_rows"]
                rec["header_merges"] = src.get("header_merges", [])
            results.append(
                _finalize_consensus(rec, ocr_text, backends, review_threshold, templates))
        return results
    finally:
        if tmp is not None:
            os.unlink(tmp.name)


def _print_report(results):
    if not results:
        print("[PIPELINE] tablo bulunamadi")
        return
    for i, t in enumerate(results):
        flag = "REVIEW" if t["needs_review"] else "OK"
        engine = "+".join(t["backends"]) if t.get("mode") == "consensus" else t["backend"]
        extra = f", uyum={t['agreement']}" if t.get("mode") == "consensus" else ""
        print(f"\n[{flag}] tablo {i} | backend={engine} | "
              f"guven={t['confidence']} (yapi={t['structural_confidence']}, "
              f"sayi={t['number_fidelity']}{extra}) | {len(t['rows'])}x{len(t['headers'])}")
        if t["issues"]:
            print("  sorunlar:")
            for x in t["issues"][:8]:
                print("   -", x)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("kullanim: python -m pipeline.table_pipeline <image> [backend]")
        sys.exit(1)
    img = sys.argv[1]
    be = sys.argv[2] if len(sys.argv) > 2 else None
    if be == "consensus":
        print(f"[PIPELINE] {img} | consensus={'+'.join(CONSENSUS_BACKENDS)}")
        results = run_consensus(img)
    else:
        print(f"[PIPELINE] {img} | backend={be or router.TABLE_BACKEND}")
        results = run(img, backend=be)
    _print_report(results)

    # TABLE_XLSX=out.xlsx -> write the deliverable (one file per detected table)
    xlsx = os.getenv("TABLE_XLSX")
    if xlsx and results:
        base = xlsx[:-5] if xlsx.lower().endswith(".xlsx") else xlsx
        for i, t in enumerate(results):
            path = f"{base}.xlsx" if len(results) == 1 else f"{base}_{i}.xlsx"
            export_result_xlsx(t, path)
            print(f"  -> {path}")
