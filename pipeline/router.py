import os
import sys
import requests
from pathlib import Path
import pypdfium2 as pdfium

from dotenv import load_dotenv

from pipeline.text_normalize import normalize_tr
from pipeline.table_export import validate_table, parse_table_json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# router is imported by ingest_router before *it* calls load_dotenv(), so read
# the env here on our own import or the URLs below would be frozen to defaults.
load_dotenv()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}
# Overridable so paddle/gemma can be hosted off-box (RunPod, another dev's
# machine, a container) without touching code -- see .env.example.
PADDLE_OCR_URL = os.getenv("PADDLE_OCR_URL", "http://127.0.0.1:8100/ocr")
GEMMA_TABLE_URL = os.getenv("GEMMA_TABLE_URL", "http://127.0.0.1:8101/table")
TATR_TABLE_URL = os.getenv("TATR_TABLE_URL", "http://127.0.0.1:8102/table")
# Which table engine to use: "tatr" = deterministic TATR+OCR (no hallucination,
# best for financial tables), "gemma" = VLM. Default tatr.
TABLE_BACKEND = os.getenv("TABLE_BACKEND", "tatr").lower()
# Remote services over a tunnel can be far slower than local; make it tunable.
SERVICE_TIMEOUT = float(os.getenv("SERVICE_TIMEOUT", "120"))

def classify_input(path):
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    return "unknown"

def analyze_pdf_pages(path, min_chars=10):
    pdf = pdfium.PdfDocument(path)
    page_types = []
    for i in range(len(pdf)):
        text = pdf[i].get_textpage().get_text_range()
        page_types.append("native" if len(text.strip()) >= min_chars else "scanned")
    pdf.close()
    return page_types

def build_native_converter():
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

def extract_single_page(src_path, page_idx, out_path):
    src = pdfium.PdfDocument(src_path)
    new = pdfium.PdfDocument.new()
    new.import_pages(src, [page_idx])
    new.save(out_path)
    new.close()
    src.close()

def render_page_to_image(src_path, page_idx, out_path, scale=2.0):
    pdf = pdfium.PdfDocument(src_path)
    img = pdf[page_idx].render(scale=scale).to_pil()
    img.save(out_path)
    pdf.close()

def _describe_service_error(e):
    """Turn a requests exception into a message that says *why* it failed, so a
    down service (ConnectionError) reads differently from a model crash (500)."""
    if isinstance(e, requests.ConnectionError):
        return "baglanti kurulamadi (servis calisiyor mu / URL dogru mu?)"
    if isinstance(e, requests.Timeout):
        return "zaman asimi (SERVICE_TIMEOUT'u artirmayi dene)"
    if isinstance(e, requests.HTTPError) and e.response is not None:
        return f"servis hatasi HTTP {e.response.status_code}"
    return str(e)[:80]

def ocr_via_paddle(image_path):
    # Upload the file itself, not a path -- a path is meaningless to a service
    # on a different machine/filesystem (RunPod, container, another host).
    with open(image_path, "rb") as f:
        r = requests.post(
            PADDLE_OCR_URL,
            files={"file": (Path(image_path).name, f, "application/octet-stream")},
            timeout=SERVICE_TIMEOUT,
        )
    r.raise_for_status()
    return r.json()["text"]

def tables_via_gemma(image_path):
    with open(image_path, "rb") as f:
        r = requests.post(
            GEMMA_TABLE_URL,
            files={"file": (Path(image_path).name, f, "application/octet-stream")},
            timeout=SERVICE_TIMEOUT,
        )
    r.raise_for_status()
    data = r.json()
    # Parse the raw model text client-side (robust to VLM JSON quirks like fenced
    # output and botched trailing brackets) so parser fixes never require a
    # redeploy of the GPU table service. Fall back to the service's own parse
    # for older services that don't return "raw".
    raw = data.get("raw")
    if raw is not None:
        return [parse_table_json(raw)]
    return data.get("tables", [])

def tables_via_tatr(image_path):
    """Deterministic backend: TATR detects + crops the table, paddle_service OCRs
    it, and tatr_service rebuilds the grid. Returns [{headers, rows}] already
    structured (no client-side JSON parsing needed, unlike the VLM path)."""
    with open(image_path, "rb") as f:
        r = requests.post(
            TATR_TABLE_URL,
            files={"file": (Path(image_path).name, f, "application/octet-stream")},
            timeout=SERVICE_TIMEOUT,
        )
    r.raise_for_status()
    return r.json().get("tables", [])

def tables_from_image(image_path):
    """Dispatch to the configured table backend (TABLE_BACKEND)."""
    if TABLE_BACKEND == "gemma":
        return tables_via_gemma(image_path)
    return tables_via_tatr(image_path)

def _finalize_table(table, ocr_text):
    """Normalize Turkish characters in every cell, then validate the table
    against the same page's OCR text. Attaches confidence + issues so ingest
    can route low-confidence tables to human review. Corrects mechanically but
    only *flags* content problems -- nothing is dropped."""
    headers = [normalize_tr(h) for h in table.get("headers", [])]
    rows = [[normalize_tr(c) for c in row] for row in table.get("rows", [])]
    confidence, issues = validate_table(headers, rows, ocr_text=ocr_text)
    return {"headers": headers, "rows": rows, "confidence": confidence, "issues": issues}

def route_and_parse(path, tmp_dir="./output/router_tmp"):
    kind = classify_input(path)
    print("[ROUTER] Girdi:", path, "-> tip:", kind)
    native_conv = build_native_converter()
    results = []  # (source_tag, ("docling", DoclingDocument) | ("text", str))

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    if kind == "image":
        print("[ROUTER] Goruntu yolu -> PaddleOCR servisi")
        text = normalize_tr(ocr_via_paddle(path))
        results.append(("image:ocr", ("text", text)))
        try:
            tables = [_finalize_table(t, text) for t in tables_from_image(path)]
            if tables:
                results.append(("image:tables", ("tables", tables)))
                print("  ", len(tables), "tablo bulundu")
        except Exception as e:
            print("  tablo tespiti -> HATA:", _describe_service_error(e))

    elif kind == "pdf":
        page_types = analyze_pdf_pages(path)
        n_native = page_types.count("native")
        n_scan = page_types.count("scanned")
        print("[ROUTER]", len(page_types), "sayfa:", n_native, "native,", n_scan, "taranmis")
        for idx, ptype in enumerate(page_types):
            page_no = idx + 1
            try:
                if ptype == "native":
                    tmp_pdf = tmp_dir + "/page_" + str(page_no) + ".pdf"
                    extract_single_page(path, idx, tmp_pdf)
                    res = native_conv.convert(tmp_pdf, raises_on_error=True)
                    results.append(("page" + str(page_no) + ":native", ("docling", res.document)))
                    Path(tmp_pdf).unlink(missing_ok=True)
                    print("  sayfa", page_no, ": native -> OK")
                else:
                    tmp_img = tmp_dir + "/page_" + str(page_no) + ".png"
                    render_page_to_image(path, idx, tmp_img)
                    text = normalize_tr(ocr_via_paddle(tmp_img))
                    results.append(("page" + str(page_no) + ":scanned", ("text", text)))
                    try:
                        tables = [_finalize_table(t, text) for t in tables_from_image(tmp_img)]
                        if tables:
                            results.append(("page" + str(page_no) + ":tables", ("tables", tables)))
                            print("  sayfa", page_no, ":", len(tables), "tablo bulundu")
                    except Exception as e:
                        print("  sayfa", page_no, ": tablo tespiti -> HATA:", _describe_service_error(e))
                    Path(tmp_img).unlink(missing_ok=True)
                    print("  sayfa", page_no, ": scanned -> OCR OK")
            except Exception as e:
                print("  sayfa", page_no, ":", ptype, "-> HATA:", _describe_service_error(e))
    else:
        print("[ROUTER] Desteklenmeyen tip:", path)

    return results

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/2024.pdf"
    docs = route_and_parse(target)
    print("\n[ROUTER] Toplam", len(docs), "parca islendi.")
