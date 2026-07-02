import sys
import requests
from pathlib import Path
import pypdfium2 as pdfium

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}
PADDLE_OCR_URL = "http://127.0.0.1:8100/ocr"

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

def ocr_via_paddle(image_path):
    abs_path = str(Path(image_path).resolve())
    r = requests.post(PADDLE_OCR_URL, json={"image_path": abs_path}, timeout=120)
    r.raise_for_status()
    return r.json()["text"]

def route_and_parse(path, tmp_dir="./output/router_tmp"):
    kind = classify_input(path)
    print("[ROUTER] Girdi:", path, "-> tip:", kind)
    native_conv = build_native_converter()
    results = []  # (source_tag, ("docling", DoclingDocument) | ("text", str))

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    if kind == "image":
        print("[ROUTER] Goruntu yolu -> PaddleOCR servisi")
        text = ocr_via_paddle(path)
        results.append(("image:ocr", ("text", text)))

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
                    text = ocr_via_paddle(tmp_img)
                    results.append(("page" + str(page_no) + ":scanned", ("text", text)))
                    Path(tmp_img).unlink(missing_ok=True)
                    print("  sayfa", page_no, ": scanned -> OCR OK")
            except Exception as e:
                print("  sayfa", page_no, ":", ptype, "-> HATA:", str(e)[:80])
    else:
        print("[ROUTER] Desteklenmeyen tip:", path)

    return results

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/2024.pdf"
    docs = route_and_parse(target)
    print("\n[ROUTER] Toplam", len(docs), "parca islendi.")
