import sys
from pathlib import Path
import pypdfium2 as pdfium

from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}

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
        textpage = pdf[i].get_textpage()
        text = textpage.get_text_range()
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

def build_ocr_converter():
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
        }
    )

def extract_single_page(src_path, page_idx, out_path):
    src = pdfium.PdfDocument(src_path)
    new = pdfium.PdfDocument.new()
    new.import_pages(src, [page_idx])
    new.save(out_path)
    new.close()
    src.close()

def route_and_parse(path, tmp_dir="./output/router_tmp"):
    kind = classify_input(path)
    print("[ROUTER] Girdi:", path, "-> tip:", kind)

    native_conv = build_native_converter()
    ocr_conv = build_ocr_converter()
    results = []

    if kind == "image":
        print("[ROUTER] Goruntu yolu -> OCR'li Docling")
        res = ocr_conv.convert(path)
        results.append(("image:OCR", res.document))
    elif kind == "pdf":
        page_types = analyze_pdf_pages(path)
        n_native = page_types.count("native")
        n_scan = page_types.count("scanned")
        print("[ROUTER]", len(page_types), "sayfa:", n_native, "native,", n_scan, "taranmis")
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        for idx, ptype in enumerate(page_types):
            page_no = idx + 1
            tmp_pdf = tmp_dir + "/page_" + str(page_no) + ".pdf"
            extract_single_page(path, idx, tmp_pdf)
            conv = native_conv if ptype == "native" else ocr_conv
            try:
                res = conv.convert(tmp_pdf, raises_on_error=True)
                results.append(("page" + str(page_no) + ":" + ptype, res.document))
                print("  sayfa", page_no, ":", ptype, "-> OK")
            except Exception as e:
                print("  sayfa", page_no, ":", ptype, "-> HATA:", str(e)[:80])
            finally:
                Path(tmp_pdf).unlink(missing_ok=True)
    else:
        print("[ROUTER] Desteklenmeyen tip:", path)

    return results

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/2024.pdf"
    docs = route_and_parse(target)
    print("\n[ROUTER] Toplam", len(docs), "sayfa/belge islendi.")
