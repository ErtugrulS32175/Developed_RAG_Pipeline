import os
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from pipeline import db
from pipeline.router import route_and_parse
from pipeline.table_export import (
    table_to_markdown,
    save_table_xlsx,
    save_table_csv,
    save_table_json,
    estimate_table_confidence,
    export_result_xlsx,
)
from pipeline.embeddings import embed_dense, embed_sparse
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
# Tables at or below this confidence, or with any validation issue, are copied
# to output/tables/_review/ and marked needs_review so a human checks them.
REVIEW_THRESHOLD = float(os.getenv("TABLE_REVIEW_THRESHOLD", "0.9"))

hf_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
tokenizer = HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=512)
chunker = HybridChunker(tokenizer=tokenizer)

def chunk_plain_text(text, source_tag, max_tokens=480):
    """Split plain OCR text into token-bounded chunks by paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current, cur_len = [], [], 0
    for p in paragraphs:
        p_len = len(hf_tok.encode(p, add_special_tokens=False))
        if cur_len + p_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, cur_len = [], 0
        current.append(p)
        cur_len += p_len
    if current:
        chunks.append(" ".join(current))
    return [{"type": "text", "text": c, "source_tag": source_tag, "page": 0, "headings": []} for c in chunks]

def _write_review_report(path, table_id, confidence, issues, table=None):
    """Human-readable note dropped next to a flagged table's copy in _review/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Tablo: {table_id}", f"Guven: {confidence:.2f}"]
    if table and table.get("mode") == "consensus":
        lines.append(f"Modeller: {'+'.join(table.get('backends', []))}")
        lines.append(f"Uyum: {table.get('agreement')}"
                     f" ({len(table.get('disagreements', []))} hucrede ayrisma)")
    lines.append("")
    if issues:
        lines.append("Sorunlar:")
        lines.extend(f"  - {x}" for x in issues)
    else:
        lines.append("Guven esigin altinda (issue listelenmedi).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_xlsx(table, headers, rows, path):
    """Pipeline results carry disagreement/review metadata, so they get the rich
    export (amber-highlighted cells + Rapor sheet). Anything else falls back to a
    plain sheet."""
    if "needs_review" in table:
        export_result_xlsx(table, str(path))
    else:
        save_table_xlsx(headers, rows, path)

def chunks_from_tables(tables, source_tag, doc_stem, filename):
    """Turn structured table results into RAG chunks and write xlsx/csv/json exports."""
    m = re.match(r"page(\d+)", source_tag)
    page_no = int(m.group(1)) if m else 0

    chunks = []
    tables_dir = OUTPUT_DIR / "tables"
    review_dir = tables_dir / "_review"
    for i, table in enumerate(tables):
        headers, rows = table["headers"], table["rows"]
        table_id = f"{doc_stem}_{source_tag.replace(':', '_')}_{i}"
        # the table pipeline already scored this table (number verification,
        # model agreement, template stage) and decided whether it needs review;
        # fall back to the shape-only proxy for tables from elsewhere.
        confidence = table.get("confidence")
        if confidence is None:
            confidence = estimate_table_confidence(headers, rows)
        issues = table.get("issues", [])
        needs_review = table.get("needs_review")
        if needs_review is None:
            needs_review = confidence < REVIEW_THRESHOLD or bool(issues)

        _save_xlsx(table, headers, rows, tables_dir / f"{table_id}.xlsx")
        save_table_csv(headers, rows, tables_dir / f"{table_id}.csv")
        save_table_json(table_id, page_no, headers, rows, confidence, tables_dir / f"{table_id}.json")

        if needs_review:
            _save_xlsx(table, headers, rows, review_dir / f"{table_id}.xlsx")
            _write_review_report(review_dir / f"{table_id}.issues.txt", table_id,
                                 confidence, issues, table)
            print(f"  [REVIEW] {table_id}: guven {confidence:.2f}, {len(issues)} sorun -> {review_dir}")

        # Carry the trust signals into the chunk, not just the values: a later
        # consumer (retrieval, an analytics layer, a reviewer UI) needs to know
        # which cells the two models disagreed on, not only what was extracted.
        table_data = {
            "table_id": table_id,
            "page": page_no,
            "headers": headers,
            "rows": rows,
            "confidence": confidence,
            "needs_review": needs_review,
            "issues": issues,
        }
        for key in ("mode", "backends", "agreement", "disagreements",
                    "structural_confidence", "number_fidelity", "template"):
            if key in table:
                table_data[key] = table[key]

        chunks.append({
            "type": "table",
            "text": table_to_markdown(
                headers, rows,
                filename=filename, page=page_no, table_id=table_id, confidence=confidence,
            ),
            "source_tag": source_tag,
            "page": page_no,
            "headings": [],
            "table_data": table_data,
        })
    return chunks

def main(path):
    filename = Path(path).name
    parts = route_and_parse(path)
    print(f"\n[INGEST] {len(parts)} parca parse edildi, chunk'laniyor...")

    all_chunks = []
    for source_tag, (content_type, content) in parts:
        if content_type == "docling":
            for chunk in chunker.chunk(content):
                ctype = "text"
                page_no = 0
                if chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if "table" in str(item.label).lower():
                            ctype = "table"
                    if chunk.meta.doc_items[0].prov:
                        page_no = chunk.meta.doc_items[0].prov[0].page_no
                all_chunks.append({
                    "type": ctype,
                    "text": chunk.text,
                    "source_tag": source_tag,
                    "page": page_no,
                    "headings": chunk.meta.headings or [],
                })
        elif content_type == "text":
            all_chunks.extend(chunk_plain_text(content, source_tag))
        elif content_type == "tables":
            all_chunks.extend(chunks_from_tables(content, source_tag, Path(path).stem, filename))

    print(f"[INGEST] Toplam {len(all_chunks)} chunk")

    conn = db.get_conn()
    db.init_schema(conn)
    document_id = db.upsert_document(conn, filename, Path(path).suffix.lower().lstrip("."))
    db.clear_chunks_for_document(conn, document_id)
    print(f"[INGEST] Belge: {filename} ({document_id})")

    try:
        batch = []
        for i, c in enumerate(all_chunks):
            if not c["text"].strip():
                continue
            sparse_indices, sparse_values = embed_sparse(c["text"])
            batch.append({
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "type": c["type"],
                "text": c["text"],
                "source_tag": c["source_tag"],
                "page": c["page"],
                "headings": c["headings"],
                "table_data": c.get("table_data"),
                "dense": embed_dense(c["text"]),
                "sparse": db.sparse_to_literal(sparse_indices, sparse_values),
            })
            if len(batch) >= 32:
                db.upsert_chunks(conn, batch)
                batch = []
                print(f"  {i+1}/{len(all_chunks)} yazildi...")
        if batch:
            db.upsert_chunks(conn, batch)
    except Exception:
        db.set_document_status(conn, document_id, "error")
        raise

    db.set_document_status(conn, document_id, "done")

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        count = cur.fetchone()[0]
    print(f"\n[INGEST] Tamamlandi. Bu belge icin vektor sayisi: {count}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/sample.pdf"
    main(target)
