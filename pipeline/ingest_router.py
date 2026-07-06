import os
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from pipeline import db
from pipeline.router import route_and_parse
from pipeline.table_export import table_to_markdown, save_table_xlsx, save_table_csv
from pipeline.embeddings import embed_dense, embed_sparse
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

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

def chunks_from_tables(tables, source_tag, doc_stem):
    """Turn structured table results into RAG chunks and write xlsx/csv exports."""
    m = re.match(r"page(\d+)", source_tag)
    page_no = int(m.group(1)) if m else 0

    chunks = []
    tables_dir = OUTPUT_DIR / "tables"
    for i, table in enumerate(tables):
        headers, rows = table["headers"], table["rows"]
        base_name = f"{doc_stem}_{source_tag.replace(':', '_')}_{i}"
        save_table_xlsx(headers, rows, tables_dir / f"{base_name}.xlsx")
        save_table_csv(headers, rows, tables_dir / f"{base_name}.csv")
        chunks.append({
            "type": "table",
            "text": table_to_markdown(headers, rows),
            "source_tag": source_tag,
            "page": page_no,
            "headings": [],
            "table_data": table,
        })
    return chunks

def main(path):
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
            all_chunks.extend(chunks_from_tables(content, source_tag, Path(path).stem))

    print(f"[INGEST] Toplam {len(all_chunks)} chunk")

    conn = db.get_conn()
    db.init_schema(conn)
    db.clear_chunks(conn)
    print("[INGEST] Tablo hazir: chunks")

    batch = []
    for i, c in enumerate(all_chunks):
        if not c["text"].strip():
            continue
        sparse_indices, sparse_values = embed_sparse(c["text"])
        batch.append({
            "id": str(uuid.uuid4()),
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

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        count = cur.fetchone()[0]
    print(f"\n[INGEST] Tamamlandi. Toplam vektor: {count}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/sample.pdf"
    main(target)
