import os
import re
import sys
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

from router import route_and_parse
from table_export import table_to_markdown, save_table_xlsx, save_table_csv
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseIndexParams
from qdrant_client.http.models import SparseVector
from fastembed import SparseTextEmbedding

load_dotenv()

QDRANT_URL       = os.getenv("QDRANT_URL")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME  = "rag_router_test"
EMBED_API_URL    = os.getenv("EMBED_API_URL", "http://localhost:8011/v1/embeddings")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "./output"))

hf_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
tokenizer = HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=512)
chunker = HybridChunker(tokenizer=tokenizer)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def embed_dense(text):
    r = requests.post(EMBED_API_URL, json={"model": EMBED_MODEL_NAME, "input": text[:2000]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def embed_sparse(text):
    res = list(sparse_model.embed([text]))[0]
    return SparseVector(indices=res.indices.tolist(), values=res.values.tolist())

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

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
    )
    print(f"[INGEST] Collection hazir: {COLLECTION_NAME}")

    batch = []
    for i, c in enumerate(all_chunks):
        if not c["text"].strip():
            continue
        batch.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": embed_dense(c["text"]), "sparse": embed_sparse(c["text"])},
            payload=c,
        ))
        if len(batch) >= 32:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            batch = []
            print(f"  {i+1}/{len(all_chunks)} yazildi...")
    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    info = client.get_collection(COLLECTION_NAME)
    print(f"\n[INGEST] Tamamlandi. Toplam vektor: {info.points_count}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/sample.pdf"
    main(target)
