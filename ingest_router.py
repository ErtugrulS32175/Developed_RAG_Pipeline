import os
import sys
import uuid
import requests
from dotenv import load_dotenv

from router import route_and_parse
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

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3"),
    max_tokens=512
)
chunker = HybridChunker(tokenizer=tokenizer)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def embed_dense(text):
    r = requests.post(EMBED_API_URL, json={"model": EMBED_MODEL_NAME, "input": text[:2000]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def embed_sparse(text):
    res = list(sparse_model.embed([text]))[0]
    return SparseVector(indices=res.indices.tolist(), values=res.values.tolist())

def main(path):
    # 1) Router ile parse et (native -> TableFormer, scanned/image -> OCR'li Docling)
    docs = route_and_parse(path)
    print(f"\n[INGEST] {len(docs)} belge parse edildi, chunk'laniyor...")

    # 2) Her belgeyi chunk'la
    all_chunks = []
    for source_tag, doc in docs:
        for chunk in chunker.chunk(doc):
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
    print(f"[INGEST] Toplam {len(all_chunks)} chunk")

    # 3) Qdrant collection
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
    )
    print(f"[INGEST] Collection hazir: {COLLECTION_NAME}")

    # 4) Embed + upsert
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
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/2024.pdf"
    main(target)
