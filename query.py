import os
import requests
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from fastembed import SparseTextEmbedding

load_dotenv()

# --- Config ---
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "rag_cloud")

EMBED_API_URL      = os.getenv("EMBED_API_URL", "http://localhost:8001/v1/embeddings")
EMBED_MODEL_NAME   = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

RERANK_API_URL     = os.getenv("RERANK_API_URL", "http://localhost:8002/v1/score")
RERANK_MODEL_NAME  = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

LLM_API_URL        = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
LLM_MODEL_NAME     = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-14B-Instruct")

TOP_K      = 15
TOP_RERANK = 10

# --- Init ---
client       = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_dense(text: str) -> list[float]:
    """Call the vLLM embedding server for a dense vector."""
    response = requests.post(
        EMBED_API_URL,
        json={"model": EMBED_MODEL_NAME, "input": text[:2000]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def embed_sparse(text: str):
    """Compute a BM25 sparse vector locally via FastEmbed."""
    result = list(sparse_model.embed([text]))[0]
    return result.indices.tolist(), result.values.tolist()


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Hybrid search in Qdrant Cloud combining dense and sparse vectors via RRF."""
    dense_vector = embed_dense(query)
    sparse_indices, sparse_values = embed_sparse(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=top_k),
            Prefetch(query={"indices": sparse_indices, "values": sparse_values}, using="sparse", limit=top_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results.points]


def rerank(query: str, chunks: list[dict], top_n: int = TOP_RERANK) -> list[dict]:
    """Rerank retrieved chunks using the vLLM cross-encoder score endpoint."""
    scored = []
    for chunk in chunks:
        response = requests.post(
            RERANK_API_URL,
            json={
                "model": RERANK_MODEL_NAME,
                "text_1": query,
                "text_2": chunk["text"],
            },
            timeout=60,
        )
        response.raise_for_status()
        score = response.json()["data"][0]["score"]
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_n]]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        parts.append(f"[Sayfa {page}]\n{text}")
    return "\n\n---\n\n".join(parts)


def generate(question: str, context: str) -> str:
    """Call the vLLM chat completions endpoint for the final answer."""
    prompt = f"""Aşağıdaki belge pasajlarına dayanarak soruyu Türkçe olarak cevapla.
SADECE pasajlarda açıkça belirtilen bilgileri kullan.
Pasajlarda olmayan hiçbir bilgiyi ekleme veya tahmin etme.
Cevabında ilgili sayfa numarasını belirt (örn: "Sayfa 13'e göre...").
Eğer cevap pasajlarda yoksa "Bu bilgi mevcut belgelerde bulunamadı." de.

BELGE PASAJLARI:
{context}

SORU: {question}

CEVAP:"""

    response = requests.post(
        LLM_API_URL,
        json={
            "model": LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask(question: str) -> str:
    chunks  = retrieve(question)
    chunks  = rerank(question, chunks)
    context = build_context(chunks)
    return generate(question, context)


# --- Interactive loop ---
if __name__ == "__main__":
    print("RAG Pipeline hazır. Çıkmak için 'quit' yaz.\n")
    while True:
        question = input("Soru: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print("\nAranıyor ve rerank ediliyor...")
        answer = ask(question)
        print(f"\nCevap:\n{answer}\n")
        print("-" * 60)

        