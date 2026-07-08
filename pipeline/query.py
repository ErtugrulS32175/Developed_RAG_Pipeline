import os

import requests
from dotenv import load_dotenv

from pipeline import db
from pipeline.embeddings import embed_dense, embed_sparse

load_dotenv()

# --- Config ---
RERANK_API_URL     = os.getenv("RERANK_API_URL", "http://localhost:8002/v1/score")
RERANK_MODEL_NAME  = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

LLM_API_URL        = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
LLM_MODEL_NAME     = os.getenv("LLM_MODEL_NAME", "google/gemma-4-12B-it")

TOP_K      = 15
TOP_RERANK = 10

# --- Init ---
conn = db.get_conn()


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Hybrid search in Postgres (pgvector) combining dense and sparse vectors via RRF."""
    dense_vector = embed_dense(query)
    sparse_indices, sparse_values = embed_sparse(query)
    return db.hybrid_search(conn, dense_vector, sparse_indices, sparse_values, top_k=top_k)


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
        text = chunk.get("text", "")
        if chunk.get("type") == "table":
            # table_to_markdown already prepends a Belge/Sayfa/Tablo/Güven
            # citation header, so this is already fully self-describing.
            parts.append(text)
        else:
            filename = chunk.get("filename") or "?"
            page = chunk.get("page", "?")
            parts.append(f"[{filename} - Sayfa {page}]\n{text}")
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

        