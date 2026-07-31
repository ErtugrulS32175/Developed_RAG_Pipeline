"""The model call and the prompts that go with it.

Split out of the retrieval module, which had grown to own both halves of the
pipeline. They change for different reasons and at different rates: retrieval
settings move when the index or the ranking changes, prompts move whenever we
learn something about how the model reads a table. Keeping them together meant
every prompt edit touched the file that decides what reaches the model.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

LLM_API_URL    = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-4-12B-it")
# Empty by default: reached over a tunnel or on localhost the endpoint needs no
# credential. It matters when the server is a rented GPU behind a provider's
# PUBLIC proxy URL -- there, vLLM's own --api-key plus this header is what stops
# the thing from answering anyone who guesses the address.
LLM_API_KEY    = os.getenv("LLM_API_KEY", "")


def llm_headers() -> dict:
    """Auth header for the answering endpoint, or nothing when no key is set.

    Kept as a function rather than a module constant so a test (or a caller
    that sets LLM_API_KEY late) sees the current value.
    """
    return {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}


def complete(prompt: str) -> str:
    """Call the vLLM chat completions endpoint."""
    response = requests.post(
        LLM_API_URL,
        json={
            "model": LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        headers=llm_headers(),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate(question: str, context: str) -> str:
    """The plain answer: prose, with a page citation."""
    return complete(f"""Aşağıdaki belge pasajlarına dayanarak soruyu Türkçe olarak cevapla.
SADECE pasajlarda açıkça belirtilen bilgileri kullan.
Pasajlarda olmayan hiçbir bilgiyi ekleme veya tahmin etme.
Cevabında ilgili sayfa numarasını belirt (örn: "Sayfa 204'e göre...").
Eğer cevap pasajlarda yoksa "Bu bilgi mevcut belgelerde bulunamadı." de.

BELGE PASAJLARI:
{context}

SORU: {question}

CEVAP:""")


def generate_structured(question: str, context: str) -> str:
    """The same answer, preceded by the lines it rests on.

    The evidence comes FIRST in the requested JSON, and that ordering is the
    point rather than a formatting preference: the model writes the quote
    before it writes the answer, so the answer is produced with the exact line
    already in front of it. Quoting the target line before answering is the
    cheapest intervention in the table-QA literature, and it is what makes the
    answer checkable afterwards -- a quote can be matched against the passage
    it claims to come from, a paraphrase cannot.

    Needs a context built with numbered=True, or there is nothing to point at.
    """
    return complete(f"""Aşağıdaki belge pasajlarına dayanarak soruyu Türkçe olarak cevapla.
SADECE pasajlarda açıkça belirtilen bilgileri kullan.

Yanıtını YALNIZCA aşağıdaki JSON biçiminde ver, öncesinde ve sonrasında hiçbir şey yazma:
{{
  "dayanak": [
    {{"pasaj": <pasaj numarasi>, "alinti": "<o pasajdan BIREBIR kopyalanmis satir>"}}
  ],
  "cevap": "<Türkçe cevap; ilgili sayfa numarasını belirt>"
}}

Kurallar:
- Önce "dayanak" alanını doldur, sonra "cevap" alanını yaz.
- "alinti" pasajdaki metinden kelimesi kelimesine kopyalanmalı; özetleme, düzeltme.
- Cevabındaki her sayı, alıntıladığın satırlarda geçmeli.
- Cevap pasajlarda yoksa "dayanak" boş liste olsun ve "cevap" alanına
  "Bu bilgi mevcut belgelerde bulunamadı." yaz.

BELGE PASAJLARI:
{context}

SORU: {question}

JSON:""")
