import os
import json
import uuid
import requests
from pathlib import Path
from dotenv import load_dotenv

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.settings import settings
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseIndexParams
from qdrant_client.http.models import SparseVector
from fastembed import SparseTextEmbedding

load_dotenv()

# --- Config ---
PDF_PATH         = os.getenv("PDF_PATH", "./data/sample.pdf")
OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "./output"))
QDRANT_URL       = os.getenv("QDRANT_URL")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "rag_oyak_cloud")
EMBED_API_URL    = os.getenv("EMBED_API_URL", "http://localhost:8011/v1/embeddings")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
GRANITE_API_URL  = os.getenv("GRANITE_API_URL", "http://localhost:8003/v1/chat/completions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = OUTPUT_DIR / "temp_pages"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Set up the GraniteDocling vLLM converter (once, reused for all pages) ---
engine_options = ApiVlmEngineOptions(
    runtime_type=VlmEngineType.API,
    url=GRANITE_API_URL,
    params=dict(
        model="ibm-granite/granite-docling-258M",
        max_tokens=4096,
        temperature=0.0,
        skip_special_tokens=False,
    ),
    timeout=90,
    response_format="doctags",
)
vlm_options = VlmConvertOptions.from_preset("granite_docling", engine_options=engine_options)
pipeline_options = VlmPipelineOptions(
    generate_page_images=True,
    vlm_options=vlm_options,
    enable_remote_services=True,
)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3"),
    max_tokens=512
)
chunker = HybridChunker(tokenizer=tokenizer)

# --- Step 1: Split PDF into single-page PDFs and process each individually ---
print("Splitting PDF into single pages...")
src_pdf = pdfium.PdfDocument(PDF_PATH)
num_pages = len(src_pdf)
print(f"Total pages: {num_pages}")

all_chunks = []
failed_pages = []

for page_idx in range(num_pages):
    page_num = page_idx + 1
    page_pdf_path = TEMP_DIR / f"page_{page_num}.pdf"

    # Extract single page to its own PDF
    single = pdfium.PdfDocument.new()
    single.import_pages(src_pdf, [page_idx])
    single.save(str(page_pdf_path))
    single.close()

    # Convert this single page
    try:
        result = converter.convert(str(page_pdf_path), raises_on_error=True)
        page_chunks = list(chunker.chunk(result.document))

        for chunk in page_chunks:
            chunk_type = "text"
            if chunk.meta.doc_items:
                for item in chunk.meta.doc_items:
                    if "table" in str(item.label).lower():
                        chunk_type = "table"
                        break
            headings = chunk.meta.headings or []
            all_chunks.append({
                "type": chunk_type,
                "text": chunk.text,
                "page": page_num,
                "headings": headings,
                "source": str(PDF_PATH),
            })
        print(f"  Page {page_num}/{num_pages}: OK ({len(page_chunks)} chunks)")

    except Exception as e:
        failed_pages.append({"page": page_num, "error": str(e)[:200]})
        print(f"  Page {page_num}/{num_pages}: FAILED - {str(e)[:80]}")

    finally:
        # Clean up temp page file
        if page_pdf_path.exists():
            page_pdf_path.unlink()

src_pdf.close()

# --- Merge consecutive table chunks with same heading ---
merged_chunks = []
i = 0
while i < len(all_chunks):
    chunk = all_chunks[i]
    if chunk["type"] == "table":
        merged_text = chunk["text"]
        while (i + 1 < len(all_chunks)
               and all_chunks[i + 1]["type"] == "table"
               and all_chunks[i + 1]["headings"] == chunk["headings"]
               and all_chunks[i + 1]["page"] == chunk["page"]):
            merged_text += "\n" + all_chunks[i + 1]["text"]
            i += 1
        merged_chunks.append({**chunk, "text": merged_text})
    else:
        merged_chunks.append(chunk)
    i += 1

chunks_data = merged_chunks

# --- Save chunks and failure report ---
chunks_path = OUTPUT_DIR / "chunks.json"
with open(chunks_path, "w", encoding="utf-8") as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=2)
print(f"\nChunks saved: {chunks_path} ({len(chunks_data)} chunks)")

report_path = OUTPUT_DIR / "failed_pages.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(failed_pages, f, ensure_ascii=False, indent=2)
print(f"Failed pages: {len(failed_pages)} (report saved to {report_path})")
if failed_pages:
    print("Failed page numbers:", [fp["page"] for fp in failed_pages])

# --- Step 2: Embedding via vLLM (dense) + FastEmbed (sparse/BM25) ---
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def embed_dense(text: str) -> list[float]:
    response = requests.post(
        EMBED_API_URL,
        json={"model": EMBED_MODEL_NAME, "input": text[:2000]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def embed_sparse(text: str) -> SparseVector:
    result = list(sparse_model.embed([text]))[0]
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())

# --- Step 3: Set up the Qdrant Cloud collection ---
print("\nConnecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
)
print(f"Created collection: {COLLECTION_NAME}")

# --- Step 4: Embed and upsert ---
print("\nIngesting chunks...")
batch = []
for i, chunk in enumerate(chunks_data):
    if not chunk["text"].strip():
        continue
    dense = embed_dense(chunk["text"])
    sparse = embed_sparse(chunk["text"])
    batch.append(PointStruct(
        id=str(uuid.uuid4()),
        vector={"dense": dense, "sparse": sparse},
        payload=chunk,
    ))
    if len(batch) >= 32:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        batch = []
        print(f"  {i+1}/{len(chunks_data)} chunks ingested...")

if batch:
    client.upsert(collection_name=COLLECTION_NAME, points=batch)

info = client.get_collection(COLLECTION_NAME)
print(f"\nCollection '{COLLECTION_NAME}' ready. Total vectors: {info.points_count}")
