# Cloud RAG Pipeline — GraniteDocling + vLLM + Qdrant Cloud

An end-to-end, fully self-hosted (vLLM-based) RAG pipeline for Turkish PDF documents. Runs on RunPod GPU infrastructure.

## Architecture

INGESTION: PDF -> GraniteDocling (vLLM, DocTags) -> HybridChunker -> bge-m3 (dense) + BM25 (sparse) -> Qdrant Cloud

QUERY: Question -> hybrid search (RRF: dense + sparse) -> bge-reranker-v2-m3 -> Qwen3-14B -> Turkish answer + source page

All models are served via vLLM as OpenAI-compatible APIs, on the same GPU across separate ports:

| Service | Model | Port | vLLM task |
|---|---|---|---|
| LLM | Qwen/Qwen3-14B | 8000 | generate |
| Reranker | BAAI/bge-reranker-v2-m3 | 8002 | score |
| Parser | ibm-granite/granite-docling-258M | 8003 | generate (VLM) |
| Embedding | BAAI/bge-m3 | 8011 | embed |

## Setup

Run ./setup.sh after every pod migration/restart, then cp .env.example .env and fill in your real Qdrant credentials.

## Starting the services

Embedding (needed for both ingestion and query):

    nohup vllm serve BAAI/bge-m3 --task embed --gpu-memory-utilization 0.1 --port 8011 > embed.log 2>&1 &

GraniteDocling — only during ingestion (the untied revision is MANDATORY):

    nohup vllm serve ibm-granite/granite-docling-258M --revision untied --port 8003 --gpu-memory-utilization 0.3 > granite.log 2>&1 &

LLM — during query (16K context is enough for the KV cache):

    nohup vllm serve Qwen/Qwen3-14B --gpu-memory-utilization 0.85 --max-model-len 16384 --port 8000 > llm.log 2>&1 &

Reranker — during query:

    nohup vllm serve BAAI/bge-reranker-v2-m3 --task score --gpu-memory-utilization 0.05 --port 8002 > rerank.log 2>&1 &

## Usage

    python3 ingest.py   # parse the PDF and upsert into Qdrant Cloud
    python3 query.py    # interactive querying

## GraniteDocling + vLLM Integration — Critical Notes

Using GraniteDocling via vLLM with Docling requires several critical settings that are not clearly documented. Until these are discovered, the result is 0 chunks:

1. --revision untied — GraniteDocling uses tied weights, which current vLLM versions do not support (AttributeError: 'LlamaModel' object has no attribute 'wte'). IBM provides an untied branch.

2. skip_special_tokens=False — by default vLLM strips DocTags special tokens like text, table, section_header; Docling cannot parse the output without them, giving an empty result.

3. response_format="doctags" — Docling must be explicitly told the response is in DocTags format.

4. Use the new API — instead of vlm_model_specs.GRANITEDOCLING_VLLM_API (legacy), use ApiVlmEngineOptions + VlmConvertOptions.from_preset("granite_docling", ...).

Related open GitHub issues: docling #3403, #2925, #2398; HF discussion #20.

## Page-by-Page Ingestion Strategy

GraniteDocling produces malformed bounding-box coordinates on some complex pages (ValueError: Coordinate 'right' is less than 'left'). If the whole PDF is processed at once, a single bad page empties the entire document with raises_on_error=False (0 chunks).

Solution: the PDF is split so each page becomes its own single-page PDF, and each page is processed separately in a try/except with raises_on_error=True. Failed pages are reported to output/failed_pages.json while the rest are processed normally.

## Known Issues / Findings

- Page 40: Could not be parsed due to a coordinate error (reported to failed_pages.json). 107 of 108 pages succeeded, producing 442 chunks.

- Page 13 table loss: When parsing the "OYAK Publicly Traded Companies" table, the table header and footnotes were extracted but the rows containing company names (including HEKTAS) were lost. This is a parsing issue, not a retrieval issue — the question about which holding HEKTAS belongs to is answered correctly from other pages (51, 58, 59), but HEKTAS does not appear for the stock-exchange query because that information only exists in the incomplete table on page 13.

## Test Results

- "Which holding is HEKTAS affiliated with?" -> OYAK group, 1963 investment, largest shareholder in 1981 (pages 58, 51, 19) OK
- "Which OYAK companies are traded on Borsa Istanbul?" -> OYAK Cimento, OYAK Yatirim Menkul Degerler, OYAK Yatirim Ortakligi (pages 36, 19, 70, 73) OK (excluding HEKTAS — page 13 table issue)

## Environment Notes (RunPod)

- Pod migration wipes pip packages but preserves the /workspace disk, so run ./setup.sh after every migration.
- transformers 5.x is incompatible with vLLM 0.11.0, so 4.57.x is required (setup.sh handles this).
- Stop (not Terminate) preserves the disk.

## Future Work / TODO

This is currently a working prototype. Planned improvements:

1. Table parsing quality (high priority): GraniteDocling loses rows in complex tables like page 13. Inspect the raw DocTags of page 13 to see whether the table is parsed incompletely or lost during chunking; experiment with generate_page_images and scale; evaluate a hybrid VLM + TableFormer approach for complex tables.

2. Recovering failed pages: investigate why page 40 failed; add a retry with different scale or a coordinate-fixing fallback.

3. Full test on a larger GPU: run all services (LLM + embedding + reranker + GraniteDocling) at once on an H100 (80GB) or H200 (141GB) to avoid VRAM juggling on the A40 (48GB).

4. Front-end / demo: add an OpenWebUI or Gradio interface with a shareable link.

5. Hide the thinking block: Qwen3 emits a think block; suppress it in query.py so users only see the final answer.

6. Ingestion performance: explore batching multiple single-page requests concurrently against the GraniteDocling endpoint.
