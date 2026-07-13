#!/bin/bash
# Gemma-only FAST path for the A/B: serve JUST gemma-4-E4B behind vLLM, fronted by
# one vllm_table_service adapter that exposes our {tables} contract. Use this when
# you only want to test gemma (no PaddleOCR-VL / HunyuanOCR) -- one model loads, so
# the pod is ready far sooner and gemma gets the whole A40.
#
#   model         vLLM server   adapter (contract)   router env
#   gemma-4-E4B   :8113         :8101                GEMMA_TABLE_URL
#
# Alone on a 48GB A40 gemma can take a generous slice (0.5 -> ~24GB for weights +
# KV cache). The LOCAL tatr backend is unaffected. Run from repo root after
# scripts/setup_vllm.sh.
set -e
VLLM=/workspace/vllm_env/bin/vllm
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "Starting vLLM OpenAI server for gemma-4-E4B (weights download on first run)..."
nohup $VLLM serve google/gemma-4-E4B-it \
  --max-model-len 8192 --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.5 --port 8113 > vllm_gemma.log 2>&1 &

echo "Waiting for vLLM (model load can take a few minutes)..."
until curl -s "http://127.0.0.1:8113/health" >/dev/null 2>&1; do sleep 5; done
echo "  vLLM :8113 ready"

echo "Starting adapter service (our contract port :8101)..."
VLLM_BASE_URL="http://127.0.0.1:8113/v1" VLLM_MODEL="google/gemma-4-E4B-it" PYTHONPATH="$REPO" \
  nohup gemma_env/bin/uvicorn vllm_table_service:app --app-dir services \
  --host 127.0.0.1 --port 8101 > wrap_gemma.log 2>&1 &

sleep 6
printf "Health :8101 -> "; curl -s "localhost:8101/health"; echo
echo
echo "Done. Locally: tunnel 8201->8101 and run:"
echo "  GEMMA_TABLE_URL=http://127.0.0.1:8201/table python -m eval.run_eval --backends gemma --images sample1"
