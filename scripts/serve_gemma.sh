#!/bin/bash
# Gemma-only FAST path for the A/B: serve ONE Gemma-4 checkpoint behind vLLM,
# fronted by one vllm_table_service adapter that exposes our {tables} contract.
#
#   model (default)   vLLM server   adapter (contract)   router env
#   gemma-4-31B-it     :8113         :8101                GEMMA_TABLE_URL
#
# The model is an env var so switching checkpoints is one line, no edit:
#   GEMMA_MODEL=google/gemma-4-12B-it bash scripts/serve_gemma.sh
#
# SIZING: 31B is bf16 ~62GB of weights -> needs an 80GB card (H100/A100), it does
# NOT fit a 48GB L40S/A40. GPU_FRAC=0.90 on 80GB = ~72GB (weights + KV). 12B is
# ~24GB and fits either card; 0.90 just gives it more KV cache. The LOCAL tatr
# backend is unaffected. Run from repo root after scripts/setup_vllm.sh.
set -e
VLLM=/workspace/vllm_env/bin/vllm
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODEL="${GEMMA_MODEL:-google/gemma-4-31B-it}"
GPU_FRAC="${GPU_FRAC:-0.90}"

# HF weights must land on the big /workspace volume -- a default ~/.cache download
# hits the near-full container disk and dies with "Disk quota exceeded (os error
# 122)". Keep HF_HUB_DISABLE_XET=1 in the environment too (xet ~doubles peak disk).
export HF_HOME="${HF_HOME:-/workspace/hf}"

echo "Starting vLLM OpenAI server for $MODEL (weights download on first run)..."
nohup $VLLM serve "$MODEL" \
  --max-model-len 8192 --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --gpu-memory-utilization "$GPU_FRAC" --port 8113 > vllm_gemma.log 2>&1 &

echo "Waiting for vLLM (31B load + download can take a while)..."
until curl -s "http://127.0.0.1:8113/health" >/dev/null 2>&1; do sleep 5; done
echo "  vLLM :8113 ready"

echo "Starting adapter service (our contract port :8101)..."
VLLM_BASE_URL="http://127.0.0.1:8113/v1" VLLM_MODEL="$MODEL" PYTHONPATH="$REPO" \
  nohup gemma_env/bin/uvicorn vllm_table_service:app --app-dir services \
  --host 127.0.0.1 --port 8101 > wrap_gemma.log 2>&1 &

sleep 6
printf "Health :8101 -> "; curl -s "localhost:8101/health"; echo
echo
echo "Done. Locally: tunnel 8201->8101 and run:"
echo "  GEMMA_TABLE_URL=http://127.0.0.1:8201/table python -m eval.run_eval --backends gemma --images sample1"
