#!/bin/bash
# Launch the ALL-vLLM path on the pod: every VLM backend (gemma, PaddleOCR-VL,
# HunyuanOCR) runs behind a vLLM OpenAI server for speed, each fronted by a
# vllm_table_service adapter that exposes our {tables} contract on the usual
# backend port. The router/harness reach the ADAPTERS over the SSH tunnel; the
# adapters reach vLLM over pod-localhost.
#
#   model          vLLM server    adapter (contract)   router env
#   gemma-4-E4B     :8113          :8101                GEMMA_TABLE_URL
#   PaddleOCR-VL    :8114          :8104                PADDLEOCR_VL_URL
#   HunyuanOCR      :8115          :8105                HUNYUAN_TABLE_URL
#
# The LOCAL tatr backend is unaffected. Caps keep all three on one A40
# (0.2*48GB ~= 9.6GB each). Run from repo root after scripts/setup_vllm.sh.
set -e
VLLM=/workspace/vllm_env/bin/vllm
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "Starting vLLM OpenAI servers (weights download to /workspace on first run)..."
nohup $VLLM serve google/gemma-4-E4B-it \
  --max-model-len 8192 --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.2 --port 8113 > vllm_gemma.log 2>&1 &
nohup $VLLM serve PaddlePaddle/PaddleOCR-VL --trust-remote-code \
  --max-num-batched-tokens 16384 --max-model-len 8192 --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 --gpu-memory-utilization 0.2 --port 8114 > vllm_vl.log 2>&1 &
nohup $VLLM serve tencent/HunyuanOCR \
  --max-model-len 8192 --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.2 --port 8115 > vllm_hy.log 2>&1 &

echo "Waiting for vLLM servers (model load can take a few minutes)..."
for p in 8113 8114 8115; do
  until curl -s "http://127.0.0.1:$p/health" >/dev/null 2>&1; do sleep 5; done
  echo "  vLLM :$p ready"
done

echo "Starting adapter services (our contract ports)..."
start_adapter () {  # $1=vllm_port  $2=model  $3=adapter_port  $4=log
  VLLM_BASE_URL="http://127.0.0.1:$1/v1" VLLM_MODEL="$2" PYTHONPATH="$REPO" \
    nohup gemma_env/bin/uvicorn vllm_table_service:app --app-dir services \
    --host 127.0.0.1 --port "$3" > "$4" 2>&1 &
}
start_adapter 8113 google/gemma-4-E4B-it   8101 wrap_gemma.log
start_adapter 8114 PaddlePaddle/PaddleOCR-VL 8104 wrap_vl.log
start_adapter 8115 tencent/HunyuanOCR      8105 wrap_hy.log

sleep 6
echo "Health:"
for p in 8101 8104 8105; do printf ":%s -> " "$p"; curl -s "localhost:$p/health"; echo; done
echo
echo "Done. Locally: tunnel 8201->8101, 8204->8104, 8205->8105 and run the harness."
