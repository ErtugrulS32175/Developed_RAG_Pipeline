#!/bin/bash
# Launch the vLLM fast path on the pod: one vLLM OpenAI server per model + one
# vllm_table_service adapter per model exposing our {tables} contract on the
# usual backend ports (8104 paddleocr_vl, 8105 hunyuan). The router/harness reach
# THESE over the SSH tunnel; the adapters reach vLLM over pod-localhost.
#
# gemma (:8101, transformers -- already fast at ~280 visual tokens) and the LOCAL
# tatr backend are started separately; this script only owns the two heavy VLMs.
# Caps keep both vLLM servers + gemma on one A40 (0.2*48GB ~= 9.6GB each).
# Run from repo root after scripts/setup_vllm.sh.
set -e
VLLM=/workspace/vllm_env/bin/vllm
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "Starting vLLM OpenAI servers (weights download to /workspace on first run)..."
nohup $VLLM serve PaddlePaddle/PaddleOCR-VL --trust-remote-code \
  --max-num-batched-tokens 16384 --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.2 --port 8114 > vllm_vl.log 2>&1 &
nohup $VLLM serve tencent/HunyuanOCR \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.2 --port 8115 > vllm_hy.log 2>&1 &

echo "Waiting for vLLM servers (model load can take a few minutes)..."
for p in 8114 8115; do
  until curl -s "http://127.0.0.1:$p/health" >/dev/null 2>&1; do sleep 5; done
  echo "  vLLM :$p ready"
done

echo "Starting adapter services (our contract ports)..."
VLLM_BASE_URL=http://127.0.0.1:8114/v1 VLLM_MODEL=PaddlePaddle/PaddleOCR-VL PYTHONPATH="$REPO" \
  nohup gemma_env/bin/uvicorn vllm_table_service:app --app-dir services \
  --host 127.0.0.1 --port 8104 > wrap_vl.log 2>&1 &
VLLM_BASE_URL=http://127.0.0.1:8115/v1 VLLM_MODEL=tencent/HunyuanOCR PYTHONPATH="$REPO" \
  nohup gemma_env/bin/uvicorn vllm_table_service:app --app-dir services \
  --host 127.0.0.1 --port 8105 > wrap_hy.log 2>&1 &

sleep 6
echo "Health:"
curl -s localhost:8104/health; echo
curl -s localhost:8105/health; echo
echo
echo "Done. Now (locally): tunnel 8204->8104, 8205->8105 (+ 8201->8101 for gemma)"
echo "and run:  python -m eval.run_eval --backends paddleocr_vl hunyuan --images sample1 ..."
