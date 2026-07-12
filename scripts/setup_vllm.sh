#!/bin/bash
# vLLM setup for the FAST A/B path: PaddleOCR-VL + HunyuanOCR served through
# vLLM's OpenAI API (batched attention/paged KV -> same greedy output as the
# transformers wrappers, far faster). Consumed by services/vllm_table_service.py.
#
# Two constraints drive the choices below:
#  * PaddleOCR-VL needs a vLLM NIGHTLY (until 0.11.1 releases); nightly also
#    serves HunyuanOCR, so one env covers both.
#  * vLLM + torch is big and the container disk (/) is nearly full, so the env
#    lives on the /workspace volume (huge, network-backed).
# Run from repo root on the pod, after the other setup scripts.
set -e

echo "[1/3] Creating vLLM nightly env on /workspace (container disk is full)"
pip install -q uv
uv venv /workspace/vllm_env
uv pip install --python /workspace/vllm_env/bin/python -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

echo "[2/3] Wrapper deps into gemma_env (runs services/vllm_table_service.py)"
# The adapter only needs a web stack + requests + table_export's openpyxl; reuse
# the existing gemma_env rather than build another venv.
gemma_env/bin/pip install -q openpyxl requests

echo "[3/3] Verify"
/workspace/vllm_env/bin/vllm --version

echo "Done. Start everything with:  bash scripts/serve_vllm.sh"
