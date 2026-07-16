#!/bin/bash
# Start the two small-VLM backends for the accuracy A/B, each in its own venv,
# in-process (no vLLM needed -- greedy transformers/paddlex output equals vLLM's,
# and this is the exact code proven locally). Run from repo root after the setup
# scripts (setup.sh, setup_paddleocrvl.sh, setup_hunyuan.sh).
#
#   backend        venv               service                 port
#   PaddleOCR-VL   paddleocrvl_env    paddleocrvl_service     8104
#   HunyuanOCR     hunyuan_env        hunyuan_service         8105
#
# HF weights + cache go on the Network Volume (/workspace) so they survive a pod
# migrate and download once; XET off avoids the disk-quota (EDQUOT) stalls.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME"

echo "Starting PaddleOCR-VL (:8104) ..."
PYTHONPATH="$REPO" HF_HOME="$HF_HOME" HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  nohup paddleocrvl_env/bin/uvicorn paddleocrvl_service:app --app-dir services \
  --host 127.0.0.1 --port 8104 > paddleocrvl_service.log 2>&1 &

echo "Starting HunyuanOCR (:8105) ..."
PYTHONPATH="$REPO" HF_HOME="$HF_HOME" HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  nohup hunyuan_env/bin/uvicorn hunyuan_service:app --app-dir services \
  --host 127.0.0.1 --port 8105 > hunyuan_service.log 2>&1 &

echo "Waiting for health (first call also downloads weights -> can take minutes)..."
for p in 8104 8105; do
  until curl -s "http://127.0.0.1:$p/health" >/dev/null 2>&1; do sleep 3; done
  printf "  :%s -> " "$p"; curl -s "http://127.0.0.1:$p/health"; echo
done
echo "Both up. Score with eval/pod_eval.py (once per adapter):"
echo "  POD_EVAL_OUT=output/eval/vl.json ADAPTER_URL=http://127.0.0.1:8104/table paddle_env/bin/python -m eval.pod_eval"
echo "  POD_EVAL_OUT=output/eval/hy.json ADAPTER_URL=http://127.0.0.1:8105/table paddle_env/bin/python -m eval.pod_eval"
