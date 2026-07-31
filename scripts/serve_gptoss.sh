#!/bin/bash
# Serve the enterprise target LLM (GPT-OSS-120B) on a rented 80GB GPU, reached
# from the laptop over an SSH tunnel. This is the RAG answering path only -- no
# table adapters, no gemma_env; scripts/serve_vllm.sh is the table workstream.
#
#   pod   : vLLM OpenAI server on :8000
#   laptop: ssh -N -L 8000:127.0.0.1:8000 root@<ip> -p <port>
#           then LLM_API_URL=http://localhost:8000/v1/chat/completions
#
# Everything else (pgvector, bge-m3, reranker) stays on the laptop, so no
# document ever reaches the pod -- only a question and its assembled context.
set -e

MODEL="${GPTOSS_MODEL:-openai/gpt-oss-120b}"
PORT="${GPTOSS_PORT:-8000}"
# 65GB of weights on an 80GB card leaves ~11GB for KV cache at 0.95. The eval
# sends 15 chunks plus a question -- nowhere near 32k -- but a longer window
# costs KV cache, not quality, so there is no reason to raise it.
MAXLEN="${GPTOSS_MAXLEN:-32768}"
GPU_FRAC="${GPU_FRAC:-0.95}"
VENV=/workspace/vllm_env

# WHERE THE WEIGHTS LAND IS A REAL CHOICE, not a detail:
#   /workspace  persists across stop/start, but on RunPod it is a MooseFS FUSE
#               mount. vLLM does not recognise it as a network FS and reads
#               shards serially -- a 58GB checkpoint once sat at 0/2 for 20+
#               minutes. Also, `df` there reports the whole cluster, NOT your
#               quota, so "226T free" tells you nothing about your limit.
#   /           the container overlay: local, fast, and wiped when the pod
#               stops. Right choice for a single measured session.
# Default is the persistent one; set HF_ROOT=/hf to use the fast local disk.
HF_ROOT="${HF_ROOT:-/workspace/hf}"
export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="$HF_ROOT/hub"
# hf-xet stages chunks AND reconstructs the file -> ~2x peak disk. With 65GB of
# weights and a 100GB disk that does not fit, so it stays off.
export HF_HUB_DISABLE_XET=1
# hf_transfer downloads in parallel chunks WITHOUT the 2x staging cost, which is
# the one lever that helps here: a single stream measured 12MB/s on this pod,
# i.e. ~87 minutes for 65GB.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# uv defaults its cache next to $HOME, and on RunPod $HOME is /workspace -- the
# MooseFS mount. Writing a multi-GB package cache through FUSE is what makes the
# install appear to hang (measured: 4.9MB of venv after minutes, byte counters
# frozen). Keep the cache on the container disk; it is scratch either way.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "[1/3] vLLM env yok, kuruluyor ($VENV)"
  pip install -q uv
  uv venv "$VENV"
  uv pip install --python "$VENV/bin/python" -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match
  # flashinfer JIT-compiles a sampling kernel through `ninja` at startup; without
  # it the engine dies LATE, after CUDA-graph capture, and the real error sits
  # above the APIServer traceback rather than at the tail.
  apt-get install -y -qq ninja-build 2>/dev/null || "$VENV/bin/pip" install -q ninja
  # Parallel-chunk downloader for the 65GB fetch below. Optional on purpose: if
  # the wheel is unavailable the download still works, just serially.
  "$VENV/bin/pip" install -q hf_transfer || export HF_HUB_ENABLE_HF_TRANSFER=0
else
  echo "[1/3] vLLM env hazir"
fi
export PATH="$VENV/bin:$PATH"

# The HF repo ships THREE ~65GB copies of the same model: the root MXFP4 shards
# (what vLLM loads), original/ and metal/. A `*.safetensors` match would pull
# 130GB+ instead of 65GB, so the copies are excluded explicitly rather than
# left to whatever the loader's default pattern happens to be.
echo "[2/3] Agirliklar indiriliyor -> $HF_ROOT (sadece kok shardlar, ~65GB)"
mkdir -p "$HF_ROOT"
AVAIL=$(df -Pk "$HF_ROOT" | awk 'NR==2 {print int($4/1024/1024)}')
echo "      hedefte bos alan: ${AVAIL}GB"
# Only meaningful on a real filesystem: on the MooseFS mount df reports the
# cluster, so this catches the container-disk case and stays quiet otherwise.
if [ "$AVAIL" -lt 75 ]; then
  echo "HATA: 65GB model icin yer yok (${AVAIL}GB). HF_ROOT baska bir diske ver."
  exit 1
fi
"$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL", ignore_patterns=["original/*", "metal/*"])
print("indirildi:", p)
PY

echo "[3/3] vLLM baslatiliyor :$PORT"
# Serial shard reads are what make a FUSE-mounted checkpoint crawl; prefetch
# pulls them into the page cache in parallel. Added only if this build has the
# flag -- vLLM nightly moves, and an unknown flag would kill the server.
EXTRA=""
if "$VENV/bin/vllm" serve --help 2>&1 | grep -q -- "--safetensors-load-strategy"; then
  EXTRA="--safetensors-load-strategy=prefetch"
  echo "      prefetch acik (agirliklar $HF_ROOT uzerinde)"
fi

# Detached on purpose: RunPod SSH idles out and a foreground server dies with the
# session. tmux exits on these images -- use nohup.
nohup "$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_FRAC" \
  $EXTRA \
  > /workspace/vllm_gptoss.log 2>&1 &

echo "Model yukleniyor (birkac dakika). Bekleniyor..."
until curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "HATA: vLLM oldu. Son loglar:"
    grep -iE "error|traceback" /workspace/vllm_gptoss.log | tail -20
    exit 1
  fi
  sleep 10
done

echo
echo "Hazir. Dogrula:  curl -s localhost:$PORT/v1/models"
curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 400
echo
echo
echo "Laptoptan tunel:"
echo "  ssh -N -L $PORT:127.0.0.1:$PORT root@<ip> -p <sshport> -o ServerAliveInterval=60"
echo "Sonra .env:  LLM_API_URL=http://localhost:$PORT/v1/chat/completions"
echo "            LLM_MODEL_NAME=$MODEL"
