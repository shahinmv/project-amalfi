#!/usr/bin/env bash
# Manual localhost proof: 2 rpc-servers + coordinator on this machine with a small model.
# Requires: scripts/build_llamacpp.sh <backend>   AND   a small GGUF at models/$MODEL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$HERE/vendor/llama.cpp/build/bin"
MODEL="${1:?usage: loopback_demo.sh <gguf-filename-in-models/>}"

"$BIN/ggml-rpc-server" --host 127.0.0.1 --port 50060 & W1=$!
"$BIN/ggml-rpc-server" --host 127.0.0.1 --port 50061 & W2=$!
sleep 3
trap 'kill $W1 $W2 $CO 2>/dev/null || true' EXIT

"$BIN/llama-server" --model "$HERE/models/$MODEL" \
  --rpc 127.0.0.1:50060,127.0.0.1:50061 --tensor-split 0.5,0.5 \
  --n-gpu-layers 999 --ctx-size 2048 --host 127.0.0.1 --port 8080 & CO=$!
sleep 8

echo ">> healthcheck:"; python3 "$HERE/scripts/healthcheck.py" --fleet <(echo '{"rpc":"127.0.0.1:50060,127.0.0.1:50061"}')
echo ">> benchmark:"; python3 "$HERE/bench/run_bench.py" --url http://127.0.0.1:8080 --concurrency 4 --max-tokens 64
