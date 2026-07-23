#!/usr/bin/env bash
# Start this node's rpc-server. Reads port from argument (default 50052).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-50052}"
BIN="$HERE/vendor/llama.cpp/build/bin/ggml-rpc-server"
[ -x "$BIN" ] || { echo "ERROR: $BIN not built. Run scripts/build_llamacpp.sh first."; exit 1; }
echo ">> starting ggml-rpc-server on 0.0.0.0:$PORT (LAN-only; do not expose to internet)"
exec "$BIN" --host 0.0.0.0 --port "$PORT"
