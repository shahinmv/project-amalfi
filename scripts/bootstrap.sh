#!/usr/bin/env bash
# One-command per-laptop setup for Amalfi: venv + deps + build llama.cpp + probe.
# Usage:
#   scripts/bootstrap.sh [--backend auto|cuda|metal|vulkan|cpu] [--rpc-host IP]
#                        [--rpc-port N] [--start-worker]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

BACKEND="auto"; RPC_HOST=""; RPC_PORT="50052"; START_WORKER="no"
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --rpc-host) RPC_HOST="$2"; shift 2;;
    --rpc-port) RPC_PORT="$2"; shift 2;;
    --start-worker) START_WORKER="yes"; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

echo "== [1/4] python venv + deps =="
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt
# cmake as a pip package so no separate system CMake install is needed
./.venv/bin/python -m pip install -q cmake

if [ "$BACKEND" = "auto" ]; then
  BACKEND="$(./.venv/bin/python scripts/probe.py --print-backend)"
fi
echo "== [2/4] building llama.cpp (backend: $BACKEND) =="
PATH="$HERE/.venv/bin:$PATH" ./scripts/build_llamacpp.sh "$BACKEND"

echo "== [3/4] probing this machine =="
if [ -z "$RPC_HOST" ]; then
  RPC_HOST="$(./.venv/bin/python scripts/probe.py --print-ip)"
  echo ">> detected LAN IP: $RPC_HOST  (override with --rpc-host if this is wrong)"
fi
OUT="node_$(hostname -s).json"
./.venv/bin/python scripts/probe.py --rpc-host "$RPC_HOST" --rpc-port "$RPC_PORT" --out "$OUT"
echo ">> wrote $OUT (copy this to the coordinator and merge all node_*.json into nodes.json)"

echo "== [4/4] next steps =="
if [ "$START_WORKER" = "yes" ]; then
  echo ">> starting worker (rpc-server) on port $RPC_PORT ..."
  exec ./scripts/start_worker.sh "$RPC_PORT"
else
  cat <<EOF
Done. On EVERY laptop run this script (add --start-worker to also launch the worker).
On the COORDINATOR only:
  1) merge all node_*.json into a JSON array 'nodes.json'
  2) ./.venv/bin/python scripts/plan_split.py --nodes nodes.json --model qwen3-30b-a3b-q4 --out fleet.json
  3) download the GGUF into models/ (see docs/runbook.md)
  4) start workers on every node, then ./scripts/start_coordinator.sh fleet.json
  5) ./.venv/bin/python bench/run_bench.py --url http://<coordinator-IP>:8080 --concurrency 8
EOF
fi
