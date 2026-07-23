#!/usr/bin/env bash
# Start llama-server (coordinator) using the command from fleet.json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
FLEET="${1:-$HERE/fleet.json}"
[ -f "$FLEET" ] || { echo "ERROR: $FLEET not found. Run plan_split.py first."; exit 1; }
CMD="$(python3 -c "import json;print(json.load(open('$FLEET'))['coordinator_cmd'])")"
BIN_DIR="$HERE/vendor/llama.cpp/build/bin"
echo ">> $CMD"
cd "$HERE"
exec env PATH="$BIN_DIR:$PATH" bash -c "$CMD"
