#!/usr/bin/env bash
# Build a pinned llama.cpp (rpc-server + llama-server) for a given backend.
# Usage: scripts/build_llamacpp.sh <cuda|metal|vulkan|cpu>
set -euo pipefail
BACKEND="${1:?usage: build_llamacpp.sh <cuda|metal|vulkan|cpu>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/config/llamacpp.pin"

echo ">> verifying pinned tag $LLAMACPP_REF exists..."
git ls-remote --tags "$LLAMACPP_REPO" "refs/tags/$LLAMACPP_REF" | grep -q "$LLAMACPP_REF" \
  || { echo "ERROR: tag $LLAMACPP_REF not found in $LLAMACPP_REPO"; exit 1; }

SRC="$HERE/vendor/llama.cpp"
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$LLAMACPP_REF" "$LLAMACPP_REPO" "$SRC"
fi
FLAGS="$(python3 "$HERE/scripts/build_flags.py" "$BACKEND")"
echo ">> cmake flags: $FLAGS"
cmake -S "$SRC" -B "$SRC/build" $FLAGS
cmake --build "$SRC/build" --config Release -j --target ggml-rpc-server llama-server
echo ">> built: $SRC/build/bin/{ggml-rpc-server,llama-server}"
