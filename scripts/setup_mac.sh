#!/usr/bin/env bash
# Amalfi — one-shot macOS setup for a node.
# Ensures Xcode Command Line Tools (git + clang + python3), clones the repo if needed,
# then builds llama.cpp (Metal) and probes this machine. Idempotent — safe to re-run.
#
#   A) already cloned:   ./scripts/setup_mac.sh
#   B) fresh Mac:        curl -fsSL https://raw.githubusercontent.com/shahinmv/project-amalfi/main/scripts/setup_mac.sh | bash
#
# Add a backend explicitly:  ./scripts/setup_mac.sh metal|cpu   (default: auto -> metal on Apple Silicon)
set -euo pipefail
REPO_URL="https://github.com/shahinmv/project-amalfi"
BACKEND="${1:-auto}"

echo "== [1/4] Xcode Command Line Tools (git + clang + python3) =="
if ! xcode-select -p >/dev/null 2>&1; then
  echo ">> installing Command Line Tools — a GUI dialog will pop up; click Install."
  xcode-select --install || true
  echo ">> When it finishes, re-run this command. (Exiting for now.)"
  exit 1
fi
echo ">> present, skipping"

echo "== [2/4] locating repo =="
if [ -f "./requirements.txt" ] && [ -d "./scripts" ]; then
  REPO="$(pwd)"
elif [ -f "$(dirname "$0")/../requirements.txt" ]; then
  REPO="$(cd "$(dirname "$0")/.." && pwd)"
else
  REPO="$HOME/Desktop/project-amalfi"
  if [ ! -f "$REPO/requirements.txt" ]; then
    echo ">> cloning $REPO_URL -> $REPO"
    git clone "$REPO_URL" "$REPO"
  fi
fi
cd "$REPO"
git pull --ff-only >/dev/null 2>&1 || true
echo ">> repo: $REPO"

echo "== [3/4] build + probe (backend: $BACKEND) =="
chmod +x scripts/*.sh 2>/dev/null || true
./scripts/bootstrap.sh --backend "$BACKEND"

echo "== [4/4] done =="
echo "Copy this machine's node_*.json to the coordinator, then merge_nodes.py + plan_split.py there."
