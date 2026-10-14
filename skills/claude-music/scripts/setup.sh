#!/usr/bin/env bash
# Setup and verify ACE-Step installation for claude-music
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/../config.json"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
ACE_STEP_DIR=$(read_ace_step_dir "$CONFIG")
OUTPUT_DIR="$HOME/Music/claude-music-output"

echo "=== claude-music Setup ==="
echo ""

if [ ! -d "$ACE_STEP_DIR" ]; then
    echo "ERROR: ACE-Step 1.5 not found at $ACE_STEP_DIR"
    exit 1
fi
echo "[OK] ACE-Step 1.5 found at $ACE_STEP_DIR"

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "[OK] uv $(uv --version 2>&1 | head -1)"

command -v ffmpeg &>/dev/null && echo "[OK] ffmpeg found" || echo "WARNING: ffmpeg not found (brew install ffmpeg / sudo apt install ffmpeg)"
command -v jq &>/dev/null && echo "[OK] jq found" || echo "WARNING: jq not found (brew install jq / sudo apt install jq)"

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    echo "[OK] GPU: $CHIP ($(( $(sysctl -n hw.memsize) / 1073741824 ))GB unified memory, MLX/MPS)"
elif command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "[OK] GPU: $GPU (${VRAM}MB VRAM)"
else
    echo "WARNING: nvidia-smi not found"
fi

echo ""
echo "Checking model checkpoints..."
MISSING=0
for model in "acestep-v15-turbo" "vae" "Qwen3-Embedding-0.6B"; do
    [ -d "$ACE_STEP_DIR/checkpoints/$model" ] && echo "  [OK] $model" || { echo "  [MISSING] $model"; MISSING=$((MISSING + 1)); }
done
for model in "acestep-5Hz-lm-0.6B" "acestep-5Hz-lm-1.7B" "acestep-v15-base" "acestep-v15-xl-turbo"; do
    [ -d "$ACE_STEP_DIR/checkpoints/$model" ] && echo "  [OK] $model (optional)" || echo "  [--] $model (optional)"
done

[ "$MISSING" -gt 0 ] && echo "" && echo "Missing required checkpoints. Run: cd $ACE_STEP_DIR && uv run acestep-download"

mkdir -p "$OUTPUT_DIR"
echo ""
echo "[OK] Output: $OUTPUT_DIR"
echo ""
echo "=== Setup Complete ==="
