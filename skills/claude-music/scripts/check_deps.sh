#!/usr/bin/env bash
# Check dependencies for claude-music skill
# Output: JSON array of tool statuses
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/../config.json"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
ACE_STEP_DIR=$(read_ace_step_dir "$CONFIG")

if [ "$(uname -s)" = "Darwin" ]; then
    PKG="brew install"
else
    PKG="sudo apt install"
fi

check_tool() {
    local name="$1" cmd="$2" install_hint="$3" required="$4"
    # Split cmd into array for safe execution (no eval)
    read -ra CMD_ARGS <<< "$cmd"
    if "${CMD_ARGS[@]}" &>/dev/null; then
        local version
        version=$("${CMD_ARGS[@]}" 2>&1 | head -1 | sed 's/[^0-9.]//g' | head -c 20)
        echo "{\"name\":\"$name\",\"installed\":true,\"version\":\"$version\",\"required\":$required}"
    else
        echo "{\"name\":\"$name\",\"installed\":false,\"install\":\"$install_hint\",\"required\":$required}"
    fi
}

RESULTS=()
RESULTS+=("$(check_tool "uv" "uv --version" "curl -LsSf https://astral.sh/uv/install.sh | sh" true)")
RESULTS+=("$(check_tool "ffmpeg" "ffmpeg -version" "$PKG ffmpeg" true)")
RESULTS+=("$(check_tool "ffprobe" "ffprobe -version" "$PKG ffmpeg" true)")
RESULTS+=("$(check_tool "jq" "jq --version" "$PKG jq" true)")
RESULTS+=("$(check_tool "ffplay" "ffplay -version" "$PKG ffmpeg" false)")

if [ -d "$ACE_STEP_DIR" ] && [ -f "$ACE_STEP_DIR/pyproject.toml" ]; then
    RESULTS+=("{\"name\":\"ace-step\",\"installed\":true,\"version\":\"1.5\",\"path\":\"$ACE_STEP_DIR\",\"required\":true}")
else
    RESULTS+=("{\"name\":\"ace-step\",\"installed\":false,\"install\":\"Install ACE-Step 1.5 to $ACE_STEP_DIR\",\"required\":true}")
fi

for model in "acestep-v15-turbo" "vae" "Qwen3-Embedding-0.6B"; do
    if [ -d "$ACE_STEP_DIR/checkpoints/$model" ]; then
        RESULTS+=("{\"name\":\"checkpoint:$model\",\"installed\":true,\"required\":true}")
    else
        RESULTS+=("{\"name\":\"checkpoint:$model\",\"installed\":false,\"install\":\"uv run acestep-download\",\"required\":true}")
    fi
done

for lm in "acestep-5Hz-lm-0.6B" "acestep-5Hz-lm-1.7B"; do
    if [ -d "$ACE_STEP_DIR/checkpoints/$lm" ]; then
        RESULTS+=("{\"name\":\"checkpoint:$lm\",\"installed\":true,\"required\":false}")
    else
        RESULTS+=("{\"name\":\"checkpoint:$lm\",\"installed\":false,\"install\":\"uv run acestep-download\",\"required\":false}")
    fi
done

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    RESULTS+=("{\"name\":\"apple-silicon-gpu\",\"installed\":true,\"version\":\"$CHIP\",\"required\":true}")
elif command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
    RESULTS+=("{\"name\":\"nvidia-gpu\",\"installed\":true,\"version\":\"$GPU_NAME\",\"required\":true}")
else
    RESULTS+=("{\"name\":\"nvidia-gpu\",\"installed\":false,\"install\":\"Install NVIDIA drivers\",\"required\":true}")
fi

mkdir -p "$HOME/Music/claude-music-output" 2>/dev/null
RESULTS+=("{\"name\":\"output-dir\",\"installed\":true,\"path\":\"$HOME/Music/claude-music-output\",\"required\":true}")

echo "["
for i in "${!RESULTS[@]}"; do
    [ "$i" -lt "$(( ${#RESULTS[@]} - 1 ))" ] && echo "  ${RESULTS[$i]}," || echo "  ${RESULTS[$i]}"
done
echo "]"
