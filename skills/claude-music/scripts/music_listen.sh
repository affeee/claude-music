#!/usr/bin/env bash
# Listening report for a generated vocal: transcript, pitch, balance, timing,
# spectrogram. Usage: bash music_listen.sh <audio> [--source ...] [--lyrics ...]
# Runs in a throwaway uv environment so nothing is installed into ACE-Step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v uv &>/dev/null; then
    echo '{"success":false,"error":"uv not found","suggestion":"Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"}'
    exit 1
fi

# Whisper backend: MLX on Apple Silicon, CTranslate2 elsewhere.
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    WHISPER="mlx-whisper"
else
    WHISPER="faster-whisper"
fi

exec uv run --quiet --no-project \
    --with librosa --with soundfile --with matplotlib --with "$WHISPER" \
    python3 "$SCRIPT_DIR/music_listen.py" "$@"
