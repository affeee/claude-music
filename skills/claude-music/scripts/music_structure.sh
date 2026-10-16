#!/usr/bin/env bash
# Structure analysis (tempo, key, beat entry, drop) for placing a vocal layer.
# Usage: bash music_structure.sh <audio>
# Runs in a throwaway uv environment so nothing is installed into ACE-Step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v uv &>/dev/null; then
    echo '{"success":false,"error":"uv not found","suggestion":"Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"}'
    exit 1
fi

exec uv run --quiet --no-project --with librosa --with soundfile \
    python3 "$SCRIPT_DIR/music_structure.py" "$@"
