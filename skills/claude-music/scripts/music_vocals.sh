#!/usr/bin/env bash
# Sing over an existing instrumental, end to end (analysis, vocal, clean, mix, listen).
# Usage: bash music_vocals.sh <instrumental> --lyrics lyrics.txt [--voice aria] [...]
# Stdlib-only orchestrator; each step runs in its own environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/music_vocals.py" "$@"
