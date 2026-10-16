#!/usr/bin/env bash
# Sing over an existing instrumental, end to end (analysis, vocal, optional clean, mix, listen).
# Usage: bash music_vocals.sh <instrumental> --lyrics lyrics.txt [--voice aria] [--clean] [...]
#        bash music_vocals.sh --clean-run <run folder>   (clean a take afterwards)
# Stdlib-only orchestrator; each step runs in its own environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/music_vocals.py" "$@"
