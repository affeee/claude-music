#!/usr/bin/env bash
# Train a named voice adapter (LoKr) from one clean vocal: clips, dataset, preprocess, train, install.
# Usage: bash music_train_voice.sh <name> <clean-vocal.flac> [--lyrics listen.json] [--caption "..."] [--dry-run]
# Stdlib-only orchestrator; Side-Step runs in the ACE-Step venv via sidestep_runner.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/music_train_voice.py" "$@"
