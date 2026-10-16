#!/usr/bin/env bash
# Clean a generated lego layer down to the isolated stem.
#
# ACE-Step's lego output is not a clean stem: it carries quiet backing and
# extra drums. Demucs (htdemucs_ft, two stems) keeps only the voice, then
# ffmpeg adeclick removes the faint clicks the separation leaves behind.
#
# Usage: bash music_clean.sh <layer> [--out PATH] [--declick-threshold N]
# Output: JSON on stdout ({"success", "clean_path", "seconds"}); progress on stderr.
# Runs Demucs in a throwaway uv environment with pinned versions, so nothing
# is installed into ACE-Step. --shifts 0 turns off Demucs' random time shift,
# so the same layer always gives the same clean stem.
set -euo pipefail

DEMUCS_PKGS=(--python 3.11 --with "demucs==4.0.1" --with "torch==2.5.1"
             --with "torchaudio==2.5.1" --with "soundfile")
DEMUCS_MODEL="htdemucs_ft"
DECLICK_THRESHOLD="1.5"

fail() {
    printf '{"success":false,"error":"%s"}\n' "$1"
    exit 1
}

[ $# -ge 1 ] || fail "usage: music_clean.sh <layer> [--out PATH] [--declick-threshold N]"
LAYER="$1"; shift
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --declick-threshold) DECLICK_THRESHOLD="$2"; shift 2 ;;
        *) fail "unknown option: $1" ;;
    esac
done

[ -f "$LAYER" ] || fail "layer not found: $LAYER"
command -v uv &>/dev/null || fail "uv not found"
command -v ffmpeg &>/dev/null || fail "ffmpeg not found"
[[ "$DECLICK_THRESHOLD" =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "--declick-threshold must be a number"

if [ -z "$OUT" ]; then
    OUT="${LAYER%.*}_clean.flac"
    n=2
    while [ -e "$OUT" ]; do OUT="${LAYER%.*}_clean-$n.flac"; n=$((n + 1)); done
fi
[ -e "$OUT" ] && fail "output exists: $OUT"

# MPS fails in Demucs ("Output channels > 65536 not supported"); CPU takes ~3 min
# for a 3-minute track. Use CUDA when there is one.
DEVICE="cpu"
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then DEVICE="cuda"; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
START=$(date +%s)

echo "Separating vocals with Demucs ($DEMUCS_MODEL, $DEVICE) ..." >&2
uv run --quiet --no-project "${DEMUCS_PKGS[@]}" python3 -I -m demucs \
    --two-stems vocals -n "$DEMUCS_MODEL" -d "$DEVICE" --shifts 0 --flac -o "$TMP" "$LAYER" >&2 \
    || fail "demucs separation failed"

STEM="$(find "$TMP" -name 'vocals.flac' | head -1)"
[ -n "$STEM" ] || fail "demucs produced no vocals stem"

echo "Removing clicks ..." >&2
ffmpeg -n -v error -i "$STEM" \
    -af "adeclick=window=55:overlap=75:arorder=2:threshold=$DECLICK_THRESHOLD" \
    -c:a flac "$OUT" || fail "ffmpeg adeclick failed"

python3 -c 'import json,sys; print(json.dumps({"success": True, "clean_path": sys.argv[1], "seconds": int(sys.argv[2]), "model": sys.argv[3], "declick_threshold": float(sys.argv[4])}))' \
    "$OUT" "$(( $(date +%s) - START ))" "$DEMUCS_MODEL" "$DECLICK_THRESHOLD"
