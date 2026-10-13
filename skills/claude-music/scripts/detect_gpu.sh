#!/usr/bin/env bash
# Detect GPU capabilities for claude-music
# Output: JSON with gpu_name, driver, vram_total_mb, vram_free_mb, tier, recommendations
set -euo pipefail

# Apple Silicon: ACE-Step runs on MLX/MPS and the GPU shares system memory, so
# the tier comes from total RAM and there is no separate VRAM figure.
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    MEM_TOTAL=$(( $(sysctl -n hw.memsize) / 1048576 ))
    if [ "$MEM_TOTAL" -ge 48000 ]; then
        TIER="high"; REC_LM="acestep-5Hz-lm-1.7B"
    elif [ "$MEM_TOTAL" -ge 24000 ]; then
        TIER="standard"; REC_LM="acestep-5Hz-lm-1.7B"
    else
        TIER="low"; REC_LM="none"
    fi
    cat <<EOF
{
  "gpu_detected": true,
  "backend": "mps",
  "gpu_name": "$CHIP",
  "unified_memory": true,
  "memory_total_mb": $MEM_TOTAL,
  "tier": "$TIER",
  "recommended_model": "acestep-v15-turbo",
  "recommended_lm": "$REC_LM",
  "notes": "Timings and memory limits: references/apple-silicon.md"
}
EOF
    exit 0
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo '{"gpu_detected":false,"error":"nvidia-smi not found"}'
    exit 0
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
CUDA_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)

if [ "$VRAM_FREE" -ge 16000 ]; then
    TIER="xl"; REC_MODEL="acestep-v15-xl-turbo"; REC_LM="acestep-5Hz-lm-1.7B"
elif [ "$VRAM_FREE" -ge 12000 ]; then
    TIER="high"; REC_MODEL="acestep-v15-turbo"; REC_LM="acestep-5Hz-lm-1.7B"
elif [ "$VRAM_FREE" -ge 8000 ]; then
    TIER="standard"; REC_MODEL="acestep-v15-turbo"; REC_LM="acestep-5Hz-lm-0.6B"
elif [ "$VRAM_FREE" -ge 4000 ]; then
    TIER="low"; REC_MODEL="acestep-v15-turbo"; REC_LM="none"
else
    TIER="minimal"; REC_MODEL="acestep-v15-turbo"; REC_LM="none"
fi

cat <<EOF
{
  "gpu_detected": true,
  "gpu_name": "$GPU_NAME",
  "driver_version": "$DRIVER",
  "vram_total_mb": $VRAM_TOTAL,
  "vram_free_mb": $VRAM_FREE,
  "vram_used_mb": $VRAM_USED,
  "compute_capability": "$CUDA_CAP",
  "tier": "$TIER",
  "recommended_model": "$REC_MODEL",
  "recommended_lm": "$REC_LM"
}
EOF
