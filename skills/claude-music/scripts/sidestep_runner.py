"""Run ACE-Step's Side-Step trainer (train_fixed) with three Apple Silicon fixes.

Runs inside the ACE-Step venv (music_train_voice.py starts it with
`uv run python sidestep_runner.py <train_fixed args>`, cwd = ace_step_dir).
ACE-Step itself is left untouched; only these calls are patched:

1. torchaudio.load -> soundfile. torchaudio >= 2.9 loads audio through
   torchcodec, whose prebuilt dylibs only link against FFmpeg 4-8, so every
   load fails next to Homebrew FFmpeg 9. The clips are plain WAV.
2. Fabric.setup casts trainable params to fp32. On MPS Fabric runs
   "16-mixed" with a GradScaler, which refuses fp16 gradients, and
   trainer_fixed casts the whole model to fp16 right before setup.
3. Fabric.clip_gradients gets error_if_nonfinite=False. The GradScaler starts
   at a huge scale, so the first steps overflow to inf; the scaler would skip
   them and lower its scale, but clip_gradients raises first.
"""
import runpy
import sys

import soundfile as sf
import torch
import torchaudio
from lightning.fabric import Fabric

TRAIN_MODULE = "acestep.training_v2.cli.train_fixed"


def _load(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


_orig_setup = Fabric.setup
_orig_clip = Fabric.clip_gradients


def _setup_fp32_trainables(self, module, *optimizers, **kwargs):
    # In-place .data keeps the Parameter objects the optimizer already holds.
    with torch.no_grad():
        for p in module.parameters():
            if p.requires_grad and p.is_floating_point() and p.dtype != torch.float32:
                p.data = p.data.float()
    return _orig_setup(self, module, *optimizers, **kwargs)


def _clip_allow_nonfinite(self, module, optimizer, *args, **kwargs):
    kwargs["error_if_nonfinite"] = False
    return _orig_clip(self, module, optimizer, *args, **kwargs)


torchaudio.load = _load
Fabric.setup = _setup_fp32_trainables
Fabric.clip_gradients = _clip_allow_nonfinite

if __name__ == "__main__":
    sys.argv[0] = TRAIN_MODULE
    runpy.run_module(TRAIN_MODULE, run_name="__main__", alter_sys=True)
