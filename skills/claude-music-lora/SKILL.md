---
name: claude-music-lora
description: >
  LoRA and LoKr fine-tuning for ACE-Step 1.5. Trains custom styles from 3-10 songs,
  manages trained adapters, and applies them during generation. Uses ACE-Step's
  built-in training pipeline.
when_to_use: >
  Use when the user asks to train a LoRA, fine-tune, clone a voice, make a custom
  style, or use LoKr for personalized music models.
allowed-tools:
  - Bash
  - Read
  - Write
---

# claude-music-lora — LoRA Fine-Tuning

## Overview

Train custom LoRA adapters to capture specific vocal styles, genres, instrument sounds, or production aesthetics. Requires 3-10 songs as training data.

## Training a Named Voice (one command)

To lock in a voice the user likes (e.g. "Aria") from one clean, isolated vocal, use the voice-training flow. It reproduces the recipe that trained Aria on a 16 GB M1 and installs the result as a voice profile that `--voice <name>` picks up.

```bash
bash ~/.claude/skills/claude-music/scripts/music_train_voice.sh <name> <clean-vocal.flac> \
  --lyrics listen.json --caption "female singer, breathy airy voice, ..." [--bpm 129 --key "C minor"]

# check the clips and dataset first (seconds, no model load):
bash ~/.claude/skills/claude-music/scripts/music_train_voice.sh <name> <clean-vocal.flac> --caption "..." --dry-run

# retrain on the first take plus approved new takes (bootstrapping), and keep the
# current adapter until the user has picked a checkpoint by ear:
bash ~/.claude/skills/claude-music/scripts/music_train_voice.sh <name> take1.flac --lyrics take1.json \
  --add t2_clean.flac t2.json --add t4_clean.flac t4.json --epochs 60 --no-install

# install another checkpoint from an earlier run after listening:
bash ~/.claude/skills/claude-music/scripts/music_train_voice.sh <name> --install <run> --epoch 70
```

Aria v3 (2026-10-17) came from this: take 1 plus four approved 30 s takes (31 clips, 60 epochs, about 5.5 min per epoch on the M1). Pitch wobble (p90 cents of frame-to-frame jitter / deviation from a 150 ms median) fell from 14/18 with v2 to 8/12 at epoch 30, against 6/10 for take 1. Changing the caption or `lora_scale` barely moved it, so more approved takes is the lever. Correct Whisper's timed lyrics by hand before training on them: it mishears generated vocals.

What it does, in order:

1. **Clips**: overlapping 20 s windows with an 8 s hop, cut to 48 kHz stereo WAV. Only windows with sung lines are kept: with timed lyrics, a window must hold at least one line (more than half of it inside); without, at least 5 s of the window must be within 30 dB of the loudest part.
2. **Dataset**: Side-Step JSON with `custom_tag` = the trigger word (default: the name capitalized), `tag_position` prepend, `genre_ratio` 0, per-clip lyrics. Captions are stored without the tag; Side-Step prepends it.
3. **Preprocess** with `--precision fp32`. fp16 on MPS put NaN into `encoder_hidden_states`, and the loss was NaN.
4. **Train**: LoKr dim 16 / alpha 32, lr 3e-4, Adafactor, base model, fp32, batch 1, `--num-workers 0`, `--offload-encoder`, `--gradient-checkpointing`, a checkpoint every 10 epochs, 80 epochs, `PYTHONUNBUFFERED=1`.
5. **Install** (skipped with `--no-install`): the chosen checkpoint (default: lowest loss; `--epoch N` to pick) goes to `<voices_dir>/<name>/adapter/lokr_weights.safetensors`, and `profile.json` gets `lego.lora: "adapter"` and a caption starting with "<Tag>, ...". The previous adapter (`adapter/previous/`) and profile (`profile.<timestamp>.json`) are kept.

`--lyrics` takes timed lyrics: a listening report from `music_listen.sh` (its `transcript.segments`), a JSON list of `{start, end, text}`, or LRC (`[mm:ss.xx] line`; bare `[Chorus]` lines set the section). Untimed lyrics or none give `[Verse]` tags only. Each `--add VOCAL [LYRICS]` is another clean take of the same voice, cut the same way; its clips get no bpm/key, since they come from other songs. `--caption` defaults to the voice profile's `lego.caption` when the voice already has one.

Files: clips, `dataset.json`, `preprocess.log`, `train.log` and `run.json` (settings and the exact commands) in `<voices_dir>/<name>/training/<run>/`. Side-Step only accepts paths inside the ACE-Step dir, so tensors go to `<ace_step_dir>/datasets/<name>/<run>/` and checkpoints to `<ace_step_dir>/lora_output/<name>/<run>/checkpoints/epoch_N_loss_L/`. It runs with cwd = `ace_step_dir` through `scripts/sidestep_runner.py`, which patches three things without touching ACE-Step: `torchaudio.load` → soundfile (torchcodec cannot load Homebrew FFmpeg 9), trainable params cast to fp32 at `Fabric.setup` (the GradScaler refuses fp16 gradients), and `error_if_nonfinite=False` on `Fabric.clip_gradients` (the first scaled steps overflow).

Rules:

- **It takes hours.** Aria (19 clips, 80 epochs) took about 5 h on the M1 at ~3.7 min per epoch. Always run it in the background, tell the user the estimate up front, and follow the NDJSON `epoch` events on stderr (`eta_sec` included). Do a `--dry-run` first and show the clip count and per-clip lyrics.
- **Never kill a run because the log looks quiet** before checking the TensorBoard events in `<output>/runs/`.
- **Stay with LoKr dim 16 on 16 GB.** A PEFT LoRA r32 in fp32 thrashed swap, and the engine refuses PEFT LoRA on Apple Silicon anyway.
- **Let the user choose the checkpoint by ear.** Lowest loss is only a default; for Aria, epochs 50-80 sounded alike. Render the same seed and caption with each candidate (`lego --lora <checkpoint dir>`), then `--install <run> --epoch N`.
- **Training data must be the voice the user approved**, clean and isolated (run `music_clean.sh` first). Takes the user rejected teach the wrong voice.

## Dataset Preparation

1. Collect 3-10 songs in the target style (WAV/FLAC preferred, MP3 OK)
2. Place in a directory: `~/Music/lora-datasets/<style_name>/`
3. Songs should be 30-300 seconds each
4. Consistent style/genre across the dataset
5. High audio quality (no noise, no clipping)

## Training

```bash
cd "$(python3 -c "import json; print(json.load(open('$HOME/.claude/skills/claude-music/config.json'))['ace_step_dir'])")"

# LoRA training (standard, ~1 hour on RTX 5070 Ti)
uv run python3 -m acestep.training.train_lora \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --dataset-dir ~/Music/lora-datasets/my_style/ \
  --output-dir ./lora_output/my_style \
  --rank 16 \
  --learning-rate 1e-4 \
  --steps 1000

# LoKr training (5x faster, ~12 min)
uv run python3 -m acestep.training.train_lora \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --dataset-dir ~/Music/lora-datasets/my_style/ \
  --output-dir ./lora_output/my_style \
  --method lokr \
  --rank 16 \
  --learning-rate 1e-4 \
  --steps 500
```

## LoRA vs LoKr

| Aspect | LoRA | LoKr |
|--------|------|------|
| Training time | ~1 hour | ~12 min |
| Quality | Higher fidelity | Good, slightly less detailed |
| VRAM | ~10GB | ~8GB |
| Use case | Voice cloning, precise style | Genre adaptation, quick experiments |

## Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Rank (r) | 16 | 4-64 | Higher = more capacity, more VRAM |
| Learning rate | 1e-4 | 1e-5 to 5e-4 | Lower for voice cloning |
| Steps | 1000 | 200-5000 | More data = more steps needed |
| Batch size | 1 | 1-4 | Limited by VRAM |

## Using Trained LoRA

After training, the LoRA is available in ACE-Step's generation pipeline. Refer to ACE-Step documentation at:
`<ace_step_dir>/docs/en/LoRA_Training_Tutorial.md` (see config.json for path)

For detailed reference: load `references/lora-training.md`
