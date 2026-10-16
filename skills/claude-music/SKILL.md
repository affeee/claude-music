---
name: claude-music
description: >
  Music production suite using ACE-Step 1.5 via Python API. Routes /music commands
  for generation, cover, repaint, compose, analyze, export, enhance, random, and
  LoRA training. 50+ languages, up to 10-minute tracks, 48kHz stereo.
when_to_use: >
  Use when the user says /music, asks to generate a song, create music, make a
  track, cover/remix/repaint/edit audio, write lyrics, compose, analyze BPM/key,
  export for Spotify/TikTok, master audio, train a LoRA, or mentions ACE-Step.
compatibility: >
  Requires ACE-Step 1.5 installed locally + NVIDIA GPU with ≥4 GB VRAM
  (≥12 GB recommended). Path configured in skills/claude-music/config.json.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

# claude-music — AI Music Production for Claude Code

## Quick Reference

| Command | What it does |
|---------|-------------|
| `/music` | Interactive mode — describe what you want |
| `/music generate` | Text/lyrics to full song (text2music) |
| `/music cover` | Style transfer from reference audio |
| `/music repaint` | Edit a specific section of a song |
| `/music extract` | Separate tracks/stems (base model only) |
| `/music lego` | Add instrument layer (base model only) |
| `/music vocals` | Sing over an existing instrumental, end to end (analysis, vocal, clean, mix, listen) |
| `/music complete` | Continue/extend audio (base model only) |
| `/music compose` | Songwriting: craft caption + lyrics + params |
| `/music analyze` | BPM, key, loudness, duration analysis |
| `/music export` | Platform-optimized export (Spotify, YouTube, etc.) |
| `/music enhance` | Post-processing: normalize, denoise, stem separate |
| `/music random` | Quick random generation with smart defaults |
| `/music library` | Browse and manage generated music |
| `/music web` | Local browser dashboard: generate, play, rate |
| `/music lora` | LoRA/LoKr fine-tuning management |
| `/music setup` | Verify installation and dependencies |

## Orchestration Logic

### Command Routing

When the user provides a specific command, load the matching sub-skill:

- `/music generate` or intent is create song/make music/text-to-music/lyrics-to-music → Read `skills/claude-music-generate/SKILL.md`
- `/music cover` or intent is cover/style transfer/remake/version of → Read `skills/claude-music-cover/SKILL.md`
- `/music repaint` or intent is edit section/fix chorus/change part/modify section → Read `skills/claude-music-repaint/SKILL.md`
- `/music compose` or intent is write lyrics/craft caption/plan song/songwriting → Read `skills/claude-music-compose/SKILL.md`
- `/music analyze` or intent is BPM/key detection/loudness/audio info → Read `skills/claude-music-analyze/SKILL.md`
- `/music export` or intent is export for Spotify/YouTube/platform/format conversion → Read `skills/claude-music-export/SKILL.md`
- `/music enhance` or intent is normalize/denoise/stem separate/master → Read `skills/claude-music-enhance/SKILL.md`
- `/music random` or intent is quick generation/surprise me/random song → Read `skills/claude-music-random/SKILL.md`
- `/music library` or intent is list songs/browse output/manage music → Read `skills/claude-music-library/SKILL.md`
- `/music web` or intent is dashboard/browser app/web UI/visual player → Read `skills/claude-music-web/SKILL.md`
- `/music lora` or intent is train/fine-tune/LoRA/custom style → Read `skills/claude-music-lora/SKILL.md`
- `/music vocals` or intent is add vocals/singing/lyrics to an existing instrumental **without changing it** → see "Singing Over an Instrumental (vocal flow)" below. Do not use cover for this: cover regenerates the whole song.
- `/music lego` or intent is add an instrument (or a vocal with hand-picked settings) on top of an existing track → see "Adding a Layer to an Existing Track" below.
- `/music setup` → Run `bash ~/.claude/skills/claude-music/scripts/setup.sh`

### Interactive Mode

When user says `/music` without arguments or describes a task in natural language:
1. Run `bash ~/.claude/skills/claude-music/scripts/check_deps.sh` to verify tools
2. Run `bash ~/.claude/skills/claude-music/scripts/detect_gpu.sh` for GPU info
3. Identify intent from the user's description
4. Route to the appropriate sub-skill
5. If ambiguous, ask the user to clarify

### Multi-Step Pipelines

For complex requests spanning multiple sub-skills (e.g., "compose lyrics, generate a song, then export for Spotify"):
1. Compose lyrics/caption with `/music compose`
2. Generate with `/music generate` using composed output
3. Export with `/music export`
4. Clean up temp files

### Generate-Listen-Iterate Loop

After any generation:
1. Present output file paths and metadata (seed, duration, format)
2. Suggest playback: `ffplay -nodisp -autoexit "<path>"`
3. Ask if user wants to:
   - **Re-generate** with different seed (same params)
   - **Refine** params (adjust caption, BPM, quality)
   - **Repaint** a specific section
   - **Cover** to change style while keeping structure
   - **Export** for a platform

## Safety Rules — MANDATORY

1. **Run preflight before writes**: `bash ~/.claude/skills/claude-music/scripts/preflight.sh "$INPUT" "$OUTPUT"`
2. **Never overwrite source files** — all operations produce new files
3. **Check VRAM before GPU operations**: `bash ~/.claude/skills/claude-music/scripts/detect_gpu.sh`
4. **Confirm before**: batch >4 generations, operations with --quality max (3-5 min), and on Apple Silicon any base-model run on a full-length track (15-25 min)
5. **Auto-execute without confirmation**: single generation (draft/standard), analysis, format conversion, setup
6. **Temp files**: `/tmp/claude-music/` with cleanup trap
7. **Output directory**: `~/Music/claude-music-output/` (auto-created)

## ACE-Step Configuration

- **Installation**: Set `ace_step_dir` in `config.json` (default: see config.json)
- **Invocation**: `bash ~/.claude/skills/claude-music/scripts/music_engine.sh <command> [args]`
- **Config**: `~/.claude/skills/claude-music/config.json`
- **Output**: `~/Music/claude-music-output/`

## Singing Over an Instrumental (vocal flow)

The default for "put vocals on my song". One command runs the whole flow, the
same way every time, with the original left untouched:

```bash
bash ~/.claude/skills/claude-music/scripts/music_vocals.sh "$SRC" \
  --lyrics lyrics.txt --voice aria
# or without a voice profile:
bash ~/.claude/skills/claude-music/scripts/music_vocals.sh "$SRC" \
  --lyrics lyrics.txt --caption "warm male baritone, clear diction" --seed 7
```

1. **Structure** (`music_structure.sh`): bpm, key, beat entry (`first_beat_sec`), drop (`drop_sec`, the largest rise onto the loud plateau).
2. **Vocal layer** (`music_engine.sh lego --track vocals`) from the beat entry. When it starts mid-song, the voice profile's throwaway `[Intro]` line ("Ooh, ooh") is prepended; without it the model tends to drop the first lyric line.
3. **Clean** (`--clean` → `music_clean.sh`): the lego layer carries quiet backing and extra drums. Demucs `htdemucs_ft` (two stems, `--shifts 0`, pinned versions, CPU on Apple Silicon) keeps the voice, then ffmpeg `adeclick` removes faint clicks. ~3-8 min.
4. **Mix** (`--mix`): the clean stem over the original; base level `--mix-gain`, lifted by `--boost` dB (default 5) from the drop with a 0.5 s ramp, so the vocal holds up against the full-energy part.
5. **Listening report** on the clean stem (see below).

Each run gets its own folder `<output_dir>/<song>-vocals_<timestamp>/` with `lyrics.txt` (as sung, intro line included), `structure.json`, `engine.json`, `listen.json`, `listen.png` and `run.json` (every setting plus the repo commit). Rerunning with the same voice, lyrics and seed reproduces the run. Overrides: `--start auto|SEC`, `--boost-at auto|none|SEC`, `--boost DB`, `--mix-gain DB`, `--steps N`, `--no-intro-line`, `--no-listen`.

- **Write lyrics first** (`/music compose`): short lines, few words and a repeated hook are sung far more clearly than dense verses.
- **It is slow on Apple Silicon**: ~17 min generation + ~5 min cleaning + ~2 min report. Run it in the background, tell the user up front, and show the progress dashboard if the user has asked for one.
- **Report the listening facts**, then ask the user how it sounds.

### Voice profiles

A voice the user likes can be kept as a profile: `<voices_dir>/<name>/profile.json` (`voices_dir` in config.json, default `~/Music/claude-music-voices`). `--voice <name>` (on `music_vocals.sh` and `lego`) fills caption, seed, steps, track and post-processing the user left unset; explicit flags always win. Example: `examples/voices/profile.example.json`.

```json
{"name": "Aria", "version": 1,
 "lego": {"caption": "...", "seed": 20261016, "inference_steps": 50, "track": "vocals"},
 "post": {"clean": true, "mix_gain_db": 0.0, "boost_db": 5.0},
 "lyrics": {"intro_line": "Ooh, ooh"}}
```

**Trained voice adapter**: `lego.lora` (path relative to the profile folder, e.g. `"adapter"` holding `lokr_weights.safetensors`) plus `lora_scale` locks the voice far better than caption + seed; start the caption with the adapter's trigger word (e.g. "Aria, ..."). Engine flags: `--lora PATH --lora-scale 0-1`. On Apple Silicon only LoKr adapters work: the engine merges them into the DiT weights and builds the MLX decoder once afterwards (`merge_lokr_into_mlx`), so memory matches a normal run. Never fall back to the PyTorch decoder on a 16 GB Mac: it went past 18 GB and froze the machine. The melody differs per take by design; the adapter carries the timbre.

Without an adapter, be upfront that a profile gives a similar voice, not the identical one: a new song, key or lyrics changes it. Keep the clean stems of takes the user likes in `<voices_dir>/<name>/samples/`; they are the dataset for training the voice later (`/music lora`).

## Adding a Layer to an Existing Track (lego)

Use for instruments, or for a vocal when you need settings the vocal flow does
not expose. For "sing over my song", use the vocal flow above.

```bash
bash ~/.claude/skills/claude-music/scripts/music_engine.sh --quality standard --progress lego \
  --src-audio "$SRC" --track vocals --mix \
  --caption "clear female lead vocal, catchy pop topline" \
  --lyrics "$LYRICS" --language sv --bpm 129 --key "C minor" \
  --inference-steps 50
```

- **Analyze first** (`/music analyze`): pass the source's real `--bpm` and `--key`, and lay the lyric sections out against its energy curve.
- **`--track`** is one of vocals, backing_vocals, drums, bass, guitar, keyboard, percussion, strings, synth, fx, brass, woodwinds. Without it the model guesses which layer to add.
- **`--start` / `--end`** (seconds) limit where the layer plays, e.g. `--start 29.8` to bring vocals in after a 16-bar intro at 129 BPM. Snap them to bar lines: one bar is `240 / bpm` seconds in 4/4. Outside that range the layer file contains the source audio, not silence; `--mix` mutes it there, and a manual mix must do the same.
- **Listen afterwards**: run `music_listen.sh` on the layer (next section) and report what it finds.
- **The output is the new layer alone**, not the full song, and it carries backing bleed. `--clean` writes `<name>_clean.flac` (Demucs + declick) and reports it as `clean_path`. `--mix` also writes `<name>_mix.<ext>` (source + layer, the clean stem when `--clean`, limiter on top) and reports it as `mix_path`. Rebalance with `--mix-gain <dB>`, lift from a point on with `--mix-boost-at SEC --mix-boost DB`, or re-mix the stem with ffmpeg; that needs no new generation.
- **`--voice <name>`** applies a voice profile (see above).
- **Needs the base model**: `cd <ace_step_dir> && uv run acestep-download --model acestep-v15-base --skip-main` (~4.8 GB). `--quality standard` keeps the LM unloaded, which matters on 16 GB machines; the engine still switches to the base model and its step/guidance settings.
- **It is slow**: about 15 s per step for a 3-minute track on Apple Silicon (16 GB), so ~19 min at 50 steps. Run it in the background and tell the user up front. The engine raises ACE-Step's 600 s generation timeout to an hour (`ACESTEP_GENERATION_TIMEOUT` overrides).
- **Progress**: with `--progress`, stderr carries `{"event": "step", "step": 11, "total": 50, "eta_sec": 547}` lines from the real diffusion loop. Report those.

## Listening to a Result (music_listen)

Claude cannot hear audio. After generating a vocal, run the listening report and
read it before describing the result or asking the user how it sounds:

```bash
bash ~/.claude/skills/claude-music/scripts/music_listen.sh "$LAYER" \
  --source "$SRC" --lyrics lyrics.txt --language en \
  --key "C minor" --bpm 129 --start 29.8 --mix-gain -3 --spectrogram auto
```

Pass the same `--start`, `--end` and `--mix-gain` as the lego run. The JSON reports:

| Field | Answers |
|-------|---------|
| `transcript`, `lyrics` | What is actually sung and when; which expected lines are missing or garbled; the language Whisper heard |
| `activity` | First onset, phrases, and silences of 4 s or more (`gaps`) |
| `pitch` | Median pitch, range, register (bass to soprano), share of notes in the key |
| `balance` | Voice against backing in dB per 10 s window, and how much that varies |
| `timing` | How far phrase starts land from the nearest eighth note |
| `spectrogram` | Path to a PNG; open it with Read to look for dropouts and noise |

- **Report facts, then ask about taste.** "Line 2 of verse 2 is missing, the voice sits around A2, it drops out 80-100 s" is yours to say. Whether it sounds good is the user's.
- **Best on a lego layer**, where the voice is isolated. On a full mix the transcript still works but pitch and balance mean little.
- **Whisper mishears singing.** A low `match` on one line is a hint, not proof; several in a row, or a wrong `language`, is a real finding.
- The first run downloads the Whisper model (~1.6 GB). The report takes 1-2 min for a 3-minute track. `--no-transcribe` skips Whisper.

## Quality Presets

| Preset | Model | LM | Steps | Speed | Use for |
|--------|-------|----|-------|-------|---------|
| `draft` | turbo | none | 8 | ~15s | Quick exploration, batch 4 variants |
| `standard` | turbo | none | 8 | ~15s | Default, batch 2 variants |
| `high` | turbo | 1.7B LM | 8 | ~25s | Better lyrics/structure, thinking mode |
| `max` | base | 1.7B LM | 65 | ~3-5min | Highest quality, single output |

## VRAM Management (RTX 5070 Ti — 16GB)

| Configuration | VRAM | Offload | Notes |
|---------------|------|---------|-------|
| Turbo (no LM) | ~8GB | CPU offload | Default, fast generation |
| Turbo + 0.6B LM | ~10GB | CPU + DiT offload | Thinking mode, lightweight |
| Turbo + 1.7B LM | ~14GB | CPU + DiT offload | Full thinking, tight on VRAM |
| XL Turbo | ~14-16GB | Full offload | Maximum quality DiT, no LM room |

**Rule**: Never run two heavy models simultaneously. The music_engine.py handles VRAM automatically.

## Apple Silicon

When `detect_gpu.sh` reports `"backend": "mps"`, the tables above do not apply: memory is shared, and a 16 GB Mac is far tighter than a 16 GB card. Read `references/apple-silicon.md` before quoting a time or choosing a batch.

- **Timings (M1, 16 GB)**: a full 3-minute track takes about 6 min per variant with turbo and 15-25 min with the base model. A 30 s draft takes 2-3 min.
- **The engine adapts by itself**: one variant for full-length tracks, CPU offload chosen per job, MLX backend for the LM, and an LM that does not fit is swapped or skipped with a warning on stderr. Pass `--batch` or `--offload on|off` to override.
- **On 16 GB, `high` runs without the 1.7B LM**, so it gives the same result as `standard`. Say so rather than promising better lyrics adherence.
- **For another take, run again** instead of raising the batch: two variants at once take as long as two runs.

## Script Invocation

All ACE-Step operations go through the bash wrapper:
```bash
bash ~/.claude/skills/claude-music/scripts/music_engine.sh <subcommand> [args]
```

The wrapper handles: path setup, environment variables, VRAM pre-check, `uv run` invocation.

**Output is always JSON to stdout.** Parse with `jq` for specific fields.

## Reference Files (Load On-Demand)

| Reference | When to load |
|-----------|-------------|
| `references/prompt-guide.md` | When crafting captions or lyrics |
| `references/parameters.md` | When user asks about specific params or tuning |
| `references/genre-recipes.md` | When targeting a specific genre |
| `references/music-theory.md` | When discussing keys, scales, BPM, song structure |
| `references/post-processing.md` | When exporting, mastering, or enhancing |
| `references/song-structures.md` | When planning song layout |
| `references/lora-training.md` | When training custom LoRA models |
| `references/apple-silicon.md` | When `detect_gpu.sh` reports `"backend": "mps"`: timings, memory limits, progress events |

## Sub-Skills

| Skill | Type | Description |
|-------|------|-------------|
| `claude-music-generate` | Generation | Core text2music via ACE-Step Python API |
| `claude-music-cover` | Generation | Style transfer from reference audio |
| `claude-music-repaint` | Editing | Selective section regeneration |
| `claude-music-compose` | Reference | Songwriting guide (caption, lyrics, params) |
| `claude-music-analyze` | Analysis | BPM, key, loudness, metadata via ffprobe/librosa |
| `claude-music-export` | Processing | Platform-specific audio export via FFmpeg |
| `claude-music-enhance` | Processing | Loudness normalization, denoise, stems (reuses video skills) |
| `claude-music-random` | Generation | Quick random generation with genre presets |
| `claude-music-library` | Management | Browse/search generated music output |
| `claude-music-web` | UI | Local browser dashboard (generate, play, rate) |
| `claude-music-lora` | Training | LoRA/LoKr fine-tuning wrapper |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/music_engine.py` | Core ACE-Step Python API wrapper (all 6 task types) |
| `scripts/music_engine.sh` | Bash wrapper (env, VRAM, `uv run`, show-once star nudge) |
| `scripts/music_export.sh` | FFmpeg platform export commands |
| `scripts/music_listen.sh` / `.py` | Listening report: transcript, pitch, balance, timing, spectrogram → JSON |
| `scripts/rank.py` | Batch-rank outputs vs caption (stub — Theme 3 of research plan) |
| `scripts/detect_gpu.sh` | GPU detection (NVIDIA or Apple Silicon) + tier recommendation → JSON |
| `scripts/preflight.sh` | Safety checks for audio files → JSON |
| `scripts/check_deps.sh` | Dependency verification → JSON |
| `scripts/setup.sh` | Installation verification (invoked by `/music setup`) |

Audio analysis (BPM, key, loudness) is handled by `claude-music-analyze` via `ffprobe` + FFmpeg's `loudnorm` filter — no separate Python script; see that sub-skill.
