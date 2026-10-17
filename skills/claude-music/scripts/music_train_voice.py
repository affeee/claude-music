#!/usr/bin/env python3
"""Train a named voice adapter (LoKr) from one clean vocal — claude-music skill.

Reproduces the recipe that made the voice "Aria" on a 16 GB M1:

  1. Clips      overlapping 20 s windows, 8 s hop, 48 kHz WAV, only windows
                with sung lines (timed lyrics, else an energy check)
  2. Dataset    Side-Step JSON, custom_tag = the voice's trigger word
                (tag_position prepend, genre_ratio 0), lyrics per clip
  3. Preprocess fp32 tensors (fp16 on MPS put NaN into encoder_hidden_states)
  4. Train      LoKr dim 16 / alpha 32, lr 3e-4, Adafactor, base model, fp32,
                batch 1, checkpoint every 10 epochs, 80 epochs
  5. Install    chosen checkpoint -> <voices_dir>/<name>/adapter/, profile.json
                gets lego.lora "adapter" and a caption starting with the tag

Side-Step only accepts paths inside the ACE-Step dir, so tensors go to
<ace_step_dir>/datasets/<name>/<run>/ and checkpoints to
<ace_step_dir>/lora_output/<name>/<run>/; it runs with cwd = ace_step_dir via
sidestep_runner.py (Apple Silicon patches). Clips, dataset, logs and run.json
live in <voices_dir>/<name>/training/<run>/.

Usage:
  music_train_voice.py aria clean_vocal.flac --lyrics listen.json --caption "female singer, ..."
  music_train_voice.py aria clean_vocal.flac --dry-run          # stop after the dataset
  music_train_voice.py aria --install aria-20261016-0100 --epoch 70

--lyrics takes timed lyrics: a listening report (music_listen.sh JSON), a JSON
list of {start, end, text}, or LRC ("[mm:ss.xx] line"; bare "[Chorus]" lines
set the section). Untimed or missing lyrics give "[Verse]" tags only.

One JSON object on stdout. Progress on stderr: NDJSON {"event": "stage"|"epoch"}
lines plus plain log lines. Stdlib only; training runs in the ACE-Step venv.
"""

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from array import array
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "config.json"
RUNNER = SCRIPT_DIR / "sidestep_runner.py"
DEFAULT_VOICES_DIR = "~/Music/claude-music-voices"
VOICE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
ADAPTER_FILE = "lokr_weights.safetensors"

CLIP_SEC = 20.0
HOP_SEC = 8.0
SAMPLE_RATE = 48000
ENERGY_FRAME_SEC = 0.25
ENERGY_RATE = 8000          # decode rate for the energy check only
ENERGY_FLOOR_DB = -50.0     # never call quieter than this "sung"
ENERGY_RANGE_DB = 30.0      # sung = within this many dB of the loudest frame
MIN_ACTIVE = 0.25           # a clip needs this share of sung frames (5 s of 20)

# The Aria recipe. Change only after a re-measured run on the target machine.
TRAIN_ARGS = {
    "--model-variant": "base",
    "--adapter-type": "lokr",
    "--lokr-linear-dim": "16",
    "--lokr-linear-alpha": "32",
    "--lr": "3e-4",
    "--optimizer-type": "adafactor",
    "--batch-size": "1",
    "--gradient-accumulation": "1",
    "--warmup-steps": "50",
    "--num-inference-steps": "50",
    "--num-workers": "0",
    "--precision": "fp32",
}
TRAIN_FLAGS = ("--offload-encoder", "--gradient-checkpointing")
PREPROCESS_MAX_SEC = "60"

EPOCH_RE = re.compile(r"\[OK\] Epoch (\d+)/(\d+) in ([\d.]+)s, Loss: ([\d.]+|nan|inf)")
CHECKPOINT_RE = re.compile(r"^epoch_(\d+)_loss_([\d.]+|nan|inf)$")
LRC_RE = re.compile(r"^\s*\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$")
SECTION_RE = re.compile(r"^\s*\[([A-Za-z][A-Za-z0-9 \-]*)\]\s*$")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def stage(name, **fields):
    print(json.dumps({"event": "stage", "stage": name, **fields}), file=sys.stderr, flush=True)


def fail(msg, **extra):
    print(json.dumps({"success": False, "error": msg, **extra}, indent=2))
    sys.exit(1)


def load_config():
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def default_tag(name):
    """Trigger word from a voice name: 'aria' -> 'Aria', 'deep-alto' -> 'Deep Alto'."""
    return " ".join(w.capitalize() for w in name.split("-") if w)


def clip_windows(duration, length=CLIP_SEC, hop=HOP_SEC):
    """Overlapping (start, end) windows covering the whole file.

    A file no longer than one window is one clip. Otherwise windows step by
    hop, and a last window is pinned to the end when the tail would be lost.
    """
    if duration <= 0:
        return []
    if duration <= length:
        return [(0.0, round(duration, 3))]
    wins, start = [], 0.0
    while start + length <= duration + 1e-6:
        wins.append((round(start, 3), round(start + length, 3)))
        start += hop
    if wins[-1][1] < duration - 0.5:
        wins.append((round(duration - length, 3), round(duration, 3)))
    return wins


def parse_timed_lyrics(text):
    """Timed lines [{start, end, text, section}] from JSON or LRC, else None.

    JSON: a music_listen report (transcript.segments), {"segments": [...]},
    or a list of {start, end, text}. LRC: "[mm:ss.xx] line"; a line ends where
    the next one starts (the last one 5 s later); bare "[Chorus]" lines set the
    section of the lines after them.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if data is not None:
        if isinstance(data, dict):
            data = (data.get("transcript") or {}).get("segments") or data.get("segments")
        if not isinstance(data, list):
            return None
        lines = []
        for seg in data:
            if not isinstance(seg, dict) or not str(seg.get("text") or "").strip():
                continue
            try:
                start, end = float(seg["start"]), float(seg["end"])
            except (KeyError, TypeError, ValueError):
                continue
            lines.append({"start": start, "end": end, "text": str(seg["text"]).strip(),
                          "section": seg.get("section") or "Verse"})
        return lines or None

    lines, section = [], "Verse"
    for raw in text.splitlines():
        m = LRC_RE.match(raw)
        if m:
            words = m.group(3).strip()
            if words:
                lines.append({"start": int(m.group(1)) * 60 + float(m.group(2)), "end": None,
                              "text": words, "section": section})
            elif lines and lines[-1]["end"] is None:
                lines[-1]["end"] = int(m.group(1)) * 60 + float(m.group(2))  # an empty stamp ends a line
            continue
        s = SECTION_RE.match(raw)
        if s:
            section = s.group(1).strip().title()
    if not lines:
        return None
    for i, line in enumerate(lines):
        if line["end"] is None:
            line["end"] = lines[i + 1]["start"] if i + 1 < len(lines) else line["start"] + 5.0
    return lines


def lines_in_window(lines, start, end):
    """Lines sung mostly (at least half their length) inside [start, end)."""
    picked = []
    for line in lines:
        length = max(line["end"] - line["start"], 0.01)
        overlap = min(line["end"], end) - max(line["start"], start)
        if overlap >= length / 2:
            picked.append(line)
    return picked


def clip_lyrics(lines):
    """Structure-tagged lyrics for one clip: a tag whenever the section changes."""
    if not lines:
        return "[Verse]"
    out, section = [], None
    for line in lines:
        if line.get("section") != section:
            section = line.get("section") or "Verse"
            out.append(f"[{section}]")
        out.append(line["text"])
    return "\n".join(out)


def frame_levels_db(samples, rate=ENERGY_RATE, frame_sec=ENERGY_FRAME_SEC):
    """RMS level per frame in dBFS from 16-bit mono samples."""
    n = max(int(rate * frame_sec), 1)
    levels = []
    for i in range(0, len(samples), n):
        chunk = samples[i:i + n]
        if not chunk:
            break
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk)) / 32768.0
        levels.append(20 * math.log10(rms) if rms > 0 else -120.0)
    return levels


def active_fraction(levels, start, end, frame_sec=ENERGY_FRAME_SEC,
                    floor_db=ENERGY_FLOOR_DB, range_db=ENERGY_RANGE_DB):
    """Share of frames in [start, end) loud enough to count as sung."""
    if not levels:
        return 0.0
    threshold = max(floor_db, max(levels) - range_db)
    frames = levels[int(start / frame_sec):max(int(end / frame_sec), int(start / frame_sec) + 1)]
    return sum(1 for lv in frames if lv >= threshold) / len(frames) if frames else 0.0


def select_clips(windows, lines=None, levels=None, min_active=MIN_ACTIVE):
    """Windows worth training on, each with its lyrics.

    With timed lyrics a window must hold at least one line; without, its
    energy must say someone sings in it (share of loud frames >= min_active).
    """
    clips = []
    for start, end in windows:
        if lines is not None:
            picked = lines_in_window(lines, start, end)
            if not picked:
                continue
        else:
            if levels is not None and active_fraction(levels, start, end) < min_active:
                continue
            picked = []
        clips.append({"start": start, "end": end, "lyrics": clip_lyrics(picked)})
    return clips


def parse_extra_sources(values):
    """--add values -> [(vocal, lyrics or None)]: each --add is VOCAL [LYRICS]."""
    out = []
    for v in values or []:
        if not 1 <= len(v) <= 2:
            raise ValueError(f"--add takes VOCAL [LYRICS], got {len(v)} values")
        out.append((v[0], v[1] if len(v) == 2 else None))
    return out


def build_dataset(run_name, tag, caption, clips, bpm=None, key="", language="en"):
    """Side-Step dataset JSON. Preprocessing prepends custom_tag to each caption.

    A clip's own "bpm"/"key" (an --add source has none) win over the run's.
    """
    return {
        "metadata": {"name": run_name, "custom_tag": tag, "tag_position": "prepend",
                     "genre_ratio": 0},
        "samples": [{
            "filename": Path(c["path"]).name,
            "audio_path": str(c["path"]),
            "caption": caption,
            "lyrics": c["lyrics"],
            "genre": "",
            "bpm": c.get("bpm", bpm),
            "keyscale": c.get("key", key) or "",
            "timesignature": "4",
            "duration": round(c["end"] - c["start"], 3),
            "is_instrumental": False,
            "language": language,
            "labeled": True,
        } for c in clips],
    }


def strip_tag(caption, tag):
    """'Aria, female singer' -> 'female singer' (the tag is added back by Side-Step)."""
    caption = (caption or "").strip()
    prefix = f"{tag},"
    return caption[len(prefix):].strip() if caption.lower().startswith(prefix.lower()) else caption


def parse_epoch_line(line):
    """(epoch, epochs, seconds, loss) from Side-Step's '[OK] Epoch 3/80 in 41.2s, Loss: 0.71'."""
    m = EPOCH_RE.search(line)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))


def list_checkpoints(names):
    """[{epoch, loss, name}] from checkpoint folder names, by epoch."""
    found = []
    for name in names:
        m = CHECKPOINT_RE.match(name)
        if m:
            found.append({"epoch": int(m.group(1)), "loss": float(m.group(2)), "name": name})
    return sorted(found, key=lambda c: c["epoch"])


def pick_checkpoint(checkpoints, epoch=None):
    """The requested epoch, else the lowest finite loss (Aria: epoch 70, lowest too)."""
    if epoch is not None:
        return next((c for c in checkpoints if c["epoch"] == epoch), None)
    finite = [c for c in checkpoints if math.isfinite(c["loss"])]
    return min(finite, key=lambda c: (c["loss"], -c["epoch"])) if finite else None


def updated_profile(profile, tag, caption, adapter_info, today):
    """profile.json after installing an adapter. Keeps every other setting."""
    p = json.loads(json.dumps(profile or {}))
    p.setdefault("name", tag)
    p["version"] = int(p["version"]) + 1 if isinstance(p.get("version"), int) else 1
    p.setdefault("created", today)
    lego = p.setdefault("lego", {})
    lego["caption"] = f"{tag}, {strip_tag(caption, tag)}"
    lego["lora"] = "adapter"
    lego.setdefault("lora_scale", 1.0)
    lego.setdefault("track", "vocals")
    lego.setdefault("inference_steps", 50)
    p.setdefault("post", {"clean": True, "mix_gain_db": 0.0, "boost_db": 5.0})
    p["adapter"] = adapter_info
    return p


def sidestep_cmd(runner, args_map, flags=()):
    cmd = ["uv", "run", "--quiet", "python", str(runner), "--plain", "--yes"]
    for k, v in args_map.items():
        cmd += [k, str(v)]
    return cmd + list(flags)


def inside_rel(path, root):
    """path relative to root ('./x') when inside it, else the absolute path."""
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        return "./" + str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Audio (ffmpeg)
# ---------------------------------------------------------------------------

def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, check=False)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def decode_levels(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1",
                          "-ar", str(ENERGY_RATE), "-f", "s16le", "-"],
                         capture_output=True, check=False).stdout
    samples = array("h")
    samples.frombytes(raw[:len(raw) - len(raw) % 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return frame_levels_db(samples)


def cut_clip(src, start, length, dest):
    proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-ss", f"{start:.3f}",
                           "-t", f"{length:.3f}", "-i", str(src), "-ar", str(SAMPLE_RATE),
                           "-ac", "2", "-c:a", "pcm_s16le", str(dest)],
                          capture_output=True, text=True, check=False)
    return proc.returncode == 0, proc.stderr.strip()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def run_logged(cmd, cwd, log_file, on_line=None):
    """Run a command, write its combined output to log_file, return the exit code."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # else Side-Step's prints sit in a buffer for 20+ min
    with open(log_file, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if on_line:
                on_line(line)
        return proc.wait()


def install_adapter(voice_dir, checkpoint_dir, profile_update):
    """Copy the checkpoint's weights into <voice>/adapter and rewrite profile.json.

    The previous adapter and profile are kept with a timestamp, never overwritten.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    src = Path(checkpoint_dir) / ADAPTER_FILE
    if not src.is_file():
        return None, f"no {ADAPTER_FILE} in {checkpoint_dir}"
    adapter_dir = voice_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    dest = adapter_dir / ADAPTER_FILE
    backups = {}
    if dest.exists():
        old = adapter_dir / "previous" / f"{ts}_{ADAPTER_FILE}"
        old.parent.mkdir(exist_ok=True)
        shutil.move(str(dest), old)
        backups["adapter"] = str(old)
    shutil.copy2(src, dest)

    profile_path = voice_dir / "profile.json"
    profile = {}
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text())
        except ValueError:
            profile = {}
        old = voice_dir / f"profile.{ts}.json"
        shutil.copy2(profile_path, old)
        backups["profile"] = str(old)
    profile_path.write_text(json.dumps(profile_update(profile), indent=2, ensure_ascii=False) + "\n")
    return {"adapter": str(dest), "profile": str(profile_path), "backups": backups}, None


def finish_install(voice_dir, run_dir, manifest, epoch):
    output_dir = Path(manifest["output_dir"])
    ck_root = output_dir / "checkpoints"
    checkpoints = list_checkpoints(os.listdir(ck_root)) if ck_root.is_dir() else []
    chosen = pick_checkpoint(checkpoints, epoch)
    if not chosen:
        fail(f"no checkpoint {'for epoch ' + str(epoch) if epoch else 'with a finite loss'} in {ck_root}",
             checkpoints=checkpoints, run_dir=str(run_dir))
    stage("install", epoch=chosen["epoch"], loss=chosen["loss"])
    tag, caption = manifest["tag"], manifest["caption"]
    info = {"run": manifest["run"], "epoch": chosen["epoch"], "loss": chosen["loss"],
            "clips": manifest.get("clips"), "trained": manifest.get("finished") or manifest.get("started"),
            "source": manifest.get("source"), "checkpoint": str(ck_root / chosen["name"])}
    result, error = install_adapter(
        voice_dir, ck_root / chosen["name"],
        lambda p: updated_profile(p, tag, caption, info, datetime.now().strftime("%Y-%m-%d")))
    if error:
        fail(error, run_dir=str(run_dir))
    manifest["installed"] = {**info, **result}
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    return chosen, checkpoints, result


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Train a named voice adapter (LoKr) from a clean vocal")
    p.add_argument("name", help="Voice name (a-z, 0-9, -), e.g. aria")
    p.add_argument("vocal", nargs="?", help="Clean, isolated vocal (FLAC/WAV), ideally 1-4 min")
    p.add_argument("--lyrics", help="Timed lyrics: listen report JSON, segment JSON or LRC")
    p.add_argument("--add", nargs="+", action="append", metavar="VOCAL [LYRICS]",
                   help="Another clean take of the same voice, with optional timed lyrics "
                        "(repeatable). Its clips get no bpm/key")
    p.add_argument("--caption", help="Voice description without the trigger word "
                                     "(default: the voice profile's lego.caption)")
    p.add_argument("--tag", help="Trigger word (default: the name capitalized, e.g. Aria)")
    p.add_argument("--bpm", type=int, help="BPM of the source song (optional)")
    p.add_argument("--key", default="", help="Key of the source song, e.g. 'C minor' (optional)")
    p.add_argument("--language", default=(cfg.get("defaults") or {}).get("language", "en"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--save-every", type=int, default=10, help="Checkpoint every N epochs")
    p.add_argument("--epoch", type=int, help="Checkpoint to install (default: lowest loss)")
    p.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    p.add_argument("--hop-sec", type=float, default=HOP_SEC)
    p.add_argument("--dry-run", action="store_true", help="Stop after writing clips and dataset.json")
    p.add_argument("--no-install", action="store_true",
                   help="Train only; pick and install a checkpoint later with --install RUN --epoch N")
    p.add_argument("--install", metavar="RUN",
                   help="Skip training: install a checkpoint from an earlier run of this voice")
    args = p.parse_args()

    name = args.name.strip().lower()
    if not VOICE_NAME_RE.match(name):
        fail(f"invalid voice name {args.name!r} (use a-z, 0-9 and -)")
    tag = args.tag or default_tag(name)
    voice_dir = Path(os.path.expanduser(cfg.get("voices_dir") or DEFAULT_VOICES_DIR)) / name

    if args.install:
        run_dir = voice_dir / "training" / Path(args.install).name
        try:
            manifest = json.loads((run_dir / "run.json").read_text())
        except (OSError, ValueError) as e:
            fail(f"cannot read {run_dir / 'run.json'}: {e}")
        chosen, checkpoints, result = finish_install(voice_dir, run_dir, manifest, args.epoch)
        print(json.dumps({"success": True, "voice": name, "run_dir": str(run_dir),
                          "checkpoint": chosen, "checkpoints": checkpoints, **result}, indent=2))
        return

    if not args.vocal:
        fail("a clean vocal file is required (or --install RUN)")
    src = Path(args.vocal).expanduser().resolve()
    if not src.is_file():
        fail(f"vocal not found: {src}")
    ace_dir = os.path.expanduser(cfg.get("ace_step_dir") or "")
    if not args.dry_run and not os.path.isdir(ace_dir):
        fail(f"ace_step_dir not found: {ace_dir!r} (set it in {CONFIG_PATH})")
    for tool in ("ffmpeg", "ffprobe") + (() if args.dry_run else ("uv",)):
        if not shutil.which(tool):
            fail(f"{tool} not found on PATH")

    profile = {}
    profile_path = voice_dir / "profile.json"
    if profile_path.is_file():
        try:
            profile = json.loads(profile_path.read_text())
        except ValueError:
            profile = {}
    caption = strip_tag(args.caption or (profile.get("lego") or {}).get("caption"), tag)
    if not caption:
        fail("no caption: pass --caption 'female singer, breathy airy voice, ...' "
             "(describe the voice; the trigger word is added for you)")

    try:
        extras = parse_extra_sources(args.add)
    except ValueError as e:
        fail(str(e))
    sources = [(src, args.lyrics)]
    for vocal, lyr in extras:
        path = Path(vocal).expanduser().resolve()
        if not path.is_file():
            fail(f"vocal not found: {path}")
        sources.append((path, lyr))

    timed = []
    for path, lyr in sources:
        lines = None
        if lyr:
            try:
                lines = parse_timed_lyrics(Path(lyr).expanduser().read_text())
            except OSError as e:
                fail(f"cannot read lyrics: {e}")
            if lines is None:
                log(f"Lyrics for {path.name} have no timing: [Verse] tags only, chosen by energy")
        timed.append(lines)

    run = f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = voice_dir / "training" / run
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=False)
    log(f"Run folder: {run_dir}")
    t0 = time.time()

    # 1. Clips
    stage("clips")
    clips, n_windows, duration, extra_info = [], 0, 0.0, []
    for k, ((path, _), lines) in enumerate(zip(sources, timed)):
        dur = probe_duration(path)
        if not dur:
            fail(f"cannot read the duration of {path}", run_dir=str(run_dir))
        windows = clip_windows(dur, args.clip_sec, args.hop_sec)
        levels = decode_levels(path) if lines is None else None
        picked = select_clips(windows, lines=lines, levels=levels)
        if not picked:
            fail(f"no window of {path.name} holds sung lines; is the vocal silent or the "
                 "lyrics timing off?", run_dir=str(run_dir), windows=len(windows))
        for clip in picked:
            clip["src"] = path
            if k:  # an --add take: its song's bpm/key are not the run's
                clip["bpm"], clip["key"] = None, ""
        clips += picked
        n_windows += len(windows)
        if k:
            extra_info.append({"source": str(path), "source_sec": round(dur, 2), "clips": len(picked),
                               "timed_lyrics": lines is not None})
        else:
            duration = dur
        log(f"{path.name}: {len(picked)} clips of {len(windows)} windows ({dur:.1f} s)")
    for i, clip in enumerate(clips, 1):
        clip["path"] = clips_dir / f"{name}_{i:02d}.wav"
        ok, err = cut_clip(clip["src"], clip["start"], clip["end"] - clip["start"], clip["path"])
        if not ok:
            fail(f"ffmpeg could not cut clip {i}: {err}", run_dir=str(run_dir))
    lines = timed[0]
    log(f"{len(clips)} clips in all")

    # 2. Dataset
    stage("dataset", clips=len(clips))
    dataset = build_dataset(run, tag, caption, clips, args.bpm, args.key, args.language)
    dataset_path = run_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))

    ace = Path(ace_dir).resolve() if ace_dir else None
    tensors_dir = ace / "datasets" / name / run if ace else None
    output_dir = ace / "lora_output" / name / run if ace else None
    ckpt_cfg = cfg.get("checkpoint_dir")
    checkpoint_arg = inside_rel(os.path.expanduser(ckpt_cfg), ace) if ckpt_cfg and ace else "./checkpoints"
    device = "mps" if sys.platform == "darwin" and platform.machine() == "arm64" else "auto"
    common = {"--checkpoint-dir": checkpoint_arg, "--model-variant": "base", "--device": device}
    pre_cmd = sidestep_cmd(RUNNER, {**common, "--dataset-json": dataset_path,
                                    "--tensor-output": inside_rel(tensors_dir, ace) if ace else "",
                                    "--max-duration": PREPROCESS_MAX_SEC, "--precision": "fp32"},
                           ("--preprocess",))
    train_cmd = sidestep_cmd(RUNNER, {**common, **TRAIN_ARGS,
                                      "--dataset-dir": inside_rel(tensors_dir, ace) if ace else "",
                                      "--output-dir": inside_rel(output_dir, ace) if ace else "",
                                      "--epochs": args.epochs, "--save-every": args.save_every},
                             TRAIN_FLAGS)
    manifest = {
        "run": run, "voice": name, "tag": tag, "caption": caption, "source": str(src),
        "source_sec": round(duration, 2), "extra_sources": extra_info, "clips": len(clips),
        "clip_sec": args.clip_sec,
        "hop_sec": args.hop_sec, "timed_lyrics": lines is not None, "bpm": args.bpm, "key": args.key,
        "language": args.language, "epochs": args.epochs, "save_every": args.save_every,
        "ace_step_dir": str(ace) if ace else None, "tensors_dir": str(tensors_dir) if ace else None,
        "output_dir": str(output_dir) if ace else None,
        "preprocess_cmd": pre_cmd, "train_cmd": train_cmd, "started": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = run_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    summary = {"success": True, "voice": name, "tag": tag, "run": run, "run_dir": str(run_dir),
               "dataset": str(dataset_path), "clips": len(clips), "windows": n_windows,
               "timed_lyrics": lines is not None}

    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True, "preprocess_cmd": pre_cmd, "train_cmd": train_cmd,
                          "cwd": str(ace) if ace else None,
                          "elapsed_sec": round(time.time() - t0, 1)}, indent=2))
        return

    # 3. Preprocess
    stage("preprocess")
    rc = run_logged(pre_cmd, ace, run_dir / "preprocess.log")
    pt_files = list(tensors_dir.glob("*.pt")) if tensors_dir.is_dir() else []
    if rc != 0 or len(pt_files) < len(clips):
        fail(f"preprocessing failed (exit {rc}, {len(pt_files)}/{len(clips)} tensors)",
             log=str(run_dir / "preprocess.log"), run_dir=str(run_dir))

    # 4. Train
    stage("train", epochs=args.epochs, clips=len(clips))
    last = {}

    def on_line(line):
        parsed = parse_epoch_line(line)
        if parsed:
            ep, total, sec, loss = parsed
            last.update(epoch=ep, epochs=total, loss=loss)
            print(json.dumps({"event": "epoch", "epoch": ep, "epochs": total, "sec": sec, "loss": loss,
                              "eta_sec": round((total - ep) * sec)}), file=sys.stderr, flush=True)

    rc = run_logged(train_cmd, ace, run_dir / "train.log", on_line)
    if rc != 0 or last.get("epoch") != args.epochs:
        fail(f"training stopped at epoch {last.get('epoch', 0)}/{args.epochs} (exit {rc})",
             log=str(run_dir / "train.log"), run_dir=str(run_dir))
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.no_install:
        ck_root = output_dir / "checkpoints"
        checkpoints = list_checkpoints(os.listdir(ck_root)) if ck_root.is_dir() else []
        stage("done")
        print(json.dumps({**summary, "installed": False, "checkpoints": checkpoints,
                          "install_cmd": f"music_train_voice.sh {name} --install {run} --epoch N",
                          "elapsed_sec": round(time.time() - t0, 1)}, indent=2))
        return

    # 5. Install
    chosen, checkpoints, result = finish_install(voice_dir, run_dir, manifest, args.epoch)
    stage("done")
    print(json.dumps({**summary, "checkpoint": chosen, "checkpoints": checkpoints, **result,
                      "elapsed_sec": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
