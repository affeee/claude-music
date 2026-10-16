#!/usr/bin/env python3
"""Vocal flow: sing over an existing instrumental, end to end — claude-music skill.

  1. Structure analysis  (music_structure.sh): bpm, key, beat entry, drop
  2. Vocal layer         (music_engine.sh lego): placed from the beat entry
  3. Clean               (--clean): Demucs + declick, no backing bleed
  4. Mix                 (--mix): over the untouched original, lifted at the drop
  5. Listening report    (music_listen.sh) on the clean stem

Every run gets its own folder with the lyrics used, each step's JSON and a
run.json manifest holding all settings, so a result can be reproduced.

Usage:
  music_vocals.py song.mp3 --lyrics lyrics.txt --voice aria
  music_vocals.py song.mp3 --lyrics lyrics.txt --caption "warm male baritone" --seed 7

One JSON object on stdout. Progress on stderr: NDJSON {"event": "stage", ...}
lines from this script plus the engine's own --progress events.
Stdlib only; each step runs in the environment it needs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "config.json"
DEFAULT_VOICES_DIR = "~/Music/claude-music-voices"
STAGES = ("analyze", "generate", "listen", "done")


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

def slug(text, max_len=40):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return "-".join(words)[:max_len].strip("-") or "song"


def with_intro_line(lyrics, intro_line, start):
    """Prepend a throwaway [Intro] line when the vocal starts mid-song.

    With --start the model tends to drop the first lyric line; a short
    throwaway line ("Ooh, ooh") absorbs that instead.
    """
    if not intro_line or not start or start <= 0:
        return lyrics
    first = next((ln.strip() for ln in lyrics.splitlines() if ln.strip()), "")
    if first.lower().startswith("[intro"):
        return lyrics
    return f"[Intro]\n{intro_line}\n\n{lyrics.lstrip()}"


def resolve_time(value, detected):
    """'auto' -> detected value, 'none' -> None, else a float in seconds."""
    if value is None or value == "auto":
        return detected
    if value == "none":
        return None
    return float(value)


def last_json(text):
    """The last JSON object printed on stdout (scripts print exactly one)."""
    text = (text or "").strip()
    start = text.rfind("\n{")
    for candidate in ([text[start + 1:]] if start >= 0 else []) + [text]:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def run_json(cmd, out_file, stream_stderr=False):
    """Run a script that prints one JSON object; save it to out_file."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True,
                          stderr=None if stream_stderr else subprocess.PIPE)
    result = last_json(proc.stdout)
    if result is None:
        tail = (proc.stderr or "")[-800:] if not stream_stderr else ""
        return {"success": False, "error": f"no JSON from {Path(cmd[1]).name}", "stderr": tail}
    out_file.write_text(json.dumps(result, indent=2))
    return result


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(SCRIPT_DIR), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def voice_profile(name):
    if not name:
        return {}
    base = os.path.expanduser(load_config().get("voices_dir") or DEFAULT_VOICES_DIR)
    path = Path(base) / name.lower() / "profile.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        fail(f"voice profile not found or unreadable: {path}")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Sing over an existing instrumental, end to end")
    p.add_argument("src", help="The instrumental (left untouched)")
    p.add_argument("--lyrics", required=True, help="Lyrics file with [Verse]/[Chorus] tags")
    p.add_argument("--voice", help="Voice profile (e.g. aria) for caption, seed and post settings")
    p.add_argument("--caption", help="Vocal description (overrides the voice profile)")
    p.add_argument("--seed", type=int, help="Seed (overrides the voice profile)")
    p.add_argument("--language", default=(cfg.get("defaults") or {}).get("language", "en"))
    p.add_argument("--start", default="auto",
                   help="Where the vocal begins: auto (beat entry), or seconds")
    p.add_argument("--boost-at", default="auto",
                   help="Where the vocal is lifted in the mix: auto (the drop), none, or seconds")
    p.add_argument("--boost", type=float, help="Lift in dB at --boost-at (default: voice profile, else 5)")
    p.add_argument("--mix-gain", type=float, help="Base vocal level in dB (default: voice profile, else 0)")
    p.add_argument("--steps", type=int, help="Diffusion steps (default: voice profile, else 50)")
    p.add_argument("--quality", default="standard")
    p.add_argument("--no-intro-line", action="store_true",
                   help="Do not prepend the voice profile's throwaway [Intro] line")
    p.add_argument("--no-listen", action="store_true", help="Skip the listening report")
    p.add_argument("--output-dir", default=cfg.get("output_dir") or "~/Music/claude-music-output")
    args = p.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    if not os.path.isfile(src):
        fail(f"instrumental not found: {src}")
    try:
        lyrics_in = Path(args.lyrics).expanduser().read_text()
    except OSError as e:
        fail(f"cannot read lyrics: {e}")

    profile = voice_profile(args.voice)
    post = profile.get("post") or {}
    lyrics_cfg = profile.get("lyrics") or {}

    run_dir = Path(os.path.expanduser(args.output_dir)) / (
        f"{slug(Path(src).stem)}-vocals_{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=False)
    log(f"Run folder: {run_dir}")
    t0 = time.time()

    # 1. Structure
    stage("analyze")
    structure = run_json(["bash", str(SCRIPT_DIR / "music_structure.sh"), src],
                         run_dir / "structure.json")
    if not structure.get("success"):
        fail("structure analysis failed", detail=structure, run_dir=str(run_dir))
    start = resolve_time(args.start, structure.get("first_beat_sec"))
    boost_at = resolve_time(args.boost_at, structure.get("drop_sec"))
    boost = args.boost if args.boost is not None else post.get("boost_db", 5.0)
    mix_gain = args.mix_gain if args.mix_gain is not None else post.get("mix_gain_db", 0.0)
    if boost_at is not None and start and boost_at <= start:
        boost_at = None  # a lift before the vocal starts would just be a louder vocal
    bpm = int(round(structure["bpm"]))
    key = structure.get("key") or ""

    lyrics = lyrics_in if args.no_intro_line else with_intro_line(
        lyrics_in, lyrics_cfg.get("intro_line"), start)
    lyrics_file = run_dir / "lyrics.txt"
    lyrics_file.write_text(lyrics)

    # 2-4. Generate, clean, mix (one engine run)
    stage("generate", start=start, boost_at=boost_at, bpm=bpm, key=key)
    engine = ["bash", str(SCRIPT_DIR / "music_engine.sh"), "--quality", args.quality,
              "--progress", "--output-dir", str(run_dir)]
    if args.seed is not None:
        engine += ["--seed", str(args.seed)]
    engine += ["lego", "--src-audio", src, "--track", "vocals", "--clean", "--mix",
               "--lyrics", lyrics, "--language", args.language,
               "--bpm", str(bpm), "--mix-gain", str(mix_gain)]
    if key:
        engine += ["--key", key]
    if start:
        engine += ["--start", f"{start:.2f}"]
    if boost_at is not None and boost:
        engine += ["--mix-boost-at", f"{boost_at:.2f}", "--mix-boost", str(boost)]
    if args.voice:
        engine += ["--voice", args.voice]
    if args.caption:
        engine += ["--caption", args.caption]
    if args.steps:
        engine += ["--inference-steps", str(args.steps)]
    result = run_json(engine, run_dir / "engine.json", stream_stderr=True)
    if not result.get("success") or not result.get("outputs"):
        fail("vocal generation failed", detail=result, run_dir=str(run_dir))
    out = result["outputs"][0]
    vocal = out.get("clean_path") or out["path"]

    # 5. Listening report
    listen = None
    if not args.no_listen:
        stage("listen")
        cmd = ["bash", str(SCRIPT_DIR / "music_listen.sh"), vocal, "--source", src,
               "--lyrics", str(lyrics_file), "--language", args.language,
               "--bpm", str(bpm), "--mix-gain", str(mix_gain),
               "--spectrogram", str(run_dir / "listen.png")]
        if key:
            cmd += ["--key", key]
        if start:
            cmd += ["--start", f"{start:.2f}"]
        listen = run_json(cmd, run_dir / "listen.json")

    summary = {
        "success": True,
        "run_dir": str(run_dir),
        "source": src,
        "voice": args.voice,
        "files": {"layer": out["path"], "clean": out.get("clean_path"),
                  "mix": out.get("mix_path"), "lyrics": str(lyrics_file)},
        "placement": {"start": start, "boost_at": boost_at, "boost_db": boost,
                      "mix_gain_db": mix_gain, "bpm": bpm, "key": key},
        "engine_settings": result.get("settings"),
        "warnings": [w for w, bad in (
            ("cleaning failed, the mix uses the raw layer", not out.get("clean_path")),
            ("mix failed", not out.get("mix_path"))) if bad],
        "minutes": round((time.time() - t0) / 60, 1),
    }
    if listen and listen.get("success", True) and "lyrics" in listen:
        summary["listen"] = {
            "lines_found": listen["lyrics"].get("lines_found"),
            "lines_expected": listen["lyrics"].get("lines_expected"),
            "median_note": (listen.get("pitch") or {}).get("median_note"),
            "in_key_ratio": (listen.get("pitch") or {}).get("in_key_ratio"),
            "gaps": (listen.get("activity") or {}).get("gaps"),
            "report": str(run_dir / "listen.json"),
        }
    (run_dir / "run.json").write_text(json.dumps({
        **summary, "args": vars(args), "repo_commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds")}, indent=2))
    stage("done", run_dir=str(run_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
