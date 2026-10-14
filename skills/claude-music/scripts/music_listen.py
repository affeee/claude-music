#!/usr/bin/env python3
"""Listening report for a generated vocal (or any audio) — claude-music skill.

Turns audio into things a text model can read: what is sung and when, how
high the voice sits and whether it is in key, how loud it is against the
backing, how its phrases land on the beat, and a spectrogram picture.

Usage:
  music_listen.py vocals.flac --source original.mp3 --lyrics lyrics.txt \\
      --key "C minor" --bpm 129 --start 29.8 --mix-gain -3 --language en

One JSON object on stdout; progress on stderr. Run it through
music_listen.sh, which supplies librosa, matplotlib and a Whisper backend.
It measures; it does not judge taste.
"""

import argparse
import contextlib
import difflib
import json
import math
import os
import re
import sys
import tempfile

SR = 16000
HOP = 512
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
WHISPER_MLX = "mlx-community/whisper-large-v3-turbo"
WHISPER_FASTER = "large-v3-turbo"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Pure helpers (stdlib only, unit-tested without audio libraries)
# ---------------------------------------------------------------------------

def parse_key(key):
    """'C minor' / 'F# major' / 'Am' -> set of pitch classes, or None."""
    m = re.match(r"^\s*([A-Ga-g])([#b]?)\s*(.*)$", key or "")
    if not m:
        return None
    name = (m.group(1).upper() + m.group(2)).upper()
    name = FLATS.get(name, name)
    if name not in NOTE_NAMES:
        return None
    mode = m.group(3).strip().lower()
    scale = SCALES["minor" if mode in ("m", "min", "minor") else "major"]
    tonic = NOTE_NAMES.index(name)
    return {(tonic + step) % 12 for step in scale}


def hz_to_midi(hz):
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def note_name(hz):
    midi = round(hz_to_midi(hz))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def voice_register(median_hz):
    """Rough register from the median sung pitch. A label, not a diagnosis."""
    for limit, label in ((120, "bass"), (175, "baritone"), (260, "tenor"),
                         (400, "alto"), (float("inf"), "soprano")):
        if median_hz < limit:
            return label
    return "soprano"


def pitch_summary(f0_hz, scale=None):
    """Summarise voiced f0 values (Hz). `scale` is a set of pitch classes."""
    voiced = sorted(h for h in f0_hz if h and h > 0 and not math.isnan(h))
    if not voiced:
        return None

    def pct(p):
        return voiced[min(len(voiced) - 1, int(p * len(voiced)))]

    median = pct(0.5)
    out = {
        "median_hz": round(median, 1),
        "median_note": note_name(median),
        "range_hz": [round(pct(0.1), 1), round(pct(0.9), 1)],
        "range_notes": [note_name(pct(0.1)), note_name(pct(0.9))],
        "register": voice_register(median),
        "typical_range": ("male" if median < 200 else
                          "high male or low female" if median < 300 else "female"),
    }
    cents = [abs((hz_to_midi(h) - round(hz_to_midi(h))) * 100) for h in voiced]
    out["mean_cents_off_semitone"] = round(sum(cents) / len(cents), 1)
    if scale:
        hits = sum(1 for h in voiced if round(hz_to_midi(h)) % 12 in scale)
        out["in_key_ratio"] = round(hits / len(voiced), 3)
    return out


def active_regions(flags, frame_sec, offset=0.0, bridge=0.4, min_len=0.2):
    """Runs of True in `flags` -> [(start, end)] seconds.

    Pauses shorter than `bridge` are sung through (breaths, consonants) and
    blips shorter than `min_len` are dropped.
    """
    runs, start = [], None
    for i, on in enumerate(list(flags) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start * frame_sec, i * frame_sec])
            start = None
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < bridge:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(round(a + offset, 2), round(b + offset, 2))
            for a, b in merged if b - a >= min_len]


def find_gaps(regions, min_gap=4.0):
    """Silences of at least `min_gap` seconds between sung regions."""
    return [{"start": a_end, "end": b_start, "sec": round(b_start - a_end, 1)}
            for (_, a_end), (b_start, _) in zip(regions, regions[1:], strict=False)
            if b_start - a_end >= min_gap]


def normalize_words(text):
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", (text or "").lower())


def lyric_lines(lyrics):
    """Sung lines of a lyric sheet: no [Section] tags, no blank lines."""
    lines = []
    for raw in (lyrics or "").splitlines():
        line = re.sub(r"\[[^\]]*\]", "", raw).strip()
        if normalize_words(line):
            lines.append(line)
    return lines


def match_lyrics(lines, words, found_at=0.5, lookahead=40, close_enough=0.15):
    """Find each expected line in the transcript, in order.

    `words` is [{"word", "start", "end"}]. Monotonic, and each line is looked
    for only in the next `lookahead` words and takes the earliest window that
    is nearly as good as the best one. Otherwise a slightly misheard chorus
    ("Ooh, you don't scare me") loses to a clean repeat a minute later and
    every line in between is reported missing. Returns per-line results.
    """
    heard = [normalize_words(w["word"]) for w in words]
    heard = [(h[0] if h else "") for h in heard]
    results, cursor = [], 0
    for line in lines:
        want = normalize_words(line)
        candidates = []
        for i in range(cursor, min(len(heard), cursor + lookahead)):
            for size in {max(1, len(want) - 2), len(want), len(want) + 2}:
                window = heard[i:i + size]
                if window:
                    ratio = difflib.SequenceMatcher(
                        None, " ".join(want), " ".join(window)).ratio()
                    candidates.append((ratio, i, i + len(window)))
        top = max((c[0] for c in candidates), default=0.0)
        near = [c for c in candidates if c[0] >= max(found_at, top - close_enough)]
        first = min((c[1] for c in near), default=None)
        ratio, i, j = max((c for c in near if c[1] == first), default=(top, None, None))
        found = i is not None
        entry = {"line": line, "match": round(ratio, 2), "found": found}
        if found:
            entry["at"] = round(words[i]["start"], 1)
            entry["heard"] = " ".join(w["word"].strip() for w in words[i:j])
            cursor = j
        results.append(entry)
    return results


def beat_offsets_ms(onsets, beats, subdivision=2):
    """Signed distance (ms) from each onset to the nearest beat subdivision."""
    if len(beats) < 2:
        return []
    grid = []
    for a, b in zip(beats, beats[1:], strict=False):
        grid.extend(a + (b - a) * k / subdivision for k in range(subdivision))
    grid.append(beats[-1])
    return [round((t - min(grid, key=lambda g: abs(g - t))) * 1000)
            for t in onsets if grid[0] <= t <= grid[-1]]


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else None


# ---------------------------------------------------------------------------
# Audio analysis (needs numpy / librosa)
# ---------------------------------------------------------------------------

def load_region(path, start, end):
    import librosa
    duration = None if end is None or end <= 0 else max(0.0, end - start)
    y, _ = librosa.load(path, sr=SR, mono=True, offset=max(0.0, start), duration=duration)
    return y


def analyze_activity(y, start, threshold_db=-40.0):
    import librosa
    import numpy as np
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    flags = db > threshold_db
    regions = active_regions(flags.tolist(), HOP / SR, offset=start)
    sung = round(sum(b - a for a, b in regions), 1)
    span = len(y) / SR
    return {
        "first_onset": regions[0][0] if regions else None,
        "last_offset": regions[-1][1] if regions else None,
        "sung_sec": sung,
        "coverage": round(sung / span, 3) if span else 0.0,
        "phrases": len(regions),
        "gaps": find_gaps(regions),
    }, regions, flags


def analyze_pitch(y, key):
    import librosa
    import numpy as np
    f0, voiced, _ = librosa.pyin(y, fmin=65.0, fmax=880.0, sr=SR,
                                 frame_length=1024, hop_length=HOP)
    values = [float(v) for v, on in zip(f0, voiced, strict=False) if on and not np.isnan(v)]
    return pitch_summary(values, parse_key(key))


def analyze_balance(y, source, flags, start, mix_gain, window=10.0):
    """Vocal level against the backing, window by window (RMS dB)."""
    import numpy as np
    n = min(len(y), len(source))
    per_window = int(window * SR)
    rows = []
    for a in range(0, n - per_window // 2, per_window):
        b = min(n, a + per_window)
        active = np.repeat(flags[a // HOP:b // HOP + 1], HOP)[:b - a]
        if active.mean() < 0.3:
            continue
        voc = y[a:b][active]
        back = source[a:b][active]
        if not len(voc) or not np.any(back):
            continue
        ratio = (20 * np.log10(max(float(np.sqrt((voc ** 2).mean())), 1e-9)) + mix_gain
                 - 20 * np.log10(max(float(np.sqrt((back ** 2).mean())), 1e-9)))
        rows.append({"at": round(start + a / SR), "vocal_vs_backing_db": round(ratio, 1)})
    if not rows:
        return None
    values = [r["vocal_vs_backing_db"] for r in rows]
    return {
        "median_db": median(values),
        "min_db": min(values),
        "max_db": max(values),
        "spread_db": round(max(values) - min(values), 1),
        "windows": rows,
        "note": "RMS of the layer (with --mix-gain) minus RMS of the source while "
                "the voice is active. Positive = voice louder than the backing.",
    }


def analyze_timing(source_path, regions, bpm):
    import librosa
    y, sr = librosa.load(source_path, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm or 120.0, units="time")
    offsets = beat_offsets_ms([a for a, _ in regions], [float(b) for b in beats])
    if not offsets:
        return None
    absolute = [abs(o) for o in offsets]
    return {
        "detected_bpm": round(float(tempo if not hasattr(tempo, "__len__") else tempo[0]), 1),
        "phrase_starts": len(offsets),
        "median_offset_ms": median(offsets),
        "median_abs_offset_ms": median(absolute),
        "within_60ms": round(sum(1 for o in absolute if o <= 60) / len(absolute), 2),
        "note": "Distance from each phrase start to the nearest eighth note. "
                "Rough: sung phrases often start between grid points on purpose.",
    }


def transcribe(y, start, language):
    """Whisper on the region. Returns (result dict, error string)."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        sf.write(path, y, SR)
        try:
            import mlx_whisper
            raw = mlx_whisper.transcribe(
                path, path_or_hf_repo=WHISPER_MLX, word_timestamps=True,
                language=language, condition_on_previous_text=False, verbose=None)
            segments = [{"start": s["start"], "end": s["end"], "text": s["text"],
                         "words": s.get("words", [])} for s in raw["segments"]]
            detected, backend = raw.get("language"), "mlx-whisper"
        except ImportError:
            from faster_whisper import WhisperModel
            model = WhisperModel(WHISPER_FASTER)
            # The 16 kHz array, not the path: faster-whisper 1.2.1 opens files
            # with av.open(metadata_errors=...), which PyAV 14+ rejects
            found, info = model.transcribe(y.astype("float32"), word_timestamps=True,
                                           language=language,
                                           condition_on_previous_text=False)
            segments = [{"start": s.start, "end": s.end, "text": s.text,
                         "words": [{"word": w.word, "start": w.start, "end": w.end}
                                   for w in (s.words or [])]} for s in found]
            detected, backend = info.language, "faster-whisper"
    except ImportError:
        return None, "no Whisper backend installed (run through music_listen.sh)"
    except Exception as e:
        return None, f"transcription failed: {e}"
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)

    words = [{"word": w["word"], "start": round(w["start"] + start, 2),
              "end": round(w["end"] + start, 2)}
             for s in segments for w in s["words"]]
    return {
        "backend": backend,
        "language": detected,
        "text": " ".join(s["text"].strip() for s in segments).strip(),
        "segments": [{"start": round(s["start"] + start, 1), "end": round(s["end"] + start, 1),
                      "text": s["text"].strip()} for s in segments],
        "words": words,
    }, None


def draw_spectrogram(y, source, start, out_path):
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tracks = [("layer", y)] + ([("source", source)] if source is not None else [])
    fig, axes = plt.subplots(len(tracks), 1, figsize=(16, 4 * len(tracks)), squeeze=False)
    for ax, (label, audio) in zip(axes[:, 0], tracks, strict=False):
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=2048, hop_length=HOP,
                                             n_mels=128, fmax=8000)
        librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=SR, hop_length=HOP,
                                 x_axis="time", y_axis="mel", fmax=8000, ax=ax)
        ax.set_title(f"{label} (time axis starts at {start:.1f}s of the file)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_lyrics(value):
    if value and os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return f.read()
    return value or ""


def build_parser():
    p = argparse.ArgumentParser(description="Listening report for a generated vocal")
    p.add_argument("audio", help="Audio to listen to (ideally a vocal layer from lego)")
    p.add_argument("--source", default=None,
                   help="The track the layer was generated over (enables balance + timing)")
    p.add_argument("--lyrics", default=None, help="Expected lyrics: text or a file path")
    p.add_argument("--language", default=None, help="Sung language code (en, sv, ...)")
    p.add_argument("--key", default=None, help="Musical key, e.g. 'C minor'")
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--start", type=float, default=0.0,
                   help="Analyse from here (the lego --start; before it the layer is the source)")
    p.add_argument("--end", type=float, default=-1.0)
    p.add_argument("--mix-gain", type=float, default=0.0,
                   help="Layer gain used in the mix (dB), for the balance figures")
    p.add_argument("--no-transcribe", action="store_true", help="Skip Whisper")
    p.add_argument("--spectrogram", default=None,
                   help="Write a spectrogram PNG here ('auto' = next to the audio)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.audio):
        print(json.dumps({"success": False, "error": f"Audio not found: {args.audio}"}))
        sys.exit(1)
    if args.source and not os.path.isfile(args.source):
        print(json.dumps({"success": False, "error": f"Source not found: {args.source}"}))
        sys.exit(1)

    report = {"success": True, "audio": args.audio,
              "region": {"start": args.start, "end": args.end}, "warnings": []}

    log("Loading audio...")
    y = load_region(args.audio, args.start, args.end)
    source = load_region(args.source, args.start, args.end) if args.source else None

    report["activity"], regions, flags = analyze_activity(y, args.start)
    if not regions:
        report["warnings"].append("No audible signal in the analysed region")

    log("Tracking pitch...")
    report["pitch"] = analyze_pitch(y, args.key)

    if source is not None:
        report["balance"] = analyze_balance(y, source, flags, args.start, args.mix_gain)
        log("Finding beats...")
        report["timing"] = analyze_timing(args.source, regions, args.bpm)

    if not args.no_transcribe:
        log("Transcribing (the first run downloads the Whisper model)...")
        transcript, error = transcribe(y, args.start, args.language)
        if error:
            report["warnings"].append(error)
        else:
            words = transcript.pop("words")
            report["transcript"] = transcript
            lines = lyric_lines(read_lyrics(args.lyrics))
            if lines:
                matched = match_lyrics(lines, words)
                found = sum(1 for m in matched if m["found"])
                report["lyrics"] = {
                    "lines_expected": len(lines),
                    "lines_found": found,
                    "lines_missing": [m["line"] for m in matched if not m["found"]],
                    "lines": matched,
                }
            if args.language and transcript["language"] not in (None, args.language):
                report["warnings"].append(
                    f"Whisper heard '{transcript['language']}', expected '{args.language}'")

    if args.spectrogram:
        path = args.spectrogram
        if path == "auto":
            path = os.path.splitext(args.audio)[0] + "_listen.png"
        log("Drawing spectrogram...")
        report["spectrogram"] = draw_spectrogram(y, source, args.start, path)

    for gap in report["activity"]["gaps"]:
        report["warnings"].append(
            f"Voice is silent for {gap['sec']}s from {gap['start']}s to {gap['end']}s")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
