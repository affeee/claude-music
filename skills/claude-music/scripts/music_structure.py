#!/usr/bin/env python3
"""Structure analysis of an instrumental — claude-music skill.

Finds what a vocal layer needs to be placed: tempo, key, where the beat
comes in, where the drop (the big, sustained rise in loudness) lands, and
the bar grid. The vocal flow uses the beat entry as lego --start and the
drop as the point where the vocal is lifted in the mix.

Usage:
  music_structure.py song.mp3

One JSON object on stdout; progress on stderr. Run it through
music_structure.sh, which supplies librosa. Module-level imports stay
stdlib-only so the pure helpers are testable in CI.
"""

import argparse
import json
import math
import sys

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler key profiles
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
RMS_HOP_SEC = 0.5          # loudness curve resolution
DROP_WINDOW_SEC = 4.0      # compare this much audio before and after a candidate
DROP_MIN_RISE_DB = 4.0     # smaller rises are build-ups, not drops
DROP_PLATEAU_MARGIN_DB = 3.0  # a drop lands within this of the loud plateau


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Pure helpers (stdlib only, unit-tested without audio libraries)
# ---------------------------------------------------------------------------

def _corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def estimate_key(chroma_mean):
    """12 mean chroma values (C..B) -> ("C minor", confidence 0-1)."""
    best = (-2.0, "")
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated = profile[-tonic:] + profile[:-tonic] if tonic else profile
            score = _corr(rotated, chroma_mean)
            if score > best[0]:
                best = (score, f"{NOTE_NAMES[tonic]} {mode}")
    return best[1], round(max(0.0, best[0]), 3)


def find_drop(rms_db, hop_sec=RMS_HOP_SEC, window_sec=DROP_WINDOW_SEC,
              min_rise_db=DROP_MIN_RISE_DB, plateau_margin_db=DROP_PLATEAU_MARGIN_DB):
    """Time (s) of the largest sustained loudness rise into the loud part, or None.

    For each candidate point, compares the mean level of the window after it
    with the window before it. Only rises that land on the track's loud
    plateau count (after-window within plateau_margin_db of the 90th
    percentile level); that keeps a beat entering a quiet intro from being
    taken for the drop. The drop is the valid point with the largest rise,
    provided it is at least min_rise_db.
    """
    w = max(1, int(round(window_sec / hop_sec)))
    if len(rms_db) < 2 * w:
        return None
    ranked = sorted(rms_db)
    plateau = ranked[min(len(ranked) - 1, int(0.9 * len(ranked)))]
    best_rise, best_i = 0.0, None
    for i in range(w, len(rms_db) - w + 1):
        before = sum(rms_db[i - w:i]) / w
        after = sum(rms_db[i:i + w]) / w
        rise = after - before
        if after >= plateau - plateau_margin_db and rise > best_rise:
            best_rise, best_i = rise, i
    if best_i is None or best_rise < min_rise_db:
        return None
    return round(best_i * hop_sec, 2)


def snap_to_beat(t, first_beat, bpm):
    """Snap a time to the nearest beat of a grid starting at first_beat."""
    if t is None or not bpm:
        return t
    beat = 60.0 / bpm
    return round(first_beat + round((t - first_beat) / beat) * beat, 2)


def bar_seconds(bpm, beats_per_bar=4):
    return 60.0 * beats_per_bar / bpm


# ---------------------------------------------------------------------------
# Analysis (needs librosa, numpy)
# ---------------------------------------------------------------------------

def analyze(path):
    import librosa
    import numpy as np

    log(f"Loading {path} ...")
    y, sr = librosa.load(path, sr=22050, mono=True)
    duration = len(y) / sr

    log("Tempo and beats ...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    first_beat = float(beat_times[0]) if len(beat_times) else 0.0

    log("Key ...")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, key_conf = estimate_key([float(v) for v in chroma.mean(axis=1)])

    log("Loudness curve ...")
    hop = int(sr * RMS_HOP_SEC)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    rms_db = [float(20 * np.log10(v + 1e-9)) for v in rms]
    drop = find_drop(rms_db)
    drop_snapped = snap_to_beat(drop, first_beat, bpm)

    return {
        "success": True,
        "audio": path,
        "duration_sec": round(duration, 2),
        "bpm": round(bpm, 2),
        "key": key,
        "key_confidence": key_conf,
        "first_beat_sec": round(first_beat, 2),
        "bar_sec": round(bar_seconds(bpm), 3),
        "drop_sec": drop_snapped,
        "loudness_db": [round(v, 1) for v in rms_db[::int(round(4 / RMS_HOP_SEC))]],
        "loudness_hop_sec": 4.0,
        "note": "first_beat_sec is where the beat comes in (use as lego --start); "
                "drop_sec is the largest sustained loudness rise, snapped to the beat "
                "(null if the track has none).",
    }


def main():
    parser = argparse.ArgumentParser(description="Structure analysis for vocal placement")
    parser.add_argument("audio")
    args = parser.parse_args()
    try:
        result = analyze(args.audio)
    except Exception as e:  # report, don't crash with a traceback on stdout
        result = {"success": False, "error": str(e)}
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
