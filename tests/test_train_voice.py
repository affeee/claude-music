"""Pure helpers of music_train_voice.py: clip windows, timed lyrics, dataset JSON,
checkpoint choice and the profile update. Stdlib only, no ACE-Step or ffmpeg."""
import json
import math

import music_train_voice as tv


def test_default_tag():
    assert tv.default_tag("aria") == "Aria"
    assert tv.default_tag("deep-alto") == "Deep Alto"


def test_clip_windows_aria_recipe():
    # 20 s windows, 8 s hop: the last regular window ends exactly at 164 s
    wins = tv.clip_windows(164.0)
    assert len(wins) == 19
    assert wins[0] == (0.0, 20.0) and wins[1] == (8.0, 28.0) and wins[-1] == (144.0, 164.0)


def test_clip_windows_pins_tail():
    wins = tv.clip_windows(50.0)
    assert wins == [(0.0, 20.0), (8.0, 28.0), (16.0, 36.0), (24.0, 44.0), (30.0, 50.0)]


def test_clip_windows_short_and_empty():
    assert tv.clip_windows(12.5) == [(0.0, 12.5)]
    assert tv.clip_windows(0) == []


def test_parse_listen_report():
    report = {"transcript": {"segments": [
        {"start": 1.0, "end": 4.0, "text": " City lights "},
        {"start": 5.0, "end": 7.0, "text": ""},
    ]}}
    lines = tv.parse_timed_lyrics(json.dumps(report))
    assert lines == [{"start": 1.0, "end": 4.0, "text": "City lights", "section": "Verse"}]


def test_parse_segment_list():
    lines = tv.parse_timed_lyrics('[{"start": 2, "end": 3, "text": "Hi", "section": "Chorus"}]')
    assert lines[0]["section"] == "Chorus" and lines[0]["end"] == 3.0


def test_parse_lrc_sections_and_ends():
    lrc = "[Verse]\n[00:01.00] One\n[00:04.50] Two\n[00:06.00]\n[Chorus]\n[01:00.00] Three\n"
    lines = tv.parse_timed_lyrics(lrc)
    assert [(ln["start"], ln["end"], ln["text"], ln["section"]) for ln in lines] == [
        (1.0, 4.5, "One", "Verse"),
        (4.5, 6.0, "Two", "Verse"),     # empty stamp ends the line
        (60.0, 65.0, "Three", "Chorus"),  # the last line runs 5 s
    ]


def test_parse_untimed_is_none():
    assert tv.parse_timed_lyrics("[Verse]\nJust words\nMore words") is None
    assert tv.parse_timed_lyrics("") is None
    assert tv.parse_timed_lyrics('{"foo": 1}') is None


def test_lines_in_window_needs_half_the_line():
    lines = [{"start": 18.0, "end": 22.0, "text": "a"},   # half inside 0-20
             {"start": 19.0, "end": 25.0, "text": "b"}]   # a sixth inside
    assert [ln["text"] for ln in tv.lines_in_window(lines, 0, 20)] == ["a"]


def test_clip_lyrics_tags_on_section_change():
    lines = [{"text": "a", "section": "Verse"}, {"text": "b", "section": "Verse"},
             {"text": "c", "section": "Chorus"}]
    assert tv.clip_lyrics(lines) == "[Verse]\na\nb\n[Chorus]\nc"
    assert tv.clip_lyrics([]) == "[Verse]"


def test_select_clips_with_timed_lyrics_drops_silent_windows():
    lines = [{"start": 2.0, "end": 5.0, "text": "one", "section": "Verse"},
             {"start": 40.0, "end": 43.0, "text": "two", "section": "Verse"}]
    clips = tv.select_clips(tv.clip_windows(50.0), lines=lines)
    assert [(c["start"], c["lyrics"]) for c in clips] == [
        (0.0, "[Verse]\none"), (24.0, "[Verse]\ntwo"), (30.0, "[Verse]\ntwo")]


def test_energy_selection():
    frame = tv.ENERGY_FRAME_SEC
    # 40 s: silence for 20 s, then loud
    levels = [-90.0] * int(20 / frame) + [-12.0] * int(20 / frame)
    assert tv.active_fraction(levels, 0, 20) == 0.0
    assert tv.active_fraction(levels, 20, 40) == 1.0
    clips = tv.select_clips(tv.clip_windows(40.0), levels=levels)
    assert [c["start"] for c in clips] == [8.0, 16.0, 20.0]
    assert all(c["lyrics"] == "[Verse]" for c in clips)


def test_frame_levels_db():
    n = int(tv.ENERGY_RATE * tv.ENERGY_FRAME_SEC)
    levels = tv.frame_levels_db([0] * n + [16384] * n)
    assert levels[0] == -120.0
    assert math.isclose(levels[1], 20 * math.log10(0.5), abs_tol=1e-6)


def test_build_dataset_matches_side_step_format():
    clips = [{"path": "/v/aria_01.wav", "start": 0.0, "end": 20.0, "lyrics": "[Verse]\nx"}]
    ds = tv.build_dataset("aria-run", "Aria", "female singer", clips, bpm=129, key="C minor")
    assert ds["metadata"] == {"name": "aria-run", "custom_tag": "Aria", "tag_position": "prepend",
                              "genre_ratio": 0}
    s = ds["samples"][0]
    assert s["filename"] == "aria_01.wav" and s["audio_path"] == "/v/aria_01.wav"
    assert s["caption"] == "female singer"   # Side-Step prepends the tag itself
    assert (s["bpm"], s["keyscale"], s["duration"], s["is_instrumental"], s["labeled"]) == (
        129, "C minor", 20.0, False, True)


def test_strip_tag():
    assert tv.strip_tag("Aria, female singer", "Aria") == "female singer"
    assert tv.strip_tag("aria,female singer", "Aria") == "female singer"
    assert tv.strip_tag("Ariana style", "Aria") == "Ariana style"
    assert tv.strip_tag(None, "Aria") == ""


def test_parse_epoch_line():
    assert tv.parse_epoch_line("  [OK] Epoch 3/80 in 221.4s, Loss: 0.7086\n") == (3, 80, 221.4, 0.7086)
    assert tv.parse_epoch_line("Epoch 3/80, Step 50, Loss: 0.7") is None


def test_checkpoints_pick_lowest_loss_or_epoch():
    names = ["epoch_10_loss_0.7812", "epoch_70_loss_0.6647", "epoch_80_loss_0.7245",
             "epoch_20_loss_nan", "runs"]
    cks = tv.list_checkpoints(names)
    assert [c["epoch"] for c in cks] == [10, 20, 70, 80]
    assert tv.pick_checkpoint(cks)["epoch"] == 70
    assert tv.pick_checkpoint(cks, epoch=80)["name"] == "epoch_80_loss_0.7245"
    assert tv.pick_checkpoint(cks, epoch=30) is None
    assert tv.pick_checkpoint([]) is None


def test_updated_profile_keeps_settings():
    old = {"name": "Aria", "version": 2, "lego": {"caption": "Aria, old", "lora_scale": 0.8, "seed": 7},
           "post": {"clean": False}}
    new = tv.updated_profile(old, "Aria", "Aria, female singer", {"epoch": 70}, "2026-10-16")
    assert new["version"] == 3
    assert new["lego"] == {"caption": "Aria, female singer", "lora_scale": 0.8, "seed": 7,
                           "lora": "adapter", "track": "vocals", "inference_steps": 50}
    assert new["post"] == {"clean": False}
    assert new["adapter"] == {"epoch": 70}
    assert old["version"] == 2  # input untouched


def test_updated_profile_from_nothing():
    new = tv.updated_profile({}, "Nova", "male tenor", {}, "2026-10-16")
    assert new["name"] == "Nova" and new["version"] == 1 and new["created"] == "2026-10-16"
    assert new["lego"]["caption"] == "Nova, male tenor"


def test_train_args_are_the_aria_recipe():
    cmd = tv.sidestep_cmd("r.py", tv.TRAIN_ARGS, tv.TRAIN_FLAGS)
    pairs = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
    assert pairs["--adapter-type"] == "lokr" and pairs["--lokr-linear-dim"] == "16"
    assert pairs["--lokr-linear-alpha"] == "32" and pairs["--lr"] == "3e-4"
    assert pairs["--optimizer-type"] == "adafactor" and pairs["--precision"] == "fp32"
    assert pairs["--num-workers"] == "0" and pairs["--batch-size"] == "1"
    assert pairs["--model-variant"] == "base"
    assert "--offload-encoder" in cmd and "--gradient-checkpointing" in cmd


def test_inside_rel(tmp_path):
    root = tmp_path / "ace"
    (root / "datasets").mkdir(parents=True)
    assert tv.inside_rel(root / "datasets" / "aria", root) == "./datasets/aria"
    assert tv.inside_rel(tmp_path / "elsewhere", root) == str((tmp_path / "elsewhere").resolve())
