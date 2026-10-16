"""Contract tests for music_engine.py.

Goals (per Refactor Plan R7):
  (a) Quality presets resolve to expected model/step/batch values.
  (b) JSON output contract is enforced (stdout = one JSON object).
  (c) Cover-mode parameter mapping: src_audio + cover_noise_strength (NOT
      reference_audio / audio_cover_strength — regression guard for the bug
      fixed in Session 1).
  (d) config.json has the CHANGE_ME placeholder or a valid absolute path.

These tests do NOT import torch or acestep — they must be GPU-free and run
in under 1 second so CI stays fast.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import subprocess
import sys
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# (a) Quality preset contract
# ---------------------------------------------------------------------------

EXPECTED_PRESETS = {
    "draft":    {"model": "acestep-v15-turbo", "lm_model": None,                    "inference_steps": 8,  "guidance_scale": 0.0, "thinking": False, "batch_size": 4, "shift": 1.0},
    "standard": {"model": "acestep-v15-turbo", "lm_model": None,                    "inference_steps": 8,  "guidance_scale": 0.0, "thinking": False, "batch_size": 2, "shift": 1.0},
    "high":     {"model": "acestep-v15-turbo", "lm_model": "acestep-5Hz-lm-1.7B",  "inference_steps": 8,  "guidance_scale": 0.0, "thinking": True,  "batch_size": 2, "shift": 1.0},
    "max":      {"model": "acestep-v15-base",  "lm_model": "acestep-5Hz-lm-1.7B",  "inference_steps": 65, "guidance_scale": 4.0, "thinking": True,  "batch_size": 1, "shift": 6.0},
}


def test_quality_presets_structure():
    from music_engine import QUALITY_PRESETS

    assert set(QUALITY_PRESETS.keys()) == set(EXPECTED_PRESETS.keys()), (
        f"Preset names changed. Update refactor plan + parameters.md. "
        f"Expected {set(EXPECTED_PRESETS.keys())}, got {set(QUALITY_PRESETS.keys())}"
    )

    for name, expected in EXPECTED_PRESETS.items():
        actual = QUALITY_PRESETS[name]
        for key, value in expected.items():
            assert actual.get(key) == value, (
                f"Preset '{name}' field '{key}': expected {value!r}, got {actual.get(key)!r}"
            )


def test_max_preset_uses_base_model():
    """Guard: 'max' quality must NOT use turbo — base model is required for
    repaint/extract/lego/complete task types."""
    from music_engine import QUALITY_PRESETS

    assert QUALITY_PRESETS["max"]["model"] == "acestep-v15-base"
    assert QUALITY_PRESETS["max"]["inference_steps"] >= 50
    assert QUALITY_PRESETS["max"]["guidance_scale"] >= 1.0


def test_draft_preset_optimized_for_batch():
    """Guard: 'draft' ships batch_size=4 for random/exploration workflows."""
    from music_engine import QUALITY_PRESETS

    assert QUALITY_PRESETS["draft"]["batch_size"] == 4
    assert QUALITY_PRESETS["draft"]["inference_steps"] == 8  # turbo fast path


# ---------------------------------------------------------------------------
# (b) JSON output contract
# ---------------------------------------------------------------------------

def test_output_json_is_valid_stdout():
    """output_json() writes exactly one parseable JSON object to stdout."""
    from music_engine import output_json

    buf = io.StringIO()
    with redirect_stdout(buf):
        output_json({"success": True, "count": 3, "outputs": []})

    payload = buf.getvalue()
    parsed = json.loads(payload)  # will raise if not valid JSON
    assert parsed["success"] is True
    assert parsed["count"] == 3


def test_error_json_exits_nonzero_with_json():
    """error_json() writes JSON error to stdout AND exits with code 1."""
    from music_engine import error_json

    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc_info:
        error_json("test error", suggestion="run --help")

    assert exc_info.value.code == 1
    parsed = json.loads(buf.getvalue())
    assert parsed["success"] is False
    assert parsed["error"] == "test error"
    assert parsed["suggestion"] == "run --help"


def test_help_subcommand_is_available():
    """Sanity: --help must return nonzero? No — argparse returns 0 on --help."""
    import sys as _sys
    from pathlib import Path as _Path

    engine = _Path(__file__).resolve().parents[1] / "skills" / "claude-music" / "scripts" / "music_engine.py"
    result = subprocess.run(
        [_sys.executable, str(engine), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "generate" in result.stdout
    assert "cover" in result.stdout
    assert "repaint" in result.stdout


# ---------------------------------------------------------------------------
# (c) Cover-mode parameter mapping — REGRESSION GUARD
# ---------------------------------------------------------------------------

def test_cover_mode_uses_src_audio_not_reference_audio(engine_source):
    """The ACE-Step API field is `src_audio` for cover mode, NOT
    `reference_audio`. The latter was a bug fixed in Session 1 —
    regression guard per Refactor Plan R7."""
    # AST walk: find cmd_cover function and look at its GenerationParams(...) call
    tree = ast.parse(engine_source)
    cover_fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cmd_cover"),
        None,
    )
    assert cover_fn is not None, "cmd_cover function not found in music_engine.py"

    gen_params_calls = [
        n for n in ast.walk(cover_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "GenerationParams"
    ]
    assert gen_params_calls, "cmd_cover does not call GenerationParams(...)"

    kwargs = {kw.arg for kw in gen_params_calls[0].keywords if kw.arg}

    assert "src_audio" in kwargs, (
        "Cover mode must pass src_audio — this is the real ACE-Step API field. "
        "If you see this failing, someone reintroduced the Session-1 bug."
    )
    assert "cover_noise_strength" in kwargs, (
        "Cover mode must pass cover_noise_strength — this is the real "
        "ACE-Step API field. Not audio_cover_strength."
    )
    assert "reference_audio" not in kwargs, (
        "Cover mode must NOT pass reference_audio — that field name was "
        "wrong and was removed in Session 1. This is a regression."
    )
    assert "audio_cover_strength" not in kwargs, (
        "Cover mode must NOT pass audio_cover_strength — that field name "
        "was wrong. Use cover_noise_strength instead."
    )


def test_cover_task_type_string(engine_source):
    """Cover mode must set task_type='cover'."""
    tree = ast.parse(engine_source)
    cover_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cmd_cover"
    )
    gen_params = next(
        n for n in ast.walk(cover_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "GenerationParams"
    )
    task_type_kw = next((kw for kw in gen_params.keywords if kw.arg == "task_type"), None)
    assert task_type_kw is not None
    assert isinstance(task_type_kw.value, ast.Constant)
    assert task_type_kw.value.value == "cover"


# ---------------------------------------------------------------------------
# (d) config.json path resolution
# ---------------------------------------------------------------------------

def test_config_json_has_ace_step_dir_key(config_json):
    assert "ace_step_dir" in config_json
    assert "checkpoint_dir" in config_json
    assert "output_dir" in config_json
    assert "defaults" in config_json


def test_config_json_placeholder_or_absolute(config_json):
    """ace_step_dir must be either CHANGE_ME (pre-install) or a concrete path.
    Anything else suggests an accidentally-committed user path (VULN-002
    regression guard)."""
    ace_dir = config_json["ace_step_dir"]
    assert ace_dir == "CHANGE_ME" or ace_dir.startswith("/") or ace_dir.startswith("~"), (
        f"ace_step_dir should be 'CHANGE_ME' pre-install or an absolute path, "
        f"got: {ace_dir!r}. If a user's personal path leaked, run the installer "
        f"or reset to 'CHANGE_ME' before committing."
    )


def test_config_defaults_shape(config_json):
    """Guard: config defaults declare the keys music_engine.py actually reads.

    model, lm_model, batch_size and thinking are deliberately NOT required —
    the quality preset owns them. They are honoured as optional pins if a user
    adds them, but shipping them would override every preset (a config
    batch_size of 2 would silently defeat draft's batch of 4).
    """
    defaults = config_json["defaults"]
    required = {"quality", "format", "language"}
    assert required.issubset(set(defaults.keys())), (
        f"config.defaults is missing keys: {required - set(defaults.keys())}"
    )


def test_config_defaults_are_honoured_by_parser(engine_module):
    """Regression: the defaults block must reach argparse.

    Before this was wired, --quality hardcoded default="standard" and every
    key in config.defaults was dead config that looked authoritative.
    Compares against what the engine itself loads, so it holds both on dev
    machines (real config.json) and in CI (no config.json, built-in
    fallbacks).
    """
    defaults = engine_module.load_config().get("defaults") or {}
    parser = engine_module.build_parser()
    args = parser.parse_args(["generate", "-c", "x"])
    assert args.quality == (defaults.get("quality") or "standard")
    assert args.format == (defaults.get("format") or "flac")
    assert args.language == (defaults.get("language") or "en")


def test_quality_preset_survives_explicit_flag(engine_module):
    """CLI > config > preset. An explicit --quality must restore that preset's
    model/LM/thinking/batch even when config.json defaults to something else."""
    parser = engine_module.build_parser()

    draft = engine_module.resolve_quality(
        parser.parse_args(["--quality", "draft", "generate", "-c", "x"]))
    assert draft.lm_model is None
    assert draft.thinking is False
    assert draft.batch == 4, "config must not clobber the draft preset's batch"

    high = engine_module.resolve_quality(
        parser.parse_args(["--quality", "high", "generate", "-c", "x"]))
    assert high.lm_model == "acestep-5Hz-lm-1.7B"
    assert high.thinking is True


def test_cli_flag_beats_config_default(engine_module):
    """An explicit CLI flag must win over config.json."""
    parser = engine_module.build_parser()
    args = engine_module.resolve_quality(
        parser.parse_args(["--batch", "7", "--format", "wav", "generate", "-c", "x"]))
    assert args.batch == 7
    assert args.format == "wav"


def test_invalid_config_value_falls_back(engine_module):
    """A bad enum in config.json must warn and fall back, not crash.

    argparse does not validate defaults against choices, so an unchecked bad
    value would otherwise flow straight into the pipeline.
    """
    assert engine_module.config_default(
        {"quality": "ultra"}, "quality", "standard",
        ["draft", "standard", "high", "max"]) == "standard"
    assert engine_module.config_default({}, "quality", "standard") == "standard"
    assert engine_module.config_default({"quality": None}, "quality", "standard") == "standard"
    assert engine_module.config_default({"quality": "high"}, "quality", "standard") == "high"


# ---------------------------------------------------------------------------
# (d2) Output naming
# ---------------------------------------------------------------------------

def test_slugify_strips_path_and_shell_metacharacters(engine_module):
    """A caption reaches the filesystem, so it must not carry separators."""
    dirty = "../../etc/passwd; rm -rf ~ && echo $HOME"
    slug = engine_module.slugify(dirty)
    assert "/" not in slug and "\\" not in slug
    assert not any(c in slug for c in ";&$|`'\" ")
    assert engine_module.slugify("") == ""
    assert engine_module.slugify("!!! ???") == ""


def test_slugify_bounds_length(engine_module):
    slug = engine_module.slugify("one two three four five six seven eight nine")
    assert slug.count("-") <= 5, "should keep at most 6 words"
    assert len(engine_module.slugify("x" * 200)) <= 48


def test_informative_stem_drops_uuids_keeps_labels(engine_module):
    """UUID names are noise; real stem labels must survive renaming."""
    assert engine_module.informative_stem("/out/fdd0cca5-0d06-2c6b-e7dd-1bd0a41273ce.flac") == ""
    assert engine_module.informative_stem("/out/8e71acc2d949ef877beb.flac") == ""
    assert engine_module.informative_stem("/out/vocals.flac") == "vocals"
    assert engine_module.informative_stem("/out/drums_stem.wav") == "drums-stem"


def test_rename_outputs_builds_readable_name(engine_module, tmp_path):
    src = tmp_path / "fdd0cca5-0d06-2c6b-e7dd-1bd0a41273ce.flac"
    src.write_bytes(b"audio")

    args = argparse.Namespace(naming="descriptive")
    outputs = [{"path": str(src), "seed": 3128607774, "index": 1}]
    engine_module.rename_outputs(args, outputs, label="lo-fi, afro-latin percussion")

    name = Path(outputs[0]["path"]).name
    assert name.startswith("lo-fi-afro-latin-percussion_")
    assert name.endswith("_01_s3128607774.flac")
    assert Path(outputs[0]["path"]).exists()
    assert not src.exists(), "original UUID file should have been renamed, not copied"


def test_rename_outputs_never_overwrites(engine_module, tmp_path):
    """Two runs with the same caption in the same minute must not clobber."""
    args = argparse.Namespace(naming="descriptive")
    made = []
    for _ in range(2):
        src = tmp_path / f"{uuid.uuid4()}.flac"
        src.write_bytes(b"audio")
        outputs = [{"path": str(src), "seed": 7, "index": 1}]
        engine_module.rename_outputs(args, outputs, label="same caption")
        made.append(outputs[0]["path"])

    assert made[0] != made[1], "second file must get a -2 suffix, not overwrite"
    assert all(Path(p).exists() for p in made)
    assert len(list(tmp_path.glob("*.flac"))) == 2, "no file may be lost"


def test_naming_uuid_mode_is_a_no_op(engine_module, tmp_path):
    src = tmp_path / "keep-this-name.flac"
    src.write_bytes(b"audio")
    args = argparse.Namespace(naming="uuid")
    outputs = [{"path": str(src), "seed": 1, "index": 1}]
    engine_module.rename_outputs(args, outputs, label="whatever")
    assert outputs[0]["path"] == str(src)
    assert src.exists()


def test_rename_outputs_survives_missing_file(engine_module, tmp_path):
    """A path the pipeline reported but did not write must not raise."""
    args = argparse.Namespace(naming="descriptive")
    outputs = [{"path": str(tmp_path / "gone.flac"), "seed": 1, "index": 1}]
    engine_module.rename_outputs(args, outputs, label="x")  # must not raise


def test_naming_default_comes_from_config(engine_module):
    parser = engine_module.build_parser()
    assert parser.parse_args(["generate", "-c", "x"]).naming in ("descriptive", "uuid")


# ---------------------------------------------------------------------------
# (e) Security regression guards (from Session 1 audit)
# ---------------------------------------------------------------------------

def test_no_eval_in_scripts():
    """VULN-001 regression: eval() must not appear in any shell script."""
    from pathlib import Path
    scripts = (Path(__file__).resolve().parents[1]
               / "skills" / "claude-music" / "scripts")
    for sh in scripts.glob("*.sh"):
        content = sh.read_text()
        # eval with a following space = the dangerous form; allow `evaluate`
        assert " eval " not in content and not content.startswith("eval "), (
            f"{sh.name} uses eval — replace with array execution "
            f"(VULN-001 regression)."
        )


def test_export_uses_no_overwrite():
    """VULN-006 regression: ffmpeg must use -n (no-overwrite), not -y."""
    from pathlib import Path
    export_sh = (Path(__file__).resolve().parents[1]
                 / "skills" / "claude-music" / "scripts" / "music_export.sh")
    content = export_sh.read_text()
    # Every ffmpeg invocation in export must use -n, not -y
    # (allow mentions in comments)
    lines = [ln for ln in content.splitlines()
             if "ffmpeg " in ln and not ln.strip().startswith("#")]
    for ln in lines:
        assert " -y " not in ln, f"ffmpeg -y (overwrite) in export: {ln.strip()}"


# ---------------------------------------------------------------------------
# (g) Base-model tasks: extract / lego / complete
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["lego", "--src-audio", "x.mp3", "--track", "vocals"],
    ["extract", "--src-audio", "x.mp3"],
    ["complete", "--src-audio", "x.mp3"],
])
def test_base_tasks_do_not_inherit_turbo_steps(engine_module, argv):
    """Regression: a turbo preset switched to the base model kept turbo's
    8 steps and guidance 0.0, which the base model cannot work with."""
    parser = engine_module.build_parser()
    args = engine_module.resolve_base_quality(
        parser.parse_args(["--quality", "standard", *argv]), argv[0])
    best = engine_module.QUALITY_PRESETS["max"]
    assert args.model == "acestep-v15-base"
    assert args.inference_steps == best["inference_steps"]
    assert args.guidance_scale == best["guidance_scale"]
    assert args.shift == best["shift"]


def test_base_tasks_keep_explicit_steps(engine_module):
    parser = engine_module.build_parser()
    args = engine_module.resolve_base_quality(parser.parse_args(
        ["--quality", "standard", "lego", "--src-audio", "x.mp3",
         "--inference-steps", "50"]), "lego")
    assert args.inference_steps == 50
    assert args.guidance_scale == engine_module.QUALITY_PRESETS["max"]["guidance_scale"]


def test_task_instruction_names_the_track(engine_module):
    """ACE-Step reads the track from the instruction text; without it lego
    does not know which layer to add."""
    assert engine_module.task_instruction("lego", "vocals") == (
        "Generate the VOCALS track based on the audio context:")
    assert engine_module.task_instruction("lego") == (
        "Generate the track based on the audio context:")
    assert engine_module.task_instruction("extract", "drums") == (
        "Extract the DRUMS track from the audio:")


def test_lego_passes_instruction_and_language(engine_source):
    assert 'instruction=task_instruction("lego", args.track)' in engine_source
    parser_args = ["lego", "--src-audio", "x.mp3", "--track", "vocals",
                   "--language", "sv", "--bpm", "129", "--key", "C minor", "--mix"]
    import music_engine
    args = music_engine.build_parser().parse_args(parser_args)
    assert (args.track, args.language, args.bpm, args.key, args.mix) == (
        "vocals", "sv", 129, "C minor", True)
    assert (args.start, args.end) == (0.0, -1.0)
    late = music_engine.build_parser().parse_args([*parser_args, "--start", "29.8"])
    assert late.start == 29.8
    assert "repainting_start=args.start" in engine_source


def test_generation_timeout_is_raised(engine_source):
    """ACE-Step's 600s default kills a base-model run on a full-length track."""
    assert 'os.environ.setdefault("ACESTEP_GENERATION_TIMEOUT"' in engine_source


def test_mix_command_never_overwrites_or_touches_source(engine_module):
    cmd = engine_module.build_mix_command("src.mp3", "layer.flac", "out.flac", -2.0)
    assert cmd[0] == "ffmpeg" and "-n" in cmd and "-y" not in cmd
    assert cmd[-1] == "out.flac"
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "volume=-5.0dB[layer]" in graph
    assert "alimiter" in graph
    assert "afade" not in graph


def test_mix_mutes_the_layer_outside_its_region(engine_module):
    """Outside --start/--end the lego output is the source audio (measured:
    correlation 1.0), so mixing it unmuted doubles the source there."""
    cmd = engine_module.build_mix_command("s.mp3", "l.flac", "o.flac", 0.0, 29.8, 150.0)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "afade=t=in:st=29.75:d=0.05" in graph
    assert "afade=t=out:st=150.00:d=0.05" in graph


def test_mix_with_source_writes_beside_layer(engine_module, tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    src, layer = tmp_path / "src.wav", tmp_path / "layer.wav"
    for path, freq in ((src, 220), (layer, 440)):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=frequency={freq}:duration=1", str(path)], check=True)
    first = engine_module.mix_with_source(str(src), str(layer))
    second = engine_module.mix_with_source(str(src), str(layer))
    assert first == str(tmp_path / "layer_mix.wav")
    assert second == str(tmp_path / "layer_mix-2.wav")
    assert Path(first).stat().st_size > 0 and Path(second).stat().st_size > 0


# ---------------------------------------------------------------------------
# (h) Unified memory (Apple Silicon): batch, LM, offload, progress
# ---------------------------------------------------------------------------

MAC_16 = {"unified_memory": True, "memory_gb": 16.0}
MAC_64 = {"unified_memory": True, "memory_gb": 64.0}
NVIDIA = {"unified_memory": False, "memory_gb": 32.0}


def test_memory_safe_batch_caps_full_tracks_on_small_macs(engine_module):
    safe = engine_module.memory_safe_batch
    assert safe(2, 182, profile=MAC_16) == 1, "two full-length variants ran out of MPS memory"
    assert safe(4, 30, profile=MAC_16) == 4, "short drafts keep their variants"
    assert safe(2, None, profile=MAC_16) == 1, "unknown length counts as a full track"
    assert safe(2, 182, profile=MAC_64) == 2
    assert safe(2, 600, profile=MAC_16) == 1, "never below one"


def test_memory_policy_leaves_discrete_gpus_alone(engine_module):
    assert engine_module.memory_safe_batch(4, 600, profile=NVIDIA) == 4


def test_explicit_batch_is_never_capped(engine_module, monkeypatch):
    monkeypatch.setattr(engine_module, "machine_profile", lambda: MAC_16)
    parser = engine_module.build_parser()
    args = engine_module.resolve_quality(parser.parse_args(
        ["--quality", "high", "--batch", "2", "generate", "-c", "x", "--duration", "240"]))
    assert args.batch == 2
    capped = engine_module.resolve_quality(parser.parse_args(
        ["--quality", "high", "generate", "-c", "x", "--duration", "240"]))
    if not (engine_module.load_config().get("defaults") or {}).get("batch_size"):
        assert capped.batch == 1


def test_lm_is_skipped_for_tasks_that_ignore_it(engine_module):
    """ACE-Step skips the LM for cover/repaint/extract; loading it anyway cost
    several GB and helped kill a batch-2 cover on a 16GB Mac."""
    ns = argparse.Namespace
    assert engine_module.task_uses_lm(ns(command="generate"))
    assert engine_module.task_uses_lm(ns(command="lego"))
    for task in ("cover", "repaint", "extract"):
        assert not engine_module.task_uses_lm(ns(command=task))


def _job(command="cover", batch=1, duration=182.0, lm=False, offload=None):
    return argparse.Namespace(
        command=command, batch=batch, _duration_sec=duration, offload=offload,
        lm_model="acestep-5Hz-lm-1.7B" if lm else None, thinking=lm)


def test_auto_offload_on_unified_memory(engine_module):
    """Measured on an M1/16GB, 182s track: offload off is ~24% faster for one
    variant, but two variants run out of MPS memory unless offloaded."""
    auto = engine_module.auto_offload
    assert auto(_job(), MAC_16) is False
    assert auto(_job(batch=2), MAC_16) is True
    assert auto(_job(batch=2, duration=None), MAC_16) is True
    assert auto(_job(), NVIDIA) is True, "discrete GPUs keep offloading"


def test_offload_precedence(engine_module, monkeypatch):
    resolve = engine_module.resolve_offload
    monkeypatch.setattr(engine_module, "machine_profile", lambda: MAC_16)
    assert resolve(_job(), {}, "offload_to_cpu") is False
    assert resolve(_job(), {"offload_to_cpu": "auto"}, "offload_to_cpu") is False
    assert resolve(_job(), {"offload_to_cpu": True}, "offload_to_cpu") is True
    assert resolve(_job(offload="off"), {"offload_to_cpu": True}, "offload_to_cpu") is False
    monkeypatch.setattr(engine_module, "machine_profile", lambda: NVIDIA)
    assert resolve(_job(), {}, "offload_to_cpu") is True
    assert resolve(_job(offload="off"), {}, "offload_to_cpu") is False


def test_diffusion_pct_follows_real_steps(engine_module):
    """ACE-Step's estimator said 58% with 0 of 50 steps done."""
    state = engine_module._diffusion
    saved = dict(state)
    try:
        state.update(step=0, total=50)
        assert engine_module.diffusion_pct(0.58) == 0.52
        state.update(step=25, total=50)
        assert engine_module.diffusion_pct(0.58) == pytest.approx(0.655)
        assert engine_module.diffusion_pct(0.30) == 0.30, "other phases pass through"
        assert engine_module.diffusion_pct(0.90) == 0.90
        assert engine_module.diffusion_pct(None) is None
    finally:
        state.update(saved)


def test_detect_gpu_reports_apple_silicon(scripts_dir):
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Apple Silicon only")
    out = subprocess.run(["bash", str(scripts_dir / "detect_gpu.sh")],
                         check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    assert info["gpu_detected"] is True
    assert info["backend"] == "mps" and info["unified_memory"] is True
    assert info["memory_total_mb"] > 0


def test_lm_that_does_not_fit_unified_memory_is_replaced_or_skipped(engine_module):
    """The 1.7B LM ran an M1/16GB out of memory on a 180s track with the pt
    and mlx backends, offload on and off. ACE-Step lists only 0.6B there."""
    fit = engine_module.fit_lm_to_memory
    big, small = "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B"
    model, note = fit(big, [small], installed=[big], profile=MAC_16)
    assert model is None and "acestep-download --model " + small in note
    model, note = fit(big, [small], installed=[big, small], profile=MAC_16)
    assert model == small and note
    assert fit(big, [small, big], installed=[big], profile=MAC_64) == (big, "")
    assert fit(big, [small], installed=[big], profile=NVIDIA) == (big, ""), (
        "discrete GPUs keep the preset's LM")
    assert fit(big, [], installed=[big], profile=MAC_16) == (big, ""), "unknown tier: leave it"


def test_lm_backend_is_mlx_on_apple_silicon(engine_module):
    assert engine_module.lm_backend(MAC_16) == "mlx"
    assert engine_module.lm_backend(NVIDIA) == "vllm"


# ---------------------------------------------------------------------------
# (i) Listening report (music_listen.py) — pure helpers, no audio libraries
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def listen(_scripts_importable):
    import music_listen
    return music_listen


def test_parse_key(listen):
    assert listen.parse_key("C minor") == {0, 2, 3, 5, 7, 8, 10}
    assert listen.parse_key("Am") == {9, 11, 0, 2, 4, 5, 7}
    assert listen.parse_key("Bb major") == {10, 0, 2, 3, 5, 7, 9}
    assert listen.parse_key("") is None and listen.parse_key("H dur") is None


def test_pitch_summary_reads_register_and_key(listen):
    c3, eb3, g3 = 130.81, 155.56, 196.0
    low = listen.pitch_summary([c3, eb3, g3, eb3, c3], listen.parse_key("C minor"))
    assert low["register"] == "baritone" and low["typical_range"] == "male"
    assert low["median_note"] == "D#3"
    assert low["in_key_ratio"] == 1.0
    assert low["mean_cents_off_semitone"] < 5
    e3 = 164.81  # E natural is outside C minor
    assert listen.pitch_summary([e3, e3, c3, c3], listen.parse_key("C minor"))["in_key_ratio"] == 0.5
    assert listen.pitch_summary([440.0])["typical_range"] == "female"
    f4 = listen.pitch_summary([349.2])
    assert (f4["register"], f4["typical_range"]) == ("alto", "female")
    assert listen.pitch_summary([233.1])["typical_range"] == "high male or low female"
    assert listen.pitch_summary([0, float("nan")]) is None


def test_active_regions_bridge_breaths_and_report_gaps(listen):
    # 0.1s frames: 1s sung, 0.2s breath, 1s sung, 6s silence, 1s sung, one blip
    flags = [True] * 10 + [False] * 2 + [True] * 10 + [False] * 60 + [True] * 10 + [False] * 20 + [True]
    regions = listen.active_regions(flags, 0.1, offset=30.0)
    assert regions == [(30.0, 32.2), (38.2, 39.2)]
    assert listen.find_gaps(regions) == [{"start": 32.2, "end": 38.2, "sec": 6.0}]


def test_lyric_lines_drop_section_tags(listen):
    assert listen.lyric_lines("[Verse 1]\nHello there\n\n[Chorus]\nBoo, boo\n") == [
        "Hello there", "Boo, boo"]


def test_match_lyrics_finds_lines_in_order_and_flags_missing(listen):
    def words(text, t0):
        return [{"word": w, "start": t0 + i, "end": t0 + i + 0.5}
                for i, w in enumerate(text.split())]

    heard = (words("the stairs are creaking while the city sleeps", 30)
             + words("boo you don't scare me", 60)
             + words("boo you dont scare me", 120))
    lines = ["The stairs are creaking while the city sleeps",
             "A light in your window that never goes out",
             "Boo, you don't scare me",
             "Boo, you don't scare me"]
    got = listen.match_lyrics(lines, heard)
    assert [m["found"] for m in got] == [True, False, True, True]
    assert got[0]["at"] == 30 and got[2]["at"] == 60
    assert got[3]["at"] == 120, "the repeated chorus matches its own occurrence"


def test_match_lyrics_prefers_the_earliest_good_match(listen):
    """Regression: 'Ooh, you don't scare me' at 60s lost to a clean repeat at
    130s, and the whole second verse was reported missing."""
    def words(text, t0):
        return [{"word": w, "start": t0 + i, "end": t0 + i + 0.5}
                for i, w in enumerate(text.split())]

    heard = (words("ooh you don't scare me", 60)
             + words("we run down shirts where the apples lay", 100)
             + words("boo you don't scare me", 130))
    got = listen.match_lyrics(
        ["Boo, you don't scare me", "We run down streets where the shadows play",
         "Boo, you don't scare me"], heard)
    assert [m["at"] for m in got] == [60, 100, 130]


def test_beat_offsets_use_eighth_notes(listen):
    beats = [0.0, 0.5, 1.0, 1.5]
    assert listen.beat_offsets_ms([0.51, 0.75, 1.02], beats) == [10, 0, 20]
    assert listen.beat_offsets_ms([0.3], [0.0]) == []


def test_listen_has_no_heavy_imports_at_module_level(scripts_dir):
    """The helpers above must import without numpy/librosa (CI has neither)."""
    tree = ast.parse((scripts_dir / "music_listen.py").read_text())
    top = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)}
    top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert not top & {"numpy", "librosa", "matplotlib", "soundfile", "mlx_whisper"}


# ---------------------------------------------------------------------------
# (j) Vocal flow: structure, clean, boost, voice profiles, orchestrator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def structure(_scripts_importable):
    import music_structure
    return music_structure


@pytest.fixture(scope="module")
def vocals(_scripts_importable):
    import music_vocals
    return music_vocals


def test_estimate_key_finds_c_minor(structure):
    # The C minor profile itself must score as C minor, and rotated to A as A minor.
    key, conf = structure.estimate_key(list(structure.MINOR_PROFILE))
    assert key == "C minor" and conf > 0.99
    rotated = structure.MINOR_PROFILE[-9:] + structure.MINOR_PROFILE[:-9]
    assert structure.estimate_key(rotated)[0] == "A minor"


def test_find_drop_skips_the_beat_entering_a_quiet_intro(structure):
    """Affe - Boo: quiet pad, beat in at ~15 s (+6 dB), drop at ~56 s onto the
    loud plateau. The beat entry is the bigger relative jump from silence-ish
    levels but does not reach the plateau, so it must not count."""
    hop = structure.RMS_HOP_SEC
    curve = [-34.0] * int(15 / hop) + [-22.0] * int(41 / hop) + [-12.5] * int(120 / hop)
    assert structure.find_drop(curve) == pytest.approx(56.0, abs=hop)


def test_find_drop_returns_none_without_a_rise(structure):
    assert structure.find_drop([-12.0] * 300) is None
    assert structure.find_drop([-12.0] * 3) is None


def test_snap_to_beat(structure):
    assert structure.snap_to_beat(56.1, 14.63, 129.2) == pytest.approx(55.96, abs=0.01)
    assert structure.snap_to_beat(None, 0.0, 120) is None


def test_structure_has_no_heavy_imports_at_module_level(scripts_dir):
    tree = ast.parse((scripts_dir / "music_structure.py").read_text())
    top = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)}
    top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert not top & {"numpy", "librosa", "soundfile"}


def test_mix_boost_lifts_the_layer_from_the_drop(engine_module):
    cmd = engine_module.build_mix_command("s.mp3", "l.flac", "o.flac", 0.0, 14.63, -1.0,
                                          boost_at=55.96, boost_db=5.0)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "eval=frame" in graph
    assert "lt(t,55.46)" in graph and "lt(t,55.96)" in graph, "ramp ends at the drop"
    assert "1.778279" in graph, "+5 dB"
    assert "0.707946" in graph, "-3 dB headroom kept"
    assert "afade=t=in:st=14.58" in graph


def test_mix_without_boost_is_unchanged(engine_module):
    plain = engine_module.build_mix_command("s", "l", "o", -2.0)
    assert plain == engine_module.build_mix_command("s", "l", "o", -2.0, boost_at=None,
                                                    boost_db=5.0)
    assert plain == engine_module.build_mix_command("s", "l", "o", -2.0, boost_at=50.0,
                                                    boost_db=0.0)


def test_clean_script_is_deterministic_and_safe(scripts_dir):
    src = (scripts_dir / "music_clean.sh").read_text()
    assert "--shifts 0" in src, "Demucs' random shift makes two runs differ"
    assert '"demucs==4.0.1"' in src and '"torch==2.5.1"' in src, "pinned for repeatable output"
    assert "ffmpeg -n" in src
    assert "-d mps" not in src, "MPS fails inside Demucs"


def test_lego_parser_has_flow_flags(engine_module):
    args = engine_module.build_parser().parse_args(
        ["lego", "--src-audio", "x.mp3", "--clean", "--voice", "aria",
         "--mix-boost-at", "55.96", "--mix-boost", "5"])
    assert args.clean is True and args.voice == "aria"
    assert args.mix_boost_at == 55.96 and args.mix_boost == 5.0
    unset = engine_module.build_parser().parse_args(["lego", "--src-audio", "x.mp3"])
    assert unset.clean is None and unset.mix_gain is None and unset.mix_boost is None


def _write_profile(tmp_path, name="aria", **post):
    d = tmp_path / name
    d.mkdir()
    (d / "profile.json").write_text(json.dumps({
        "name": name.title(),
        "lego": {"caption": "airy female vocal", "seed": 20261016,
                 "inference_steps": 50, "track": "vocals"},
        "post": {"clean": True, "mix_gain_db": 0.0, "boost_db": 5.0, **post},
        "lyrics": {"intro_line": "Ooh, ooh"},
    }))
    return d


def test_voice_profile_fills_only_unset_values(engine_module, tmp_path):
    _write_profile(tmp_path)
    profile = engine_module.load_voice_profile("aria", base_dir=tmp_path)
    args = engine_module.build_parser().parse_args(["lego", "--src-audio", "x.mp3"])
    engine_module.apply_voice_profile(args, profile)
    assert (args.caption, args.seed, args.inference_steps, args.track) == (
        "airy female vocal", 20261016, 50, "vocals")
    assert args.clean is True and args.mix_gain == 0.0 and args.mix_boost == 5.0

    args = engine_module.build_parser().parse_args(
        ["--seed", "7", "lego", "--src-audio", "x.mp3", "--caption", "mine",
         "--no-clean", "--mix-boost", "2"])
    engine_module.apply_voice_profile(args, profile)
    assert (args.caption, args.seed, args.clean, args.mix_boost) == ("mine", 7, False, 2.0)


@pytest.mark.parametrize("name", ["../etc", "Aria/x", "", "a" * 60, "ari a"])
def test_voice_profile_rejects_unsafe_names(engine_module, tmp_path, name):
    with pytest.raises(ValueError):
        engine_module.load_voice_profile(name, base_dir=tmp_path)


def test_voice_profile_missing(engine_module, tmp_path):
    with pytest.raises(ValueError, match="no voice profile"):
        engine_module.load_voice_profile("nobody", base_dir=tmp_path)


def test_intro_line_only_when_starting_mid_song(vocals):
    lyr = "[Verse]\nCity lights"
    assert vocals.with_intro_line(lyr, "Ooh, ooh", 14.63).startswith("[Intro]\nOoh, ooh\n\n[Verse]")
    assert vocals.with_intro_line(lyr, "Ooh, ooh", 0) == lyr
    assert vocals.with_intro_line(lyr, None, 14.63) == lyr
    has_intro = "[Intro]\nMmm\n\n[Verse]\nx"
    assert vocals.with_intro_line(has_intro, "Ooh, ooh", 14.63) == has_intro


def test_resolve_time(vocals):
    assert vocals.resolve_time("auto", 55.96) == 55.96
    assert vocals.resolve_time(None, 14.63) == 14.63
    assert vocals.resolve_time("none", 55.96) is None
    assert vocals.resolve_time("30", 55.96) == 30.0


def test_last_json_skips_log_noise(vocals):
    assert vocals.last_json('noise\n{"a": 1}') == {"a": 1}
    assert vocals.last_json('{\n  "a": {"b": 2}\n}') == {"a": {"b": 2}}
    assert vocals.last_json("no json here") is None


def test_vocals_is_stdlib_only(scripts_dir):
    tree = ast.parse((scripts_dir / "music_vocals.py").read_text())
    top = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)}
    top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert top <= set(sys.stdlib_module_names)


def test_every_generation_config_honours_seed(engine_source):
    """Regression: lego/extract/complete passed --seed in GenerationParams but
    left GenerationConfig.use_random_seed at ACE-Step's default (True), so the
    seed was silently replaced by a random one and runs could not be repeated."""
    tree = ast.parse(engine_source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "GenerationConfig"]
    assert len(calls) == 6
    for call in calls:
        assert "use_random_seed" in {k.arg for k in call.keywords}, ast.unparse(call)


def test_voice_profile_reference_audio_is_relative_to_profile(engine_module, tmp_path):
    d = _write_profile(tmp_path)
    data = json.loads((d / "profile.json").read_text())
    data["lego"]["reference_audio"] = "reference.wav"
    (d / "profile.json").write_text(json.dumps(data))
    profile = engine_module.load_voice_profile("aria", base_dir=tmp_path)
    args = engine_module.build_parser().parse_args(["lego", "--src-audio", "x.mp3"])
    engine_module.apply_voice_profile(args, profile)
    assert args.reference_audio == str(d / "reference.wav")
    args = engine_module.build_parser().parse_args(
        ["lego", "--src-audio", "x.mp3", "--reference-audio", "/mine.wav"])
    engine_module.apply_voice_profile(args, profile)
    assert args.reference_audio == "/mine.wav"


def test_voice_profile_lora_resolves_and_cli_wins(engine_module, tmp_path):
    d = _write_profile(tmp_path)
    data = json.loads((d / "profile.json").read_text())
    data["lego"].update(lora="adapter", lora_scale=0.8)
    (d / "profile.json").write_text(json.dumps(data))
    profile = engine_module.load_voice_profile("aria", base_dir=tmp_path)
    args = engine_module.build_parser().parse_args(["lego", "--src-audio", "x.mp3"])
    engine_module.apply_voice_profile(args, profile)
    assert args.lora == str(d / "adapter") and args.lora_scale == 0.8
    args = engine_module.build_parser().parse_args(
        ["lego", "--src-audio", "x.mp3", "--lora", "/other", "--lora-scale", "0.5"])
    engine_module.apply_voice_profile(args, profile)
    assert args.lora == "/other" and args.lora_scale == 0.5


def test_voice_adapter_on_mlx_merges_lokr(engine_module, tmp_path, monkeypatch):
    """On MLX the adapter must be merged into the weights: the PyTorch
    fallback needed >18 GB and froze a 16 GB Mac (watchdog panic)."""
    lokr = tmp_path / "ck"
    lokr.mkdir()
    (lokr / "lokr_weights.safetensors").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(engine_module, "merge_lokr_into_mlx",
                        lambda h, w, s: calls.append((w, s)) or "✅ merged")

    class Mlx:
        use_mlx_dit = True

        def add_voice_lora(self, *a, **k):
            raise AssertionError("PyTorch adapter path must not be used on MLX")

    assert engine_module.load_voice_adapter(Mlx(), str(lokr), 0.7) == "✅ merged"
    assert calls == [(str(lokr / "lokr_weights.safetensors"), 0.7)]
    peft = tmp_path / "peft"
    peft.mkdir()
    assert engine_module.load_voice_adapter(Mlx(), str(peft)).startswith("❌")


def test_voice_adapter_without_mlx_uses_acestep_loader(engine_module):
    class Cuda:
        use_mlx_dit = False

        def add_voice_lora(self, path, scale=1.0):
            return f"✅ {path} {scale}"

    assert engine_module.load_voice_adapter(Cuda(), "/a", 0.5) == "✅ /a 0.5"


def test_lokr_weights_file(engine_module, tmp_path):
    f = tmp_path / "lokr_weights.safetensors"
    f.write_bytes(b"x")
    assert engine_module.lokr_weights_file(str(tmp_path)) == str(f)
    assert engine_module.lokr_weights_file(str(f)) == str(f)
    assert engine_module.lokr_weights_file(str(tmp_path / "nope")) is None
