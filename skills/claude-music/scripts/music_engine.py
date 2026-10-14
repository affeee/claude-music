#!/usr/bin/env python3
"""ACE-Step Music Engine for claude-music skill.

Subcommands:
  generate    Text-to-music generation
  cover       Style transfer using reference audio
  repaint     Selective section regeneration
  extract     Track separation (base model only)
  lego        Add instrument layer (base model only)
  complete    Audio completion (base model only)

Usage:
  python3 music_engine.py generate --caption "upbeat pop" --duration 60
  python3 music_engine.py cover --src-audio song.mp3 --caption "jazz version"
  python3 music_engine.py repaint --src-audio song.mp3 --start 30 --end 60

All structured output is JSON to stdout. Progress/debug info goes to stderr.
"""

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------

QUALITY_PRESETS = {
    "draft": {
        "model": "acestep-v15-turbo",
        "lm_model": None,
        "inference_steps": 8,
        "guidance_scale": 0.0,
        "thinking": False,
        "batch_size": 4,
        "shift": 1.0,
    },
    "standard": {
        "model": "acestep-v15-turbo",
        "lm_model": None,
        "inference_steps": 8,
        "guidance_scale": 0.0,
        "thinking": False,
        "batch_size": 2,
        "shift": 1.0,
    },
    "high": {
        "model": "acestep-v15-turbo",
        "lm_model": "acestep-5Hz-lm-1.7B",
        "inference_steps": 8,
        "guidance_scale": 0.0,
        "thinking": True,
        "batch_size": 2,
        "shift": 1.0,
    },
    "max": {
        "model": "acestep-v15-base",
        "lm_model": "acestep-5Hz-lm-1.7B",
        "inference_steps": 65,
        "guidance_scale": 4.0,
        "thinking": True,
        "batch_size": 1,
        "shift": 6.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    """Print to stderr for progress messages (keeps stdout clean for JSON)."""
    print(msg, file=sys.stderr, flush=True)


def get_free_vram_mb():
    """Return free VRAM in MB, or -1 if not available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            return int(free / (1024 * 1024))
    except Exception:
        pass
    return -1


def emit_event(event, **fields):
    """Print one NDJSON progress event to stderr (only with --progress).

    stderr stays the machine-readable progress channel for wrappers such as
    the web dashboard; stdout remains reserved for the single result JSON.
    """
    with contextlib.suppress(Exception):
        print(json.dumps({"event": event, **fields}), file=sys.stderr, flush=True)


# ACE-Step reports the diffusion phase as this slice of its 0..1 progress, and
# fills it from a wall-clock guess (5s/step until it has timing history). On a
# slow backend that guess runs minutes ahead of the real step counter.
DIFFUSION_PCT_RANGE = (0.52, 0.79)
_diffusion = {"step": 0, "total": 0}


def diffusion_pct(pct):
    """Replace ACE-Step's time-based diffusion estimate with real steps."""
    start, end = DIFFUSION_PCT_RANGE
    if pct is None or not start <= pct < end:
        return pct
    total = _diffusion["total"]
    done = _diffusion["step"] / total if total else 0.0
    return round(start + (end - start) * done, 4)


def is_diffusion_bar(desc, total, expected_steps):
    """Whether a tqdm bar is the diffusion loop. Turbo labels its bar; the base
    model's has no desc at all, so an unlabelled bar of exactly the requested
    step count counts too. Labelled bars (VAE chunks etc.) never do."""
    desc = str(desc or "")
    if "diffusion" in desc.lower():
        return True
    return not desc and bool(expected_steps) and total == expected_steps


def track_diffusion_steps(expected_steps=None):
    """Turn tqdm's diffusion bar into NDJSON 'step' events (--progress only).

    Bars are silenced so they cannot glue themselves to the front of a JSON
    line, which made line-based readers drop the event.
    """
    try:
        from tqdm.std import tqdm
    except Exception:
        return
    if getattr(tqdm, "_claude_music_patched", False):
        return
    orig_init, orig_iter = tqdm.__init__, tqdm.__iter__

    def _init(self, *a, **kw):
        desc = kw.get("desc") or (a[1] if len(a) > 1 else "") or ""
        total = kw.get("total") or (a[2] if len(a) > 2 else None)
        iterable = kw.get("iterable", a[0] if a else None)
        if total is None:
            with contextlib.suppress(Exception):
                total = len(iterable)
        self._cm_steps = (int(total or 0)
                          if is_diffusion_bar(desc, total, expected_steps) else 0)
        kw["disable"] = True
        orig_init(self, *a, **kw)

    def _iter(self):
        total = getattr(self, "_cm_steps", 0)
        if not total:
            yield from orig_iter(self)
            return
        _diffusion.update(step=0, total=total)
        t0 = time.time()
        for step, item in enumerate(orig_iter(self), start=1):
            yield item
            per_step = (time.time() - t0) / step
            _diffusion["step"] = step
            emit_event("step", step=step, total=total,
                       sec_per_step=round(per_step, 2),
                       eta_sec=round(per_step * (total - step)))

    tqdm.__init__, tqdm.__iter__ = _init, _iter
    tqdm._claude_music_patched = True


def make_progress(args):
    """Return a Gradio-style progress callable for generate_music(), or None.

    ACE-Step forwards this to the LM and DiT stages. It must never raise:
    a broken progress line must not kill a generation.
    """
    if not getattr(args, "progress", False):
        return None
    last = {}

    def _progress(frac=None, desc=None, *a, **kw):
        try:
            pct = float(frac) if frac is not None else None
        except (TypeError, ValueError):
            pct = None
        event = {"pct": diffusion_pct(pct), "stage": str(desc) if desc else ""}
        if event != last:  # the estimator ticks twice a second
            last.clear()
            last.update(event)
            emit_event("progress", **event)

    return _progress


def output_json(data):
    """Print JSON result to stdout."""
    print(json.dumps(data, indent=2, default=str))


def error_json(msg, suggestion=None):
    """Print error JSON and exit."""
    result = {"success": False, "error": msg}
    if suggestion:
        result["suggestion"] = suggestion
    output_json(result)
    sys.exit(1)


def load_config():
    """Load config.json. Returns {} if missing or unreadable.

    Never raises: a broken config must not stop --help from working.
    """
    config_path = Path(__file__).parent.parent / "config.json"
    if not config_path.exists():
        return {}
    try:
        # utf-8-sig: install.ps1 writes config.json with a BOM
        with open(config_path, encoding="utf-8-sig") as f:
            cfg = json.load(f)
    except Exception as e:
        log(f"Warning: could not read config.json ({e}), using built-in defaults")
        return {}
    return cfg if isinstance(cfg, dict) else {}


def config_default(defaults, key, fallback, choices=None):
    """Read one key from the config 'defaults' block.

    Falls back to the built-in default when the key is absent, null, or not one
    of `choices`. An invalid value warns rather than crashing argparse, which
    does not validate defaults against choices on its own.
    """
    value = defaults.get(key)
    if value is None:
        return fallback
    if choices is not None and value not in choices:
        log(f"Warning: config.json defaults.{key}={value!r} is not one of "
            f"{sorted(choices)}, using {fallback!r}")
        return fallback
    return value


def slugify(text, max_words=6, max_len=48):
    """Filesystem-safe slug from free text. Returns '' if nothing usable.

    Only [a-z0-9-] survives, so this doubles as the basename sanitizer: a
    caption can never introduce a path separator or shell metacharacter.
    """
    words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    if not words:
        return ""
    return "-".join(words[:max_words])[:max_len].strip("-")


def informative_stem(path):
    """Return the original basename if it carries meaning, else ''.

    ACE-Step names its own output files with UUIDs, which are pure noise. But
    some task types (extract) encode the stem — 'vocals', 'drums' — in that
    name, and that must survive renaming.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    compact = re.sub(r"[^0-9a-zA-Z]", "", stem).lower()
    if not compact:
        return ""
    # UUID / hex blob: long and hex-only, or the 8-4-4-4-12 shape
    if len(compact) >= 16 and re.fullmatch(r"[0-9a-f]+", compact):
        return ""
    if re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", stem.lower()):
        return ""
    return slugify(stem, max_words=3, max_len=24)


def rename_outputs(args, outputs, label="", task_type="track"):
    """Rename pipeline-written files to readable names, in place.

    ACE-Step's pipeline chooses its own filenames (UUIDs), so this runs after
    generation rather than passing a path in. Never overwrites an existing
    file: on collision it appends -2, -3, ... A failed rename is logged and
    the original path kept, so a permissions problem cannot lose a track.

    Result: lofi-afro-latin-percussion_20260812-1530_01_s3128607774.flac
    The seed is embedded because it is the one field you cannot recover from
    the file later, and it is what makes a track reproducible.
    """
    if getattr(args, "naming", "descriptive") != "descriptive":
        return outputs

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    slug = slugify(label) or slugify(task_type) or "track"

    for out in outputs:
        old = out.get("path")
        if not old or not os.path.isfile(old):
            continue

        directory = os.path.dirname(old)
        ext = os.path.splitext(old)[1]
        parts = [slug, stamp, f"{out.get('index', 1):02d}"]

        kept = informative_stem(old)
        if kept:
            parts.append(kept)

        seed = out.get("seed")
        if isinstance(seed, int) and seed >= 0:
            parts.append(f"s{seed}")

        base = "_".join(parts)
        new = os.path.join(directory, base + ext)
        counter = 2
        while os.path.exists(new):
            new = os.path.join(directory, f"{base}-{counter}{ext}")
            counter += 1

        try:
            os.rename(old, new)
            out["path"] = new
        except OSError as e:
            log(f"Warning: could not rename {os.path.basename(old)} ({e}), keeping original name")

    return outputs


def resolve_quality(args):
    """Resolve generation params with precedence: CLI flag > config.json > preset.

    The preset owns model, lm_model, batch_size and thinking, so config.json
    ships without those keys and the presets stay intact out of the box. A user
    who adds one to defaults pins it across every preset.
    """
    preset = QUALITY_PRESETS.get(args.quality, QUALITY_PRESETS["standard"])
    defaults = getattr(args, "_config_defaults", None) or {}

    def pick(cli_value, key):
        """CLI wins if set; then config.json; then the quality preset."""
        if cli_value is not None:
            return cli_value
        if defaults.get(key) is not None:
            return defaults[key]
        return preset[key]

    if args.model is None:
        args.model = pick(None, "model")
    if args.lm_model is None and not hasattr(args, "_lm_model_set"):
        args.lm_model = pick(None, "lm_model")
    if args.inference_steps is None:
        args.inference_steps = preset["inference_steps"]
    if args.guidance_scale is None:
        args.guidance_scale = preset["guidance_scale"]
    if args.batch is None:
        # config.json calls this batch_size; the CLI flag is --batch
        if defaults.get("batch_size") is not None:
            args.batch = defaults["batch_size"]
        else:
            args.batch = preset["batch_size"]
            args._batch_from_preset = True
    if args.shift is None:
        args.shift = preset["shift"]
    if not hasattr(args, "thinking") or args.thinking is None:
        args.thinking = pick(None, "thinking")

    return apply_memory_policy(args)


def machine_profile():
    """Memory layout of this machine, without importing torch.

    Apple Silicon shares one pool between CPU and GPU, so the VRAM tiers the
    presets were tuned for (a 16GB card next to system RAM) do not apply.
    """
    memory_gb = 0.0
    with contextlib.suppress(Exception):
        memory_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    unified = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {"unified_memory": unified, "memory_gb": round(memory_gb, 1)}


def source_duration(path):
    """Length of an audio file in seconds via ffprobe, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            check=True, capture_output=True, text=True, timeout=30).stdout
        return float(out.strip())
    except Exception:
        return None


# Seconds of audio (batch x duration) a 16GB unified-memory machine holds with
# every model resident, i.e. without CPU offload. Measured on an M1/16GB with
# a 182s track: one variant fits, two run out of MPS memory. Scales with RAM.
# The runs behind this are in references/apple-silicon.md.
UNIFIED_AUDIO_BUDGET_SEC = 240.0


def unified_audio_budget(profile):
    return UNIFIED_AUDIO_BUDGET_SEC * profile["memory_gb"] / 16.0


def memory_safe_batch(batch, duration, profile=None):
    """Cap a preset's batch so batch x duration fits unified memory."""
    profile = profile or machine_profile()
    if not profile["unified_memory"] or not profile["memory_gb"]:
        return batch
    if not duration or duration <= 0:
        return 1  # unknown length: assume a full track
    return max(1, min(batch, int(unified_audio_budget(profile) // duration)))


def lm_backend(profile=None):
    """LM backend to ask ACE-Step for. vllm needs CUDA; on Apple Silicon the
    native MLX backend is what ACE-Step itself recommends, and the PyTorch
    one ran a 16GB machine out of memory."""
    profile = profile or machine_profile()
    return "mlx" if profile["unified_memory"] else "vllm"


def fit_lm_to_memory(lm_model, allowed, installed, profile=None):
    """LM that fits this machine: (model or None, note for the log).

    `allowed` is ACE-Step's own list for the detected memory tier. Only
    unified memory is held to it: there the 1.7B LM ran a 16GB machine out of
    memory in every configuration tried, where a 16GB card runs it fine.
    """
    profile = profile or machine_profile()
    if not lm_model or not profile["unified_memory"] or not allowed:
        return lm_model, ""
    if lm_model in allowed:
        return lm_model, ""
    for candidate in allowed:
        if candidate in installed:
            return candidate, (f"{lm_model} does not fit {profile['memory_gb']:.0f} GB "
                               f"unified memory, using {candidate}")
    return None, (f"{lm_model} does not fit {profile['memory_gb']:.0f} GB unified memory; "
                  f"continuing without thinking mode. ACE-Step suggests {allowed[0]} here: "
                  f"uv run acestep-download --model {allowed[0]} --skip-main")


def lm_models_for_this_machine():
    """ACE-Step's LM list for the detected memory tier, or [] if unknown."""
    try:
        from acestep.gpu_config import get_gpu_config
        return list(get_gpu_config().available_lm_models or [])
    except Exception:
        return []


def task_uses_lm(args):
    """ACE-Step ignores the LM for these tasks, so loading it only costs memory."""
    return getattr(args, "command", None) not in ("cover", "repaint", "extract")


def wants_lm(args):
    return bool(args.lm_model and args.thinking and task_uses_lm(args))


def apply_memory_policy(args):
    """Shrink a preset-chosen batch on unified-memory machines.

    An explicit --batch or config.json batch_size is left alone.
    """
    duration = getattr(args, "duration", None)
    if (not duration or duration <= 0) and getattr(args, "src_audio", None):
        duration = source_duration(args.src_audio)
    args._duration_sec = duration
    if not getattr(args, "_batch_from_preset", False):
        return args
    safe = memory_safe_batch(args.batch, duration)
    if safe < args.batch:
        log(f"Batch {args.batch} -> {safe}: unified memory "
            f"({machine_profile()['memory_gb']:.0f} GB). Pass --batch to override.")
        args.batch = safe
    return args


def auto_offload(args, profile=None):
    """Whether idle models should be moved to CPU when nobody chose.

    A discrete GPU has its own memory, so offloading frees VRAM. On unified
    memory it only shuffles within one pool and costs about a quarter of the
    run, so it stays off until the job holds more audio than fits with
    everything loaded.
    """
    profile = profile or machine_profile()
    if not profile["unified_memory"] or not profile["memory_gb"]:
        return True
    duration = getattr(args, "_duration_sec", None)
    if not duration or duration <= 0:
        return (args.batch or 1) > 1
    return (args.batch or 1) * duration > unified_audio_budget(profile)


def resolve_offload(args, defaults, key):
    """CLI --offload > a true/false in config.json > auto_offload()."""
    choice = getattr(args, "offload", None) or "auto"
    if choice != "auto":
        return choice == "on"
    value = defaults.get(key)
    if isinstance(value, bool):
        return value
    return auto_offload(args)


def resolve_base_quality(args, task):
    """resolve_quality() for the base-model-only tasks (extract/lego/complete).

    These switch a turbo preset to the base model. Turbo's 8 steps / no CFG
    would starve it, so whatever the CLI left unset comes from the max preset.
    """
    cli = {k: getattr(args, k, None)
           for k in ("inference_steps", "guidance_scale", "shift")}
    args = resolve_quality(args)
    if "turbo" in (args.model or ""):
        args.model = QUALITY_PRESETS["max"]["model"]
        for key, value in cli.items():
            if value is None:
                setattr(args, key, QUALITY_PRESETS["max"][key])
        log(f"Switched to base model ({task} requires base)")
    return args


TRACK_NAMES = (
    "vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard",
    "percussion", "strings", "synth", "fx", "brass", "woodwinds",
)

# Mirrors acestep.constants.TASK_INSTRUCTIONS. ACE-Step picks the track from
# the instruction text, and the Python API does not derive it from task_type.
TASK_INSTRUCTIONS = {
    "extract": ("Extract the {track} track from the audio:",
                "Extract the track from the audio:"),
    "lego": ("Generate the {track} track based on the audio context:",
             "Generate the track based on the audio context:"),
    "complete": ("Complete the input track with {track}:",
                 "Complete the input track:"),
}


def task_instruction(task, track=None):
    """Instruction prompt for a base-model task, naming the track if given."""
    named, default = TASK_INSTRUCTIONS[task]
    return named.format(track=track.upper()) if track else default


def build_mix_command(src_audio, layer, out_path, layer_gain_db=0.0,
                      start=0.0, end=-1.0):
    """ffmpeg argv that lays a generated layer over the untouched source.

    Both inputs drop 3 dB before summing and a limiter catches what is left,
    so the mix cannot clip. -n keeps the no-overwrite guarantee.

    Outside --start/--end ACE-Step returns the source itself, not silence, so
    the layer is faded to nothing there or the source would be doubled.
    """
    fades = ""
    if start and start > 0:
        fades += f",afade=t=in:st={max(0.0, start - 0.05):.2f}:d=0.05"
    if end and end > 0:
        fades += f",afade=t=out:st={end:.2f}:d=0.05"
    graph = (
        "[0:a]aresample=48000,volume=-3dB[src];"
        f"[1:a]aresample=48000,volume={layer_gain_db - 3.0:.1f}dB{fades}[layer];"
        "[src][layer]amix=inputs=2:normalize=0:duration=first,"
        "alimiter=limit=0.89:level=false[mix]"
    )
    return ["ffmpeg", "-n", "-v", "error", "-i", src_audio, "-i", layer,
            "-filter_complex", graph, "-map", "[mix]", out_path]


def mix_with_source(src_audio, layer, layer_gain_db=0.0, start=0.0, end=-1.0):
    """Mix a lego layer with its source. Returns the mix path, or None."""
    if not shutil.which("ffmpeg"):
        log("Warning: ffmpeg not found, skipping --mix")
        return None
    stem, ext = os.path.splitext(layer)
    out_path = f"{stem}_mix{ext}"
    counter = 2
    while os.path.exists(out_path):
        out_path = f"{stem}_mix-{counter}{ext}"
        counter += 1
    try:
        subprocess.run(build_mix_command(src_audio, layer, out_path, layer_gain_db,
                                         start, end),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True)
    except (OSError, subprocess.CalledProcessError) as e:
        log(f"Warning: mix failed ({getattr(e, 'stderr', '') or e}), keeping the layer only")
        return None
    return out_path


# ---------------------------------------------------------------------------
# ACE-Step initialization
# ---------------------------------------------------------------------------

def initialize_acestep(args):
    """Load ACE-Step models. Returns (handler, llm_handler, status_msg)."""
    project_root = Path(args.ace_step_dir)
    sys.path.insert(0, str(project_root))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHAUDIO_USE_BACKEND"] = "ffmpeg"
    import torch
    cpu_only = not torch.cuda.is_available() and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    # ACE-Step aborts a generation after 600s, model load included, reading
    # the limit at import. The base model on a full-length track needs longer
    # than that off a big GPU, and hours on a CPU alone.
    os.environ.setdefault("ACESTEP_GENERATION_TIMEOUT",
                          "21600" if cpu_only else "3600")
    if cpu_only:
        # torch uses one thread per physical core; adding the hyperthreads
        # ran ~10% faster on a 4-core i5-1135G7
        torch.set_num_threads(os.cpu_count() or torch.get_num_threads())
        log(f"CPU only: {torch.get_num_threads()} threads, "
            f"{os.environ['ACESTEP_GENERATION_TIMEOUT']}s generation timeout")

    from acestep.handler import AceStepHandler
    if getattr(args, "progress", False):
        track_diffusion_steps(getattr(args, "inference_steps", None))
        emit_event("stage", stage="loading_model")
    log(f"Loading DiT model: {args.model}...")

    # Memory/attention settings come from config.json so 16GB-VRAM cards can
    # keep CPU offload on while larger cards can turn it off for speed.
    defaults = getattr(args, "_config_defaults", None) or {}
    use_flash = config_default(defaults, "use_flash_attention", True)
    offload = resolve_offload(args, defaults, "offload_to_cpu")
    offload_dit = resolve_offload(args, defaults, "offload_dit_to_cpu")
    log(f"CPU offload: {'on' if offload else 'off'}")

    handler = AceStepHandler()
    t0 = time.time()
    status, success = handler.initialize_service(
        project_root=str(project_root),
        config_path=args.model,
        device="auto",
        use_flash_attention=use_flash,
        offload_to_cpu=offload,
        offload_dit_to_cpu=offload_dit,
    )

    if not success:
        return None, None, f"DiT init failed: {status}"

    load_time = time.time() - t0
    log(f"DiT ready in {load_time:.1f}s: {status}")

    llm_handler = None
    if args.lm_model and args.thinking and not task_uses_lm(args):
        log(f"Skipping LM: ACE-Step does not use it for {args.command}")
    elif wants_lm(args):
        checkpoints = project_root / "checkpoints"
        installed = [d.name for d in checkpoints.iterdir()] if checkpoints.is_dir() else []
        args.lm_model, note = fit_lm_to_memory(
            args.lm_model, lm_models_for_this_machine(), installed)
        if note:
            log(f"Warning: {note}")
    if wants_lm(args):
        log(f"Loading LM: {args.lm_model}...")
        t1 = time.time()
        try:
            from acestep.llm_inference import LLMHandler
            llm_handler = LLMHandler()

            backend = lm_backend()
            if backend == "vllm":
                # Try vllm first, fall back to PyTorch native
                try:
                    import vllm  # noqa: F401
                except ImportError:
                    backend = "pt"
                    log("vllm not available, using PyTorch backend for LM")

            lm_status, lm_success = llm_handler.initialize(
                checkpoint_dir=str(project_root / "checkpoints"),
                lm_model_path=args.lm_model,
                backend=backend,
                device="auto",
                offload_to_cpu=offload,
            )
            if not lm_success:
                log(f"LM init failed: {lm_status}, continuing without thinking mode")
                llm_handler = None
            else:
                log(f"LM ready in {time.time() - t1:.1f}s ({backend} backend)")
        except Exception as e:
            log(f"LM load failed ({e}), continuing without thinking mode")
            llm_handler = None

    if getattr(args, "progress", False):
        emit_event("stage", stage="generating")
    return handler, llm_handler, status


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    """Text-to-music generation."""
    args = resolve_quality(args)

    if not args.caption and not args.lyrics:
        error_json(
            "Must provide --caption and/or --lyrics",
            "Example: music_engine.py generate --caption 'upbeat pop' --lyrics '[Verse] Hello world'"
        )

    # Validate input lengths (ACE-Step limits)
    if args.caption and len(args.caption) > 512:
        log(f"WARNING: Caption truncated from {len(args.caption)} to 512 chars")
        args.caption = args.caption[:512]
    if args.lyrics and len(args.lyrics) > 4096:
        log(f"WARNING: Lyrics truncated from {len(args.lyrics)} to 4096 chars")
        args.lyrics = args.lyrics[:4096]

    handler, llm_handler, status = initialize_acestep(args)
    if handler is None:
        error_json(status)

    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    params = GenerationParams(
        caption=args.caption or "",
        lyrics=args.lyrics or "",
        instrumental=args.instrumental,
        vocal_language=args.language,
        bpm=args.bpm,
        keyscale=args.key or "",
        timesignature=args.time_sig or "",
        duration=args.duration if args.duration and args.duration > 0 else -1.0,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        shift=args.shift,
        task_type="text2music",
        thinking=args.thinking if llm_handler else False,
    )

    config = GenerationConfig(
        batch_size=args.batch,
        use_random_seed=(args.seed < 0),
        audio_format=args.format,
    )

    save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    log(f"Generating {args.batch} variant(s), {args.duration}s, {args.model}...")
    t0 = time.time()
    result = generate_music(
        dit_handler=handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=save_dir,
        progress=make_progress(args),
    )
    gen_time = time.time() - t0

    if not result or not result.success:
        err = result.error if result else "Unknown generation error"
        error_json(err)

    outputs = []
    for i, audio in enumerate(result.audios or []):
        path = None
        for key in ("path", "audio_path", "save_path", "file_path"):
            if key in audio and audio[key]:
                path = audio[key]
                break
        if not path:
            for v in audio.values():
                if isinstance(v, str) and os.path.isfile(v):
                    path = v
                    break

        if path and os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            seed = audio.get("params", {}).get("seed", -1)
            outputs.append({
                "path": path,
                "seed": seed,
                "duration_sec": args.duration if args.duration and args.duration > 0 else "auto",
                "size_mb": round(size_mb, 2),
                "index": i + 1,
            })

    outputs = rename_outputs(args, outputs,
                             label=args.caption or args.lyrics, task_type="text2music")

    output_json({
        "success": True,
        "task_type": "text2music",
        "model": args.model,
        "lm_model": args.lm_model if llm_handler else None,
        "quality": args.quality,
        "outputs": outputs,
        "params": {
            "caption": args.caption,
            "lyrics": args.lyrics[:200] + "..." if args.lyrics and len(args.lyrics) > 200 else args.lyrics,
            "bpm": args.bpm,
            "key": args.key,
            "duration": args.duration,
            "instrumental": args.instrumental,
            "language": args.language,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "thinking": args.thinking if llm_handler else False,
        },
        "timing": {
            "generation_sec": round(gen_time, 1),
        },
        "count": len(outputs),
        "playback_hint": f"ffplay -nodisp -autoexit \"{outputs[0]['path']}\"" if outputs else None,
    })


# ---------------------------------------------------------------------------
# Subcommand: cover
# ---------------------------------------------------------------------------

def cmd_cover(args):
    """Style transfer / cover generation from reference audio."""
    args = resolve_quality(args)

    if not args.src_audio:
        error_json("Must provide --src-audio for cover mode")
    if not os.path.isfile(args.src_audio):
        error_json(f"Source audio not found: {args.src_audio}")

    handler, llm_handler, status = initialize_acestep(args)
    if handler is None:
        error_json(status)

    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    params = GenerationParams(
        caption=args.caption or "",
        lyrics=args.lyrics or "",
        instrumental=args.instrumental,
        vocal_language=args.language,
        bpm=args.bpm,
        keyscale=args.key or "",
        timesignature=args.time_sig or "",
        duration=args.duration if args.duration and args.duration > 0 else -1.0,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        shift=args.shift,
        task_type="cover",
        src_audio=args.src_audio,
        cover_noise_strength=args.cover_strength,
        thinking=args.thinking if llm_handler else False,
    )

    config = GenerationConfig(
        batch_size=args.batch,
        use_random_seed=(args.seed < 0),
        audio_format=args.format,
    )

    save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    log(f"Generating cover (strength={args.cover_strength})...")
    t0 = time.time()
    result = generate_music(
        dit_handler=handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=save_dir,
        progress=make_progress(args),
    )
    gen_time = time.time() - t0

    if not result or not result.success:
        error_json(result.error if result else "Cover generation failed")

    outputs = []
    for i, audio in enumerate(result.audios or []):
        path = None
        for key in ("path", "audio_path", "save_path", "file_path"):
            if key in audio and audio[key]:
                path = audio[key]
                break
        if not path:
            for v in audio.values():
                if isinstance(v, str) and os.path.isfile(v):
                    path = v
                    break
        if path and os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            seed = audio.get("params", {}).get("seed", -1)
            outputs.append({
                "path": path, "seed": seed,
                "size_mb": round(size_mb, 2), "index": i + 1,
            })

    outputs = rename_outputs(
        args, outputs,
        label=args.caption or informative_stem(args.src_audio) or "cover",
        task_type="cover")

    output_json({
        "success": True,
        "task_type": "cover",
        "model": args.model,
        "outputs": outputs,
        "params": {
            "src_audio": args.src_audio,
            "cover_strength": args.cover_strength,
            "caption": args.caption,
        },
        "timing": {"generation_sec": round(gen_time, 1)},
        "count": len(outputs),
        "playback_hint": f"ffplay -nodisp -autoexit \"{outputs[0]['path']}\"" if outputs else None,
    })


# ---------------------------------------------------------------------------
# Subcommand: repaint
# ---------------------------------------------------------------------------

def cmd_repaint(args):
    """Selective section regeneration."""
    args = resolve_quality(args)

    if not args.src_audio:
        error_json("Must provide --src-audio for repaint mode")
    if not os.path.isfile(args.src_audio):
        error_json(f"Source audio not found: {args.src_audio}")

    handler, llm_handler, status = initialize_acestep(args)
    if handler is None:
        error_json(status)

    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    params = GenerationParams(
        caption=args.caption or "",
        lyrics=args.lyrics or "",
        instrumental=args.instrumental,
        vocal_language=args.language,
        bpm=args.bpm,
        keyscale=args.key or "",
        duration=args.duration if args.duration and args.duration > 0 else -1.0,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        shift=args.shift,
        task_type="repaint",
        src_audio=args.src_audio,
        repainting_start=args.start,
        repainting_end=args.end,
        thinking=args.thinking if llm_handler else False,
    )

    config = GenerationConfig(
        batch_size=args.batch,
        use_random_seed=(args.seed < 0),
        audio_format=args.format,
    )

    save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    log(f"Repainting {args.start}s-{args.end}s...")
    t0 = time.time()
    result = generate_music(
        dit_handler=handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=save_dir,
        progress=make_progress(args),
    )
    gen_time = time.time() - t0

    if not result or not result.success:
        error_json(result.error if result else "Repaint failed")

    outputs = []
    for i, audio in enumerate(result.audios or []):
        path = None
        for key in ("path", "audio_path", "save_path", "file_path"):
            if key in audio and audio[key]:
                path = audio[key]
                break
        if not path:
            for v in audio.values():
                if isinstance(v, str) and os.path.isfile(v):
                    path = v
                    break
        if path and os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            seed = audio.get("params", {}).get("seed", -1)
            outputs.append({
                "path": path, "seed": seed,
                "size_mb": round(size_mb, 2), "index": i + 1,
            })

    outputs = rename_outputs(
        args, outputs,
        label=args.caption or informative_stem(args.src_audio) or "repaint",
        task_type="repaint")

    output_json({
        "success": True,
        "task_type": "repaint",
        "model": args.model,
        "outputs": outputs,
        "params": {
            "src_audio": args.src_audio,
            "start": args.start, "end": args.end,
            "caption": args.caption,
        },
        "timing": {"generation_sec": round(gen_time, 1)},
        "count": len(outputs),
        "playback_hint": f"ffplay -nodisp -autoexit \"{outputs[0]['path']}\"" if outputs else None,
    })


# ---------------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------------

def cmd_extract(args):
    """Track separation (base model only)."""
    args = resolve_base_quality(args, "extract")

    if not args.src_audio:
        error_json("Must provide --src-audio for extract mode")
    if not os.path.isfile(args.src_audio):
        error_json(f"Source audio not found: {args.src_audio}")

    handler, llm_handler, status = initialize_acestep(args)
    if handler is None:
        error_json(status)

    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    params = GenerationParams(
        caption=args.caption or "",
        task_type="extract",
        instruction=task_instruction("extract", args.track),
        src_audio=args.src_audio,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        shift=args.shift,
    )

    config = GenerationConfig(
        batch_size=1,
        audio_format=args.format,
    )

    save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    log("Extracting tracks...")
    t0 = time.time()
    result = generate_music(
        dit_handler=handler, llm_handler=None,
        params=params, config=config, save_dir=save_dir,
        progress=make_progress(args),
    )
    gen_time = time.time() - t0

    if not result or not result.success:
        error_json(result.error if result else "Extract failed")

    outputs = []
    for i, audio in enumerate(result.audios or []):
        path = None
        for key in ("path", "audio_path", "save_path", "file_path"):
            if key in audio and audio[key]:
                path = audio[key]
                break
        if path and os.path.isfile(path):
            outputs.append({"path": path, "index": i + 1})

    outputs = rename_outputs(
        args, outputs,
        label=informative_stem(args.src_audio) or "extract",
        task_type="extract")

    output_json({
        "success": True, "task_type": "extract",
        "outputs": outputs,
        "timing": {"generation_sec": round(gen_time, 1)},
        "count": len(outputs),
    })


# ---------------------------------------------------------------------------
# Subcommand: lego
# ---------------------------------------------------------------------------

def cmd_lego(args):
    """Multi-track / add layer (base model only)."""
    args = resolve_base_quality(args, "lego")

    if not args.src_audio:
        error_json("Must provide --src-audio for lego mode")
    if not os.path.isfile(args.src_audio):
        error_json(f"Source audio not found: {args.src_audio}")

    handler, llm_handler, status = initialize_acestep(args)
    if handler is None:
        error_json(status)

    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    params = GenerationParams(
        caption=args.caption or "",
        lyrics=args.lyrics or "",
        vocal_language=args.language,
        bpm=args.bpm,
        keyscale=args.key or "",
        task_type="lego",
        instruction=task_instruction("lego", args.track),
        src_audio=args.src_audio,
        repainting_start=args.start,
        repainting_end=args.end,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        shift=args.shift,
        thinking=args.thinking if llm_handler else False,
    )

    config = GenerationConfig(batch_size=1, audio_format=args.format)

    save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    log("Adding layer...")
    t0 = time.time()
    result = generate_music(
        dit_handler=handler, llm_handler=llm_handler,
        params=params, config=config, save_dir=save_dir,
        progress=make_progress(args),
    )
    gen_time = time.time() - t0

    if not result or not result.success:
        error_json(result.error if result else "Lego failed")

    outputs = []
    for i, audio in enumerate(result.audios or []):
        path = None
        for key in ("path", "audio_path", "save_path", "file_path"):
            if key in audio and audio[key]:
                path = audio[key]
                break
        if path and os.path.isfile(path):
            outputs.append({"path": path, "index": i + 1})

    outputs = rename_outputs(
        args, outputs,
        label="_".join(filter(None, [informative_stem(args.src_audio) or "lego",
                                     args.track])),
        task_type="lego")

    # lego returns the new layer alone, not the source with the layer on top
    if args.mix:
        for out in outputs:
            mix_path = mix_with_source(args.src_audio, out["path"], args.mix_gain,
                                       args.start, args.end)
            if mix_path:
                out["mix_path"] = mix_path

    output_json({
        "success": True, "task_type": "lego",
        "track": args.track,
        "outputs": outputs,
        "timing": {"generation_sec": round(gen_time, 1)},
        "count": len(outputs),
    })


# ---------------------------------------------------------------------------
# Subcommand: complete
# ---------------------------------------------------------------------------

def cmd_complete(args):
    """Audio completion (base model only)."""
    args = resolve_base_quality(args, "complete")

    if not args.src_audio:
        error_json("Must provide --src-audio for complete mode")
    if not os.path.isfile(args.src_audio):
        error_json(f"Source audio not found: {args.src_audio}")

    handler, llm_handler, status = initialize_acestep(args)
    if handler is None:
        error_json(status)

    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    params = GenerationParams(
        caption=args.caption or "",
        lyrics=args.lyrics or "",
        task_type="complete",
        instruction=task_instruction("complete"),
        src_audio=args.src_audio,
        duration=args.duration if args.duration and args.duration > 0 else -1.0,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        shift=args.shift,
    )

    config = GenerationConfig(batch_size=1, audio_format=args.format)

    save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    log("Completing audio...")
    t0 = time.time()
    result = generate_music(
        dit_handler=handler, llm_handler=llm_handler,
        params=params, config=config, save_dir=save_dir,
        progress=make_progress(args),
    )
    gen_time = time.time() - t0

    if not result or not result.success:
        error_json(result.error if result else "Complete failed")

    outputs = []
    for i, audio in enumerate(result.audios or []):
        path = None
        for key in ("path", "audio_path", "save_path", "file_path"):
            if key in audio and audio[key]:
                path = audio[key]
                break
        if path and os.path.isfile(path):
            outputs.append({"path": path, "index": i + 1})

    outputs = rename_outputs(
        args, outputs,
        label=informative_stem(args.src_audio) or "complete",
        task_type="complete")

    output_json({
        "success": True, "task_type": "complete",
        "outputs": outputs,
        "timing": {"generation_sec": round(gen_time, 1)},
        "count": len(outputs),
    })


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="ACE-Step Music Engine for claude-music skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    # Read default from config.json if available. Do NOT exit on a missing/
    # unconfigured path here — --help must always work. Validation happens in
    # main() once a subcommand is selected.
    _cfg = load_config()
    _defaults = _cfg.get("defaults") or {}
    if not isinstance(_defaults, dict):
        _defaults = {}

    default_ace_dir = _cfg.get("ace_step_dir") or None
    if default_ace_dir == "CHANGE_ME":
        default_ace_dir = None

    quality_choices = ["draft", "standard", "high", "max"]
    format_choices = ["flac", "wav", "mp3", "wav32", "opus", "aac"]
    _lang = config_default(_defaults, "language", "en")

    parser.add_argument("--ace-step-dir", default=default_ace_dir,
                        help="ACE-Step installation directory")
    parser.add_argument("--model", default=None,
                        help="DiT model variant (acestep-v15-turbo, acestep-v15-base, acestep-v15-xl-turbo, etc.)")
    parser.add_argument("--lm-model", default=None,
                        help="LM model (acestep-5Hz-lm-0.6B, acestep-5Hz-lm-1.7B, acestep-5Hz-lm-4B, or none)")
    parser.add_argument("--quality",
                        default=config_default(_defaults, "quality", "standard", quality_choices),
                        choices=quality_choices,
                        help="Quality preset (overridable by specific params)")
    parser.add_argument("--format",
                        default=config_default(_defaults, "format", "flac", format_choices),
                        choices=format_choices,
                        help="Output audio format")
    parser.add_argument("--naming",
                        default=config_default(_defaults, "naming", "descriptive",
                                               ["descriptive", "uuid"]),
                        choices=["descriptive", "uuid"],
                        help="Output filenames: 'descriptive' (caption_date_index_seed) "
                             "or 'uuid' (leave ACE-Step's raw filenames)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (number of variants)")
    parser.add_argument("--progress", action="store_true",
                        help="Emit NDJSON progress events to stderr (for wrappers)")
    parser.add_argument("--offload", choices=["auto", "on", "off"], default=None,
                        help="Move idle models to CPU. auto: config.json, else on "
                             "for discrete GPUs and off for unified memory")
    parser.add_argument("--output", default=None, help="Output file path (for single output)")
    parser.add_argument("--output-dir",
                        default=config_default(_cfg, "output_dir", "~/Music/claude-music-output"),
                        help="Output directory for generated files")

    # Preset-owned keys (model, lm_model, batch_size, thinking) are resolved in
    # resolve_quality(), which needs the config block at that point.
    parser.set_defaults(_config_defaults=_defaults)

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Text-to-music generation")
    p_gen.add_argument("--caption", "-c", default="", help="Music description/style tags")
    p_gen.add_argument("--lyrics", "-l", default="", help="Lyrics with structure tags")
    p_gen.add_argument("--instrumental", action="store_true", help="Instrumental only (no vocals)")
    p_gen.add_argument("--language", default=_lang, help="Vocal language code (en, zh, ja, etc.)")
    p_gen.add_argument("--bpm", type=int, default=None, help="Beats per minute (30-300)")
    p_gen.add_argument("--key", default=None, help="Musical key (e.g., 'C major', 'Am')")
    p_gen.add_argument("--time-sig", default=None, help="Time signature (2, 3, 4, 6)")
    p_gen.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (10-600)")
    p_gen.add_argument("--inference-steps", type=int, default=None, help="Diffusion steps")
    p_gen.add_argument("--guidance-scale", type=float, default=None, help="CFG guidance scale")
    p_gen.add_argument("--shift", type=float, default=None, help="Timestep shift factor")
    p_gen.add_argument("--thinking", action="store_true", default=None, help="Enable LM thinking mode")
    p_gen.add_argument("--no-thinking", dest="thinking", action="store_false", help="Disable LM thinking")
    p_gen.set_defaults(func=cmd_generate)

    # --- cover ---
    p_cov = sub.add_parser("cover", help="Cover/style transfer from reference audio")
    p_cov.add_argument("--src-audio", required=True, help="Source audio file")
    p_cov.add_argument("--caption", "-c", default="", help="New style description")
    p_cov.add_argument("--lyrics", "-l", default="", help="New lyrics (optional)")
    p_cov.add_argument("--cover-strength", type=float, default=0.5,
                        help="Reference fidelity (0.0=reimagine, 1.0=faithful)")
    p_cov.add_argument("--instrumental", action="store_true")
    p_cov.add_argument("--language", default=_lang)
    p_cov.add_argument("--bpm", type=int, default=None)
    p_cov.add_argument("--key", default=None)
    p_cov.add_argument("--time-sig", default=None)
    p_cov.add_argument("--duration", type=float, default=-1.0)
    p_cov.add_argument("--inference-steps", type=int, default=None)
    p_cov.add_argument("--guidance-scale", type=float, default=None)
    p_cov.add_argument("--shift", type=float, default=None)
    p_cov.add_argument("--thinking", action="store_true", default=None)
    p_cov.add_argument("--no-thinking", dest="thinking", action="store_false")
    p_cov.set_defaults(func=cmd_cover)

    # --- repaint ---
    p_rep = sub.add_parser("repaint", help="Selective section regeneration")
    p_rep.add_argument("--src-audio", required=True, help="Source audio file")
    p_rep.add_argument("--start", type=float, default=0.0, help="Repaint start (seconds)")
    p_rep.add_argument("--end", type=float, default=-1.0, help="Repaint end (seconds, -1=end of file)")
    p_rep.add_argument("--caption", "-c", default="", help="Style description for repainted section")
    p_rep.add_argument("--lyrics", "-l", default="")
    p_rep.add_argument("--instrumental", action="store_true")
    p_rep.add_argument("--language", default=_lang)
    p_rep.add_argument("--bpm", type=int, default=None)
    p_rep.add_argument("--key", default=None)
    p_rep.add_argument("--duration", type=float, default=-1.0)
    p_rep.add_argument("--inference-steps", type=int, default=None)
    p_rep.add_argument("--guidance-scale", type=float, default=None)
    p_rep.add_argument("--shift", type=float, default=None)
    p_rep.add_argument("--thinking", action="store_true", default=None)
    p_rep.add_argument("--no-thinking", dest="thinking", action="store_false")
    p_rep.set_defaults(func=cmd_repaint)

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Track separation (base model only)")
    p_ext.add_argument("--src-audio", required=True)
    p_ext.add_argument("--caption", "-c", default="")
    p_ext.add_argument("--track", choices=TRACK_NAMES, default=None,
                       help="Which track to extract (e.g. vocals, drums, bass)")
    p_ext.add_argument("--inference-steps", type=int, default=None)
    p_ext.add_argument("--guidance-scale", type=float, default=None)
    p_ext.add_argument("--shift", type=float, default=None)
    p_ext.add_argument("--thinking", action="store_true", default=None)
    p_ext.add_argument("--no-thinking", dest="thinking", action="store_false")
    # extract doesn't need lyrics/bpm/key/etc, set defaults
    p_ext.set_defaults(func=cmd_extract, lyrics="", instrumental=False, language="en",
                       bpm=None, key=None, time_sig=None, duration=-1.0)

    # --- lego ---
    p_lego = sub.add_parser("lego", help="Add instrument layer (base model only)")
    p_lego.add_argument("--src-audio", required=True)
    p_lego.add_argument("--caption", "-c", default="", help="Description of layer to add")
    p_lego.add_argument("--lyrics", "-l", default="")
    p_lego.add_argument("--track", choices=TRACK_NAMES, default=None,
                        help="Which layer to add (e.g. vocals, drums, bass)")
    p_lego.add_argument("--start", type=float, default=0.0,
                        help="Where the new layer begins (seconds)")
    p_lego.add_argument("--end", type=float, default=-1.0,
                        help="Where the new layer ends (seconds, -1=end of file)")
    p_lego.add_argument("--mix", action="store_true",
                        help="Also write the layer mixed over the source (<name>_mix)")
    p_lego.add_argument("--mix-gain", type=float, default=0.0,
                        help="Layer level in the mix, dB relative to the source")
    p_lego.add_argument("--language", default=_lang)
    p_lego.add_argument("--bpm", type=int, default=None)
    p_lego.add_argument("--key", default=None)
    p_lego.add_argument("--inference-steps", type=int, default=None)
    p_lego.add_argument("--guidance-scale", type=float, default=None)
    p_lego.add_argument("--shift", type=float, default=None)
    p_lego.add_argument("--thinking", action="store_true", default=None)
    p_lego.add_argument("--no-thinking", dest="thinking", action="store_false")
    p_lego.set_defaults(func=cmd_lego, instrumental=False,
                        time_sig=None, duration=-1.0)

    # --- complete ---
    p_comp = sub.add_parser("complete", help="Audio completion (base model only)")
    p_comp.add_argument("--src-audio", required=True)
    p_comp.add_argument("--caption", "-c", default="")
    p_comp.add_argument("--lyrics", "-l", default="")
    p_comp.add_argument("--duration", type=float, default=-1.0, help="Target total duration")
    p_comp.add_argument("--inference-steps", type=int, default=None)
    p_comp.add_argument("--guidance-scale", type=float, default=None)
    p_comp.add_argument("--shift", type=float, default=None)
    p_comp.add_argument("--thinking", action="store_true", default=None)
    p_comp.add_argument("--no-thinking", dest="thinking", action="store_false")
    p_comp.set_defaults(func=cmd_complete, instrumental=False, language="en",
                        bpm=None, key=None, time_sig=None)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help(sys.stderr)
        error_json("No command specified", "Use: music_engine.py generate|cover|repaint|extract|lego|complete")

    # Validate ACE-Step path only when actually running a subcommand.
    # --help / --version and missing-subcommand cases are already handled above.
    if not args.ace_step_dir:
        error_json(
            "ACE-Step path not configured",
            "Run install.sh, or edit skills/claude-music/config.json and set "
            "ace_step_dir to your ACE-Step 1.5 installation path"
        )

    try:
        args.func(args)
    except KeyboardInterrupt:
        log("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        error_json(f"Unexpected error: {e}", traceback.format_exc()[:500])


if __name__ == "__main__":
    main()
