# Apple Silicon (MLX / MPS)

ACE-Step runs on Apple Silicon through MLX (DiT, VAE, LM) and MPS (text
encoder). The GPU shares system memory with everything else, so the VRAM tiers
in `SKILL.md` do not apply: a 16 GB Mac behaves like a much smaller card than a
16 GB RTX. ACE-Step itself reports a 16 GB M1 as 11.8 GB, tier 4.

Load this when `detect_gpu.sh` reports `"backend": "mps"`.

## What the engine does on its own

All of this is automatic on unified memory. An explicit flag or a `config.json`
value always wins.

| Setting | Automatic choice | Override |
|---------|------------------|----------|
| Batch | Capped so `batch x duration` stays within 240 s of audio per 16 GB (scales with RAM). A full-length track gets 1 variant; a 30 s draft keeps its 4. | `--batch N` |
| CPU offload | Off while the job fits the budget above, on when it does not (for example an explicit `--batch 2` on a full track). | `--offload on\|off`, or `offload_to_cpu` / `offload_dit_to_cpu` in `config.json` (`"auto"`, `true`, `false`) |
| LM | Not loaded for cover, repaint, extract (ACE-Step ignores it there). For other tasks, an LM that ACE-Step does not list for the machine's tier is swapped for a listed one if installed, else skipped with a warning. | `--lm-model` picks the model; it is still checked against the tier |
| LM backend | `mlx` | none |
| Generation timeout | 3600 s instead of ACE-Step's 600 s | `ACESTEP_GENERATION_TIMEOUT` |

## Measured timings: M1, 16 GB

One 182 s track, turbo model, 8 steps. Wall time is the whole command,
including about 70-90 s of model loading that every run pays.

| Run | Result | Wall time | Per step |
|-----|--------|-----------|----------|
| cover, 1 variant, offload off | ok | 5 min 54 s | 10.7 s |
| cover, 1 variant, offload on | ok | 7 min 45 s | 11.1 s |
| cover, 2 variants, offload off | MPS out of memory | - | - |
| cover, 2 variants, offload on | ok | 11 min 35 s | 19.0 s |
| generate 180 s, no LM, 1 variant | ok | 5 min 48 s | 10.8 s |
| generate 180 s, 1.7B LM (pt), offload off | LM failed to load, ran without it | 5 min 48 s | - |
| generate 180 s, 1.7B LM (pt), offload on | killed by the OS before diffusion | - | - |
| generate 180 s, 1.7B LM (mlx), offload off | MPS out of memory | - | - |
| generate 180 s, 1.7B LM (mlx), offload on | MPS out of memory | - | - |
| lego vocals, base model, 50 steps, full track | ok | 19 min | 15 s |
| lego vocals, base model, 65 steps, 20 s clip | ok | 5 min 30 s | 2.2 s |

What follows from that:

- **Two variants of a full track are not faster than two runs.** 11 min 35 s
  together against 2 x 5 min 54 s apart. Generate one at a time and re-run for
  another take.
- **Offload off saves about a quarter of the run** when the job fits.
- **The 1.7B LM does not fit a 16 GB machine** for a full-length track. On such
  a machine `high` runs as `standard` (the engine says so on stderr) and `max`
  runs the base model without thinking. ACE-Step lists `acestep-5Hz-lm-0.6B`
  for this tier; it is not part of the default download and has not been
  measured here. Install it with
  `cd <ace_step_dir> && uv run acestep-download --model acestep-5Hz-lm-0.6B --skip-main`
  and the engine picks it up.
- Machines with more memory have not been measured. The batch budget scales
  linearly with RAM as a first guess.

## Telling the user what to expect

The `~15s` / `~25s` / `3-5 min` figures in the preset tables are for a 16 GB
NVIDIA card. On a 16 GB Mac, plan with:

| Job | Expect |
|-----|--------|
| 30 s draft | 2-3 min, mostly model loading |
| Full track (3 min), turbo | about 6 min per variant |
| Full track, base model (`max`, lego, extract, complete) | 15-25 min |

Say the estimate before starting anything over a couple of minutes, run it in
the background, and confirm first for base-model runs on full tracks.

## Reading progress

With `--progress`, stderr carries NDJSON:

```json
{"event": "step", "step": 11, "total": 50, "sec_per_step": 14.02, "eta_sec": 547}
```

`step` events come from the real diffusion loop. `progress` events carry
ACE-Step's overall 0..1 figure; inside the diffusion phase the engine replaces
ACE-Step's time-based guess with the real step fraction, because the guess
assumes 5 s per step and ran minutes ahead here. Model loading before the first
step (about a minute) produces no events.

## Out of memory

`MPS backend out of memory` in the result JSON, or a process that dies with
exit code 137, both mean the job did not fit. In order: drop to one variant,
shorten the track, close other applications, then try `--offload on`.
