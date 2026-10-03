# Experimental HAND modules

**None of these modules is part of the evaluated HAND system.** The architecture section of
the manuscript describes `FCN_Encoder` + `GlobalHTADecoder` (in `hand/models/baseline/`);
these modules are designs that were specified, implemented, and then either measured and
rejected or never evaluated at all. They were not part of any released checkpoint.

Despite the name, **`HAND_Encoder` is not the encoder behind the HAND results.** It is an
experimental encoder that shares the project name; the evaluated encoder is `FCN_Encoder`.

Verified 2026-09-01 against all six checkpoints in `models/`: the encoder state dicts are
`FCN_Encoder` (84 tensors, 1.706 M parameters) with no octave, gate or squeeze-excitation
keys, and the decoder state dicts contain no `memory`, `sparse` or `fusion` keys. Full
evidence in [`docs/ablations.md`](../../../docs/ablations.md) section 1.1.

Nothing here is deleted. It is research history and it is the starting point for Option B,
should Experiment 2 justify escalating to it.

## What is here

| Module | Status in the manuscript | State in this repository |
|---|---|---|
| `hand_encoder.py` | reported as a **measured negative result** (encoder section) | **Selectable but never shipped.** Reachable via `--encoder hand`; lost to the baseline by 7.18 pp |
| `encoder_components.py` | the components of that negative result | Used by `hand_encoder.py` |
| `memory_attention.py` | listed as **specified, not evaluated** (supplementary) | **Inactive.** Never trained |
| `msap.py` | listed as **specified, not evaluated** (supplementary) | **Inactive.** Never trained |
| `hand_decoder.py` | not described as the evaluated decoder | **Inactive.** Never trained |
| `complete_hand_model.py` | The full proposed model: HAND encoder + MSAP + HAND decoder | **Inactive.** Never trained |
| `advanced_decoder.py`, `trainer_advanced.py` | earlier decoder variants | **Inactive** |
| `train_curriculum.py`, `train_multigpu.py` | the only entry points that reach MSAP | **Inactive.** No checkpoint |
| `compute_metrics.py` | metrics helper for the above | **Inactive** |

## Why it is quarantined rather than removed

Two parallel model stacks with no marking is what allowed the manuscript to describe one
architecture while the results came from the other. The separation is the fix: `baseline/`
is what runs, `experimental/` is what was proposed, and the boundary is now a directory
rather than an undocumented import graph.

## Before using any of this

`msap.py` cannot be trained as it stands: manuscript Eq. 32 requires a complexity target
`C_target(x)` that the paper never defines. Any use must state its definition first — see
`experiments/recovery_plan.md`, Experiment 3.
