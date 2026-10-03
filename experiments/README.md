# experiments/

The evidence behind the published numbers. **A number that is not backed by an artefact in
this directory is not a number.**

Each artefact's purpose, configuration, split, hardware assumptions, metrics and the claim
it supports are documented one by one in
[`../docs/REPRODUCIBILITY_ARTIFACTS.md`](../docs/REPRODUCIBILITY_ARTIFACTS.md). The results
they establish — including the negative ones — are in
[`../docs/ablations.md`](../docs/ablations.md).

## Layout

| Path | What it holds |
|---|---|
| `benchmark_suite/registry/` | **one JSON per run** — the complete argv verbatim, the environment (Python, torch, CUDA, cuDNN, driver), parameter counts, epochs, timings and metrics. Eleven records are published: the reported page model and its two extra seeds, and both phases of E4, E5 and B-3a |
| `benchmark_suite/profiling/` | the measurement artefacts: the HAND-vs-DAN cost row, seed variance, speculative decoding in AMP and fp32 for both bases, the E4 efficiency row, the E2+E3 composition, and the exactness gate |
| `benchmark_suite/manifests/` | split manifests for the corpora the paper reports on. **Hash-only** — `image_sha256`, `text_sha256`, `n_chars` and the sample name. No transcription text, no image data |
| `benchmark_suite/record.py` | the run-record writer that every training run calls |
| `p0_3_khatt_l6_audit/` | the character 20-gram test-in-train coverage measurement |
| `TEST_ACCESS_LOG.md` | **every evaluation that touched an official test split**, with its date and reason |

## Reading a number

1. Find it in [`../docs/REPRODUCIBILITY_ARTIFACTS.md`](../docs/REPRODUCIBILITY_ARTIFACTS.md).
   That names the JSON.
2. Open the JSON. It names the checkpoint, the argv and the environment.
3. If it is a test-split number, check `TEST_ACCESS_LOG.md`.

To recover the exact command that produced a run:

```bash
python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['config']['argv']))" \
  experiments/benchmark_suite/registry/20260917T160949Z_e14_budget_1p26M_s0_bd9339.json
```

## Two standing rules

- **Never compare latency across sessions.** Absolute latency on the reference host moves up
  to 9 %. Two honest greedy figures coexist here — 1.960 s/page and 2.415 s/page, from two
  harnesses — and neither is "the" latency. Quote a latency with its artefact and its session.
- **Quote a CER with its evaluation batch size.** Padding to the widest image in a batch
  perturbs the encoder output by ~0.01 pp. Everything reported here uses batch size 1.

## What is not here

Screening probes, cost-model screens, step-attribution traces, training-collapse
diagnostics, reconstructed V1 run records, resume bookkeeping, and manifests for corpora
outside the reported results were **not** published. They are development artefacts, not
evidence for a reported claim; `../docs/REPRODUCIBILITY_ARTIFACTS.md` §4 lists each class
and the reason.

Per-sample prediction dumps are excluded **by rule**, not by accident: they contain IAM and
KHATT ground-truth transcriptions, which are registration-licensed and may not be
redistributed.
