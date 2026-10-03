# Multi-page (double / triple page) experiments — reproducibility note

Date: 2026-10-02 (UTC). Host: single workstation, 2x NVIDIA RTX PRO 6000 Blackwell Server Edition
(97,887 MiB each), driver 595.84, Linux 6.8.0-138-generic. Software: Python 3.11.14,
torch 2.9.1+cu128 (CUDA 12.8, cuDNN 91002), Pillow 12.2.0, numpy 2.4.6, networkx 3.5,
safetensors 0.8.0. Code state: the development tree this repository was exported from (recorded
in each run record's `git` and `effective_code_state` blocks), plus the additive changes listed
at the end.

GPU 0 ran every evaluation, GPU 1 every fine-tune; every CUDA process was launched as
`CUDA_VISIBLE_DEVICES=<N> flock /tmp/hand-gpu<N>.lock <command>`. Two processes belonging to
another project were resident on the GPUs throughout (GPU 0: 1.4 GB, GPU 1: 19.9 GB); each result
JSON records `nvidia-smi` utilisation at start and end (`env.gpu_snapshot_*`), and latencies are
labelled "contended" wherever that process showed non-zero utilisation.

## Checkpoints

| name | path | sha256 | parameters |
|---|---|---|---|
| BASE page model (e14, epoch 3580, 1,258,600 samples) | `outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt` | `d94af1419554a057ae671733af2fe4cf3f63d1f8b3652f9de17277b432632f5f` | 7,033,700 |
| BASE release export | `outputs/export_e14/model.safetensors` | `b842d1e1e1d9568825253b79d01b91702635a665d19227c9a2748de0ec49515b` | 7,033,700 |
| E3 draft heads for BASE (m=5) | `outputs/spec_heads_m5/heads.pt` -> `outputs/export_e14/spec_heads_m5.safetensors` | `3598151fd92a50fd722b3ac0931310ad0ca2776b62dd84339ad51ade2dd71976` / `fc73d191871da0241bcd82ea68d2844ffea2385635712c9951802ecfe9bbbf2a` | 365,968 |
| E4 page model (shared K/V, epoch 1295) | `outputs/e4_sharekv_s0/checkpoints/best_1295.pt` | `cb70b5ff13c690504a355003e7c53a426b336d8d4bd409f89c27b4a7a755106e` | 6,112,612 |
| E4 release export | `outputs/export_e4/model.safetensors` | `e4237f30d27bb58898288726f341bf15a3ed0aea2244231725f735f65c6481dc` | 6,112,612 |
| E3 draft heads for E4 (m=5) | `outputs/spec_heads_e4_m5/heads.pt` -> `outputs/export_e4/spec_heads_m5.safetensors` | `9a592a0b4e79108e8c0fd0f1e9e62aa0043430bd38ab8b447e09e3b5e008389d` / `11f7fb8713e018251f8db7d2299c1b0acdfc9b68b9e57805a496dae0d93823ad` | 365,968 |
| V1 double-page model (three-token scheme) | `models/read_double_page/best_model.pt` | `c5493d8011705ca7355d4e27f0760fdaa2003dd434a89d5426d50ff625f2f63c` | V1 |
| V1 triple-page model (three-token scheme) | `models/read_triple_page/best_model.pt` | `6cce715f1a880c834e74a15d29bc74465ce1f276fc5137005c378514ac960e05` | V1 |
| adapted double-page model (this study, 4 h) | `outputs/ft_double_page_from_e14_s0/checkpoints/best_220.pt` -> `outputs/export_ft_double/model.safetensors` | `8de29302713cffc559c3804308889efea2a8a9739df00bb1fe345a31c121f3d9` / `2dd86b07e42232d22f4b6fcaf26259b21556608a69c292f89bad41cd39bb3f63` | 7,033,700 |
| E3 heads for the adapted double model (m=5, 60 epochs) | `outputs/spec_heads_ft_double_m5/heads.pt` -> `outputs/export_ft_double/spec_heads_m5.safetensors` | `47d70a999f8cd2ff66f3f56082bce74445799651cbdc528efdd28fba17162761` / `cb933b41cfd94f148d3206e0023210c31e322cf5ec801825578c1c7e420d9211` | 365,968 |
| adapted triple-page model (this study, 1 h, from the double model) | `outputs/ft_triple_page_from_double_s0/checkpoints/best_50.pt` -> `outputs/export_ft_triple/model.safetensors` | `9e53f3b6714178ee34876f4b52b4d79d654fbb324bc4c971261ba1f53c1d1579` / `084df87b2b8d23fb16c8d1e7a7e7353168cf2fe4db843e9ffef162f05e376a2b` | 7,033,700 |

Exports: `python3 release/tools/export_release_checkpoint.py --ckpt <pt> --out <dir> --expect-params <n> --expect-charset 99 --heads <heads.pt>`.

## Datasets (all 150 dpi, charset 99, five-token layout scheme ⓟ ⓝ ⓢ ⓐ ⓑ + closing forms)

| level | directory | train/valid/test | construction | manifest |
|---|---|---|---|---|
| page | `formatted/READ_2016_page_sem_dan` | 350/50/50 | one scan per sample | `experiments/benchmark_suite/manifests/READ_2016_page_sem_dan.json` |
| double | `formatted/READ_2016_double_page_sem_dan` | 169/24/24 | DAN pairing: two scans with the same written page number, concatenated horizontally; unpaired scans dropped | `.../READ_2016_double_page_sem_dan.json` |
| triple | `formatted/READ_2016_triple_page_sem_dan` | 116/16/16 | three consecutive scans of the split in scan order, non-overlapping, remainder dropped; shorter scans padded at the bottom with their median colour (0 px on every test triple, max 136 px at 300 dpi in train); NOT a DAN construction | `.../READ_2016_triple_page_sem_dan.json` |

Build: `python3 hand/Datasets/format_read_dan_splits.py --levels triple_page` then
`python3 tools/build_read_triple_page_dan_manifest.py` (asserts the source triples equal the V1
manifest `READ_2016_triple_page_sem.json`). `labels.pkl` sha256
`f086776d61b93ef5ead1c730a206cb6baeda0c7b4be282bef90432bd9a9415a1`.

Test-split sizes: page 465 chars / 19.8 lines / 1,190x1,755 px; double 937 chars / 39.0 lines /
2,380x1,755 px; triple 1,407 chars / 58.1 lines / 3,571x1,755 px (means).

## Zero-shot scaling sweep (T2)

`experiments/multipage/run_zero_shot.sh` (exact commands inside) runs `tools/multipage_eval.py`
for BASE, +E3, +E4, +E3+E4 over page / double_page / triple_page in one process per configuration
(so cross-level ratios are within-process), AMP fp16, batch 1, warm-up 2 documents, decoding budget
3000 / 6000 / 9000 tokens (the page model's config caps at 3000; raised explicitly), line cap raised
from the harness's 100 to 1000, per-line guard 150. A BASE fp32 single-page row is run last.
Metrics are the training harness's `MetricManager` (CER, WER with `format_string_for_wer`, LOER and
mAP-CER with the READ post-processing module). Outputs: `zero_shot/<CONFIG>_<level>.json`
(aggregate + per-sample + raw strings), `zero_shot/per_sample_<CONFIG>_<level>.csv`,
`zero_shot/summary_<CONFIG>.json`; raw predictions in `outputs/multipage_predictions/` (git-ignored).
Speculative rows carry `identity_vs_ref` = documents whose output equals the KV-cache greedy output
of the same base.

## Adaptation (T3)

Double-page fine-tune from the page model (DAN's protocol: continue from the page model), 4 h cap:
```
CUDA_VISIBLE_DEVICES=1 flock /tmp/hand-gpu1.lock python3 tools/train.py --dataset READ_2016 \
  --level double_page --variant _sem_dan --encoder fcn \
  --init-from outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt --no-hand-encoding --additional-tokens 1 \
  --batch-size 1 --lr 1e-4 --fonts-dir fonts_dan_read --output ft_double_page_from_e14_s0 --seed 0 \
  --eval-interval 10 --experiment multipage_adaptation --curr-step 100 --syn-min-lines 10 --syn-max-lines 60 \
  --syn-init-proba 0.5 --syn-end-proba 0.2 --syn-proba-steps 20000 --start-valid-from-steps 0 \
  --workers 4 --max-hours 4 --max-epochs 10000
```
Smoke first (`--smoke`, 2 epochs): `transfered weights for encoder` / `decoder` printed, 7,033,700
parameters, validation CER 0.3326 after 338 samples (registry
`20261002T031036Z_ft_double_page_smoke_d6d930.json`), i.e. the page weights loaded strictly.
Why the synthetic schedule differs from the page run (0.9 -> 0.2 over 200k samples, curr-step 10,000):
with `start_scheduler_at_max_line` the synthetic ratio stays at its initial value until the line
curriculum reaches `max_nb_lines`, which at curr-step 10,000 is 500k samples away; a 4-hour run
(~100k samples) would therefore see 90 % synthetic 10-15-line pages and only ~10 % real double
pages. The compressed schedule (lines 10 -> 60 over 5k samples, ratio 0.5 -> 0.2 over the next
20k) keeps the recipe's shape inside the budget. The run record (train_seconds, epochs, best
validation CER) is in `experiments/benchmark_suite/registry/<run_id>_ft_double_page_from_e14_s0_*.json`.
Result: train_seconds 14,451, 311 epochs, 52,559 samples seen, best epoch 220 with validation CER
0.0410 / WER 0.1569 (`registry/20261002T031727Z_ft_double_page_from_e14_s0_f44211.json`).

Triple-page adaptation, continued from the double-page checkpoint, real triples only (the synthetic
generator produces single pages at this level), 1 h cap:
```
CUDA_VISIBLE_DEVICES=1 flock /tmp/hand-gpu1.lock python3 tools/train.py --dataset READ_2016 \
  --level triple_page --variant _sem_dan --encoder fcn \
  --init-from outputs/ft_double_page_from_e14_s0/checkpoints/best_220.pt --no-hand-encoding --additional-tokens 1 \
  --batch-size 1 --lr 1e-4 --fonts-dir fonts_dan_read --output ft_triple_page_from_double_s0 --seed 0 \
  --eval-interval 10 --experiment multipage_adaptation --no-synthetic --start-valid-from-steps 0 \
  --workers 4 --max-hours 1.0 --max-epochs 10000
```
Result: train_seconds 3,654, 66 epochs, 7,656 samples seen, best epoch 50 with validation CER 0.0545 /
WER 0.1724 (`registry/20261002T072130Z_ft_triple_page_from_double_s0_f29b66.json`).

Draft heads for the adapted double model: `tools/train_spec_heads.py --ckpt <best_220.pt> --m 5 --epochs 60
--batch-size 2 --level double_page --max-hours 0.5 --out outputs/spec_heads_ft_double_m5` (new `--level`
option; 672 s). No heads were trained for the triple model.

Evaluation of the adapted checkpoints: `experiments/multipage/run_adaptation.sh` and
`run_adaptation_double.sh` (the double branch was relaunched once after a LOER hang, see caveats);
outputs in `adaptation/FT_DOUBLE_*.json`, `FT_DOUBLE_E3_*.json`, `FT_TRIPLE_*.json`.

These are REDUCED-DURATION adaptations (4.2 % and 0.6 % of the page model's 1,258,600 samples): the
comparison with the page model is not matched in budget.

## Baselines (T4)

`CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock python3 tools/evaluate_hand.py --model read_double_page read_triple_page --split test --batch-size 1 --max-chars 9000 --out experiments/multipage/v1`
evaluates the V1 checkpoints on their own V1 test sets (three-token scheme, consecutive-scan
construction; 25 / 16 documents). DAN double-page weights are not available locally, so no DAN
double-page number was measured here; published DAN / Faster-DAN / DANCER numbers are quoted from
the papers by the manuscript, not by this directory.

## Tables (T5)

`python3 tools/multipage_summary.py` -> `tables.json`, `tables.md` ((a) zero-shot scale table incl. the
forced-continuation diagnostic rows `BASE_FORCED_*` produced with `--force-pages`, (b) within-process
scaling ratios, (c) adaptation rows, (d) V1 baselines). Headline numbers (test splits):

| model | level | CER | WER | LOER | mAP-CER |
|---|---|---|---|---|---|
| BASE page model | single | 0.0355 | 0.1328 | 0.0529 | 0.9264 |
| BASE page model, zero-shot | double / triple | 0.5443 / 0.6871 | 0.5875 / 0.7228 | 0.4425 / 0.6019 | 0.4906 / 0.3024 |
| adapted double model (4 h) | double | 0.0360 | 0.1327 | 0.0474 | 0.925 |
| adapted triple model (+1 h) | triple | 0.0348 | 0.1341 | 0.0470 | 0.939 |
| V1 double / triple models (own sets) | double / triple | 0.0486 / 0.0681 | 0.1516 / 0.1779 | n/a | n/a |

## Caveats

- Zero-shot rows use a model that never saw a multi-page image; the token budget and line cap were
  raised so that only the model's own end token, the 150-character per-line guard or the budget stop it.
- Latencies are batch-1 wall-clock on a shared GPU; see the contention notes.
- The triple-page construction is this repository's own (no external reference numbers exist).
- Zero-shot, the page model decodes every double/triple page as exactly one ⓟ...Ⓟ block (the first
  scan, read at CER 0.032-0.038) and stops; the error is omission, not misrecognition. Forced
  continuation (`BASE_FORCED_*`) does not move its attention to the second scan.
- Each adapted model emits exactly its trained page count: the double model hallucinates extra page
  blocks on single pages (CER 2.66) and omits the third page of triples (CER 0.36); the triple model
  does the same on single and double pages. Their own-level numbers are the only meaningful ones.
- LOER is an exact graph edit distance; on predictions with many hallucinated page blocks it becomes
  exponential, so `tools/multipage_eval.py` applies a 120 s per-document limit (`aggregate.loer_timeouts`,
  timed-out documents are excluded from the LOER sum; CER/WER/mAP-CER are unaffected). Only the
  off-level rows of the adapted models hit it (6/50 and 5/50 single pages).
- Speculative heads were retrained only for the adapted double model (60 epochs, reduced).
