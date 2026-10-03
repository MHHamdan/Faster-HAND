# Test-set access ledger

Every evaluation that touches an **official test split** is recorded here, with the reason and
whether it was pre-registered. Model and hyper-parameter selection uses **validation only**;
the test split of a track is opened once the configuration, seeds and checkpoint-selection rule
are frozen.

Why this file exists: repeatedly looking at the test set while iterating is the standard way a
0.3 pp "improvement" turns into a selection artefact. The ledger makes that visible.

**Rules**
1. A test evaluation is legitimate if it is (a) a harness validation on a third-party released
   checkpoint, (b) a frozen historical artefact being re-measured, or (c) a pre-registered final
   evaluation of a frozen configuration.
2. Screening never touches test; screening arms close on validation.
3. Anything else needs a line here explaining why, written *before* the run.
4. Numbers from rows marked "not pre-registered" may not be used to choose between models.

| # | Date (UTC) | Commit | Model / checkpoint | Split | Reason | Pre-registered? | Result |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-07 | 00b27df | **official DAN** `dan_read_page.pt` (Zenodo 7244382) via **DAN's own code** | READ 2016 page test | harness validation — anchor (a) | yes, `experiments/v2_stage1_anchor.md` | CER 3.41 / WER 13.05 / LOER 5.17 / mAP_CER 93.26 |
| 2 | 2026-09-07 | 00b27df | **official DAN** `dan_read_page.pt` via **this repository's harness** | READ 2016 page test | harness validation — anchor (b); the acceptance test | yes, same document | CER 3.41 / WER 13.05 / LOER 5.17 / mAP_CER 93.26 — **0.00 pp from (a)** |
| 3 | 2026-09-07 | 00b27df | frozen V1 released `models/read_page/best_model.pt` | READ 2016 page test (three-token split) | Stage 1 arm **A0 by re-evaluation** instead of retraining (§6) | yes, `experiments/v2_stage1_anchor.md` §3 | CER 4.55 / WER 15.23 / mAP_CER 92.92 — reproduces the frozen number exactly |
| 4 | 2026-09-07 | 00b27df | **deep-convolutional control-baseline encoder smoke run** (DANCER-class geometry, Stage 2 arm E1) (`smoke_dancer`, 2 epochs, untrained) | READ 2016 page test | integration smoke test of a new module; `hand_v2/train.py` evaluates test at the end of every run | not pre-registered — **artefact only** | CER 88.6 % (an untrained model); **not usable for any comparison**, recorded for completeness. Follow-up: give the trainer a `--no-test` switch so screening runs cannot touch test at all |

## Not yet opened

- READ 2016 page test for **A1** (this repository, DAN recipe) — will be opened once the run
  completes its pre-registered 500,000-sample budget. Selection and monitoring use validation.
- READ 2016 page test for **A2** (official DAN code, our environment) — same rule.
- Everything in Stage 2 (B1–B6, E0–E2): **validation only**, by construction.

## Historical note

Numbers produced before this ledger existed (the `results_recovery/` measurements, Experiment 2,
the Stage 0 profiling) were test-set evaluations of **already-frozen** artefacts, i.e. category
(b) above. They are recorded in `experiments/benchmark_suite/registry/` with their run ids and
are not re-litigated here.
