# Reproducibility artefacts

Every JSON this repository publishes, and why it is here. **An artefact is published only
if a reported result cannot be reproduced or independently verified without it.** Internal
diagnostics, screening probes, campaign bookkeeping and superseded measurements are not
published, however interesting they are; §4 lists what was withheld and why.

Common to every artefact below unless stated otherwise:

| | |
|---|---|
| Hardware | **1 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB)**, single-host workstation, no cluster |
| Software | Python 3.11.14, PyTorch 2.9.1+cu128, CUDA 12.8, cuDNN 9.10.2, driver 595.84, Linux 6.8 |
| Batch size | **1**, for every evaluation |
| Decoding | greedy, unless the artefact is a speculative-decoding run |
| Hostname and absolute paths | **redacted** — see §5 |

**Latency artefacts are comparable only within one file.** Absolute latency on this host
moves up to 9 % between sessions; each efficiency JSON is one process, and only ratios
inside it are evidence. Two honest greedy figures therefore coexist in this repository
(1.960 and 2.415 s/page) and neither is "the" latency.

---

## 1. Can the final results be reproduced from a clean clone?

| Final result | Reproducible from this repository? | With what |
|---|---|---|
| **DAN vs HAND baseline comparison** (3.41 % vs 3.55 %, +0.138 pp, *p* = 0.42) | **Verifiable from the artefact; re-measurable only with both checkpoints** | `efficiency_hand_vs_dan.json` + `seed_variance_test.json`; re-run with `tools/efficiency_bench.py` |
| **E4 parameter reduction** (7,033,700 → 6,112,612, −13.1 %) | **Yes, fully, on CPU, with no checkpoint** | `hand_v2/tests/test_fast_decode_paths.py` recomputes the arithmetic; `tools/validate_install_cpu.py` S2 checks the anchor count |
| **E4 accuracy parity** (4.41 % vs 4.41 %, *p* = 0.997) | Verifiable from the artefact; re-measurable with the E4 checkpoint | `e4_vs_anchor_valid.json`, `e4_vs_anchor_test.json` |
| **E3 acceleration** (2.73×, 50/50 identical) | Verifiable from the artefact; re-measurable with the base checkpoint + heads | `spec_decode_test.json`, `spec_decode_test_fp32.json` |
| **E4 + E3 acceleration** (2.886× AMP / 2.947× fp32) | Verifiable from the artefact; re-measurable with the E4 checkpoint + heads | `e4_spec_decode_test.json`, `e4_spec_decode_test_fp32.json` |
| **Final CER / WER / LOER / mAP-CER** | Verifiable from the artefacts; re-measurable with the checkpoint and READ 2016 | the artefacts below + `results_real/*.json` |
| **Final latency measurements** | Verifiable from the artefacts; **not** re-measurable to the same absolute values on other hardware | `efficiency_hand_vs_dan{,_uncontended}.json`, `e4_vs_anchor_efficiency.json` |

**The honest limit.** No checkpoint is distributed in this repository, and none is published
elsewhere yet. A reader can therefore **verify** every number against the artefact that
produced it, **recompute** the parameter arithmetic and the decode-path equivalences on CPU
with no checkpoint at all, and **re-measure** anything once weights exist. Re-measuring the
accuracy numbers additionally needs READ 2016 (CC BY 4.0, freely downloadable) or IAM/KHATT
(registration required, not redistributed). This is stated in
[`reproducibility.md`](reproducibility.md) section 9 and is not worked around here.

---

## 2. Published profiling artefacts

### 2.1 `efficiency_hand_vs_dan.json` · `efficiency_hand_vs_dan_uncontended.json`

| | |
|---|---|
| **Purpose** | the matched HAND-vs-DAN cost and accuracy row — the only accuracy comparison against DAN this project makes |
| **Source experiment** | the efficiency baseline, 2026-09-22 (contended) and its uncontended re-run |
| **Produced by** | `tools/efficiency_bench.py --n-pages 50 --split test` |
| **Configuration** | both checkpoints in **one process**, batch 1, AMP fp16, greedy, K/V cached and reference-decode rows; identical architecture, identical harness, paired over the same pages |
| **Dataset / split** | READ 2016 page `_sem_dan`, **test**, 50 pages |
| **Hardware** | as above; the `_uncontended` file was taken on an otherwise idle device, the other while the host was shared |
| **Metrics** | parameters, encoder/decoder GFLOPs over the real decode, latency ± sd, peak MiB, tokens, ms/token, CER, WER, LOER, mAP-CER |
| **Supports** | README *Results* rows 1–2; manuscript Table V (`tab:multi_page_results`); [`ablations.md`](ablations.md) preamble |

### 2.2 `seed_variance_test.json` · `seed_variance_valid.json`

| | |
|---|---|
| **Purpose** | **the admissibility floor.** Without this, no accuracy comparison in this repository may be read as a difference |
| **Source experiment** | seeds 0/1/2 of the 500 k anchor (`s1_A1fixedR1_s{0,1,2}`) |
| **Produced by** | `tools/seed_variance_analysis.py` |
| **Configuration** | per-page decode of each checkpoint in one process, then between-seed spread, per-page correlation, paired bootstrap contrasts with CI and *p* |
| **Dataset / split** | READ 2016 page `_sem_dan`, test and valid, 50 pages each |
| **Metrics** | per-seed corpus CER, mean ± sd, per-page *r*, paired bootstrap CI95 and *p*, detectable effect at *n* seeds |
| **Supports** | the **0.70 pp** spread (test 4.574 ± 0.388) quoted in the README, the model card and `ablations.md`; the *p* = 0.42 DAN contrast |

### 2.3 `spec_decode_test.json` · `spec_decode_test_fp32.json`

| | |
|---|---|
| **Purpose** | **E3** — lossless speculative decoding on the full-budget base |
| **Source experiment** | Track A, E3 |
| **Produced by** | `tools/spec_decode.py --m 5 --split test` (`--no-amp` for the fp32 control) |
| **Configuration** | base `e14/best_3580.pt` **frozen**; 4 draft heads, **365,968 parameters**; *m* = 2…5 and greedy, all in one process; `amp: true` / `amp: false` |
| **Dataset / split** | READ 2016 page `_sem_dan`, **test**, 50 pages |
| **Metrics** | per-arm latency s/page, speed-up, passes/page, tokens/pass, CER, WER, LOER, mAP-CER, **and per-page identity against greedy**; `per_page` holds timings and token counts only — no transcriptions |
| **Supports** | **2.73×, 50/50 identical** — README *Highlights*, efficiency table row 3, `ablations.md` §3.1 |

### 2.4 `e4_spec_decode_test.json` · `e4_spec_decode_test_fp32.json`

| | |
|---|---|
| **Purpose** | **E4 + E3** — the headline efficiency configuration, and its fp32 exactness control |
| **Source experiment** | Track A/B combination, 2026-09-26 |
| **Produced by** | `tools/train_spec_heads.py` then `tools/spec_decode.py --m 5 --split test` |
| **Configuration** | base `e4_sharekv_s0/best_1295.pt` **frozen**; E3's head recipe unchanged (*m* = 5, hidden 256, 300 epochs, batch 2, lr 3e-4), **365,968 head parameters** |
| **Dataset / split** | READ 2016 page `_sem_dan`, **test**, 50 pages |
| **Metrics** | as 2.3, plus per-position draft accuracy |
| **Supports** | **2.886× under AMP (48/50 identical)** and **2.947× in fp32 (50/50 identical)** — README *Highlights*, efficiency table row 5 and its fp32 sub-row, `ablations.md` §3.2. **These two files exist as a pair precisely so the AMP and fp32 claims cannot be collapsed into one.** |

### 2.5 `e4_vs_anchor_valid.json` · `e4_vs_anchor_test.json` · `e4_vs_anchor_efficiency.json`

| | |
|---|---|
| **Purpose** | **E4** — the 13.1 % parameter reduction: that accuracy is unchanged, and that the saving is storage and not speed |
| **Source experiment** | Track B, E4 (`--share-memory-kv`) |
| **Produced by** | `tools/seed_variance_analysis.py` (paired `*_valid`/`*_test`) and `tools/efficiency_bench.py` (`*_efficiency`) |
| **Configuration** | the 500 k anchor argv with exactly one flag added, seed 0, paired against `s1_A1fixedR1_s0`; both checkpoints in one process |
| **Dataset / split** | READ 2016 page `_sem_dan`; valid for the gate, test for the single licensed test opening |
| **Metrics** | paired per-page CER with bootstrap CI and *p*; parameters, GFLOPs, latency ± sd, peak MiB, CER/WER/LOER/mAP-CER |
| **Supports** | **−921,088 parameters (−13.1 %) at 4.41 % vs 4.41 %, *p* = 0.997**, and **1.012× latency — a null by construction**. README *Highlights*, efficiency table row 4, `ablations.md` §1.2 |

### 2.6 `e5_vs_anchor_valid.json`

| | |
|---|---|
| **Purpose** | **E5, a negative result** — decoder depth 8 → 6 fails its gate, but the paired test cannot establish the sign |
| **Produced by** | `tools/seed_variance_analysis.py`, paired, one process |
| **Dataset / split** | READ 2016 page `_sem_dan`, **valid**, 50 pages. The test split was not opened |
| **Metrics** | corpus CER, page-CER mean, CI95, paired bootstrap SD and *p*, per-page *r*, win counts |
| **Supports** | `ablations.md` §1.3 — gate FAIL at 4.69 % vs 4.41 %, **paired *p* = 0.183, CI95 [−0.666, +0.148] spans zero** |

### 2.7 `b3a_vs_anchor_valid.json`

| | |
|---|---|
| **Purpose** | **B-3a, a negative result** — label smoothing 0.1 is worse, and the failure is distributional, not a statistic artefact |
| **Produced by** | `tools/seed_variance_analysis.py`, paired, one process |
| **Dataset / split** | READ 2016 page `_sem_dan`, **valid**, 50 pages. The test split was not opened |
| **Supports** | `ablations.md` §2.2 — 4.67 % vs 4.41 %, **+0.26 pp worse**, 0 of 277 evaluations reached the anchor |

### 2.8 `exact_decoding_equivalence.json`

| | |
|---|---|
| **Purpose** | **the exactness gate.** A cost change must prove it changed nothing else |
| **Produced by** | `tools/verify_exact_decoding.py` |
| **Configuration** | every page decoded under the reference path and under each optimisation (`HAND_FAST_MASKS`, `HAND_FAST_STEP`, `use_mem_cache`), **in one process**, requiring character-for-character identity |
| **Dataset / split** | READ 2016 page `_sem_dan`, **test**, 50 pages, for HAND **and** DAN weights |
| **Metrics** | per-optimisation identity counts, `all_exact` |
| **Supports** | **50/50 for each optimisation, separately and combined, for both weight sets** — `ablations.md` §3.4; it is what licenses every "identical output" phrase in this repository |

### 2.9 `efficiency_combined_test.json`

| | |
|---|---|
| **Purpose** | **E2 + E3 do not compose** — a negative result that changed the reported configuration |
| **Produced by** | `tools/efficiency_combined.py --modes baseline,e2,e3,e2_e3 --m 5` |
| **Configuration** | four arms in **one process**, each verified page-by-page against the baseline |
| **Dataset / split** | READ 2016 page `_sem_dan`, **test**, 50 pages, AMP fp16 |
| **Supports** | `ablations.md` §3.3 — predicted 6.2× by multiplication, **measured 2.63×, worse than E3 alone at 2.73×**. This is why the repository measures compositions instead of multiplying factors |

### 2.10 `spec_heads_m5_history.json`

| | |
|---|---|
| **Purpose** | the draft-head training record — evidence that the base was **frozen** and that *m* = 5 sits at the knee |
| **Produced by** | `tools/train_spec_heads.py` |
| **Metrics** | `n_parameters` (365,968), per-epoch loss and per-position draft accuracy over 300 epochs |
| **Supports** | the head parameter count in every efficiency row; the *m* = 5 choice in `ablations.md` §3.1 |

### 2.11 `anchor_dan_official.json`

| | |
|---|---|
| **Purpose** | DAN's published checkpoint evaluated through **its own upstream harness**, so the comparison in 2.1 is not an artefact of this project's harness |
| **Metrics** | harness identity, checkpoint digest, wall clock, result files |
| **Supports** | the provenance of the DAN row in the efficiency table |

### 2.12 `corpus_text_overlap.json`

| | |
|---|---|
| **Purpose** | character 20-gram test-in-train **coverage** — a different question from transcription overlap, and the companion to `dataset_audit.json` |
| **Produced by** | `tools/audit_corpus_text_overlap.py` |
| **Dataset / split** | all audited corpora, paragraph and line levels |
| **Supports** | the coverage table in the README and `ablations.md` §4.1; the **withdrawal of RIMES' unseen-text role** (0.394 against a READ control of 0.065) |

### 2.13 `dataset_audit.json` (repository root)

| | |
|---|---|
| **Purpose** | transcription overlap, duplicate documents by content hash, and writer overlap across splits, for all 26 formatted corpora |
| **Produced by** | `tools/audit_dataset_integrity.py --fail-on-violation` — exits non-zero on a blocking violation, so it can gate CI |
| **Supports** | `KHATT_line` **47.0 % test-instance overlap**; AHAWP **excluded from every results table**; `ablations.md` §4.2 |

### 2.14 `release/PARITY_CPU.json`

| | |
|---|---|
| **Purpose** | the CPU reference run: per-page edit distances that `tools/validate_install_cpu.py` stage S4 checks an install against |
| **Configuration** | CPU **fp32**, all 50 test pages, 827 edits total, 12.55 s/page on an unloaded host |
| **Supports** | the install validator, and the claim that the CPU and GPU paths produce the same string |

---

## 3. Published run records

`experiments/benchmark_suite/registry/` — one JSON per run, each storing the **complete argv
verbatim**, the environment (Python, torch, CUDA, cuDNN, driver), parameter counts, epochs
reached, wall clock and metrics. Eleven are published; the rest are withheld (§4).

| Record | Why it is published |
|---|---|
| `20260911T032022Z_s1_A1fixedR1_s0_b09a58` | **phase A** of the reported page model — 1,429 epochs, 91,899 s. Also the 500 k anchor every Track B arm is paired against |
| `20260917T160949Z_e14_budget_1p26M_s0_bd9339` | **phase B — the reported page model.** 3,596 epochs = 1,258,600 samples (DAN's published budget), best epoch 3,580, 229,150 s, `n_params.total` 7,033,700. Every headline accuracy number traces here |
| `20260917T160951Z_s1_A1fixedR1_s1_af4532` · `20260919T101946Z_s1_A1fixedR1_s2_1baf26` | seeds 1 and 2 — without these the **0.70 pp seed spread** is a bare assertion |
| `20260924T030209Z_b1_budget_cont_s0_1485fa` | **B-1**: +550 epochs changed nothing. Evidence that the reported model is not under-trained |
| `20260924T135940Z_e4_sharekv_s0_4c50d4` + `20260925T201240Z_e4_sharekv_s0_d0408c` | **E4**, both phases (capped at the 30 h limit, then resumed). `n_params.total` **6,112,612** |
| `20260924T135940Z_e5_depth6_s0_e554ad` + `20260925T201240Z_e5_depth6_s0_3583cf` | **E5**, both phases. `n_params.total` 5,714,788 |
| `20260924T150342Z_b3a_labelsmooth_s0_a61f5e` + `20260925T211803Z_b3a_labelsmooth_s0_efff0c` | **B-3a**, both phases. `n_params.total` 7,033,700 — unchanged, as the arm requires |

Records marked `status: "failed"` or `status: "running"` from the same campaign are **not**
published: they are resume bookkeeping and carry no measurement.

## 3.1 Split manifests

`experiments/benchmark_suite/manifests/` — five manifests, for the corpora the paper reports
on. Each lists per-sample `image_sha256`, `text_sha256`, `n_chars` and the formatted sample
name. They let anyone confirm they built the same splits **without redistributing a single
character of licensed ground truth**: no transcription text and no image data is present.

READ 2016 manifests additionally carry source-scan filenames, one manuscript page-number
field and one region-geometry field, all from a CC BY 4.0 corpus; this is enumerated in
`release/NOTICE.md` §4.

---

## 4. Withheld, and why

Not published. Each is a development artefact, not evidence for a reported claim.

| Withheld | Count | Why |
|---|---|---|
| Step-attribution and pipeline profiles (`line_*`, `page_*`, `*_attribution.json`) | 18 | internal optimisation traces; no published number depends on them, and they embedded interpreter and workstation paths |
| Screening and probe run records (`probe_*`, `p01_*`, `x1_*`, `smoke_*`, `b2_scratch_*`) | 15 | screening that never reached a reported result |
| `backfill_*` run records | 31 | run records reconstructed after the fact for V1 checkpoints; the V1 metrics themselves are published in `results_real/` |
| Resume bookkeeping records (`status: failed` / `running`) | 6 | no measurement |
| Cost-model screens (`depth_inference_screen.json`, `batch_scaling.json`, `multitoken_m_analysis.json`, `cudnn_shape_bench*.json`, `b5_boundary_oracle.json`) | 7 | they chose what to run next; no reported number cites them |
| D1 / X1 / X2 layout-ablation and training-collapse diagnostics | 12 | internal diagnosis of a training collapse during recovery, not a paper claim |
| Layout-metric and learning probes (`layout_metrics_*`, `probe_learning_*`, `init_diagnostics`, `stage0_*`) | 20 | development instrumentation |
| Split manifests for corpora the paper does not report on (RIMES, Bentham, Saint Gall, Washington, IAM line, …) | 14 | 10 MB of manifests for corpora outside the reported results |
| Per-sample prediction dumps | — | excluded **by rule**, not by accident: they contain IAM and KHATT ground-truth transcriptions, which are registration-licensed and may not be redistributed |

---

## 5. Redactions, and their effect on digests

Three classes of change were made to published artefacts before release. Each is mechanical,
touches metadata only, and changes **no measured value**:

| Change | Files | What |
|---|---|---|
| Absolute repository root | 11 run records | the author’s absolute checkout path → **`${HAND_ROOT}`** |
| Hostname | 14 run records and profiling files | `"host": "<workstation name>"` → `"host": "<redacted: single-host workstation>"` |
| Local export path | `release/PARITY_CPU.json` | a scratch directory path → `<exported model directory>` |
| User name | every run record | `"user": "<account>"` → `"user": "<redacted>"` |
| Local weight directory | run records and profiling files | `private/dan_weights/` → `weights/dan/`, the directory `data/README.md` tells a reader to create |
| Pointers to unpublished planning notes | `notes` field of five run records | the trailing "See …" / "Pre-registration: …" reference removed; the rest of the note is unchanged |

**Every redacted file's SHA-256 therefore differs from the digest recorded in the private
archive.** `release/MANIFEST.md` lists the digest of each file **as published here**, which
is the digest a reader can check. No manifest in this repository claims the digest of a
pre-redaction file. The before/after digest pairs are retained in the development archive.

Run records also carry the git commit and branch of the development repository they ran in
(`git.commit`, `git.branch`). This public repository was exported from that tree with a new
history, so those hashes identify the code state recorded in `effective_code_state` but do not
resolve in this repository's log.

Every published record still parses as JSON and still carries its complete argv, environment
and metrics.

### Two things inside a run record that are *not* redacted, deliberately

**`effective_code_state.per_file_sha256`** records the digest of every source file **as it
executed**, over 141 files under `hand/`, `hand_v2/` and `tools/`. Those digests describe
the private working tree at launch time and **do not match the files published here**, for
two reasons, neither of which changes any measured value:

- the CeCILL-C notices restored on nine files and the Article 5.2 modification statements
  added to twenty-four (see `release/NOTICE.md` §1) are **comment-only**, so the digests
  moved while the executed behaviour did not;
- some of the 141 files are development utilities that are not published (§4).

The run record is a record of **what ran**; `release/MANIFEST.md` is a record of **what is
published**. Both are correct, they answer different questions, and neither was edited to
agree with the other.

**The `notes` field** carries the launch note the author wrote at the time. Pointers in it to
planning notes that are not published were removed (table above); the descriptive text is as
written.
