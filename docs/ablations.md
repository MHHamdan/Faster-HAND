# Ablations and negative results

Every row below was pre-registered before the run, decided by a rule fixed in advance, and
read from an artefact published in this repository. The artefact for each is named in
[`REPRODUCIBILITY_ARTIFACTS.md`](REPRODUCIBILITY_ARTIFACTS.md).

**Most of these experiments failed.** They are published because the paper's restraint
depends on them: a reader who cannot see the failures has to take the restraint on trust.

Two measurement rules apply throughout and are not negotiable here:

- **Accuracy.** The measured between-seed spread on READ 2016 page is **0.70 pp** over
  *n* = 3 seeds (test 4.574 ± 0.388). Every accuracy difference below is smaller than that.
  A paired per-page bootstrap, not a difference of two numbers, is what makes a comparison
  admissible.
- **Latency.** Absolute latency on the reference host moves up to **9 %** between sessions.
  Only within-session, within-process ratios are evidence.

---

## 1. Architecture

### 1.1 The octave/gated encoder loses to the FCN baseline — Experiment 2

An encoder variant built from octave convolutions with gated depth-wise separable blocks
and squeeze-excitation was trained against the `FCN_Encoder` baseline at
convergence, three seeds each, under a rule fixed before any run: accept the variant
only if its mean test CER is lower by more than 0.3 pp **and** the direction holds
on every seed.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean | SD | 95 % CI |
|---|---:|---:|---:|---:|---:|---|
| `fcn` (baseline) | 3.610 % | 3.120 % | 4.320 % | **3.683 %** | 0.603 | [2.185, 5.182] |
| `hand` (proposed) | 11.010 % | 10.820 % | 10.770 % | **10.867 %** | 0.127 | [10.552, 11.181] |
| `hand`, octave path removed | — | 2.680 % | — | 2.680 % | — | — |
| `hand`, SE removed | — | 11.280 % | — | 11.280 % | — | — |
| `hand`, gating removed | 10.080 % | — | — | 10.080 % | — | — |

**Verdict: H₀ not rejected, by 7.183 pp in the wrong direction.** No seed was dropped,
excluded or re-run.

The single-component arms localise the failure: removing the **octave path** takes the
variant from 10.87 % to 2.68 %, while removing squeeze-excitation or gating
changes nothing material. The defect is the octave branch's low-frequency reconstruction,
not the design as a whole. Those three arms are one seed each and are a diagnosis, not a
measurement.

**Consequence.** Every checkpoint in this repository uses the `FCN_Encoder`, and that is
the encoder the manuscript's architecture section describes. The variant is published,
untrained, in `hand/models/experimental/`, and the manuscript reports it as a negative
result. Artefact: `results_recovery/experiment2/analysis.json`.

### 1.2 Shared visual key/value projection — E4. **The one arm that passed.**

One `in_proj_k` / `in_proj_v` pair over the encoder features, shared by all eight decoder
layers, instead of eight independent pairs. Sharing was verified **in the weights**: across
all 8 layers `att.in_proj_k` and `att.in_proj_v` are byte-identical, while
`self_att.in_proj_q` differs layer to layer as a control.

| | |
|---|---|
| parameters | **7,033,700 → 6,112,612** — **−921,088, −13.1 %** |
| best validation CER | **4.41 %** at epoch 1,295, against the anchor's **4.41 %** at epoch 1,390 |
| difference | **+0.00 pp**, reached 95 epochs earlier; paired *p* = 0.997 |
| gate | within +0.21 pp (threshold 4.62 %) → **PASS** |

The pass is not an artefact of taking a minimum over 277 noisy evaluations: E4's *mean*
validation CER is lower than the anchor's in every late window (1,200–1,300: 0.0495 vs
0.0539; 1,300–1,400: 0.0489 vs 0.0508; 1,400–1,429: 0.0483 vs 0.0490), and 5 of its 277
evaluations reached 4.62 % or better against E5's 0 of 277.

**What the 13.1 % buys is a null, by construction.**

| | anchor → E4 |
|---|---|
| parameters | −921,088 (**−13.1 %**) |
| latency, K/V cached | 2.336 → 2.308 s/page — **1.012×** |
| peak memory allocated | 1239.3 → 1235.8 MiB — **−0.28 %** |
| decoder GFLOPs/page | 125.1 → 125.5 — **unchanged** |

Sharing weights reduces storage, not computation: every decoder layer still applies the
projection to the encoder features, only the stored tensor is reused. The FLOPs, latency and
activation memory *cannot* fall, and 1.012× is noise around unity, well inside the ±9 %
session variation. **E4 is a parameter-count and checkpoint-size result. It must never be
quoted as a speed or memory result.**

### 1.3 Decoder depth 8 → 6 — E5. **Gate: FAIL. Paired test: indistinguishable.**

| | |
|---|---|
| parameters | **5,714,788** — −18.7 % |
| best validation CER | **4.69 %** at epoch 1,305, against the anchor's 4.41 % |
| difference | **+0.28 pp**; gate threshold 4.62 % → **FAIL** |
| 277 evaluations | **0 reached 4.62 %**; 0 reached 4.41 % |

But on the paired per-page test over the same 50 validation pages in one process:
**anchor − E5 = −0.273 pp, bootstrap SD 0.204 pp, CI95 [−0.666, +0.148], *p* = 0.183**,
per-page *r* = 0.762, 26 pages the anchor wins and 19 pages E5 wins.

**The confidence interval spans zero.** The two configurations are statistically
indistinguishable and the *sign* of the difference is not established. The gate still fails,
because the gate was fixed in advance and is a decision rule, not a significance test. Both
facts are reported; neither is allowed to cancel the other.

---

## 2. Training

### 2.1 Extending the training budget — B-1. **Verdict S: exhausted.**

`e14_budget_1p26M_s0` resumed for **550 further epochs (3,596 → 4,145)**: +192,500 samples,
**+15.3 % of the budget**, 10.27 GPU-h, its own argv unchanged except `--max-epochs`.

| | |
|---|---|
| post-resume evaluations | **110** (every 5 epochs) |
| the run's own best | **0.0395 at epoch 3,580** |
| evaluations beating it | **0 of 110** |
| new best checkpoint written | **none** — `best_3580.pt` still stands |

More compute at this scale buys nothing. The reported model is not under-trained, and the
0.12 pp gap to DAN is not a budget artefact.

### 2.2 Label smoothing 0.1 — B-3a. **Gate: FAIL.**

`CrossEntropyLoss(label_smoothing=0.1)` on the decoder objective; architecture untouched, so
parameters are identical to the anchor at 7,033,700. This arm buys accuracy or nothing, so
its gate was the strict one: it had to *beat* 4.41 % by more than 0.21 pp.

| | |
|---|---|
| best validation CER | **4.67 %** at epoch 1,380 |
| difference | **+0.26 pp — worse, not better** |
| gate | ≤ 4.20 % → **FAIL, by 0.47 pp** |
| 277 evaluations | **0 reached 4.20 %**; 0 reached even 4.41 % |

Unlike E5 this needs no statistical caveat: B-3a's whole distribution is worse than the
anchor's in every late window, the direction was visible from epoch 600 (0.5463 vs 0.3515)
and never reversed across 829 further epochs. Label smoothing led on this project's own
calibration data and still failed on the metric that matters.

---

## 3. Inference

### 3.1 Speculative decoding on the full base — E3

*m* − 1 draft heads trained on a **frozen** base (encoder and decoder in `eval`, `no_grad`,
gradients only in the heads), so the base model's output cannot change. **+365,968
parameters, +5.2 %.** 50 READ 2016 test pages, AMP fp16, one process.

| arm | CER | WER | LOER | mAP-CER | s/page | speed-up | tokens/pass | identical to greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.0355 | 0.1331 | 0.0529 | 0.9264 | 1.960 | 1.00× | 1.000 | 50/50 |
| *m* = 2 | 0.0355 | 0.1331 | 0.0529 | 0.9264 | 1.134 | 1.73× | 1.949 | **50/50** |
| ***m* = 5** | 0.0355 | 0.1331 | 0.0529 | 0.9264 | **0.717** | **2.73×** | 3.113 | **50/50** |

Accuracy is *unchanged to four decimals* because the output is the same string. *m* = 5 sits
at the knee of the acceptance distribution: the marginal head is worth +0.57× at *m* = 3 and
+0.14× at *m* = 5, so the head budget stops there. *m* = 3 is the Pareto choice (2.30× for
+2.6 % parameters); *m* = 5 is the pure-latency choice.

### 3.2 Speculative decoding on the compact base — E4 + E3

Heads retrained on E4's frozen base with the E3 recipe unchanged — a single-variable change.
**365,968 head parameters, identical to E3's.** Draft accuracy improves at *k*+2 and *k*+3
(0.9002 / 0.6325 against 0.8452 / 0.6105) and is marginally lower at *k*+1 (0.9976 vs
0.9999), so more tokens are accepted per pass.

**AMP fp16 — the deployed precision:**

| arm | s/page | speed-up | tokens/pass | CER % | WER % | LOER | identical |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 1.9223 | 1.00× | 1.000 | 4.00 | 15.74 | 0.0349 | 50/50 |
| *m* = 4 | 0.7062 | 2.722× | 3.050 | 4.01 | 15.81 | 0.0349 | 47/50 |
| ***m* = 5** | **0.6660** | **2.886×** | 3.230 | 4.00 | 15.79 | 0.0349 | **48/50** |

**fp32 control — exact at every *m*:**

| arm | s/page | speed-up | tokens/pass | CER % | identical |
|---|---:|---:|---:|---:|---:|
| greedy | 1.7230 | 1.00× | 1.000 | 4.00 | 50/50 |
| *m* = 4 | 0.6211 | 2.774× | 3.050 | 4.00 | **50/50** |
| ***m* = 5** | **0.5846** | **2.947×** | 3.232 | 4.00 | **50/50** |

**How to state this, and how not to.** `tokens_per_pass` agrees between the two precisions
to three decimals, so the accept/reject sequence is identical and only floating-point
rounding differs. The honest statement is:

> E4 + E3 provides **2.886× acceleration under AMP**; the fp32 control is exactly
> token-identical on **50/50 pages at 2.947×**.

The AMP divergence is bounded at **≤ 0.01 pp CER with LOER unmoved**. "Token-identical under
AMP" is licensed for **E3 on the full base** (§3.1, 50/50) and **must not** be carried to
this configuration. The three results are distinct and must not be collapsed into one:

| | Precision | Speed-up | Identical to greedy |
|---|---|---:|---:|
| E3, full base (7,033,700 + 365,968) | AMP | 2.73× | 50/50 |
| E4 + E3 (6,112,612 + 365,968) | AMP | 2.886× | 48/50 |
| E4 + E3 (6,112,612 + 365,968) | fp32 | 2.947× | 50/50 |

### 3.3 Static shapes with a compiled decoder step, and its composition — E2, and E2 + E3. **Do not compose.**

E2 pads the visual memory to a static length (`s_pad = 8640`) and runs a compiled decoder
step. Measured in `efficiency_combined_test.json` (one process, 50 test pages, AMP): on its
own it is worth **2.27×** (2.2696×) and is token-identical to greedy on **48/50** pages — the
two differing pages are fp16 tie-breaks under the padded shape, as for the compact model.

Composed with E3 the arithmetic prediction was 2.27 × 2.73 = 6.2×. **The measured answer is
2.63×, which is worse than E3 alone (2.73×).** Both E3 arms are 50/50 identical; the number of
passes is unchanged at 153.0. E2's saving is per-step overhead that E3 has already removed by
taking fewer steps, so there is nothing left for E2 to save, and the padded shape makes each
remaining pass slightly more expensive.

**The efficiency contribution is E3 alone.** This is the reason the campaign measured the
composition instead of multiplying the two factors. (An earlier version of this section
described E2 as an exact mask construction worth 1.033×; that described a different, earlier
arm and did not match the published artefact.)

### 3.4 Exactness gates

A cost change must prove it changed nothing else. `tools/verify_exact_decoding.py` decodes
every page under the reference path and under each optimisation (`HAND_FAST_MASKS`,
`HAND_FAST_STEP`, `use_mem_cache`) **in one process** and requires character-for-character
identity: **50/50 for each, separately and combined, for both HAND and DAN weights.**
`tests/test_fast_decode_paths.py` holds the same claims on CPU permanently — the
band mask against the sliced square over 57 (T, num_pred) combinations, teacher-forced and
130-step incremental decodes with the paths on and off, and the E4 parameter arithmetic.

---

## 4. Data

### 4.1 Corpora that cannot carry an unseen-text claim

Mean character 20-gram test-in-train coverage, measured by
`tools/audit_corpus_text_overlap.py`:

| Corpus | paragraph / line |
|---|---|
| READ 2016 `_sem_dan` (control) | 0.065 |
| **RIMES** | **0.394 / 0.402** — not text-disjoint |
| KHATT | 0.892 / 0.949 |
| **IAM Aachen** | **0.0007 / 0.0003** |

RIMES had been assigned the unseen-text role in an earlier plan. At 0.394 against a READ
control of 0.065 **that role was withdrawn**: any RIMES result is partly recall of training
text. **IAM Aachen is the only corpus here that supports an unseen-text claim**, and the
only one used for one.

### 4.2 Transcription overlap across splits

`tools/audit_dataset_integrity.py`, verdict stored in `dataset_audit.json`. Overlap is
reported two ways because they differ: distinct strings, and held-out **instances**.
Evaluation scores instances, so the instance figure is the one that bounds a CER.

| Dataset | Verdict |
|---|---|
| READ 2016 (all levels), IAM (line, page) | clean |
| `KHATT_paragraph` | acceptable — 8 of 213 test instances (3.8 %) |
| **`KHATT_line`** | **contaminated — 470 of 999 test instances (47.0 %)** |
| **AHAWP** (character / word / paragraph) | **violation** — 65 / 10 / 3 distinct transcriptions in total, shared across every split |

**AHAWP results measure memorisation, not recognition**, and are excluded from every results
table in this repository. No identical document appears across splits in any of the 26
corpora audited.
