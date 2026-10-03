# Model card

For the HAND page-level recognition model and the five V1 scale checkpoints. Measurement
provenance for every number here is in
[`REPRODUCIBILITY_ARTIFACTS.md`](REPRODUCIBILITY_ARTIFACTS.md); the negative results are in
[`ablations.md`](ablations.md).

---

## Availability

The four READ 2016 models are released as assets of the GitHub release
[v1.1.0](https://github.com/DocumentRecognitionModels/HAND-Decoding/releases/tag/v1.1.0), not in
the git history. `release/hand_release/hub.py` downloads them and refuses any file whose
SHA-256 differs from the pinned value. Each package carries its own README with the reported
result, its scope and the attribution.

| Model | Reads | Reported result (paper) | Package | SHA-256 |
|---|---|---|---|---|
| `hand-read2016-page` | single pages | READ 2016 test CER 3.55 %, WER 13.31 % | 27.5 MB, with draft heads | `877246ec2503da82636a820d99091c9a99c234d5396b8cb8e9bc9ab8fc4f645c` |
| `hand-read2016-page-compact` | single pages | CER 4.00 %, WER 15.74 % (500,000 samples) | 27.5 MB, with draft heads | `c53984243256c9e90aee7aea887fd8706034a39cf31ddc81631251f58d6d7791` |
| `hand-read2016-double-page` | double pages | double-page test CER 3.60 %, WER 13.27 % | 27.5 MB, with draft heads | `42928757e0d561343db7a7ef7452886ef002310770ea8ffa7164361728fbe9ea` |
| `hand-read2016-triple-page` | triple pages | triple-page test CER 3.48 %, WER 13.41 % | 26.1 MB | `1c09c18ecd3aac0a99011dee05b593832b9e4a1ec818efc2779e3c742262d58d` |

Before release, every model was decoded on its stored example image and reproduced the recorded
prediction token for token; with the draft heads, speculative decoding reproduced greedy
decoding. Each model reads the number of pages it was trained on and over-generates on shorter
inputs. The IAM and KHATT models are not released, because they were trained on corpora licensed
for registered research use.

`release/` carries the licence notices, the inference contract, the model configuration and
the export/evaluation tools — everything needed to *use* a checkpoint, and no checkpoint.
`release/MANIFEST.md` lists every released file with its SHA-256.

## Architecture

A fully convolutional encoder feeding an eight-layer autoregressive transformer decoder,
trained with a single sequence cross-entropy over an interleaved text-and-layout token
stream. Output is the transcription and the layout tags in one stream, with no explicit
segmentation step. Figure 3 of the manuscript draws this exactly, from
`docs/assets/hand_architecture.tex`.

| | |
|---|---|
| Input | RGB page, `B x 3 x H x W` |
| Encoder | 6 convolutional blocks (3 -> 128 channels, all downsampling) then 4 depth-wise separable blocks (-> 256 channels), 3x3 kernels, InstanceNorm, residual where shapes match |
| Feature grid | `B x 256 x H/32 x W/8`; 2D sinusoidal PE added, then flattened to `T_enc x B x 256` |
| Decoder | 8 post-norm layers, `d_model` 256, 4 heads, FFN width 256 |
| Self-attention | causal **and** banded to a window of `w = 100` tokens |
| Cross-attention | full, over the flattened visual memory; `K = V`, encoder padding masked |
| Output head | ReLU, dropout, 1x1 convolution to `|V| = 100` (text + layout symbols) |

**The evaluated system is `FCN_Encoder` + `GlobalHTADecoder`**, both in
`hand/models/baseline/`, optionally with E4 (`--share-memory-kv`) and E3
(`hand/models/baseline/spec_heads.py`). Despite the name, the class `HAND_Encoder` in
`hand/models/experimental/` is **not** the encoder behind these results; it is an
experimental encoder sharing the project name that was measured and rejected
([`ablations.md`](ablations.md) §1.1).

| Configuration | Encoder | Decoder | Total |
|---|---|---|---|
| Reported page model | 1,706,240 | 5,327,460 | **7,033,700** |
| **+ E4**, shared visual K/V across all 8 decoder layers | 1,706,240 | 4,406,372 | **6,112,612** (−921,088, **−13.1 %**) |
| **+ E3**, *m* = 5 speculative draft heads | | +365,968 | 6,112,612 + 365,968 |

The draft heads add **365,968 parameters** and are trained on a frozen base, so the base
model's output cannot change. On the full-budget base the same head budget gives
7,033,700 + 365,968.

The page model **is** DAN's architecture: parameter count equal to the digit, identical
state-dict key sets and tensor shapes, identical FLOPs and peak memory. That identity is
the basis of every matched comparison in the paper, and it is checked mechanically by
`tools/validate_install_cpu.py` stage S2 and by `tests/test_fast_decode_paths.py`.

A separate set of designed components — the gated/octave HAND encoder, MSAP,
memory-augmented and sparse attention, adaptive fusion — lives in
`hand/models/experimental/` and is **inactive on the trained path**. No checkpoint uses it,
its `README.md` says so, and the manuscript's architecture section describes the trained
model rather than these. They remain published so that the manuscript's training-strategy
and supplementary sections can be checked against code. The encoder variant that *was*
trained lost to the baseline by 7.2 pp; see [`ablations.md`](ablations.md) §1.1.

## Training data

READ 2016, page level, `_sem_dan` five-token layout scheme. 350 training pages, 3,596
epochs = **1,258,600 samples**, which is DAN's published budget. Initialised from Denis
Coquenet's released READ 2016 line checkpoint. A synthetic-page curriculum over 41 audited
fonts runs alongside. Exact argv: [`reproducibility.md`](reproducibility.md) section 5.

## Evaluation

READ 2016 page test/valid, greedy decoding, **evaluation batch size 1**, seed 0. Run
`e14_budget_1p26M_s0`, best epoch 3580.

| Split | n | chars | CER | WER |
|---|---|---|---|---|
| test | 50 | 23,262 | **3.55 %** | 13.31 % |
| valid | 50 | 21,609 | 3.95 % | 15.01 % |

**How to read 3.55.** Against DAN's published 3.43 on the same split that is a 0.12 pp gap.
The measured between-seed spread over *n* = 3 seeds is **0.70 pp** (test 4.574 ± 0.388), so
0.12 pp is inside the noise: *these two numbers are not distinguished by this measurement*.
Do not read the table as either system being ahead.

The V1 scale checkpoints (`read_page`, `read_double_page`, `read_triple_page`, `iam_page`,
`khatt_paragraph`) are a separate, earlier line of work; their numbers are in the README and
in `results_real/*.json`.

## Intended use

Research on end-to-end handwritten document recognition and layout analysis. Not validated
for production transcription, for archival cataloguing decisions, or for any use where an
incorrect transcription carries cost.

## Limitations

- **Scale.** Trained on 350 READ 2016 pages. Generalisation beyond that corpus is
  unmeasured except where a table says otherwise.
- **Contamination.** `KHATT_line` has **47.0 % test-instance transcription overlap** with
  training; `AHAWP_{character,word,paragraph}` share their entire (tiny) transcription set
  across all splits, so AHAWP numbers measure **memorisation**, not recognition, and must
  never be reported as either. See `dataset_audit.json` and the README.
- **Text coverage.** Mean 20-gram test-in-train coverage: RIMES 0.394, KHATT 0.892–0.949,
  READ 2016 `_sem_dan` 0.065, IAM Aachen **0.0007**. IAM Aachen is the only corpus here that
  supports an unseen-text claim, and the only one used for one.
- **Latency is session-dependent.** Absolute latency on the reference host moves up to 9 %
  between sessions. Two honest greedy figures exist (1.960 and 2.415 s/page) from two
  harnesses; neither is "the" latency. Quote a figure with its artefact and session.
- **Speculative decoding exactness is precision-dependent and must be stated per
  configuration.** E4 + E3 provides **2.886× acceleration under AMP**; the fp32 control is
  exactly token-identical on **50/50 pages at 2.947×**. Under AMP the E4 + E3 output is
  **48/50** identical, with the CER difference bounded at ≤ 0.01 pp and LOER unmoved.
  **E3 alone on the full base is 2.73× and 50/50 identical under AMP** — that exactness is
  licensed for E3 and must not be carried to E4 + E3. Never write "2.886× lossless".
- **The `hand` encoder is not converged.** Its ablation table is not at a common epoch
  across rows; `PENDING.md` says so.
- **No accuracy claim is attached to post-OCR correction.** That code is under
  re-measurement.

## Licence chain — read before redistributing

The weights are **not** covered by the code's MIT licence.

| Component | Licence |
|---|---|
| This author's own code | MIT ([`LICENSE`](../LICENSE)) |
| 27 files derived from Denis Coquenet's DAN / VerticalAttentionOCR | **CeCILL-C** — list in [`release/NOTICE.md`](../release/NOTICE.md) §1.1 |
| Trained weights | **CC BY 4.0**, inherited from the initialisation checkpoint (Zenodo [10.5281/zenodo.7244382](https://doi.org/10.5281/zenodo.7244382)); §3(a) attribution is binding |
| READ 2016 | CC BY 4.0 — not redistributed here |
| IAM, KHATT | research licences requiring registration — not redistributed here |

[`release/NOTICE.md`](../release/NOTICE.md) is the text that discharges CeCILL-C Art. 6.4(3).
`release/NOTICE.md` §1.1 and §1.2 carry the per-file list.

## Citation

`CITATION.cff`. It deliberately separates **software** authorship (sole author: Mohammed
Hamdan) from the **paper**'s author list, which is a bibliographic fact and not a statement
about who contributed code.
