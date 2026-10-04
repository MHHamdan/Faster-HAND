# Reproducibility

Everything below runs from the repository root on a single machine. No cluster, no
`/scratch` paths. This is the single reproduction document for FasterHAND; the README links to
it and nothing else duplicates it.

### Required to reproduce the paper, or optional?

| Section | | |
|---|---|---|
| **1** Environment | **required** | the exact versions every number was measured in |
| **2** Inputs you must acquire | **required** | corpora, the initialisation checkpoint, the fonts |
| **3** Validate the install | **required** | one CPU command; tells you whether the rest can work |
| **4** Evaluate a checkpoint | **required** | reproduces the reported CER / WER / LOER / mAP-CER |
| **5** Train the current page model | **required** for the model itself; ~89 GPU-hours |
| **6** The V1 training path | optional | an earlier, separate line of work |
| **7** Regenerate the paper tables | **required** to re-derive the published tables |
| **8** Measured results | reference | the numbers themselves |
| **9** Hardware and limits | **read before quoting any latency** | |
| **10** Efficiency and exactness | **required** for the E3 / E4 / E4+E3 claims | |
| **11** Optional development utilities | optional | nothing here produces a published number |

Artefact-by-artefact provenance for every published JSON is in
[`REPRODUCIBILITY_ARTIFACTS.md`](REPRODUCIBILITY_ARTIFACTS.md). The negative results are in
[`ablations.md`](ablations.md).

> Numbers in this repository come from artefacts, not from prose. Evaluations of the V1
> released checkpoints are in `results_real/`; every run of the current `tools/train.py` path
> writes a record into `experiments/benchmark_suite/registry/`. Where a document and an
> artefact disagree, the artefact is right. Where a table cell has no measurement behind
> it, the generator prints `--` and names the gap in `PENDING.md`.

**Read section 3 first.** It is one CPU command, it needs no GPU and no licensed corpus,
and it tells you whether the rest of this document can work on your machine.

---

## 0. What is in the repository

```
hand/              the model library — encoders, decoders, dataset managers and formatters,
                   trainer, layout metrics. Parts are CeCILL-C-derived from DAN (see NOTICE)
tools/             entry points — training (tools/train.py), evaluation, tables, dataset
                   checks, CPU validation
tests/             the CPU test suite
release/           the licence notices, the inference contract, the export/eval tools
experiments/       run records and profiling artefacts behind the published numbers
results_real/      evaluations of the V1 released checkpoints, one JSON per run
docs/              this file, the model card, the ablations and the changelog
```

### Which entry point do I use?

| I want to | Use | Why |
|---|---|---|
| train the current page model | `tools/train.py` | it produced `outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt`, the checkpoint behind the reported READ 2016 page numbers |
| evaluate a released export, on CPU or GPU | `release/tools/evaluate_release.py` | it loads the exported payload strictly and checks the result against a stored measurement |
| check that an install works, without a GPU | `tools/validate_install_cpu.py` | section 3 |
| train or evaluate one of the **V1** checkpoints in `models/` | `tools/train_hand.py`, `tools/evaluate_hand.py` | those checkpoints were produced by that path; `results_real/` is its output |

`tools/train.py` does **not** replace `hand/`. It imports `tools/train_hand.py`'s argument
parser and parameter builder, and it runs `hand.OCR.document_OCR.hand.trainer_std_hand.Manager`
— the same trainer, the same encoder, the same decoder, the same loss and schedule. What it
adds is the cuDNN API selection, optional shape bucketing, an opt-in `--eval-test` protocol
gate, and an automatic run record. Nothing in `hand/` is deprecated and nothing here says it
is.

Cluster job scripts are deliberately **not** part of this repository. The launchers that
drove these runs were loops over the commands documented below; they hardcoded scheduler
accounts and hostnames and none of them is needed to reproduce a result. The three scripts
section 2 depends on — `scripts/setup_dan_fonts.sh`, `scripts/verify_dan_fonts.py` and
`scripts/setup_dataset_links.sh` — **are** here.

**One input that this repository does not ship is load-bearing**: Denis Coquenet's released
DAN line checkpoint, which initialises training. Section 2.3 says where to get it. The run
records stored under `experiments/benchmark_suite/registry/` show it at the author's own
local path `private/dan_weights/fcn_read_2016_line_syn.pt`; put it wherever you like and
pass that path to `--init-from`.

---

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-pinned.txt
pip install -e .
```

Two constraints are load-bearing and are documented in the requirements file:

- **Do not install TensorFlow.** TensorBoard probes for it on import; a TF built against
  NumPy 1.x aborts `import torch.utils.tensorboard` under NumPy 2.x. This breaks *training
  logging only* — inference is unaffected — but it breaks it silently until a run starts.
  `tools/validate_install_cpu.py` reports it as a warning rather than guessing.
- **Pillow ≥ 10** removed `FreeTypeFont.getsize()`. The synthetic-page generator used it in
  three places and was completely broken; `hand/OCR/ocr_dataset_manager.py` now uses a
  `font_getsize()` shim that reproduces the legacy geometry exactly.

The reference environment for every measurement quoted in this document, read from
`experiments/benchmark_suite/registry/20260917T160949Z_e14_budget_1p26M_s0_bd9339.json`:
Python 3.11.14, PyTorch 2.9.1+cu128, CUDA 12.8, cuDNN 9.10.2, Linux 6.8, driver 595.84.

---

## 2. Inputs you must acquire

Four inputs are git-ignored and are **not** in a fresh clone. Three are needed to train or
evaluate the page model; the fourth (`raw_READ2016/`) is what the others are built from. A
clone without any of them still imports, and still passes stages S1 and S2 of section 3.

| Input | Ignored by | Needed for |
|---|---|---|
| `raw_READ2016/` | `.gitignore:64` | rebuilding any READ dataset |
| `formatted/READ_2016_page_sem_dan/` | `.gitignore:63` | training and evaluating the page model |
| the DAN line checkpoint (see 2.3) | not shipped | initialising training (section 5) |
| `fonts_dan_read/` (41 `.ttf`) | `.gitignore:149` | the synthetic-page curriculum during training |

### 2.1 Raw READ 2016

READ 2016 (the Bozen Ratsprotokolle collection) is on Zenodo and is **CC BY 4.0**:
DOI [10.5281/zenodo.1297399](https://doi.org/10.5281/zenodo.1297399), which supersedes
[10.5281/zenodo.218236](https://doi.org/10.5281/zenodo.218236) and adds the ICFHR 2016
competition test set. Attribution is required if you redistribute any of it; this
repository redistributes none of it.

Unpack so the tree looks like this — the formatter reads exactly these paths:

```
raw_READ2016/
├── PublicData/Training/      { *.JPG, page/*.xml }
├── PublicData/Validation/    { *.JPG, page/*.xml }
└── Test-ICFHR-2016/          { *.JPG, page/*.xml }
```

### 2.2 The formatted page dataset

`formatted/READ_2016_page_sem_dan/` is derived, not downloaded. Build it:

```bash
python hand/Datasets/format_read_dan_splits.py --levels page
```

The script symlinks the raw scans into a temporary view and runs the unchanged formatter
`hand/Datasets/dataset_formatters/read2016_formatter.py`; nothing is copied or downloaded.
It asserts its own split counts and fails loudly rather than write a wrong dataset:

| Dataset | train | valid | test | charset |
|---|---|---|---|---|
| `READ_2016_page_sem_dan` | 350 | 50 | 50 | 99 |
| `READ_2016_page_sem` (V1) | 350 | 50 | 50 | 95 |

The two differ in the layout-token scheme, not in the images: `_sem_dan` carries DAN's five
tokens (`ⓟ ⓝ ⓢ ⓐ ⓑ` with their closing forms), `_sem` carries three. **They are not
interchangeable.** A checkpoint trained on one will not load against the other's charset,
and the charset sizes above are the check.

Pages in `formatted/` are already at the model's 150 dpi working resolution; the raw scans
are 300 dpi and the formatter halved them once. Anything that resizes a formatted page
again is wrong — see the docstring of `release/hand_release/inference.py`, which measured
that failure dropping a whole line of transcription while the model still looked confident.

`--levels double_page paragraph` rebuilds the other DAN-construction levels (169/24/24 and
1,602/182/199). Neither is used by the page model.

### 2.3 The DAN line checkpoint

Training does not start from random weights. It starts from Denis Coquenet's released
READ 2016 line model, which is on Zenodo under **CC BY 4.0**:

- Record **7244382**, DOI [10.5281/zenodo.7244382](https://doi.org/10.5281/zenodo.7244382)
- File `fcn_read_2016_line_syn.pt`; place it anywhere and pass that path to `--init-from`.
  The commands in section 5 use `weights/dan/fcn_read_2016_line_syn.pt`
- 20,855,873 bytes
- `md5    191a4c5659be189b2160f56f3759850c`
- `sha256 557d6c349131b113c1e316effdb028eff9b37bf39a48d0689125a3783b86a7c3`

Both digests were computed from the local copy while writing this section; check yours
against them before training.

This file is why the licence on the trained weights is **CC BY 4.0 with attribution
(section 3(a))** and not the CeCILL-C of the surrounding code. CeCILL-C Art. 1 defines
Source Code as the Software's instructions and program lines and Object Code as the binaries
produced by compiling them, and the licence has no clause about a program's output — trained
weights are neither. The binding constraint arrives through this initialisation checkpoint
instead. Do not re-host the file; link to the Zenodo record.

### 2.4 The 41 fonts

The synthetic-page curriculum draws from a specific font inventory. DAN does not publish a
font list as an input — `get_valid_fonts()` walks a tree and keeps every `.ttf` whose cmap
covers the dataset charset — so the inventory is reconstructed from primary sources:

```bash
bash scripts/setup_dan_fonts.sh        # Lato 2.0 and Gentium Plus 5.000; DejaVu 2.370 from Fonts/
python scripts/verify_dan_fonts.py     # versions, charset coverage, DAN's own selector
sha256sum -c fonts_dan_read/CHECKSUMS.sha256
```

The font binaries are not committed; `fonts_dan_read/CHECKSUMS.sha256` and
`fonts_dan_read/VERIFICATION.json` are, and they pin all 41 files (21 DejaVu 2.370, 18 Lato
2.006/2.007, 2 Gentium Plus 5.000; every one reported `covers_charset: true` over the
88-character synthetic charset). The script downloads from Debian's snapshot mirror and
SIL's download server and verifies an archive hash before extracting.

### 2.5 The other corpora

| Corpus | Status |
|---|---|
| **IAM** | register at <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>, place `lines.tgz` / `forms*.tgz` and the XML under `IAM/`, run `hand/Datasets/dataset_formatters/iam_formatter.py`. **Not redistributable.** |
| **KHATT** | request from <https://khatt.ideas2serve.net/>, unpack under `KHATT/`, run `khatt_formatter.py`. **Not redistributable.** |
| **AHAWP** | in `AHAWP_extracted/`. Read the warning in section 4.4 before reporting any AHAWP number. |
| **RIMES / MAURDOR** | formatters exist (`rimes_formatter.py`, `maurdor_formatter.py`); neither corpus is present here and both require licensed access. |

---

## 3. Validate the install — one CPU command

```bash
python tools/validate_install_cpu.py
```

No GPU, no CUDA context, no network. It runs four stages and reports each as PASS, WARN,
FAIL or SKIP, printing the acquisition instruction next to every SKIP:

| Stage | Needs | Checks |
|---|---|---|
| `S1-env` | nothing | interpreter, torch, numpy, Pillow; the two constraints in section 1 |
| `S2-arch` | `release/hand-read2016-page/config.json` only | builds the page model with **no weights and no dataset** and checks parameter counts and tensor shapes |
| `S3-weights` | an exported checkpoint | loads it `strict=True` on CPU and re-checks the counts and the charset |
| `S4-parity` | weights **and** `formatted/READ_2016_page_sem_dan/` | transcribes the first N test pages and compares per-page edit distances against `release/PARITY_CPU.json` |

**On a fresh clone with nothing downloaded, S1 and S2 run and S3/S4 skip.** That is the
useful case: S2 alone catches a broken install, a wrong torch, or a model that does not have
the architecture it claims to. Measured output of S2 on the reference environment:

```
[PASS] S2-arch  encoder 1706240 + decoder 5327460 = 7033700 parameters;
                1755x1161 page -> 55x146 visual positions; output layer 100 wide
```

7,033,700 is the training record's `n_params.total`. 55 × 146 = 8,030 is the encoder grid
that `experiments/benchmark_suite/profiling/efficiency_hand_vs_dan_uncontended.json` records
for `test_0`. The output layer is `vocab_size + additional_tokens` = 99 + 1.

With weights and the dataset in place, S4 is the accuracy check:

```bash
# export once (CPU only, no CUDA context)
python release/tools/export_release_checkpoint.py \
    --ckpt outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt \
    --out  /path/to/export

python tools/validate_install_cpu.py --model /path/to/export               # 3 pages
python tools/validate_install_cpu.py --model /path/to/export --pages 50    # full split
```

Measured on the reference host, CPU fp32, at the default 3 pages:

```
page              edits    ref    chars        s
test_0.jpeg          10     10      404    202.3
test_1.jpeg          11     11      400     13.2
test_2.jpeg          41     41      581     17.1
[PASS] S4-parity  62 edits / 1385 chars, CER 0.044765; reference 62 / 1385, CER 0.044765;
                  |delta| = 0, tolerance 2; 232.5 s total
```

A 5-page run on the same host gave 128 / 2386 against a reference of 128 / 2386, also
|delta| = 0. Note the first page: 202 s against 13-17 s for the next two. The first decode
pays for thread ramp-up and allocator warm-up, so **do not extrapolate the total from page
one** — and on a 50-page run that cost is amortised.

Four things about that output are deliberate:

- **The tolerance is 2 edits, and it comes from a measurement.** `release/PARITY_CPU.json`
  is a CPU fp32 run over all 50 test pages: 827 edits / 23,262 characters. The published
  GPU AMP fp16 artefact records 826 over the same 23,262. One edit on one page out of fifty
  is the only fp32-versus-fp16 divergence this model has been measured to have; 2 is that
  rounded up. There is **no** measurement of CPU-to-CPU divergence across machines, so a
  failure at tolerance 2 means investigate, not automatically "broken" — the per-page table
  says which page moved.
- **The character count is itself a check.** 23,262 (2,386 over the first five pages) is a
  property of the ground truth and the CER definition, not of the model. If it comes out
  different, the dataset is different and no CER below is comparable.
- **A subset CER is not the model's CER.** Those three pages score 4.48 % and the first
  five score 5.36 %; the corpus figure is 3.56 %. The command prints the subset figure and
  the corpus figure side by side and says which is which.
- **Page-level agreement is what is checked, not just the total.** Two errors of opposite
  sign on different pages would cancel in a corpus CER. The per-page column is the check.

The command writes nothing unless you pass `--out`, writes that atomically, and refuses an
`--out` path inside `release/`, `results_real/`, `experiments/` or `outputs/` — checked
before any work is done, so a bad path costs 0.26 s rather than ten minutes of decoding. It
cannot overwrite a measured artefact, on success or on failure.

The `1755x1161` in the S2 line is not arbitrary: it is the size of `test_0.jpeg`, whose
8,030 visual positions the profiling artefact records directly
(`efficiency_hand_vs_dan_uncontended.json`, `.rows["HAND_e14_1p26M::baseline"].per_page[0]`:
`img_h 1755, img_w 1161, visual_positions 8030`). Pass `--page-size H W` for a different
shape.

### The CPU test suite

```bash
python -m pytest tests -q
```

All CPU. They are equivalence tests for every replaced or vectorised function — layout
metrics against DAN's reference implementation, the fast decoding paths against the frozen
one, charset and token handling, the branch-free attention NaN replacement. They need no
dataset and no weights, and they are the check that a code change has not silently changed
what the model computes.

### What the validation command does not prove

It validates the page model's **inference** path. It does not train anything, it never
touches a GPU, and passing it says nothing about whether a training run will reproduce.

---

## 4. Evaluate a checkpoint

### 4.1 The current page model

```bash
python release/tools/evaluate_release.py --model /path/to/export --split test --device cuda
python release/tools/evaluate_release.py --model /path/to/export --split test --device cpu --tol-edits 2
```

Both device paths produce the same string; `--device cpu` runs at ~12.5 s/page on an
unloaded host (`release/PARITY_CPU.json`), so the full split is about ten minutes. That
script carries its own expected values and their sources in its docstring.

### 4.2 The V1 released checkpoints

This is how every number in `results_real/` was produced.

```bash
python tools/evaluate_hand.py --model read_page --split test --batch-size 1
python tools/evaluate_hand.py --model all --split test valid --batch-size 1
```

Writes `results_real/<model>.json` containing measured CER/WER, sample counts, the
checkpoint epoch, the validation CER the checkpoint itself recorded, and wall-clock time.
Augmentation and synthetic data are forced off; seeds are pinned. Available `--model` keys:
`read_page`, `read_double_page`, `read_triple_page`, `iam_page`, `khatt_paragraph`,
`ahawp_paragraph`. Place a checkpoint at `models/<name>/best_model.pt` and the script finds
it.

### 4.3 Fix the evaluation batch size when comparing numbers

CER is **not** strictly invariant to evaluation batch size. Samples are padded to the widest
image in their batch, so different batching perturbs the encoder output slightly. Observed
on `pg_control`: 4.43 % test at batch size 4 versus 4.42 % at batch size 1; validation was
unaffected at 3.89 % on both. The effect is ~0.01 pp and changes no conclusion, but it is
enough to make an independent rerun disagree with a published table in the last digit.
**All numbers reported from this repository use batch size 1.** Quote `--batch-size`
alongside any CER.

### 4.4 Split integrity — check before trusting any number

```bash
python tools/audit_dataset_integrity.py --json dataset_audit.json --fail-on-violation
python tools/audit_corpus_text_overlap.py          # text COVERAGE, a different question
python tools/check_splits.py --json results_real/split_integrity.json
```

| Corpus | Verdict |
|---|---|
| READ 2016 (all levels), IAM (line, page) | clean — no transcription or document overlap |
| `KHATT_paragraph` | acceptable — 7 of 212 distinct test transcriptions (3.3 %), 8 of 213 instances (3.8 %) |
| `KHATT_line` | **contaminated** — 230 of 744 distinct test transcriptions (30.9 %), **470 of 999 test instances (47.0 %)**. Evaluation scores instances, so 47.0 % is the figure that bounds any line-level CER here |
| `AHAWP_character` / `AHAWP_word` / `AHAWP_paragraph` | **violation** — 65 / 10 / 3 distinct transcriptions in total, all shared across every split |

AHAWP results are memorisation, not recognition, and must not be reported as generalisation.

Transcription overlap and text *coverage* are different questions, and the two audits
disagree on KHATT because they measure different things — `dataset_audit.json` counts whole
transcriptions, `corpus_text_overlap.json` counts character 20-grams. Mean 20-gram
test-in-train coverage, test split, from
`experiments/p0_3_khatt_l6_audit/corpus_text_overlap.json`:

| Corpus | paragraph | line |
|---|---|---|
| READ 2016 `_sem_dan` (control) | 0.065 | — |
| RIMES | 0.394 | 0.402 |
| KHATT | 0.892 | 0.949 |
| **IAM Aachen** | **0.0007** | **0.0003** |

**IAM Aachen is the one corpus here whose text supports an unseen-text claim**, and it is
the one this repository uses for that purpose. RIMES at 0.394 against a control of 0.065 is
not text-disjoint: a RIMES result is partly recall of training text and must not be
presented as unseen-text performance. Cite both audits; neither substitutes for the other.

---

## 5. Train the current page model

**This section needs a GPU and READ 2016.** It cannot be run on CPU in any useful sense: the
two phases below took 91,899 s and 229,150 s — about 89 hours in total — on a single
RTX PRO 6000 Blackwell. Both figures are the `train_seconds` fields of the run records named
below. There is no CPU path and this document does not pretend there is one.

The reported page model is a **two-phase run**, and the phases are not independent.

### Phase A — 1,429 epochs

```bash
python tools/train.py \
    --dataset READ_2016 --level page --variant _sem_dan --encoder fcn \
    --output s1_A1fixedR1_s0 \
    --batch-size 1 --lr 1e-4 --curr-step 10000 \
    --syn-min-lines 1 --syn-max-lines 30 \
    --syn-init-proba 0.9 --syn-end-proba 0.2 --syn-proba-steps 200000 \
    --no-hand-encoding --additional-tokens 1 \
    --init-from weights/dan/fcn_read_2016_line_syn.pt \
    --fonts-dir fonts_dan_read \
    --max-epochs 1429 --max-hours 240 --eval-interval 5 --seed 0 \
    --experiment v2_stage1_recovery
```

Record: `experiments/benchmark_suite/registry/20260911T032022Z_s1_A1fixedR1_s0_b09a58.json`.
Best epoch 1390, last epoch 1428, 91,899 s.

### The step between the phases — do not skip it

Phase B runs with `--resume`, and `--resume` continues from the last checkpoint **in its own
output folder**. Phase B uses a different output folder, so Phase A's run directory must be
copied first:

```bash
cp -r outputs/s1_A1fixedR1_s0 outputs/e14_budget_1p26M_s0
```

Omitting this is the most damaging mistake available in this document, and it is a **silent**
one: Phase B would find no checkpoint, fall back to `--init-from`, and train 3,596 epochs
from the DAN line initialisation instead of 2,167 further epochs from Phase A's page model.
It would run, it would produce a `best_*.pt`, and the number would be wrong. The same
convention is used for every continuation run here — see the note recorded on
`b1_budget_cont_s0`: "Copy of e14; e14 itself is never written to."

### Phase B — to 3,596 epochs

```bash
python tools/train.py \
    --dataset READ_2016 --level page --variant _sem_dan --encoder fcn \
    --output e14_budget_1p26M_s0 \
    --batch-size 1 --lr 1e-4 --curr-step 10000 \
    --syn-min-lines 1 --syn-max-lines 30 \
    --syn-init-proba 0.9 --syn-end-proba 0.2 --syn-proba-steps 200000 \
    --no-hand-encoding --additional-tokens 1 \
    --init-from weights/dan/fcn_read_2016_line_syn.pt \
    --fonts-dir fonts_dan_read \
    --max-epochs 3596 --max-hours 240 --eval-interval 5 --seed 0 --resume \
    --experiment v2_e1.4_budget
```

That is the argv the run record stores verbatim. **Add `--eval-test` if you rerun it today**:
the test split was evaluated unconditionally when this run executed, and the opt-in gate
described below was added afterwards, so a rerun without the flag now stops at validation.
The flag is present in the working tree and is **not yet committed** — check
`python tools/train.py --help` for it before relying on either behaviour.

3,596 epochs × 350 training pages = 1,258,600 samples, which is DAN's published budget.
Record: `experiments/benchmark_suite/registry/20260917T160949Z_e14_budget_1p26M_s0_bd9339.json`.
Best epoch 3580, last 3595, 229,150 s, `n_params.total` 7,033,700.

Result, read from `outputs/e14_budget_1p26M_s0/results/predict_READ_2016-{test,valid}_3580.txt`:

| Split | n | chars | CER | WER |
|---|---|---|---|---|
| test | 50 | 23,262 | 3.55 % | 13.31 % |
| valid | 50 | 21,609 | 3.95 % | 15.01 % |

### Two things to know before interpreting that number

- **The test split is opened once.** `--eval-test` is off by default in the working-tree
  `tools/train.py` precisely so that a probe or screening run cannot quietly write a test
  metric into the registry; two screening runs did so before the gate existed, which is
  why the gate exists. Under that gate a run reporting a test number must have asked for it at launch, on
  the record, and its metrics carry `test_evaluated`. The `e14` record predates the gate and
  carries no such field; its test numbers were produced by the unconditional behaviour, and
  the phase-B command above is the argv as recorded.
- **The seed spread is larger than most differences you will want to read into.** Over
  n = 3 seeds: test 4.574 ± 0.388 (spread 0.70 pp), validation 4.47 ± 0.06. The smallest
  effect three seeds resolve is 0.887 pp on test and 0.122 pp on validation. An earlier
  "0.3 pp noise floor" is superseded and should not be quoted.

### The budget question is settled

`b1_budget_cont_s0` resumed the finished run for a further ~550 epochs (+15.3 % budget) and
completed at epoch 4,145. Its best epoch is **3,580 — the same epoch — at validation CER
0.0395, unchanged.** Running roughly 190k samples past DAN's budget produced no improvement;
the budget curve is flat at that point. Record:
`experiments/benchmark_suite/registry/20260924T030209Z_b1_budget_cont_s0_1485fa.json`
(`test_evaluated: false`, by protocol).

---

## 6. The V1 training path

`tools/train_hand.py` produced the checkpoints in `models/` and the evaluations in
`results_real/`. It is still the argument parser and parameter builder that
`tools/train.py` imports, so its flags are the V2 flags.

```bash
python tools/train_hand.py \
    --dataset READ_2016 --level page --variant _sem \
    --encoder fcn --output read_page_fcn \
    --batch-size 8 --lr 1e-4 --max-hours 10 --seed 0
```

| Flag | Meaning |
|---|---|
| `--encoder {fcn,hand}` | `fcn` = DAN's `FCN_Encoder`, the encoder the manuscript's architecture section describes and every released checkpoint uses; `hand` = the gated DSC + octave + SE variant, kept only to reproduce the negative result in `ablations.md` §1.1, **not recommended** |
| `--no-octave` / `--no-se` / `--no-gate` / `--no-residual` | ablate one component of the FasterHAND encoder |
| `--init-from PATH` | warm-start encoder+decoder from a checkpoint |
| `--no-synthetic` / `--no-augment` | disable synthetic pages / augmentation |
| `--resume` | continue from the last checkpoint **in the output folder** |
| `--smoke` | tiny run that verifies train → checkpoint → evaluate, then exits |

Each run writes `outputs/<name>/` with `run_command.json` (argv, resolved args, torch
version, GPU, code version, start time), `results/params.txt`, `checkpoints/best_*.pt` and
`last_*.pt`, and `results/predict_*.txt` per split. After training the script reloads the
**best** checkpoint; if no `best_*.pt` exists it falls back to `last_*.pt` and says so
loudly.

`--elastic-depths 2 4 6 8` trains a shared-weight decoder stack usable at several depths;
`tools/evaluate_depth_sweep.py --run <name> --split test valid` writes
`results_real/depth_sweep_<run>.json`.

---

## 7. Regenerate the paper tables

```bash
python tools/make_tables.py --out paper_tables
```

Reads `results_real/*.json`, the run record of the current page model, the checkpoints in
`models/`, and the stored per-sample prediction files. Emits `main_results.tex`,
`headline_read2016_page.tex`, `efficiency.tex`, `encoder_ablation.tex`,
`postcorrection.tex`, plus two audit files:

- **`SOURCES.md`** — one row per emitted number, with the artefact path and the key inside
  it. A reader can check any cell without reading the generator.
- **`PENDING.md`** — every cell that printed `--`, with the reason.

Two runs over an unchanged tree produce **byte-identical** output. That is a property the
generator did not previously have. It keyed a dict on the `model` field inside each JSON
while iterating an unsorted `glob.glob`, and that field is not unique: five files declare
`"model": "read_page"`, and four each declare the double- and triple-page names. Whichever
file the filesystem happened to return last won, so all three READ rows reported a
coverage-sweep variant — 3.96 / 4.73 / 8.50, which are `read_page_cov2.0`,
`read_double_page_cov4.0` and `read_triple_page_cov2.0` — instead of the runs named in the
rows. Runs are now identified by filename, the glob is sorted, prediction files are chosen
by numeric epoch rather than string sort, and duplicated `model` fields are reported in
`PENDING.md`.

Table generation reads measurements only. Any earlier script that emitted tables from a
hardcoded dict of numbers has been removed from the tracked tree; do not reintroduce one.

---

## 8. Measured results — V1 released checkpoints

Every row is read straight out of `results_real/*.json`; regenerate it with section 7 rather
than editing it by hand. Batch size 1 throughout. **These are the V1 checkpoints. The
current page model is in section 5.**

| Run | Split | n | CER | WER |
|---|---|---|---|---|
| `ahawp_paragraph` | test | 37 | 2.56 % | 3.09 % |
| `ahawp_paragraph` | valid | 36 | 0.00 % | 0.00 % |
| `iam_page` | test | 232 | 6.34 % | 17.62 % |
| `iam_page` | valid | 105 | 4.92 % | 15.97 % |
| `khatt_paragraph` | test | 213 | 21.89 % | 30.87 % |
| `khatt_paragraph` | valid | 220 | 16.62 % | 24.90 % |
| `read_double_page` | valid | 25 | 3.68 % | 15.12 % |
| `read_double_page_cov2.0` | valid | 25 | 4.39 % | 15.86 % |
| `read_double_page_cov4.0` | valid | 25 | 4.73 % | 16.43 % |
| `read_double_page_cov8.0` | valid | 25 | 3.90 % | 15.38 % |
| `read_page` | valid | 50 | 3.87 % | 15.29 % |
| `read_page_beam3` | test | 50 | 4.53 % | 15.09 % |
| `read_page_beam3` | valid | 50 | 3.91 % | 15.34 % |
| `read_page_cov2.0` | valid | 50 | 3.96 % | 15.39 % |
| `read_page_cov4.0` | valid | 50 | 4.05 % | 15.26 % |
| `read_page_cov8.0` | valid | 50 | 4.09 % | 15.36 % |
| `read_triple_page` | valid | 16 | 10.00 % | 21.45 % |
| `read_triple_page_cov2.0` | valid | 16 | 8.50 % | 19.87 % |
| `read_triple_page_cov4.0` | valid | 16 | 9.09 % | 20.59 % |
| `read_triple_page_cov8.0` | valid | 16 | 9.52 % | 20.88 % |
| `run_abl_line_fcn` | test | 685 | 4.65 % | 12.55 % |
| `run_pg_control` | test | 50 | 4.42 % | 15.35 % |
| `run_pg_control` | valid | 50 | 3.89 % | 15.54 % |
| `run_pg_repaired` | test | 50 | 4.28 % | 15.07 % |
| `run_pg_repaired` | valid | 50 | 4.18 % | 15.47 % |
| `run_pg_tf04` | test | 50 | 5.10 % | 16.24 % |
| `run_pg_tf04` | valid | 50 | 3.90 % | 15.36 % |
| `run_pg_tf06` | test | 50 | 4.37 % | 15.05 % |
| `run_pg_tf06` | valid | 50 | 4.01 % | 15.36 % |
| `run_read_page_ft_control` | test | 50 | 4.42 % | 15.35 % |
| `run_read_page_ft_control` | valid | 50 | 3.89 % | 15.54 % |

Three gaps to be aware of, all visible above:

- **`read_page` / `read_double_page` / `read_triple_page` have no test-split entry.** The
  coverage sweep overwrote those JSONs with validation-only runs. Re-run
  `python tools/evaluate_hand.py --model read_page read_double_page read_triple_page --split test`
  before quoting a V1 READ test number; the only stored V1 READ test measurement is
  `read_page_beam3` (4.53 %, beam width 3, not the greedy baseline).
- **Several files share a `model` field.** `read_page_beam3` and the three `read_page_cov*`
  files all declare `"model": "read_page"`. They are separate runs with different decoding
  settings. Identify a run by its filename, never by that field — see section 7.
- **`ahawp_paragraph` validation CER is 0.00 %.** That is memorisation, not recognition; see
  section 4.4.

### Post-OCR correction — no claim, pending re-measurement

A set of post-OCR correction experiments (ByT5 conservative / selective / line-by-line /
direct, T5 spelling, mT5-small) was run against the READ 2016 page predictions. A previous
pass indicated that every configuration scored worse than the uncorrected output, but the
pickles behind those numbers are not in this repository and no JSON artefact reproduces
them, so no figure is quoted here. `post_correction/` contains the code. Until it writes
measurements into `results_real/`, treat post-OCR correction as **unmeasured** — in
particular, do not report an improvement from it.

---

## 9. Hardware, and what cannot be reproduced here

| Task | Needs | Cost | Artefact |
|---|---|---|---|
| `tools/validate_install_cpu.py` S1+S2 | nothing beyond the clone | seconds | — |
| `tools/validate_install_cpu.py` S4, 3 pages | weights + READ 2016 | 232 s measured here (202 s of it the first page's warm-up), against ~38 s implied by an unloaded 12.55 s/page. **CPU contention and first-page warm-up both dominate a short run**; on a host running other people's jobs single pages measured anywhere from 12.6 s to 305 s. Budget generously | `release/PARITY_CPU.json` for the unloaded per-page figure; the rest measured while writing this section |
| full 50-page CPU evaluation | weights + READ 2016 | ~10 min unloaded (627.6 s, 12.55 s/page) | `release/PARITY_CPU.json` |
| GPU page decoding, greedy, AMP | 1 GPU | **1.960 s/page** | `experiments/benchmark_suite/profiling/spec_decode_test.json` `.arms.greedy.latency_s_per_page` |
| GPU page decoding, speculative m=5 | 1 GPU | **0.717 s/page**, 2.73×, token-identical on 50/50 pages | same file, `.arms.spec_m5` |
| GPU page decoding, greedy, AMP, second harness | 1 GPU | **2.415 s/page** (kv-cache 2.136), peak 1,229 MiB allocated | `efficiency_hand_vs_dan_uncontended.json` `.rows["HAND_e14_1p26M::baseline"]` |
| **training, section 5** | **1 GPU, ~89 h** | not reproducible on CPU | the two `train_seconds` fields in section 5 |

The two dedicated harnesses disagree (1.960 against 2.415 s/page) because they are different
harnesses measured in different sessions on the same checkpoint — both are honest numbers and
neither is "the" latency. The `sample_time` fields inside the training records (11.75 s/page
for `e14`, 2.13 s/page for `b1_budget_cont_s0`) are end-of-training evaluations taken while
the GPU was shared and are not comparable to either. **Quote a latency figure with the
artefact and the session it came from, never on its own.**

All GPU measurements quoted in this document: single NVIDIA RTX PRO 6000 Blackwell (97 GB),
CUDA 12.8, PyTorch 2.9.1, batch size 1 for evaluation.

Stated plainly, so nobody discovers it the hard way:

- **Training cannot be reproduced without a GPU.** Section 5 is ~89 GPU-hours.
- **Nothing on READ 2016, IAM or KHATT can be reproduced without the corpus.** READ 2016 is
  CC BY 4.0 and freely downloadable; IAM and KHATT require registration and are not
  redistributed here, so their numbers cannot be checked by a reader who has not registered.
- **Checkpoints are not kept in git.** The four READ 2016 models are GitHub release assets
  (`release/hand_release/hub.py` downloads and verifies them); the IAM and KHATT models are not
  released. Sections 3 and 4.1 describe how to use one.
- **The `hand` encoder (`--encoder hand`) is under evaluation.** No released checkpoint uses
  it, and the ablation table generated in section 7 is not trained to convergence and is not
  at a common epoch across its rows — `PENDING.md` says so.

---

## 10. Efficiency and exactness — the E3 / E4 / E4+E3 claims

Every command writes one JSON under `experiments/benchmark_suite/profiling/`, and that JSON
is the artefact the paper cites. [`REPRODUCIBILITY_ARTIFACTS.md`](REPRODUCIBILITY_ARTIFACTS.md)
maps each one to the table and claim it supports.

```bash
# full cost row: parameters, peak memory, FLOPs over the REAL decode, latency +- sd,
# throughput, tokens generated, and CER/WER/LOER/mAP-CER, all in one process
python tools/efficiency_bench.py --n-pages 50 --split test --out <out>.json

# the four-arm comparison in one process, each arm verified page-by-page against baseline
python tools/efficiency_combined.py --n-pages 50 --warmup 2 --split test --m 5 \
    --modes baseline,e2,e3,e2_e3 --out <out>.json

# speculative decoding: latency, passes, tokens/pass, CER/WER/LOER/mAP-CER,
# AND per-page identity against greedy
python tools/spec_decode.py --m 5 --split test --out <out>.json

# train the m-1 draft heads on a FROZEN base (encoder and decoder in eval, no_grad,
# gradients only in the heads), so the base model's output cannot change
python tools/train_spec_heads.py --m 5 --out <out>.json

# the exactness gate: every page decoded under the reference path and under each
# optimisation, in one process, requiring character-for-character identity
python tools/verify_exact_decoding.py

# between-seed spread, per-page correlation, paired bootstrap contrasts with CI and p.
# This is what makes a CER comparison admissible in this repository.
python tools/seed_variance_analysis.py
```

Each GPU command above assumes an otherwise idle device. On a shared host, serialise them
behind a lock (`flock /tmp/hand-gpu0.lock ...`) and discard anything measured under
contention.

### What the speed-up numbers mean, exactly

| Configuration | Precision | Speed-up | Pages identical to greedy |
|---|---|---|---|
| **E3 on the full base** (7,033,700 + 365,968, *m* = 5) | AMP fp16 | **2.73×** | **50 / 50** |
| **E4 + E3** (6,112,612 + 365,968, *m* = 5) | AMP fp16 | **2.886×** | **48 / 50** |
| **E4 + E3** (6,112,612 + 365,968, *m* = 5) | fp32 | **2.947×** | **50 / 50** |

**Do not collapse these into one number.** The 48/50 result is bounded at ≤ 0.01 pp CER with
LOER unmoved, and it disappears in fp32: `tokens_per_pass` agrees between the two precisions
to three decimals, so the accept/reject sequence is identical and only floating-point
rounding differs. "Token-identical under AMP" is licensed for **E3 on the full base** and
must not be carried to the E4 + E3 configuration.

### Two standing rules

- **Never compare latency across sessions.** Absolute latency on the reference host moves up
  to 9 %. This repository deliberately reports two different greedy figures — **1.960 s/page**
  (`spec_decode_test.json`) and **2.415 s/page** (`efficiency_hand_vs_dan_uncontended.json`)
  — from two harnesses, rather than picking one. Quote a latency with its artefact and its
  session, never on its own.
- **A cost change must prove it changed nothing else.** `tools/verify_exact_decoding.py` and
  `tests/test_fast_decode_paths.py` are the gates. A speed claim without the identity
  column is not admissible here.

### Training flags added by the efficiency work

| Flag | Default | Effect |
|---|---|---|
| `--grad-clip` | 1.0 | unchanged behaviour; `0` disables clipping |
| `--label-smoothing` | 0.0 | unchanged behaviour; ε on the decoder cross-entropy |
| `--share-memory-kv` | off | one visual K/V projection shared by all decoder layers: 7,033,700 → 6,112,612 (E4) |
| `HAND_FAST_MASKS`, `HAND_FAST_STEP` | on | exact decode-path optimisations; `0` restores the reference path for an A/B |

**Every one defaults to the behaviour that produced the existing results**, so no previously
reported number depends on any of them.

---

## 11. Optional development utilities

Nothing in this section produces a number in the paper. It is published so the reported
results can be interrogated, not because a reproduction needs it.

| Path | What it is for |
|---|---|
| `hand/models/experimental/` | designed components that are **inactive on the trained path** — the gated/octave FasterHAND encoder, MSAP, memory-augmented and sparse attention, adaptive fusion. No checkpoint uses them and the manuscript's architecture section does not describe them; published so the manuscript's training-strategy and supplementary sections can be checked against code |
| `tools/dump_predictions.py`, `tools/recompute_layout_metrics.py` | prediction dumps and layout-metric recomputation |
| `hand/Datasets/format_read_dan_splits.py`, `hand/Datasets/bucketing.py` | dataset formatting (single, double and triple page) and shape bucketing |
| `tests/` | the CPU test suite — `python -m pytest tests -q` |
| `tools/make_tables.py` | regenerates the LaTeX tables, `SOURCES.md` and `PENDING.md` from the measurements (section 7) |
| `release/tools/generate_manifest.py` | regenerates `release/MANIFEST.md` with checksums |
| `release/hand_release/inference.py` | the inference contract: image in, `{text, raw, regions, confidence, latency_s}` out |

---

## 12. Multi-page experiments (double- and triple-page READ 2016)

Everything for the scaling study of the manuscript's multi-page section lives under
`experiments/multipage/`: `README.md` there records the checkpoints (paths and SHA-256), the
dataset construction (`formatted/READ_2016_double_page_sem_dan`, DAN pairing, 169/24/24;
`formatted/READ_2016_triple_page_sem_dan`, three consecutive scans, 116/16/16, manifest
`experiments/benchmark_suite/manifests/READ_2016_triple_page_sem_dan.json`), the exact
commands, and the caveats. The evaluation harness is `tools/multipage_eval.py`; the triple-page
partition is built by `hand/Datasets/format_read_dan_splits.py --levels triple_page`.

## 13. Provenance of the DAN page checkpoint used for the paired comparison

The DAN single-page numbers measured in this repository (`anchor_dan_official.json`,
`efficiency_hand_vs_dan*.json`) load the DAN authors' public READ 2016 page checkpoint
`dan_read_page.pt` (md5 `f2779a5887c7a8d7b81255059392dd51`, epoch 3595, 1,258,600 training
samples, 7,033,700 parameters). It is not redistributed here; obtain it from the DAN authors'
release alongside the line checkpoint of section 2.3 and verify the md5 before use.
