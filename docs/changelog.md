# Changelog

## 1.1.0 — released weights and demo (2026-10-03)

- The four READ 2016 models (single page, compact single page, double-page adapted,
  triple-page adapted) are attached to the GitHub release as CC BY 4.0 packages, with draft
  heads where they exist. Each was checked to reproduce its stored example prediction token for
  token. `release/hand_release/hub.py` downloads them and verifies the pinned SHA-256.
- The same weights are published as the Hugging Face models `MHamdan/hand-read2016-*`, loadable
  with `HANDRecognizer.from_pretrained`.
- `demo/app.py`: Gradio demo with three READ 2016 example images (single, double, triple page).
- `safetensors` added to the requirements; it is needed to load the released weights.

## 1.0.0 — initial public release (2026-10-02)

First public release of the code and evidence for
*HAND: Unified Text–Layout Decoding for Handwritten Document Recognition*. No experiment was
rerun for the release and no measured value was changed.

### Contents
- `hand/`, `tools/`, `tests/`, `release/hand_release/` — the model, training, evaluation and
  inference code behind every reported number, and the CPU test suite.
- `experiments/` — run records, profiling measurements and the multi-page scaling study
  (single-, double- and triple-page READ 2016 evaluations, with and without adaptation).
- `results_real/`, `results_recovery/`, `dataset_audit.json` — stored metrics and the dataset
  overlap audit.
- `docs/` — reproduction guide, ablations and negative results, per-artefact provenance, model
  card, and a qualitative gallery of examples not included in the PDFs.
- `release/NOTICE.md`, `release/MANIFEST.md` — CeCILL-C notice for the DAN-derived files, CC BY 4.0
  attribution for the initialisation checkpoint, and a SHA-256 digest for every published file.

### Licensing
- Nine DAN-derivative source files carry the upstream CeCILL-C notice byte-identical to upstream,
  and the twenty-four files that are measurably modified carry an Art. 5.2 modification statement
  (`release/NOTICE.md` §1).

### Redactions
- Absolute paths, a hostname, user names, a local weight-directory name and pointers to
  unpublished planning notes were removed from published run records. Each redaction is
  metadata-only and changes no measured value; `docs/REPRODUCIBILITY_ARTIFACTS.md` §5 lists them.

### Not distributed
- Datasets (READ 2016, IAM, KHATT) — see `data/README.md` for sources and preparation.
- The manuscript and supplementary material, which are under review.
- Per-sample IAM and KHATT prediction dumps, which contain registration-licensed ground truth.
