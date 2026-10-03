# models/

**This directory is empty in a clone.** Everything in it except this file is git-ignored.

## Released weights

Checkpoints are not kept in git (`/models/`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors` are
ignored). The exported READ 2016 models are assets of the GitHub release v1.1.0; download one
with `cd release && python -m hand_release.hub <name>`. The list, digests and reported results
are in the README's *Model weights* section and in [`docs/model_card.md`](../docs/model_card.md).

## Where a checkpoint goes

Place one at `models/<name>/best_model.pt` and `tools/evaluate_hand.py` will find it:

```bash
python tools/evaluate_hand.py --model read_page --split test --batch-size 1
```

Recognised names: `read_page`, `read_double_page`, `read_triple_page`, `iam_page`,
`khatt_paragraph`, `ahawp_paragraph`.

Every evaluation records a **fingerprint of the weights it loaded** — parameter counts,
decoder depth, architecture markers — so a result can always be traced to the exact
checkpoint that produced it.

For the current page model, export first and then evaluate against the export:

```bash
python release/tools/export_release_checkpoint.py \
    --ckpt outputs/<run>/checkpoints/best_<epoch>.pt --out /path/to/export
python release/tools/evaluate_release.py --model /path/to/export --split test --device cuda
```

## The weights are not MIT

They initialise from Denis Coquenet's released READ 2016 line checkpoint (Zenodo
[10.5281/zenodo.7244382](https://doi.org/10.5281/zenodo.7244382), **CC BY 4.0**), so
**CC BY 4.0 §3(a) attribution is the binding constraint on them** — not the repository's
MIT licence, which covers this author's code only.

Read [`release/NOTICE.md`](../release/NOTICE.md) before redistributing any checkpoint
derived from this tree.
