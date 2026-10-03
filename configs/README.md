# configs/

## What is here

| File | Contents |
|---|---|
| `datasets_registry.yaml` | one entry per corpus: where the raw data lives, the formatted variants derived from it, level, script, licence |

## Where the training configuration actually lives

**HAND is configured by argv, not by config files.** There is no `configs/train/` or
`configs/eval/` because no such file was ever the source of truth for a reported number —
the argv was.

That argv is not lost. Every run writes a record to
`experiments/benchmark_suite/registry/<timestamp>_<name>_<hash>.json`, and that record
stores the **complete argv verbatim** alongside the environment, the parameter counts, the
epoch reached and the metrics. To reproduce a run, read its record:

```bash
python -c "import json,sys; r=json.load(open(sys.argv[1])); print(' '.join(r['argv']))" \
  experiments/benchmark_suite/registry/20260917T160949Z_e14_budget_1p26M_s0_bd9339.json
```

The two commands that produced the reported page model are transcribed in
[`docs/reproducibility.md`](../docs/reproducibility.md) section 5.

Evaluation is likewise argv: `release/tools/evaluate_release.py` and
`tools/evaluate_hand.py`, both at `--batch-size 1`. See `docs/reproducibility.md` section 4.

## Model configuration

The exported-checkpoint format carries its own `config.json` — see
`release/hand-read2016-page/config.json`, which is also what
`tools/validate_install_cpu.py` stage S2 builds the model from, with no weights and no
dataset.
