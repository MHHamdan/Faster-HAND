# hand_v2

Training and evaluation layer on top of `hand/`. It contains **no model code of its own**: the
model, dataset formatters and trainers live in `hand/`, which this package imports, subclasses
and wraps. `hand_v2/train.py` is the entry point that trained the reported page model; the name
refers to this second engineering layer, not to a second version of the model.

| Module | Purpose |
|---|---|
| `profiling/profile_pipeline.py` | measures where a training step's wall-clock goes: per-sample input cost, DataLoader throughput vs. workers, the model's own step ceiling, end-to-end throughput with GPU/CPU sampled, autoregressive eval cost |
| `profiling/gpu_monitor.py` | background NVML + psutil sampler used by the above |
| `profiling/attribute_step.py` | counts every host-device synchronisation in a training step and attributes it to a source line |
| `profiling/cudnn_shape_bench.py` | cost of unseen input shapes per cuDNN strategy (v8 heuristic, benchmark, v7 API, bucketing) |
| `profiling/eval_page_timing.py` | autoregressive page decoding time, vectorised vs reference line-index function |
| `data/bucketing.py` | pads training batches to shape multiples so cuDNN sees few distinct shapes (`--shape-bucket H W`) |
| `tests/` | equivalence tests for every replaced function |
| `train.py` | V2 training entry point: `tools/train_hand.py`'s flags, plus `--cudnn-api {v7,v8}` (default v7), `--shape-bucket H W`, `--experiment`, `--notes`, and automatic recording into `experiments/benchmark_suite/` |

Measurements that justify anything in `data/` live in
`experiments/benchmark_suite/profiling/` and are summarised in `docs/ablations.md`.
