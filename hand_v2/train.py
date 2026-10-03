#!/usr/bin/env python3
"""
HAND V2 training entry point.

Same flags as tools/train_hand.py (they are imported, not copied), plus:

  --cudnn-api {v7,v8}          v7 (default) avoids cuDNN 9's per-shape plan building
  --shape-bucket H W           pad training batches to shape multiples (for v8); 0 0 = off
  --experiment ID              experiment tag stored in the run record (e.g. exp2.1, v2_e1)
  --notes TEXT                 free text stored in the run record

Every run is recorded in experiments/benchmark_suite/registry/ before training starts
(status "running") and again when it ends, with config, commit, dataset, seed, GPU and
metrics. Stage 0 adds nothing to the model: the encoder, decoder, loss, optimiser and
schedule are exactly those of tools/train_hand.py.
"""
import json
import os
import sys
import time
import traceback

# --cudnn-api must act before torch is imported: cuDNN's default v8 heuristic path spends
# ~1.3-1.8 s building a plan for every unseen input shape (and every padded batch is a new
# shape); the legacy v7 API has no such cost and the same steady-state speed at page
# scale (experiments/benchmark_suite/profiling/cudnn_shape_bench*.json).
if "--cudnn-api" in sys.argv:
    _api = sys.argv[sys.argv.index("--cudnn-api") + 1]
    if _api == "v7":
        os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
elif os.environ.get("TORCH_CUDNN_V8_API_DISABLED") is None:
    os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"          # default: v7

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.train_hand import LEVEL_DEFAULTS, build_arg_parser, build_params, git_or_mtime_stamp, set_seed  # noqa: E402
from experiments.benchmark_suite.record import RunRecord, parse_predict_file  # noqa: E402


def dataset_descriptor(a, data_path):
    d = {"name": a.dataset, "level": a.level, "variant": a.variant,
         "id": "{}_{}{}".format(a.dataset, a.level, a.variant), "path": data_path, "splits": {}}
    for split in ("train", "valid", "test"):
        p = os.path.join(data_path, split)
        d["splits"][split] = len(os.listdir(p)) if os.path.isdir(p) else 0
    return d


def main():
    ap = build_arg_parser()
    ap.add_argument("--experiment", default=None)
    # Protocol: the test split is opened ONCE, after the recipe is frozen. Diagnostic and probe
    # runs must not evaluate it -- see experiments/PROVENANCE_AUDIT.md section 5, where two probes
    # were found to have written test metrics into the registry. Test evaluation is therefore
    # OPT-IN: a run that reports test numbers must say so at launch, on the record.
    ap.add_argument("--eval-test", action="store_true",
                    help="evaluate the test split at the end of training. Off by default. Pass it "
                         "only for a run explicitly registered as a test evaluation; every "
                         "diagnostic, probe or screening run must leave it off.")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--cudnn-api", default="v7", choices=["v7", "v8"],
                    help="v7 (default): legacy cuDNN API, no per-shape plan building. v8: "
                         "PyTorch's default; pair it with --shape-bucket or pay ~1.3-1.8 s "
                         "per unseen input shape")
    ap.add_argument("--shape-bucket", type=int, nargs=2, default=[0, 0], metavar=("H", "W"),
                    help="pad training batches up to multiples of (H, W) so cuDNN sees few "
                         "distinct shapes; 0 0 disables (default). Suggested with v8: 32 128")
    # Stage 1 recipe overrides (arm B: DAN's synthetic-curriculum settings). Defaults keep the
    # repository recipe unchanged; see experiments/v2_stage1_baseline.md.
    ap.add_argument("--syn-min-lines", type=int, default=None,
                    help="curriculum start line count for synthetic documents (repo page default 5; DAN 1)")
    ap.add_argument("--syn-max-lines", type=int, default=None,
                    help="curriculum end line count (repo page default 30 = DAN)")
    ap.add_argument("--syn-proba-steps", type=int, default=None,
                    help="samples over which the synthetic ratio decays init->end (repo 100000; DAN 200000)")
    ap.add_argument("--fonts-dir", default=None,
                    help="directory of .ttf files to use for synthetic text (sets HAND_FONTS_PATH); "
                         "default: the repository's Fonts/ (154 fonts)")
    a = ap.parse_args()
    if a.fonts_dir:
        os.environ["HAND_FONTS_PATH"] = os.path.abspath(a.fonts_dir)

    if a.smoke:
        a.max_hours = min(a.max_hours, 0.05)
        a.max_epochs = min(a.max_epochs, 2)
        a.eval_interval = 1

    os.chdir(ROOT)
    set_seed(a.seed)               # cudnn.deterministic=True, benchmark=False, as the baseline
    params = build_params(a)
    syn = params["dataset_params"]["config"].get("synthetic_data")
    if syn:
        if a.syn_min_lines is not None:
            syn["min_nb_lines"] = a.syn_min_lines
        if a.syn_max_lines is not None:
            syn["max_nb_lines"] = a.syn_max_lines
        if a.syn_proba_steps is not None:
            syn["num_steps_proba"] = a.syn_proba_steps
    if a.shape_bucket[0] > 1 or a.shape_bucket[1] > 1:
        from hand_v2.data.bucketing import install_shape_bucketing
        install_shape_bucketing(params, *a.shape_bucket)

    run_dir = os.path.join(ROOT, "outputs", a.output)
    os.makedirs(run_dir, exist_ok=True)
    data_path = list(params["dataset_params"]["datasets"].values())[0]
    with open(os.path.join(run_dir, "run_command.json"), "w") as f:
        json.dump({"argv": sys.argv, "args": vars(a), "code_version": git_or_mtime_stamp(),
                   "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "entry": "hand_v2/train.py"}, f, indent=2)

    rec = RunRecord.start(name=a.output, kind="train", experiment=a.experiment,
                          dataset=dataset_descriptor(a, data_path), seed=a.seed, config=vars(a),
                          notes=a.notes, output_dir=os.path.relpath(run_dir, ROOT))
    print("=" * 72)
    print("HAND V2 training | dataset={} level={}{} encoder={} cudnn_api={} shape_bucket={}".format(
        a.dataset, a.level, a.variant, a.encoder, a.cudnn_api, tuple(a.shape_bucket)))
    print("output=outputs/{}  batch={}  lr={}  seed={}  hand_encoding={}  run_id={}".format(
        a.output, a.batch_size, a.lr, a.seed,
        LEVEL_DEFAULTS[a.level]["hand_encoding"] and not a.no_hand_encoding, rec.run_id))
    print("=" * 72, flush=True)

    from hand.OCR.document_OCR.hand.trainer_std_hand import Manager
    t0 = time.time()
    try:
        manager = Manager(params)
        manager.load_model()
        n_params = {k: sum(p.numel() for p in m.parameters()) for k, m in manager.models.items()}
        n_params["total"] = sum(n_params.values())
        print("Parameters: encoder={:.3f}M decoder={:.3f}M total={:.3f}M".format(
            n_params["encoder"] / 1e6, n_params["decoder"] / 1e6, n_params["total"] / 1e6), flush=True)
        rec.update(n_params=n_params, resumed_from_epoch=manager.latest_epoch)

        manager.train()
        train_seconds = time.time() - t0

        ckpt_dir = os.path.join(run_dir, "checkpoints")
        available = os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else []
        which = "best" if any("best" in f for f in available) else ("last" if any("last" in f for f in available) else None)
        if which is None:
            rec.finish(metrics={}, status="failed", train_seconds=train_seconds,
                       failure="no checkpoint produced")
            print("ERROR: no checkpoint in {} -- skipping evaluation.".format(ckpt_dir), flush=True)
            return
        if which == "last":
            print("WARNING: no 'best' checkpoint; reporting the LAST checkpoint.", flush=True)
        manager.params["training_params"]["load_epoch"] = which
        manager.load_model()
        print("Evaluating checkpoint: {} (epoch {})".format(which, manager.latest_epoch), flush=True)

        metrics = {"reported_checkpoint": which, "best_epoch": manager.latest_epoch,
                   "last_epoch": max([int(f.split("_")[-1].split(".")[0]) for f in available if "last" in f] or [None]),
                   "eval_batch_size": a.batch_size, "decoding": "greedy"}
        eval_splits = (["test", "valid"] if a.eval_test else ["valid"])
        metrics["test_evaluated"] = bool(a.eval_test)
        metrics["protocol_note"] = (
            "test evaluated at launch request (--eval-test)" if a.eval_test else
            "TEST NOT EVALUATED -- diagnostic/screening run; test is opened once, after the "
            "recipe is frozen, by a run launched with --eval-test")
        for split in eval_splits:
            split_dir = os.path.join(data_path, split)
            if os.path.isdir(split_dir) and os.listdir(split_dir):
                name = "{}-{}".format(a.dataset, split)
                manager.predict(name, [(a.dataset, split)], ["cer", "wer", "time"], output=True)
                pf = os.path.join(run_dir, "results", "predict_{}_{}.txt".format(name, manager.latest_epoch))
                if os.path.exists(pf):
                    metrics[split] = parse_predict_file(pf)
        rec.finish(metrics=metrics, status="complete", train_seconds=train_seconds,
                   artifacts={"checkpoints": sorted(os.listdir(ckpt_dir)),
                              "params": "results/params.txt", "run_command": "run_command.json"})
        print("[record] {} complete -> {}".format(rec.run_id, os.path.relpath(rec.path, ROOT)), flush=True)
    except KeyboardInterrupt:
        rec.finish(metrics={}, status="interrupted", train_seconds=time.time() - t0)
        raise
    except Exception:
        rec.finish(metrics={}, status="failed", train_seconds=time.time() - t0,
                   failure=traceback.format_exc()[-4000:])
        raise


if __name__ == "__main__":
    main()
