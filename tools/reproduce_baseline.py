#!/usr/bin/env python3
"""
Phase 2, Task 2 — reproduce every released checkpoint and record full evidence.

Re-evaluates the frozen baseline (docs/model_card.md) and writes, per checkpoint:

    results_recovery/<checkpoint>/metrics.json      CER, WER, LOER, mAP_CER per split
    results_recovery/<checkpoint>/predictions.json  per-sample prediction and ground truth
    results_recovery/<checkpoint>/runtime.json      latency, peak GPU memory, environment

It reuses `tools/evaluate_hand.py`'s Manager/predict path unchanged, so decoding and
metric computation are identical to training. What it adds is:

  * **layout metrics** — LOER and mAP_CER, which the existing evaluation never requested.
    Only the READ `_sem` datasets carry layout tokens; IAM, KHATT and AHAWP were formatted
    with `constraints: ["add_eot", "add_sot"]` and have none, so requesting layout metrics
    there would score an empty graph against an empty graph. Those runs record
    `layout_metrics: "not_applicable"` rather than a misleading 0.
  * **latency and peak memory**, measured per split with `torch.cuda.max_memory_allocated`
    reset immediately before decoding, and reported per document alongside batch size and
    device — the manuscript quotes a latency with neither.
  * **per-sample predictions**, so a disputed CER can be traced to the document that
    produced it.

Usage:
    python tools/reproduce_baseline.py --model all --split test valid
    python tools/reproduce_baseline.py --model read_page --split test --batch-size 1
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.evaluate_hand import (  # noqa: E402
    MODELS, build_params, stage_checkpoint, detect_additional_tokens,
    set_seed, _jsonable,
)
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402

# Layout metrics need layout tokens in the ground truth. Only the READ `_sem` datasets
# have them (their formatter constraints omit 'hand_encoding').
LAYOUT_CAPABLE = {"read_page", "read_double_page", "read_triple_page"}

MAX_CHARS = {
    "line": 200, "paragraph": 1000, "page": 3000,
    "double_page": 6000, "triple_page": 9000,
}


def environment():
    """Everything needed to explain a number that fails to reproduce elsewhere."""
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu_total_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        env["git_commit"] = None  # repository has no commits yet
    return env


def checkpoint_fingerprint(path):
    """Identify the weights independently of the file path they were loaded from."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {"epoch": ck.get("epoch"), "step": ck.get("step"),
           "recorded_best_valid_cer": ck.get("best"),
           "charset_size": len(ck.get("charset", []))}
    total = 0
    for key in ("encoder_state_dict", "decoder_state_dict"):
        sd = ck.get(key, {})
        n = sum(t.numel() for t in sd.values() if torch.is_tensor(t))
        out[key.replace("_state_dict", "_parameters")] = n
        total += n
    out["total_parameters"] = total
    dec = ck.get("decoder_state_dict", {})
    out["decoder_layers"] = len({k.split(".")[2] for k in dec
                                 if k.startswith("att_decoder.decoder_layers.")})
    # The audit's central check, re-run on every reproduction so it cannot silently drift.
    enc = ck.get("encoder_state_dict", {})
    out["encoder_has_octave_gate_se_keys"] = any(
        t in k.lower() for k in enc for t in ("octave", "h2h", "l2l", "gated", "excit"))
    out["decoder_has_memory_sparse_fusion_keys"] = any(
        t in k.lower() for k in dec for t in ("memory", "sparse", "fusion"))
    del ck
    return out


def run_one(model_key, splits, batch_size, out_root):
    spec = MODELS[model_key]
    ckpt_path = os.path.join(ROOT, spec["ckpt"])
    if not os.path.exists(ckpt_path):
        print("[skip] {}: checkpoint missing at {}".format(model_key, spec["ckpt"]))
        return None

    out_dir = os.path.join(out_root, model_key)
    os.makedirs(out_dir, exist_ok=True)

    run_dir = "recover_{}".format(model_key)
    stage_checkpoint(spec["ckpt"], run_dir)
    set_seed(0)

    additional_tokens = detect_additional_tokens(ckpt_path)
    params = build_params(spec, run_dir, batch_size, MAX_CHARS[spec["level"]],
                          GlobalHTADecoder, additional_tokens)

    layout = model_key in LAYOUT_CAPABLE
    metric_names = ["cer", "wer", "time"] + (["loer", "map_cer"] if layout else [])
    params["training_params"]["eval_metrics"] = metric_names

    manager = Manager(params)
    manager.load_model()

    metrics = {
        "model": model_key,
        "dataset": spec["dataset"],
        "level": spec["level"],
        "checkpoint": spec["ckpt"],
        "checkpoint_fingerprint": checkpoint_fingerprint(ckpt_path),
        "loaded_epoch": manager.latest_epoch,
        "charset_size": len(manager.dataset.charset),
        "hand_encoding": spec.get("hand_encoding", True),
        "additional_tokens": additional_tokens,
        "layout_metrics": "measured" if layout else "not_applicable",
        "layout_metrics_note": (
            None if layout else
            "dataset formatted without layout tokens (constraints lack 'hand_encoding'), "
            "so LOER and mAP_CER are undefined for it"),
        "decoding": "greedy",
        "batch_size": batch_size,
        "splits": {},
    }
    runtime = {"model": model_key, "batch_size": batch_size,
               "environment": environment(), "splits": {}}
    predictions = {"model": model_key, "splits": {}}

    for split in splits:
        split_dir = os.path.join(
            ROOT, "formatted",
            "{}_{}{}".format(spec["dataset"], spec["level"], spec["variant"]), split)
        if not os.path.isdir(split_dir) or not os.listdir(split_dir):
            metrics["splits"][split] = {"status": "missing_or_empty"}
            print("[skip] {} {}: split missing or empty".format(model_key, split))
            continue

        custom_name = "{}-{}".format(spec["dataset"], split)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        manager.predict(custom_name, [(spec["dataset"], split)], metric_names, output=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        mm = manager.metric_manager[custom_name]
        values = {k: _jsonable(v) for k, v in mm.get_display_values(output=True).items()}
        n = values.get("nb_samples") or 1

        metrics["splits"][split] = {"status": "ok", "metrics": values}
        runtime["splits"][split] = {
            "n_samples": n,
            "wall_clock_s": round(elapsed, 3),
            "seconds_per_document": round(elapsed / n, 4),
            "peak_gpu_memory_gb": (
                round(torch.cuda.max_memory_allocated() / 1024 ** 3, 4)
                if torch.cuda.is_available() else None),
        }
        # Per-sample predictions, so any reported number is traceable to a document.
        try:
            predictions["splits"][split] = [
                {"id": i, "prediction": x, "ground_truth": y}
                for i, x, y in zip(mm.epoch_metrics.get("ids", []),
                                   mm.epoch_metrics.get("str_x", []),
                                   mm.epoch_metrics.get("str_y", []))
            ]
        except Exception as exc:  # never lose the metrics because of the dump
            predictions["splits"][split] = {"error": str(exc)}

        print("[{}] {}: {}".format(model_key, split, values), flush=True)
        print("    {:.3f}s total, {:.3f}s/doc, {} GB peak".format(
            elapsed, elapsed / n, runtime["splits"][split]["peak_gpu_memory_gb"]), flush=True)

    for name, payload in (("metrics", metrics), ("runtime", runtime),
                          ("predictions", predictions)):
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    print("wrote", out_dir, flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["all"],
                    help="checkpoint key(s), or 'all'")
    ap.add_argument("--split", nargs="+", default=["test", "valid"])
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 is the reporting default; CER is not invariant to this")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_recovery"))
    a = ap.parse_args()

    keys = sorted(MODELS) if a.model == ["all"] else a.model
    for k in keys:
        if k not in MODELS:
            raise SystemExit("unknown model {!r}; choose from {}".format(k, sorted(MODELS)))

    summary = {}
    for k in keys:
        try:
            rec = run_one(k, a.split, a.batch_size, a.out)
            if rec:
                summary[k] = rec
        except Exception as exc:
            import traceback
            traceback.print_exc()
            summary[k] = {"status": "failed", "error": repr(exc)}
            print("[FAIL] {}: {!r}".format(k, exc), flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nwrote", os.path.join(a.out, "_summary.json"))


if __name__ == "__main__":
    main()
