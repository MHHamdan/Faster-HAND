#!/usr/bin/env python3
"""Efficiency baseline: HAND and the DAN reference under ONE evaluation harness.

Everything that is not the model is held identical: the same 50 READ 2016 pages, the same
preprocessing, the same greedy decoding rule, the same stopping rule, the same batch size,
the same precision, the same GPU, the same measurement code. Both checkpoints are executed
through THIS repository's modules, so the only difference between the rows is the weights
(and, for the `kvcache` mode, the documented inference-time change below).

Reported per row:
  parameters (encoder / decoder / total) - peak memory - encoder FLOPs - decoder FLOPs -
  total FLOPs - wall-clock latency - throughput - batch size - hardware - precision -
  number of generated tokens - CER / WER on the decoded output.

Modes:
  baseline   the frozen decoding path (BASELINE_FREEZE.md), which re-projects the visual
             keys and values at every decoding step
  kvcache    identical numerics, but the memory K/V projection is computed once per page
             (attention.py, `use_mem_cache`). Equivalence is CHECKED, not assumed: the
             decoded string must be identical to the baseline's on every page.

FLOPs are counted with torch.utils.flop_counter.FlopCounterMode over the REAL decode of a
page (every step actually executed), not over an analytic estimate, and are measured in a
separate pass so that the counter's overhead never enters a timing measurement.

  CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock python3 tools/efficiency_bench.py \
      --n-pages 50 --split test --out experiments/benchmark_suite/profiling/efficiency_hand_vs_dan.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.evaluate_hand import (apply_checkpoint_architecture, build_params,  # noqa: E402
                                 stage_checkpoint, detect_additional_tokens, set_seed)
from tools.reproduce_baseline import MAX_CHARS  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402
from hand.basic.metric_manager import MetricManager  # noqa: E402
from hand.OCR.ocr_utils import LM_ind_to_str  # noqa: E402

MODELS = {
    "HAND_e14_1p26M": "outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt",
    "DAN_published": "weights/dan/dan_read_page.pt",
}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_page(manager, pos_features, features_size, reduced_size, max_chars, use_mem_cache,
                depth=None):
    """Greedy free-running decode of one page. Mirrors trainer_std_hand.evaluate_batch_greedy."""
    dev, ds = manager.device, manager.dataset
    dec = manager.models["decoder"]
    if use_mem_cache:
        dec.reset_mem_kv_cache()
    b = pos_features.size(1)
    max_lines = manager.params["training_params"].get("max_line_pred", 100)
    max_per_line = manager.params["training_params"].get("max_pred_per_line", 150)
    reached_end = torch.zeros((b,), dtype=torch.bool, device=dev)
    toks = torch.ones((b, 1), dtype=torch.long, device=dev) * ds.tokens["start"]
    plen = torch.ones((b,), dtype=torch.int, device=dev)
    nl = ds.charset.index("\n") if "\n" in ds.charset else -1
    line_count = torch.zeros((b,), dtype=torch.int, device=dev)
    char_in_line = torch.zeros((b,), dtype=torch.int, device=dev)
    conf, cache = [], None
    for i in range(1, max_chars + 1):
        _, pred, cache, _ = dec(pos_features, pos_features, toks, reduced_size, plen,
                                features_size, start=0, cache=cache, num_pred=1,
                                padding_value=ds.tokens["pad"], use_mem_cache=use_mem_cache,
                                depth=depth)
        conf.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
        new = torch.argmax(pred[:, :, -1], dim=1, keepdim=True)
        toks = torch.cat([toks, new], dim=1)
        if nl >= 0:
            is_nl = torch.eq(new.squeeze(-1), nl)
            line_count = line_count + is_nl.int()
            char_in_line = torch.where(is_nl, torch.zeros_like(char_in_line), char_in_line + 1)
            reached_end = torch.logical_or(
                reached_end, torch.logical_or(line_count >= max_lines, char_in_line >= max_per_line))
        reached_end = torch.logical_or(reached_end, torch.eq(toks[:, -1], ds.tokens["end"]))
        plen[torch.eq(reached_end, False)] = i + 1
        if torch.all(reached_end):
            break
    plen[torch.eq(reached_end, False)] = max_chars
    toks = toks[:, 1:]
    conf = torch.cat(conf, dim=1).cpu().detach().numpy()
    pt = [toks[k, :plen[k]] for k in range(b)]
    cs = [conf[k, :plen[k]].tolist() for k in range(b)]
    rm = [ds.tokens["end"]] + ([ds.tokens["blank"]] if "blank" in ds.tokens else [])
    pt, cs = manager.remove_ind_from_pred_list(pt, rm, cs)
    return [LM_ind_to_str(ds.charset, t, oov_symbol="") for t in pt], cs, int(plen[0])


def build(manager_label, ckpt, variant, level, dataset, split, device):
    set_seed(0)
    spec = dict(dataset=dataset, level=level, variant=variant, ckpt=ckpt, hand_encoding=False)
    run_dir = "eff_{}".format(manager_label)
    stage_checkpoint(ckpt, run_dir)
    add_tok = detect_additional_tokens(os.path.join(ROOT, ckpt))
    params = build_params(spec, run_dir, 1, MAX_CHARS[level], GlobalHTADecoder, add_tok)
    apply_checkpoint_architecture(params, os.path.join(ROOT, ckpt), manager_label)
    if device == "cpu":
        params["training_params"]["force_cpu"] = True
        params["training_params"]["use_amp"] = False
    m = Manager(params)
    # Evaluation only: nothing is stepped, and a shared-K/V checkpoint has fewer optimiser
    # parameter groups than an untied model, which makes load_optimizers raise.
    m.load_model(reset_optimizer=True)
    for mod in m.models.values():
        mod.eval()
    return m, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="READ_2016")
    ap.add_argument("--level", default="page")
    ap.add_argument("--variant", default="_sem_dan")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n-pages", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--flops-pages", type=int, default=2)
    ap.add_argument("--modes", default="baseline,kvcache")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    modes = a.modes.split(",")
    env = {
        "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
        "gpu_total_mem_MiB": (torch.cuda.get_device_properties(0).total_memory // 2**20
                              if a.device == "cuda" else None),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
        "host": platform.node(), "batch_size": 1,
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True).stdout.strip().splitlines()[0]
        if a.device == "cuda" else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results, decoded = {}, {}

    selected = {}
    for entry in a.models.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            k, path = (x.strip() for x in entry.split("=", 1))
        elif entry in MODELS:
            k, path = entry, MODELS[entry]
        else:
            raise SystemExit("unknown model {!r}: pass 'label=path/to/ckpt.pt', or one of: {}"
                             .format(entry, ", ".join(MODELS)))
        if not os.path.exists(os.path.join(ROOT, path)):
            raise SystemExit("checkpoint for {!r} does not exist: {}".format(k, path))
        selected[k] = path

    for label, ckpt in selected.items():
        manager, params = build(label, ckpt, a.variant, a.level, a.dataset, a.split, a.device)
        amp = params["training_params"]["use_amp"] and a.device == "cuda"
        if a.split == "test":
            ds_key = "{}-test".format(a.dataset)
            manager.dataset.generate_test_loader(ds_key, [(a.dataset, "test")])
        else:
            ds_key = list(manager.dataset.valid_loaders)[0]
        loader = (manager.dataset.test_loaders[ds_key] if a.split == "test"
                  else manager.dataset.valid_loaders[ds_key])
        max_chars = params["training_params"]["max_char_prediction"]
        n_enc = sum(p.numel() for p in manager.models["encoder"].parameters())
        n_dec = sum(p.numel() for p in manager.models["decoder"].parameters())

        batches = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= a.n_pages:
                    break
                batches.append(batch)

        for mode in modes:
            depth = int(mode.split("@")[1]) if "@" in mode else None
            kv = mode.split("@")[0] == "kvcache"
            mm = MetricManager(["cer", "wer", "time", "loer", "map_cer"], ds_key)
            rows, preds = [], []
            torch.cuda.reset_peak_memory_stats() if a.device == "cuda" else None
            with torch.no_grad():
                # warm-up (excluded from every statistic)
                for batch in batches[:a.warmup]:
                    x = batch["imgs"].to(manager.device)
                    with autocast("cuda", enabled=amp):
                        f = manager.models["encoder"](x)
                        pf = manager.models["decoder"].features_updater.get_pos_features(f)
                        pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
                        decode_page(manager, pf, f.size(), [s[:2] for s in batch["imgs_reduced_shape"]],
                                    max_chars, kv, depth)
                if a.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                for i, batch in enumerate(batches):
                    x = batch["imgs"].to(manager.device)
                    reduced = [s[:2] for s in batch["imgs_reduced_shape"]]
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with autocast("cuda", enabled=amp):
                        f = manager.models["encoder"](x)
                        pf = manager.models["decoder"].features_updater.get_pos_features(f)
                        pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
                        if a.device == "cuda":
                            torch.cuda.synchronize()
                        t_enc = time.perf_counter()
                        str_x, cs, ntok = decode_page(manager, pf, f.size(), reduced, max_chars,
                                                     kv, depth)
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    mm.update_metrics(mm.compute_metrics(
                        {"nb_samples": 1, "str_y": [batch["raw_labels"][0]], "str_x": [str_x[0]],
                         "confidence_score": cs, "time": t1 - t0, "names": batch["names"]},
                        ["cer", "wer", "time", "loer", "map_cer"]))
                    preds.append(str_x[0])
                    rows.append({"page": i, "name": batch["names"][0],
                                 "img_h": int(x.size(2)), "img_w": int(x.size(3)),
                                 "visual_positions": int(f.size(2) * f.size(3)),
                                 "encoder_s": t_enc - t0, "decode_s": t1 - t_enc,
                                 "total_s": t1 - t0, "tokens": ntok})
            peak = (torch.cuda.max_memory_allocated() / 2**20) if a.device == "cuda" else None
            reserved = (torch.cuda.max_memory_reserved() / 2**20) if a.device == "cuda" else None

            # FLOPs on the real decode, separate pass so the counter never times anything
            flops = []
            with torch.no_grad():
                for batch in batches[:a.flops_pages]:
                    x = batch["imgs"].to(manager.device)
                    reduced = [s[:2] for s in batch["imgs_reduced_shape"]]
                    with autocast("cuda", enabled=amp):
                        fc = FlopCounterMode(display=False)
                        with fc:
                            f = manager.models["encoder"](x)
                        enc_fl = fc.get_total_flops()
                        pf = manager.models["decoder"].features_updater.get_pos_features(f)
                        pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
                        fc2 = FlopCounterMode(display=False)
                        with fc2:
                            _, _, ntok = decode_page(manager, pf, f.size(), reduced, max_chars,
                                                    kv, depth)
                        dec_fl = fc2.get_total_flops()
                    flops.append({"name": batch["names"][0], "tokens": ntok,
                                  "encoder_GFLOPs": enc_fl / 1e9, "decoder_GFLOPs": dec_fl / 1e9,
                                  "total_GFLOPs": (enc_fl + dec_fl) / 1e9,
                                  "decoder_GFLOPs_per_token": dec_fl / 1e9 / max(ntok, 1)})

            key = "{}::{}".format(label, mode)
            decoded[key] = preds
            results[key] = {
                "model": label, "mode": mode, "inference_depth": depth or 8,
                "ckpt": ckpt, "ckpt_sha256": sha256(ckpt),
                "dataset": "{}_{}{}".format(a.dataset, a.level, a.variant), "split": a.split,
                "n_pages": len(rows), "batch_size": 1,
                "precision": "AMP fp16 (autocast)" if amp else "fp32",
                "params_encoder": n_enc, "params_decoder": n_dec, "params_total": n_enc + n_dec,
                "peak_memory_MiB_allocated": peak, "peak_memory_MiB_reserved": reserved,
                "latency_s_per_page_mean": float(np.mean([r["total_s"] for r in rows])),
                "latency_s_per_page_std": float(np.std([r["total_s"] for r in rows])),
                "latency_s_per_page_median": float(np.median([r["total_s"] for r in rows])),
                "encoder_s_mean": float(np.mean([r["encoder_s"] for r in rows])),
                "decode_s_mean": float(np.mean([r["decode_s"] for r in rows])),
                "throughput_pages_per_s": float(1.0 / np.mean([r["total_s"] for r in rows])),
                "tokens_generated_mean": float(np.mean([r["tokens"] for r in rows])),
                "tokens_generated_total": int(sum(r["tokens"] for r in rows)),
                "ms_per_generated_token": float(1000 * np.sum([r["decode_s"] for r in rows]) /
                                                max(sum(r["tokens"] for r in rows), 1)),
                "visual_positions_mean": float(np.mean([r["visual_positions"] for r in rows])),
                "flops": flops,
                "metrics": {k: (v.item() if hasattr(v, "item") else v)
                            for k, v in mm.get_display_values(output=True).items()},
                "per_page": rows,
            }
            print("{:>28} | {:.3f} s/page | {:.1f} tok | {} GFLOPs dec | CER {}".format(
                key, results[key]["latency_s_per_page_mean"], results[key]["tokens_generated_mean"],
                round(flops[0]["decoder_GFLOPs"], 1) if flops else "-",
                results[key]["metrics"].get("cer")), flush=True)
        del manager
        torch.cuda.empty_cache() if a.device == "cuda" else None

    # exact-equivalence check: kvcache must reproduce the baseline token stream
    equiv = {}
    for label in a.models.split(","):
        b, k = "{}::baseline".format(label), "{}::kvcache".format(label)
        if b in decoded and k in decoded:
            same = [x == y for x, y in zip(decoded[b], decoded[k])]
            equiv[label] = {"pages": len(same), "identical_pages": int(sum(same)),
                            "token_identical": bool(all(same))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"environment": env, "equivalence": equiv, "rows": results},
              open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(equiv, indent=1))
    print("written:", a.out)


if __name__ == "__main__":
    main()
