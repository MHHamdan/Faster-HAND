#!/usr/bin/env python3
"""E3 — fused draft-and-verify decoding, and its equivalence and speed measurement.

One forward pass per iteration does two jobs:

  VERIFY   the pass is given the accepted sequence plus the m-1 drafted tokens and asked for the
           last m distributions. Distribution j is the base model's own next-token distribution
           after the prefix ending at drafted token j, so comparing argmax(dist j) with draft j+1
           is exactly the check "would the base model have produced this token?". The longest
           matching prefix is accepted, and distribution ell -- which is a true base-model
           distribution -- supplies one more token for free.
  DRAFT    the hidden state that produced distribution ell is fed to the heads, giving the next
           m-1 drafts in the same pass.

Tokens emitted per pass = ell + 1 >= 1, so the loop cannot be slower than single-token greedy by
more than the head cost, and every emitted token is one the base model would have emitted.
The decoded string is therefore IDENTICAL to the base model's greedy decode; that identity is
asserted here page by page rather than assumed.

Why this works on this hardware: a forward over m tokens costs 0.054 ms more than over one,
against a fixed 3.83 ms per step (profiling/batch_scaling.json), so verification is nearly free.

  CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock python3 tools/spec_decode.py \
      --heads outputs/spec_heads_m5/heads.pt --m 2 3 4 5 --split test --n-pages 50 \
      --out experiments/benchmark_suite/profiling/spec_decode.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.amp import autocast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.evaluate_hand import (apply_checkpoint_architecture, build_params,  # noqa: E402
                                 stage_checkpoint, detect_additional_tokens, set_seed)
from tools.reproduce_baseline import MAX_CHARS  # noqa: E402
from tools.efficiency_bench import decode_page  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402
from hand.models.baseline.spec_heads import SpeculativeHeads  # noqa: E402
from hand.basic.metric_manager import MetricManager  # noqa: E402
from hand.OCR.ocr_utils import LM_ind_to_str  # noqa: E402


def spec_decode_page(manager, heads, m, pf, fsize, reduced, max_chars):
    """Fused draft-and-verify greedy decode of one page. Returns (string, stats)."""
    dev, ds = manager.device, manager.dataset
    dec = manager.models["decoder"]
    dec.reset_mem_kv_cache()
    sot, eot, pad = ds.tokens["start"], ds.tokens["end"], ds.tokens["pad"]
    nl = ds.charset.index("\n") if "\n" in ds.charset else -1
    max_lines = manager.params["training_params"].get("max_line_pred", 100)
    max_per_line = manager.params["training_params"].get("max_pred_per_line", 150)

    seq = [sot]
    emitted, conf, line_count, char_in_line = [], [], 0, 0
    draft, passes, accepted_hist, cache = [], 0, [], None
    done = False
    while not done and len(emitted) < max_chars:
        feed = seq + draft                                  # draft holds m-1 tokens (or none)
        n_new = 1 + len(draft)                              # distributions we want back
        toks = torch.tensor([feed], dtype=torch.long, device=dev)
        plen = torch.tensor([len(feed)], dtype=torch.int, device=dev)
        out, pred, cache, _ = dec(pf, pf, toks, reduced, plen, fsize, start=0, cache=cache,
                                  num_pred=n_new, padding_value=pad, use_mem_cache=True)
        passes += 1
        # pred: (1, V, n_new); distribution j predicts the token after feed[:len(seq)+j]
        probs = torch.softmax(pred[0].float(), dim=0)       # (V, n_new)
        argmax = pred[0].argmax(0).tolist()                 # length n_new
        top1 = probs.max(0).values.tolist()
        ell = 0
        while ell < len(draft) and argmax[ell] == draft[ell]:
            ell += 1
        new_tokens = draft[:ell] + [argmax[ell]]            # ell accepted drafts + 1 true token
        new_conf = top1[:ell + 1]
        accepted_hist.append(len(new_tokens))
        for t, c in zip(new_tokens, new_conf):
            if t == eot:
                done = True
                break
            emitted.append(t)
            conf.append(c)
            seq.append(t)
            if nl >= 0 and t == nl:
                line_count += 1
                char_in_line = 0
            else:
                char_in_line += 1
            if line_count >= max_lines or char_in_line >= max_per_line:
                done = True
                break
        # Cache rollback. The pass added one entry per query position, i.e. n_new entries per
        # layer, for the prefixes seq+draft[:0] .. seq+draft[:n_new-1]. Only the first ell+1 of
        # those prefixes were actually accepted, so the rest are states for a continuation that
        # did not happen and must be dropped or the next pass would attend to them. Without this
        # the decode is not equal to the base model's and the equivalence check fails.
        drop = n_new - (ell + 1)
        if drop > 0 and cache is not None:
            cache = cache[:, :cache.size(1) - drop]
        if done:
            break
        # the hidden state that produced distribution `ell` drafts the next window
        h = out[ell, 0, :].float()
        with torch.no_grad():
            draft = [int(l.argmax(-1)) for l in heads(h)][:m - 1]
    toks_t = torch.tensor(emitted, dtype=torch.long, device=dev)
    s = LM_ind_to_str(ds.charset, toks_t, oov_symbol="")
    return s, conf, {"passes": passes, "tokens": len(emitted),
               "accepted_mean": float(np.mean(accepted_hist)) if accepted_hist else 0.0,
               "accepted_hist": accepted_hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt")
    ap.add_argument("--heads", required=True)
    ap.add_argument("--m", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n-pages", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true",
                    help="run the decode in fp32. The method is exact in exact arithmetic; "
                         "under fp16 autocast a different query-block shape can round a near-tie "
                         "the other way, so this isolates arithmetic from logic.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    set_seed(0)
    spec = dict(dataset="READ_2016", level="page", variant="_sem_dan", ckpt=a.ckpt,
                hand_encoding=False)
    stage_checkpoint(a.ckpt, "spec_decode")
    params = build_params(spec, "spec_decode", 1, MAX_CHARS["page"], GlobalHTADecoder,
                          detect_additional_tokens(os.path.join(ROOT, a.ckpt)))
    apply_checkpoint_architecture(params, os.path.join(ROOT, a.ckpt), "spec_decode")
    man = Manager(params)
    man.load_model(reset_optimizer=True)
    for mod in man.models.values():
        mod.eval()
    st = torch.load(a.heads, map_location=man.device, weights_only=False)
    heads = SpeculativeHeads(256, st["vocab_out"], st["m"], st["hidden"]).to(man.device)
    heads.load_state_dict(st["heads"])
    heads.eval()
    print("heads m={} hidden={} params={} trained to epoch {}".format(
        st["m"], st["hidden"], heads.n_parameters(), st["epoch"]), flush=True)

    if a.split == "test":
        key = "READ_2016-test"
        man.dataset.generate_test_loader(key, [("READ_2016", "test")])
        loader = man.dataset.test_loaders[key]
    else:
        key = list(man.dataset.valid_loaders)[0]
        loader = man.dataset.valid_loaders[key]
    max_chars = params["training_params"]["max_char_prediction"]
    amp = params["training_params"]["use_amp"] and not a.no_amp
    metric_names = ["cer", "wer", "time", "loer", "map_cer"]

    arms = ["greedy"] + ["spec_m{}".format(m) for m in a.m]
    mm = {k: MetricManager(metric_names, key) for k in arms}
    rows = {k: [] for k in arms}
    preds = {k: [] for k in arms}

    with torch.no_grad():
        batches = []
        for i, b in enumerate(loader):
            if i >= a.n_pages:
                break
            batches.append(b)
        # warm-up on the first pages, scored by nobody: cuDNN plan selection and allocator
        # behaviour on the first call would otherwise land entirely in the first timed page.
        for b in batches[:a.warmup]:
            x = b["imgs"].to(man.device)
            with autocast("cuda", enabled=amp):
                f = man.models["encoder"](x)
                pf = man.models["decoder"].features_updater.get_pos_features(f)
                pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
                decode_page(man, pf, f.size(), [s_[:2] for s_ in b["imgs_reduced_shape"]],
                            max_chars, True)
                spec_decode_page(man, heads, max(a.m), pf, f.size(),
                                 [s_[:2] for s_ in b["imgs_reduced_shape"]], max_chars)
        for i, b in enumerate(batches):
            x = b["imgs"].to(man.device)
            reduced = [s[:2] for s in b["imgs_reduced_shape"]]
            gt = b["raw_labels"][0]
            with autocast("cuda", enabled=amp):
                f = man.models["encoder"](x)
                pf = man.models["decoder"].features_updater.get_pos_features(f)
                pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
                for arm in arms:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if arm == "greedy":
                        sx, cs, ntok = decode_page(man, pf, f.size(), reduced, max_chars, True)
                        s, cf = sx[0], cs[0]
                        stat = {"passes": ntok, "tokens": ntok, "accepted_mean": 1.0}
                    else:
                        s, cf, stat = spec_decode_page(man, heads, int(arm.split("m")[1]), pf,
                                                       f.size(), reduced, max_chars)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    preds[arm].append(s)
                    rows[arm].append({"page": i, "seconds": dt, **stat})
                    mm[arm].update_metrics(mm[arm].compute_metrics(
                        {"nb_samples": 1, "str_y": [gt], "str_x": [s],
                         "confidence_score": [cf], "time": dt, "names": b["names"]},
                        metric_names))
            if True:
                print("  page {}: ".format(i) + " | ".join(
                    "{} {:.2f}s {}p acc{:.2f}".format(k, rows[k][-1]["seconds"],
                                                      rows[k][-1]["passes"],
                                                      rows[k][-1]["accepted_mean"])
                    for k in arms), flush=True)

    res = {"ckpt": a.ckpt, "heads": a.heads, "head_parameters": heads.n_parameters(),
           "split": a.split, "n_pages": len(rows["greedy"]), "amp": bool(amp),
           "arms": {}}
    base = None
    for arm in arms:
        r = rows[arm]
        d = mm[arm].get_display_values(output=True)
        lat = float(np.mean([x["seconds"] for x in r]))
        if base is None:
            base = lat
        res["arms"][arm] = {
            "cer": float(d["cer"]), "wer": float(d["wer"]),
            "loer": float(d["loer"]) if "loer" in d else None,
            "map_cer": float(d["map_cer"]) if "map_cer" in d else None,
            "latency_s_per_page": lat, "speedup_vs_greedy": base / lat,
            "passes_per_page": float(np.mean([x["passes"] for x in r])),
            "tokens_per_page": float(np.mean([x["tokens"] for x in r])),
            "tokens_per_pass": float(np.sum([x["tokens"] for x in r])
                                     / max(np.sum([x["passes"] for x in r]), 1)),
            "identical_to_greedy": bool(all(u == v for u, v in zip(preds["greedy"], preds[arm]))),
            "identical_pages": int(sum(1 for u, v in zip(preds["greedy"], preds[arm]) if u == v)),
        }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({**res, "per_page": rows}, open(a.out, "w"), indent=1)
    print("\n%-10s %7s %7s %7s %8s %9s %8s %9s %9s %10s"
          % ("arm", "CER", "WER", "LOER", "mAP-CER", "s/page", "speedup", "passes",
             "tok/pass", "identical"))
    for k, v in res["arms"].items():
        print("%-10s %7.4f %7.4f %7s %8s %9.3f %8.2f %9.1f %9.3f %10s"
              % (k, v["cer"], v["wer"], v["loer"], v["map_cer"], v["latency_s_per_page"],
                 v["speedup_vs_greedy"], v["passes_per_page"], v["tokens_per_pass"],
                 v["identical_to_greedy"]))
    print("written:", a.out)


if __name__ == "__main__":
    main()
