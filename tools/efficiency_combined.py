#!/usr/bin/env python3
"""The combined efficiency experiment: baseline -> E2 -> E3 -> E2+E3, one process, one gate.

E2 (static-shape decoding with CUDA-graph capture) and E3 (lossless structured speculative
decoding) attack the same measured cost from opposite ends: E3 removes decoding steps, E2 makes
each remaining step cheaper. They are independent and should compose, but **composition is a
measurement, not an inference** -- E3's verify pass has a different query-block shape from the
single-token step, so the captured graph has to cover it.

Four arms, every one decoded on the same pages in the same process:

  baseline   the frozen reference decode
  e2         static shapes + torch.compile(mode="reduce-overhead")
  e3         speculative draft-and-verify, eager
  e2_e3      speculative draft-and-verify with the static, compiled step

**No speed-up is claimed for any arm unless its output is identical to the baseline's on every
page.** That check is the gate, and it is reported per arm alongside CER, WER, LOER and mAP-CER.

CPU smoke (validates the plumbing and the cross-arm identity without a GPU):
  python3 tools/efficiency_combined.py --device cpu --n-pages 1 --max-steps 24 --warmup 0 \
      --modes baseline,e3 --heads outputs/spec_heads_m5/heads.pt --out /tmp/x.json
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

from tools.evaluate_hand import build_params, stage_checkpoint, detect_additional_tokens, set_seed  # noqa: E402
from tools.reproduce_baseline import MAX_CHARS  # noqa: E402
from tools.graph_decode import pad_memory  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402
from hand.models.baseline.spec_heads import SpeculativeHeads  # noqa: E402
from hand.basic.metric_manager import MetricManager  # noqa: E402
from hand.OCR.ocr_utils import LM_ind_to_str  # noqa: E402

WIN = None          # set from the decoder


def greedy_loop(man, step, ds, max_chars, max_steps):
    """Single-token greedy decode driven by `step(tokens, start, cache, n_new)`."""
    dev = man.device
    sot, eot, pad = ds.tokens["start"], ds.tokens["end"], ds.tokens["pad"]
    nl = ds.charset.index("\n") if "\n" in ds.charset else -1
    max_lines = man.params["training_params"].get("max_line_pred", 100)
    max_per_line = man.params["training_params"].get("max_pred_per_line", 150)
    seq, emitted, conf, cache = [sot], [], [], None
    lines = chars = 0
    for _ in range(min(max_chars, max_steps)):
        _, pred, cache = step(seq, 1, cache)
        p = torch.softmax(pred[0, :, -1].float(), dim=0)
        t = int(p.argmax())
        if t == eot:
            break
        conf.append(float(p.max()))
        emitted.append(t)
        seq.append(t)
        if nl >= 0 and t == nl:
            lines, chars = lines + 1, 0
        else:
            chars += 1
        if lines >= max_lines or chars >= max_per_line:
            break
    emitted, conf = emitted[:max_steps], conf[:max_steps]
    return emitted, conf, {"passes": len(emitted), "tokens": len(emitted)}


def spec_loop(man, step, heads, m, ds, max_chars, max_steps):
    """Fused draft-and-verify decode driven by the same `step` contract."""
    dev = man.device
    sot, eot, pad = ds.tokens["start"], ds.tokens["end"], ds.tokens["pad"]
    nl = ds.charset.index("\n") if "\n" in ds.charset else -1
    max_lines = man.params["training_params"].get("max_line_pred", 100)
    max_per_line = man.params["training_params"].get("max_pred_per_line", 150)
    seq, emitted, conf, cache = [sot], [], [], None
    draft, passes, acc = [], 0, []
    lines = chars = 0
    done = False
    while not done and len(emitted) < min(max_chars, max_steps):
        feed = seq + draft
        n_new = 1 + len(draft)
        out, pred, cache = step(feed, n_new, cache)
        passes += 1
        probs = torch.softmax(pred[0].float(), dim=0)
        argmax = pred[0].argmax(0).tolist()
        top1 = probs.max(0).values.tolist()
        ell = 0
        while ell < len(draft) and argmax[ell] == draft[ell]:
            ell += 1
        acc.append(ell + 1)
        for t, c in zip(draft[:ell] + [argmax[ell]], top1[:ell + 1]):
            if t == eot:
                done = True
                break
            emitted.append(t)
            conf.append(c)
            seq.append(t)
            if nl >= 0 and t == nl:
                lines, chars = lines + 1, 0
            else:
                chars += 1
            if lines >= max_lines or chars >= max_per_line:
                done = True
                break
        drop = n_new - (ell + 1)
        if drop > 0 and cache is not None:
            cache = cache[:, :cache.size(1) - drop]
        if done:
            break
        with torch.no_grad():
            draft = [int(l.argmax(-1)) for l in heads(out[ell, 0, :].float())][:m - 1]
    # `max_steps` caps EMITTED TOKENS, not passes, so a capped speculative run and a capped
    # greedy run are comparable: a block may overshoot the cap and is trimmed back to it.
    emitted, conf = emitted[:max_steps], conf[:max_steps]
    return emitted, conf, {"passes": passes, "tokens": len(emitted),
                           "accepted_mean": float(np.mean(acc)) if acc else 0.0}


def make_step(man, dec, pf, fsize, reduced, pad_tok, static, fwd, enc_key=None, enc_mask=None):
    """`step(seq, n_new, cache) -> (hidden, pred, cache)`.

    static=False: the sequence is passed as it is, exactly as the reference decode does.
    static=True:  the token window is held at n_new + win - 1 once the sequence is long enough,
                  the cache is trimmed to the last win-1 entries so its shape is constant, and the
                  padded visual memory and its injected mask keep the memory dimension constant.
                  All three are exact; the first steps, before the window fills, use the plain path.
    """
    dev = man.device

    def step(seq, n_new, cache):
        if static:
            dec._enc_mask_cache = (enc_key, enc_mask)
        ntk = n_new + WIN - 1
        if static and len(seq) >= ntk:
            toks = torch.tensor([seq[-ntk:]], dtype=torch.long, device=dev)
            start = len(seq) - ntk
            if cache is not None and cache.size(1) > WIN - 1:
                cache = cache[:, -(WIN - 1):].contiguous()
            f = fwd
        else:
            toks = torch.tensor([seq], dtype=torch.long, device=dev)
            start = 0
            f = dec.forward
        plen = torch.tensor([toks.size(1)], dtype=torch.int, device=dev)
        if static and f is not dec.forward:
            torch.compiler.cudagraph_mark_step_begin()
        out, pred, cache, _ = f(pf, pf, toks, reduced, plen, fsize, start=start, cache=cache,
                                num_pred=n_new, padding_value=pad_tok, use_mem_cache=True)
        if static and f is not dec.forward:
            return out.clone(), pred.clone(), cache.clone()
        return out, pred, cache

    return step


def main():
    global WIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt")
    ap.add_argument("--heads", default="outputs/spec_heads_m5/heads.pt")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n-pages", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=10 ** 9,
                    help="cap on EMITTED TOKENS (not passes), so greedy and speculative arms stay "
                         "comparable under truncation. For smoke runs only.")
    ap.add_argument("--modes", default="baseline,e2,e3,e2_e3")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--s-pad", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    set_seed(0)
    spec = dict(dataset="READ_2016", level="page", variant="_sem_dan", ckpt=a.ckpt,
                hand_encoding=False)
    stage_checkpoint(a.ckpt, "eff_combined")
    params = build_params(spec, "eff_combined", 1, MAX_CHARS["page"], GlobalHTADecoder,
                          detect_additional_tokens(os.path.join(ROOT, a.ckpt)))
    if a.device == "cpu":
        params["training_params"]["force_cpu"] = True
        params["training_params"]["use_amp"] = False
    man = Manager(params)
    man.load_model()
    for mod in man.models.values():
        mod.eval()
    dec = man.models["decoder"]
    ds = man.dataset
    WIN = dec.dec_att_win
    pad_tok = ds.tokens["pad"]
    max_chars = params["training_params"]["max_char_prediction"]
    amp = params["training_params"]["use_amp"] and a.device == "cuda"
    st = torch.load(a.heads, map_location=man.device, weights_only=False)
    heads = SpeculativeHeads(256, st["vocab_out"], st["m"], st["hidden"]).to(man.device)
    heads.load_state_dict(st["heads"])
    heads.eval()

    if a.split == "test":
        key = "READ_2016-test"
        ds.generate_test_loader(key, [("READ_2016", "test")])
        loader = ds.test_loaders[key]
    else:
        key = list(ds.valid_loaders)[0]
        loader = ds.valid_loaders[key]

    batches = []
    for i, b in enumerate(loader):
        if i >= a.n_pages:
            break
        batches.append(b)

    feats, s_max = [], 0
    with torch.no_grad(), autocast("cuda", enabled=amp):
        for b in batches:
            f = man.models["encoder"](b["imgs"].to(man.device))
            pf = dec.features_updater.get_pos_features(f)
            pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
            feats.append((pf, f.size()))
            s_max = max(s_max, pf.size(0))
    s_pad = a.s_pad or int(np.ceil(s_max / 64) * 64)

    modes = a.modes.split(",")
    metric_names = ["cer", "wer", "time", "loer", "map_cer"]
    mm = {k: MetricManager(metric_names, key) for k in modes}
    rows, preds = {k: [] for k in modes}, {k: [] for k in modes}
    compiled = {}

    for mode in modes:
        static = mode in ("e2", "e2_e3")
        spec_arm = mode in ("e3", "e2_e3")
        # Warm-up pages are decoded and discarded, but they are NOT taken out of the scored set:
        # this split has exactly 50 pages and dropping two would silently change the CER the arms
        # are compared at (3.58 on 48 pages against 3.55 on 50).
        warm = list(range(min(a.warmup, len(batches))))
        order = warm + list(range(len(batches)))          # warm-up pages, then every page scored
        for i, bi in enumerate(order):
            b = batches[bi]
            pf, fsize = feats[bi]
            reduced = [s_[:2] for s_ in b["imgs_reduced_shape"]]
            gt = b["raw_labels"][0]
            with torch.no_grad(), autocast("cuda", enabled=amp):
                if static:
                    pf_use, mask_pad = pad_memory(pf, reduced, fsize, s_pad, man.device)
                    fsize_use = (1, int(fsize[1]), 1, s_pad)
                    enc_key = (tuple(fsize_use), tuple(tuple(r) for r in reduced))
                    if a.device == "cuda":
                        if "f" not in compiled:
                            compiled["f"] = torch.compile(dec.forward, mode="reduce-overhead",
                                                          dynamic=False)
                        fwd = compiled["f"]
                    else:
                        fwd = dec.forward
                else:
                    pf_use, mask_pad, fsize_use, enc_key, fwd = pf, None, fsize, None, dec.forward
                dec.reset_mem_kv_cache()
                step = make_step(man, dec, pf_use, fsize_use, reduced, pad_tok, static, fwd,
                                 enc_key, mask_pad)
                if a.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                if spec_arm:
                    em, cf, stat = spec_loop(man, step, heads, a.m, ds, max_chars, a.max_steps)
                else:
                    em, cf, stat = greedy_loop(man, step, ds, max_chars, a.max_steps)
                if a.device == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            s = LM_ind_to_str(ds.charset, torch.tensor(em, dtype=torch.long, device=man.device),
                              oov_symbol="")
            if i < len(warm):
                continue
            page = i - len(warm)
            preds[mode].append(s)
            rows[mode].append({"page": page, "seconds": dt, **stat})
            mm[mode].update_metrics(mm[mode].compute_metrics(
                {"nb_samples": 1, "str_y": [gt], "str_x": [s], "confidence_score": [cf],
                 "time": dt, "names": b["names"]}, metric_names))
        print("{:>8} done, {:.3f} s/page".format(
            mode, float(np.mean([r["seconds"] for r in rows[mode]]))), flush=True)

    res = {"ckpt": a.ckpt, "heads": a.heads, "m": a.m, "split": a.split, "s_pad": s_pad,
           "device": a.device, "amp": bool(amp), "n_pages": len(rows[modes[0]]),
           "head_parameters": heads.n_parameters(), "arms": {}}
    base = None
    for mode in modes:
        d = mm[mode].get_display_values(output=True)
        lat = float(np.mean([r["seconds"] for r in rows[mode]]))
        base = base or lat
        res["arms"][mode] = {
            "cer": float(d["cer"]), "wer": float(d["wer"]),
            "loer": float(d["loer"]) if "loer" in d else None,
            "map_cer": float(d["map_cer"]) if "map_cer" in d else None,
            "latency_s_per_page": lat, "speedup_vs_baseline": base / lat,
            "passes_per_page": float(np.mean([r["passes"] for r in rows[mode]])),
            "tokens_per_page": float(np.mean([r["tokens"] for r in rows[mode]])),
            "identical_pages": int(sum(1 for u, v in zip(preds[modes[0]], preds[mode]) if u == v)),
            "token_identical": bool(all(u == v for u, v in zip(preds[modes[0]], preds[mode])))}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({**res, "per_page": rows}, open(a.out, "w"), indent=1)
    print("\n%-8s %7s %7s %7s %8s %9s %8s %9s %10s"
          % ("arm", "CER", "WER", "LOER", "mAP-CER", "s/page", "speedup", "passes", "identical"))
    for k, v in res["arms"].items():
        print("%-8s %7.4f %7.4f %7s %8s %9.3f %8.2f %9.1f %10s"
              % (k, v["cer"], v["wer"], v["loer"], v["map_cer"], v["latency_s_per_page"],
                 v["speedup_vs_baseline"], v["passes_per_page"],
                 "{}/{}".format(v["identical_pages"], res["n_pages"])))
    print("written:", a.out)


if __name__ == "__main__":
    main()
