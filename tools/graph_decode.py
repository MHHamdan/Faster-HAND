#!/usr/bin/env python3
"""E2 — static-shape decoding, and whether removing launch overhead is worth what it costs.

The decode step was measured at ~98 % overhead: adding seven more pages to a batch costs 9.8 %
more time per step, so the marginal arithmetic per page is 0.054 ms against a 3.83 ms fixed cost
spread over ~160 small kernels (profiling/batch_scaling.json). The lever is therefore the
launches, and the standard way to remove them is to capture the step as a CUDA graph -- which
requires every per-step shape to be constant.

E1 removed the last growing input. What remains variable is the number of VISUAL positions,
which differs from page to page (8,085 / 8,470 / 8,525 ...). This tool makes the step fully
static:

  * the visual memory and its padding mask are padded to a fixed S_pad, with the padded
    positions masked -- masked keys contribute nothing, so the output is unchanged;
  * the token window is held at exactly `dec_att_win` by left-padding with the padding token,
    which the key mask already marks, with the positional-encoding `start` set to the window's
    true absolute position so the real tokens keep their encodings.

Both are exact. `--mode compile` then wraps the step in torch.compile(mode="reduce-overhead"),
which records CUDA graphs when shapes are static; `--mode eager` is the control.

Acceptance (campaign gate): token-identical output on every page AND >= 1.5x latency.

  CUDA_VISIBLE_DEVICES=0 python3 tools/graph_decode.py --n-pages 50 --split test \
      --out experiments/benchmark_suite/profiling/graph_decode.json
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
from tools.efficiency_bench import decode_page  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402
from hand.basic.metric_manager import MetricManager  # noqa: E402
from hand.OCR.ocr_utils import LM_ind_to_str  # noqa: E402


class FixedPE(torch.nn.Module):
    """Positional encoding as a constant-address buffer instead of a changing Python int.

    The decoder adds pe[:, :, start:start+L], and inside a compiled region a `start` that
    changes every step is a guard that forces a recompile per step. The slice is therefore
    copied into a fixed buffer before each call and added from there: same arithmetic, no int
    in the traced graph, and an address stable enough for CUDA-graph replay.
    """

    def __init__(self, pe_1d, win, device):
        super().__init__()
        self.src = pe_1d.pe
        self.buf = torch.zeros((1, pe_1d.dim, win), device=device, dtype=self.src.dtype)

    @torch.no_grad()
    def set_start(self, start):
        self.buf.copy_(self.src[:, :, start:start + self.buf.size(2)])

    def forward(self, x, start=0):
        return x + self.buf[:, :, :x.size(2)].to(x.dtype)


def pad_memory(pf, reduced, fsize, s_pad, device):
    """Pad the visual memory to a fixed length and build the matching padding mask.

    The decoder derives its encoder mask from (features_size, reduced_size); the padded mask is
    injected into its per-document memo so the derivation is bypassed. Padded positions are
    masked, so they contribute nothing to any attention output.
    """
    s = pf.size(0)
    assert s <= s_pad, "S={} exceeds S_pad={}".format(s, s_pad)
    out = torch.zeros((s_pad, pf.size(1), pf.size(2)), dtype=pf.dtype, device=device)
    out[:s] = pf
    h, w = int(fsize[2]), int(fsize[3])
    real = torch.ones((1, h, w), dtype=torch.bool, device=device)
    real[0, :int(reduced[0][0]), :int(reduced[0][1])] = False
    mask = torch.ones((1, s_pad), dtype=torch.bool, device=device)
    mask[:, :s] = real.flatten(1)
    return out, mask


def static_decode(manager, dec_step, dyn_step, max_chars, win, pad_tok, ds, fixed_pe=None):
    """Greedy decode whose step has a constant shape once the causal window is full.

    The positional encoding is absolute, so a left-padded window would need a negative `start`
    while the sequence is shorter than the window. Those first `win - 1` steps therefore run
    through the ordinary variable-shape path (`dyn_step`) and everything after -- 79 % of a
    477-token page -- runs through the constant-shape one.
    """
    dev = manager.device
    sot, eot = ds.tokens["start"], ds.tokens["end"]
    nl = ds.charset.index("\n") if "\n" in ds.charset else -1
    max_lines = manager.params["training_params"].get("max_line_pred", 100)
    max_per_line = manager.params["training_params"].get("max_pred_per_line", 150)
    seq = [sot]
    emitted, conf, cache = [], [], None
    line_count = char_in_line = 0
    for _ in range(max_chars):
        # the window is always `win` long: left-padded while the sequence is shorter
        if len(seq) >= win:
            toks = torch.tensor([seq[-win:]], dtype=torch.long, device=dev)
            if fixed_pe is not None:
                fixed_pe.set_start(len(seq) - win)
            pred, cache = dec_step(toks, 0, cache)
        else:
            toks = torch.tensor([seq], dtype=torch.long, device=dev)
            if fixed_pe is not None:
                fixed_pe.set_start(0)        # buf[:, :, :L] is then pe[:, :, 0:L], as required
            pred, cache = dyn_step(toks, 0, cache)
        p = torch.softmax(pred[0, :, -1].float(), dim=0)
        t = int(p.argmax())
        if t == eot:
            break
        conf.append(float(p.max()))
        emitted.append(t)
        seq.append(t)
        if nl >= 0 and t == nl:
            line_count += 1
            char_in_line = 0
        else:
            char_in_line += 1
        if line_count >= max_lines or char_in_line >= max_per_line:
            break
    return LM_ind_to_str(ds.charset, torch.tensor(emitted, dtype=torch.long, device=dev),
                         oov_symbol=""), conf, len(emitted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n-pages", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--modes", default="reference,static_eager,static_compile")
    ap.add_argument("--fixed-pe", action="store_true",
                    help="replace the positional encoding's changing int `start` with a fixed "
                         "buffer. Removes a recompile guard but mutates a compiled input, which "
                         "measured SLOWER (1.15x vs 2.53x): inductor falls back off cudagraphs.")
    ap.add_argument("--s-pad", type=int, default=0, help="0 = max over the split, rounded up to 64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    set_seed(0)
    spec = dict(dataset="READ_2016", level="page", variant="_sem_dan", ckpt=a.ckpt,
                hand_encoding=False)
    stage_checkpoint(a.ckpt, "graph_decode")
    params = build_params(spec, "graph_decode", 1, MAX_CHARS["page"], GlobalHTADecoder,
                          detect_additional_tokens(os.path.join(ROOT, a.ckpt)))
    man = Manager(params)
    man.load_model()
    for mod in man.models.values():
        mod.eval()
    dec = man.models["decoder"]
    ds = man.dataset
    win = dec.dec_att_win
    pad_tok = ds.tokens["pad"]
    max_chars = params["training_params"]["max_char_prediction"]
    amp = params["training_params"]["use_amp"]
    if a.split == "test":
        key = "READ_2016-test"
        ds.generate_test_loader(key, [("READ_2016", "test")])
        loader = ds.test_loaders[key]
    else:
        key = list(ds.valid_loaders)[0]
        loader = ds.valid_loaders[key]

    with torch.no_grad():
        batches = []
        for i, b in enumerate(loader):
            if i >= a.n_pages:
                break
            batches.append(b)
    s_max = 0
    feats = []
    with torch.no_grad(), autocast("cuda", enabled=amp):
        for b in batches:
            f = man.models["encoder"](b["imgs"].to(man.device))
            pf = dec.features_updater.get_pos_features(f)
            pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
            feats.append((pf, f.size()))
            s_max = max(s_max, pf.size(0))
    s_pad = a.s_pad or int(np.ceil(s_max / 64) * 64)
    print("visual positions: max {} -> S_pad {}".format(s_max, s_pad), flush=True)

    modes = a.modes.split(",")
    metric_names = ["cer", "wer", "time", "loer", "map_cer"]
    mm = {k: MetricManager(metric_names, key) for k in modes}
    rows = {k: [] for k in modes}
    preds = {k: [] for k in modes}
    compiled = {}

    for mode in modes:
        for i, b in enumerate(batches):
            pf, fsize = feats[i]
            reduced = [s_[:2] for s_ in b["imgs_reduced_shape"]]
            gt = b["raw_labels"][0]
            with torch.no_grad(), autocast("cuda", enabled=amp):
                if mode == "reference":
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    sx, cs, ntok = decode_page(man, pf, fsize, reduced, max_chars, True)
                    torch.cuda.synchronize(); dt = time.perf_counter() - t0
                    s, cf = sx[0], cs[0]
                else:
                    pf_pad, mask_pad = pad_memory(pf, reduced, fsize, s_pad, man.device)
                    fsize_pad = (1, int(fsize[1]), 1, s_pad)
                    key_cache = (tuple(fsize_pad), tuple(tuple(r) for r in reduced))
                    fwd = dec.forward
                    if mode == "static_compile":
                        if "f" not in compiled:
                            compiled["f"] = torch.compile(dec.forward, mode="reduce-overhead",
                                                          dynamic=False)
                        torch.compiler.cudagraph_mark_step_begin()
                        fwd = compiled["f"]

                    def make_step(_fwd):
                        def step(toks, start, cache, _pf=pf_pad, _m=mask_pad, _fs=fsize_pad,
                                 _k=key_cache):
                            dec._enc_mask_cache = (_k, _m)      # inject the padded mask
                            plen = torch.tensor([toks.size(1)], dtype=torch.int,
                                                device=man.device)
                            # With cudagraph trees the outputs live in the graph's private
                            # pool and are overwritten by the next replay. The cache is fed
                            # back in as an input on the following step, so it must leave the
                            # pool: one 0.8 MB clone per step, against a 3.83 ms step.
                            _, pred, cache, _ = _fwd(_pf, _pf, toks, reduced, plen, _fs,
                                                     start=start, cache=cache, num_pred=1,
                                                     padding_value=pad_tok, use_mem_cache=True)
                            return pred.clone(), cache.clone()
                        return step

                    dec.reset_mem_kv_cache()
                    dec._enc_mask_cache = (key_cache, mask_pad)
                    if a.fixed_pe and mode == "static_compile" and not isinstance(dec.pe_1d, FixedPE):
                        real_pe = dec.pe_1d
                        dec.pe_1d = FixedPE(real_pe, win, man.device)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    s, cf, ntok = static_decode(
                        man, make_step(fwd), make_step(dec.forward), max_chars, win, pad_tok, ds,
                        dec.pe_1d if isinstance(dec.pe_1d, FixedPE) else None)
                    torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if i < a.warmup:
                continue
            preds[mode].append(s)
            rows[mode].append({"page": i, "seconds": dt, "tokens": ntok})
            mm[mode].update_metrics(mm[mode].compute_metrics(
                {"nb_samples": 1, "str_y": [gt], "str_x": [s], "confidence_score": [cf],
                 "time": dt, "names": b["names"]}, metric_names))
        print("{:>16} done, {:.3f} s/page".format(
            mode, float(np.mean([r["seconds"] for r in rows[mode]]))), flush=True)

    res = {"ckpt": a.ckpt, "split": a.split, "s_pad": s_pad, "amp": bool(amp),
           "n_pages": len(rows[modes[0]]), "gpu": torch.cuda.get_device_name(0), "arms": {}}
    base = None
    for mode in modes:
        d = mm[mode].get_display_values(output=True)
        lat = float(np.mean([r["seconds"] for r in rows[mode]]))
        base = base or lat
        res["arms"][mode] = {
            "cer": float(d["cer"]), "wer": float(d["wer"]),
            "loer": float(d["loer"]) if "loer" in d else None,
            "map_cer": float(d["map_cer"]) if "map_cer" in d else None,
            "latency_s_per_page": lat, "speedup_vs_reference": base / lat,
            "tokens_per_page": float(np.mean([r["tokens"] for r in rows[mode]])),
            "identical_pages": int(sum(1 for u, v in zip(preds[modes[0]], preds[mode]) if u == v)),
            "token_identical": bool(all(u == v for u, v in zip(preds[modes[0]], preds[mode])))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({**res, "per_page": rows}, open(a.out, "w"), indent=1)
    print("\n%-16s %7s %7s %7s %8s %9s %8s %10s" % ("mode", "CER", "WER", "LOER", "mAP-CER",
                                                    "s/page", "speedup", "identical"))
    for k, v in res["arms"].items():
        print("%-16s %7.4f %7.4f %7s %8s %9.3f %8.2f %10s"
              % (k, v["cer"], v["wer"], v["loer"], v["map_cer"], v["latency_s_per_page"],
                 v["speedup_vs_reference"], "{}/{}".format(v["identical_pages"], res["n_pages"])))
    print("written:", a.out)


if __name__ == "__main__":
    main()
