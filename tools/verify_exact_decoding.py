#!/usr/bin/env python3
"""Equivalence gate for the exact decoding optimisations.

Three changes to the decode path claim to change cost and nothing else:

  HAND_FAST_MASKS   build the (1, <=win) causal-mask slice directly instead of the (T x T) square,
                    memoise the constant encoder-padding mask, slice the token mask
  HAND_FAST_STEP    embed and positionally-encode only the kept token window, with the positional
                    encoding's `start` advanced by the number of dropped tokens
  use_mem_cache     project the visual memory's K/V once per document instead of once per step

"Exact" is a claim, so it is tested rather than asserted: every page is decoded under the
reference path and under the optimised path in ONE process, and the decoded strings must be
identical character for character. Any difference fails the gate and the optimisation is a
defect, not an optimisation.

  CUDA_VISIBLE_DEVICES=0 python3 tools/verify_exact_decoding.py --n-pages 50 --split test \
      --out experiments/benchmark_suite/profiling/exact_decoding_equivalence.json
"""
import argparse
import json
import os
import sys

import torch
from torch.amp import autocast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.evaluate_hand import build_params, stage_checkpoint, detect_additional_tokens, set_seed  # noqa: E402
from tools.reproduce_baseline import MAX_CHARS  # noqa: E402
from tools.efficiency_bench import decode_page, MODELS  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline import dan_decoder  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n-pages", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = {"split": a.split, "n_pages": a.n_pages, "models": {}}
    for label in a.models.split(","):
        ckpt = MODELS[label]
        set_seed(0)
        spec = dict(dataset="READ_2016", level="page", variant="_sem_dan", ckpt=ckpt,
                    hand_encoding=False)
        stage_checkpoint(ckpt, "verify_{}".format(label))
        params = build_params(spec, "verify_{}".format(label), 1, MAX_CHARS["page"],
                              GlobalHTADecoder, detect_additional_tokens(os.path.join(ROOT, ckpt)))
        m = Manager(params)
        m.load_model()
        for mod in m.models.values():
            mod.eval()
        if a.split == "test":
            key = "READ_2016-test"
            m.dataset.generate_test_loader(key, [("READ_2016", "test")])
            loader = m.dataset.test_loaders[key]
        else:
            key = list(m.dataset.valid_loaders)[0]
            loader = m.dataset.valid_loaders[key]
        max_chars = params["training_params"]["max_char_prediction"]

        # (fast_masks, fast_step, use_mem_cache)
        settings = {"reference": (False, False, False), "optimised": (True, True, True),
                    "masks_only": (True, False, False), "step_only": (False, True, False),
                    "cache_only": (False, False, True)}
        out = {k: [] for k in settings}
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= a.n_pages:
                    break
                x = b["imgs"].to(m.device)
                reduced = [s[:2] for s in b["imgs_reduced_shape"]]
                with autocast("cuda", enabled=params["training_params"]["use_amp"]):
                    f = m.models["encoder"](x)
                    pf = m.models["decoder"].features_updater.get_pos_features(f)
                    pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
                    for name, (fm, fs, kv) in settings.items():
                        dan_decoder.FAST_MASKS, dan_decoder.FAST_STEP = fm, fs
                        sx, _, _ = decode_page(m, pf, f.size(), reduced, max_chars, kv)
                        out[name].append(sx[0])
        dan_decoder.FAST_MASKS = dan_decoder.FAST_STEP = True
        ref = out["reference"]
        res["models"][label] = {
            k: {"identical_pages": sum(1 for u, v in zip(ref, out[k]) if u == v),
                "pages": len(ref),
                "token_identical": all(u == v for u, v in zip(ref, out[k]))}
            for k in settings if k != "reference"}
        print(label, json.dumps(res["models"][label]), flush=True)
        del m
        torch.cuda.empty_cache()

    res["all_exact"] = all(v["token_identical"] for mm in res["models"].values()
                           for v in mm.values())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("ALL EXACT:", res["all_exact"])
    print("written:", a.out)
    sys.exit(0 if res["all_exact"] else 1)


if __name__ == "__main__":
    main()
