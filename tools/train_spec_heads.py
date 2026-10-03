#!/usr/bin/env python3
"""E3 — train speculative draft heads on a FROZEN baseline.

The base model's weights never move: `torch.no_grad()` around the encoder and decoder, both in
eval mode, so the hidden states the heads see at training time are the ones they will see at
decoding time. Only the heads carry gradients. Consequently the base model's greedy output --
and therefore its CER, WER, LOER and mAP-CER -- cannot change; the heads can only make the
decode faster or fail to.

Head k is trained with cross-entropy against the target shifted by 1 + k, on the real training
pages under evaluation-time preprocessing (no augmentation, no synthetic pages), teacher-forced.

  CUDA_VISIBLE_DEVICES=1 flock /tmp/hand-gpu1.lock python3 tools/train_spec_heads.py \
      --m 3 --epochs 120 --out outputs/spec_heads_m3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.evaluate_hand import (apply_checkpoint_architecture, build_params,  # noqa: E402
                                 stage_checkpoint, detect_additional_tokens, set_seed)
from tools.reproduce_baseline import MAX_CHARS  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402
from hand.models.baseline.spec_heads import SpeculativeHeads  # noqa: E402


def build(ckpt, batch_size, device="cuda", level="page", variant="_sem_dan"):
    # `level`/`variant` added 2026-10 (multi-page study) so heads can be trained on the data an
    # adapted base was fine-tuned on; the defaults reproduce every existing head run exactly.
    set_seed(0)
    spec = dict(dataset="READ_2016", level=level, variant=variant, ckpt=ckpt,
                hand_encoding=False)
    stage_checkpoint(ckpt, "spec_base")
    params = build_params(spec, "spec_base", batch_size, MAX_CHARS[level], GlobalHTADecoder,
                          detect_additional_tokens(os.path.join(ROOT, ckpt)))
    apply_checkpoint_architecture(params, os.path.join(ROOT, ckpt), "spec_base")
    m = Manager(params)
    # the base is frozen for head training, so optimiser state is irrelevant; a shared-K/V
    # checkpoint also has fewer optimiser groups than an untied model and would raise.
    m.load_model(reset_optimizer=True)
    for mod in m.models.values():
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    return m, params


def hidden_states(manager, batch, amp):
    """Teacher-forced pass through the frozen model. Returns (T, B, C) hidden states."""
    x = batch["imgs"].to(manager.device)
    y = batch["labels"].to(manager.device)
    reduced = [s[:2] for s in batch["imgs_reduced_shape"]]
    with torch.no_grad(), autocast("cuda", enabled=amp):
        f = manager.models["encoder"](x)
        dec = manager.models["decoder"]
        pf = dec.features_updater.get_pos_features(f)
        pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
        out, _, _, _ = dec(pf, pf, y[:, :-1], reduced, batch["labels_len"], f.size(),
                           start=0, padding_value=manager.dataset.tokens["pad"])
    return out.float(), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256, help="0 = linear heads")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="page", help="dataset level for head training (default page)")
    ap.add_argument("--variant", default="_sem_dan")
    a = ap.parse_args()

    manager, params = build(a.ckpt, a.batch_size, level=a.level, variant=a.variant)
    amp = params["training_params"]["use_amp"]
    pad = manager.dataset.tokens["pad"]
    vocab_out = manager.models["decoder"].end_conv.weight.shape[0]
    heads = SpeculativeHeads(256, vocab_out, a.m, a.hidden).to(manager.device)
    opt = AdamW(heads.parameters(), lr=a.lr)
    lossf = CrossEntropyLoss(ignore_index=pad)
    os.makedirs(a.out, exist_ok=True)
    print("heads: m={} hidden={} vocab_out={} parameters={}".format(
        a.m, a.hidden, vocab_out, heads.n_parameters()), flush=True)

    loader = manager.dataset.train_loader
    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        tot = np.zeros(a.m - 1)
        acc = np.zeros(a.m - 1)
        n = np.zeros(a.m - 1)
        for batch in loader:
            out, y = hidden_states(manager, batch, amp)          # (T,B,C), (B,L)
            h = out.permute(1, 0, 2)                             # (B,T,C)
            logits = heads(h)
            loss = 0.0
            for k in range(1, a.m):
                # hidden at input position t predicts the token at t+1+k
                lg = logits[k - 1][:, :h.size(1) - k, :]         # (B, T-k, V)
                tg = y[:, 1 + k:1 + k + lg.size(1)]              # (B, T-k)
                if tg.size(1) < lg.size(1):
                    lg = lg[:, :tg.size(1), :]
                lk = lossf(lg.reshape(-1, lg.size(-1)), tg.reshape(-1))
                loss = loss + lk
                m_ = tg != pad
                tot[k - 1] += float(lk) * int(m_.sum())
                acc[k - 1] += int(((lg.argmax(-1) == tg) & m_).sum())
                n[k - 1] += int(m_.sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        row = {"epoch": ep, "loss": (tot / np.maximum(n, 1)).tolist(),
               "acc": (acc / np.maximum(n, 1)).tolist(), "seconds": time.time() - t0}
        hist.append(row)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print("epoch {:4d}  loss {}  head acc {}  {:.0f}s".format(
                ep, [round(v, 4) for v in row["loss"]],
                [round(v, 4) for v in row["acc"]], row["seconds"]), flush=True)
            torch.save({"heads": heads.state_dict(), "m": a.m, "hidden": a.hidden,
                        "vocab_out": vocab_out, "base_ckpt": a.ckpt, "epoch": ep},
                       os.path.join(a.out, "heads.pt"))
            json.dump({"args": vars(a), "n_parameters": heads.n_parameters(), "history": hist},
                      open(os.path.join(a.out, "history.json"), "w"), indent=1)
        if (time.time() - t0) / 3600 > a.max_hours:
            print("time budget reached", flush=True)
            break
    torch.save({"heads": heads.state_dict(), "m": a.m, "hidden": a.hidden,
                "vocab_out": vocab_out, "base_ckpt": a.ckpt, "epoch": len(hist) - 1},
               os.path.join(a.out, "heads.pt"))
    json.dump({"args": vars(a), "n_parameters": heads.n_parameters(), "history": hist},
              open(os.path.join(a.out, "history.json"), "w"), indent=1)
    print("written:", a.out)


if __name__ == "__main__":
    main()
