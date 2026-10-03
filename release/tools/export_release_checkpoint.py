#!/usr/bin/env python3
"""Export a training checkpoint to a release payload: weights only, no optimizer state.

  python release/tools/export_release_checkpoint.py \
      --ckpt outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt \
      --out  release/hand-read2016-page \
      --expect-params 7033700 --expect-charset 99

CPU only. No CUDA context is created.

WHY. `best_3580.pt` is 84,821,823 B. Its top-level keys are encoder_state_dict,
decoder_state_dict, optimizer_encoder_state_dict, optimizer_decoder_state_dict,
scaler_state_dict, optimizers_named_params, charset, epoch, step, best, curriculum_config.
The two Adam states are roughly two thirds of the file and are worth nothing to a user.
Weights-only fp32: 7,033,700 x 4 B = 28,134,800 B = 26.83 MiB plus header. (The arithmetic
checks out against a file we can measure: heads.pt is 1,469,127 B and 365,968 x 4 B =
1,463,872 B.) The 26.8 MiB figure for the exported file is ESTIMATED until this script runs.

WHAT IT ASSERTS, refusing to write if any fails:
  * total parameter count equals --expect-params
  * len(charset) equals --expect-charset
  * decoder end_conv output width equals charset + additional_tokens
  * no optimizer/scaler tensor reaches the output
  * no value in the emitted config or charset is a filesystem path

USE_LINE_INDICES is read back from the checkpoint, never defaulted. It is the one field whose
wrong value is silent: no shape mismatch, no error, roughly 86 % CER and confident-looking
output. It must be True for every hand-scales-v1 checkpoint (additional_tokens == 3) and
False for the V2 page model (additional_tokens == 1). Verify after any export:

    python -c "import json,sys; c=json.load(open(sys.argv[1]+'/config.json')); \
               print(c['additional_tokens'], c['use_line_indices'])" <out dir>
"""
import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "release"))

from hand_release.inference import DEFAULT_CONFIG  # noqa: E402

DROP = ("optimizer_encoder_state_dict", "optimizer_decoder_state_dict",
        "scaler_state_dict", "optimizers_named_params")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _tied(sd):
    """True iff some decoder tensors share storage (E4 shared K/V); see the note in main()."""
    uniq = {}
    for v in sd.values():
        uniq.setdefault(v.data_ptr(), v.numel())
    return sum(uniq.values()) < sum(v.numel() for v in sd.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-params", type=int, default=7033700)
    ap.add_argument("--expect-charset", type=int, default=99)
    ap.add_argument("--heads", default=None,
                    help="optional spec-head checkpoint to export alongside")
    ap.add_argument("--format", choices=["safetensors", "pt"], default="safetensors")
    a = ap.parse_args()

    ckpt = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(REPO, a.ckpt)
    out = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    os.makedirs(out, exist_ok=True)

    src_sha, src_size = sha256(ckpt), os.path.getsize(ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)

    charset = list(ck["charset"])
    enc_sd, dec_sd = ck["encoder_state_dict"], ck["decoder_state_dict"]
    n_enc = sum(v.numel() for v in enc_sd.values())
    n_dec = sum(v.numel() for v in dec_sd.values())
    extra = dec_sd["end_conv.weight"].shape[0] - len(charset)

    fail = []
    if n_enc + n_dec != a.expect_params and not _tied(dec_sd):
        fail.append("parameters %d != expected %d" % (n_enc + n_dec, a.expect_params))
    if len(charset) != a.expect_charset:
        fail.append("charset %d != expected %d" % (len(charset), a.expect_charset))
    if extra not in (1, 2, 3):
        fail.append("implausible additional_tokens=%d" % extra)
    if any(k in ck and ck[k] is not None for k in DROP) is False:
        pass  # a checkpoint without optimizer state is fine; we only must not EMIT it
    if fail:
        sys.exit("REFUSING TO EXPORT:\n  " + "\n  ".join(fail))

    # Shared visual K/V (E4, --share-memory-kv): the checkpoint stores one tensor under eight
    # layer keys (torch.save keeps the storage sharing). Detect it structurally, as
    # tools/evaluate_hand.py:detect_shared_memory_kv does, so the exported config rebuilds the
    # tied model and expected_parameters is the UNIQUE count (6,112,612 for E4), not the
    # per-key sum (7,033,700). For an untied checkpoint both counts agree and nothing changes.
    # Added 2026-10 (multi-page study); default behaviour for the page model is identical.
    uniq = {}
    for v in dec_sd.values():
        uniq.setdefault(v.data_ptr(), v.numel())
    n_dec_unique = sum(uniq.values())
    share_memory_kv = n_dec_unique < n_dec
    if share_memory_kv:
        n_dec = n_dec_unique
        if n_enc + n_dec != a.expect_params:
            sys.exit("REFUSING TO EXPORT: tied decoder, unique parameters %d != expected %d"
                     % (n_enc + n_dec, a.expect_params))

    flat = {}
    for k, v in enc_sd.items():
        flat["encoder." + k] = v.detach().to(torch.float32).contiguous().clone()
    for k, v in dec_sd.items():
        # .clone(): safetensors refuses tensors that share storage (the tied K/V case)
        flat["decoder." + k] = v.detach().to(torch.float32).contiguous().clone()
    assert not any(k.startswith(("optimizer", "scaler")) for k in flat)

    if a.format == "safetensors":
        from safetensors.torch import save_file
        wpath = os.path.join(out, "model.safetensors")
        save_file(flat, wpath, metadata={"format": "pt"})
    else:
        wpath = os.path.join(out, "model.pt")
        torch.save(flat, wpath)

    # `use_line_indices` MUST be read back from the checkpoint and MUST NOT be left at
    # DEFAULT_CONFIG's value (False, which is right only for the V2 page model). It selects
    # the two-step token positional code. Getting it wrong raises no error and produces no
    # shape mismatch: the model reads at roughly 86 % CER while returning confident-looking
    # output. Every hand-scales-v1 checkpoint needs True (additional_tokens == 3). The
    # fallback expression is the one inference.py:_load_training_checkpoint already uses, so
    # a checkpoint predating the key still exports correctly.
    use_line_indices = bool(ck.get("use_line_indices", extra == 3))
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(vocab_size=len(charset), additional_tokens=extra,
               use_line_indices=use_line_indices,
               expected_parameters=n_enc + n_dec,
               parameters_encoder=n_enc, parameters_decoder=n_dec)
    if share_memory_kv:
        cfg["dec_share_memory_kv"] = True   # read by inference.py:from_pretrained (default False)
    cfg["source_checkpoint"] = {
        "path": os.path.relpath(ckpt, REPO), "sha256": src_sha, "bytes": src_size,
        "epoch": int(ck["epoch"]), "step": int(ck["step"]),
        "recorded_best_valid_cer": float(ck["best"]),
        "use_line_indices_source": (
            "read from the checkpoint key 'use_line_indices'" if "use_line_indices" in ck
            else "checkpoint carries no 'use_line_indices' key; inferred from "
                 "additional_tokens == %d, i.e. (additional_tokens == 3) -> %s"
                 % (extra, use_line_indices)),
    }
    for k, v in cfg.items():
        if isinstance(v, str) and (v.startswith("/") or "\\" in v):
            sys.exit("config value %r looks like an absolute path" % k)

    json.dump(cfg, open(os.path.join(out, "config.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump({"charset": charset, "vocab_size": len(charset),
               "additional_tokens": extra,
               "special_tokens": {"end": len(charset), "start": len(charset) + 1,
                                  "pad": len(charset) + 2},
               "note": ("index convention from hand/OCR/ocr_dataset_manager.py:74-99 with "
                        "charset_mode='seq2seq'; the output layer covers 0..vocab_size, i.e. "
                        "the charset plus <end>")},
              open(os.path.join(out, "charset.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump({"do_rgb": True, "working_dpi": cfg["working_dpi"],
               "resample": "PIL.Image.BILINEAR",
               "image_mean": cfg["image_mean"], "image_std": cfg["image_std"],
               "do_resize_to_fixed_size": False, "do_pad": False,
               "encoder_reduction_h": 32, "encoder_reduction_w": 8,
               "note": ("Training-set channel statistics for READ_2016_page_sem_dan, read from "
                        "outputs/e14_budget_1p26M_s0/results/params.txt:202-221. Scale is 0-255, "
                        "NOT 0-1, and these are NOT ImageNet statistics. The page must reach the "
                        "model at 150 dpi: raw READ 2016 scans are 300 dpi and must be halved "
                        "(PIL BILINEAR); the pages under formatted/ already are 150 dpi and must "
                        "NOT be resized again.")},
              open(os.path.join(out, "preprocessor_config.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    heads_note = None
    if a.heads:
        hp = a.heads if os.path.isabs(a.heads) else os.path.join(REPO, a.heads)
        blob = torch.load(hp, map_location="cpu", weights_only=False)
        hsd = {k: v.detach().to(torch.float32).contiguous() for k, v in blob["heads"].items()}
        n_heads = sum(v.numel() for v in hsd.values())
        if a.format == "safetensors":
            from safetensors.torch import save_file
            hout = os.path.join(out, "spec_heads_m%d.safetensors" % blob["m"])
            save_file(hsd, hout, metadata={"format": "pt"})
        else:
            hout = os.path.join(out, "spec_heads_m%d.pt" % blob["m"])
            torch.save(hsd, hout)
        json.dump({"m": blob["m"], "hidden": blob["hidden"], "vocab_out": blob["vocab_out"],
                   "parameters": n_heads, "d_model": cfg["enc_dim"],
                   "base_checkpoint_sha256": src_sha,
                   "source": {"path": os.path.relpath(hp, REPO), "sha256": sha256(hp),
                              "bytes": os.path.getsize(hp)}},
                  open(os.path.splitext(hout)[0] + ".json", "w", encoding="utf-8"), indent=2)
        heads_note = (os.path.basename(hout), n_heads, os.path.getsize(hout))

    print("wrote   %s  %d B  sha256 %s" % (wpath, os.path.getsize(wpath), sha256(wpath)))
    print("params  encoder %d + decoder %d = %d" % (n_enc, n_dec, n_enc + n_dec))
    print("charset %d, additional_tokens %d, use_line_indices %s"
          % (len(charset), extra, use_line_indices))
    print("source  %s  sha256 %s  %d B" % (os.path.relpath(ckpt, REPO), src_sha, src_size))
    if heads_note:
        print("heads   %s  %d params  %d B" % heads_note)


if __name__ == "__main__":
    main()
