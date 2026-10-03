#!/usr/bin/env python3
"""
Reproducible test-set evaluation for trained HAND checkpoints.

Drives the project's own Manager/predict path so decoding and metric computation are
identical to those used during training, then writes measured CER/WER (and per-sample
predictions) to JSON under results_real/.

Usage:
    python tools/evaluate_hand.py --model read_page --split test
    python tools/evaluate_hand.py --model all --split test valid
"""
import argparse
import json
import os
import re
import random
import sys
import time

import numpy as np
import torch
from torch.optim import Adam

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hand.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder as StdDecoder  # noqa: E402
from hand.models.baseline.fcn_encoder import FCN_Encoder  # noqa: E402

# Each entry pins the formatted dataset, the document level and the checkpoint the
# corresponding training run produced. variant="_sem" marks datasets whose labels
# carry layout (semantic) tokens.
MODELS = {
    # `hand_encoding` must match what the run was trained with, and it travels with
    # `use_line_indices`. READ and AHAWP used ["add_eot", "add_sot", "hand_encoding"] plus
    # use_line_indices=True, so their decoder builds a two-part line/offset positional
    # encoding. IAM and KHATT dropped hand_encoding and omitted use_line_indices entirely
    # (the cluster training entry points for those two runs), and
    # dan_decoder defaults that flag to False -- so those models use a plain 1D
    # sinusoidal encoding instead. Forcing the wrong one silently yields ~86% CER.
    "read_page": dict(dataset="READ_2016", level="page", variant="_sem",
                      ckpt="models/read_page/best_model.pt", hand_encoding=True),
    "read_double_page": dict(dataset="READ_2016", level="double_page", variant="_sem",
                             ckpt="models/read_double_page/best_model.pt", hand_encoding=True),
    "read_triple_page": dict(dataset="READ_2016", level="triple_page", variant="_sem",
                             ckpt="models/read_triple_page/best_model.pt", hand_encoding=True),
    "iam_page": dict(dataset="IAM", level="page", variant="",
                     ckpt="models/iam_page/best_model.pt", hand_encoding=False),
    "khatt_paragraph": dict(dataset="KHATT", level="paragraph", variant="",
                            ckpt="models/khatt_paragraph/best_model.pt", hand_encoding=False),
    # trained by tools/train_hand.py (see outputs/read_line_fcn/run_command.json)
    "read_line": dict(dataset="READ_2016", level="line", variant="_synthetic",
                      ckpt="outputs/read_line_fcn/checkpoints/best_238.pt",
                      hand_encoding=False),
    # recovered DAN-recipe baseline, five-token scheme, 500 k samples (registry
    # 20260911T032022Z_s1_A1fixedR1_s0; trained with --no-hand-encoding --additional-tokens 1)
    "read_page_a1fixed": dict(dataset="READ_2016", level="page", variant="_sem_dan",
                              ckpt="outputs/s1_A1fixedR1_s0/checkpoints/best_1390.pt",
                              hand_encoding=False),
    "ahawp_paragraph": dict(dataset="AHAWP", level="paragraph", variant="",
                            ckpt="models/ahawp_paragraph/best_model.pt", hand_encoding=True),
}


def apply_checkpoint_architecture(params, ckpt_path, label=None):
    """
    Make `params` describe the architecture the checkpoint was actually trained with.

    Every tool that loads a HAND checkpoint defaulted to the anchor's architecture -- 8 decoder
    layers, untied projections. That assumption broke three times in two days:

      * seed_variance_analysis on E5's 6-layer checkpoint -> load_state_dict, 52 missing keys
      * seed_variance_analysis on E4's shared-K/V checkpoint -> load_optimizers, group mismatch
      * efficiency_bench on E4 -> would NOT have crashed, and would have reported 7,033,700
        parameters instead of 6,112,612, silently voiding the measurement

    The last is the dangerous one, so this lives in one place and every tool calls it. Returns the
    list of adjustments made, for logging.
    """
    notes = []
    n_layers = detect_decoder_layers(ckpt_path)
    if n_layers != params["model_params"]["dec_num_layers"]:
        params["model_params"]["dec_num_layers"] = n_layers
        notes.append("decoder depth {}".format(n_layers))
    if detect_shared_memory_kv(ckpt_path):
        params["model_params"]["dec_share_memory_kv"] = True
        notes.append("shared visual K/V")
    if notes and label:
        print("  {}: {} (from the checkpoint)".format(label, ", ".join(notes)), flush=True)
    return notes


def detect_shared_memory_kv(ckpt_path):
    """
    Recover whether a checkpoint was trained with `--share-memory-kv`.

    The E4 arm ties one cross-attention key/value projection across all decoder layers. Sharing is
    invisible in the state-dict KEY SET -- one copy of the tensor is written per layer key -- so a
    model built without the flag loads those weights and computes an identical forward pass, but
    reports the UNTIED parameter count (7,033,700 instead of 6,112,612). For an efficiency
    measurement that is the whole number of interest, so detect it rather than assume.

    torch.save preserves storage sharing within a file, so the signal is structural and exact:
    summing numel over unique storages is smaller than summing over keys iff weights are tied.
    Verified 2026-09-26 -- E4 decoder 4,406,372 unique vs 5,327,460 summed (and 183 vs the anchor's
    211 optimiser named params, -28 = 7 layers x 2 projections x 2 tensors); the anchor's two sums
    are equal.
    """
    import torch as _torch
    sd = _torch.load(ckpt_path, map_location="cpu", weights_only=False).get("decoder_state_dict")
    if not sd:
        return False
    summed = sum(v.numel() for v in sd.values())
    uniq = {}
    for v in sd.values():
        uniq.setdefault(v.data_ptr(), v.numel())
    return sum(uniq.values()) < summed


def detect_decoder_layers(ckpt_path, default=8):
    """
    Recover the decoder depth a checkpoint was trained with.

    `dec_num_layers` defaults to 8, but the E5 arm trains with `--dec-layers 6`. Building an
    8-layer decoder for a 6-layer checkpoint fails in load_state_dict with 52 missing keys
    (observed 2026-09-26 on the E5 paired analysis), so read the depth back from the weights
    instead of assuming it: count the distinct `att_decoder.decoder_layers.<i>.` indices.
    """
    import torch as _torch
    d = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = d.get("decoder_state_dict")
    if not sd:
        return default
    idx = set()
    for k in sd:
        m = re.match(r"att_decoder\.decoder_layers\.(\d+)\.", k)
        if m:
            idx.add(int(m.group(1)))
    return (max(idx) + 1) if idx else default


def detect_additional_tokens(ckpt_path):
    """
    Recover the `additional_tokens` a checkpoint was trained with.

    All document-level runs used dan_decoder.GlobalHTADecoder, which sizes its output
    layer as vocab_size + additional_tokens while the embedding is always vocab_size + 3.
    The READ and AHAWP runs set additional_tokens=3; the IAM and KHATT runs set it to 1
    ("only <eot>"). Loading with the wrong value fails with a size mismatch on end_conv,
    so read it back from the checkpoint instead of assuming.

    Do NOT infer a different decoder *class* from this: advanced_decoder also happens
    to produce a vocab+1 output layer, but none of the delivered checkpoints were trained
    with it, and evaluating a dan_decoder model through the two-pass decoder yields
    garbage (~86% CER).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    charset = len(ck.get("charset") or [])
    end_conv = ck["decoder_state_dict"]["end_conv.weight"].shape[0]
    extra = end_conv - charset
    if extra not in (1, 2, 3):
        raise ValueError("{}: implausible additional_tokens={} (charset={}, end_conv={})"
                         .format(ckpt_path, extra, charset, end_conv))
    return extra


def _jsonable(v):
    """MetricManager returns numpy scalars; make them plain Python for json.dump."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_params(spec, run_dir, batch_size, max_chars, decoder_cls, additional_tokens,
                 eval_depth=None, beam_size=1, beam_lp=1.0,
                 coverage_gain=0.0, coverage_mode='log'):
    dataset_name = spec["dataset"]
    data_path = os.path.join(ROOT, "formatted", "{}_{}{}".format(
        dataset_name, spec["level"], spec["variant"]))
    if not os.path.isdir(data_path):
        raise FileNotFoundError(data_path)

    return {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class": OCRDataset,
            "datasets": {dataset_name: data_path},
            "train": {"name": "{}-train".format(dataset_name),
                      "datasets": [(dataset_name, "train")]},
            "valid": {"{}-valid".format(dataset_name): [(dataset_name, "valid")]},
            "config": {
                "balance_datasets": False,
                "load_in_memory": False,
                "worker_per_gpu": 4,
                "width_divisor": 8,
                "height_divisor": 32,
                "padding_value": 0,
                "padding_token": None,
                "charset_mode": "seq2seq",
                "constraints": (["add_eot", "add_sot", "hand_encoding"]
                                if spec.get("hand_encoding", True)
                                else ["add_eot", "add_sot"]),
                "normalize": True,
                "preprocessings": [{"type": "to_RGB"}],
                # Evaluation must never see augmentation or synthetic pages.
                "augmentation": None,
                "synthetic_data": None,
            },
        },
        "model_params": {
            "models": {"encoder": FCN_Encoder, "decoder": decoder_cls},
            "transfer_learning": None,
            "transfered_charset": True,
            "keep_charset_from_checkpoint": True,
            "additional_tokens": additional_tokens,
            "input_channels": 3,
            "dropout": 0.5,
            "enc_dim": 256,
            "nb_layers": 5,
            "pe_h_max": 500,
            "pe_w_max": 1000,
            "l_max": 15000,
            "dec_num_layers": 8,
            "dec_num_heads": 4,
            "dec_res_dropout": 0.1,
            "dec_pred_dropout": 0.1,
            "dec_att_dropout": 0.1,
            "dec_dim_feedforward": 256,
            "attention_win": 100,
            "use_tokens_from_all_lines": True,
            "use_first_pass_tokens": True,
            "use_line_indices": spec.get("hand_encoding", True),
            "two_step_pos_enc_mode": "cat",
        },
        "training_params": {
            "output_folder": run_dir,
            "max_nb_epochs": 0,
            "load_epoch": "best",
            "interval_save_weights": None,
            "batch_size": batch_size,
            "valid_batch_size": batch_size,
            "test_batch_size": batch_size,
            "use_amp": True,
            "nb_gpu": 1,
            "optimizers": {"all": {"class": Adam, "args": {"lr": 1e-4, "amsgrad": False}}},
            "lr_schedulers": None,
            "eval_on_valid": False,
            "eval_on_valid_interval": 1,
            "focus_metric": "cer",
            "expected_metric_value": "low",
            "set_name_focus_metric": "{}-valid".format(dataset_name),
            "train_metrics": ["loss", "cer", "wer"],
            "eval_metrics": ["cer", "wer"],
            "force_cpu": False,
            "max_line_pred": 100,
            "max_pred_per_line": 150,
            "max_char_prediction": max_chars,
            "eval_decoder_depth": eval_depth,
            "beam_size": beam_size,
            "coverage_gain": coverage_gain,
            "coverage_mode": coverage_mode,
            "beam_length_penalty": beam_lp,
            "ddp_rank": 0,
        },
    }


def stage_checkpoint(ckpt_src, run_dir):
    """Manager.get_checkpoint() scans outputs/<run_dir>/checkpoints for a file whose
    name contains 'best', so expose the checkpoint there without copying 84MB."""
    ckpt_dir = os.path.join(ROOT, "outputs", run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    dst = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.join(ROOT, ckpt_src), dst)
    return dst


def evaluate(model_key, splits, batch_size, max_chars, out_dir, eval_depth=None,
             beam_size=1, beam_lp=1.0, coverage_gain=0.0, coverage_mode='log'):
    spec = MODELS[model_key]
    ckpt_path = os.path.join(ROOT, spec["ckpt"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    run_dir = "eval_{}".format(model_key)
    stage_checkpoint(spec["ckpt"], run_dir)

    set_seed(0)
    additional_tokens = detect_additional_tokens(ckpt_path)
    decoder_cls = StdDecoder
    print("decoder: dan_decoder (additional_tokens={})".format(additional_tokens),
          flush=True)
    params = build_params(spec, run_dir, batch_size, max_chars, decoder_cls,
                          additional_tokens, eval_depth, beam_size, beam_lp,
                          coverage_gain, coverage_mode)
    manager = Manager(params)
    manager.load_model()

    dataset_name = spec["dataset"]
    record = {
        "model": model_key,
        "dataset": dataset_name,
        "level": spec["level"],
        "checkpoint": spec["ckpt"],
        "checkpoint_epoch": manager.latest_epoch,
        "checkpoint_best_valid_cer_recorded": manager.best,
        "charset_size": len(manager.dataset.charset),
        "hand_encoding": spec.get("hand_encoding", True),
        "use_line_indices": spec.get("hand_encoding", True),
        "decoder_variant": decoder_cls.__module__.rsplit(".", 1)[-1],
        "additional_tokens": additional_tokens,
        "eval_decoder_depth": eval_depth,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "splits": {},
    }

    for split in splits:
        split_dir = os.path.join(ROOT, "formatted",
                                 "{}_{}{}".format(dataset_name, spec["level"], spec["variant"]),
                                 split)
        if not os.path.isdir(split_dir) or not os.listdir(split_dir):
            record["splits"][split] = {"status": "missing_or_empty"}
            print("[skip] {} {} split is missing/empty".format(model_key, split))
            continue

        custom_name = "{}-{}".format(dataset_name, split)
        t0 = time.time()
        manager.predict(custom_name, [(dataset_name, split)],
                        ["cer", "wer", "time"], output=True)
        elapsed = time.time() - t0

        mm = manager.metric_manager[custom_name]
        values = mm.get_display_values(output=True)
        record["splits"][split] = {
            "status": "ok",
            "metrics": {k: _jsonable(v) for k, v in values.items()},
            "wall_clock_s": round(elapsed, 2),
        }
        print("[{}] {} -> {}".format(model_key, split, record["splits"][split]["metrics"]))

    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if eval_depth is None else "_d{}".format(eval_depth)
    if beam_size > 1:
        suffix += "_beam{}".format(beam_size)
    if coverage_gain:
        suffix += "_cov{}".format(coverage_gain)
    out_path = os.path.join(out_dir, "{}{}.json".format(model_key, suffix))
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print("wrote", out_path)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["all"],
                    help="model key(s) from MODELS, or 'all'")
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=3000,
                    help="decoding cap; raise for multi-page levels")
    ap.add_argument("--depth", type=int, nargs="+", default=[None],
                    help="pin decoder depth(s) at inference; sweep for the "
                         "accuracy/compute curve of an adaptive-depth model")
    ap.add_argument("--coverage-gain", type=float, default=0.0, metavar="G",
                    help="inference-time attention coverage penalty; 0 disables. "
                         "Discourages re-attending to already-consumed encoder positions.")
    ap.add_argument("--coverage-mode", default="log", choices=["log", "linear"])
    ap.add_argument("--beam", type=int, default=1,
                    help="beam width at inference (1 = greedy argmax, the default)")
    ap.add_argument("--beam-length-penalty", type=float, default=1.0,
                    help="divide beam score by length**penalty (1.0 = mean log-prob)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_real"))
    args = ap.parse_args()

    keys = list(MODELS) if args.model == ["all"] else args.model
    os.chdir(ROOT)  # init_paths() builds 'outputs/...' relative to cwd

    summary = {}
    for k in keys:
        print("\n" + "=" * 80 + "\nEVALUATING {}\n".format(k) + "=" * 80, flush=True)
        try:
            for d in args.depth:
                key = k if d is None else "{}_d{}".format(k, d)
                if args.beam > 1:
                    key += "_beam{}".format(args.beam)
                summary[key] = evaluate(k, args.split, args.batch_size,
                                        args.max_chars, args.out, d,
                                        args.beam, args.beam_length_penalty,
                                        args.coverage_gain, args.coverage_mode)
        except Exception as exc:  # keep going; record the failure honestly
            import traceback
            traceback.print_exc()
            summary[k] = {"model": k, "status": "FAILED", "error": repr(exc)}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSummary written to", os.path.join(args.out, "_summary.json"))


if __name__ == "__main__":
    main()
