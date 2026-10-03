#!/usr/bin/env python3
"""
Single, parameterised training entry point for HAND.

Replaces the ~40 one-off `main_hand_*.py` / `train_*.sh` scripts that hardcoded cluster
paths. Everything that varies between runs is a flag, so a run is fully described by its
command line (which is recorded into the output folder alongside the resolved config).

The `--encoder` flag selects between:
  fcn  : DAN's FCN_Encoder (the architecture the released checkpoints actually used)
  hand : HAND_Encoder -- gated depth-wise separable + octave conv + squeeze-excitation,
         i.e. the encoder the paper describes

Having both behind one flag is what makes the encoder ablation possible at all.

Examples:
    # Baseline page-level run on READ 2016
    python tools/train_hand.py --dataset READ_2016 --level page --variant _sem \
        --encoder fcn --output read_page_fcn --max-hours 12

    # Same, with the paper's encoder
    python tools/train_hand.py --dataset READ_2016 --level page --variant _sem \
        --encoder hand --output read_page_hand --max-hours 12
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.optim import Adam

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hand.basic.transforms import aug_config  # noqa: E402
from hand.models.baseline.fcn_encoder import FCN_Encoder  # noqa: E402
# The one experimental import on this path, and it is deliberate: --encoder hand is
# how Experiment 2 tests the manuscript's encoder against the baseline one. No
# released checkpoint uses it. See hand/models/experimental/README.md.
from hand.models.experimental.hand_encoder import HAND_Encoder  # noqa: E402
from hand.basic.scheduler import exponential_dropout_scheduler, linear_scheduler  # noqa: E402
from hand.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402

from hand_v2.models.dancer_encoder import DANCER_Encoder  # noqa: E402

# "dancer" is the Stage 2 arm E1 encoder, implemented from the DANCER supplement's Table SI 1
# (hand_v2/models/dancer_encoder.py). It is NOT the failed V1 octave encoder ("hand").
ENCODERS = {"fcn": FCN_Encoder, "hand": HAND_Encoder, "dancer": DANCER_Encoder}

# Per-level defaults: synthetic-curriculum line budget, decode cap, and whether the
# two-step "hand_encoding" layout scheme applies.
#
# hand_encoding splits the token stream into a line pass and a character pass, which
# requires a newline in the charset to mark line boundaries. Sub-line levels (line, word,
# character) have no newline, so enabling it raises ValueError: '\n' is not in list.
# It also travels with use_line_indices: with hand_encoding off the decoder falls back to
# a plain 1D sinusoidal positional encoding.
LEVEL_DEFAULTS = {
    "line":        dict(max_nb_lines=1,  min_nb_lines=1,  max_char_prediction=200,  hand_encoding=False),
    "word":        dict(max_nb_lines=1,  min_nb_lines=1,  max_char_prediction=50,   hand_encoding=False),
    "character":   dict(max_nb_lines=1,  min_nb_lines=1,  max_char_prediction=20,   hand_encoding=False),
    "paragraph":   dict(max_nb_lines=10, min_nb_lines=1,  max_char_prediction=1000, hand_encoding=True),
    "page":        dict(max_nb_lines=30, min_nb_lines=5,  max_char_prediction=3000, hand_encoding=True),
    "double_page": dict(max_nb_lines=60, min_nb_lines=10, max_char_prediction=6000, hand_encoding=True),
    "triple_page": dict(max_nb_lines=90, min_nb_lines=15, max_char_prediction=9000, hand_encoding=True),
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_or_mtime_stamp():
    """Best-effort provenance stamp for the results record."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git:{}".format(int(os.path.getmtime(os.path.join(ROOT, "hand"))))


def build_params(a):
    lvl = LEVEL_DEFAULTS[a.level]
    hand_encoding = lvl["hand_encoding"] and not a.no_hand_encoding
    data_path = os.path.join(ROOT, "formatted", "{}_{}{}".format(a.dataset, a.level, a.variant))
    if not os.path.isdir(data_path):
        raise FileNotFoundError(data_path)

    synthetic = None
    if not a.no_synthetic:
        synthetic = {
            "mode": "document",
            "page_syn_mode": "typed",
            "init_proba": a.syn_init_proba,
            "end_proba": a.syn_end_proba,
            "num_steps_proba": 100000,
            "proba_scheduler_function": linear_scheduler,
            "start_scheduler_at_max_line": True,
            "dataset_level": a.level,
            "curriculum": True,
            "crop_curriculum": True,
            "curr_start": 0,
            "curr_step": a.curr_step,
            "min_nb_lines": lvl["min_nb_lines"],
            "max_nb_lines": lvl["max_nb_lines"],
            "padding_value": 255,
            "max_char_per_line": 100,
            "mix_paragraphs": True,
            "rimes_sem_order": False,
            "config": {
                "background_color_default": (255, 255, 255),
                "background_color_eps": 15,
                "text_color_default": (0, 0, 0),
                "text_color_eps": 15,
                "font_size_min": 35,
                "font_size_max": 45,
                "color_mode": "RGB",
                "padding_left_ratio_min": 0.00,
                "padding_left_ratio_max": 0.05,
                "padding_right_ratio_min": 0.02,
                "padding_right_ratio_max": 0.2,
                "padding_top_ratio_min": 0.02,
                "padding_top_ratio_max": 0.1,
                "padding_bottom_ratio_min": 0.02,
                "padding_bottom_ratio_max": 0.1,
            },
        }

    transfer = None
    if a.init_from:
        init_path = a.init_from if os.path.isabs(a.init_from) else os.path.join(ROOT, a.init_from)
        if not os.path.exists(init_path):
            raise FileNotFoundError(init_path)
        # [state-dict key, checkpoint path, strict, reset-optimizer-state]
        transfer = {
            "encoder": ["encoder", init_path, True, True],
            "decoder": ["decoder", init_path, True, False],
        }

    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class": OCRDataset,
            "datasets": {a.dataset: data_path},
            "train": {"name": "{}-train".format(a.dataset),
                      "datasets": [(a.dataset, "train")]},
            "valid": {"{}-valid".format(a.dataset): [(a.dataset, "valid")]},
            "config": {
                "balance_datasets": True,
                "load_in_memory": False,
                "worker_per_gpu": a.workers,
                "width_divisor": 8,
                "height_divisor": 32,
                "padding_value": 0,
                "padding_token": None,
                "charset_mode": "seq2seq",
                "constraints": (["add_eot", "add_sot", "hand_encoding"] if hand_encoding
                                else ["add_eot", "add_sot"]),
                "layout_token_identity": a.layout_token_identity,
                "normalize": True,
                "preprocessings": [{"type": "to_RGB"}],
                "augmentation": None if a.no_augment else aug_config(0.9, 0.1),
                "synthetic_data": synthetic,
            },
        },
        "model_params": {
            "models": {"encoder": ENCODERS[a.encoder], "decoder": GlobalHTADecoder},
            "transfer_learning": transfer,
            "transfered_charset": True,
            # DAN uses 1 (<eot> only); HAND V1's released runs used 3. The default keeps every
            # historical run reproducible; Stage 1 arm A1 selects DAN's value explicitly.
            "additional_tokens": a.additional_tokens,
            "input_channels": 3,
            "dropout": a.dropout,
            "enc_dim": 256,
            "nb_layers": 5,
            "pe_h_max": 500,
            "pe_w_max": 1000,
            "l_max": 15000,
            "dec_num_layers": a.dec_layers,
            "dec_num_heads": 4,
            "dec_res_dropout": 0.1,
            "dec_pred_dropout": 0.1,
            "dec_att_dropout": 0.1,
            "dec_dim_feedforward": 256,
            "attention_win": 100,
            "use_tokens_from_all_lines": True,
            "use_first_pass_tokens": True,
            "use_line_indices": hand_encoding,
            "two_step_pos_enc_mode": "cat",
            "dropout_scheduler": {"function": exponential_dropout_scheduler, "T": 5e4},
            # HAND_Encoder ablation switches (ignored by FCN_Encoder)
            "enc_use_octave": not a.no_octave,
            "enc_use_gate": not a.no_gate,
            "enc_use_se": not a.no_se,
            "enc_use_residual": not a.no_residual,
            "enc_octave_alpha": a.octave_alpha,
            # Experiment 2.1: octave implementation and encoder depth.
            # Defaults ("legacy", 5) reproduce Experiment 2 exactly.
            "enc_octave_impl": a.octave_impl,
            "enc_blocks": a.enc_blocks,
            # Adaptive-depth decoder: train one shared-weight stack that stays usable at
            # several depths, so inference can trade compute for accuracy per document.
            "dec_elastic_depths": a.elastic_depths,
            "dec_share_memory_kv": getattr(a, "share_memory_kv", False),
        },
        "training_params": {
            "output_folder": a.output,
            "max_nb_epochs": a.max_epochs,
            "max_training_time": int(a.max_hours * 3600),
            "load_epoch": "last" if a.resume else None,
            "interval_save_weights": None,
            "batch_size": a.batch_size,
            "valid_batch_size": a.batch_size,
            "test_batch_size": a.batch_size,
            "use_amp": True,
            "label_smoothing": getattr(a, "label_smoothing", 0.0),
            **({"gradient_clipping": {"models": ["encoder", "decoder"],
                                       "max": getattr(a, "grad_clip", 1.0)}}
               if getattr(a, "grad_clip", 1.0) > 0 else {}),
            "nb_gpu": 1,
            "optimizers": {"all": {"class": Adam,
                                   "args": {"lr": a.lr, "amsgrad": False}}},
            "lr_schedulers": None,
            "eval_on_valid": True,
            "eval_on_valid_interval": a.eval_interval,
            "focus_metric": "cer",
            "expected_metric_value": "low",
            "set_name_focus_metric": "{}-valid".format(a.dataset),
            "train_metrics": ["loss", "cer", "wer"],
            "eval_metrics": ["cer", "wer"],
            "force_cpu": False,
            "max_line_pred": 100,
            "max_pred_per_line": 150,
            "max_char_prediction": lvl["max_char_prediction"],
            # Teacher forcing injects errors into the decoder's input so training
            # conditions look more like autoregressive inference. The released recipe
            # pinned this at a constant 0.2. The measured train/test gap (train CER 1.6%
            # vs test CER 4.6% at page level) is exposure bias, so this rate is the direct
            # lever on it: ramping it up trades teacher-forced accuracy for robustness to
            # the model's own mistakes at inference.
            # Scheduled sampling: fraction of decoder-input positions replaced by the
            # model's own prediction (not random noise). 0 disables it.
            "scheduled_sampling_rate": a.scheduled_sampling,
            "teacher_forcing_scheduler": {"min_error_rate": a.tf_min_error,
                                          "max_error_rate": a.tf_max_error,
                                          "total_num_steps": a.tf_ramp_steps},
            "ddp_rank": 0,
        },
    }

    if synthetic:
        # The original recipe suppressed validation until the synthetic curriculum had
        # ramped to full page complexity: curr_start + curr_step * (max_lines - min_lines).
        # At page level that is 125k steps, which a fixed-hours budget never reaches at
        # batch 8 (~44 steps/epoch) -- the run then ends with best=None and no `best`
        # checkpoint, making it useless for comparison. Cap it so every run validates and
        # produces a best checkpoint within its budget.
        curriculum_steps = (synthetic["curr_start"]
                            + synthetic["curr_step"]
                            * (synthetic["max_nb_lines"] - synthetic["min_nb_lines"]))
        params["training_params"]["start_valid_from_steps"] = (
            a.start_valid_from_steps if a.start_valid_from_steps is not None
            else min(curriculum_steps, 15000))

    return params


def build_arg_parser():
    """The run's full flag set. Exposed so hand_v2/ tooling can reuse it verbatim."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["READ_2016", "IAM", "KHATT", "AHAWP"])
    ap.add_argument("--level", required=True, choices=sorted(LEVEL_DEFAULTS))
    ap.add_argument("--variant", default="", help='"_sem" for layout-token datasets')
    ap.add_argument("--encoder", default="fcn", choices=sorted(ENCODERS))
    ap.add_argument("--output", required=True, help="folder under outputs/")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to warm-start encoder+decoder from")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    # D1 diagnostic (experiments/p0_1_layout_ablation): this repository clips encoder and
    # decoder gradients separately at norm 1.0; DAN's reference implementation does not clip
    # at all, and that is one of the five behaviourally live differences left by
    # X1_CONFIG_AUDIT.md. 0 disables clipping entirely. The default is unchanged (1.0), so
    # every existing run and every existing config hash is unaffected.
    ap.add_argument("--share-memory-kv", action="store_true",
                    help="E4: one visual key/value projection shared by every decoder layer "
                         "(-921,088 parameters at depth 8). Off by default")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label-smoothing epsilon on the decoder cross-entropy; 0.0 (default) is "
                         "the unsmoothed loss every existing result was produced with")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="gradient-clipping norm applied per model; 0 disables clipping")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--dec-layers", type=int, default=8)
    ap.add_argument("--additional-tokens", type=int, default=3,
                    help="decision-layer tokens beyond the charset (DAN uses 1, HAND V1 used 3)")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--max-epochs", type=int, default=10000)
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--curr-step", type=int, default=5000)
    ap.add_argument("--syn-init-proba", type=float, default=0.8)
    ap.add_argument("--syn-end-proba", type=float, default=0.2)
    # --- HAND_Encoder component ablation ---
    ap.add_argument("--no-octave", action="store_true", help="ablate octave convolution")
    ap.add_argument("--no-gate", action="store_true",
                    help="ablate gating (plain depth-wise separable conv instead)")
    ap.add_argument("--no-se", action="store_true", help="ablate squeeze-and-excitation")
    ap.add_argument("--no-residual", action="store_true", help="ablate residual connections")
    ap.add_argument("--octave-impl", default="legacy", choices=["legacy", "corrected"],
                    help="'legacy' is what Experiment 2 used; 'corrected' keeps the "
                         "low-frequency path always active and upsamples bilinearly")
    ap.add_argument("--enc-blocks", type=int, default=5,
                    help="HAND_Encoder depth. 5 is the paper's specification; extra "
                         "blocks are stride-1 and preserve the output geometry")
    ap.add_argument("--octave-alpha", type=float, default=0.5)
    ap.add_argument("--elastic-depths", type=int, nargs="+", default=None,
                    metavar="D",
                    help="train the decoder to be usable at these depths, e.g. 2 4 6 8")
    ap.add_argument("--scheduled-sampling", type=float, default=0.0, metavar="P",
                    help="replace this fraction of decoder inputs with the model's own "
                         "predictions (true scheduled sampling; costs one extra forward "
                         "pass). Distinct from --tf-*, which injects random tokens.")
    ap.add_argument("--tf-min-error", type=float, default=0.2,
                    help="teacher-forcing error rate at the start of the ramp")
    ap.add_argument("--tf-max-error", type=float, default=0.2,
                    help="teacher-forcing error rate at the end of the ramp "
                         "(raise above --tf-min-error to attack exposure bias)")
    ap.add_argument("--tf-ramp-steps", type=float, default=5e4,
                    help="steps over which the teacher-forcing error rate ramps")
    ap.add_argument("--start-valid-from-steps", type=int, default=None,
                    help="suppress validation until this step (default: "
                         "min(curriculum ramp, 15000))")
    ap.add_argument("--no-hand-encoding", action="store_true",
                    help="force the plain 1D positional encoding (auto-set for sub-line levels)")
    ap.add_argument("--layout-token-identity", action="store_true",
                    help="identify layout tokens by identity instead of by charset position. "
                         "The shipped positional test (id >= len(char_only_set)) assumes every "
                         "non-character sorts to the tail, but char_only_set also drops the "
                         "newline, which sorts to index 0 -- so the boundary lands one index low "
                         "and exactly one real character per corpus is read as a layout token "
                         "(READ_2016_page_sem_dan: the em dash; IAM_page: 'z'). Off by default: "
                         "every recorded run used the positional test")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: verify the pipeline trains and evaluates, then exit")
    return ap


def main():
    a = build_arg_parser().parse_args()

    if a.smoke:
        a.max_hours = min(a.max_hours, 0.05)
        a.max_epochs = min(a.max_epochs, 2)
        a.eval_interval = 1

    os.chdir(ROOT)  # init_paths() builds 'outputs/...' relative to cwd
    set_seed(a.seed)

    params = build_params(a)
    run_dir = os.path.join(ROOT, "outputs", a.output)
    os.makedirs(run_dir, exist_ok=True)

    # Record exactly how this run was invoked, for reproduction.
    with open(os.path.join(run_dir, "run_command.json"), "w") as f:
        json.dump({
            "argv": sys.argv,
            "args": vars(a),
            "code_version": git_or_mtime_stamp(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)

    print("=" * 72)
    print("HAND training | dataset={} level={}{} encoder={}".format(
        a.dataset, a.level, a.variant, a.encoder))
    print("output=outputs/{}  batch={}  lr={}  seed={}  hand_encoding={}".format(
        a.output, a.batch_size, a.lr, a.seed,
        LEVEL_DEFAULTS[a.level]["hand_encoding"] and not a.no_hand_encoding))
    print("=" * 72, flush=True)

    manager = Manager(params)
    manager.load_model()

    n_enc = sum(p.numel() for p in manager.models["encoder"].parameters())
    n_dec = sum(p.numel() for p in manager.models["decoder"].parameters())
    print("Parameters: encoder={:.3f}M decoder={:.3f}M total={:.3f}M".format(
        n_enc / 1e6, n_dec / 1e6, (n_enc + n_dec) / 1e6), flush=True)

    manager.train()

    # Reload the best-CER weights and report on every split. If training never produced
    # a "best" checkpoint (e.g. it stopped before the first validation), get_checkpoint()
    # would return None and the model would be silently re-initialised at random -- so
    # fall back to "last" explicitly and say which weights are being reported.
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    available = os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else []
    if any("best" in f for f in available):
        which = "best"
    elif any("last" in f for f in available):
        which = "last"
        print("WARNING: no 'best' checkpoint was produced (training may have stopped "
              "before the first validation). Reporting the LAST checkpoint instead.",
              flush=True)
    else:
        print("ERROR: no checkpoint in {} -- skipping evaluation.".format(ckpt_dir),
              flush=True)
        return
    manager.params["training_params"]["load_epoch"] = which
    manager.load_model()
    print("Evaluating checkpoint: {} (epoch {})".format(which, manager.latest_epoch),
          flush=True)

    for split in ["test", "valid"]:
        split_dir = os.path.join(ROOT, "formatted",
                                 "{}_{}{}".format(a.dataset, a.level, a.variant), split)
        if os.path.isdir(split_dir) and os.listdir(split_dir):
            manager.predict("{}-{}".format(a.dataset, split), [(a.dataset, split)],
                            ["cer", "wer", "time"], output=True)


if __name__ == "__main__":
    main()
