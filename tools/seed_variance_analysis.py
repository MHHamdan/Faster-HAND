#!/usr/bin/env python3
"""How many seeds does a comparison actually need? Measure it instead of assuming it.

`docs/ablations.md` records test CER 4.57 +- 0.39 over three 500 k seeds and a detectable
effect of ~0.9 pp at n = 3 -- a floor so high that it would swallow most real interventions
and would cost 127 GPU-h to establish at the full budget. That floor is computed from the
*unpaired* seed-to-seed spread of a corpus-level number. Two things can make it much smaller
and neither needs another training run:

  1. PAIRING. Two arms evaluated on the SAME 50 pages share the page-difficulty component of
     the error. What matters for a comparison is the variance of the per-page DIFFERENCE, not
     the variance of each arm. This script measures the across-seed per-page correlation, which
     is exactly the quantity that says how much pairing buys.
  2. THE SPLIT. Validation (50 pages) has sigma 0.06 pp across the same three seeds while test
     has 0.39 pp. Screening on validation is therefore ~6x more sensitive per GPU-hour.

Outputs, per split: each checkpoint's corpus CER; the between-seed spread; the per-page
Pearson correlation of errors between seeds; the paired per-page difference statistics; a
page-level bootstrap CI for a single run; and the detectable effect under a paired vs an
unpaired design at n = 1, 2, 3 seeds.

  CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock python3 tools/seed_variance_analysis.py \
      --split valid --out experiments/benchmark_suite/profiling/seed_variance.json
"""
import argparse
import json
import os
import sys

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
from hand.basic.metric_manager import edit_cer_from_string, nb_chars_cer_from_string, MetricManager  # noqa: E402

RUNS = {
    "s1_seed0_500k": "outputs/s1_A1fixedR1_s0/checkpoints/best_1390.pt",
    "s1_seed1_500k": "outputs/s1_A1fixedR1_s1/checkpoints/best_1265.pt",
    "s1_seed2_500k": "outputs/s1_A1fixedR1_s2/checkpoints/best_1360.pt",
    "e14_seed0_1p26M": "outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt",
    "DAN_published": "weights/dan/dan_read_page.pt",
}


def per_page(label, ckpt, split, n_pages, device="cuda"):
    set_seed(0)
    spec = dict(dataset="READ_2016", level="page", variant="_sem_dan", ckpt=ckpt,
                hand_encoding=False)
    stage_checkpoint(ckpt, "sv_{}".format(label))
    params = build_params(spec, "sv_{}".format(label), 1, MAX_CHARS["page"], GlobalHTADecoder,
                          detect_additional_tokens(os.path.join(ROOT, ckpt)))
    apply_checkpoint_architecture(params, os.path.join(ROOT, ckpt), label)
    m = Manager(params)
    # reset_optimizer=True: this is evaluation, nothing is ever stepped, and loading optimiser
    # state costs correctness rather than buying it. The E4 arm shares one visual K/V projection
    # across decoder layers, so its checkpoint's optimiser has fewer parameter groups than an
    # untied model and load_optimizers raises "parameter group that doesn't match the size of
    # optimizer's group" (observed 2026-09-26). Model weights are unaffected: sharing still writes
    # one copy of the tensor per layer key, verified identical across all 8 layers for in_proj_k
    # and in_proj_v with self_att.in_proj_q differing as a control.
    m.load_model(reset_optimizer=True)
    for mod in m.models.values():
        mod.eval()
    if split == "test":
        key = "READ_2016-test"
        m.dataset.generate_test_loader(key, [("READ_2016", "test")])
        loader = m.dataset.test_loaders[key]
    else:
        key = list(m.dataset.valid_loaders)[0]
        loader = m.dataset.valid_loaders[key]
    lt = MetricManager(["cer"], key).layout_tokens
    amp = params["training_params"]["use_amp"] and device == "cuda"
    rows = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= n_pages:
                break
            x = b["imgs"].to(m.device)
            with autocast("cuda", enabled=amp):
                f = m.models["encoder"](x)
                pf = m.models["decoder"].features_updater.get_pos_features(f)
                pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
                sx, _, _ = decode_page(m, pf, f.size(), [s[:2] for s in b["imgs_reduced_shape"]],
                                       params["training_params"]["max_char_prediction"], True)
            gt = b["raw_labels"][0]
            rows.append({"page": i, "name": b["names"][0],
                         "edit": int(edit_cer_from_string(gt, sx[0], lt)),
                         "nb": int(nb_chars_cer_from_string(gt, lt))})
    del m
    torch.cuda.empty_cache()
    return rows


def corpus_cer(rows):
    return sum(r["edit"] for r in rows) / sum(r["nb"] for r in rows)


def analyse(data, boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    labels = list(data)
    n = len(data[labels[0]])
    E = {k: np.array([r["edit"] for r in v], float) for k, v in data.items()}
    N = np.array([r["nb"] for r in data[labels[0]]], float)
    out = {"n_pages": n, "corpus_cer": {k: corpus_cer(v) for k, v in data.items()},
           "page_cer_mean": {k: float(np.mean(E[k] / N)) for k in labels}}

    # page-level bootstrap of one run's corpus CER: the uncertainty from the 50-page sample
    out["bootstrap_page_ci"] = {}
    idx = rng.integers(0, n, size=(boot, n))
    for k in labels:
        b = E[k][idx].sum(1) / N[idx].sum(1)
        out["bootstrap_page_ci"][k] = {"mean": float(b.mean()), "sd": float(b.std()),
                                       "ci95": [float(np.percentile(b, 2.5)),
                                                float(np.percentile(b, 97.5))]}
    # pairing: correlation of per-page error rates, and the paired difference
    seeds = [k for k in labels if k.startswith("s1_seed")]
    out["pairs"] = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = seeds[i], seeds[j]
            ra, rb = E[a] / N, E[b] / N
            d = (E[a] - E[b])
            db = d[idx].sum(1) / N[idx].sum(1)          # paired bootstrap of the CER difference
            out["pairs"]["{} vs {}".format(a, b)] = {
                "pearson_r_page_cer": float(np.corrcoef(ra, rb)[0, 1]),
                "corpus_cer_diff": corpus_cer(data[a]) - corpus_cer(data[b]),
                "paired_bootstrap_sd": float(db.std()),
                "paired_bootstrap_ci95": [float(np.percentile(db, 2.5)),
                                          float(np.percentile(db, 97.5))],
                "unpaired_bootstrap_sd": float(np.sqrt(
                    (E[a][idx].sum(1) / N[idx].sum(1)).var()
                    + (E[b][idx].sum(1) / N[idx].sum(1)).var()))}
    # the comparison the manuscript needs: every run against every other, paired over the
    # SAME pages. This is the statistic a "below 3.43" claim has to survive.
    out["paired_contrasts"] = {}
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i >= j:
                continue
            a, b = labels[i], labels[j]
            d = E[a] - E[b]
            db = d[idx].sum(1) / N[idx].sum(1)
            lo, hi = float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))
            out["paired_contrasts"]["{} - {}".format(a, b)] = {
                "diff_pp": (corpus_cer(data[a]) - corpus_cer(data[b])) * 100,
                "paired_bootstrap_sd_pp": float(db.std()) * 100,
                "ci95_pp": [lo * 100, hi * 100],
                "p_two_sided_bootstrap": float(2 * min((db >= 0).mean(), (db <= 0).mean())),
                "pearson_r_page_cer": float(np.corrcoef(E[a] / N, E[b] / N)[0, 1]),
                "n_pages_a_better": int((E[a] < E[b]).sum()),
                "n_pages_b_better": int((E[b] < E[a]).sum())}

    # between-seed spread of the corpus number (the quantity the ledger reports)
    if len(seeds) >= 2:
        v = np.array([corpus_cer(data[k]) for k in seeds])
        out["between_seed"] = {"mean": float(v.mean()), "sd_ddof1": float(v.std(ddof=1)),
                               "spread": float(v.max() - v.min()), "n": len(seeds)}
        # detectable effect, two-sided alpha .05, power .8, paired vs unpaired, per n
        sd_between = float(v.std(ddof=1))
        sd_paired = float(np.mean([p["paired_bootstrap_sd"] for p in out["pairs"].values()]))
        out["detectable_effect_pp"] = {
            "unpaired_seeds": {str(k): 2.8 * sd_between * np.sqrt(2.0 / k) * 100
                               for k in (1, 2, 3, 5)},
            "paired_pages_one_seed_each": 2.8 * sd_paired * 100,
            "note": "2.8 = z_{.975}+z_{.8}; unpaired uses the between-seed sd, paired uses the "
                    "page-level bootstrap sd of the per-page difference at one seed per arm"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--n-pages", type=int, default=50)
    ap.add_argument("--runs", default=",".join(RUNS),
                    help="comma-separated. Either a name from RUNS, or 'label=path/to/ckpt.pt' for "
                         "a checkpoint not listed there -- so a new arm can be analysed without "
                         "editing this file at verdict time.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # Resolve every entry BEFORE loading anything, so a typo or a missing checkpoint fails in
    # milliseconds instead of after the first arm has been decoded.
    selected = {}
    for entry in a.runs.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            k, path = entry.split("=", 1)
            k, path = k.strip(), path.strip()
        elif entry in RUNS:
            k, path = entry, RUNS[entry]
        else:
            raise SystemExit("unknown run {!r}: pass 'label=path/to/checkpoint.pt', or one of: {}"
                             .format(entry, ", ".join(RUNS)))
        if not os.path.exists(os.path.join(ROOT, path)):
            raise SystemExit("checkpoint for {!r} does not exist: {}".format(k, path))
        selected[k] = path

    data = {}
    for k, path in selected.items():
        print("== {} ({})".format(k, path), flush=True)
        data[k] = per_page(k, path, a.split, a.n_pages)
        print("   corpus CER {:.4f}".format(corpus_cer(data[k])), flush=True)
    res = {"split": a.split, "runs": {k: selected[k] for k in data},
           "per_page": data, "analysis": analyse(data)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["analysis"], indent=1))
    print("written:", a.out)
