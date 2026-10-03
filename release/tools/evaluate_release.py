#!/usr/bin/env python3
"""One command that reproduces a published number from a released checkpoint.

  python release/tools/evaluate_release.py --model release/hand-read2016-page \
         --split test --device cuda

EXPECTED VALUES AND TOLERANCE are written next to the check, below, and are read from the
artefacts, never from prose:

  corpus CER  0.0355   (826 edits / 23,262 characters)
      experiments/benchmark_suite/profiling/seed_variance_test.json
        .per_page.e14_seed0_1p26M  -> sum(edit)=826, sum(nb)=23262, ratio 0.035508554724443295
      experiments/benchmark_suite/profiling/spec_decode_test.json .arms.greedy.cer -> 0.0355
      experiments/benchmark_suite/profiling/efficiency_hand_vs_dan_uncontended.json
        .rows["HAND_e14_1p26M::baseline"].metrics.cer -> 0.0355
  corpus WER  0.1331   (same three artefacts)
  LOER        0.0529   mAP-CER 0.9264  (same three artefacts)

TOLERANCE, set from a measurement rather than from taste. The published numbers were taken
under AMP fp16 on an RTX PRO 6000 Blackwell. This script was run on CPU in fp32 over the full
test split and produced **827 edits over 23,262 characters** against the artefact's **826 over
23,262**: 49 of 50 pages matched the reference edit distance exactly, and test_24 read 23
against 22. One edit in 23,262 characters is 0.0043 pp.

  --tol-edits 2   THE DEFAULT. CPU or GPU, fp32. Tight enough that a wrong charset, a wrong
                  additional_tokens, a wrong positional-encoding mode or a wrong
                  normalisation all fail it by orders of magnitude, loose enough to survive
                  the fp32-versus-fp16 divergence measured above.
  --tol-edits 0   only when comparing an fp32 run against an fp32 reference. This was the
                  default until 2026-09-24, which made the documented reproduction command
                  print PARITY FAIL on a correct export.
  --tol-edits 30  suggested for an AMP fp16 run - ~0.13 pp of slack. fp16 tie-breaking
                  across query-block shapes is a real effect in this codebase: it is what
                  makes m=3 and m=4 read 49/50 instead of 50/50 in spec_decode_test.json.

--out AND THE REFERENCE ARTEFACT. `release/PARITY_CPU.json` is the recorded reference run.
Do NOT pass `--out release/PARITY_CPU.json`: a reproduction attempt that goes wrong would
overwrite the reference with its own failure. As a backstop this script REFUSES to overwrite
an existing file when parity fails; write new runs somewhere else.

The character count is itself a check: 23,262 must come out exactly, because it is a property
of the ground truth and the CER definition, not of the model.

WER is tokenised with hand.basic.metric_manager.format_string_for_wer, which is the harness
that produced the published 0.1331. Splitting on whitespace instead gives 3,399 words against
the harness's 4,276 and a WER that is not comparable to anything.

MEASURED HERE, WITHOUT A GPU: see release/PARITY_CPU.json.
"""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "release"))

import editdistance  # noqa: E402

# WER must be tokenised the way the harness that produced the published number tokenises it.
# Whitespace splitting gives 3,399 words on this split against the harness's 4,276 and a WER
# that is not comparable to anything. hand/basic/metric_manager.py:254 (format_string_for_wer)
# is the definition: punctuation is separated into its own word, layout tokens are removed,
# runs of whitespace collapse to one space.
from hand.basic.metric_manager import format_string_for_wer  # noqa: E402

from hand_release.inference import (HANDRecognizer, strip_layout,  # noqa: E402
                                    LAYOUT_TOKENS)

EXPECTED = {
    "test": {"edits": 826, "chars": 23262, "cer": 0.0355, "wer": 0.1331,
             "loer": 0.0529, "map_cer": 0.9264, "n_pages": 50,
             "source": "experiments/benchmark_suite/profiling/seed_variance_test.json"
                       " (.per_page.e14_seed0_1p26M) and spec_decode_test.json (.arms.greedy)"},
    "valid": {"edits": None, "chars": 21609, "cer": 0.0395, "wer": 0.1501, "n_pages": 50,
              "source": "outputs/e14_budget_1p26M_s0/results/predict_READ_2016-valid_3580.txt"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="release/hand-read2016-page",
                    help="release directory (config.json + charset.json + model.safetensors) "
                         "or a raw training checkpoint .pt")
    ap.add_argument("--data", default="formatted/READ_2016_page_sem_dan",
                    help="formatted dataset root; its pages are ALREADY at 150 dpi")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N pages only (a smoke run)")
    ap.add_argument("--kv-cache", action="store_true")
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--source-dpi", type=int, default=None,
                    help="resolution of the input pages; omit for formatted/ pages (150 dpi)")
    ap.add_argument("--tol-edits", type=int, default=2,
                    help="edit-count slack against the reference. DEFAULT 2, set from the "
                         "measurement in this module's docstring (CPU fp32 reads 827 against "
                         "the AMP-fp16 artefact's 826). 0 is correct only when comparing an "
                         "fp32 run against an fp32 reference; 30 for an AMP fp16 run.")
    ap.add_argument("--layout-metrics", action="store_true",
                    help="also compute LOER and mAP-CER (slow: exact graph edit distance)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model = a.model if os.path.isabs(a.model) else os.path.join(REPO, a.model)
    data = a.data if os.path.isabs(a.data) else os.path.join(REPO, a.data)
    labels = pickle.load(open(os.path.join(data, "labels.pkl"), "rb"))
    gts = labels["ground_truth"][a.split]
    names = sorted(gts, key=lambda n: int(n.split("_")[1].split(".")[0]))
    if a.limit:
        names = names[:a.limit]

    r = HANDRecognizer.from_pretrained(model, device=a.device, kv_cache=a.kv_cache,
                                       speculative=a.speculative, m=a.m)
    exp = EXPECTED[a.split]
    print("model        %s" % os.path.relpath(model, REPO))
    print("device       %s   amp=%s   path=%s" % (
        r.device, a.amp, "speculative_m%d" % a.m if a.speculative
        else ("kv_cache" if a.kv_cache else "reference")))
    print("pages        %d of %d" % (len(names), exp["n_pages"]))
    print("expecting    CER %.4f  (%s edits / %s chars)   source: %s"
          % (exp["cer"], exp["edits"], exp["chars"], exp["source"]))
    print("-" * 78)

    pairs, edits, chars, w_e, w_n, t_total = [], 0, 0, 0, 0, 0.0
    per_page = []   # counts only. NEVER predictions: a per-sample dump embeds ground truth.
    for i, name in enumerate(names):
        gt = gts[name]["text"]
        t0 = time.time()
        out = r.read(os.path.join(data, a.split, name), source_dpi=a.source_dpi, amp=a.amp)
        t_total += time.time() - t0
        g, p = strip_layout(gt), out.text
        e = editdistance.eval(p, g)
        edits += e
        chars += len(g)
        lt = "".join(sorted(LAYOUT_TOKENS))
        gw = format_string_for_wer(gt, lt)
        pw = format_string_for_wer(out.raw, lt)
        w_e += editdistance.eval(pw, gw)
        w_n += len(gw)
        pairs.append((gt, out.raw))
        per_page.append({"page": i, "name": name, "edit": e, "nb": len(g),
                         "word_edit": editdistance.eval(pw, gw), "nb_words": len(gw),
                         "tokens": out.n_tokens, "wall_s": out.latency_s})
        print("  %-16s edits %4d / %4d   cum CER %.4f   %.2f s"
              % (name, e, len(g), edits / chars, out.latency_s), flush=True)

    cer, wer = edits / chars, w_e / w_n
    res = {"model": os.path.relpath(model, REPO), "split": a.split, "n_pages": len(names),
           "device": str(r.device), "amp": a.amp, "dtype": "fp16-autocast" if a.amp else "fp32",
           "decode_path": ("speculative_m%d" % a.m if a.speculative
                           else ("kv_cache" if a.kv_cache else "reference")),
           "edits": edits, "chars": chars, "cer": cer,
           "word_edits": w_e, "words": w_n, "wer": wer,
           "wer_tokenizer": "hand.basic.metric_manager.format_string_for_wer",
           "wall_s_total": t_total, "wall_s_per_page": t_total / len(names),
           "expected": exp, "tol_edits": a.tol_edits, "per_page": per_page}

    if a.layout_metrics:
        from hand.basic.layout_metrics import loer, order_invariant_metrics, READ_MATCHING_TOKENS
        res["loer"] = loer(pairs, dataset="read")
        toks = "".join(READ_MATCHING_TOKENS) + "".join(READ_MATCHING_TOKENS.values())
        res["order_invariant"] = order_invariant_metrics(pairs, toks)

    print("-" * 78)
    print("CER  %.6f   (%d edits / %d chars)    expected %.4f" % (cer, edits, chars, exp["cer"]))
    print("WER  %.6f   (%d / %d)                expected %.4f" % (wer, w_e, w_n, exp["wer"]))
    print("wall %.2f s/page on %s" % (t_total / len(names), r.device))

    ok = True
    if a.limit is None and exp["edits"] is not None:
        delta = abs(edits - exp["edits"])
        ok = delta <= a.tol_edits
        print("PARITY  %s   |%d - %d| = %d, tolerance %d edits"
              % ("PASS" if ok else "FAIL", edits, exp["edits"], delta, a.tol_edits))
        res["parity_pass"], res["edit_delta"] = ok, delta
    else:
        print("PARITY  NOT CHECKED (partial run: --limit set, or no edit reference for split)")
        res["parity_pass"] = None

    if a.out:
        out_p = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
        # A failing run must never be able to destroy a recorded reference artefact.
        # release/PARITY_CPU.json is such an artefact; so is any file the caller already has.
        if os.path.exists(out_p) and not ok:
            print("NOT WRITING %s: parity failed and that file already exists. A failed "
                  "reproduction must not overwrite a recorded result. Write elsewhere."
                  % out_p)
        else:
            json.dump(res, open(out_p, "w"), indent=2)
            print("wrote %s" % out_p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
