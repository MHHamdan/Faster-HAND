#!/usr/bin/env python3
"""
Corpus-level text overlap between held-out splits and training text, for every formatted corpus.

Open audit item: an earlier dataset audit recorded RIMES leakage as *"letter templates
recur thematically; not measured"*, and a later KHATT audit then assigned the unseen-text
generalization role to *"IAM Aachen and RIMES (both text-disjoint by construction)"* on the strength
of that unmeasured assumption. This tool measures it, for RIMES and for every other formatted corpus
that `dataset_audit.json` did not then cover (it was dated 2026-09-01 and held 13 of the 26
corpora on disk; it was regenerated over all 26 on 2026-09-24 and now carries a provenance
header). The two tools remain complementary and neither supersedes the other: integrity asks
whether the same string or the same file appears in two splits, this asks how much of the
held-out text is already present in training.

Method is **identical** to the L6 audit's analysis B, and the functions are imported from it rather
than reimplemented: NFKC normalisation, diacritic/tatweel stripping, whitespace collapse, character
20-grams, and coverage = |held-out grams ∩ train grams| / |held-out grams|. The only generalisation
is that layout tokens are stripped by Unicode range (U+24B6–U+24E9, the circled-letter block used by
every layout scheme in this repository) instead of by the ten hard-coded READ tokens; on READ the two
are the same set, so the control reproduces.

READ 2016 page is the control: a corpus with no known text leakage. Its value is the reference every
other row is read against, not a threshold.

Output: experiments/p0_3_khatt_l6_audit/corpus_text_overlap.json and a printed table. Reads only.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_khatt_fixed_text import coverage_stats, ngrams, norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMATTED = os.path.join(ROOT, "formatted")
OUT = os.path.join(ROOT, "experiments", "p0_3_khatt_l6_audit", "corpus_text_overlap.json")

# The circled-letter block. Every layout scheme in this repository draws its tokens from it
# (READ five-class and three-class, RIMES), so stripping the range covers all of them.
LAYOUT = {chr(c) for c in range(0x24B6, 0x24EA)}


def texts(corpus):
    """split -> list of normalised, layout-token-free transcriptions."""
    path = os.path.join(FORMATTED, corpus, "labels.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    gt = data.get("ground_truth", data)
    out = {}
    for split in ("train", "valid", "test"):
        items = gt.get(split)
        if not isinstance(items, dict):
            continue
        acc = []
        for sample in items.values():
            text = sample.get("text") if isinstance(sample, dict) else sample
            if isinstance(text, str):
                acc.append(norm("".join(c for c in text if c not in LAYOUT)))
        out[split] = acc
    return out


def audit(corpus):
    split_texts = texts(corpus)
    if not split_texts or not split_texts.get("train"):
        return None
    train_grams = set().union(*[ngrams(t) for t in split_texts["train"]]) or set()
    train_exact = set(split_texts["train"])
    row = {"splits": {s: len(v) for s, v in split_texts.items()}}
    for split in ("valid", "test"):
        held = split_texts.get(split) or []
        if not held:
            continue
        stats = coverage_stats(train_grams, [ngrams(t) for t in held])
        if not stats:
            continue
        stats["exact_dup_in_train"] = sum(t in train_exact for t in held)
        row[split] = stats
    return row or None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="*", default=None,
                    help="formatted/ subdirectory names; default: every corpus with a labels.pkl")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    names = args.corpora or sorted(
        d for d in os.listdir(FORMATTED)
        if os.path.exists(os.path.join(FORMATTED, d, "labels.pkl")))

    results = {}
    for name in names:
        try:
            row = audit(name)
        except Exception as exc:  # a malformed corpus must not abort the sweep
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if row:
            results[name] = row

    payload = {
        "metric": "character 20-gram coverage of held-out text by training text",
        "normalisation": "NFKC, diacritics/tatweel stripped, whitespace collapsed, "
                         "layout tokens U+24B6-U+24E9 removed",
        "source": "tools/audit_corpus_text_overlap.py; helpers imported from "
                  "tools/audit_khatt_fixed_text.py",
        "control": "READ_2016_page_sem_dan",
        "corpora": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
        f.write("\n")

    hdr = f"{'corpus':38s} {'split':6s} {'n':>5s} {'mean':>7s} {'median':>7s} {'>0.5':>6s} {'>0.9':>6s} {'exact':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(results):
        row = results[name]
        if "error" in row:
            print(f"{name:38s} {row['error']}")
            continue
        for split in ("valid", "test"):
            s = row.get(split)
            if not s:
                continue
            print(f"{name:38s} {split:6s} {s['n']:5d} {s['mean']:7.3f} {s['median']:7.3f} "
                  f"{s['frac_above_0.5']:6.2f} {s['frac_above_0.9']:6.2f} {s['exact_dup_in_train']:6d}")
    print("\nwrote", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
