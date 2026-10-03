#!/usr/bin/env python3
"""
KHATT leakage control L6 — decide the fixed-/unique-text assignment of every local paragraph.

Open audit item (data/README.md §2.2): KHATT mixes fixed-text paragraphs (every
writer copies the same source text) with unique-text paragraphs. L6 requires fixed-text paragraphs
to be excluded from query sets. The assignment per paragraph index was never established locally,
and an earlier string-repetition probe was recorded as not trusted.

Two defects of that probe are avoided here:

1. **Label source.** `KHATT/labels/` holds no transcription for the test split (836 forms, none of
   them test); the authoritative source is `{train,val,test}_text.txt`, which is what the formatter
   consumes. Auditing `labels/` silently drops the test set.
2. **Exact matching.** Writers wrote different numbers of lines, so two copies of the same fixed
   source text differ in length: the paragraph texts are *prefixes of one source at varying
   truncation*. Exact-duplicate counting therefore reports ~5% where prefix matching reports ~87%.

Method: reconstruct paragraphs by grouping line ids (AHTD3A####_ParaN_L) exactly as the formatter
does, normalise (NFKC, strip diacritics/tatweel, collapse whitespace), and classify a paragraph as
fixed-text when the first N characters match the canonical fixed sentence under a fuzzy ratio,
which absorbs the transcription variants (ضرغام / ضرعام / خرغام, بصحبة / بصحبه).

Three analyses are reported, the last two decisive:

A. fixed-/unique-text classification per split and paragraph index (the original open item);
B. **corpus-level n-gram leakage**: the fraction of each held-out paragraph's character 20-grams
   that occur anywhere in the training text, with READ 2016 page as a control;
C. **can a re-split repair it**: paragraphs are clustered by 20-gram Jaccard and whole clusters are
   assigned to a synthetic train/test partition, then B is recomputed.

Output: experiments/p0_3_khatt_l6_audit/khatt_l6_audit.json and a printed table. Reads only.
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys
import statistics
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "KHATT")
TEXT_FILES = {"train": "train_text.txt", "valid": "val_text.txt", "test": "test_text.txt"}
OUT = os.path.join(ROOT, "experiments", "p0_3_khatt_l6_audit", "khatt_l6_audit.json")
LINE_RE = re.compile(r"^(AHTD3A\d+_Para\d+)_(\d+)$")


def norm(t):
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[ً-ْـ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def load_paragraphs():
    """id -> normalised paragraph text, grouped from the split text files."""
    out = {}
    for split, fname in TEXT_FILES.items():
        path = os.path.join(RAW, fname)
        lines = collections.defaultdict(list)
        with open(path, encoding="utf-8") as f:
            for row in f:
                row = row.strip()
                if not row:
                    continue
                sid, _, text = row.partition(" ")
                sid = sid.split("/")[-1]
                m = LINE_RE.match(sid)
                if m:
                    lines[m.group(1)].append((int(m.group(2)), text))
        for para, items in lines.items():
            out[(split, para)] = norm("\n".join(t for _, t in sorted(items)))
    return out


def ngrams(t, n=20):
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def coverage_stats(train_grams, held_out):
    cov = [len(g & train_grams) / len(g) for g in held_out if g]
    if not cov:
        return {}
    return {"n": len(cov), "mean": round(statistics.mean(cov), 4),
            "median": round(statistics.median(cov), 4),
            "frac_above_0.5": round(sum(c > 0.5 for c in cov) / len(cov), 4),
            "frac_above_0.9": round(sum(c > 0.9 for c in cov) / len(cov), 4)}


def read_control():
    """READ 2016 page, layout tokens stripped: the same statistic on a corpus known to be clean."""
    import pickle
    path = os.path.join(ROOT, "formatted", "READ_2016_page_sem_dan", "labels.pkl")
    if not os.path.exists(path):
        return {}
    layout = set("\u24b6\u24b7\u24c3\u24c5\u24c8\u24d0\u24d1\u24dd\u24df\u24e2")
    with open(path, "rb") as f:
        gt = pickle.load(f)["ground_truth"]
    txt = lambda s: norm("".join(c for c in s["text"] if c not in layout))
    train = set().union(*[ngrams(txt(s)) for s in gt["train"].values()])
    return {sp: coverage_stats(train, [ngrams(txt(s)) for s in gt[sp].values()])
            for sp in ("valid", "test")}


def cluster_disjoint_probe(keys, grams, jaccard=0.5):
    """Assign whole text-clusters to a synthetic 80/20 split, then re-measure leakage."""
    n = len(keys)
    inv = collections.defaultdict(list)
    for i, g in enumerate(grams):
        for x in g:
            inv[x].append(i)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = set()
    for lst in inv.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                pair = (lst[i], lst[j])
                if pair in seen:
                    continue
                seen.add(pair)
                a, b = grams[pair[0]], grams[pair[1]]
                if len(a | b) and len(a & b) / len(a | b) >= jaccard:
                    ra, rb = find(pair[0]), find(pair[1])
                    if ra != rb:
                        parent[rb] = ra
    clusters = collections.defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    ordered = sorted(clusters.values(), key=len, reverse=True)
    train, test = [], []
    for c in ordered:
        (train if len(train) <= 4 * len(test) else test).extend(c)
    train_grams = set().union(*[grams[i] for i in train])
    return {"n_clusters": len(ordered), "largest_cluster": len(ordered[0]),
            "singletons": sum(1 for c in ordered if len(c) == 1),
            "train_paragraphs": len(train), "test_paragraphs": len(test),
            "leakage_after_resplit": coverage_stats(train_grams, [grams[i] for i in test])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-chars", type=int, default=40)
    ap.add_argument("--ratio", type=float, default=0.80)
    a = ap.parse_args()

    texts = load_paragraphs()
    if not texts:
        sys.exit("no paragraphs reconstructed")

    # The canonical fixed text is the most frequent prefix across the corpus, taken from the data
    # rather than hard-coded, so the audit does not assume which sentence KHATT used.
    pref_counts = collections.Counter(t[:a.prefix_chars] for t in texts.values())
    canonical, canonical_n = pref_counts.most_common(1)[0]

    def is_fixed(t):
        return difflib.SequenceMatcher(None, t[:a.prefix_chars], canonical).ratio() >= a.ratio

    per_split = collections.defaultdict(lambda: collections.Counter())
    per_index = collections.defaultdict(lambda: collections.Counter())
    unique_texts = collections.defaultdict(dict)
    for (split, para), t in texts.items():
        kind = "fixed" if is_fixed(t) else "unique"
        per_split[split][kind] += 1
        per_index[para.split("_Para")[1]][kind] += 1
        if kind == "unique":
            unique_texts[split][para] = t

    train_unique = set(unique_texts["train"].values())
    leakage = {}
    for split in ("valid", "test"):
        vals = list(unique_texts[split].values())
        exact = sum(1 for t in vals if t in train_unique)
        near = sum(1 for t in vals
                   if any(difflib.SequenceMatcher(None, t[:60], u[:60]).ratio() >= 0.80
                          for u in train_unique))
        leakage[split] = {"n": len(vals), "exact_in_train": exact, "near_dup_in_train": near}

    grams = [ngrams(texts[k]) for k in texts]
    key_list = list(texts)
    train_grams = set().union(*[ngrams(t) for (sp, _), t in texts.items() if sp == "train"])
    leak = {sp: coverage_stats(train_grams, [ngrams(t) for (s2, _), t in texts.items() if s2 == sp])
            for sp in ("valid", "test")}
    control = read_control()
    resplit = cluster_disjoint_probe(key_list, grams)

    report = {
        "canonical_fixed_prefix": canonical,
        "canonical_prefix_occurrences": canonical_n,
        "params": {"prefix_chars": a.prefix_chars, "ratio": a.ratio},
        "paragraphs_total": len(texts),
        "per_split": {k: dict(v) for k, v in per_split.items()},
        "per_paragraph_index": {k: dict(v) for k, v in per_index.items()},
        "unique_only_leakage": leakage,
        "ngram_leakage_khatt": leak,
        "ngram_leakage_read_control": control,
        "cluster_disjoint_resplit": resplit,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("canonical fixed prefix (from data, x{}): {!r}".format(canonical_n, canonical))
    print("\nsplit    total   fixed          unique")
    for split in ("train", "valid", "test"):
        c = per_split[split]
        n = sum(c.values())
        print("  {:6s} {:5d}   {:4d} ({:5.1f}%)  {:4d} ({:5.1f}%)".format(
            split, n, c["fixed"], 100 * c["fixed"] / n, c["unique"], 100 * c["unique"] / n))
    print("\nparagraph index   total   fixed          unique")
    for idx in sorted(per_index):
        c = per_index[idx]
        n = sum(c.values())
        print("  Para{:12s} {:4d}   {:4d} ({:5.1f}%)  {:4d} ({:5.1f}%)".format(
            idx, n, c["fixed"], 100 * c["fixed"] / n, c["unique"], 100 * c["unique"] / n))
    print("\nafter L6 exclusion (unique-text only):")
    for split, v in leakage.items():
        print("  {:6s} {:3d} paragraphs | exact text also in train: {:3d} | near-duplicate in train: {:3d}".format(
            split, v["n"], v["exact_in_train"], v["near_dup_in_train"]))
    print("\ncorpus-level leakage — fraction of each held-out paragraph's char 20-grams present in train:")
    for sp in ("valid", "test"):
        v = leak[sp]
        print("  KHATT {:6s} n={:4d} mean {:.3f} median {:.3f} | >0.5: {:.0%} | >0.9: {:.0%}".format(
            sp, v["n"], v["mean"], v["median"], v["frac_above_0.5"], v["frac_above_0.9"]))
    for sp, v in control.items():
        print("  READ  {:6s} n={:4d} mean {:.3f} median {:.3f} | >0.5: {:.0%} | >0.9: {:.0%}   (control)".format(
            sp, v["n"], v["mean"], v["median"], v["frac_above_0.5"], v["frac_above_0.9"]))
    r = resplit
    print("\ncan a text-disjoint re-split repair it? clusters={} largest={} singletons={}".format(
        r["n_clusters"], r["largest_cluster"], r["singletons"]))
    la = r["leakage_after_resplit"]
    print("  cluster-disjoint {}/{} split -> test coverage mean {:.3f} median {:.3f} | >0.5: {:.0%}".format(
        r["train_paragraphs"], r["test_paragraphs"], la["mean"], la["median"], la["frac_above_0.5"]))

    print("\nwrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
