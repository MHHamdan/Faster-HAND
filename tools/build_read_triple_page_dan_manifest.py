#!/usr/bin/env python3
"""Manifest for formatted/READ_2016_triple_page_sem_dan (added 2026-10, multi-page scaling study).

The dataset itself is produced by
    python3 hand/Datasets/format_read_dan_splits.py --levels triple_page
which also writes formatted/READ_2016_triple_page_sem_dan/provenance.json (sample -> the three
source scans). This script turns that into the same manifest schema the other READ manifests use
(experiments/benchmark_suite/manifests/*.json: per sample name, source_scans, source_sha256,
image_sha256, text_sha256; no transcription is embedded) and cross-checks the source triples
against the V1 manifest READ_2016_triple_page_sem.json, whose triples it must reproduce exactly.
"""
import hashlib
import json
import os
import pickle
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = "READ_2016_triple_page_sem_dan"
FORMATTED = os.path.join(ROOT, "formatted", DS)
RAW = {"train": "PublicData/Training/Images", "valid": "PublicData/Validation/Images",
       "test": "Test-ICFHR-2016"}
OUT = os.path.join(ROOT, "experiments", "benchmark_suite", "manifests", DS + ".json")
V1 = os.path.join(ROOT, "experiments", "benchmark_suite", "manifests", "READ_2016_triple_page_sem.json")


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def strip_layout(s):
    return "".join(c for c in s if not ("ⓐ" <= c <= "ⓩ" or "Ⓐ" <= c <= "Ⓩ"))


def main():
    labels = pickle.load(open(os.path.join(FORMATTED, "labels.pkl"), "rb"))
    prov = json.load(open(os.path.join(FORMATTED, "provenance.json")))
    v1 = json.load(open(V1))["splits"]
    m = {"dataset": DS,
         "description": ("READ 2016 triple pages, five-token scheme (ⓟ ⓝ ⓢ ⓐ ⓑ + end tokens), 150 dpi: three "
                         "consecutive scans of the same split in scan order concatenated horizontally, "
                         "non-overlapping, remainder dropped (NOT a DAN construction; same source triples as "
                         "the V1 READ_2016_triple_page_sem manifest). Shorter scans are padded at the bottom "
                         "with their median colour before concatenation. 116/16/16. Built by "
                         "hand/Datasets/format_read_dan_splits.py --levels triple_page "
                         "(read2016_formatter.format_read2016_triple_page)."),
         "charset_size": len(labels["charset"]), "splits": {}}
    for s in ("train", "valid", "test"):
        gt = labels["ground_truth"][s]
        names = sorted(gt, key=lambda n: int(re.findall(r"\d+", n)[0]))
        v1_triples = [tuple(x["source_scans"]) for x in v1[s]["samples"]]
        samples = []
        for name in names:
            scans = prov[s][name]
            text = gt[name]["text"]
            samples.append({
                "name": name, "source_scans": scans,
                "source_sha256": [sha256_file(os.path.join(ROOT, "raw_READ2016", RAW[s], sc)) for sc in scans],
                "image_sha256": sha256_file(os.path.join(FORMATTED, s, name)),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "n_chars": len(strip_layout(text)), "n_lines": text.count("\n") + 1,
                "pad_bottom_px_300dpi": [pg["pad_bottom_px"] for pg in gt[name]["pages"]]})
        got = [tuple(x["source_scans"]) for x in samples]
        assert got == v1_triples, (s, "source triples differ from the V1 manifest")
        m["splits"][s] = {"n": len(samples), "samples": samples,
                          "v1_source_triples_identical": True}
    json.dump(m, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT, {s: m["splits"][s]["n"] for s in m["splits"]})
    print("V1 source triples identical in every split: True")
    for s in ("test",):
        for name in ("test_0.jpeg", "test_7.jpeg"):
            print("\n=== %s/%s  sources=%s" % (s, name, prov[s][name]))
            print(repr(labels["ground_truth"][s][name]["text"]))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
