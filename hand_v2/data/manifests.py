"""
Evaluation manifests with checksums.

A manifest pins exactly which documents an evaluation split contains, which raw scans each
formatted sample was derived from, and the SHA-256 of every file involved, so that a number
reported against a split can be checked against the data that produced it. Manifests carry
**no transcriptions** (they embed only the SHA-256 of each transcription), so they can be
committed even for research-licensed corpora.

Written to experiments/benchmark_suite/manifests/<name>.json plus CHECKSUMS.sha256 over all
manifests. Re-run after any formatter change; the diff shows what moved.

Manifests produced here:
  READ_2016_page_sem            350/50/50 pages, scan-level provenance (V1 three-token scheme)
  READ_2016_page_sem_dan        same pages, DAN five-token scheme (Stage 1 training data)
  READ_2016_double_page_sem_dan 169/24/24 DAN pairing (same written page number)
  READ_2016_paragraph_dan       1,602/182/199 one sample per TextRegion
  READ_2016_triple_page_sem     116/16/16 consecutive-scan triples (HAND V1 construction)
  READ_2016_page_sequences      ordered scan sequences per split with 2-, 3-, 5-page windows
  <any formatted dataset>       generic: names + formatted-image and transcription hashes
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "experiments", "benchmark_suite", "manifests")
RAW_READ = os.path.join(ROOT, "raw_READ2016")
FORMATTED = os.path.join(ROOT, "formatted")


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_labels(dataset):
    with open(os.path.join(FORMATTED, dataset, "labels.pkl"), "rb") as f:
        return pickle.load(f)


def strip_layout_tokens(s):
    return "".join(c for c in s if not ("ⓐ" <= c <= "ⓩ" or "Ⓐ" <= c <= "Ⓩ"))


def read_raw_pages():
    """Ordered raw scans per split, in the order the formatter enumerates them (sorted XML
    names), together with the page-level text so formatted samples can be matched by content."""
    from hand.Datasets.dataset_formatters.read2016_formatter import READ2016DatasetFormatter
    from hand_v2.data.format_read_dan_splits import make_temp_tree
    temp = os.path.join(FORMATTED, "_read_raw_view")
    make_temp_tree(temp)
    f = READ2016DatasetFormatter("page", sem_token=True, end_token=True)
    f.temp_fold = temp
    ds = f.preformat_read2016()
    out = {}
    for s in ("train", "valid", "test"):
        pages = []
        for p in ds[s]:
            scan = os.path.basename(os.path.realpath(p["img_path"]))
            pages.append({"scan": scan, "scan_path": os.path.realpath(p["img_path"]),
                          "n_regions": len(p["text_regions"]),
                          "n_lines": sum(len(r["lines"]) for r in p["text_regions"]),
                          "lines_hash": sha256_text("\n".join(l["label"] for r in p["text_regions"] for l in r["lines"]))})
        out[s] = pages
    return out, ds, f


def manifest_generic(dataset, description):
    labels = load_labels(dataset)
    m = {"dataset": dataset, "description": description, "charset_size": len(labels["charset"]),
         "splits": {}}
    for s, gt in labels["ground_truth"].items():
        samples = []
        for name in sorted(gt):
            img = os.path.join(FORMATTED, dataset, s, name)
            text = gt[name]["text"] if isinstance(gt[name], dict) else str(gt[name])
            samples.append({"name": name, "image_sha256": sha256_file(img) if os.path.exists(img) else None,
                            "text_sha256": sha256_text(text), "n_chars": len(strip_layout_tokens(text))})
        m["splits"][s] = {"n": len(samples), "samples": samples}
    return m


def manifest_read_page(raw):
    labels = load_labels("READ_2016_page_sem")
    m = {"dataset": "READ_2016_page_sem", "description": "READ 2016 single pages, 150 dpi, DAN layout tokens; "
         "official ICFHR 2016 split 350/50/50", "splits": {}}
    for s in ("train", "valid", "test"):
        gt = labels["ground_truth"][s]
        by_scan = {}
        for name in gt:
            scan = re.sub(r"\.(jpe?g|png)$", "", name, flags=re.I) + ".JPG"
            by_scan[scan] = name
        samples = []
        for p in raw[s]:
            name = by_scan.get(p["scan"])
            assert name is not None, ("formatted sample missing for", p["scan"])
            text = gt[name]["text"]
            samples.append({"name": name, "source_scan": p["scan"], "source_sha256": sha256_file(p["scan_path"]),
                            "image_sha256": sha256_file(os.path.join(FORMATTED, "READ_2016_page_sem", s, name)),
                            "text_sha256": sha256_text(text), "n_regions": p["n_regions"], "n_lines": p["n_lines"]})
        assert len(samples) == len(gt), (s, len(samples), len(gt))
        m["splits"][s] = {"n": len(samples), "samples": samples}
    return m


def manifest_read_dan_split(dataset, raw, ds, formatter, level):
    """DAN pairing / paragraph enumeration reproduced with the formatter's own functions, then
    matched to the formatted sample order (the formatter enumerates in the same order)."""
    labels = load_labels(dataset)
    m = {"dataset": dataset, "splits": {}}
    if level == "double_page":
        m["description"] = ("READ 2016 double pages built like DAN (arXiv 2203.12273 §5.1.2): two scans that carry "
                            "the same written page number are concatenated; unpaired scans dropped; 169/24/24")
        for s in ("train", "valid", "test"):
            pages = ds[s]
            for p in pages:
                if "mode" not in p["text_regions"][0]:
                    p["label"], p["text_regions"], p["nb_cols"], p["side"] = formatter.sort_text_regions(p["text_regions"], p["width"])
            grouped = formatter.group_by_page_number({"train": [], "valid": [], "test": [], s: pages})[s]
            pairs = [d for d in grouped if len(d["pages"]) == 2]
            names = sorted(labels["ground_truth"][s], key=lambda n: int(re.findall(r"\d+", n)[0]))
            assert len(pairs) == len(names), (s, len(pairs), len(names))
            samples = []
            for name, d in zip(names, pairs):
                scans = [os.path.basename(os.path.realpath(pg["img_path"])) for pg in d["pages"]]
                samples.append({"name": name, "written_page_number": d["page_num"], "source_scans": scans,
                                "source_sha256": [sha256_file(os.path.realpath(pg["img_path"])) for pg in d["pages"]],
                                "image_sha256": sha256_file(os.path.join(FORMATTED, dataset, s, name)),
                                "text_sha256": sha256_text(labels["ground_truth"][s][name]["text"])})
            m["splits"][s] = {"n": len(samples), "samples": samples,
                              "unpaired_scans": [os.path.basename(os.path.realpath(pg["img_path"]))
                                                 for d in grouped if len(d["pages"]) != 2 for pg in d["pages"]]}
    elif level == "page":
        m["description"] = ("READ 2016 single pages rebuilt with DAN's five-token layout scheme (ⓟ ⓝ ⓢ ⓐ ⓑ + end "
                            "tokens); the V1 set READ_2016_page_sem carries only ⓟ ⓐ ⓑ. Same 350/50/50 scans.")
        for s in ("train", "valid", "test"):
            names = sorted(labels["ground_truth"][s], key=lambda n: int(re.findall(r"\d+", n)[0]))
            assert len(names) == len(raw[s]), (s, len(names), len(raw[s]))
            samples = []
            for name, p in zip(names, raw[s]):
                samples.append({"name": name, "source_scan": p["scan"], "source_sha256": sha256_file(p["scan_path"]),
                                "image_sha256": sha256_file(os.path.join(FORMATTED, dataset, s, name)),
                                "text_sha256": sha256_text(labels["ground_truth"][s][name]["text"]),
                                "n_regions": p["n_regions"], "n_lines": p["n_lines"]})
            m["splits"][s] = {"n": len(samples), "samples": samples}
    elif level == "paragraph":
        m["description"] = "READ 2016 paragraphs = one sample per PAGE-XML TextRegion (DAN), 1,602/182/199"
        for s in ("train", "valid", "test"):
            names = sorted(labels["ground_truth"][s], key=lambda n: int(re.findall(r"\d+", n)[0]))
            regions = [(os.path.basename(os.path.realpath(p["img_path"])), ri, r)
                       for p in ds[s] for ri, r in enumerate(p["text_regions"])]
            assert len(regions) == len(names), (s, len(regions), len(names))
            samples = []
            for name, (scan, ri, r) in zip(names, regions):
                samples.append({"name": name, "source_scan": scan, "region_index": ri,
                                "region_bbox_300dpi": {k: int(v) for k, v in r["coords"].items()}, "n_lines": len(r["lines"]),
                                "image_sha256": sha256_file(os.path.join(FORMATTED, dataset, s, name)),
                                "text_sha256": sha256_text(labels["ground_truth"][s][name]["text"])})
            m["splits"][s] = {"n": len(samples), "samples": samples}
    return m


def manifest_read_triple(raw):
    labels = load_labels("READ_2016_triple_page_sem")
    scan_hash = {p["scan"]: sha256_file(p["scan_path"]) for s in raw for p in raw[s]}
    m = {"dataset": "READ_2016_triple_page_sem", "description": "HAND V1 triple pages: three consecutive scans of the "
         "same split concatenated horizontally (not a DAN construction; no external baseline)", "splits": {}}
    for s, gt in labels["ground_truth"].items():
        samples = []
        for name in sorted(gt):
            scans = [x + ".JPG" for x in re.sub(r"\.(jpe?g|png)$", "", name, flags=re.I).split("_")]
            samples.append({"name": name, "source_scans": scans, "source_sha256": [scan_hash.get(x) for x in scans],
                            "image_sha256": sha256_file(os.path.join(FORMATTED, "READ_2016_triple_page_sem", s, name)),
                            "text_sha256": sha256_text(gt[name]["text"])})
        m["splits"][s] = {"n": len(samples), "samples": samples}
    return m


def manifest_read_sequences(raw, windows=(2, 3, 5)):
    """Ordered scan sequences per split (archival scan order = Seite number), for the multi-page
    benchmark: every window of k consecutive scans within a split."""
    m = {"dataset": "READ_2016_page_sequences", "description": "Consecutive-scan windows over each split's "
         "pages in scan order (Seite number); pages come from READ_2016_page_sem. Gaps in the scan "
         "numbering (e.g. missing Seite0006) break a sequence.", "windows": list(windows), "splits": {}}
    for s in ("train", "valid", "test"):
        order = sorted(raw[s], key=lambda p: int(re.findall(r"\d+", p["scan"])[0]))
        nums = [int(re.findall(r"\d+", p["scan"])[0]) for p in order]
        runs, cur = [], [order[0]]
        for prev, p in zip(order, order[1:]):
            if int(re.findall(r"\d+", p["scan"])[0]) == int(re.findall(r"\d+", prev["scan"])[0]) + 1:
                cur.append(p)
            else:
                runs.append(cur)
                cur = [p]
        runs.append(cur)
        wins = {}
        for k in windows:
            wins[str(k)] = [[p["scan"] for p in run[i:i + k]] for run in runs for i in range(len(run) - k + 1)]
        m["splits"][s] = {"n_pages": len(order), "scan_numbers": nums, "n_runs": len(runs),
                          "run_lengths": [len(r) for r in runs], "windows": wins,
                          "n_windows": {k: len(v) for k, v in wins.items()}}
    return m


def manifest_bentham_sequences(windows=(2, 3, 5)):
    """Bentham R0 page ids are <box>_<folder>_<page>; consecutive pages of the same box/folder
    within a split form a run, windowed like the READ sequences."""
    prov = json.load(open(os.path.join(FORMATTED, "Bentham_page", "provenance.json")))
    m = {"dataset": "Bentham_page_sequences", "description": "Consecutive Bentham R0 pages (same box and folder, "
         "page number +1) within each partition, windowed for the multi-page benchmark", "windows": list(windows), "splits": {}}
    by_split = {}
    for name, info in prov.items():
        box, folder, page = info["document"].split("_")
        by_split.setdefault(info["split"], []).append((box, folder, int(page), name, info["document"]))
    for s, items in by_split.items():
        items.sort()
        runs, cur = [], [items[0]]
        for prev, it in zip(items, items[1:]):
            if it[0] == prev[0] and it[1] == prev[1] and it[2] == prev[2] + 1:
                cur.append(it)
            else:
                runs.append(cur)
                cur = [it]
        runs.append(cur)
        wins = {str(k): [[it[4] for it in run[i:i + k]] for run in runs for i in range(len(run) - k + 1)] for k in windows}
        m["splits"][s] = {"n_pages": len(items), "n_runs": len(runs), "run_lengths": sorted([len(r) for r in runs], reverse=True),
                          "name_by_document": {it[4]: it[3] for it in items}, "windows": wins,
                          "n_windows": {k: len(v) for k, v in wins.items()}}
    return m


def write(m, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".json")
    with open(path, "w") as f:
        json.dump(m, f, indent=1, ensure_ascii=False, sort_keys=True)
    counts = {s: (v.get("n") if isinstance(v, dict) else None) for s, v in m.get("splits", {}).items()}
    print("wrote", os.path.relpath(path, ROOT), counts)


def write_checksums():
    lines = []
    for fn in sorted(os.listdir(OUT)):
        if fn.endswith(".json"):
            lines.append("{}  {}".format(sha256_file(os.path.join(OUT, fn)), fn))
    with open(os.path.join(OUT, "CHECKSUMS.sha256"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote CHECKSUMS.sha256 ({} manifests)".format(len(lines)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", action="store_true", help="READ 2016 manifests (page, DAN double, DAN paragraph, triple, sequences)")
    ap.add_argument("--generic", nargs="*", default=[], help="formatted dataset names for generic manifests")
    ap.add_argument("--bentham-sequences", action="store_true")
    a = ap.parse_args()
    if a.read:
        raw, ds, f = read_raw_pages()
        write(manifest_read_page(raw), "READ_2016_page_sem")
        write(manifest_read_dan_split("READ_2016_double_page_sem_dan", raw, ds, f, "double_page"), "READ_2016_double_page_sem_dan")
        if os.path.exists(os.path.join(FORMATTED, "READ_2016_page_sem_dan", "labels.pkl")):
            write(manifest_read_dan_split("READ_2016_page_sem_dan", raw, ds, f, "page"), "READ_2016_page_sem_dan")
        write(manifest_read_dan_split("READ_2016_paragraph_dan", raw, ds, f, "paragraph"), "READ_2016_paragraph_dan")
        write(manifest_read_triple(raw), "READ_2016_triple_page_sem")
        write(manifest_read_sequences(raw), "READ_2016_page_sequences")
    if a.bentham_sequences:
        write(manifest_bentham_sequences(), "Bentham_page_sequences")
    for d in a.generic:
        write(manifest_generic(d, "generic manifest: formatted sample names with image and transcription hashes"), d)
    write_checksums()


if __name__ == "__main__":
    main()
