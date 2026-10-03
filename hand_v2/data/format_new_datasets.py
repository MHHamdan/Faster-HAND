#!/usr/bin/env python3
"""
Format the three historical corpora added for HAND V2, from the raw trees linked under
data/, into formatted/<Name>_<level>/ (git-ignored):

    Bentham_line, Bentham_page        data/Bentham      (Bentham Dataset R0)
    SaintGall_line, SaintGall_page    data/SaintGall    (IAM-HistDB)
    Washington_line                   data/Washington   (IAM-HistDB, fold cv1 only;
                                                         no page images locally)

Nothing is downloaded and the raw trees are read in place. Prints the split counts,
checks them against the published partition sizes, and writes a JSON summary
(counts, charset, skipped samples, decoding token tallies, wall-clock) for the report.

    python hand_v2/data/format_new_datasets.py [--only Bentham_line ...] [--summary out.json]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from hand.Datasets.dataset_formatters.bentham_formatter import BenthamDatasetFormatter  # noqa: E402
from hand.Datasets.dataset_formatters.iam_histdb_formatter import IAMHistDBDatasetFormatter  # noqa: E402

DATA = os.path.join(ROOT, "data")
FORMATTED = os.path.join(ROOT, "formatted")

# Published partition sizes (README / partition files), used only to flag deviations.
EXPECTED = {
    "Bentham_line": {"train": 9198, "valid": 1415, "test": 860},
    "Bentham_page": {"train": 350, "valid": 50, "test": 33},
    "SaintGall_line": {"train": 468, "valid": 235, "test": 707},
    "SaintGall_page": {"train": 20, "valid": 10, "test": 30},
    "Washington_line": {"train": 325, "valid": 168, "test": 163},
}


def build(name, dpi, output_root):
    if name == "Bentham_line":
        return BenthamDatasetFormatter("line", dpi=dpi, raw_data_path=os.path.join(DATA, "Bentham"),
                                       output_root=output_root)
    if name == "Bentham_page":
        return BenthamDatasetFormatter("page", dpi=dpi, raw_data_path=os.path.join(DATA, "Bentham"),
                                       output_root=output_root)
    if name == "SaintGall_line":
        return IAMHistDBDatasetFormatter("SaintGall", "line", dpi=dpi,
                                         raw_data_path=os.path.join(DATA, "SaintGall"), output_root=output_root)
    if name == "SaintGall_page":
        return IAMHistDBDatasetFormatter("SaintGall", "page", dpi=dpi,
                                         raw_data_path=os.path.join(DATA, "SaintGall"), output_root=output_root)
    if name == "Washington_line":
        return IAMHistDBDatasetFormatter("Washington", "line", dpi=dpi, fold="cv1",
                                         raw_data_path=os.path.join(DATA, "Washington"), output_root=output_root)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(EXPECTED), default=sorted(EXPECTED))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--output_root", default=FORMATTED)
    parser.add_argument("--summary", default=None, help="write a JSON summary here")
    args = parser.parse_args()

    summary = {}
    for name in args.only:
        print("=== {} ===".format(name), flush=True)
        t0 = time.time()
        fmt = build(name, args.dpi, args.output_root)
        fmt.format()
        counts = {s: len(fmt.gt[s]) for s in fmt.set_names}
        deviations = {s: (counts[s], EXPECTED[name][s]) for s in counts if counts[s] != EXPECTED[name][s]}
        summary[name] = {
            "output_dir": fmt.target_fold_path,
            "counts": counts,
            "expected": EXPECTED[name],
            "deviations": deviations,
            "charset_size": len(fmt.charset),
            "charset": "".join(sorted(fmt.charset)),
            "unusual_codepoints": fmt.unusual_codepoints(),
            "skipped": fmt.skipped,
            "notes": fmt.notes,
            "token_counts": getattr(fmt, "token_counts", {}),
            "elapsed_s": round(time.time() - t0, 1),
        }
        print("{}: {} | charset {} | skipped {} | {:.1f}s{}".format(
            name, " ".join("{} {}".format(s, counts[s]) for s in counts), len(fmt.charset), len(fmt.skipped),
            summary[name]["elapsed_s"],
            "" if not deviations else " | DEVIATES FROM EXPECTED: {}".format(deviations)), flush=True)

    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
        print("summary written to", args.summary)


if __name__ == "__main__":
    main()
