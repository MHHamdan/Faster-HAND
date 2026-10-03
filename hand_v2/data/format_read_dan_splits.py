#
#  Modified by Mohammed Hamdan, 2025-2026, under CeCILL-C Article 5.3.2. The measured
#  divergence from the pinned upstream commit is recorded in release/NOTICE.md
#  section 1.1. This file remains governed by CeCILL-C; the repository's top-level
#  MIT LICENSE does not apply to it.
"""
Re-derive the READ 2016 double-page and paragraph datasets with DAN's exact construction.

Why: the repository's `READ_2016_double_page_sem` (175/25/25) pairs consecutive *scans*
(Seite0405+Seite0406, ...), whereas DAN (arXiv 2203.12273, `group_by_page_number`) pairs
scans that carry the same *written* page number and drops the unpaired ones, giving
169/24/24. `READ_2016_paragraph` (837/79/95) was built by a different tool that kept only
2-10-line regions; DAN's paragraph level is one sample per TextRegion, 1,602/182/199.
Both DAN splits are reproduced here with the unchanged formatter code
(`hand/Datasets/dataset_formatters/read2016_formatter.py`, a CeCILL-C derivative of DAN's),
fed from the raw tree through symlinks: nothing is copied or downloaded.

Outputs (git-ignored): formatted/READ_2016_page_sem_dan (five-token scheme), formatted/READ_2016_double_page_sem_dan, formatted/READ_2016_paragraph_dan
and, for the record, a JSON listing every pair.
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from hand.Datasets.dataset_formatters.read2016_formatter import READ2016DatasetFormatter  # noqa: E402

RAW = os.path.join(ROOT, "raw_READ2016")
SPLIT_SRC = {"train": "PublicData/Training", "valid": "PublicData/Validation", "test": "Test-ICFHR-2016"}


def make_temp_tree(temp_fold):
    """train/valid/test dirs holding symlinks to the raw scans and the PAGE-XML folder."""
    for set_name, rel in SPLIT_SRC.items():
        src = os.path.join(RAW, rel)
        dst = os.path.join(temp_fold, set_name)
        os.makedirs(dst, exist_ok=True)
        page_link = os.path.join(dst, "page")
        if not os.path.islink(page_link):
            os.symlink(os.path.join(src, "page"), page_link)
        img_dir = os.path.join(src, "Images") if os.path.isdir(os.path.join(src, "Images")) else src
        for f in os.listdir(img_dir):
            if f.upper().endswith(".JPG"):
                link = os.path.join(dst, f)
                if not os.path.islink(link):
                    os.symlink(os.path.join(img_dir, f), link)


class _LocalREAD(READ2016DatasetFormatter):
    """Same formatter, but the temp tree is a symlink view of the raw data and is never deleted."""

    def __init__(self, level, temp_fold, target_name, **kw):
        super().__init__(level, **kw)
        self.temp_fold = temp_fold
        self.target_fold_path = os.path.join(ROOT, "formatted", target_name)

    def format(self):
        os.makedirs(self.target_fold_path, exist_ok=True)
        for set_name in self.set_names:
            os.makedirs(os.path.join(self.target_fold_path, set_name), exist_ok=True)
        self.map_datasets_files[self.dataset_name][self.level]["format_function"]()
        # end_format() of the base class writes labels.pkl and then rmtree(temp_fold);
        # write labels.pkl ourselves and keep the symlink view.
        import pickle
        with open(os.path.join(self.target_fold_path, "labels.pkl"), "wb") as f:
            pickle.dump({"ground_truth": self.gt, "charset": sorted(list(self.charset))}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", default=os.path.join(ROOT, "formatted", "_read_raw_view"))
    ap.add_argument("--levels", nargs="+", default=["double_page", "paragraph", "page"],
                    help="any of double_page, page, paragraph, triple_page (the last is opt-in)")
    a = ap.parse_args()
    make_temp_tree(a.temp)
    if "double_page" in a.levels:
        f = _LocalREAD("double_page", a.temp, "READ_2016_double_page_sem_dan", sem_token=True, end_token=True)
        f.format()
        counts = {s: len(f.gt[s]) for s in f.gt}
        print("double_page (DAN pairing):", counts)
        assert counts == {"train": 169, "valid": 24, "test": 24}, counts
    if "page" in a.levels:
        # Same construction as READ_2016_page_sem but with DAN's five-token scheme
        # (ⓟ ⓝ ⓢ ⓐ ⓑ + end tokens); the existing READ_2016_page_sem (V1) carries only ⓟ ⓐ ⓑ.
        f = _LocalREAD("page", a.temp, "READ_2016_page_sem_dan", sem_token=True, end_token=True)
        f.format()
        counts = {s: len(f.gt[s]) for s in f.gt}
        print("page (DAN token scheme):", counts)
        assert counts == {"train": 350, "valid": 50, "test": 50}, counts
    if "triple_page" in a.levels:
        # Added 2026-10 (multi-page scaling study). NOT a DAN construction: three consecutive scans
        # of the same split in scan order, non-overlapping, remainder dropped -- the V1
        # READ_2016_triple_page_sem source triples -- rebuilt with the five-token scheme at 150 dpi.
        # Not in the default --levels list, so the three original outputs are unaffected.
        f = _LocalREAD("triple_page", a.temp, "READ_2016_triple_page_sem_dan", sem_token=True, end_token=True)
        f.format()
        counts = {s: len(f.gt[s]) for s in f.gt}
        print("triple_page (consecutive scans, five-token scheme):", counts)
        assert counts == {"train": 116, "valid": 16, "test": 16}, counts
        with open(os.path.join(f.target_fold_path, "provenance.json"), "w") as fh:
            json.dump({s: {k: [os.path.basename(os.path.realpath(p)) for p in v] for k, v in f.provenance[s].items()}
                       for s in f.provenance}, fh, indent=1)
    if "paragraph" in a.levels:
        f = _LocalREAD("paragraph", a.temp, "READ_2016_paragraph_dan")
        f.format()
        counts = {s: len(f.gt[s]) for s in f.gt}
        print("paragraph (one sample per TextRegion):", counts)
        assert counts == {"train": 1602, "valid": 182, "test": 199}, counts


if __name__ == "__main__":
    main()
