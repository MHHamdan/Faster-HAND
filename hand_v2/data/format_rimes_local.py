"""RIMES 2011 formatter for the local, already-extracted layout.

`hand.Datasets.dataset_formatters.rimes_formatter.RIMESDatasetFormatter` expects the two
tar archives under ./Datasets/raw/RIMES and extracts them into `<target>/temp/images_gray/`.
On this machine the images already sit at

    data/RIMES/training_2011/images/train2011-N.png   (1,500)
    data/RIMES/eval_2011/images/eval2011-N.png        (100)
    data/RIMES/training_2011.xml, data/RIMES/eval_2011_annotated.xml

so this wrapper builds the temp layout with SYMLINKS (no copies), points
`source_fold_path` at data/RIMES, and runs the unchanged `format_rimes_line` /
`preformat_rimes_paragraph`. Deviations from the original file, all confined here:

  * `format_rimes_paragraph` does not exist in the hand copy of the formatter (it only
    carries `line` and `page`); it is re-implemented here after VAN's original
    (whole paragraph image resized 300 -> 150 dpi, line boxes scaled, {"text", "lines"}).
  * the local PNGs are 1-bit (mode "1"). `resize` is overridden to promote bool arrays /
    mode-"1" images to 8-bit grayscale first; otherwise PIL falls back to NEAREST for
    mode "1" and the saved lines are binary, nearest-downsampled images.
  * a `provenance.json` sidecar (formatted name -> source paragraph image, split) is
    written for tools/audit_dataset_integrity.py; RIMES 2011 carries no writer identity.

Usage (from the repo root):
    python hand_v2/data/format_rimes_local.py --level line
    python hand_v2/data/format_rimes_local.py --level paragraph
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hand.Datasets.dataset_formatters.rimes_formatter import RIMESDatasetFormatter  # noqa: E402

IMAGE_DIRS = [os.path.join("training_2011", "images"), os.path.join("eval_2011", "images")]


class LocalRIMESDatasetFormatter(RIMESDatasetFormatter):

    def __init__(self, level, raw_dir, output_path, dpi=150):
        assert level in ("line", "paragraph"), level
        super().__init__("line", dpi=dpi)  # the parent only registers "line" and "page"
        self.level = level
        self.source_fold_path = raw_dir
        self.target_fold_path = output_path
        self.map_datasets_files["RIMES"]["paragraph"] = {
            "arx_files": [],
            "needed_files": ["eval_2011_annotated.xml", "training_2011.xml"],
            "format_function": self.format_rimes_paragraph,
        }
        self.map_datasets_files["RIMES"]["line"]["arx_files"] = []  # no archives: symlinked images
        self._dataset = None

    # ---------------------------------------------------------------- temp layout
    def init_format(self):
        os.makedirs(self.target_fold_path, exist_ok=True)
        img_fold = os.path.join(self.temp_fold, "images_gray")
        os.makedirs(img_fold, exist_ok=True)
        for fn in self.map_datasets_files["RIMES"][self.level]["needed_files"]:
            assert os.path.exists(os.path.join(self.source_fold_path, fn)), fn
        n = 0
        for sub in IMAGE_DIRS:
            d = os.path.join(self.source_fold_path, sub)
            for name in sorted(os.listdir(d)):
                if not name.endswith(".png"):
                    continue
                dst = os.path.join(img_fold, name)
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(os.path.join(d, name)), dst)
                n += 1
        print("symlinked {} images into {}".format(n, img_fold))
        for set_name in self.set_names:
            d = os.path.join(self.target_fold_path, set_name)
            os.makedirs(d, exist_ok=True)
            for fn in os.listdir(d):  # fresh build: the parent names files by len(os.listdir)
                os.remove(os.path.join(d, fn))

    # ---------------------------------------------------------------- 1-bit images
    def resize(self, img, source_dpi, target_dpi):
        if isinstance(img, np.ndarray):
            if img.dtype == bool:
                img = img.astype(np.uint8) * 255
        elif img.mode != "L":
            img = img.convert("L")
        return super().resize(img, source_dpi, target_dpi)

    def preformat_rimes_paragraph(self):
        self._dataset = super().preformat_rimes_paragraph()
        return self._dataset

    # ---------------------------------------------------------------- paragraph level
    def format_rimes_paragraph(self):
        """After VAN's original: one sample per paragraph image, line boxes scaled to dpi."""
        dataset = self.preformat_rimes_paragraph()
        for set_name in self.set_names:
            fold = os.path.join(self.target_fold_path, set_name)
            for sample in dataset[set_name]:
                new_name = "{}_{}.png".format(set_name, len(os.listdir(fold)))
                new_path = os.path.join(fold, new_name)
                self.load_resize_save(sample["img_path"], new_path, 300, self.dpi)
                paragraph = {
                    "text": sample["text"],
                    "lines": [self.adjust_coord_ratio(dict(line), self.dpi / 300) for line in sample["lines"]],
                    "nb_cols": 1,
                }
                self.charset = self.charset.union(set(paragraph["text"]))
                self.gt[set_name][new_name] = paragraph

    # ---------------------------------------------------------------- outputs
    def provenance(self):
        """Reconstruct formatted-name -> source image from the parent's sequential naming."""
        prov = {}
        for set_name in self.set_names:
            i = 0
            for sample in self._dataset[set_name]:
                doc = os.path.basename(sample["img_path"])
                units = sample["lines"] if self.level == "line" else [sample]
                for _ in units:
                    prov["{}_{}.png".format(set_name, i)] = {"document": doc, "split": set_name}
                    i += 1
        return prov

    def end_format(self):
        super().end_format()  # removes temp (symlinks only) and writes labels.pkl
        prov = self.provenance()
        assert set(prov) == {k for s in self.set_names for k in self.gt[s]}, "provenance/name mismatch"
        with open(os.path.join(self.target_fold_path, "provenance.json"), "w") as f:
            json.dump(prov, f, indent=0)
        stats = {"level": self.level, "charset_size": len(self.charset),
                 "n_samples": {s: len(self.gt[s]) for s in self.set_names},
                 "n_paragraphs": {s: len(self._dataset[s]) for s in self.set_names}}
        with open(os.path.join(self.target_fold_path, "format_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(stats)


def main():
    ap = argparse.ArgumentParser(description="Format RIMES 2011 from the local extracted layout")
    ap.add_argument("--level", choices=["line", "paragraph"], required=True)
    ap.add_argument("--raw_path", default=os.path.join(REPO_ROOT, "data", "RIMES"))
    ap.add_argument("--output_path", default=None, help="default formatted/RIMES_<level>")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    out = a.output_path or os.path.join(REPO_ROOT, "formatted", "RIMES_{}".format(a.level))
    LocalRIMESDatasetFormatter(a.level, a.raw_path, out, dpi=a.dpi).format()


if __name__ == "__main__":
    main()
