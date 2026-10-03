#  IAM-HistDB formatter for the HAND framework: Saint Gall and George Washington.
#
#  Both corpora come from the IAM Historical Document Database (Fischer et al.) and share
#  the same transcription conventions (ground_truth/transcription.txt):
#
#      <line_id> <word_1>|...|<word_n> [<edition label_1>|...|<label_n>]
#
#  where each word is spelled character by character, "-" separated. Decoding, per each
#  corpus' README.txt:
#
#  Saint Gall (Latin, 9th c.)   "pt" -> "."   (punctuation mark, typically a dot)
#                               "et" -> "&"   (the et ligature, also inside words)
#                               anything else is a single letter, kept as-is.
#                               The third column (edition labels, lower-case, no
#                               punctuation, "BREAK" markers) is NOT used.
#
#  Washington (English, 1755)   s_pt "." s_cm "," s_mi "-" s_qo ":" s_sq ";" s_qt "'"
#                               s_bl "(" s_br ")" s_et "&" s_lb "£" s_sl "/" s_GW "G.W."
#                               s_s  -> "ſ" (U+017F, long s; kept distinct from "s")
#                               s_0..s_9 -> the digit; s_<n>th/st/nd/rd -> "<n>th" etc.
#                               Any token not in the table raises: nothing is dropped
#                               silently.
#  Words are joined with a single space. Punctuation stays attached to its word exactly as
#  in the spelling (e.g. "haberetur-pt" -> "haberetur."). A trailing s_mi is the
#  end-of-line hyphenation mark and is kept as "-".
#
#  Line level : data/line_images_normalized/<line_id>.png (bilevel, height-normalised to
#               120 px, 300 dpi metadata) -> grey, resized to `dpi`.
#  Page level : (Saint Gall only) data/page_images/<page_id>.jpg (300 dpi) with the line
#               polygons of ground_truth/line_location/<page_id>.svg; one paragraph per
#               page, lines in file order (= line-number order = reading order).
#
#  Splits: Saint Gall sets/{train,valid,test}.txt are page ids; Washington
#  sets/cv<k>/{train,valid,test}.txt are line ids (cross-validation folds; cv1 is used).

import os
import re
import json
import pickle
import time
import unicodedata

import numpy as np
from PIL import Image

from hand.Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUT = os.path.join(REPO_ROOT, "formatted")
DEFAULT_RAW = {
    "SaintGall": os.path.join(REPO_ROOT, "data", "SaintGall"),
    "Washington": os.path.join(REPO_ROOT, "data", "Washington"),
}

SAINTGALL_TOKENS = {"pt": ".", "et": "&"}

WASHINGTON_TOKENS = {
    "s_pt": ".", "s_cm": ",", "s_mi": "-", "s_qo": ":", "s_sq": ";", "s_qt": "'",
    "s_bl": "(", "s_br": ")", "s_et": "&", "s_lb": "£", "s_sl": "/", "s_GW": "G.W.",
    "s_s": "ſ",
}
WASHINGTON_TOKENS.update({"s_%d" % d: str(d) for d in range(10)})
_WASHINGTON_ORDINAL = re.compile(r"^s_(\d)(st|nd|rd|th)$")


def decode_saintgall_word(word):
    out = []
    for tok in word.split("-"):
        if tok in SAINTGALL_TOKENS:
            out.append(SAINTGALL_TOKENS[tok])
        elif len(tok) == 1:
            out.append(tok)
        else:
            raise ValueError("unknown Saint Gall token {!r} in {!r}".format(tok, word))
    return "".join(out)


def decode_washington_word(word):
    out = []
    for tok in word.split("-"):
        if tok in WASHINGTON_TOKENS:
            out.append(WASHINGTON_TOKENS[tok])
        elif _WASHINGTON_ORDINAL.match(tok):
            m = _WASHINGTON_ORDINAL.match(tok)
            out.append(m.group(1) + m.group(2))
        elif len(tok) == 1:
            out.append(tok)
        else:
            raise ValueError("unknown Washington token {!r} in {!r}".format(tok, word))
    return "".join(out)


DECODERS = {"SaintGall": decode_saintgall_word, "Washington": decode_washington_word}


def decode_spelling(dataset_name, spelling):
    """'<word>|<word>|...' with '-'-separated character tokens -> plain text."""
    return " ".join(DECODERS[dataset_name](w) for w in spelling.split("|"))


def svg_path_bbox(d_attr):
    """Bounding box of an SVG path made of M/L absolute commands ('M x y L x y ... Z')."""
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d_attr)]
    xs, ys = nums[0::2], nums[1::2]
    return {"left": int(np.floor(min(xs))), "right": int(np.ceil(max(xs))),
            "top": int(np.floor(min(ys))), "bottom": int(np.ceil(max(ys)))}


class IAMHistDBDatasetFormatter(OCRDatasetFormatter):
    """
    Formatter for Saint Gall and Washington (IAM-HistDB).

    Args:
        dataset_name: "SaintGall" or "Washington"
        level: "line" (both) or "page" (Saint Gall only)
        raw_data_path: corpus root (holding data/, ground_truth/, sets/, README.txt)
        output_root: directory under which <output_root>/<dataset_name>_<level>/ is written
        fold: Washington cross-validation fold used for the splits (default "cv1")
        dpi / source_dpi: target / raw resolution
    """

    def __init__(self, dataset_name, level, set_names=("train", "valid", "test"), dpi=150, source_dpi=300,
                 raw_data_path=None, output_root=None, fold="cv1", jpeg_quality=95):
        if dataset_name not in DECODERS:
            raise ValueError(dataset_name)
        if level == "page" and dataset_name != "SaintGall":
            raise ValueError("page level is only available for SaintGall (no page images for Washington)")
        set_names = list(set_names)
        super().__init__(dataset_name, level, "", set_names)
        self.dpi = dpi
        self.source_dpi = source_dpi
        self.ratio = dpi / source_dpi
        self.fold = fold
        self.jpeg_quality = jpeg_quality
        self.source_fold_path = raw_data_path or DEFAULT_RAW[dataset_name]
        self.target_fold_path = os.path.join(output_root or DEFAULT_OUT, "{}_{}".format(dataset_name, level))
        self.map_datasets_files.update({
            dataset_name: {
                "line": {"arx_files": [], "needed_files": [], "format_function": self.format_line},
                "page": {"arx_files": [], "needed_files": [], "format_function": self.format_page},
            }
        })
        self.skipped = []
        self.notes = []
        self.provenance = dict()
        self.token_counts = dict()
        self.elapsed = None

    # ------------------------------------------------------------------ plumbing
    def init_format(self):
        if not os.path.isdir(self.source_fold_path):
            raise FileNotFoundError(self.source_fold_path)
        os.makedirs(self.target_fold_path, exist_ok=True)
        for set_name in self.set_names:
            os.makedirs(os.path.join(self.target_fold_path, set_name), exist_ok=True)

    def format(self):
        t0 = time.time()
        self.init_format()
        self.map_datasets_files[self.dataset_name][self.level]["format_function"]()
        self.end_format()
        self.elapsed = time.time() - t0

    def end_format(self):
        with open(os.path.join(self.target_fold_path, "labels.pkl"), "wb") as f:
            pickle.dump({"ground_truth": self.gt, "charset": sorted(list(self.charset))}, f)
        with open(os.path.join(self.target_fold_path, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(self.provenance, f, indent=1, ensure_ascii=False)
        print("{}_{}: charset {} | ".format(self.dataset_name, self.level, len(self.charset))
              + " ".join("{} {}".format(s, len(self.gt[s])) for s in self.set_names)
              + " | skipped {}".format(len(self.skipped)))

    def sets_dir(self):
        if self.dataset_name == "Washington":
            return os.path.join(self.source_fold_path, "sets", self.fold)
        return os.path.join(self.source_fold_path, "sets")

    def read_set(self, set_name):
        with open(os.path.join(self.sets_dir(), set_name + ".txt"), encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    @staticmethod
    def page_of(line_id):
        return line_id.rsplit("-", 1)[0]

    def load_transcriptions(self):
        """line_id -> decoded text, in file order. Also tallies the special tokens seen."""
        out = dict()
        path = os.path.join(self.source_fold_path, "ground_truth", "transcription.txt")
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.rstrip("\n")
                if not raw.strip():
                    continue
                parts = raw.split(" ")
                line_id, spelling = parts[0], parts[1]
                for tok in re.split(r"[|-]", spelling):
                    if len(tok) > 1:
                        self.token_counts[tok] = self.token_counts.get(tok, 0) + 1
                out[line_id] = decode_spelling(self.dataset_name, spelling)
        return out

    def unusual_codepoints(self):
        out = []
        for ch in sorted(self.charset):
            if ord(ch) < 32 or ord(ch) > 126:
                out.append((ch, "U+%04X" % ord(ch), unicodedata.name(ch, "?")))
        return out

    # ------------------------------------------------------------------ line level
    def line_ids_for_split(self, set_name, transcriptions):
        ids = self.read_set(set_name)
        if self.dataset_name == "SaintGall":
            # sets are page ids; expand to the lines of those pages, in transcription order
            pages = set(ids)
            return [lid for lid in transcriptions if self.page_of(lid) in pages]
        return ids

    def load_line_image(self, line_id):
        path = os.path.join(self.source_fold_path, "data", "line_images_normalized", line_id + ".png")
        if not os.path.isfile(path):
            return None
        # bilevel ("1") PNGs: convert to 8-bit grey first so the bilinear downscale antialiases
        return np.array(Image.open(path).convert("L"))

    def format_line(self):
        transcriptions = self.load_transcriptions()
        for set_name in self.set_names:
            for line_id in self.line_ids_for_split(set_name, transcriptions):
                text = transcriptions.get(line_id)
                if text is None:
                    self.skipped.append((set_name, line_id, "no transcription"))
                    continue
                if text == "":
                    self.skipped.append((set_name, line_id, "empty transcription"))
                    continue
                img = self.load_line_image(line_id)
                if img is None:
                    self.skipped.append((set_name, line_id, "missing line image"))
                    continue
                img = self.resize(img, self.source_dpi, self.dpi)
                new_img_name = "{}_{}.png".format(set_name, len(self.gt[set_name]))
                Image.fromarray(img, mode="L").save(os.path.join(self.target_fold_path, set_name, new_img_name))
                self.gt[set_name][new_img_name] = {"text": text}
                self.charset = self.charset.union(set(text))
                self.provenance[new_img_name] = {"document": self.page_of(line_id), "source_id": line_id,
                                                 "split": set_name}

    # ------------------------------------------------------------------ page level (Saint Gall)
    def load_svg_lines(self, page_id):
        """[(line_id, bbox)] in file order."""
        path = os.path.join(self.source_fold_path, "ground_truth", "line_location", page_id + ".svg")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            svg = f.read()
        out = []
        for m in re.finditer(r"<path\b[^>]*?\bid=\"([^\"]+)\"[^>]*?\bd=\"([^\"]+)\"", svg):
            out.append((m.group(1), svg_path_bbox(m.group(2))))
        return out

    def format_page(self):
        transcriptions = self.load_transcriptions()
        for set_name in self.set_names:
            for page_id in self.read_set(set_name):
                img_path = os.path.join(self.source_fold_path, "data", "page_images", page_id + ".jpg")
                if not os.path.isfile(img_path):
                    self.skipped.append((set_name, page_id, "missing page image"))
                    continue
                svg_lines = self.load_svg_lines(page_id)
                if not svg_lines:
                    self.skipped.append((set_name, page_id, "missing or empty line_location svg"))
                    continue
                lines = []
                for line_id, bbox in svg_lines:
                    text = transcriptions.get(line_id)
                    if text is None:
                        self.skipped.append(("page:" + page_id, line_id, "svg line without transcription"))
                        continue
                    if text == "":
                        self.skipped.append(("page:" + page_id, line_id, "empty transcription"))
                        continue
                    entry = {"text": text}
                    entry.update(bbox)
                    lines.append(self.adjust_coord_ratio(entry, self.ratio))
                for line_id in transcriptions:
                    if self.page_of(line_id) == page_id and line_id not in {l for l, _ in svg_lines}:
                        self.skipped.append(("page:" + page_id, line_id, "transcribed line without svg polygon"))
                if not lines:
                    self.skipped.append((set_name, page_id, "no usable line"))
                    continue

                img = Image.open(img_path)
                img = self.resize(img, self.source_dpi, self.dpi)
                new_img_name = "{}_{}.jpg".format(set_name, len(self.gt[set_name]))
                Image.fromarray(img).save(os.path.join(self.target_fold_path, set_name, new_img_name),
                                          quality=self.jpeg_quality)
                paragraph = {
                    "label": "\n".join(l["text"] for l in lines),
                    "lines": lines,
                    "mode": "body",
                    "top": min(l["top"] for l in lines),
                    "bottom": max(l["bottom"] for l in lines),
                    "left": min(l["left"] for l in lines),
                    "right": max(l["right"] for l in lines),
                }
                page_label = {
                    "text": paragraph["label"],
                    "paragraphs": [paragraph],
                    "nb_cols": 1,
                    "side": "left",
                    "top": paragraph["top"],
                    "bottom": paragraph["bottom"],
                    "left": paragraph["left"],
                    "right": paragraph["right"],
                    "page_width": int(img.shape[1]),
                }
                self.gt[set_name][new_img_name] = {"text": paragraph["label"], "nb_cols": 1, "pages": [page_label]}
                self.charset = self.charset.union(set(paragraph["label"]))
                self.provenance[new_img_name] = {"document": page_id, "source_id": page_id, "split": set_name}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format Saint Gall and Washington (IAM-HistDB) for HAND")
    parser.add_argument("--saintgall_path", default=DEFAULT_RAW["SaintGall"])
    parser.add_argument("--washington_path", default=DEFAULT_RAW["Washington"])
    parser.add_argument("--output_root", default=DEFAULT_OUT)
    parser.add_argument("--fold", default="cv1", help="Washington cross-validation fold")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    IAMHistDBDatasetFormatter("SaintGall", "line", dpi=args.dpi, raw_data_path=args.saintgall_path,
                              output_root=args.output_root).format()
    IAMHistDBDatasetFormatter("SaintGall", "page", dpi=args.dpi, raw_data_path=args.saintgall_path,
                              output_root=args.output_root).format()
    IAMHistDBDatasetFormatter("Washington", "line", dpi=args.dpi, raw_data_path=args.washington_path,
                              output_root=args.output_root, fold=args.fold).format()
