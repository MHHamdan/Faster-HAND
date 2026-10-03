#  Bentham R0 formatter for the HAND framework (line and page level).
#
#  Source: "Bentham Dataset R0" (PRHLT, UPV, June 2014), read in place from an
#  unpacked tree:
#      <raw>/BenthamDatasetR0-GT/{Partitions,PAGE,Transcriptions,Images/Lines,README.txt}
#      <raw>/BenthamDatasetR0-Images/Images/Pages/*.jpg          (300 dpi, colour)
#
#  Line level  : the pre-extracted polygon crops GT/Images/Lines/<line_id>.png (RGBA,
#                alpha = manual line polygon) with GT/Transcriptions/<line_id>.txt, split
#                by Partitions/{Train,Validation,Test}Lines.lst.
#  Page level  : the page JPG with GT/PAGE/<page_id>.xml (PAGE 2010), split by
#                Partitions/{Train,Validation,Test}.lst. One paragraph per TextRegion.
#                Line ids in the XML are NOT in reading order (GT/README.txt); regions and
#                lines are re-sorted geometrically exactly as the README describes:
#                lines by the mean y of their polygon (ties: smallest x), regions by the
#                mean y of their top-most line.
#
#  Transcriptions are kept verbatim apart from stripping trailing whitespace: no
#  case-folding, no entity/tag removal, no space collapsing (punctuation is tokenised
#  with surrounding spaces in this corpus, and 70 lines contain a literal "<gap/>").

import os
import json
import pickle
import time
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from hand.Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_RAW = os.path.join(REPO_ROOT, "data", "Bentham")
DEFAULT_OUT = os.path.join(REPO_ROOT, "formatted")


def _local(tag):
    """Strip the XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]


def _coords_points(coords_el):
    """
    Polygon points of a PAGE <Coords> element, as a list of (x, y) ints.
    Train/valid files use the 2010 form (<Point x= y=/> children); the test files use the
    2013 form (points="x,y x,y ..." attribute). Both are handled.
    """
    pts = []
    for child in coords_el:
        if _local(child.tag) == "Point":
            pts.append((int(float(child.attrib["x"])), int(float(child.attrib["y"]))))
    if not pts and "points" in coords_el.attrib:
        for p in coords_el.attrib["points"].split():
            x, y = p.split(",")
            pts.append((int(float(x)), int(float(y))))
    return pts


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {"left": min(xs), "right": max(xs), "top": min(ys), "bottom": max(ys)}


class BenthamDatasetFormatter(OCRDatasetFormatter):
    """
    Formatter for Bentham R0 at line or page level.

    Args:
        level: "line" or "page"
        raw_data_path: directory holding BenthamDatasetR0-GT/ and BenthamDatasetR0-Images/
        output_root: directory under which <output_root>/Bentham_<level>/ is written
        dpi: target resolution (images and coordinates are rescaled from source_dpi)
        source_dpi: resolution of the raw images (300 for both pages and line crops)
        jpeg_quality: quality of the page JPEGs (line crops are written as PNG)
    """

    def __init__(self, level, set_names=("train", "valid", "test"), dpi=150, source_dpi=300,
                 raw_data_path=None, output_root=None, jpeg_quality=95):
        set_names = list(set_names)
        super().__init__("Bentham", level, "", set_names)
        self.dpi = dpi
        self.source_dpi = source_dpi
        self.ratio = dpi / source_dpi
        self.jpeg_quality = jpeg_quality
        self.source_fold_path = raw_data_path or DEFAULT_RAW
        self.target_fold_path = os.path.join(output_root or DEFAULT_OUT, "Bentham_{}".format(level))
        self.gt_root = os.path.join(self.source_fold_path, "BenthamDatasetR0-GT")
        self.pages_root = os.path.join(self.source_fold_path, "BenthamDatasetR0-Images", "Images", "Pages")
        self.partition_files = {
            "line": {"train": "TrainLines.lst", "valid": "ValidationLines.lst", "test": "TestLines.lst"},
            "page": {"train": "Train.lst", "valid": "Validation.lst", "test": "Test.lst"},
        }
        self.map_datasets_files.update({
            "Bentham": {
                "line": {"arx_files": [], "needed_files": [], "format_function": self.format_bentham_line},
                "page": {"arx_files": [], "needed_files": [], "format_function": self.format_bentham_page},
            }
        })
        # Everything that was not converted 1:1, for the report.
        self.skipped = []          # (split, source id, reason)
        self.notes = []            # free-text findings
        self.provenance = dict()   # formatted file name -> {"document", "source_id", "split"}
        self.elapsed = None

    # ------------------------------------------------------------------ plumbing
    def init_format(self):
        for sub in [self.gt_root, self.pages_root]:
            if not os.path.isdir(sub):
                raise FileNotFoundError(sub)
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
        print("Bentham_{}: charset {} | ".format(self.level, len(self.charset))
              + " ".join("{} {}".format(s, len(self.gt[s])) for s in self.set_names)
              + " | skipped {}".format(len(self.skipped)))

    def read_partition(self, set_name):
        path = os.path.join(self.gt_root, "Partitions", self.partition_files[self.level][set_name])
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    @staticmethod
    def clean_text(text):
        """Verbatim except trailing whitespace (the files end with '\\n', some with several)."""
        return text.rstrip()

    def read_transcription(self, line_id):
        path = os.path.join(self.gt_root, "Transcriptions", line_id + ".txt")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return self.clean_text(f.read())

    def unusual_codepoints(self):
        """Characters outside printable ASCII, with their Unicode names (for the report)."""
        out = []
        for ch in sorted(self.charset):
            if ord(ch) < 32 or ord(ch) > 126:
                out.append((ch, "U+%04X" % ord(ch), unicodedata.name(ch, "?")))
        return out

    # ------------------------------------------------------------------ line level
    def load_line_image(self, line_id):
        """
        Line crops are RGBA: RGB is the grey scan, alpha is the manual line polygon
        (0 outside, 255 inside). Composite over white so that only the annotated polygon
        remains, then keep a single grey channel (R == G == B in these files).
        """
        path = os.path.join(self.gt_root, "Images", "Lines", line_id + ".png")
        if not os.path.isfile(path):
            return None
        img = Image.open(path)
        if img.mode == "RGBA":
            rgba = np.array(img)
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            grey = rgba[..., :3].astype(np.float32).mean(axis=2, keepdims=True)
            out = grey * alpha + 255.0 * (1.0 - alpha)
            return out[..., 0].round().astype(np.uint8)
        return np.array(img.convert("L"))

    def format_bentham_line(self):
        for set_name in self.set_names:
            ids = self.read_partition(set_name)
            for i, line_id in enumerate(ids):
                text = self.read_transcription(line_id)
                if text is None:
                    self.skipped.append((set_name, line_id, "missing transcription file"))
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
                self.provenance[new_img_name] = {"document": line_id[:11], "source_id": line_id, "split": set_name}
                if (i + 1) % 1000 == 0:
                    print("  Bentham_line {}: {}/{}".format(set_name, i + 1, len(ids)))

    # ------------------------------------------------------------------ page level
    def parse_page_xml(self, page_id):
        """
        Returns (page_width, page_height, regions). Each region is
        {"id", "lines": [{"text", "pts", "coords"}]} with lines already sorted in reading
        order and regions sorted top-down (GT/README.txt steps 1-6).
        """
        root = ET.parse(os.path.join(self.gt_root, "PAGE", page_id + ".xml")).getroot()
        page = [el for el in root if _local(el.tag) == "Page"][0]
        width, height = int(page.attrib["imageWidth"]), int(page.attrib["imageHeight"])
        regions = []
        for region in page:
            if _local(region.tag) != "TextRegion":
                continue
            lines = []
            for tl in region:
                if _local(tl.tag) != "TextLine":
                    continue
                coords = [c for c in tl if _local(c.tag) == "Coords"]
                pts = _coords_points(coords[0]) if coords else []
                text = None
                for te in tl:
                    if _local(te.tag) == "TextEquiv":
                        for u in te:
                            if _local(u.tag) == "Unicode":
                                text = u.text
                if text is None:
                    self.skipped.append(("page:" + page_id, tl.attrib.get("id"), "TextLine without Unicode text"))
                    continue
                text = self.clean_text(text)
                if text == "":
                    self.skipped.append(("page:" + page_id, tl.attrib.get("id"), "TextLine with empty text"))
                    continue
                if not pts:
                    self.skipped.append(("page:" + page_id, tl.attrib.get("id"), "TextLine without polygon"))
                    continue
                # README step 2: mean y of the polygon; ties broken by the smallest x.
                mean_y = float(np.mean([p[1] for p in pts]))
                min_x = min(p[0] for p in pts)
                lines.append({"text": text, "pts": pts, "coords": _bbox(pts), "sort_key": (mean_y, min_x)})
            if not lines:
                self.skipped.append(("page:" + page_id, region.attrib.get("id"), "TextRegion without lines"))
                continue
            lines.sort(key=lambda l: l["sort_key"])                 # step 3
            regions.append({"id": region.attrib.get("id"), "lines": lines,
                            "sort_key": lines[0]["sort_key"]})       # step 4: height of top line
        regions.sort(key=lambda r: r["sort_key"])                    # step 5
        return width, height, regions

    def format_bentham_page(self):
        for set_name in self.set_names:
            ids = self.read_partition(set_name)
            for i, page_id in enumerate(ids):
                img_path = os.path.join(self.pages_root, page_id + ".jpg")
                xml_path = os.path.join(self.gt_root, "PAGE", page_id + ".xml")
                if not os.path.isfile(img_path):
                    self.skipped.append((set_name, page_id, "missing page image"))
                    continue
                if not os.path.isfile(xml_path):
                    self.skipped.append((set_name, page_id, "missing PAGE xml"))
                    continue
                xml_w, xml_h, regions = self.parse_page_xml(page_id)
                if not regions:
                    self.skipped.append((set_name, page_id, "PAGE xml has no text line"))
                    continue

                img = Image.open(img_path)
                if img.size != (xml_w, xml_h):
                    self.notes.append("{}: image {} vs xml {}x{}".format(page_id, img.size, xml_w, xml_h))
                img = self.resize(img, self.source_dpi, self.dpi)
                new_img_name = "{}_{}.jpg".format(set_name, len(self.gt[set_name]))
                Image.fromarray(img).save(os.path.join(self.target_fold_path, set_name, new_img_name),
                                          quality=self.jpeg_quality)

                paragraphs = []
                for region in regions:
                    lines = []
                    for line in region["lines"]:
                        entry = {"text": line["text"]}
                        entry.update(line["coords"])
                        lines.append(self.adjust_coord_ratio(entry, self.ratio))
                    paragraphs.append({
                        "label": "\n".join(l["text"] for l in lines),
                        "lines": lines,
                        "mode": "body",
                        "top": min(l["top"] for l in lines),
                        "bottom": max(l["bottom"] for l in lines),
                        "left": min(l["left"] for l in lines),
                        "right": max(l["right"] for l in lines),
                    })
                page_text = "\n".join(p["label"] for p in paragraphs)
                page_label = {
                    "text": page_text,
                    "paragraphs": paragraphs,
                    "nb_cols": 1,
                    "side": "left",
                    "top": min(p["top"] for p in paragraphs),
                    "bottom": max(p["bottom"] for p in paragraphs),
                    "left": min(p["left"] for p in paragraphs),
                    "right": max(p["right"] for p in paragraphs),
                    "page_width": int(img.shape[1]),
                }
                self.gt[set_name][new_img_name] = {"text": page_text, "nb_cols": 1, "pages": [page_label]}
                self.charset = self.charset.union(set(page_text))
                self.provenance[new_img_name] = {"document": page_id, "source_id": page_id, "split": set_name}
                if (i + 1) % 50 == 0:
                    print("  Bentham_page {}: {}/{}".format(set_name, i + 1, len(ids)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format Bentham R0 (line and page level) for HAND")
    parser.add_argument("--raw_path", default=DEFAULT_RAW,
                        help="directory holding BenthamDatasetR0-GT/ and BenthamDatasetR0-Images/")
    parser.add_argument("--output_root", default=DEFAULT_OUT, help="formatted/ directory")
    parser.add_argument("--levels", nargs="+", default=["line", "page"], choices=["line", "page"])
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    for level in args.levels:
        BenthamDatasetFormatter(level, dpi=args.dpi, raw_data_path=args.raw_path,
                                output_root=args.output_root).format()
