#  IAM "Aachen" (RWTH) split formatter for the HAND framework.
#
#  The Aachen split is the form-level partition used by VAN, DAN, HTR-VT and TrOCR
#  comparisons: 747 / 116 / 336 forms (train / valid / test), 6,482 / 976 / 2,915 lines.
#  Form lists come from OpenSLR 56 (`{train,validation,test}.uttlist`); VAN's own
#  annotation files (`{train,valid,test}.xml`) list the same forms and are used as an
#  optional cross-check.
#
#  Produces, from the raw IAM distribution (data/IAM/{forms,lines,xml,ascii}):
#    level="line": one sample per line, ALL lines of every form in the list (no `err`
#                  filtering); the lines.txt segmentation flag is kept as
#                  "segmentation_flag".
#  Text: "text" is VAN's own `Line Value` (natural punctuation spacing, "tomorrow. Mr.",
#  but spaced double quotes, `" prop up "`), i.e. exactly what VAN/DAN train and score on;
#  without VAN's XML it falls back to the decoded form-XML text (data/IAM/xml), which
#  differs from VAN only in the spacing around double quotes. The lines.txt string
#  (`|` -> space, tokenised punctuation, "tomorrow . Mr.") is kept as "text_lines_txt";
#  it differs from "text" for ~62% of lines, by spacing only. Same 79-char charset.
#    level="page": one sample per form = the handwritten block, cropped exactly like
#                  IAMDatasetFormatter.format_iam_paragraph (union of the line boxes,
#                  10 px padding at 300 dpi) and written with the IAM_page schema
#                  ({"text", "nb_cols", "pages": [...]}) plus "writer_id" / "form_id".
#  Both levels write a `provenance.json` sidecar (formatted name -> writer, form, split)
#  so tools/audit_dataset_integrity.py can verify writer independence.

import argparse
import json
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from hand.Datasets.dataset_formatters.iam_formatter import IAMDatasetFormatter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

UTTLIST_NAMES = {"train": "train.uttlist", "valid": "validation.uttlist", "test": "test.uttlist"}
VAN_XML_NAMES = {"train": "train.xml", "valid": "valid.xml", "test": "test.xml"}
PAD_300DPI = 10  # same paragraph padding as IAMDatasetFormatter.format_iam_paragraph


class IAMAachenDatasetFormatter(IAMDatasetFormatter):

    def __init__(self, level, split_dir, raw_data_path, output_path, van_xml_dir=None,
                 dpi=150, source_dpi=300):
        assert level in ("line", "page"), level
        super().__init__(level=level, dpi=dpi, source_dpi=source_dpi, raw_data_path=raw_data_path)
        self.split_dir = split_dir
        self.van_xml_dir = van_xml_dir
        self.target_fold_path = output_path
        self.map_datasets_files["IAM"]["line"]["format_function"] = self.format_aachen_line
        self.map_datasets_files["IAM"]["page"]["format_function"] = self.format_aachen_page
        self.provenance = dict()
        self.stats = {"level": level, "splits": {}, "text_differs_from_lines_txt": [],
                      "text_differs_from_form_xml": [], "text_source": None}

    # ------------------------------------------------------------------ inputs
    def load_form_lists(self):
        forms = {}
        for set_name in self.set_names:
            path = os.path.join(self.split_dir, UTTLIST_NAMES[set_name])
            with open(path) as f:
                ids = [l.strip() for l in f if l.strip()]
            assert len(ids) == len(set(ids)), "duplicate form ids in {}".format(path)
            forms[set_name] = sorted(ids)
        for a in self.set_names:
            for b in self.set_names:
                if a < b:
                    assert not set(forms[a]) & set(forms[b]), "form overlap between {} and {}".format(a, b)
        return forms

    def load_van_forms(self):
        """VAN's train/valid/test.xml: form id -> (writer id, [line values])."""
        if self.van_xml_dir is None:
            return None
        out = {}
        for set_name in self.set_names:
            path = os.path.join(self.van_xml_dir, VAN_XML_NAMES[set_name])
            if not os.path.exists(path):
                return None
            root = ET.parse(path).getroot()
            out[set_name] = {}
            for page in root:
                form_id = os.path.basename(page.get("FileName")).split(".")[0]
                writer = page.find("WriterID").get("Name")
                lines = [self.format_text_label(l.get("Value"))
                         for p in page.findall("Paragraph") for l in p.findall("Line")]
                out[set_name][form_id] = (writer, lines)
        return out

    def load_writer_ids(self):
        """data/IAM/ascii/forms.txt: form id -> writer id (3-digit string)."""
        path = os.path.join(self.source_fold_path, "ascii", "forms.txt")
        writers = {}
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                writers[parts[0]] = parts[1]
        return writers

    def lines_of_form(self, transcriptions, form_id):
        """All lines.txt entries of a form, in line-number order."""
        ids = [lid for lid in transcriptions if lid.rsplit("-", 1)[0] == form_id]
        return sorted(ids, key=lambda lid: int(lid.rsplit("-", 1)[1]))

    def prepare(self):
        self.forms = self.load_form_lists()
        self.writers = self.load_writer_ids()
        self.transcriptions = self.load_lines_txt()
        assert self.transcriptions, "ascii/lines.txt not found under {}".format(self.source_fold_path)
        # index lines.txt by form once
        self.form_lines = {}
        for lid in self.transcriptions:
            self.form_lines.setdefault(lid.rsplit("-", 1)[0], []).append(lid)
        for fid in self.form_lines:
            self.form_lines[fid].sort(key=lambda lid: int(lid.rsplit("-", 1)[1]))

        van = self.load_van_forms()
        self.van_text = {}
        if van is not None:
            for set_name in self.set_names:
                assert set(van[set_name]) == set(self.forms[set_name]), \
                    "VAN {}.xml forms differ from {}".format(set_name, UTTLIST_NAMES[set_name])
                for fid in self.forms[set_name]:
                    assert van[set_name][fid][0] == self.writers[fid], (fid, "writer id mismatch")
                    assert len(van[set_name][fid][1]) == len(self.form_lines[fid]), (fid, "line count mismatch")
                    for lid, value in zip(self.form_lines[fid], van[set_name][fid][1]):
                        self.van_text[lid] = value
            self.stats["van_xml_cross_check"] = "forms, writer ids and per-form line counts identical"
            self.stats["text_source"] = "VAN Line Value"
        else:
            self.stats["van_xml_cross_check"] = "not available"
            self.stats["text_source"] = "data/IAM/xml line text (decoded)"

    def line_text_lines_txt(self, line_id):
        """lines.txt transcription ('|' -> space, tokenised punctuation), normalised."""
        return self.format_text_label(self.transcriptions[line_id]["text"])

    def line_text(self, xml_data, line_id):
        """Primary transcription: VAN's Line Value, else the decoded form-XML text."""
        assert xml_data is not None and line_id in xml_data, "line {} missing from form xml".format(line_id)
        xml_text = self.format_text_label(xml_data[line_id]["text"])
        text = self.van_text.get(line_id, xml_text)
        assert text.replace(" ", "") == xml_text.replace(" ", ""), (line_id, text, xml_text)
        if text != xml_text:
            self.stats["text_differs_from_form_xml"].append(line_id)
        if text != self.line_text_lines_txt(line_id):
            self.stats["text_differs_from_lines_txt"].append(line_id)
        return text

    # ------------------------------------------------------------------ line level
    def format_aachen_line(self):
        self.prepare()
        for set_name in self.set_names:
            n, n_err, n_empty = 0, 0, 0
            for form_id in self.forms[set_name]:
                xml_data, _ = self.load_xml_data(form_id)
                writer = self.writers[form_id]
                for line_id in self.form_lines[form_id]:
                    img_path = self.find_line_image(line_id)
                    assert img_path is not None, "line image missing: {}".format(line_id)
                    img = np.array(Image.open(img_path).convert("L"))
                    img = self.resize(img, self.source_dpi, self.dpi)
                    new_name = "{}_{}.png".format(set_name, n)
                    Image.fromarray(img, mode="L").save(os.path.join(self.target_fold_path, set_name, new_name))

                    text = self.line_text(xml_data, line_id)
                    flag = self.transcriptions[line_id]["status"]
                    n_err += flag == "err"
                    n_empty += len(text) == 0
                    self.charset = self.charset.union(set(text))
                    self.gt[set_name][new_name] = {
                        "text": text,
                        "text_lines_txt": self.line_text_lines_txt(line_id),
                        "segmentation_flag": flag,
                        "line_id": line_id,
                        "form_id": form_id,
                        "writer_id": writer,
                    }
                    self.provenance[new_name] = {"writer": writer, "document": form_id,
                                                 "line": line_id, "split": set_name}
                    n += 1
            self.stats["splits"][set_name] = {"n_lines": n, "n_err_lines": n_err, "n_empty_text": n_empty,
                                              "n_forms": len(self.forms[set_name]),
                                              "n_writers": len({self.writers[f] for f in self.forms[set_name]})}
            print("  {}: {} lines ({} err, {} empty) from {} forms".format(
                set_name, n, n_err, n_empty, len(self.forms[set_name])))

    # ------------------------------------------------------------------ page level
    def format_aachen_page(self):
        self.prepare()
        ratio = self.dpi / self.source_dpi
        for set_name in self.set_names:
            n, n_err = 0, 0
            for form_id in self.forms[set_name]:
                xml_data, _ = self.load_xml_data(form_id)
                assert xml_data is not None, "xml missing: {}".format(form_id)
                writer = self.writers[form_id]
                img_path = self.find_form_image(form_id)
                assert img_path is not None, "form image missing: {}".format(form_id)
                form_img = np.array(Image.open(img_path).convert("L"))

                lines = []
                for line_id in self.form_lines[form_id]:
                    n_err += self.transcriptions[line_id]["status"] == "err"
                    lines.append({"id": line_id, "text": self.line_text(xml_data, line_id),
                                  "flag": self.transcriptions[line_id]["status"],
                                  "coords": xml_data[line_id]["coords"]})

                # crop: same as IAMDatasetFormatter.format_iam_paragraph
                para_top = max(0, min(l["coords"]["top"] for l in lines) - PAD_300DPI)
                para_bottom = min(form_img.shape[0], max(l["coords"]["bottom"] for l in lines) + PAD_300DPI)
                para_left = max(0, min(l["coords"]["left"] for l in lines) - PAD_300DPI)
                para_right = min(form_img.shape[1], max(l["coords"]["right"] for l in lines) + PAD_300DPI)
                para_img = form_img[para_top:para_bottom, para_left:para_right].copy()
                para_img = self.resize(para_img, self.source_dpi, self.dpi)

                new_name = "{}_{}.png".format(set_name, n)
                Image.fromarray(para_img, mode="L").save(os.path.join(self.target_fold_path, set_name, new_name))

                page_text = "\n".join(l["text"] for l in lines)
                self.charset = self.charset.union(set(page_text))
                line_info = [{
                    "text": l["text"],
                    "top": int((l["coords"]["top"] - para_top) * ratio),
                    "bottom": int((l["coords"]["bottom"] - para_top) * ratio),
                    "left": int((l["coords"]["left"] - para_left) * ratio),
                    "right": int((l["coords"]["right"] - para_left) * ratio),
                    "line_id": l["id"],
                    "segmentation_flag": l["flag"],
                } for l in lines]
                paragraph = {
                    "label": page_text,
                    "lines": line_info,
                    "mode": "body",
                    "top": min(l["top"] for l in line_info),
                    "bottom": max(l["bottom"] for l in line_info),
                    "left": min(l["left"] for l in line_info),
                    "right": max(l["right"] for l in line_info),
                }
                page_label = {
                    "text": page_text,
                    "paragraphs": [paragraph],
                    "nb_cols": 1,
                    "side": "single",
                    "top": paragraph["top"],
                    "bottom": paragraph["bottom"],
                    "left": paragraph["left"],
                    "right": paragraph["right"],
                    "page_width": int(para_img.shape[1]),
                }
                self.gt[set_name][new_name] = {
                    "text": page_text,
                    "nb_cols": 1,
                    "pages": [page_label],
                    "writer_id": writer,
                    "form_id": form_id,
                }
                self.provenance[new_name] = {"writer": writer, "document": form_id, "split": set_name}
                n += 1
            self.stats["splits"][set_name] = {"n_pages": n, "n_lines": sum(len(v["pages"][0]["paragraphs"][0]["lines"])
                                                                          for v in self.gt[set_name].values()),
                                              "n_err_lines": n_err,
                                              "n_writers": len({self.writers[f] for f in self.forms[set_name]})}
            print("  {}: {} pages ({} err lines)".format(set_name, n, n_err))

    # ------------------------------------------------------------------ outputs
    def init_format(self):
        super().init_format()
        for set_name in self.set_names:
            d = os.path.join(self.target_fold_path, set_name)
            for fn in os.listdir(d):  # fresh build: sequential names must start at 0
                os.remove(os.path.join(d, fn))

    def end_format(self):
        super().end_format()
        self.stats["n_text_differs_from_lines_txt"] = len(self.stats["text_differs_from_lines_txt"])
        self.stats["n_text_differs_from_form_xml"] = len(self.stats["text_differs_from_form_xml"])
        self.stats["charset_size"] = len(self.charset)
        with open(os.path.join(self.target_fold_path, "provenance.json"), "w") as f:
            json.dump(self.provenance, f, indent=0)
        with open(os.path.join(self.target_fold_path, "format_stats.json"), "w") as f:
            json.dump(self.stats, f, indent=2)
        print("wrote", os.path.join(self.target_fold_path, "labels.pkl"), "+ provenance.json + format_stats.json")


def main():
    ap = argparse.ArgumentParser(description="Format IAM with the Aachen/RWTH form split")
    ap.add_argument("--level", choices=["line", "page"], required=True)
    ap.add_argument("--split_dir", required=True, help="dir with train/validation/test.uttlist")
    ap.add_argument("--raw_path", default=os.path.join(REPO_ROOT, "data", "IAM"))
    ap.add_argument("--output_path", default=None, help="default formatted/IAM_<level>_aachen")
    ap.add_argument("--van_xml_dir", default=os.path.join(REPO_ROOT, "third_party", "VerticalAttentionOCR",
                                                          "Datasets", "raw", "IAM"))
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    out = a.output_path or os.path.join(REPO_ROOT, "formatted", "IAM_{}_aachen".format(a.level))
    IAMAachenDatasetFormatter(a.level, a.split_dir, a.raw_path, out,
                              van_xml_dir=a.van_xml_dir if os.path.isdir(a.van_xml_dir) else None,
                              dpi=a.dpi).format()


if __name__ == "__main__":
    main()
