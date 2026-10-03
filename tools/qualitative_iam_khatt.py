#!/usr/bin/env python3
"""Qualitative composite figures for the text-only models (IAM page, KHATT paragraph).

These checkpoints emit plain transcriptions with line breaks and no layout tokens, so the
figure has three columns per example and nothing else:

    [ input image (cropped to the labelled text box) ] [ reference ] [ HAND prediction ]

Several examples are stacked into one page-width composite. Character-level differences
come from an exact Levenshtein alignment of the two strings: substituted and inserted
characters are red and bold in the prediction, the reference characters the prediction
does not reproduce are shaded, and a red caret in the prediction marks a deletion. Line
breaks are preserved; long lines wrap with a hook.

Predictions are never produced here. They are read from the per-sample dump written by
`hand_v2/eval/dump_predictions.py`, which runs the released checkpoint through the exact
evaluation path (`tools/evaluate_hand.py`), e.g.

    CUDA_VISIBLE_DEVICES=0 python3 hand_v2/eval/dump_predictions.py \
        --model iam_page --split test --out outputs/iam_predictions

The per-sample CER printed on the figure is computed with the metric module's own
formatting (`hand.basic.metric_manager.format_string_for_cer` + editdistance), so the mean
over the dump reproduces the corpus CER of `results_real/iam_page.json`.

    python3 tools/qualitative_iam_khatt.py \
        --predictions outputs/iam_predictions/iam_page__IAM_page__test.json \
        --data formatted/IAM_page --split test \
        --names test_12.png test_7.png test_3.png \
        --labels "representative" "difficult handwriting" "failure case" \
        --out experiments/qualitative/iam --prefix iam \
        --composite iam_examples --single "0:iam_representative:IAM (English)" --stats

`--stats` prints the per-sample CER distribution of the whole dump (min, median, mean,
max, count above a threshold) and exits unless names are given.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import editdistance
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42        # embed TrueType, not Type 3
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from hand.basic.metric_manager import format_string_for_cer  # noqa: E402
from tools.qualitative_figures import (  # noqa: E402
    MONO, MONO_ADVANCE, LINE_SPACING, ERR_FG, ERR_BG, GT_BG,
    char_error_marks, tokenize_stream, wrap_lines, draw_text_panel, _bg)


# --------------------------------------------------------------------------- data
def load_dump(path):
    d = json.load(open(path, encoding="utf-8"))
    by_name = {}
    for s in d["samples"]:
        by_name[os.path.basename(s["name"])] = s
    return d, by_name


def sample_cer(gt, pred):
    """Exactly the metric manager's per-sample edit count and character count."""
    g, p = format_string_for_cer(gt, None), format_string_for_cer(pred, None)
    return editdistance.eval(g, p), len(g)


def cer_table(dump):
    rows = []
    for s in dump["samples"]:
        e, n = sample_cer(s["ground_truth"], s["prediction"])
        rows.append((s["name"], e, n, e / max(1, n)))
    return rows


def load_labels(data_dir, split):
    with open(os.path.join(data_dir, "labels.pkl"), "rb") as f:
        return pickle.load(f)["ground_truth"][split]


def crop_to_box(img, sample, margin_frac=0.025):
    """Crop to the union of the labelled line and page boxes plus a margin (both axes).

    The page box alone is not safe: on some IAM forms it stops above the last line."""
    H, W = img.shape[:2]
    boxes = list(sample.get("pages") or [])
    for p in sample.get("pages") or []:
        for para in p.get("paragraphs", []):
            boxes += para.get("lines", [])
    if not boxes:
        return img
    top = min(b["top"] for b in boxes)
    bottom = max(b["bottom"] for b in boxes)
    left = min(b["left"] for b in boxes)
    right = max(b["right"] for b in boxes)
    my, mx = int(margin_frac * H), int(margin_frac * W)
    return img[max(0, top - my):min(H, bottom + my), max(0, left - mx):min(W, right + mx)]


# --------------------------------------------------------------------------- layout
def row_geometry(gt_lines, pr_lines, img_hw, fs, width_in, img_w_in, gap):
    """Text panel width, cell count, row height (inches) for one example."""
    cell_in = MONO_ADVANCE * fs / 72.0
    line_in = LINE_SPACING * fs / 72.0
    text_w = (width_in - img_w_in - 2 * gap) / 2.0
    max_cells = int(text_w / cell_in) - 1
    n_rows = max(len(wrap_lines(gt_lines, max_cells)), len(wrap_lines(pr_lines, max_cells)))
    text_h = (n_rows + 0.9) * line_in
    H, W = img_hw
    img_h = img_w_in * H / W
    return text_w, max_cells, text_h, img_h


def draw_legend(ax, fs):
    ax.set_axis_off()
    ax.text(0.0, 0.5, "red, bold: inserted or substituted character", fontsize=fs,
            va="center", ha="left", color=ERR_FG, transform=ax.transAxes, fontweight="bold")
    ax.text(0.33, 0.5, "▲ missing character", fontsize=fs, va="center", ha="left",
            color=ERR_FG, transform=ax.transAxes)
    ax.text(0.50, 0.5, "shaded in the reference: characters the prediction does not reproduce",
            fontsize=fs, va="center", ha="left", color="#7a5a00", transform=ax.transAxes,
            bbox=dict(boxstyle="square,pad=0.15", fc=GT_BG, ec="none"))


def composite(examples, out_base, fs=6.0, width_in=7.28, img_w_in=2.25, gap=0.10,
              legend=True, title_fs=None, dpi=220):
    """examples: list of dicts {img, gt, pred, label, name, cer, edits, nchars}."""
    title_fs = title_fs or fs + 0.6
    rows = []
    for ex in examples:
        gt, pr = ex["gt"], ex["pred"]
        _, gt_mark, pr_mark, carets = char_error_marks(gt, pr)
        gt_lines = tokenize_stream(gt, gt_mark, [], set())
        pr_lines = tokenize_stream(pr, pr_mark, [], carets)
        text_w, max_cells, text_h, img_h = row_geometry(
            gt_lines, pr_lines, ex["img"].shape[:2], fs, width_in, img_w_in, gap)
        rows.append(dict(ex, gt_lines=gt_lines, pr_lines=pr_lines, text_w=text_w,
                         max_cells=max_cells, text_h=text_h, img_h=img_h,
                         row_h=max(img_h, text_h)))
    head = 0.27                       # two-line row title band
    row_gap = 0.14
    legend_h = 0.26 if legend else 0.0
    fig_h = sum(r["row_h"] + head for r in rows) + row_gap * (len(rows) - 1) + legend_h + 0.04
    fig = plt.figure(figsize=(width_in, fig_h))
    y = 1.0 - 0.02 / fig_h
    for k, r in enumerate(rows):
        y -= head / fig_h
        tag = "(%s) %s" % ("abcdefgh"[k], r["label"]) if len(rows) > 1 else r["label"]
        yc = y + (head - 0.075) / fig_h          # bold label and column headers
        ys = y + 0.065 / fig_h                    # sample id and per-example statistics
        fig.text(0.0, yc, tag, fontsize=title_fs, fontweight="bold", ha="left", va="center",
                 family="DejaVu Sans")
        fig.text(0.0, ys, "%s: CER %.1f%% (%d edits / %d characters)" % (
            r["name"], 100 * r["cer"], r["edits"], r["nchars"]), fontsize=title_fs - 1.2,
            ha="left", va="center", color="#444444", family="DejaVu Sans")
        for h_, col in enumerate(["reference", "HAND prediction"]):
            fig.text((img_w_in + gap + h_ * (r["text_w"] + gap)) / width_in, yc, col,
                     fontsize=title_fs, fontweight="bold", ha="left", va="center",
                     family="DejaVu Sans")
        y -= r["row_h"] / fig_h
        ax = fig.add_axes([0.0, y + (r["row_h"] - r["img_h"]) / fig_h, img_w_in / width_in,
                           r["img_h"] / fig_h])
        ax.imshow(r["img"], cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                   ec="#999999", lw=0.4))
        for h_, lines in enumerate([r["gt_lines"], r["pr_lines"]]):
            x = (img_w_in + gap + h_ * (r["text_w"] + gap)) / width_in
            axt = fig.add_axes([x, y + (r["row_h"] - r["text_h"]) / fig_h,
                                r["text_w"] / width_in, r["text_h"] / fig_h])
            draw_text_panel(axt, lines, fs, bg_for=_bg, title=None, max_cells=r["max_cells"])
        y -= row_gap / fig_h
    if legend:
        axl = fig.add_axes([0.0, 0.0, 1.0, 0.18 / fig_h])
        draw_legend(axl, fs - 0.9)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=dpi)
    plt.close(fig)
    return {"figure_size_in": [width_in, round(float(fig_h), 2)],
            "rows": [{"name": r["name"], "cer": r["cer"], "edits": r["edits"],
                      "n_chars": r["nchars"], "n_lines_gt": len(r["gt_lines"]),
                      "n_lines_pred": len(r["pr_lines"])} for r in rows]}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="dump written by hand_v2/eval/dump_predictions.py")
    ap.add_argument("--data", default="formatted/IAM_page")
    ap.add_argument("--split", default="test")
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[], help="one row title per name")
    ap.add_argument("--out", default="experiments/qualitative/iam")
    ap.add_argument("--prefix", default="iam")
    ap.add_argument("--composite", default=None, help="basename of the stacked composite")
    ap.add_argument("--single", nargs="*", default=[],
                    help="<index>:<basename>[:<label>] single-example panels, index into --names")
    ap.add_argument("--font-size", type=float, default=6.0)
    ap.add_argument("--width-cm", type=float, default=18.5)
    ap.add_argument("--image-width-cm", type=float, default=5.7)
    ap.add_argument("--stats", action="store_true", help="print the per-sample CER distribution")
    ap.add_argument("--threshold", type=float, default=0.15)
    a = ap.parse_args()

    dump, by_name = load_dump(a.predictions if os.path.isabs(a.predictions)
                              else os.path.join(REPO, a.predictions))
    table = cer_table(dump)
    cers = np.array([r[3] for r in table])
    corpus = sum(r[1] for r in table) / sum(r[2] for r in table)
    if a.stats:
        print("dump: %s  model %s  split %s  n=%d" % (a.predictions, dump["model"], dump["split"], len(table)))
        print("corpus CER (sum edits / sum chars) = %.4f   manager: %s" % (
            corpus, dump.get("metrics_from_manager", {}).get("cer")))
        print("per-sample CER: min %.4f  median %.4f  mean %.4f  max %.4f  >%.0f%%: %d/%d" % (
            cers.min(), np.median(cers), cers.mean(), cers.max(), 100 * a.threshold,
            int((cers > a.threshold).sum()), len(cers)))
        for name, e, n, c in sorted(table, key=lambda r: r[3]):
            print("  %-14s %6.2f%%  %4d / %4d" % (name, 100 * c, e, n))
    if not a.names:
        return

    data_dir = a.data if os.path.isabs(a.data) else os.path.join(REPO, a.data)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    os.makedirs(out_dir, exist_ok=True)
    gt_all = load_labels(data_dir, a.split)
    labels = list(a.labels) + [""] * (len(a.names) - len(a.labels))
    examples = []
    for name, label in zip(a.names, labels):
        s = by_name[name]
        e, n = sample_cer(s["ground_truth"], s["prediction"])
        img = np.asarray(Image.open(os.path.join(data_dir, a.split, name)).convert("L"))
        img = crop_to_box(img, gt_all[name])
        examples.append(dict(img=img, gt=s["ground_truth"], pred=s["prediction"], label=label,
                             name=name.rsplit(".", 1)[0], cer=e / max(1, n), edits=e, nchars=n))
        rec = {"model": dump["model"], "checkpoint": dump["checkpoint"], "dataset": dump["dataset"],
               "split": a.split, "name": name, "image": os.path.relpath(os.path.join(data_dir, a.split, name), REPO),
               "label": label, "ground_truth": s["ground_truth"], "prediction": s["prediction"],
               "edit_distance": e, "n_chars": n, "cer": e / max(1, n),
               "confidence": s.get("confidence"), "source_dump": os.path.relpath(
                   a.predictions if os.path.isabs(a.predictions) else os.path.join(REPO, a.predictions), REPO)}
        json.dump(rec, open(os.path.join(out_dir, "%s_%s_%s.json" % (a.prefix, a.split, name.rsplit(".", 1)[0])),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    width_in = a.width_cm / 2.54
    img_w_in = a.image_width_cm / 2.54
    if a.composite:
        st = composite(examples, os.path.join(out_dir, a.composite), fs=a.font_size,
                       width_in=width_in, img_w_in=img_w_in)
        print("composite -> %s.pdf  size %s in" % (os.path.join(out_dir, a.composite), st["figure_size_in"]))
        for r in st["rows"]:
            print("  %-10s CER %.2f%%  (%d / %d)  lines gt %d pred %d" % (
                r["name"], 100 * r["cer"], r["edits"], r["n_chars"], r["n_lines_gt"], r["n_lines_pred"]))
    for spec in a.single:
        idx, base, *label = spec.split(":", 2)
        ex = dict(examples[int(idx)])
        if label:
            ex["label"] = label[0]
        st = composite([ex], os.path.join(out_dir, base), fs=a.font_size, width_in=width_in,
                       img_w_in=img_w_in)
        print("single    -> %s.pdf  size %s in" % (os.path.join(out_dir, base), st["figure_size_in"]))


if __name__ == "__main__":
    main()
