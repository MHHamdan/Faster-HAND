#!/usr/bin/env python3
"""Qualitative composites for multi-page (double / triple page) READ 2016 images.

Three figure kinds, all drawn from the raw predictions stored in the evaluation result files
of `tools/multipage_eval.py` (`experiments/multipage/...json`), never from a fresh decode, so that
every displayed string is the one the reported metrics were computed on:

  adapted      the adapted multi-page model on its own level: the full image with the
               reading-order path, the page transitions and the end-of-sequence marker on top,
               then one (ground truth | prediction) pair per page;
  zero-shot    the same layout for the single-page model applied without adaptation, where the
               pages the decoder never produced are greyed out in the image and empty in the
               text panels;
  comparison   zero-shot and adapted output on the same image, side by side, pages only.

Character errors are an exact Levenshtein alignment per page (red: substituted or inserted
characters; a caret marks a deletion; the reference characters the prediction does not reproduce
are shaded). The sample-level CER printed by `--report` is recomputed with the metric manager's
normalisation (layout tokens removed, repeated blanks collapsed) and checked against the value
stored in the result file.

    python tools/qualitative_multipage.py adapted \
        --results experiments/multipage/adaptation/FT_DOUBLE_double_page.json \
        --names test_3.jpeg --out experiments/qualitative
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import editdistance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "release"))
from hand_release.inference import strip_layout  # noqa: E402
import qualitative_figures as Q  # noqa: E402

MAX_HEIGHT_IN = 8.55      # IEEEtran journal text height is 9.67 in; leave room for the caption


# --------------------------------------------------------------------------- metrics
def harness_text(s):
    """The string the metric manager measures CER on (format_string_for_cer)."""
    s = strip_layout(s)
    s = re.sub("(\n)+", "\n", s)
    s = re.sub("( )+", " ", s).strip()
    return s


def sample_cer(gt_raw, pred_raw):
    g, p = harness_text(gt_raw), harness_text(pred_raw)
    d = editdistance.eval(g, p)
    return d, len(g), d / max(1, len(g))


def page_pairs(gt_raw, pred_raw, n_pages):
    gp, pp = Q.split_raw_pages(gt_raw), Q.split_raw_pages(pred_raw)
    pairs = []
    for k in range(n_pages):
        g = gp[k] if k < len(gp) else ""
        q = pp[k] if k < len(pp) else ""
        pairs.append((g, q))
    return pairs


def page_stats(g, q):
    d, n, cer = sample_cer(g, q)
    ld, _, _, lerr = Q.layout_error_marks(g, q)
    return {"edits": d, "chars": n, "cer": cer, "layout_token_edits": ld, "layout_token_errors": lerr}


def load_results(path):
    d = json.load(open(path, encoding="utf-8"))
    per = {r["name"]: r for r in d["per_sample"]}
    return d, per, d["raw_predictions"]


# --------------------------------------------------------------------------- text panels
UNDECODED_HEAD = 4      # reference lines shown for a page the decoder never produced


def page_panels(g, q):
    """Tokenised GT and prediction streams of one page with their error marks. For a page the
    decoder never produced, the reference is shown as its first lines and a count of the rest
    (the image panel already carries the greyed page)."""
    _, gm, pm, car = Q.char_error_marks(strip_layout(g), strip_layout(q))
    _, gb, pb, _ = Q.layout_error_marks(g, q)
    gl = Q.tokenize_stream(g, gm, gb, set())
    if q:
        return gl, Q.tokenize_stream(q, pm, pb, car)
    if len(gl) > UNDECODED_HEAD + 1:
        tail = "\u22ef %d more lines, none decoded" % (len(gl) - UNDECODED_HEAD)
        gl = gl[:UNDECODED_HEAD] + [[("char", c, "eq") for c in tail]]
    return gl, None


def draw_empty_panel(ax, fs, title, text):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 1.0, title, fontsize=fs + 1.0, family="DejaVu Sans", ha="left", va="top",
            fontweight="bold", transform=ax.transAxes)
    ax.add_patch(Rectangle((0.0, 0.04), 0.96, 0.80, fc="#f3f3f3", ec="#bbbbbb", lw=0.5,
                           transform=ax.transAxes, clip_on=False))
    import textwrap
    ax.text(0.48, 0.44, "\n".join(textwrap.wrap(text, 30)), fontsize=fs + 0.2, family="DejaVu Sans",
            ha="center", va="center", color="#555555", style="italic", transform=ax.transAxes,
            linespacing=1.4)


# --------------------------------------------------------------------------- composite
def composite(img, sample, gt_raw, pred_raw, out_base, fs=6.2, width_in=18.5 / 2.54,
              decoded_pages=None, max_height_in=MAX_HEIGHT_IN, pred_title="HAND prediction",
              empty_text=None):
    """Full-width image on top, (GT | prediction) pairs per page below, two pairs per row."""
    order = Q.reading_order(sample)
    n_pages = len(sample["pages"])
    pairs = page_pairs(gt_raw, pred_raw, n_pages)
    if decoded_pages is None:
        decoded_pages = [k for k, (_, q) in enumerate(pairs) if q]
    undecoded = [k for k in range(n_pages) if k not in decoded_pages]
    end_page = max(decoded_pages) if decoded_pages else None
    panels = [page_panels(g, q) for g, q in pairs]
    per_page = [page_stats(g, q) for g, q in pairs]
    img, order, sample = Q.crop_to_page(img, order, sample, crop_x=True)
    H, W = img.shape[:2]

    gap, top_pad, row_gap, legend_h = 0.12, 0.16, 0.10, 0.42
    page_cols = 2
    n_rows = int(np.ceil(n_pages / page_cols))
    legend_in_slot = (n_pages % page_cols) != 0
    while True:
        cell_in = Q.MONO_ADVANCE * fs / 72.0
        line_in = Q.LINE_SPACING * fs / 72.0
        text_w = (width_in - (2 * page_cols - 1) * gap) / (2 * page_cols)
        max_cells = int(text_w / cell_in) - 1
        row_h, row_nl = [], []
        for r_ in range(n_rows):
            nl = 4
            for k in range(r_ * page_cols, min(n_pages, (r_ + 1) * page_cols)):
                gl, pl = panels[k]
                nl = max(nl, len(Q.wrap_lines(gl, max_cells)), len(Q.wrap_lines(pl, max_cells)) if pl else 0)
            row_h.append((nl + 1.6) * line_in)
            row_nl.append(nl)
        # the free slot of an odd-page figure takes the legend only if its row is tall enough
        legend_in_slot = (n_pages % page_cols) != 0 and row_h[-1] >= 10 * 1.35 * line_in
        img_w = width_in
        img_h = img_w * H / W + 0.02
        fig_h = top_pad + img_h + 0.16 + sum(row_h) + row_gap * (n_rows - 1) + \
            (0.08 if legend_in_slot else legend_h + 0.02)
        if fig_h <= max_height_in or fs <= 5.6:
            break
        fs -= 0.2
    if fig_h > max_height_in:
        print("warning: %s is %.2f in tall at %.1f pt" % (out_base, fig_h, fs))

    fig = plt.figure(figsize=(width_in, fig_h))
    ax_img = fig.add_axes([0, 1 - (top_pad + img_h) / fig_h, img_w / width_in, img_h / fig_h])
    Q.draw_image_panel(ax_img, img, order, sample, title="(a) input image and reading order",
                       label_fs=4.2, page_labels=True, end_after_page=end_page,
                       undecoded_pages=undecoded)
    y = 1 - (top_pad + img_h + 0.16) / fig_h
    for r_ in range(n_rows):
        y -= row_h[r_] / fig_h
        for c_ in range(page_cols):
            k = r_ * page_cols + c_
            if k >= n_pages:
                # legend in the free slot of the last row
                x = (2 * c_ * (text_w + gap)) / width_in
                if legend_in_slot:
                    ax = fig.add_axes([x, y, (2 * text_w + gap) / width_in, row_h[r_] / fig_h])
                    draw_legend_block(ax, fs, row_h[r_])
                break
            gl, pl = panels[k]
            for h_, (lines, title) in enumerate([
                    (gl, "(b%d) ground truth, page %d" % (k + 1, k + 1)),
                    (pl, "(c%d) %s, page %d" % (k + 1, pred_title, k + 1))]):
                x = ((2 * c_ + h_) * (text_w + gap)) / width_in
                ax = fig.add_axes([x, y, text_w / width_in, row_h[r_] / fig_h])
                if lines is None:
                    draw_empty_panel(ax, fs, title, empty_text or "no output")
                else:
                    Q.draw_text_panel(ax, lines, fs, bg_for=Q._bg, title=title, max_cells=max_cells,
                                      title_fs=fs + 1.0, n_rows=row_nl[r_])
        y -= row_gap / fig_h
    if not legend_in_slot:
        ax_leg = fig.add_axes([0.0, 0.0, 1.0, legend_h / fig_h])
        Q.draw_legend(ax_leg, multipage=True)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=200)
    plt.close(fig)
    return {"figure_size_in": [round(width_in, 2), round(float(fig_h), 2)], "font_pt": round(fs, 1),
            "per_page": per_page, "decoded_pages": decoded_pages}


def draw_legend_block(ax, fs, row_h_in):
    """Legend for the free slot of an odd-page figure: the same entries as Q.draw_legend,
    one line per row at the text panels' line pitch."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    f = fs - 0.4
    step = 1.35 * Q.LINE_SPACING * fs / 72.0 / row_h_in
    x, y = 0.0, 1.0 - 1.6 * step
    for key, lab in Q.LEGEND_ITEMS:
        ax.add_patch(Rectangle((x, y - 0.025), 0.03, 0.05, fc=Q.CLASS_COLOR[key], ec="none",
                               transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.04, y, lab, fontsize=f, va="center", ha="left", transform=ax.transAxes)
        x += 0.04 + 0.016 * len(lab) + 0.04
    rows = [(Q.LEGEND_PATH, "#333333", None, "normal"), (Q.LEGEND_MULTIPAGE, "#333333", None, "normal")] \
        + list(Q.LEGEND_ERRORS)
    y -= 1.4 * step
    for text, col, bg, weight in rows:
        kw = dict(bbox=dict(boxstyle="square,pad=0.15", fc=bg, ec="none")) if bg else {}
        ax.text(0, y, text, fontsize=f, va="center", ha="left", color=col, fontweight=weight,
                transform=ax.transAxes, **kw)
        y -= step


# --------------------------------------------------------------------------- comparison
def comparison(img, sample, gt_raw, zs_raw, ad_raw, out_base, width_in=18.5 / 2.54,
               titles=("(a) page model, zero-shot", "(b) after double-page adaptation")):
    """Zero-shot and adapted output on the same image: which pages were decoded, at what CER."""
    n_pages = len(sample["pages"])
    order = Q.reading_order(sample)
    img, order, sample = Q.crop_to_page(img, order, sample, crop_x=True)
    H, W = img.shape[:2]
    gap, top_pad, bottom_pad = 0.16, 0.16, 0.06
    panel_w = (width_in - gap) / 2
    panel_h = panel_w * H / W
    fig_h = top_pad + panel_h + bottom_pad
    fig = plt.figure(figsize=(width_in, fig_h))
    stats = {}
    for k, (raw, title) in enumerate([(zs_raw, titles[0]), (ad_raw, titles[1])]):
        pairs = page_pairs(gt_raw, raw, n_pages)
        decoded = [i for i, (_, q) in enumerate(pairs) if q]
        pp = [page_stats(g, q) for g, q in pairs]
        notes = {i: "CER %.1f %%" % (100 * pp[i]["cer"]) for i in decoded}
        ax = fig.add_axes([k * (panel_w + gap) / width_in, bottom_pad / fig_h, panel_w / width_in,
                           panel_h / fig_h])
        Q.draw_image_panel(ax, img, order, sample, title=title, number_every=0, label_fs=4.2,
                           page_labels=True, end_after_page=max(decoded) if decoded else None,
                           undecoded_pages=[i for i in range(n_pages) if i not in decoded],
                           page_notes=notes, section_labels=False)
        d, n, cer = sample_cer(gt_raw, raw)
        stats[["zero_shot", "adapted"][k]] = {"decoded_pages": decoded, "per_page": pp,
                                              "edits": d, "chars": n, "cer": cer}
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=200)
    plt.close(fig)
    stats["figure_size_in"] = [round(width_in, 2), round(float(fig_h), 2)]
    return stats


# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["adapted", "zero-shot", "comparison"])
    ap.add_argument("--results", required=True,
                    help="multipage_eval result JSON holding the prediction (adapted / zero-shot)")
    ap.add_argument("--results-zero-shot", default=None,
                    help="comparison only: result JSON of the zero-shot page model on the same level")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/qualitative")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--font-size", type=float, default=6.2)
    ap.add_argument("--width-cm", type=float, default=18.5)
    ap.add_argument("--max-height-in", type=float, default=MAX_HEIGHT_IN)
    ap.add_argument("--pred-title", default="HAND prediction")
    a = ap.parse_args()

    res_path = a.results if os.path.isabs(a.results) else os.path.join(REPO, a.results)
    res, per, raw_preds = load_results(res_path)
    data_dir = os.path.join(REPO, res["data"])
    split = res["split"]
    level = res["level"]
    gt = Q.load_labels(data_dir, split)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    prefix = a.prefix or "read2016_%s_%s" % (level, {"adapted": "adapted", "zero-shot": "zeroshot",
                                                     "comparison": "zeroshot_vs_adapted"}[a.kind])
    zs = None
    if a.kind == "comparison":
        zp = a.results_zero_shot if os.path.isabs(a.results_zero_shot) else os.path.join(REPO, a.results_zero_shot)
        zs = load_results(zp)
        assert zs[0]["level"] == level and zs[0]["split"] == split
    for name in a.names:
        sample = gt[name]
        gt_raw = sample["text"]
        pred_raw = raw_preds[name]
        rec = per[name]
        d, n, cer = sample_cer(gt_raw, pred_raw)
        assert d == rec["edit_chars"] and n == rec["nb_chars"], \
            "CER recomputation differs from the result file for %s: %d/%d vs %d/%d" % (
                name, d, n, rec["edit_chars"], rec["nb_chars"])
        img_path = os.path.join(data_dir, split, name)
        img = np.asarray(Image.open(img_path).convert("RGB"))
        stem = name.rsplit(".", 1)[0]
        base = os.path.join(out_dir, "%s_%s_%s" % (prefix, split, stem))
        record = {"image": os.path.relpath(img_path, REPO), "results": os.path.relpath(res_path, REPO),
                  "config": res["config"], "export_dir": res["export_dir"], "level": level,
                  "decode_path": res.get("decode_path"), "amp": res.get("amp"),
                  "per_sample": rec, "raw": pred_raw, "gt_raw": gt_raw}
        if a.kind == "comparison":
            zs_raw = zs[2][name]
            zrec = zs[1][name]
            stats = comparison(img, sample, gt_raw, zs_raw, pred_raw, base, width_in=a.width_cm / 2.54,
                               titles=("(a) page model, zero-shot",
                                       "(b) after %s adaptation" % level.replace("_", "-")))
            record.update({"zero_shot": {"results": os.path.relpath(zp, REPO), "config": zs[0]["config"],
                                         "export_dir": zs[0]["export_dir"], "per_sample": zrec,
                                         "raw": zs_raw}})
        else:
            if a.kind == "zero-shot":
                empty = ("no output: the decoder emitted the end-of-sequence symbol after closing "
                         "the first page")
            else:
                empty = "no output"
            stats = composite(img, sample, gt_raw, pred_raw, base, fs=a.font_size,
                              width_in=a.width_cm / 2.54, max_height_in=a.max_height_in,
                              pred_title=a.pred_title, empty_text=empty)
        ld, _, _, lerr = Q.layout_error_marks(gt_raw, pred_raw)
        stats.update({"cer": cer, "edits": d, "chars": n, "layout_token_edits": ld,
                      "layout_token_errors": lerr})
        record["figure_stats"] = stats
        json.dump(record, open(base.replace(out_dir, os.path.join(out_dir, "predictions")) + ".json",
                               "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        pp = stats.get("per_page") or stats.get("adapted", {}).get("per_page", [])
        print("%-12s %-10s CER %.2f%% (%d/%d)  WER %.2f%%  LOER %.4f  mAP-CER %.3f  layout edits %d  "
              "pages %s  size %s  -> %s.pdf" % (
                  name, level, 100 * cer, d, n, 100 * rec["wer"], rec["loer"] if rec["loer"] is not None else float("nan"),
                  rec["map_cer"], ld, " ".join("%.1f%%" % (100 * p["cer"]) for p in pp),
                  stats["figure_size_in"], os.path.relpath(base, REPO)))


if __name__ == "__main__":
    main()
