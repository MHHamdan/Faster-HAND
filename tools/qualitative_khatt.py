#!/usr/bin/env python3
"""Qualitative KHATT (Arabic) paragraph figures for the supplementary material.

Each example is one row:  [ input paragraph image ] [ ground truth ] [ HAND prediction ]

Every prediction shown comes from the per-sample dump written by
``hand_v2/eval/dump_predictions.py --model khatt_paragraph --split test``, i.e. from the
released checkpoint run through the exact evaluation path. The three samples are chosen from
the per-sample CER distribution (or given with --names) and their raw predictions are written
next to the figures as JSON so that a figure can always be traced to the decode that produced
it.

Arabic rendering. The KHATT labels (and therefore the predictions) are stored per line in
reversed, visual left-to-right character order; the script reverses every line back to logical
order, shapes it with ``arabic_reshaper`` (one presentation-form glyph per logical character,
ligatures disabled, harakat kept) and then merges lam-alef pairs into their ligature glyph
itself. Glyphs are placed one by one from the right margin using the advance widths of the
embedded font (Amiri), so each glyph keeps a 1:1 link to its logical character(s) and can be
coloured individually. The python-bidi reordering is not needed because the lines contain only
right-to-left text and neutral punctuation, which is placed in logical order from the right.

Error marking. An exact Levenshtein alignment is computed on the logical strings (line breaks
included). In the prediction, substituted and inserted characters are drawn in red; a deleted
ground-truth character is marked by a small red caret at the position where it is missing. In
the ground-truth panel the characters that are substituted or deleted are shaded.

    python tools/qualitative_khatt.py \
        --dump outputs/khatt_predictions/khatt_paragraph__KHATT_paragraph__test.json \
        --out experiments/qualitative/khatt --png-dir /tmp/khatt_png
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from hand.basic.metric_manager import format_string_for_cer, format_string_for_wer, \
    edit_wer_from_formatted_split_text  # noqa: E402
import editdistance  # noqa: E402

try:
    import arabic_reshaper
    from fontTools.ttLib import TTFont
except ImportError as e:  # pragma: no cover
    raise SystemExit("arabic_reshaper and fontTools are required: {}".format(e))

# --------------------------------------------------------------------------- fonts
FONT_CANDIDATES = [
    os.path.expanduser("~/.fonts/Amiri-Regular.ttf"),
    "/usr/share/fonts/truetype/amiri/Amiri-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
SANS = "DejaVu Sans"

ERR_FG = "#c81e1e"       # substitutions / insertions in the prediction
GT_BG = "#ffe3b3"        # ground-truth characters that are substituted or deleted
DEL_FG = "#c81e1e"       # deletion caret
LINE_PITCH = 1.55        # line pitch in em

LAM_INIT, LAM_MED = "ﻟ", "ﻠ"
# alef final form -> (isolated lam-alef ligature, final lam-alef ligature)
ALEF_FINAL = {
    "ﺎ": ("ﻻ", "ﻼ"),   # alef
    "ﺂ": ("ﻵ", "ﻶ"),   # alef with madda above
    "ﺄ": ("ﻷ", "ﻸ"),   # alef with hamza above
    "ﺈ": ("ﻹ", "ﻺ"),   # alef with hamza below
}
HARAKAT = set("ًٌٍَُِّْ")

# visually similar letter groups (dots / shared skeleton) used to describe substitutions
SIMILAR = [
    set("بتثنيىئ"), set("جحخ"), set("رز"), set("دذ"), set("سش"), set("صض"), set("طظ"),
    set("عغ"), set("فق"), set("اأإآ"), set("هة"), set("وؤ"), set("ءئؤأإ"),
]


class ArabicFont:
    def __init__(self, path):
        self.path = path
        fm.fontManager.addfont(path)
        self.prop = fm.FontProperties(fname=path)
        tt = TTFont(path)
        self.cmap = tt.getBestCmap()
        self.hmtx = tt["hmtx"]
        self.upem = tt["head"].unitsPerEm
        self.reshaper = arabic_reshaper.ArabicReshaper({
            "delete_harakat": False, "support_ligatures": False, "delete_tatweel": False})

    def advance(self, g, size):
        cp = ord(g)
        if cp not in self.cmap:
            return 0.5 * size
        return self.hmtx[self.cmap[cp]][0] * size / self.upem

    def shape(self, logical):
        """List of (glyph, [logical indices]) in logical order, lam-alef merged."""
        shaped = self.reshaper.reshape(logical)
        if len(shaped) != len(logical):
            raise RuntimeError("reshaper changed the length: {!r}".format(logical))
        out, i = [], 0
        while i < len(shaped):
            g = shaped[i]
            if g in (LAM_INIT, LAM_MED) and i + 1 < len(shaped) and shaped[i + 1] in ALEF_FINAL:
                out.append((ALEF_FINAL[shaped[i + 1]][0 if g == LAM_INIT else 1], [i, i + 1]))
                i += 2
            else:
                out.append((g, [i]))
                i += 1
        return out

    def width(self, logical, size):
        return sum(self.advance(g, size) for g, _ in self.shape(logical) if g not in HARAKAT)


def pick_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ArabicFont(p)
    raise SystemExit("no Arabic-capable font found")


# --------------------------------------------------------------------------- strings
def to_logical(stored):
    """KHATT strings are stored per line in reversed (visual LTR) order."""
    return "\n".join(line[::-1] for line in stored.split("\n"))


def align(gt, pred):
    """Levenshtein alignment. Returns per-pred-char ops ('M','S','I'), per-gt-char ops
    ('M','S','D') and the list of pred positions (before index i) where a gt char is deleted."""
    n, m = len(gt), len(pred)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        gi = gt[i - 1]
        row, prev = D[i], D[i - 1]
        for j in range(1, m + 1):
            c = 0 if gi == pred[j - 1] else 1
            row[j] = min(prev[j - 1] + c, prev[j] + 1, row[j - 1] + 1)
    pred_ops, gt_ops, dels = ["?"] * m, ["?"] * n, []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if gt[i - 1] == pred[j - 1] else 1):
            op = "M" if gt[i - 1] == pred[j - 1] else "S"
            pred_ops[j - 1] = op
            gt_ops[i - 1] = op
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            gt_ops[i - 1] = "D"
            dels.append(j)
            i -= 1
        else:
            pred_ops[j - 1] = "I"
            j -= 1
    return pred_ops, gt_ops, dels, int(D[n, m])


def sample_metrics(gt_stored, pred_stored):
    g, p = format_string_for_cer(gt_stored, None), format_string_for_cer(pred_stored, None)
    ed, nb = editdistance.eval(g, p), len(g)
    wg, wp = format_string_for_wer(gt_stored, None), format_string_for_wer(pred_stored, None)
    we = edit_wer_from_formatted_split_text(wg, wp)
    return dict(edit_chars=ed, nb_chars=nb, cer=ed / nb, edit_words=we, nb_words=len(wg),
                wer=we / len(wg), n_lines_gt=g.count("\n") + 1, n_lines_pred=p.count("\n") + 1,
                len_pred=len(p))


def error_profile(gt_log, pred_log):
    pred_ops, gt_ops, dels, _ = align(gt_log, pred_log)
    subs = [(gt_log[i], pred_log[j]) for i, j in sub_pairs(gt_ops, pred_ops)]
    similar = sum(1 for a, b in subs if any(a in s and b in s for s in SIMILAR))
    return dict(n_sub=sum(o == "S" for o in pred_ops), n_ins=sum(o == "I" for o in pred_ops),
                n_del=len(dels), n_sub_similar=similar, subs=subs,
                n_newline_del=sum(1 for i, o in enumerate(gt_ops) if o == "D" and gt_log[i] == "\n"),
                n_newline_ins=sum(1 for j, o in enumerate(pred_ops) if o == "I" and pred_log[j] == "\n"),
                n_haraka_err=sum(1 for i, o in enumerate(gt_ops) if o != "M" and gt_log[i] in HARAKAT)
                + sum(1 for j, o in enumerate(pred_ops) if o == "I" and pred_log[j] in HARAKAT))


def sub_pairs(gt_ops, pred_ops):
    gi = [i for i, o in enumerate(gt_ops) if o in "MS"]
    pj = [j for j, o in enumerate(pred_ops) if o in "MS"]
    assert len(gi) == len(pj)
    return [(i, j) for i, j in zip(gi, pj) if gt_ops[i] == "S"]


# --------------------------------------------------------------------------- drawing
def layout_rows(font, logical, size, panel_w):
    """Split the logical text into lines and wrap each line into visual rows that fit panel_w.
    Returns [(a, b, rows)] where rows is a list of [(glyph, [indices])] (indices relative to a)."""
    lines, start = [], 0
    for k, ch in enumerate(logical + "\n"):
        if ch == "\n":
            lines.append((start, k))
            start = k + 1
    out = []
    for (a, b) in lines:
        seg = logical[a:b]
        glyphs = font.shape(seg) if seg else []
        rows, cur, cur_w = [], [], 0.0
        for g, idx in glyphs:
            adv = 0.0 if g in HARAKAT else font.advance(g, size)
            if cur and cur_w + adv > panel_w and g != " ":
                rows.append(cur)
                cur, cur_w = [], 0.0
            cur.append((g, idx))
            cur_w += adv
        rows.append(cur)
        out.append((a, b, rows))
    return out


def n_rows(font, logical, size, panel_w):
    return sum(len(rows) for _, _, rows in layout_rows(font, logical, size, panel_w))


def draw_text_panel(ax, font, logical, ops, dels, size, x_right, y_top, panel_w, is_pred):
    """Draw logical text line by line, right-aligned at x_right (points). ops: per-char op.
    dels: pred positions (before char index) of deleted gt chars (prediction panel only)."""
    dels_set = {}
    for d in dels:
        dels_set[d] = dels_set.get(d, 0) + 1
    y = y_top
    pitch = LINE_PITCH * size
    for (a, b, rows) in layout_rows(font, logical, size, panel_w):
        for r_i, row in enumerate(rows):
            x = x_right if r_i == 0 else x_right - 1.2 * size
            if r_i > 0:   # continuation hook for wrapped rows
                ax.text(x_right - 0.3 * size, y, "↳", fontsize=size * 0.8, color="#888888",
                        ha="right", va="baseline", family=SANS)
            prev_x = x
            for g, idx in row:
                li = a + idx[0]
                op = ops[li] if li < len(ops) else "M"
                color = ERR_FG if (is_pred and op in "SI") else "black"
                if g in HARAKAT:
                    gx = prev_x
                else:
                    adv = font.advance(g, size)
                    x -= adv
                    gx = prev_x = x
                    if not is_pred and op in "SD":
                        ax.add_patch(Rectangle((x, y - 0.35 * size), adv, 1.05 * size,
                                               facecolor=GT_BG, edgecolor="none", zorder=0))
                    for q in idx:
                        if is_pred and (a + q) in dels_set:
                            draw_caret(ax, x + adv, y, size, count=dels_set[a + q], x_right=x_right)
                ax.text(gx, y, g, fontproperties=font.prop, fontsize=size, color=color,
                        ha="left", va="baseline", zorder=2)
            if is_pred and r_i == len(rows) - 1:
                # deletions located at the line break (dropped line / merged lines)
                if b in dels_set:
                    draw_caret(ax, x, y, size, count=dels_set[b], x_right=x_right)
                if b < len(ops) and ops[b] == "I" and logical[b] == "\n":
                    ax.text(x - 0.2 * size, y, "¶", fontsize=size * 0.8, color=ERR_FG,
                            ha="right", va="baseline", family=SANS)
            y -= pitch
    return y_top - y - pitch + 0.6 * size


def draw_legend(ax, x0, y, width):
    """Legend row explaining every mark (same wording as the IAM figure)."""
    fs, size = 6.0, 7.0
    x = x0
    ax.text(x, y, "substituted / inserted character", fontsize=fs, family=SANS, color=ERR_FG,
            ha="left", va="baseline")
    x += 0.27 * width
    draw_caret(ax, x, y + 0.45 * size, size)
    ax.text(x + 1.2 * size, y, "\u00d7n: n deleted reference characters", fontsize=fs, family=SANS,
            ha="left", va="baseline")
    x += 0.28 * width
    ax.add_patch(Rectangle((x, y - 0.15 * size), 0.9 * size, 0.95 * size, facecolor=GT_BG,
                           edgecolor="none"))
    ax.text(x + 1.3 * size, y, "reference character not reproduced", fontsize=fs, family=SANS,
            ha="left", va="baseline")
    x += 0.30 * width
    ax.text(x, y, "\u00b6", fontsize=fs, family=SANS, color=ERR_FG, ha="left", va="baseline")
    ax.text(x + 1.0 * size, y, "inserted line break", fontsize=fs, family=SANS, ha="left", va="baseline")


def draw_caret(ax, x, y, size, count=1, x_right=None):
    """Small red caret below the baseline at x; a run of several deleted characters is one
    caret with the count written next to it (inside the panel)."""
    s = 0.32 * size
    ax.plot([x - s / 2, x, x + s / 2], [y - 0.58 * size, y - 0.18 * size, y - 0.58 * size],
            color=DEL_FG, lw=0.7, solid_capstyle="round", zorder=3)
    if count > 1:
        left = x_right is not None and x + 1.5 * size > x_right
        ax.text(x - 0.6 * s if left else x + 0.6 * s, y - 0.62 * size, "\u00d7{}".format(count),
                fontsize=0.6 * size, color=DEL_FG, ha="right" if left else "left", va="baseline",
                family=SANS, zorder=3)


def fit_size(font, texts, panel_w, max_size=8.5, min_size=7.0):
    longest = 0.0
    for t in texts:
        for line in t.split("\n"):
            longest = max(longest, font.width(line, 10.0))
    if longest == 0:
        return max_size
    return max(min_size, min(max_size, 10.0 * panel_w / longest))


def render(examples, font, out_pdf, png_path=None, page_w_cm=18.5, labels=("a", "b", "c")):
    """One example per row: the input image on the left, the ground truth above the prediction
    on the right (the text column is wide enough that KHATT lines do not need wrapping)."""
    pt = 72.0 / 2.54
    page_w = page_w_cm * pt
    gap, margin = 0.35 * pt, 0.15 * pt
    img_w = 5.8 * pt
    txt_w = page_w - 2 * margin - img_w - gap
    header_h = 10.0
    rows = []
    for ex in examples:
        size = fit_size(font, [ex["gt_logical"], ex["pred_logical"]], txt_w, max_size=8.0, min_size=6.5)
        n_gt = n_rows(font, ex["gt_logical"], size, txt_w)
        n_pr = n_rows(font, ex["pred_logical"], size, txt_w)
        txt_h = (n_gt + n_pr) * LINE_PITCH * size + 2 * header_h + 0.6 * size
        im = ex["image"]
        img_h = img_w * im.height / im.width
        rows.append(dict(size=size, n_gt=n_gt, h=max(txt_h, img_h + header_h) + 8, img_h=img_h))
    legend_h = 13.0
    page_h = sum(r["h"] for r in rows) + 2 * margin + legend_h
    fig = plt.figure(figsize=(page_w / 72, page_h / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.axis("off")
    y = page_h - margin
    for ex, row, lab in zip(examples, rows, labels):
        size = row["size"]
        x0 = margin
        xt = x0 + img_w + gap           # left edge of the text column
        hy = y - 7
        if lab:
            ax.text(x0, hy, "({}) input".format(lab), fontsize=6.5, family=SANS, ha="left", va="baseline")
        else:
            ax.text(x0, hy, "KHATT (Arabic), representative paragraph", fontsize=6.5, family=SANS,
                    ha="left", va="baseline")
        ax.text(xt, hy, "reference", fontsize=6.5, family=SANS, ha="left", va="baseline")
        top = y - header_h
        im = ex["image"]
        ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255,
                  extent=[x0, x0 + img_w, top - row["img_h"], top], interpolation="lanczos",
                  aspect="auto", zorder=1)
        ax.add_patch(Rectangle((x0, top - row["img_h"]), img_w, row["img_h"], fill=False,
                               edgecolor="#bbbbbb", lw=0.4, zorder=2))
        ytxt = top - 0.95 * size
        used = draw_text_panel(ax, font, ex["gt_logical"], ex["gt_ops"], [], size,
                               xt + txt_w, ytxt, txt_w, is_pred=False)
        hy2 = ytxt - used - 2
        ax.text(xt, hy2, "HAND prediction   CER {:.1f}%  ({:.1f}% in reading order)   WER {:.1f}%".format(
            100 * ex["cer"], 100 * ex["cer_logical"], 100 * ex["wer"]),
            fontsize=6.5, family=SANS, ha="left", va="baseline")
        draw_text_panel(ax, font, ex["pred_logical"], ex["pred_ops"], ex["dels"], size,
                        xt + txt_w, hy2 - 0.95 * size - 3, txt_w, is_pred=True)
        y -= row["h"]
        ax.plot([margin, page_w - margin], [y + 3, y + 3], color="#dddddd", lw=0.4)
    draw_legend(ax, margin, y - 4, page_w - 2 * margin)
    fig.savefig(out_pdf, format="pdf")
    if png_path:
        fig.savefig(png_path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- selection
def select(samples, corpus_cer):
    """(1) CER closest to the corpus value, (2) hardest 'similar-glyph' case with all lines
    recovered, (3) worst CER."""
    ok = [s for s in samples]
    rep = min(ok, key=lambda s: abs(s["cer"] - corpus_cer))
    hard_pool = [s for s in ok if 0.25 <= s["cer"] <= 0.45 and s["n_lines_pred"] == s["n_lines_gt"]
                 and s["n_newline_del"] == 0 and s["n_newline_ins"] == 0]
    hard = max(hard_pool, key=lambda s: s["n_sub_similar"] / max(1, s["edit_chars"]) + 0.0 * s["cer"])
    fail = max(ok, key=lambda s: s["cer"])
    return [rep, hard, fail]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=os.path.join(REPO, "outputs", "khatt_predictions",
                                                   "khatt_paragraph__KHATT_paragraph__test.json"))
    ap.add_argument("--data", default=os.path.join(REPO, "formatted", "KHATT_paragraph"))
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "source", "figures", "qualitative", "khatt"))
    ap.add_argument("--names", nargs="*", default=None,
                    help="three sample names (representative, difficult, failure); default: automatic")
    ap.add_argument("--representative", default=None, help="name used for khatt_representative.pdf")
    ap.add_argument("--png-dir", default=None, help="also write 300 dpi PNG previews here")
    ap.add_argument("--page-width-cm", type=float, default=18.5)
    a = ap.parse_args()

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.family"] = SANS
    font = pick_font()
    print("font:", font.path)

    with open(a.dump, encoding="utf-8") as f:
        dump = json.load(f)
    split = dump["split"]
    samples = []
    for s in dump["samples"]:
        m = sample_metrics(s["ground_truth"], s["prediction"])
        gt_log = to_logical(format_string_for_cer(s["ground_truth"], None))
        pr_log = to_logical(format_string_for_cer(s["prediction"], None))
        m.update(error_profile(gt_log, pr_log))
        # CER of the same strings in reading order: the paper metric compares the stored
        # per-line-reversed strings, so a shifted line break costs many more edits than in
        # logical order.
        m["cer_logical"] = editdistance.eval(gt_log, pr_log) / m["nb_chars"]
        m.update(name=os.path.basename(s["name"]), gt_stored=s["ground_truth"], pred_stored=s["prediction"],
                 gt_logical=gt_log, pred_logical=pr_log)
        samples.append(m)
    tot_ed, tot_nb = sum(s["edit_chars"] for s in samples), sum(s["nb_chars"] for s in samples)
    tot_we, tot_nw = sum(s["edit_words"] for s in samples), sum(s["nb_words"] for s in samples)
    corpus_cer, corpus_wer = tot_ed / tot_nb, tot_we / tot_nw
    corpus_cer_logical = sum(s["cer_logical"] * s["nb_chars"] for s in samples) / tot_nb
    cers = sorted(s["cer"] for s in samples)
    stats = dict(n=len(samples), corpus_cer=corpus_cer, corpus_wer=corpus_wer,
                 corpus_cer_logical_order=corpus_cer_logical,
                 manager_cer=dump["metrics_from_manager"].get("cer"),
                 manager_wer=dump["metrics_from_manager"].get("wer"),
                 min=cers[0], median=statistics.median(cers), mean=statistics.mean(cers), max=cers[-1],
                 n_above_40=sum(c > 0.40 for c in cers), n_above_30=sum(c > 0.30 for c in cers),
                 n_below_10=sum(c < 0.10 for c in cers),
                 n_line_count_mismatch=sum(s["n_lines_pred"] != s["n_lines_gt"] for s in samples),
                 total_sub=sum(s["n_sub"] for s in samples), total_ins=sum(s["n_ins"] for s in samples),
                 total_del=sum(s["n_del"] for s in samples),
                 total_sub_similar=sum(s["n_sub_similar"] for s in samples),
                 total_haraka_err=sum(s["n_haraka_err"] for s in samples),
                 total_newline_del=sum(s["n_newline_del"] for s in samples),
                 total_newline_ins=sum(s["n_newline_ins"] for s in samples))
    print(json.dumps(stats, indent=1))

    by_name = {s["name"]: s for s in samples}
    chosen = [by_name[n] for n in a.names] if a.names else select(samples, corpus_cer)
    os.makedirs(a.out, exist_ok=True)
    if a.png_dir:
        os.makedirs(a.png_dir, exist_ok=True)
    examples = []
    for s in chosen:
        pred_ops, gt_ops, dels, ed = align(s["gt_logical"], s["pred_logical"])
        im = Image.open(os.path.join(a.data, split, s["name"])).convert("L")
        ex = dict(s)
        ex.update(image=im, pred_ops=pred_ops, gt_ops=gt_ops, dels=dels, edit_logical=ed)
        examples.append(ex)
        rec = {k: v for k, v in s.items() if k not in ("subs",)}
        rec["subs_gt_pred"] = ["{}>{}".format(x, y) for x, y in s["subs"]]
        rec.update(model=dump["model"], checkpoint=dump["checkpoint"], dataset=dump["dataset"],
                   split=split, image=os.path.join("formatted", "KHATT_paragraph", split, s["name"]),
                   note="gt_stored/pred_stored are the strings as stored in labels.pkl / produced by the "
                        "model (per-line reversed, visual LTR order); *_logical are reversed back per line.")
        with open(os.path.join(a.out, "prediction_{}.json".format(s["name"].replace(".png", ""))),
                  "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print("{}: CER {:.4f} (reading order {:.4f}) WER {:.4f} sub {} (similar {}) ins {} del {} "
              "lines gt/pred {}/{} newline del/ins {}/{} haraka err {}".format(
                  s["name"], s["cer"], s["cer_logical"], s["wer"], s["n_sub"], s["n_sub_similar"],
                  s["n_ins"], s["n_del"],
                  s["n_lines_gt"], s["n_lines_pred"], s["n_newline_del"], s["n_newline_ins"], s["n_haraka_err"]))
    with open(os.path.join(a.out, "cer_distribution_{}.json".format(split)), "w") as f:
        json.dump(dict(stats=stats, per_sample={s["name"]: round(s["cer"], 4) for s in samples},
                       chosen=[s["name"] for s in chosen]), f, indent=1)

    render(examples, font, os.path.join(a.out, "khatt_examples.pdf"),
           os.path.join(a.png_dir, "khatt_examples.png") if a.png_dir else None, a.page_width_cm)
    rep = examples[0] if a.representative is None else \
        next(e for e in examples if e["name"] == a.representative)
    render([rep], font, os.path.join(a.out, "khatt_representative.pdf"),
           os.path.join(a.png_dir, "khatt_representative.png") if a.png_dir else None, a.page_width_cm,
           labels=("",))
    print("wrote", os.path.join(a.out, "khatt_examples.pdf"))


if __name__ == "__main__":
    main()
