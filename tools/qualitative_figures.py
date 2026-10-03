#!/usr/bin/env python3
"""Qualitative composite figures: input page with its reading-order path, the ground-truth
token stream and the HAND prediction, with character-level recognition errors and
layout-token errors highlighted.

One composite per example:

    [ page image + reading-order path ] [ ground truth ] [ HAND prediction ]

The reading-order path is drawn from the ground-truth line boxes in the order the label
stream visits them (the five-token scheme: page > {section, page number} > {body,
annotation}). The two text panels render the interleaved token stream; layout tokens are
drawn as small coloured tags and the text is monospaced so that the two panels align
line by line. Errors are an exact Levenshtein alignment between the layout-stripped
strings (insertions and substitutions in red in the prediction, the corresponding
ground-truth characters shaded, deletions marked by a red caret in the prediction).
Layout-token errors are an alignment of the two token sequences; mismatched tags carry a
red border.

Every prediction shown is produced by the released model and written next to the figure
as JSON (`<out>/predictions/<dataset>_<split>_<name>.json`), so that a figure can always be
traced to the exact decode that produced it.

    python tools/qualitative_figures.py \
        --export-dir outputs/export_e14_cpu \
        --data formatted/READ_2016_page_sem_dan --split test \
        --names test_35.jpeg test_10.jpeg \
        --out experiments/qualitative --device cpu

CPU is sufficient: a single page decodes in about ten seconds in fp32.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "release"))
from hand_release.inference import LAYOUT_TOKENS, strip_layout  # noqa: E402

# --------------------------------------------------------------------------- constants
OPEN = {"ⓟ": "P", "ⓝ": "N", "ⓢ": "S", "ⓐ": "A", "ⓑ": "B"}
CLOSE = {"Ⓟ": "P", "Ⓝ": "N", "Ⓢ": "S", "Ⓐ": "A", "Ⓑ": "B"}
TAG_CLASS = {"P": "page", "N": "page_number", "S": "section", "A": "annotation", "B": "body"}
COMBINING = {"\u0304": "\u00af", "\u0308": "\u00a8"}   # combining macron / diaeresis -> spacing glyph

# One hue per layout class, shared by the image overlay and the text tags.
CLASS_COLOR = {
    "page": "#6b6b6b",
    "page_number": "#2a9d4b",
    "section": "#7b3fb3",
    "annotation": "#e07b00",
    "body": "#1f6fd1",
}
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
PATH_CMAP = LinearSegmentedColormap.from_list("path", ["#1a237e", "#00897b", "#c62828"])
ERR_FG = "#c81e1e"
ERR_BG = "#ffd6d6"
GT_BG = "#ffe9c2"
MONO = "DejaVu Sans Mono"
MONO_ADVANCE = 0.602          # DejaVu Sans Mono advance width in em
LINE_SPACING = 1.32           # line pitch in em
TAG_W_OPEN, TAG_W_CLOSE = 2.0, 2.6   # tag widths in character cells


# --------------------------------------------------------------------------- data access
def load_labels(data_dir, split):
    with open(os.path.join(data_dir, "labels.pkl"), "rb") as f:
        return pickle.load(f)["ground_truth"][split]


def reading_order(sample):
    """Line boxes in the order the label stream visits them.

    Returns a list of dicts: {page, section, cls, text, top, bottom, left, right}. The
    paragraphs of a page are stored in an arbitrary order in the label file; the order
    that matters is the one of the page text, so each paragraph is located by its label
    inside the page text and the paragraphs are sorted by that position. A paragraph
    whose label starts with the section-open token begins a new section.
    """
    out = []
    for pi, page in enumerate(sample["pages"]):
        text = page["text"]
        paras = []
        placed = []
        for para in page["paragraphs"]:
            lab = para["label"]
            k = text.find(lab)
            while k >= 0 and k in placed:
                k = text.find(lab, k + 1)
            if k < 0:                       # fall back on the stripped text
                k = strip_layout(text).find(strip_layout(lab))
            placed.append(k)
            paras.append((k, para))
        paras.sort(key=lambda t: t[0])
        sec = -1
        for _, para in paras:
            if para["label"].startswith("ⓢ"):
                sec += 1
            for ln in para["lines"]:
                out.append({"page": pi, "section": sec if para["mode"] != "page_number" else -1,
                            "cls": para["mode"], "text": ln["text"], "top": ln["top"],
                            "bottom": ln["bottom"], "left": ln["left"], "right": ln["right"]})
    return out


# --------------------------------------------------------------------------- alignment
def levenshtein_align(a, b):
    """Exact edit-distance alignment of two sequences.

    Returns (distance, ops) where ops is a list of (tag, i, j): 'eq' and 'sub' consume
    a[i] and b[j]; 'del' consumes a[i] only (missing from b); 'ins' consumes b[j] only.
    """
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = D[i], D[i - 1]
        for j in range(1, m + 1):
            c = 0 if ai == b[j - 1] else 1
            row[j] = min(prev[j - 1] + c, prev[j] + 1, row[j - 1] + 1)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            ops.append(("eq" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return int(D[n, m]), ops


def char_error_marks(gt_text, pred_text):
    """Marks over the stripped strings: per-character status and caret positions."""
    dist, ops = levenshtein_align(gt_text, pred_text)
    gt_mark = ["eq"] * len(gt_text)
    pr_mark = ["eq"] * len(pred_text)
    carets = set()             # indices into pred_text before which a deletion happened
    jp = 0                     # prediction characters consumed so far
    for tag, i, j in ops:
        if tag == "sub":
            gt_mark[i], pr_mark[j] = "sub", "sub"
            jp = j + 1
        elif tag == "eq":
            jp = j + 1
        elif tag == "del":
            gt_mark[i] = "del"
            carets.add(jp)
        elif tag == "ins":
            pr_mark[j] = "ins"
            jp = j + 1
    return dist, gt_mark, pr_mark, carets


def layout_tokens(raw):
    return [c for c in raw if c in LAYOUT_TOKENS]


def anchored_align(a, b, big=10 ** 6):
    """Edit-distance alignment of two interleaved streams in which a layout token may only be
    paired with a layout token and a character with a character. The text anchors the tokens,
    so a missing or spurious region boundary is marked where it occurs in the page rather than
    at an arbitrary position of a periodic token sequence."""
    n, m = len(a), len(b)
    ta = np.array([c in LAYOUT_TOKENS for c in a], dtype=bool)
    tb = np.array([c in LAYOUT_TOKENS for c in b], dtype=bool)
    D = np.zeros((n + 1, m + 1), dtype=np.int64)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ai, ki = a[i - 1], ta[i - 1]
        row, prev = D[i], D[i - 1]
        for j in range(1, m + 1):
            c = 0 if ai == b[j - 1] else (1 if ki == tb[j - 1] else big)
            row[j] = min(prev[j - 1] + c, prev[j] + 1, row[j - 1] + 1)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = 0 if a[i - 1] == b[j - 1] else (1 if ta[i - 1] == tb[j - 1] else big)
            if D[i, j] == D[i - 1, j - 1] + c:
                ops.append(("eq" if c == 0 else "sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def layout_error_marks(gt_raw, pred_raw):
    """Token-level edit distance (the reported count) and per-token error marks located by the
    text-anchored alignment of the two streams."""
    g, p = layout_tokens(gt_raw), layout_tokens(pred_raw)
    dist, _ = levenshtein_align(g, p)
    g_bad = [False] * len(g)
    p_bad = [False] * len(p)
    errors = []
    gi = pi = 0
    for tag, i, j in anchored_align(gt_raw, pred_raw):
        ta = i is not None and gt_raw[i] in LAYOUT_TOKENS
        tb = j is not None and pred_raw[j] in LAYOUT_TOKENS
        if tag in ("eq", "sub"):
            if ta and tb:
                if tag == "sub":
                    g_bad[gi] = p_bad[pi] = True
                    errors.append("substitution %s -> %s" % (g[gi], p[pi]))
                gi += 1
                pi += 1
        elif tag == "del" and ta:
            g_bad[gi] = True
            errors.append("missing %s" % g[gi])
            gi += 1
        elif tag == "ins" and tb:
            p_bad[pi] = True
            errors.append("spurious %s" % p[pi])
            pi += 1
    assert gi == len(g) and pi == len(p)
    return dist, g_bad, p_bad, errors


# --------------------------------------------------------------------------- text panels
def tokenize_stream(raw, char_marks, tag_bad, carets):
    """Interleaved stream -> list of visual lines, each a list of items.

    item = ("char", c, mark) | ("tag", label, is_close, bad) | ("caret",)
    """
    lines, cur = [], []
    ci, ti = 0, 0
    for c in raw:
        if c in OPEN or c in CLOSE:
            if c == "ⓟ" and cur:
                lines.append(cur)
                cur = []
            is_close = c in CLOSE
            lab = OPEN.get(c) or CLOSE.get(c)
            cur.append(("tag", lab, is_close, tag_bad[ti] if ti < len(tag_bad) else False))
            ti += 1
            continue
        if ci in carets:
            cur.append(("caret",))
        if c == "\n":
            lines.append(cur)
            cur = []
        else:
            cur.append(("char", c, char_marks[ci] if ci < len(char_marks) else "eq"))
        ci += 1
    if ci in carets:
        cur.append(("caret",))
    lines.append(cur)
    return lines


def item_width(it):
    if it[0] == "char":
        return 0.0 if it[1] in COMBINING else 1.0
    if it[0] == "tag":
        return TAG_W_CLOSE if it[2] else TAG_W_OPEN
    return 0.0


def wrap_lines(lines, max_cells):
    """Wrap visual lines at `max_cells`, preferring a break at the last space."""
    out = []
    for ln in lines:
        cur, w, first = [], 0.0, True
        for it in ln:
            iw = item_width(it)
            if w + iw > max_cells and cur:
                # back up to the last space if that keeps most of the line
                k = max((i for i, c in enumerate(cur) if c[0] == "char" and c[1] == " "), default=-1)
                if k > 0 and sum(item_width(c) for c in cur[:k]) >= 0.55 * max_cells:
                    head, tail = cur[:k], cur[k + 1:]
                else:
                    head, tail = cur, []
                out.append((head, not first))
                cur, first = tail, False
                w = 1.5 + sum(item_width(c) for c in tail)      # indent continuation
            cur.append(it)
            w += iw
        out.append((cur, not first))
    return out


def draw_text_panel(ax, lines, fs, bg_for, title=None, max_cells=40, title_fs=None, n_rows=None):
    """Render wrapped lines into `ax`, whose data units are character cells (x) and
    text lines (y, downwards). `n_rows` fixes the vertical extent so that panels sharing a
    row keep the same line pitch whatever their length."""
    ax.set_axis_off()
    wrapped = wrap_lines(lines, max_cells)
    ax.set_xlim(-0.3, max_cells + 0.3)
    ax.set_ylim((n_rows or len(wrapped)) + 0.4, -0.9)
    for yi, (ln, is_cont) in enumerate(wrapped):
        x = 1.5 if is_cont else 0.0
        if is_cont:
            ax.text(0.2, yi, "↳", fontsize=fs * 0.9, family=MONO, color="#888888",
                    ha="left", va="center")
        run, run_mark = [], None

        def flush():
            nonlocal run, run_mark, x
            if not run:
                return
            s = "".join(run)
            ncell = sum(0.0 if ch in COMBINING else 1.0 for ch in run)
            kw = {}
            bg = bg_for(run_mark)
            if bg is not None:
                kw["bbox"] = dict(boxstyle="square,pad=0.08", fc=bg, ec="none")
            ax.text(x, yi, s, fontsize=fs, family=MONO, ha="left", va="center",
                    color=ERR_FG if run_mark in ("sub", "ins") else "#111111",
                    fontweight="bold" if run_mark in ("sub", "ins") else "normal", **kw)
            x += ncell
            run, run_mark = [], None

        for it in ln:
            if it[0] == "char":
                if run and it[2] != run_mark:
                    flush()
                run.append(it[1])
                run_mark = it[2]
                continue
            flush()
            if it[0] == "tag":
                lab, is_close, bad = it[1], it[2], it[3]
                w = TAG_W_CLOSE if is_close else TAG_W_OPEN
                col = CLASS_COLOR[TAG_CLASS[lab]]
                ax.add_patch(Rectangle((x + 0.12, yi - 0.42), w - 0.24, 0.84,
                                       fc=col if not is_close else "white",
                                       ec=ERR_FG if bad else col,
                                       lw=1.3 if bad else 0.6, zorder=2,
                                       joinstyle="round"))
                ax.text(x + w / 2, yi, ("/" if is_close else "") + lab, fontsize=fs * 0.78,
                        family=MONO, ha="center", va="center", zorder=3,
                        color="white" if not is_close else col, fontweight="bold")
                x += w
            elif it[0] == "caret":
                ax.text(x, yi + 0.42, "▲", fontsize=fs * 0.6, color=ERR_FG, ha="center",
                        va="center", zorder=4)
        flush()
    if title:
        ax.text(0, -0.75, title, fontsize=title_fs or fs + 1.5, family="DejaVu Sans", ha="left",
                va="center", fontweight="bold")
    return len(wrapped)


# --------------------------------------------------------------------------- image panel
def draw_image_panel(ax, img, order, sample, title=None, number_every=1, label_fs=4.2,
                     page_labels=None, end_after_page=None, undecoded_pages=(), page_notes=None,
                     page_number_in_label=False, section_labels=True):
    """Input image with the reference line boxes, the region outlines and the reading-order path.

    Multi-page options (all off by default): `page_labels` writes "page k" and the written page
    number above each page outline; `end_after_page=k` draws an end-of-sequence marker after the
    last line of page k (0-based); `undecoded_pages` greys out pages the decoder never produced;
    `page_notes` maps a page index to a short string appended to its label.
    """
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    H, W = img.shape[:2]
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    # page outlines
    for page in sample["pages"]:
        ax.add_patch(Rectangle((page["left"], page["top"]), page["right"] - page["left"],
                               page["bottom"] - page["top"], fill=False,
                               ec=CLASS_COLOR["page"], lw=0.7, ls=(0, (4, 2)), zorder=2))
    # section outlines: union of the line boxes of each (page, section)
    secs = {}
    for ln in order:
        if ln["section"] < 0:
            continue
        k = (ln["page"], ln["section"])
        b = secs.setdefault(k, [ln["left"], ln["top"], ln["right"], ln["bottom"]])
        b[0], b[1] = min(b[0], ln["left"]), min(b[1], ln["top"])
        b[2], b[3] = max(b[2], ln["right"]), max(b[3], ln["bottom"])
    pad = 0.012 * W
    for (pi, si), (l, t, r, b) in secs.items():
        ax.add_patch(Rectangle((l - pad, t - pad), r - l + 2 * pad, b - t + 2 * pad, fill=False,
                               ec=CLASS_COLOR["section"], lw=0.8, ls=(0, (2, 1.5)), zorder=3))
        if not section_labels:
            continue
        ax.text(l - pad, t - pad - 0.004 * H, "section %d" % (si + 1), fontsize=label_fs,
                color=CLASS_COLOR["section"], ha="left", va="bottom", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    # line boxes
    for ln in order:
        col = CLASS_COLOR[ln["cls"]]
        ax.add_patch(Rectangle((ln["left"], ln["top"]), ln["right"] - ln["left"],
                               ln["bottom"] - ln["top"], fc=col, ec=col, alpha=0.16, lw=0.0,
                               zorder=1))
        ax.add_patch(Rectangle((ln["left"], ln["top"]), ln["right"] - ln["left"],
                               ln["bottom"] - ln["top"], fill=False, ec=col, lw=0.35, zorder=4))
    # reading-order path through the line centres, coloured by progression
    n = len(order)
    cmap = PATH_CMAP
    cx = [0.5 * (ln["left"] + ln["right"]) for ln in order]
    cy = [0.5 * (ln["top"] + ln["bottom"]) for ln in order]
    for i in range(n - 1):
        col = cmap(0.1 + 0.8 * i / max(1, n - 2))
        jump = order[i + 1]["cls"] != order[i]["cls"] or order[i + 1]["section"] != order[i]["section"]
        page_jump = order[i + 1]["page"] != order[i]["page"]
        if page_jump and page_labels:
            # the transition between two pages is the event the multi-page figures are about:
            # a heavier dashed arc, labelled at its apex; greyed out when the decoder never
            # reached the next page
            rad = 0.25
            taken = order[i + 1]["page"] not in undecoded_pages
            col_t = "#111111" if taken else "#9a9a9a"
            arr = FancyArrowPatch((cx[i], cy[i]), (cx[i + 1], cy[i + 1]), arrowstyle="-|>",
                                  mutation_scale=7, lw=1.2 if taken else 0.9, color=col_t,
                                  ls=(0, (4, 2)), connectionstyle="arc3,rad=%.2f" % rad,
                                  shrinkA=2, shrinkB=2, zorder=8)
            ax.add_patch(arr)
            # label in the gutter between the two pages, at the height of the chord midpoint
            pa, pb = sample["pages"][order[i]["page"]], sample["pages"][order[i + 1]["page"]]
            apx = 0.5 * (pa["right"] + pb["left"])
            apy = 0.5 * (cy[i] + cy[i + 1])
            lab = "page %d \u2192 page %d" % (order[i]["page"] + 1, order[i + 1]["page"] + 1)
            if not taken:
                lab += ": not taken"
            ax.text(apx, apy, lab, fontsize=label_fs + 0.6, ha="center", va="center",
                    color="#111111" if taken else "#8a1c1c", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col_t, lw=0.5))
            continue
        arr = FancyArrowPatch((cx[i], cy[i]), (cx[i + 1], cy[i + 1]), arrowstyle="-|>",
                              mutation_scale=5, lw=0.9 if not jump else 1.1, color=col,
                              connectionstyle="arc3,rad=%.2f" % (0.25 if jump else 0.0),
                              shrinkA=1.5, shrinkB=1.5, zorder=6, alpha=0.95)
        ax.add_patch(arr)
    for i, ln in enumerate(order):
        if number_every <= 0 or i % number_every:
            continue
        col = cmap(0.1 + 0.8 * i / max(1, n - 1))
        ax.text(ln["left"] - 0.006 * W, cy[i], str(i + 1), fontsize=label_fs, ha="right",
                va="center", color="white", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.18", fc=col, ec="none"))
    if page_labels:
        for pi, page in enumerate(sample["pages"]):
            num = [strip_layout(p["label"]) for p in page["paragraphs"] if p["mode"] == "page_number"]
            lab = "page %d" % (pi + 1)
            if num and page_number_in_label:
                lab += " (written page number %s)" % num[0]
            if page_notes and pi in page_notes:
                lab += ("  " if page_number_in_label else " \u00b7 ") + page_notes[pi]
            ax.text(page["left"], page["top"] - 0.006 * H, lab, fontsize=label_fs + 1.0,
                    color="#222222", ha="left", va="bottom", zorder=9, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))
    for pi in undecoded_pages:
        page = sample["pages"][pi]
        ax.add_patch(Rectangle((page["left"], page["top"]), page["right"] - page["left"],
                               page["bottom"] - page["top"], fc="white", ec="none", alpha=0.72,
                               zorder=7))
        ax.text(0.5 * (page["left"] + page["right"]), 0.5 * (page["top"] + page["bottom"]),
                "not decoded", fontsize=label_fs + 3.5, ha="center", va="center", color="#8a1c1c",
                fontweight="bold", zorder=9, rotation=0,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#8a1c1c", lw=0.8))
    if end_after_page is not None:
        last = [i for i, ln in enumerate(order) if ln["page"] == end_after_page]
        if last:
            i = last[-1]
            ax.text(order[i]["right"], order[i]["bottom"] + 0.012 * H, "end of sequence",
                    fontsize=label_fs + 0.6, color="white", ha="right", va="top", zorder=9,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#111111", ec="none"))
    if title:
        ax.set_title(title, fontsize=7, loc="left", pad=3, fontweight="bold")


LEGEND_ITEMS = [("page", "page"), ("page_number", "page number"), ("section", "section"),
                ("annotation", "annotation"), ("body", "body")]
LEGEND_PATH = "numbered path: ground-truth reading order;  dashed: page and section outlines"
LEGEND_MULTIPAGE = ("dashed arc: page transition;  black tag: end of sequence;  "
                    "greyed page: not decoded")
LEGEND_ERRORS = [("red: inserted or substituted character", ERR_FG, None, "bold"),
                 ("\u25b2 missing character", ERR_FG, None, "normal"),
                 ("shaded in (b): ground-truth characters the prediction does not reproduce",
                  "#7a5a00", GT_BG, "normal"),
                 ("red border: layout token in error", ERR_FG, None, "normal")]


def draw_legend(ax, fs=5.4, multipage=False):
    """Legend strip across the full figure width: colour chips and path note on the first row,
    error marks on the second, multi-page symbols on a third when `multipage`."""
    ax.set_axis_off()
    rows = [0.84, 0.50, 0.16] if multipage else [0.75, 0.22]
    x = 0.0
    for key, lab in LEGEND_ITEMS:
        ax.add_patch(Rectangle((x, rows[0] - 0.15), 0.022, 0.30, fc=CLASS_COLOR[key], ec="none",
                               transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.028, rows[0], lab, fontsize=fs, va="center", ha="left", transform=ax.transAxes)
        x += 0.028 + 0.0085 * len(lab) + 0.025
    ax.text(x + 0.01, rows[0], LEGEND_PATH, fontsize=fs, va="center", ha="left",
            transform=ax.transAxes, color="#333333")
    xs = [0.0, 0.26, 0.40, 0.815]
    for (text, col, bg, weight), xx in zip(LEGEND_ERRORS, xs):
        kw = dict(bbox=dict(boxstyle="square,pad=0.15", fc=bg, ec="none")) if bg else {}
        ax.text(xx, rows[1], text, fontsize=fs, va="center", ha="left", color=col, fontweight=weight,
                transform=ax.transAxes, **kw)
    if multipage:
        ax.text(0.0, rows[2], LEGEND_MULTIPAGE, fontsize=fs, va="center", ha="left",
                transform=ax.transAxes, color="#333333")


# --------------------------------------------------------------------------- composite
def composite(img, sample, gt_raw, pred_raw, out_base, fs=6.2, width_in=7.28,
              image_frac=0.38, pred_label="HAND prediction", note=None):
    """Build the three-panel figure and save PDF + PNG. Returns a dict of statistics.

    Single-page layout only; double- and triple-page composites are produced by
    tools/qualitative_multipage.py from the multi-page result files."""
    order = reading_order(sample)
    gt_text, pr_text = strip_layout(gt_raw), strip_layout(pred_raw)
    dist, gt_mark, pr_mark, carets = char_error_marks(gt_text, pr_text)
    ldist, g_bad, p_bad, lerrors = layout_error_marks(gt_raw, pred_raw)
    gt_lines = tokenize_stream(gt_raw, gt_mark, g_bad, set())
    pr_lines = tokenize_stream(pred_raw, pr_mark, p_bad, carets)
    stats = {"cer": dist / max(1, len(gt_text)), "edit_distance": dist, "n_gt_chars": len(gt_text),
             "layout_token_edit_distance": ldist, "layout_token_errors": lerrors,
             "n_lines_gt": len(gt_lines), "n_lines_pred": len(pr_lines)}

    H, W = img.shape[:2]
    cell_in = MONO_ADVANCE * fs / 72.0
    line_in = LINE_SPACING * fs / 72.0
    gap = 0.12
    # image | GT | prediction, side by side. The displayed image is cropped vertically to
    # the labelled page box (plus a margin) so that blank paper does not set the scale.
    img, order, sample = crop_to_page(img, order, sample)
    H, W = img.shape[:2]
    text_w_guess = (width_in - image_frac * width_in - 2 * gap) / 2.0
    max_cells = int(text_w_guess / cell_in) - 1
    n_rows = max(len(wrap_lines(gt_lines, max_cells)), len(wrap_lines(pr_lines, max_cells)))
    text_h = (n_rows + 1.6) * line_in
    img_w = min(image_frac * width_in, text_h * W / H)
    img_h = img_w * H / W
    text_w = (width_in - img_w - 2 * gap) / 2.0
    max_cells = int(text_w / cell_in) - 1
    n_rows = max(len(wrap_lines(gt_lines, max_cells)), len(wrap_lines(pr_lines, max_cells)))
    text_h = (n_rows + 1.6) * line_in
    top_pad, bottom_pad = 0.16, 0.30
    fig_h = max(img_h + 0.2, text_h) + top_pad + bottom_pad
    fig = plt.figure(figsize=(width_in, fig_h))
    ax_img = fig.add_axes([0, 1 - (top_pad + img_h + 0.2) / fig_h, img_w / width_in, (img_h + 0.2) / fig_h])
    draw_image_panel(ax_img, img, order, sample, title="(a) input page and reading order",
                     label_fs=4.8)
    panels = [(gt_lines, "(b) ground truth"), (pr_lines, "(c) " + pred_label)]
    for k, (lines, title) in enumerate(panels):
        x = (img_w + gap + k * (text_w + gap)) / width_in
        ax = fig.add_axes([x, 1 - (top_pad + text_h) / fig_h, text_w / width_in, text_h / fig_h])
        draw_text_panel(ax, lines, fs, bg_for=_bg, title=title, max_cells=max_cells)
    ax_leg = fig.add_axes([0.0, 0.0, 1.0, bottom_pad / fig_h])
    draw_legend(ax_leg)
    if note:
        fig.text(0.995, 0.012, note, fontsize=5.2, ha="right", va="bottom", color="#444444")
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=220)
    plt.close(fig)
    stats["figure_size_in"] = [width_in, round(float(fig_h), 2)]
    return stats


def crop_to_page(img, order, sample, margin_frac=0.03, crop_x=False):
    """Crop the image rows (and, with `crop_x`, the columns) to the union of the page boxes,
    shifting the boxes accordingly."""
    H, W = img.shape[:2]
    top = min(p["top"] for p in sample["pages"])
    bottom = max(p["bottom"] for p in sample["pages"])
    m = int(margin_frac * H)
    y0, y1 = max(0, top - m), min(H, bottom + m)
    x0, x1 = 0, W
    if crop_x:
        left = min(p["left"] for p in sample["pages"])
        right = max(p["right"] for p in sample["pages"])
        mx = int(0.4 * margin_frac * W)
        x0, x1 = max(0, left - mx), min(W, right + mx)
    img = img[y0:y1, x0:x1]
    order = [dict(ln, top=ln["top"] - y0, bottom=ln["bottom"] - y0, left=ln["left"] - x0,
                  right=ln["right"] - x0) for ln in order]
    sample = dict(sample, pages=[dict(p, top=p["top"] - y0, bottom=p["bottom"] - y0, left=p["left"] - x0,
                                      right=p["right"] - x0) for p in sample["pages"]])
    return img, order, sample


def _bg(mark):
    return {"sub": ERR_BG, "ins": ERR_BG, "del": GT_BG}.get(mark)


def split_raw_pages(raw):
    """Split an interleaved stream at every page-open token."""
    pages, cur = [], ""
    for c in raw:
        if c == "ⓟ" and cur:
            pages.append(cur)
            cur = ""
        cur += c
    if cur:
        pages.append(cur)
    return pages


# --------------------------------------------------------------------------- inference
def predict(export_dir, image_path, device, max_tokens, max_lines, kv_cache=True):
    import torch
    from hand_release.inference import HANDRecognizer
    torch.set_num_threads(8)
    r = HANDRecognizer.from_pretrained(export_dir, device=device, kv_cache=kv_cache)
    if max_lines:
        r.max_lines = max_lines
    res = r.read(image_path, max_tokens=max_tokens)
    return res.to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", default="outputs/export_e14_cpu")
    ap.add_argument("--data", default="formatted/READ_2016_page_sem_dan")
    ap.add_argument("--split", default="test")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/qualitative")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--max-lines", type=int, default=None,
                    help="override the page model's 100-line stop (needed for multi-page input)")
    ap.add_argument("--predictions", default=None,
                    help="JSON {name: {raw: ...}} with cached decodes; missing names are decoded")
    ap.add_argument("--font-size", type=float, default=6.8)
    ap.add_argument("--width-cm", type=float, default=18.5)
    ap.add_argument("--prefix", default=None, help="output file prefix (default: dataset name)")
    ap.add_argument("--note", default=None, help="small annotation printed on the figure")
    a = ap.parse_args()

    data_dir = a.data if os.path.isabs(a.data) else os.path.join(REPO, a.data)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    export_dir = a.export_dir if os.path.isabs(a.export_dir) else os.path.join(REPO, a.export_dir)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    gt = load_labels(data_dir, a.split)
    cached = json.load(open(a.predictions, encoding="utf-8")) if a.predictions else {}
    prefix = a.prefix or os.path.basename(os.path.normpath(data_dir)).replace("READ_2016_", "read2016_").replace("_sem_dan", "")
    summary = {}
    for name in a.names:
        sample = gt[name]
        img_path = os.path.join(data_dir, a.split, name)
        stem = name.rsplit(".", 1)[0]
        pred_path = os.path.join(out_dir, "predictions", "%s_%s_%s.json" % (prefix, a.split, stem))
        if name in cached:
            pred = dict(cached[name])
        elif os.path.exists(pred_path):
            pred = json.load(open(pred_path, encoding="utf-8"))
        else:
            pred = predict(export_dir, img_path, a.device, a.max_tokens, a.max_lines)
        pred.update({"image": os.path.relpath(img_path, REPO), "export_dir": os.path.relpath(export_dir, REPO),
                     "gt_raw": sample["text"]})
        img = np.asarray(Image.open(img_path).convert("RGB"))
        base = os.path.join(out_dir, "%s_%s_%s" % (prefix, a.split, stem))
        stats = composite(img, sample, sample["text"], pred["raw"], base, fs=a.font_size,
                          width_in=a.width_cm / 2.54, note=a.note)
        pred["figure_stats"] = stats
        json.dump(pred, open(pred_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        summary[name] = stats
        print("%-14s CER %.2f%%  layout-token edits %d %s  -> %s.pdf" % (
            name, 100 * stats["cer"], stats["layout_token_edit_distance"],
            stats["layout_token_errors"] or "", os.path.relpath(base, REPO)))
    return summary


if __name__ == "__main__":
    main()
