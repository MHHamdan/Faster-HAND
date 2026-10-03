#!/usr/bin/env python3
"""Animated GIFs for docs/gallery/, rendered only from stored predictions (no decoding).

  speculative   the logged verification passes of speculative decoding (m = 5) on READ 2016
                test page 11, from experiments/qualitative/decoding_test_11_spec_steps.json.
                Each pass is shown twice: the four drafted symbols, then the verification
                (accepted, rejected and replaced by the base model's symbol, discarded, or the
                base model's extra symbol after a fully accepted draft). The strip below grows
                with the symbols actually emitted. The file logs the first 40 of 156 passes.

  triple-page   the decode of the triple-page adapted model on READ 2016 triple-page test
                image test_5, from experiments/multipage/adaptation/FT_TRIPLE_triple_page.json,
                revealed line by line. The image shows the reference regions; the page whose
                text is being revealed is outlined, pages not yet reached are dimmed. No
                per-line timing or attention was logged for this decode, so the animation shows
                the order of the output, not where the model attended.

    python tools/gallery_animations.py speculative --out docs/gallery/speculative_decoding.gif
    python tools/gallery_animations.py triple-page \\
        --data formatted/READ_2016_triple_page_sem_dan --out docs/gallery/triple_page_decoding.gif

The triple-page mode needs the formatted triple-page test split (labels.pkl and test_5.jpeg),
built by hand/Datasets/format_read_dan_splits.py; the speculative mode needs no dataset.
"""
import argparse
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "release"))
sys.path.insert(0, ROOT)

TAGS = {"ⓟ": "<P>", "Ⓟ": "</P>", "ⓝ": "<N>", "Ⓝ": "</N>", "ⓢ": "<S>", "Ⓢ": "</S>",
        "ⓑ": "<B>", "Ⓑ": "</B>", "ⓐ": "<A>", "Ⓐ": "</A>"}
TAG_COLOR = {"P": "#6b6b6b", "N": "#2a9d4b", "S": "#7b3fb3", "A": "#e07b00", "B": "#1f6fd1"}
MONO = "DejaVu Sans Mono"


def fig_to_image(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames, out, durations, colors=96):
    """One shared palette for all frames, so unchanged regions compress across frames. The
    palette is fitted on a mosaic of every frame, so a colour that appears in only a few frames
    (a rejected draft, a closing tag) is still represented."""
    tiles = [f.resize((max(1, f.width // 4), max(1, f.height // 4)), Image.NEAREST) for f in frames]
    mosaic = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        mosaic.paste(t, (0, y))
        y += t.height
    base = mosaic.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
    q = [f.quantize(palette=base, dither=Image.NONE) for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True)
    print("%s: %d frames, %d x %d px, %.2f MB" % (out, len(q), q[0].width, q[0].height,
                                                os.path.getsize(out) / 1e6))


# --------------------------------------------------------------------------- speculative
def show_symbol(c):
    return TAGS.get(c, {"\n": "⏎", " ": "␣"}.get(c, c))


def speculative_frame(spec, k, phase, width_in=6.4, height_in=2.7, fs=9.0):
    m = spec["m"]
    s = spec["log"][k]
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width_in)
    ax.set_ylim(height_in, 0)
    ax.text(0.15, 0.22, "Speculative decoding, m = %d  (READ 2016 test page 11)" % m,
            fontsize=fs + 1, fontweight="bold", va="center")
    ax.text(0.15, 0.48, "pass %d of %d logged  ·  prefix %d symbols" % (s["step"] + 1, len(spec["log"]),
                                                                    s["prefix_len"]),
            fontsize=fs - 1, color="#444444", va="center")
    label = "draft: the heads propose" if phase == 0 else "verify: one decoder pass checks every draft"
    ax.text(0.15, 0.80, label, fontsize=fs, va="center", color="#222222")

    cell_w, x0, y0, h = 0.86, 0.6, 1.02, 0.46
    draft, verified, acc = s["draft"], s["verified"], s["accepted"]
    for j in range(m):
        x = x0 + j * (cell_w + 0.12)
        tok, below = "", ""
        if j < m - 1:
            tok = draft[j] if j < len(draft) else ""
            if phase == 0 or not draft:
                fc, ec = "#f4f4f4", "#888888"
            elif j < acc:
                fc, ec = "#e3f4e6", "#2a9d4b"
            elif j == acc:
                fc, ec = "#ffd6d6", "#c81e1e"
                below = "base: " + show_symbol(verified[j]) if j < len(verified) else ""
            else:
                fc, ec = "#eeeeee", "#c4c4c4"
        else:
            reached = phase == 1 and acc == m - 1
            tok = verified[j] if reached and j < len(verified) else ""
            fc, ec = ("#e8f1fb", "#1f6fd1") if reached else ("white", "#c4c4c4")
        if not draft and j == 0 and phase == 1:            # first pass: no drafts yet
            tok, fc, ec = verified[0], "#e8f1fb", "#1f6fd1"
        ax.add_patch(Rectangle((x, y0), cell_w, h, fc=fc, ec=ec, lw=1.2))
        if tok:
            ax.text(x + cell_w / 2, y0 + h / 2, show_symbol(tok), fontsize=fs + 2, family=MONO,
                    ha="center", va="center")
        if below:
            ax.text(x + cell_w / 2, y0 + h + 0.17, below, fontsize=fs - 1.5, family=MONO,
                    ha="center", va="center", color="#c81e1e")
    if phase == 1:
        n_emit = len(s["emitted"])
        ax.text(0.15, 1.92, "emits %d symbol%s, rolls back %d" % (n_emit, "" if n_emit == 1 else "s",
                                                                s["rolled_back"]),
                fontsize=fs, va="center", fontweight="bold",
                color="#2a9d4b" if s["rolled_back"] == 0 else "#333333")
    emitted = "".join("".join(p["emitted"]) for p in spec["log"][:k + (1 if phase == 1 else 0)])
    shown = "".join(show_symbol(c) if c in TAGS else ("⏎" if c == "\n" else c) for c in emitted)
    ax.text(0.15, 2.36, "output: " + shown[-58:], fontsize=fs - 1, family=MONO, va="center",
            color="#111111")
    return fig


def cmd_speculative(a):
    spec = json.load(open(a.spec, encoding="utf-8"))
    emitted = "".join("".join(p["emitted"]) for p in spec["log"])
    assert spec["raw"].startswith(emitted), "logged passes do not reproduce the stored decode"
    frames, durations = [], []
    for k in range(len(spec["log"])):
        for phase in (0, 1):
            frames.append(fig_to_image(speculative_frame(spec, k, phase), a.dpi))
            durations.append(500 if phase == 0 else 900)
    durations[-1] = 2500
    save_gif(frames, a.out, durations, colors=a.colors)


# --------------------------------------------------------------------------- triple page
def tagged_lines(page_raw):
    """A page's raw stream as display lines, layout tokens rendered as tags."""
    lines, cur = [], []
    for c in page_raw:
        if c == "\n":
            lines.append(cur)
            cur = []
        elif c in TAGS:
            cur.append(("tag", TAGS[c]))
        else:
            cur.append(("chr", c))
    if cur:
        lines.append(cur)
    return lines


def draw_line(ax, x, y, items, fs):
    """Draw one display line; returns nothing. Tags are colored by region class."""
    t = ax.transData
    for kind, s in items:
        color = TAG_COLOR[s.strip("</>")] if kind == "tag" else "#111111"
        txt = ax.text(x, y, s, fontsize=fs, family=MONO, color=color, va="top", ha="left",
                      fontweight="bold" if kind == "tag" else "normal", transform=t)
        x += len(s) * fs * 0.602 / 72.0          # monospace advance in inches


def cmd_triple(a):
    import qualitative_figures as Q
    res = json.load(open(a.results, encoding="utf-8"))
    pred = res["raw_predictions"][a.name]
    gt = Q.load_labels(a.data, "test")
    sample = gt[a.name]
    img = np.array(Image.open(os.path.join(a.data, "test", a.name)).convert("RGB"))
    order = Q.reading_order(sample)
    img, order, sample = Q.crop_to_page(img, order, sample, crop_x=True)
    H, W = img.shape[:2]

    width_in = 7.2
    img_h = width_in * H / W
    base = plt.figure(figsize=(width_in, img_h))
    ax = base.add_axes([0, 0, 1, 1])
    Q.draw_image_panel(ax, img, order, sample, title=None, number_every=0, label_fs=6.0,
                       page_labels=True, section_labels=False)
    base_img = np.array(fig_to_image(base, a.dpi))

    pages = Q.split_raw_pages(pred)
    page_lines = [tagged_lines(p) for p in pages]
    text_rows, fs = 13, 7.0
    text_h = 0.42 + text_rows * fs * 1.35 / 72.0
    frames, durations = [], []
    steps = [(p, n) for p in range(len(page_lines)) for n in range(1, len(page_lines[p]) + 1, a.stride)]
    steps += [(p, len(page_lines[p])) for p in range(len(page_lines))
              if (len(page_lines[p]) - 1) % a.stride]
    steps = sorted(set(steps))
    for p, n in steps:
        fig = plt.figure(figsize=(width_in, img_h + text_h))
        axi = fig.add_axes([0, text_h / (img_h + text_h), 1, img_h / (img_h + text_h)])
        axi.imshow(base_img)
        axi.set_axis_off()
        sx = base_img.shape[1] / W
        sy = base_img.shape[0] / H
        for k, pg in enumerate(sample["pages"]):
            x, y = pg["left"] * sx, pg["top"] * sy
            w, h = (pg["right"] - pg["left"]) * sx, (pg["bottom"] - pg["top"]) * sy
            if k > p:
                axi.add_patch(Rectangle((x, y), w, h, fc="white", ec="none", alpha=0.62))
            elif k == p:
                axi.add_patch(Rectangle((x, y), w, h, fill=False, ec="#111111", lw=2.2))
        axt = fig.add_axes([0, 0, 1, text_h / (img_h + text_h)])
        axt.set_axis_off()
        axt.set_xlim(0, width_in)
        axt.set_ylim(text_h, 0)
        axt.text(0.12, 0.17, "HAND, triple-page adapted model — page %d of 3, line %d of %d"
                 % (p + 1, n, len(page_lines[p])), fontsize=fs + 1, fontweight="bold", va="center")
        shown = page_lines[p][:n][-text_rows:]
        for r, items in enumerate(shown):
            draw_line(axt, 0.12, 0.36 + r * fs * 1.35 / 72.0, items, fs)
        frames.append(fig_to_image(fig, a.dpi))
        durations.append(1600 if n == len(page_lines[p]) else 450)
    durations[-1] = 3000
    save_gif(frames, a.out, durations, colors=a.colors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speculative")
    s.add_argument("--spec", default=os.path.join(ROOT, "experiments/qualitative/decoding_test_11_spec_steps.json"))
    s.add_argument("--out", default=os.path.join(ROOT, "docs/gallery/speculative_decoding.gif"))
    s.add_argument("--dpi", type=int, default=100)
    s.add_argument("--colors", type=int, default=64)
    t = sub.add_parser("triple-page")
    t.add_argument("--results", default=os.path.join(ROOT, "experiments/multipage/adaptation/FT_TRIPLE_triple_page.json"))
    t.add_argument("--data", default=os.path.join(ROOT, "formatted/READ_2016_triple_page_sem_dan"))
    t.add_argument("--name", default="test_5.jpeg")
    t.add_argument("--out", default=os.path.join(ROOT, "docs/gallery/triple_page_decoding.gif"))
    t.add_argument("--stride", type=int, default=2, help="reveal this many lines per frame")
    t.add_argument("--dpi", type=int, default=100)
    t.add_argument("--colors", type=int, default=96)
    a = ap.parse_args()
    {"speculative": cmd_speculative, "triple-page": cmd_triple}[a.cmd](a)


if __name__ == "__main__":
    main()
