#!/usr/bin/env python3
"""Autoregressive decoding, made visible: where the decoder looks while it emits the stream.

The page model decodes one token at a time; at every step the decoder's last cross-attention
layer yields a distribution over the encoder grid (one cell per 32 x 8 pixels). This script
records that map for every emitted token and renders

  * a static multi-panel figure (`--panels 10 30 60 100`): the page with the attention mass
    of the tokens decoded so far, the token stream beneath it, and a final panel with the
    centroid path of the whole decode (the model's own reading order);
  * optionally an animated GIF of the same data (`--gif`), one frame per `--stride` tokens;
  * optionally (`--speculative`), an instrumented speculative decode with the m-1 draft heads:
    for every verification step the drafted tokens, the number accepted and the base model's
    correction are logged, and the first few steps are drawn as a small figure.

Everything is produced by the released model on the given image; the per-step data are
saved next to the outputs (`<out>_steps.npz`, `<out>_spec_steps.json`) so the figures can be
regenerated without decoding again.

    python tools/decoding_animation.py --export-dir outputs/export_e14_cpu \
        --image formatted/READ_2016_page_sem_dan/test/test_11.jpeg \
        --out experiments/qualitative/decoding_test_11 --gif \
        --gif-out docs/assets/hand_decoding_test_11.gif --fps 3 --device cpu

CPU is sufficient (about ten seconds for the greedy decode of a page, fp32).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "release"))
from hand_release.inference import HANDRecognizer, strip_layout  # noqa: E402

OPEN = {"ⓟ": "P", "ⓝ": "N", "ⓢ": "S", "ⓐ": "A", "ⓑ": "B"}
CLOSE = {"Ⓟ": "P", "Ⓝ": "N", "Ⓢ": "S", "Ⓐ": "A", "Ⓑ": "B"}
TAG_CLASS = {"P": "page", "N": "page_number", "S": "section", "A": "annotation", "B": "body"}
CLASS_COLOR = {"page": "#6b6b6b", "page_number": "#2a9d4b", "section": "#7b3fb3",
               "annotation": "#e07b00", "body": "#1f6fd1"}
MONO = "DejaVu Sans Mono"
HEAT = LinearSegmentedColormap.from_list("heat", [(0, 0, 0, 0), (0.12, 0.47, 0.71, 0.55),
                                                  (0.85, 0.1, 0.1, 0.85)])
PATH_CMAP = LinearSegmentedColormap.from_list("path", ["#1a237e", "#00897b", "#c62828"])


# --------------------------------------------------------------------------- decoding
def decode_with_attention(r, image, max_tokens):
    """Greedy K/V-cached decode (mirrors HANDRecognizer._decode_greedy) that also keeps the
    decoder's last-layer cross-attention map of every emitted token."""
    dec = r.decoder
    x = r.preprocess(image)
    h_img, w_img = x.shape[-2], x.shape[-1]
    reduced = [[int(np.ceil(h_img / 32)), int(np.ceil(w_img / 8))]]
    t0 = time.time()
    with torch.no_grad():
        feats = r.encoder(x)
        fsize = feats.size()
        pf = dec.features_updater.get_pos_features(feats)
        pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
        dec.reset_mem_kv_cache()
        seq = [r.tok_start]
        ids, conf, atts, cache = [], [], [], None
        line_count = char_in_line = 0
        truncated = True
        for _ in range(max_tokens):
            toks = torch.tensor([seq], dtype=torch.long, device=r.device)
            plen = torch.tensor([len(seq)], dtype=torch.int, device=r.device)
            _, pred, cache, w = dec(pf, pf, toks, reduced, plen, fsize, start=0, cache=cache,
                                    num_pred=1, padding_value=r.tok_pad, use_mem_cache=True)
            probs = torch.softmax(pred[0, :, -1].float(), dim=0)
            t = int(torch.argmax(pred[0, :, -1]))
            if t == r.tok_end:
                truncated = False
                break
            ids.append(t)
            conf.append(float(probs[t]))
            atts.append(w[0, 0].float().cpu().numpy())
            seq.append(t)
            line_count, char_in_line, stop = r._stop(t, line_count, char_in_line)
            if stop:
                truncated = False
                break
    raw = "".join(r.charset[i] for i in ids if 0 <= i < r.vocab)
    return {"raw": raw, "ids": ids, "confidence": conf, "attention": np.stack(atts),
            "feature_grid": [int(fsize[2]), int(fsize[3])], "image_hw": [h_img, w_img],
            "latency_s": time.time() - t0, "truncated": truncated}


def decode_speculative_logged(r, image, max_tokens, n_steps_log=40):
    """Fused draft-and-verify (mirrors HANDRecognizer._decode_speculative) with a per-step
    log: drafted tokens, how many were accepted, and the base model's own next token."""
    dec = r.decoder
    x = r.preprocess(image)
    h_img, w_img = x.shape[-2], x.shape[-1]
    reduced = [[int(np.ceil(h_img / 32)), int(np.ceil(w_img / 8))]]
    log = []
    t0 = time.time()
    with torch.no_grad():
        feats = r.encoder(x)
        fsize = feats.size()
        pf = dec.features_updater.get_pos_features(feats)
        pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
        dec.reset_mem_kv_cache()
        seq = [r.tok_start]
        ids, conf, draft, cache = [], [], [], None
        line_count = char_in_line = 0
        done, truncated, step = False, True, 0
        while not done and len(ids) < max_tokens:
            feed = seq + draft
            n_new = 1 + len(draft)
            toks = torch.tensor([feed], dtype=torch.long, device=r.device)
            plen = torch.tensor([len(feed)], dtype=torch.int, device=r.device)
            out, pred, cache, _ = dec(pf, pf, toks, reduced, plen, fsize, start=0, cache=cache,
                                      num_pred=n_new, padding_value=r.tok_pad, use_mem_cache=True)
            probs = torch.softmax(pred[0].float(), dim=0)
            argmax = pred[0].argmax(0).tolist()
            top1 = probs.max(0).values.tolist()
            ell = 0
            while ell < len(draft) and argmax[ell] == draft[ell]:
                ell += 1
            emitted = []
            for t, c in zip(draft[:ell] + [argmax[ell]], top1[:ell + 1]):
                if t == r.tok_end:
                    done, truncated = True, False
                    break
                ids.append(t)
                conf.append(c)
                seq.append(t)
                emitted.append(t)
                line_count, char_in_line, stop = r._stop(t, line_count, char_in_line)
                if stop:
                    done, truncated = True, False
                    break
            if step < n_steps_log:
                log.append({"step": step, "prefix_len": len(seq) - 1 - len(emitted),
                            "draft": [r.charset[t] if t < r.vocab else "<end>" for t in draft],
                            "verified": [r.charset[t] if t < r.vocab else "<end>" for t in argmax[:len(draft) + 1]],
                            "accepted": ell,
                            "emitted": [r.charset[t] if t < r.vocab else "<end>" for t in emitted],
                            "rolled_back": n_new - (ell + 1)})
            drop = n_new - (ell + 1)
            if drop > 0 and cache is not None:
                cache = cache[:, :cache.size(1) - drop]
            if done:
                break
            h = out[ell, 0, :].float()
            draft = [int(l.argmax(-1)) for l in r.heads(h)][:r.m - 1]
            step += 1
    raw = "".join(r.charset[i] for i in ids if 0 <= i < r.vocab)
    all_steps = step + 1
    return {"raw": raw, "n_tokens": len(ids), "n_verification_steps": all_steps,
            "tokens_per_step": len(ids) / max(1, all_steps), "latency_s": time.time() - t0,
            "m": r.m, "truncated": truncated, "log": log}


# --------------------------------------------------------------------------- rendering
def tag_label(c):
    if c in OPEN:
        return OPEN[c], False
    if c in CLOSE:
        return CLOSE[c], True
    return None, None


def accumulated_map(att, upto, grid, hw, decay=None):
    """Attention mass of the tokens decoded so far, upsampled to image size (nearest)."""
    a = att[:upto].sum(0) if upto > 0 else np.zeros(grid, dtype=np.float32)
    a = a / (a.max() + 1e-8)
    H, W = hw
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0


def content_rows(att, grid, hw, margin_frac=0.04, thresh=0.08):
    """Rows of the image that the decode attended to: the vertical extent of the content."""
    a = att.sum(0)
    a = a / (a.max() + 1e-8)
    rows = np.where(a.max(1) > thresh)[0]
    H = hw[0]
    if len(rows) == 0:
        return 0, H
    y0 = int(rows[0] * H / grid[0]) - int(margin_frac * H)
    y1 = int((rows[-1] + 1) * H / grid[0]) + int(margin_frac * H)
    return max(0, y0), min(H, y1)


def centroids(att, grid, hw):
    gh, gw = grid
    H, W = hw
    ys, xs = np.mgrid[0:gh, 0:gw]
    cy = (att * ys[None]).sum((1, 2)) / (att.sum((1, 2)) + 1e-8)
    cx = (att * xs[None]).sum((1, 2)) / (att.sum((1, 2)) + 1e-8)
    return (cx + 0.5) * W / gw, (cy + 0.5) * H / gh


def draw_stream(ax, raw, upto, fs=5.6, max_cells=44, max_rows=None):
    """Token stream with layout tags; the first `upto` tokens drawn dark, the rest faint."""
    ax.set_axis_off()
    rows, cur, w = [], [], 0.0
    for i, c in enumerate(raw):
        lab, close = tag_label(c)
        iw = 2.4 if lab and close else (1.8 if lab else 1.0)
        if c == "\n":
            rows.append(cur)
            cur, w = [], 0.0
            continue
        if w + iw > max_cells:
            rows.append(cur)
            cur, w = [], 0.0
        cur.append((i, c, lab, close, iw))
        w += iw
    rows.append(cur)
    if max_rows and len(rows) > max_rows:
        # keep the rows around the current position
        last = next((k for k, row in enumerate(rows) if row and row[-1][0] >= upto - 1), len(rows) - 1)
        start = max(0, min(last - max_rows + 2, len(rows) - max_rows))
        rows = rows[start:start + max_rows]
    ax.set_xlim(-0.3, max_cells + 0.3)
    ax.set_ylim(len(rows) + 0.3, -0.7)
    for yi, row in enumerate(rows):
        x = 0.0
        for (i, c, lab, close, iw) in row:
            decoded = i < upto
            if lab:
                col = CLASS_COLOR[TAG_CLASS[lab]]
                ax.add_patch(plt.Rectangle((x + 0.1, yi - 0.42), iw - 0.2, 0.84,
                                           fc=col if not close else "white", ec=col,
                                           lw=0.5, alpha=1.0 if decoded else 0.25, zorder=2))
                ax.text(x + iw / 2, yi, ("/" if close else "") + lab, fontsize=fs * 0.78,
                        family=MONO, ha="center", va="center", zorder=3, fontweight="bold",
                        color=("white" if not close else col), alpha=1.0 if decoded else 0.35)
            else:
                ax.text(x, yi, c, fontsize=fs, family=MONO, ha="left", va="center",
                        color="#111111" if decoded else "#b5b5b5",
                        fontweight="bold" if i == upto - 1 else "normal")
            x += iw


def render_panels(img, data, fractions, out_base, fs=5.6, width_in=7.28, gt_order=None):
    raw, att = data["raw"], data["attention"]
    grid, hw = data["feature_grid"], data["image_hw"]
    n = len(raw)
    y0, y1 = content_rows(att, grid, hw)
    H, W = img.shape[:2]
    n_pan = len(fractions) + 1
    gap = 0.08
    pan_w = (width_in - gap * (n_pan - 1)) / n_pan
    img_h = pan_w * (y1 - y0) / W
    stream_h = 1.15
    fig_h = 0.22 + img_h + 0.05 + stream_h + 0.1
    fig = plt.figure(figsize=(width_in, fig_h))
    cx, cy = centroids(att, grid, hw)
    for k, f in enumerate(fractions + [None]):
        x0 = k * (pan_w + gap) / width_in
        ax = fig.add_axes([x0, 1 - (0.22 + img_h) / fig_h, pan_w / width_in, img_h / fig_h])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_ylim(y1, y0)
        if f is not None:
            upto = max(1, int(round(f * n / 100.0)))
            heat = accumulated_map(att, upto, grid, hw)
            ax.imshow(heat, cmap=HEAT, vmin=0, vmax=1, interpolation="bilinear")
            # the current token's attention in strong red
            cur = np.asarray(Image.fromarray((att[upto - 1] / att[upto - 1].max() * 255).astype(np.uint8))
                             .resize((W, H), Image.BILINEAR)) / 255.0
            ax.contour(cur, levels=[0.5], colors="#c62828", linewidths=0.8)
            ax.set_title("(%s) %d %%: %d / %d tokens" % ("abcdefg"[k], f, upto, n),
                         fontsize=6.2, loc="left", pad=2)
            ax_s = fig.add_axes([x0, 0.08 / fig_h, pan_w / width_in, stream_h / fig_h])
            draw_stream(ax_s, raw, upto, fs=fs, max_cells=int(pan_w / (0.602 * fs / 72)) - 1,
                        max_rows=int(stream_h / (1.3 * fs / 72)) - 1)
        else:
            # final panel: the centroid path of the whole decode, coloured by token index
            idx = [i for i, c in enumerate(raw) if c not in OPEN and c not in CLOSE and c != "\n"]
            pts = np.array([(cx[i], cy[i]) for i in idx])
            # one point per text line (centroid of its characters) to keep the path legible
            lines, cur_line = [], []
            for i, c in enumerate(raw):
                if c == "\n":
                    if cur_line:
                        lines.append(cur_line)
                    cur_line = []
                elif c not in OPEN and c not in CLOSE:
                    cur_line.append(i)
            if cur_line:
                lines.append(cur_line)
            lx = [np.mean([cx[i] for i in ln]) for ln in lines]
            ly = [np.mean([cy[i] for i in ln]) for ln in lines]
            ax.scatter(pts[:, 0], pts[:, 1], c=idx, cmap=PATH_CMAP, s=1.2, lw=0, alpha=0.6, zorder=3)
            for j in range(len(lx) - 1):
                ax.annotate("", (lx[j + 1], ly[j + 1]), (lx[j], ly[j]),
                            arrowprops=dict(arrowstyle="-|>", color=PATH_CMAP(j / max(1, len(lx) - 2)),
                                            lw=0.9, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
            for j in range(len(lx)):
                ax.text(lx[j], ly[j], str(j + 1), fontsize=3.6, ha="center", va="center", color="white",
                        zorder=5, bbox=dict(boxstyle="circle,pad=0.15", fc=PATH_CMAP(j / max(1, len(lx) - 1)), ec="none"))
            ax.set_title("(%s) centroid path, full decode" % "abcdefg"[k],
                         fontsize=6.2, loc="left", pad=2)
            ax_s = fig.add_axes([x0, 0.08 / fig_h, pan_w / width_in, stream_h / fig_h])
            ax_s.set_axis_off()
            ax_s.text(0, 1, "\n".join([
                "%d tokens, %d text lines" % (n, len(lines)),
                "mean top-1 conf. %.3f" % float(np.mean(data["confidence"])),
                "greedy, K/V cache, fp32,",
                "  CPU: %.1f s" % float(data["latency_s"]),
                "blue: attention mass so far",
                "red: current token's attention",
                "arrows: per-line centroids,",
                "  the model's reading order"]),
                fontsize=fs, family=MONO, va="top", ha="left", transform=ax_s.transAxes, linespacing=1.25)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=250)
    plt.close(fig)
    return fig_h


def render_gif(img, data, gif_out, fps=3, stride=4, max_px=900, hold_last=8):
    raw, att = data["raw"], data["attention"]
    grid, hw = data["feature_grid"], data["image_hw"]
    n = len(raw)
    y0, y1 = content_rows(att, grid, hw)
    img = img[y0:y1]
    H, W = img.shape[:2]
    scale = min(1.0, max_px / float(max(H, W)))
    fig_w = W * scale / 100.0
    fig_h_img = H * scale / 100.0
    stream_h = 1.6
    frames = []
    steps = list(range(stride, n + 1, stride))
    if steps[-1] != n:
        steps.append(n)
    tot_h = max(fig_h_img, stream_h) + 0.58
    for upto in steps:
        fig = plt.figure(figsize=(fig_w + 3.2, tot_h), dpi=100, facecolor="white")
        ax = fig.add_axes([0, 0.01, fig_w / (fig_w + 3.2), fig_h_img / tot_h])
        ax.imshow(img, interpolation="bilinear")
        ax.set_axis_off()
        heat = accumulated_map(att, upto, grid, hw)[y0:y1]
        ax.imshow(heat, cmap=HEAT, vmin=0, vmax=1, interpolation="bilinear")
        cur = np.asarray(Image.fromarray((att[upto - 1] / att[upto - 1].max() * 255).astype(np.uint8))
                         .resize((hw[1], hw[0]), Image.BILINEAR))[y0:y1] / 255.0
        ax.contour(cur, levels=[0.5], colors="#c62828", linewidths=1.2)
        ax_s = fig.add_axes([fig_w / (fig_w + 3.2) + 0.01, 0.01, 1 - fig_w / (fig_w + 3.2) - 0.02,
                             fig_h_img / tot_h])
        draw_stream(ax_s, raw, upto, fs=8.5, max_cells=int(3.0 / (0.602 * 8.5 / 72)) - 1,
                    max_rows=int(fig_h_img / (1.3 * 8.5 / 72)) - 2)
        fig.text(0.01, 1 - 0.14 / tot_h, "HAND, autoregressive decoding: token %d of %d" % (upto, n),
                 fontsize=10, va="center", ha="left", fontweight="bold")
        fig.text(0.01, 1 - 0.36 / tot_h, "blue: attention mass of the tokens decoded so far;  "
                 "red: attention of the current token;  right: the token stream", fontsize=8,
                 va="center", ha="left", color="#444444")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=160))
    frames += [frames[-1]] * hold_last
    os.makedirs(os.path.dirname(os.path.abspath(gif_out)), exist_ok=True)
    frames[0].save(gif_out, save_all=True, append_images=frames[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    return os.path.getsize(gif_out), len(frames)


def render_spec_steps(spec, out_base, steps=(3, 7, 12, 15), fs=6.0, width_in=3.5):
    """Draft -> verify -> accept -> rollback, for a few real verification passes.

    One pass feeds the accepted prefix plus the m-1 drafted tokens and reads m predictions
    from the base model: prediction j is compared with draft j, left to right; the first
    mismatch is replaced by the base model's own token and everything after it is dropped
    (K/V cache rolled back). When every draft is accepted, prediction m-1 is a free extra
    token, so a pass emits between 1 and m tokens.
    """
    log = [s for s in spec["log"] if s["step"] in steps]
    m = spec["m"]
    row_h = 0.50
    top = 0.50
    fig_h = top + row_h * len(log) + 0.64
    fig = plt.figure(figsize=(width_in, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, width_in)
    ax.set_ylim(fig_h, 0)

    def show(c):
        if c in OPEN:
            return OPEN[c]
        if c in CLOSE:
            return "/" + CLOSE[c]
        return {"\n": "⏎", " ": "␣", "<end>": "END"}.get(c, c)

    ax.text(0.05, 0.13, "Speculative decoding, m = %d" % m, fontsize=fs + 1.2, fontweight="bold",
            va="center", ha="left")
    ax.text(0.05, 0.32, "one verification pass per row. Cells 1-%d: the draft heads' proposals, checked\n"
            "left to right against the base model's predictions; cell %d: the base model's\n"
            "own next token, emitted only when every draft was accepted" % (m - 1, m),
            fontsize=fs - 1.0, va="center", ha="left", color="#444444", linespacing=1.2)
    cell_w = 0.29
    x_cells = 1.02
    for k, s in enumerate(log):
        y = top + k * row_h
        ax.text(0.05, y + 0.12, "pass %d" % (s["step"] + 1), fontsize=fs, va="center", ha="left",
                fontweight="bold")
        ax.text(0.05, y + 0.28, "prefix: %d tokens" % s["prefix_len"], fontsize=fs - 1.4,
                va="center", ha="left", color="#555555")
        draft, verified, acc = s["draft"], s["verified"], s["accepted"]
        for j in range(m):
            x = x_cells + j * cell_w
            if j < m - 1:                       # a drafted position
                tok = draft[j] if j < len(draft) else ""
                if j < acc:
                    fc, ec, ls = "#e3f4e6", "#2a9d4b", "-"
                elif j == acc:
                    fc, ec, ls = "#ffd6d6", "#c81e1e", "-"
                else:
                    fc, ec, ls = "#f2f2f2", "#bbbbbb", "-"
            else:                               # the base model's extra token
                reached = acc == m - 1
                tok = verified[j] if reached and j < len(verified) else ""
                fc, ec, ls = ("#e8f1fb", "#1f6fd1", "-") if reached else ("white", "#bbbbbb", (0, (2, 2)))
            ax.add_patch(plt.Rectangle((x, y + 0.01), cell_w - 0.03, 0.23, fc=fc, ec=ec, lw=0.8, ls=ls))
            if tok:
                ax.text(x + (cell_w - 0.03) / 2, y + 0.125, show(tok), fontsize=fs + 0.8, family=MONO,
                        ha="center", va="center", color="#111111")
            if j < m - 1 and j == acc:
                ax.text(x + (cell_w - 0.03) / 2, y + 0.33, "base: %s" % show(verified[j]),
                        fontsize=fs - 1.4, ha="center", va="center", color="#c81e1e")
        ax.text(x_cells + m * cell_w + 0.05, y + 0.125,
                "emits %d\nrolls back %d" % (len(s["emitted"]), s["rolled_back"]),
                fontsize=fs - 0.8, va="center", ha="left", linespacing=1.2)
    y = top + len(log) * row_h + 0.02
    legend = [("#e3f4e6", "#2a9d4b", "draft accepted"),
              ("#ffd6d6", "#c81e1e", "draft rejected: the base token is emitted instead"),
              ("#f2f2f2", "#bbbbbb", "draft discarded after a rejection"),
              ("#e8f1fb", "#1f6fd1", "base model's extra token after a full acceptance")]
    for i, (fc, ec, lab) in enumerate(legend):
        xx = 0.05 + (i % 2) * 1.55
        yy = y + (i // 2) * 0.15
        ax.add_patch(plt.Rectangle((xx, yy), 0.11, 0.10, fc=fc, ec=ec, lw=0.8))
        ax.text(xx + 0.15, yy + 0.05, lab, fontsize=fs - 1.7, va="center", ha="left")
    ax.text(0.05, fig_h - 0.07, "page: %d tokens in %d passes = %.2f tokens per pass;\n"
            "the emitted string is identical to the greedy decode" % (spec["n_tokens"],
            spec["n_verification_steps"], spec["tokens_per_step"]),
            fontsize=fs - 1.0, va="bottom", ha="left", color="#333333", linespacing=1.2)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", default="outputs/export_e14_cpu")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output base path (without extension)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--panels", type=int, nargs="+", default=[10, 30, 60, 100],
                    help="percentages of the stream shown in the static panels")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif-out", default=None)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--stride", type=int, default=5, help="tokens per GIF frame")
    ap.add_argument("--max-px", type=int, default=800, help="longest image side in a GIF frame")
    ap.add_argument("--speculative", action="store_true", help="also instrument a speculative decode")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--spec-steps", type=int, nargs="+", default=[3, 7, 12, 15])
    ap.add_argument("--reuse", action="store_true", help="reuse <out>_steps.npz if present")
    a = ap.parse_args()

    torch.set_num_threads(8)
    export_dir = a.export_dir if os.path.isabs(a.export_dir) else os.path.join(REPO, a.export_dir)
    image = a.image if os.path.isabs(a.image) else os.path.join(REPO, a.image)
    out = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img = np.asarray(Image.open(image).convert("RGB"))

    steps_path = out + "_steps.npz"
    if a.reuse and os.path.exists(steps_path):
        z = np.load(steps_path, allow_pickle=True)
        data = {k: (z[k].tolist() if k in ("raw", "feature_grid", "image_hw", "latency_s", "truncated")
                    else z[k]) for k in z.files}
        data["raw"] = str(data["raw"])
        data["confidence"] = list(z["confidence"])
    else:
        r = HANDRecognizer.from_pretrained(export_dir, device=a.device, kv_cache=True)
        data = decode_with_attention(r, image, a.max_tokens)
        np.savez_compressed(steps_path, raw=data["raw"], ids=np.array(data["ids"]),
                            confidence=np.array(data["confidence"]), attention=data["attention"],
                            feature_grid=np.array(data["feature_grid"]), image_hw=np.array(data["image_hw"]),
                            latency_s=data["latency_s"], truncated=data["truncated"])
        json.dump({"image": os.path.relpath(image, REPO), "export_dir": os.path.relpath(export_dir, REPO),
                   "raw": data["raw"], "text": strip_layout(data["raw"]), "n_tokens": len(data["raw"]),
                   "latency_s": data["latency_s"], "truncated": data["truncated"],
                   "decode_path": "kv_cache_greedy_with_attention_log"},
                  open(out + "_prediction.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("decoded %d tokens in %.1f s (%s)" % (len(data["raw"]), float(data["latency_s"]), os.path.basename(image)))

    h = render_panels(img, data, a.panels, out)
    print("panels -> %s.pdf (%.2f x %.2f in)" % (out, 7.28, h))

    if a.gif:
        gif_out = a.gif_out or (out + ".gif")
        gif_out = gif_out if os.path.isabs(gif_out) else os.path.join(REPO, gif_out)
        size, nfr = render_gif(img, data, gif_out, fps=a.fps, stride=a.stride, max_px=a.max_px)
        print("gif -> %s  %d frames  %.1f MB" % (gif_out, nfr, size / 1e6))

    if a.speculative:
        spec_path = out + "_spec_steps.json"
        if a.reuse and os.path.exists(spec_path):
            spec = json.load(open(spec_path, encoding="utf-8"))
        else:
            r = HANDRecognizer.from_pretrained(export_dir, device=a.device, speculative=True, m=a.m)
            spec = decode_speculative_logged(r, image, a.max_tokens)
            spec["greedy_identical"] = spec["raw"] == data["raw"]
            json.dump(spec, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("speculative m=%d: %d tokens in %d passes (%.2f tok/pass), %.1f s, identical to greedy: %s"
              % (spec["m"], spec["n_tokens"], spec["n_verification_steps"], spec["tokens_per_step"],
                 spec["latency_s"], spec["greedy_identical"]))
        render_spec_steps(spec, out + "_speculative", steps=a.spec_steps)
        print("speculative steps -> %s_speculative.pdf" % out)


if __name__ == "__main__":
    main()
