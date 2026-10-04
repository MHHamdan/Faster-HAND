#!/usr/bin/env python3
"""Gradio demo: read a handwritten document image with a released FasterHAND model.

    pip install -r demo/requirements.txt
    python demo/app.py                    # http://127.0.0.1:7860
    python demo/app.py --share            # temporary public link from Gradio

The first use of a model downloads its weights from the GitHub release into weights/ and
checks the pinned SHA-256 (release/hand_release/hub.py). The output is the model's single
interleaved stream: the transcription, with layout tokens shown as colored tags, the plain
text, and the regions parsed from the tags. FasterHAND predicts layout tokens, not coordinates, so no
boxes are drawn on the image.
"""
import argparse
import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "release"))
sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from hand_release.hub import MODELS, download  # noqa: E402
from hand_release.inference import HANDRecognizer  # noqa: E402

EXAMPLES = os.path.join(ROOT, "demo", "examples")
TAGS = {"ⓟ": ("P", "page"), "Ⓟ": ("/P", "page"), "ⓝ": ("N", "page number"), "Ⓝ": ("/N", "page number"),
        "ⓢ": ("S", "section"), "Ⓢ": ("/S", "section"), "ⓐ": ("A", "annotation"),
        "Ⓐ": ("/A", "annotation"), "ⓑ": ("B", "body"), "Ⓑ": ("/B", "body")}
COLORS = {"page": "#6b6b6b", "page number": "#2a9d4b", "section": "#7b3fb3",
          "annotation": "#e07b00", "body": "#1f6fd1"}
LABELS = {
    "hand-read2016-page": "Single page (paper model)",
    "hand-read2016-page-compact": "Single page, compact (shared K/V)",
    "hand-read2016-double-page": "Double page (adapted)",
    "hand-read2016-triple-page": "Triple page (adapted)",
}
CHOICES = [(v, k) for k, v in LABELS.items()]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_loaded = {}


def get_model(name, speculative, weights_dir):
    key = (name, speculative)
    if key not in _loaded:
        path = download(name, weights_dir)
        heads = os.path.join(path, "spec_heads_m5.safetensors") if speculative else None
        _loaded[key] = HANDRecognizer.from_pretrained(path, device=DEVICE, kv_cache=True,
                                                      speculative=speculative, heads_path=heads)
    return _loaded[key]


def render_stream(raw):
    """The interleaved stream as HTML: layout tokens become colored tags, newlines breaks."""
    out = []
    for c in raw:
        if c in TAGS:
            tag, cls = TAGS[c]
            out.append('<span style="color:#fff;background:%s;border-radius:3px;padding:0 3px;'
                       'margin:0 1px;font-size:0.8em;font-weight:600">%s</span>' % (COLORS[cls], tag))
        elif c == "\n":
            out.append("<br>")
        else:
            out.append(html.escape(c))
    legend = " ".join('<span style="color:#fff;background:%s;border-radius:3px;padding:0 4px;'
                      'font-size:0.8em">%s</span>' % (col, name) for name, col in COLORS.items())
    return ('<div style="font-size:0.85em;margin-bottom:6px">%s</div>'
            '<div style="font-family:DejaVu Sans Mono,monospace;line-height:1.6;'
            'white-space:normal">%s</div>' % (legend, "".join(out)))


def transcribe(image, name, dpi, speculative, weights_dir):
    if image is None:
        raise gr.Error("Upload a document image or pick an example.")
    if speculative and not MODELS[name]["heads"]:
        speculative = False
    model = get_model(name, speculative, weights_dir)
    pages = MODELS[name]["pages"]
    result = model.read(Image.fromarray(image).convert("RGB"), source_dpi=int(dpi),
                        max_tokens=3000 * pages, amp=DEVICE == "cuda")
    regions = [[r.cls, r.text] for r in result.regions]
    stats = ("**%d symbols** in %.2f s on %s · decoding: %s%s"
             % (result.n_tokens, result.latency_s, DEVICE.upper(), result.decode_path,
                " · output truncated at the length limit" if result.truncated else ""))
    if result.n_tokens and result.raw.count("ⓟ") != pages:
        stats += ("<br>The output contains %d page element(s); this model was trained on %d-page "
                  "inputs. Choose the model that matches the number of pages in the image."
                  % (result.raw.count("ⓟ"), pages))
    return render_stream(result.raw), result.text, regions, stats


def build(weights_dir):
    with gr.Blocks(title="FasterHAND demo") as demo:
        gr.Markdown(
            "# FasterHAND demo\n"
            "Segmentation-free recognition of handwritten documents: the model reads the image and "
            "emits one sequence of characters and layout tokens (page, page number, section, "
            "annotation, body) in reading order. Models were trained on READ 2016 (Early Modern "
            "German); other scripts and layouts are outside what they have seen. "
            "[Code and paper results](https://github.com/MHHamdan/Faster-HAND)")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Document image", type="numpy")
                name = gr.Dropdown(CHOICES, value="hand-read2016-page", label="Model")
                dpi = gr.Number(value=150, precision=0, label="Image resolution (dpi)",
                                info="300 for a raw READ 2016 scan; 150 for the examples. "
                                     "The image is resampled to 150 dpi.")
                speculative = gr.Checkbox(value=False, label="Speculative decoding (m = 5)",
                                          info="Same output as greedy decoding, fewer decoder passes. "
                                               "Not available for the triple-page model.")
                run = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=2):
                stats = gr.Markdown()
                stream = gr.HTML(label="Output stream")
                text = gr.Textbox(label="Transcription (layout tokens removed)", lines=10)
                regions = gr.Dataframe(headers=["region", "text"], label="Regions", wrap=True)
        gr.Examples(
            examples=[[os.path.join(EXAMPLES, "read2016_page_test_11.jpg"), "hand-read2016-page", 150, False],
                      [os.path.join(EXAMPLES, "read2016_double_page_test_23.jpg"), "hand-read2016-double-page", 150, True],
                      [os.path.join(EXAMPLES, "read2016_triple_page_test_5.jpg"), "hand-read2016-triple-page", 150, False]],
            inputs=[image, name, dpi, speculative],
            label="READ 2016 test images (CC BY 4.0): single, double and triple page")
        wdir = gr.State(weights_dir)
        run.click(transcribe, [image, name, dpi, speculative, wdir], [stream, text, regions, stats])
        gr.Markdown(
            "Weights: CC BY 4.0, adapted from D. Coquenet's DAN line model (Zenodo 10.5281/zenodo.7244382). "
            "Example images: READ 2016 (Zenodo 10.5281/zenodo.1297399), CC BY 4.0.")
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    ap.add_argument("--weights-dir", default=os.path.join(ROOT, "weights"))
    a = ap.parse_args()
    build(a.weights_dir).launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
