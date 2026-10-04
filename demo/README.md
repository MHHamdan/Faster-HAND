# Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/Faster-HAND/blob/main/demo/FasterHAND_demo.ipynb)

A Gradio app that reads a handwritten document image with one of the released FasterHAND models.
The quickest way to try it is the Colab notebook
[`FasterHAND_demo.ipynb`](FasterHAND_demo.ipynb): run all cells and open the printed
`gradio.live` link.

```bash
pip install -r requirements.txt -r demo/requirements.txt
python demo/app.py                 # http://127.0.0.1:7860
python demo/app.py --share         # also prints a temporary public gradio.live link
```

On first use of a model the app downloads its package from the GitHub release into `weights/`
and verifies the pinned SHA-256 (`release/hand_release/hub.py`). It runs on CPU or GPU; on CPU a
single page takes several seconds, a triple page considerably longer.

## What it shows

- **Output stream** — the model's single sequence, with layout tokens as colored tags:
  page (`P`), page number (`N`), section (`S`), annotation (`A`), body (`B`); `/` closes.
- **Transcription** — the same text with layout tokens removed.
- **Regions** — the regions parsed from the tags, in reading order.

FasterHAND predicts layout tokens, not coordinates, so the app draws no boxes on the image.

## Choosing a model and resolution

| Model | Use for |
|---|---|
| Single page / single page, compact | one page per image |
| Double page (adapted) | two facing pages in one image |
| Triple page (adapted) | three pages side by side |

Each model reads the number of pages it was trained on: the single-page model stops after the
first page of a wider image, and the adapted models keep going on narrower ones. The app warns
when the number of page elements in the output differs from the model's.

Images are resampled to the 150 dpi working resolution: enter 300 for a raw READ 2016 scan and
150 for the examples. The models were trained on READ 2016 (Early Modern German, 1470–1805);
other scripts, languages and layouts are outside what they have seen.

## Examples and attribution

`examples/` holds three READ 2016 test images at 150 dpi: single page `test_11`, double page
`test_23`, triple page `test_5`. READ 2016: Sánchez, Romero, Toselli and Vidal, Zenodo
[10.5281/zenodo.1297399](https://doi.org/10.5281/zenodo.1297399), CC BY 4.0. The weights are
CC BY 4.0, adapted from D. Coquenet's DAN line model, Zenodo
[10.5281/zenodo.7244382](https://doi.org/10.5281/zenodo.7244382).

## Hosting

GitHub runs no Python servers, so the demo cannot be served from the repository page. The Colab
notebook is the free hosted option: Colab runs the app and Gradio's `share=True` gives it a
public link for as long as the notebook runs. The app can also be hosted unchanged on any Gradio
host, for example a Hugging Face Space: keep `demo/app.py` as the entry point and install
`requirements.txt` and `demo/requirements.txt`.
