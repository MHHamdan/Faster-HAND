"""HAND page-level inference: image in, {text, raw, regions, confidence, latency_s} out.

This module is the release inference contract. It is deliberately self-contained at the
*orchestration* level: it builds the encoder and the decoder directly, loads the released
weights STRICTLY, and reproduces the formatter's preprocessing exactly. It does not build a
dataset, a Manager or a training harness, so it runs from a single page image.

It imports two model classes from the `hand` package:

    hand.models.baseline.fcn_encoder.FCN_Encoder      - Denis Coquenet, CeCILL-C
    hand.models.baseline.dan_decoder.GlobalHTADecoder - derived from DAN's models_dan.py

Both are third-party or third-party-derived; see release/NOTICE.md. This file itself is the
author's own work and is offered under MIT, which CeCILL-C Art. 5.3.3 / 6.3 permits for a
Related Module that does not modify the Software.

PREPROCESSING (any deviation from this pipeline changes the measured error rates). The
measured pipeline is, in order:

  1. open the page, convert to RGB                      (preprocessings: [{"type": "to_RGB"}])
  2. bring the page to the model's WORKING RESOLUTION of 150 dpi, with PIL BILINEAR
     (read2016_formatter.py:291 and generic_dataset_formatter.py:150-162 -
     `img.resize((int(w*r), int(h*r)), BILINEAR)`, with r = target_dpi / source_dpi).
     READ 2016's raw scans are 300 dpi, so the formatter halved them once when it built
     `formatted/READ_2016_page_sem_dan/`. This step therefore runs ONLY for an image that is
     not already at 150 dpi, and `read()` will not guess: pass `source_dpi=` for a raw scan
     and nothing for a page that is already at the working resolution.
  3. (x - mean) / std with the TRAINING-SET channel statistics, on the 0-255 scale
     (outputs/e14_budget_1p26M_s0/results/params.txt:202-221)
  4. NO further resize, NO padding, NO letterbox. The page keeps its aspect ratio and its
     size; the encoder is fully convolutional.

There is no 1024x2048 LANCZOS step and no ImageNet mean/std anywhere in the measured path.

Getting step 2 wrong is not a subtle error but it is a SILENT one: resizing an already-150-dpi
page by another 0.5 was measured here to turn `Aūch das Er der / Leib aigenschafft freȳ seie`
into `Aūch das Ereistie / ist gegen`, i.e. it drops a whole line, while the model still
returns confident-looking output (mean top-1 0.92). Check the encoder grid, not the prose:
a 1755 x 1161 page must give 55 x 146 = 8,030 visual positions, which is the value
`efficiency_hand_vs_dan_uncontended.json` records for test_0.

DECODING. Three paths, all producing the same string:

  reference   the frozen decoding path (docs/model_card.md): the cross-attention memory
              keys/values are re-projected at every step.
  kv_cache    the memory K/V projection is computed once per page. Token-identical on 50/50
              pages for both HAND and DAN's published weights
              (experiments/benchmark_suite/profiling/exact_decoding_equivalence.json,
              "all_exact": true). ~1.15x on the measurement host.
  speculative fused draft-and-verify with m-1 look-ahead heads. Every emitted token is one the
              base model's own output layer produced, so the string is identical by
              construction; that identity is asserted page by page in
              spec_decode_test.json (m=5: 50/50 under AMP). 2.73x at m=5 on the
              measurement host (1.960 -> 0.717 s/page).

Both fast paths are OFF by default.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------------------
# Locate the `hand` package, which carries the two model classes imported below.
#
# Two layouts must both work, and a real clone is the one that matters:
#
#   <model repo>/hand_release/inference.py   +  <model repo>/hand/...       <- a snapshot_download
#   <source repo>/release/hand_release/...   +  <source repo>/hand/...      <- this working tree
#
# An already-installed `hand` package takes precedence over both. Nothing here reaches
# outside the directory the file was downloaded into, which is what makes the downloaded
# repository self-sufficient: the previous version computed a path three directories above
# itself and therefore only ever worked on the machine the release was assembled on.
# --------------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)                 # model-repo root, or release/
_SOURCE_ROOT = os.path.dirname(_PKG_PARENT)          # source-repo root, in this working tree
for _cand in (_PKG_PARENT, _SOURCE_ROOT):
    if os.path.isdir(os.path.join(_cand, "hand")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

try:
    from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402
    from hand.models.baseline.fcn_encoder import FCN_Encoder  # noqa: E402
except ImportError as _e:      # pragma: no cover - a diagnosis, not a fallback
    raise ImportError(
        "%s\n\nThis module needs the model classes from the `hand` package:\n"
        "    hand/models/baseline/fcn_encoder.py   (Denis Coquenet, CeCILL-C)\n"
        "    hand/models/baseline/attention.py     (DAN derivative, CeCILL-C)\n"
        "    hand/models/baseline/dan_decoder.py   (DAN derivative, CeCILL-C)\n"
        "    hand/models/baseline/spec_heads.py    (this project, MIT; speculative decoding only)\n"
        "plus the three package __init__.py files. A released model repository must carry "
        "them next to hand_release/; see MODEL_FILES.md section 3. Searched: %s"
        % (_e, [_PKG_PARENT, _SOURCE_ROOT])) from _e

# --------------------------------------------------------------------------------------
# READ 2016 layout tokens. Lower case opens a region, upper case closes it.
# Source: hand/basic/layout_metrics.py:46 (READ_MATCHING_TOKENS), which is in turn
# DAN's read2016_formatter.SEM_MATCHING_TOKENS.
# --------------------------------------------------------------------------------------
LAYOUT_OPEN_TO_CLOSE = {
    "ⓑ": "Ⓑ",   # circled b -> circled B   body
    "ⓐ": "Ⓐ",   # circled a -> circled A   annotation
    "ⓟ": "Ⓟ",   # circled p -> circled P   page
    "ⓝ": "Ⓝ",   # circled n -> circled N   page_number
    "ⓢ": "Ⓢ",   # circled s -> circled S   section
}
LAYOUT_CLASS = {
    "ⓑ": "body",
    "ⓐ": "annotation",
    "ⓟ": "page",
    "ⓝ": "page_number",
    "ⓢ": "section",
}
LAYOUT_TOKENS = set(LAYOUT_OPEN_TO_CLOSE) | set(LAYOUT_OPEN_TO_CLOSE.values())


@dataclass
class Region:
    cls: str
    text: str
    char_span: tuple  # [start, end) into PageResult.text
    line_indices: list = field(default_factory=list)

    def to_dict(self):
        return {"class": self.cls, "text": self.text,
                "char_span": list(self.char_span), "line_indices": self.line_indices}


@dataclass
class PageResult:
    text: str                 # transcription, layout tokens stripped
    raw: str                  # the single interleaved token stream, layout tokens inline
    regions: list             # list[Region]
    confidence: list          # per-emitted-token top-1 probability, aligned with `raw`
    latency_s: float
    n_tokens: int
    decode_path: str
    truncated: bool = False

    def to_dict(self):
        return {"text": self.text, "raw": self.raw,
                "regions": [r.to_dict() for r in self.regions],
                "confidence": self.confidence, "latency_s": self.latency_s,
                "n_tokens": self.n_tokens, "decode_path": self.decode_path,
                "truncated": self.truncated}


def strip_layout(s: str) -> str:
    return "".join(c for c in s if c not in LAYOUT_TOKENS)


def parse_regions(raw: str) -> list:
    """Turn the interleaved stream into regions over the layout-stripped text.

    The five-token scheme nests: page > {section, page_number} > {body, annotation}. This
    returns every opened region, innermost and outermost alike, in opening order, each with
    its span into the stripped text and the indices of the stripped-text lines it covers.

    Unbalanced output is normal for a free-running decoder and is NOT repaired here: a region
    whose close token never arrives is closed at end of stream and flagged by its span
    reaching the end. Tolerating that is the honest behaviour; silently repairing it would
    make LOER look better than it is.
    """
    stack, out, plain = [], [], []
    for ch in raw:
        if ch in LAYOUT_OPEN_TO_CLOSE:
            stack.append((ch, len(plain)))
            continue
        if ch in LAYOUT_OPEN_TO_CLOSE.values():
            opener = next((o for o, c in LAYOUT_OPEN_TO_CLOSE.items() if c == ch), None)
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == opener:
                    o, start = stack.pop(i)
                    out.append((LAYOUT_CLASS[o], start, len(plain)))
                    break
            continue
        plain.append(ch)
    while stack:
        o, start = stack.pop()
        out.append((LAYOUT_CLASS[o], start, len(plain)))
    text = "".join(plain)
    # line index of every character, so a region can name the lines it spans
    line_of, ln = [], 0
    for ch in text:
        line_of.append(ln)
        if ch == "\n":
            ln += 1
    out.sort(key=lambda t: (t[1], -t[2]))
    regions = []
    for cls, a, b in out:
        lines = sorted(set(line_of[a:b])) if b > a else []
        regions.append(Region(cls=cls, text=text[a:b], char_span=(a, b), line_indices=lines))
    return regions


class HANDRecognizer:
    """Read a handwritten page. Text and layout come out of one token stream."""

    def __init__(self, encoder, decoder, charset, config, device,
                 speculative=False, m=5, kv_cache=False, heads=None):
        self.encoder, self.decoder = encoder, decoder
        self.charset = list(charset)
        self.config = config
        self.device = device
        self.vocab = len(self.charset)
        # Index convention, from hand/OCR/ocr_dataset_manager.py:74-99 with
        # charset_mode="seq2seq": end = V, start = V + 1, pad = V + 2.
        self.tok_end, self.tok_start, self.tok_pad = self.vocab, self.vocab + 1, self.vocab + 2
        self.nl = self.charset.index("\n") if "\n" in self.charset else -1
        self.max_chars = int(config.get("max_char_prediction", 3000))
        self.max_lines = int(config.get("max_line_pred", 100))
        self.max_per_line = int(config.get("max_pred_per_line", 150))
        self.mean = np.asarray(config["image_mean"], dtype=np.float64)
        self.std = np.asarray(config["image_std"], dtype=np.float64)
        self.working_dpi = config.get("working_dpi", 150)
        self.expected_visual_divisors = (32, 8)   # encoder reduces H by 32 and W by 8
        self.speculative, self.m, self.heads = speculative, m, heads
        self.kv_cache = kv_cache or speculative   # the fused path requires the memory cache
        if speculative and heads is None:
            raise ValueError("speculative=True needs draft heads; pass heads_path= to "
                             "from_pretrained (spec_heads_m5.safetensors)")

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_pretrained(cls, path, device=None, speculative=False, m=5, kv_cache=False,
                        heads_path=None, dtype=torch.float32):
        """`path` is one of three things, tried in this order:

          * a local directory holding config.json, charset.json and the weights;
          * a local training checkpoint (.pt);
          * a Hugging Face repo id of the shape `owner/name`, which is resolved with
            huggingface_hub.snapshot_download and then treated as a directory.

        The third case is why this classmethod exists in the form the card documents. Without
        it `from_pretrained("MHHamdan/hand-read2016-page")` fell through to torch.load and
        raised FileNotFoundError on a string that was never a path.

        Weights are loaded with strict=True in every case: a silent shape mismatch is how you
        get a model that reads at 86 % CER and looks fine."""
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not os.path.exists(path) and _looks_like_repo_id(path):
            path = _resolve_from_hub(path)
        if os.path.isdir(path):
            config = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
            charset = json.load(open(os.path.join(path, "charset.json"),
                                     encoding="utf-8"))["charset"]
            enc_sd, dec_sd = _load_release_weights(path)
        else:
            config, charset, enc_sd, dec_sd = _load_training_checkpoint(path)

        if len(charset) != config["vocab_size"]:
            raise ValueError("charset.json has %d entries, config.json says vocab_size=%d"
                             % (len(charset), config["vocab_size"]))

        enc = FCN_Encoder({"input_channels": config["input_channels"],
                           "dropout": config["dropout"], "enc_dim": config["enc_dim"],
                           "nb_layers": config["nb_layers"], "device": device})
        enc.load_state_dict(enc_sd, strict=True)
        dec_params = {k: config[k] for k in (
            "enc_dim", "l_max", "dec_pred_dropout", "attention_win", "vocab_size",
            "additional_tokens", "use_line_indices", "two_step_pos_enc_mode", "pe_h_max",
            "pe_w_max", "dec_num_layers", "dec_num_heads", "dec_res_dropout",
            "dec_att_dropout", "dec_dim_feedforward", "use_tokens_from_all_lines",
            "use_first_pass_tokens", "dropout")}
        dec_params["device"] = device
        # E4 (shared visual K/V projections across decoder layers). Opt-in through the exported
        # config; absent or false leaves the decoder exactly as before. Added 2026-10.
        if config.get("dec_share_memory_kv"):
            dec_params["dec_share_memory_kv"] = True
        dec = GlobalHTADecoder(dec_params)
        dec.load_state_dict(dec_sd, strict=True)
        enc.to(device=device, dtype=dtype).eval()
        dec.to(device=device, dtype=dtype).eval()

        n = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
        expected = config.get("expected_parameters")
        if expected is not None and n != expected:
            raise ValueError("loaded %d parameters, config.json expects %d" % (n, expected))

        heads = None
        if speculative:
            if heads_path is None and os.path.isdir(path):
                for cand in ("spec_heads_m5.safetensors", "spec_heads_m5.pt"):
                    if os.path.exists(os.path.join(path, cand)):
                        heads_path = os.path.join(path, cand)
                        break
            if heads_path is None:
                raise FileNotFoundError("speculative=True but no draft-head file was found")
            heads = _load_heads(heads_path, config, device, dtype)
            if m > heads.m:
                raise ValueError("heads were trained for m=%d, asked for m=%d" % (heads.m, m))
        return cls(enc, dec, charset, config, device, speculative, m, kv_cache, heads)

    # ---------------------------------------------------------------- preprocessing
    def preprocess(self, image, source_dpi=None):
        """PIL image or path -> (1, 3, H, W) float tensor on the model's device.

        `source_dpi` is the resolution of the image you are handing in. Pass 300 for a raw
        READ 2016 scan. Pass nothing for a page that is already at the model's 150 dpi
        working resolution, which is what `formatted/READ_2016_page_sem_dan/` holds. It is
        deliberately not guessed: PIL's dpi metadata is absent or wrong on most scans, and
        guessing wrong silently costs whole lines of transcription."""
        img = Image.open(image) if isinstance(image, (str, os.PathLike)) else image
        img = img.convert("RGB")
        if source_dpi and self.working_dpi and int(source_dpi) != int(self.working_dpi):
            r = self.working_dpi / float(source_dpi)
            w, h = img.size
            img = img.resize((int(w * r), int(h * r)), Image.BILINEAR)
        a = (np.asarray(img).astype(np.float64) - self.mean) / self.std
        x = torch.tensor(a, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return x.to(device=self.device, dtype=next(self.encoder.parameters()).dtype)

    # ---------------------------------------------------------------- public API
    def read(self, image, source_dpi=None, max_tokens=None, amp=False):
        t0 = time.time()
        x = self.preprocess(image, source_dpi=source_dpi)
        h_img, w_img = x.shape[-2], x.shape[-1]
        reduced = [[int(np.ceil(h_img / 32)), int(np.ceil(w_img / 8))]]
        ctx = (torch.autocast(device_type=self.device.type, dtype=torch.float16)
               if amp and self.device.type == "cuda" else _null_ctx())
        budget = int(max_tokens) if max_tokens else self.max_chars
        with torch.no_grad(), ctx:
            feats = self.encoder(x)
            fsize = feats.size()
            pf = self.decoder.features_updater.get_pos_features(feats)
            pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
            if self.speculative:
                ids, conf, truncated = self._decode_speculative(pf, fsize, reduced, budget)
                path = "speculative_m%d" % self.m
            else:
                ids, conf, truncated = self._decode_greedy(pf, fsize, reduced, budget)
                path = "kv_cache" if self.kv_cache else "reference"
        raw = "".join(self.charset[i] for i in ids if 0 <= i < self.vocab)
        return PageResult(text=strip_layout(raw), raw=raw, regions=parse_regions(raw),
                          confidence=conf, latency_s=time.time() - t0, n_tokens=len(ids),
                          decode_path=path, truncated=truncated)

    # ---------------------------------------------------------------- decoders
    def _stop(self, tok, line_count, char_in_line):
        if self.nl >= 0 and tok == self.nl:
            line_count, char_in_line = line_count + 1, 0
        else:
            char_in_line += 1
        return line_count, char_in_line, (line_count >= self.max_lines
                                          or char_in_line >= self.max_per_line)

    def _decode_greedy(self, pf, fsize, reduced, budget):
        """Single-token greedy decode. Mirrors trainer_std_hand.evaluate_batch_greedy and
        tools/efficiency_bench.py:decode_page at batch 1."""
        dec = self.decoder
        if self.kv_cache:
            dec.reset_mem_kv_cache()
        seq = [self.tok_start]
        ids, conf, cache = [], [], None
        line_count = char_in_line = 0
        truncated = True
        for _ in range(budget):
            toks = torch.tensor([seq], dtype=torch.long, device=self.device)
            plen = torch.tensor([len(seq)], dtype=torch.int, device=self.device)
            _, pred, cache, _ = dec(pf, pf, toks, reduced, plen, fsize, start=0, cache=cache,
                                    num_pred=1, padding_value=self.tok_pad,
                                    use_mem_cache=self.kv_cache)
            probs = torch.softmax(pred[0, :, -1].float(), dim=0)
            t = int(torch.argmax(pred[0, :, -1]))
            if t == self.tok_end:
                truncated = False
                break
            ids.append(t)
            conf.append(float(probs[t]))
            seq.append(t)
            line_count, char_in_line, stop = self._stop(t, line_count, char_in_line)
            if stop:
                truncated = False
                break
        return ids, conf, truncated

    def _decode_speculative(self, pf, fsize, reduced, budget):
        """Fused draft-and-verify. Lifted from tools/spec_decode.py:spec_decode_page, which is
        the routine the 2.73x measurement was taken with. Every emitted token comes from the
        base model's own distribution, so the string equals the greedy string by construction."""
        dec = self.decoder
        dec.reset_mem_kv_cache()
        seq = [self.tok_start]
        ids, conf, draft, cache = [], [], [], None
        line_count = char_in_line = 0
        done, truncated = False, True
        while not done and len(ids) < budget:
            feed = seq + draft
            n_new = 1 + len(draft)
            toks = torch.tensor([feed], dtype=torch.long, device=self.device)
            plen = torch.tensor([len(feed)], dtype=torch.int, device=self.device)
            out, pred, cache, _ = dec(pf, pf, toks, reduced, plen, fsize, start=0, cache=cache,
                                      num_pred=n_new, padding_value=self.tok_pad,
                                      use_mem_cache=True)
            probs = torch.softmax(pred[0].float(), dim=0)
            argmax = pred[0].argmax(0).tolist()
            top1 = probs.max(0).values.tolist()
            ell = 0
            while ell < len(draft) and argmax[ell] == draft[ell]:
                ell += 1
            for t, c in zip(draft[:ell] + [argmax[ell]], top1[:ell + 1]):
                if t == self.tok_end:
                    done, truncated = True, False
                    break
                ids.append(t)
                conf.append(c)
                seq.append(t)
                line_count, char_in_line, stop = self._stop(t, line_count, char_in_line)
                if stop:
                    done, truncated = True, False
                    break
            # Roll the K/V cache back to the accepted prefix. The pass appended one entry per
            # query position; the rest describe a continuation that did not happen, and leaving
            # them in makes the decode differ from the base model's. See spec_decode.py.
            drop = n_new - (ell + 1)
            if drop > 0 and cache is not None:
                cache = cache[:, :cache.size(1) - drop]
            if done:
                break
            h = out[ell, 0, :].float()
            draft = [int(l.argmax(-1)) for l in self.heads(h)][:self.m - 1]
        return ids, conf, truncated


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _load_release_weights(d):
    """model.safetensors (preferred) or model.pt, split into encoder/decoder sub-dicts."""
    st = os.path.join(d, "model.safetensors")
    pt = os.path.join(d, "model.pt")
    if os.path.exists(st):
        from safetensors.torch import load_file
        flat = load_file(st)
    elif os.path.exists(pt):
        flat = torch.load(pt, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            "%s holds the configuration but no weights. The release directory ships "
            "config.json / charset.json / preprocessor_config.json only; the .safetensors "
            "files are produced by release/tools/export_release_checkpoint.py and are "
            "deliberately not carried in the working tree. See MODEL_FILES.md." % d)
    enc = {k[len("encoder."):]: v for k, v in flat.items() if k.startswith("encoder.")}
    dec = {k[len("decoder."):]: v for k, v in flat.items() if k.startswith("decoder.")}
    if not enc or not dec:
        raise ValueError("%s: expected keys prefixed 'encoder.' and 'decoder.'" % d)
    return enc, dec


def _load_training_checkpoint(path):
    """Read a raw training checkpoint. Provided so the example runs before the safetensors
    export has been produced; the released repo should carry model.safetensors instead."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    charset = list(ck["charset"])
    v = len(charset)
    extra = ck["decoder_state_dict"]["end_conv.weight"].shape[0] - v
    if extra not in (1, 2, 3):
        raise ValueError("%s: implausible additional_tokens=%d" % (path, extra))
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(vocab_size=v, additional_tokens=extra,
               use_line_indices=bool(ck.get("use_line_indices", extra == 3)))
    cfg["expected_parameters"] = (sum(x.numel() for x in ck["encoder_state_dict"].values())
                                 + sum(x.numel() for x in ck["decoder_state_dict"].values()))
    return cfg, charset, ck["encoder_state_dict"], ck["decoder_state_dict"]


def _load_heads(path, config, device, dtype):
    from hand.models.baseline.spec_heads import SpeculativeHeads
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
        meta = json.load(open(os.path.splitext(path)[0] + ".json", encoding="utf-8"))
        m, hidden, vocab_out = meta["m"], meta["hidden"], meta["vocab_out"]
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd, m, hidden = blob["heads"], blob["m"], blob["hidden"]
        vocab_out = blob["vocab_out"]
    heads = SpeculativeHeads(config["enc_dim"], vocab_out, m, hidden)
    heads.load_state_dict(sd, strict=True)
    return heads.to(device=device, dtype=dtype).eval()


_HUB_FILES = ["config.json", "charset.json", "preprocessor_config.json",
              "model.safetensors"]
_HUB_FILES_OPTIONAL = ["spec_heads_m5.safetensors", "spec_heads_m5.json"]


def _looks_like_repo_id(s):
    """`owner/name`, the Hugging Face repo-id shape. Deliberately narrow: exactly one slash,
    no path separators of any other kind, no leading dot, no whitespace, and not something
    that is obviously a filename."""
    if not isinstance(s, str) or s.count("/") != 1 or "\\" in s or s != s.strip():
        return False
    owner, name = s.split("/")
    if not owner or not name or owner.startswith(".") or name.startswith("."):
        return False
    if any(c.isspace() for c in s) or name.endswith((".pt", ".safetensors", ".json")):
        return False
    ok = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return set(owner) <= ok and set(name) <= ok


def _resolve_from_hub(repo_id, revision=None):
    """Download the model repository and return the local directory.

    Only the files the loader actually reads are fetched; the weights are large and there is
    no reason to pull a card, a licence text or an example script into a cache to build a
    model. `snapshot_download` is the documented entry point and honours HF_HOME / HF_TOKEN.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:      # pragma: no cover
        raise ImportError(
            "%r looks like a Hugging Face repo id, but huggingface_hub is not installed. "
            "Either `pip install huggingface_hub`, or download the repository yourself and "
            "pass the local directory to from_pretrained()." % repo_id) from e
    return snapshot_download(repo_id=repo_id, revision=revision,
                             allow_patterns=_HUB_FILES + _HUB_FILES_OPTIONAL)


# The architecture of the released page model. Every value is read back from the training
# configuration recorded at outputs/e14_budget_1p26M_s0/results/params.txt and from
# tools/evaluate_hand.py:build_params, not invented here.
DEFAULT_CONFIG = {
    "model_type": "hand-dan-page",
    "input_channels": 3,
    "dropout": 0.5,
    "enc_dim": 256,
    "nb_layers": 5,
    "pe_h_max": 500,
    "pe_w_max": 1000,
    "l_max": 15000,
    "dec_num_layers": 8,
    "dec_num_heads": 4,
    "dec_res_dropout": 0.1,
    "dec_pred_dropout": 0.1,
    "dec_att_dropout": 0.1,
    "dec_dim_feedforward": 256,
    "attention_win": 100,
    "use_tokens_from_all_lines": True,
    "use_first_pass_tokens": True,
    "two_step_pos_enc_mode": "cat",
    "use_line_indices": False,
    "vocab_size": 99,
    "additional_tokens": 1,
    "max_char_prediction": 3000,
    "max_line_pred": 100,
    "max_pred_per_line": 150,
    "working_dpi": 150,
    "image_mean": [202.70203512453844, 190.85819291654516, 136.08993740340918],
    "image_std": [71.95683449787683, 72.00679752035164, 57.33656372156373],
}
