#!/usr/bin/env python3
"""Multi-page scaling evaluation of a released HAND checkpoint (zero-shot or adapted).

One process decodes every requested level (page / double_page / triple_page) with the SAME
loaded model, so cross-level ratios (latency, memory, sequence length) are within-process.
Per sample it records CER, WER, LOER, mAP-CER (the training harness's MetricManager, i.e. the
same definitions and the same READ post-processing as every published number), the encoder
grid T_enc = H_f x W_f, the number of emitted tokens, wall-clock latency (CUDA-synchronised),
peak GPU memory (reset per sample), decoder passes (forward hook; = tokens + 1 for greedy, fewer
for speculative), the stop reason (end token / token budget / line cap / per-line cap) and
failures (exception, empty output, no closing page token).

Decoding budget: the page model's config caps max_char_prediction at 3000. For multi-page
documents the budget is raised explicitly (--max-tokens, default 3000 / 6000 / 9000 for
page / double / triple, i.e. LEVEL_DEFAULTS of tools/train_hand.py), and the line cap
(max_line_pred 100 in the harness) is raised to --max-lines (default 1000) so that only the
token budget and the per-line runaway guard (150 characters) can stop a document early.

Usage (AMP fp16, KV-cache greedy, all three levels):
  CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock python3 tools/multipage_eval.py \
      --export-dir outputs/export_e14 --config-name BASE --kv-cache --amp \
      --level page double_page triple_page --out-dir experiments/multipage/zero_shot
Speculative (E3) with identity check against the greedy run of the same base:
  ... --export-dir outputs/export_e14 --heads outputs/export_e14/spec_heads_m5.safetensors \
      --speculative --m 5 --config-name E3 --identity-ref BASE
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "release"))

from hand.basic.metric_manager import MetricManager, compute_global_mAP  # noqa: E402
from hand_release.inference import HANDRecognizer, PageResult, parse_regions, strip_layout  # noqa: E402

LEVEL_DEFAULT_TOKENS = {"page": 3000, "double_page": 6000, "triple_page": 9000}
LEVEL_PAGES = {"page": 1, "double_page": 2, "triple_page": 3}
METRICS = ["cer", "wer", "time", "loer", "map_cer"]


LOER_TIMEOUT_S = 120


class _time_limit:
    """SIGALRM-based time limit (main thread only); raises TimeoutError."""

    def __init__(self, seconds):
        self.seconds = int(seconds)

    def _handler(self, signum, frame):
        raise TimeoutError("graph edit distance exceeded %d s" % self.seconds)

    def __enter__(self):
        import signal
        self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)

    def __exit__(self, *a):
        import signal
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)
        return False


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class InstrumentedRecognizer(HANDRecognizer):
    """HANDRecognizer with (a) an explicit line cap, (b) the stop reason recorded, (c) the encoder
    grid size returned, (d) decoder passes counted. Decoding code paths are the parent's."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stop_reason = None
        self.passes = 0
        self.last_fsize = None
        self.decoder.register_forward_hook(self._count)

    def _count(self, *a):
        self.passes += 1

    def _stop(self, tok, line_count, char_in_line):
        line_count, char_in_line, stop = super()._stop(tok, line_count, char_in_line)
        if stop and self.stop_reason is None:
            self.stop_reason = "line_cap" if line_count >= self.max_lines else "per_line_cap"
        return line_count, char_in_line, stop

    force_pages = 0        # >0: forced-continuation stress test, see _decode_greedy_forced
    forced_restarts = 0

    def _decode_greedy_forced(self, pf, fsize, reduced, budget):
        """Greedy KV-cache decode in which the END token is REFUSED while fewer than
        `force_pages` page blocks have been closed: the open-page token ⓟ is appended instead and
        decoding continues. A diagnostic, not a decoding path of the model: it separates 'the
        page model learned to stop after one ⓟ...Ⓟ' from 'the page model cannot attend beyond
        the first page of a wider image'."""
        dec = self.decoder
        dec.reset_mem_kv_cache()
        tok_open = self.charset.index("ⓟ")
        tok_close = self.charset.index("Ⓟ")
        seq = [self.tok_start]
        ids, conf, cache = [], [], None
        line_count = char_in_line = 0
        truncated = True
        closed = 0
        self.forced_restarts = 0
        for _ in range(budget):
            toks = torch.tensor([seq], dtype=torch.long, device=self.device)
            plen = torch.tensor([len(seq)], dtype=torch.int, device=self.device)
            _, pred, cache, _ = dec(pf, pf, toks, reduced, plen, fsize, start=0, cache=cache,
                                    num_pred=1, padding_value=self.tok_pad, use_mem_cache=True)
            probs = torch.softmax(pred[0, :, -1].float(), dim=0)
            t = int(torch.argmax(pred[0, :, -1]))
            if t == self.tok_end:
                if closed >= self.force_pages:
                    truncated = False
                    break
                t = tok_open
                self.forced_restarts += 1
            if t == tok_close:
                closed += 1
            ids.append(t)
            conf.append(float(probs[t]))
            seq.append(t)
            line_count, char_in_line, stop = self._stop(t, line_count, char_in_line)
            if stop:
                truncated = False
                break
        return ids, conf, truncated

    def read(self, image, source_dpi=None, max_tokens=None, amp=False):
        """Copy of HANDRecognizer.read that also exposes the encoder grid; the decoders are the
        parent's unchanged methods (or the forced-continuation diagnostic when force_pages > 0)."""
        self.stop_reason, self.passes = None, 0
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
            self.last_fsize = (int(fsize[2]), int(fsize[3]), int(h_img), int(w_img))
            pf = self.decoder.features_updater.get_pos_features(feats)
            pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
            if self.force_pages > 0:
                ids, conf, truncated = self._decode_greedy_forced(pf, fsize, reduced, budget)
                path = "kv_cache_forced_%d_pages" % self.force_pages
            elif self.speculative:
                ids, conf, truncated = self._decode_speculative(pf, fsize, reduced, budget)
                path = "speculative_m%d" % self.m
            else:
                ids, conf, truncated = self._decode_greedy(pf, fsize, reduced, budget)
                path = "kv_cache" if self.kv_cache else "reference"
        if truncated:
            self.stop_reason = "token_budget"
        elif self.stop_reason is None:
            self.stop_reason = "end_token"
        raw = "".join(self.charset[i] for i in ids if 0 <= i < self.vocab)
        return PageResult(text=strip_layout(raw), raw=raw, regions=parse_regions(raw),
                          confidence=conf, latency_s=time.time() - t0, n_tokens=len(ids),
                          decode_path=path, truncated=truncated)


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL).decode().strip()
        util = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL).decode().strip()
        return {"compute_apps": out.splitlines(), "gpus": util.splitlines(), "pid": os.getpid()}
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}


def load_split(data_dir, split):
    import pickle
    labels = pickle.load(open(os.path.join(data_dir, "labels.pkl"), "rb"))
    gts = labels["ground_truth"][split]
    names = sorted(gts, key=lambda n: int(n.split("_")[1].split(".")[0]))
    return names, gts


def evaluate_level(r, level, data_dir, split, budget, amp, warmup, limit, pred_dir, config_name):
    names, gts = load_split(data_dir, split)
    if limit:
        names = names[:limit]
    mm = MetricManager(METRICS, "READ_2016-%s" % split)
    rows, raws = [], {}
    # warm-up documents are decoded and discarded; every document is still scored
    order = [("warm", n) for n in names[:warmup]] + [("score", n) for n in names]
    for kind, name in order:
        img = os.path.join(data_dir, split, name)
        gt = gts[name]["text"]
        if r.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        err = None
        try:
            out = r.read(img, max_tokens=budget, amp=amp)
        except Exception:
            err = traceback.format_exc()[-2000:]
            out = None
        if r.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if r.device.type == "cuda" else float("nan")
        if kind == "warm":
            print("  [warm-up] %s %.2fs" % (name, dt), flush=True)
            continue
        hf, wf, h_img, w_img = r.last_fsize if r.last_fsize else (0, 0, 0, 0)
        row = {"name": name, "level": level, "gt_chars": len(strip_layout(gt)),
               "gt_lines": gt.count("\n") + 1, "img_h": h_img, "img_w": w_img,
               "H_f": hf, "W_f": wf, "T_enc": hf * wf, "latency_s": dt,
               "peak_mem_MiB": peak, "error": err}
        if out is None:
            raw, conf = "", []
            row.update({"tokens": 0, "passes": r.passes, "truncated": False,
                        "stop_reason": "exception", "failure": "exception"})
        else:
            raw, conf = out.raw, out.confidence
            n_open, n_close = raw.count("ⓟ"), raw.count("Ⓟ")
            failure = None
            if len(raw.strip()) == 0:
                failure = "empty"
            elif n_close < LEVEL_PAGES[level]:
                failure = "missing_closing_page_token(%d/%d)" % (n_close, LEVEL_PAGES[level])
            row.update({"tokens": out.n_tokens, "passes": r.passes, "truncated": bool(out.truncated),
                        "stop_reason": r.stop_reason, "failure": failure, "forced_restarts": r.forced_restarts,
                        "n_page_open": n_open, "n_page_close": n_close,
                        "pred_chars": len(out.text),
                        "len_ratio": len(out.text) / max(1, len(strip_layout(gt))),
                        "mean_top1_conf": float(np.mean(conf)) if conf else float("nan")})
        vals = {"nb_samples": 1, "str_y": [gt], "str_x": [raw],
                "confidence_score": [conf], "time": dt, "names": ["READ_2016_" + name]}
        m = mm.compute_metrics(vals, [x for x in METRICS if x != "loer"])
        # LOER is an exact graph edit distance (networkx optimize_graph_edit_distance). When the
        # ground truth is a single page the whole graphs are matched, and a prediction that
        # hallucinates many page blocks (e.g. an adapted double-page model on a single page emitting
        # 8 ⓟ blocks) makes the search exponential. Guard it with a per-document time limit; a timed
        # out document contributes no LOER term and is counted in aggregate["loer_timeouts"].
        loer_timeout = False
        try:
            with _time_limit(LOER_TIMEOUT_S):
                ml = mm.compute_metrics(vals, ["loer"])
            m.update({k: ml[k] for k in ("edit_graph", "nb_nodes_and_edges", "nb_pp_op_layout",
                                         "nb_gt_layout_token") if k in ml})
        except TimeoutError:
            loer_timeout = True
            m["edit_graph"], m["nb_nodes_and_edges"] = [float("nan")], [float("nan")]
        mm.update_metrics({k: v for k, v in m.items() if not (loer_timeout and k in ("edit_graph", "nb_nodes_and_edges"))})
        row.update({"edit_chars": int(m["edit_chars"][0]), "nb_chars": int(m["nb_chars"][0]),
                    "cer": m["edit_chars"][0] / max(1, m["nb_chars"][0]),
                    "edit_words": int(m["edit_words"][0]), "nb_words": int(m["nb_words"][0]),
                    "wer": m["edit_words"][0] / max(1, m["nb_words"][0]),
                    "edit_graph": float(m["edit_graph"][0]),
                    "nb_nodes_and_edges": float(m["nb_nodes_and_edges"][0]),
                    "loer": (None if loer_timeout else
                             m["edit_graph"][0] / m["nb_nodes_and_edges"][0] if m["nb_nodes_and_edges"][0] else float("nan")),
                    "loer_timeout": loer_timeout,
                    "map_cer": float(compute_global_mAP([m["map_cer"][0]])),
                    "tokens_per_pass": row["tokens"] / max(1, row["passes"])})
        rows.append(row)
        raws[name] = raw
        with open(os.path.join(pred_dir, "%s_%s_%s.txt" % (config_name, level, os.path.splitext(name)[0])),
                  "w", encoding="utf-8") as f:
            f.write(raw)
        print("  %-12s %-14s CER %.4f WER %.4f LOER %s mAP %.3f tok %5d pass %5d T_enc %6d %.2fs %6.0fMiB %s%s"
              % (level, name, row["cer"], row["wer"], "T/O" if loer_timeout else "%.3f" % row["loer"], row["map_cer"], row["tokens"],
                 row["passes"], row["T_enc"], dt, peak, row["stop_reason"],
                 (" FAIL:" + row["failure"]) if row["failure"] else ""), flush=True)
    d = mm.get_display_values(output=True)
    lat = [x["latency_s"] for x in rows]
    agg = {"n": len(rows), "cer": float(d["cer"]), "wer": float(d["wer"]),
           "loer": float(d["loer"]) if "loer" in d else None,
           "map_cer": float(d["map_cer"]) if "map_cer" in d else None,
           "edit_chars": int(sum(x["edit_chars"] for x in rows)),
           "nb_chars": int(sum(x["nb_chars"] for x in rows)),
           "edit_words": int(sum(x["edit_words"] for x in rows)),
           "nb_words": int(sum(x["nb_words"] for x in rows)),
           "latency_s_mean": float(np.mean(lat)), "latency_s_median": float(np.median(lat)),
           "latency_s_total": float(np.sum(lat)),
           "tokens_mean": float(np.mean([x["tokens"] for x in rows])),
           "tokens_max": int(max(x["tokens"] for x in rows)),
           "passes_mean": float(np.mean([x["passes"] for x in rows])),
           "tokens_per_pass_mean": float(np.mean([x["tokens_per_pass"] for x in rows])),
           "T_enc_mean": float(np.mean([x["T_enc"] for x in rows])),
           "T_enc_max": int(max(x["T_enc"] for x in rows)),
           "peak_mem_MiB_mean": float(np.mean([x["peak_mem_MiB"] for x in rows])),
           "peak_mem_MiB_max": float(np.max([x["peak_mem_MiB"] for x in rows])),
           "truncated_n": int(sum(1 for x in rows if x["truncated"])),
           "truncation_rate": float(np.mean([1.0 if x["truncated"] else 0.0 for x in rows])),
           "stop_reasons": {k: int(sum(1 for x in rows if x["stop_reason"] == k))
                            for k in sorted(set(x["stop_reason"] for x in rows))},
           "loer_timeouts": int(sum(1 for x in rows if x.get("loer_timeout"))),
           "loer_note": "LOER aggregated over documents whose graph edit distance finished within %d s" % LOER_TIMEOUT_S,
           "failed_n": int(sum(1 for x in rows if x["failure"])),
           "failure_rate": float(np.mean([1.0 if x["failure"] else 0.0 for x in rows])),
           "len_ratio_mean": float(np.mean([x.get("len_ratio", 0.0) for x in rows])),
           "gt_chars_mean": float(np.mean([x["gt_chars"] for x in rows]))}
    return agg, rows, raws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, help="release dir (config.json, charset.json, model.safetensors)")
    ap.add_argument("--heads", default=None, help="draft heads (.safetensors or heads.pt); needed with --speculative")
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--kv-cache", action="store_true")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (the paper's setting); omit for fp32")
    ap.add_argument("--level", nargs="+", default=["page", "double_page", "triple_page"])
    ap.add_argument("--data", nargs="*", default=[],
                    help="level=dir overrides; default formatted/READ_2016_<level>_sem_dan")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-tokens", nargs="*", type=int, default=None,
                    help="per-level decoding budget, same order as --level (default 3000/6000/9000)")
    ap.add_argument("--max-lines", type=int, default=1000, help="line cap (harness default 100)")
    ap.add_argument("--max-per-line", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config-name", required=True, help="label, e.g. BASE, E4, E3, E3E4")
    ap.add_argument("--identity-ref", default=None,
                    help="config name whose per-level JSON in --out-dir holds the greedy reference strings")
    ap.add_argument("--out-dir", default="experiments/multipage/zero_shot")
    ap.add_argument("--pred-dir", default="outputs/multipage_predictions")
    ap.add_argument("--tag", default="", help="suffix for the output files (e.g. _fp32)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force-pages", action="store_true",
                    help="diagnostic: refuse the end token until the level's page count of ⓟ...Ⓟ blocks is closed")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(ROOT, a.out_dir)
    pred_dir = a.pred_dir if os.path.isabs(a.pred_dir) else os.path.join(ROOT, a.pred_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    data = {lv: os.path.join(ROOT, "formatted", "READ_2016_%s_sem_dan" % lv) for lv in a.level}
    for kv in a.data:
        k, v = kv.split("=", 1)
        data[k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    budgets = {lv: LEVEL_DEFAULT_TOKENS[lv] for lv in a.level}
    if a.max_tokens:
        assert len(a.max_tokens) == len(a.level), "--max-tokens must match --level"
        budgets = dict(zip(a.level, a.max_tokens))

    export = a.export_dir if os.path.isabs(a.export_dir) else os.path.join(ROOT, a.export_dir)
    heads = a.heads if (a.heads is None or os.path.isabs(a.heads)) else os.path.join(ROOT, a.heads)
    torch.manual_seed(0)
    r = InstrumentedRecognizer.from_pretrained(export, device=a.device, kv_cache=a.kv_cache,
                                               speculative=a.speculative, m=a.m, heads_path=heads)
    r.__class__ = InstrumentedRecognizer   # from_pretrained builds cls(...) so this is already true
    r.max_lines, r.max_per_line = a.max_lines, a.max_per_line
    n_params = (sum(p.numel() for p in r.encoder.parameters())
                + sum(p.numel() for p in r.decoder.parameters()))
    path = "speculative_m%d" % a.m if a.speculative else ("kv_cache" if a.kv_cache else "reference")
    if a.force_pages:
        path += "+forced_continuation"
    print("config %s  export %s  params %d  path %s  amp %s  heads %s" % (
        a.config_name, os.path.relpath(export, ROOT), n_params, path, a.amp, a.heads), flush=True)
    env = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(),
           "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu_snapshot_start": gpu_snapshot(), "argv": sys.argv}

    summary = {}
    for lv in a.level:
        print("=== level %s  data %s  budget %d" % (lv, os.path.relpath(data[lv], ROOT), budgets[lv]), flush=True)
        r.force_pages = LEVEL_PAGES[lv] if a.force_pages else 0
        agg, rows, raws = evaluate_level(r, lv, data[lv], a.split, budgets[lv], a.amp, a.warmup,
                                         a.limit, pred_dir, a.config_name + a.tag)
        identity = None
        if a.identity_ref:
            ref_path = os.path.join(out_dir, "%s_%s%s.json" % (a.identity_ref, lv, a.tag))
            if os.path.exists(ref_path):
                ref = json.load(open(ref_path, encoding="utf-8"))
                ref_raw = ref.get("raw_predictions", {})
                same = [n for n in raws if n in ref_raw and ref_raw[n] == raws[n]]
                diff = [n for n in raws if n in ref_raw and ref_raw[n] != raws[n]]
                identity = {"ref": os.path.relpath(ref_path, ROOT), "identical_n": len(same),
                            "compared_n": len(same) + len(diff), "differing": diff,
                            "all_identical": len(diff) == 0 and len(same) > 0}
            else:
                identity = {"ref": os.path.relpath(ref_path, ROOT), "error": "reference JSON not found"}
        res = {"config": a.config_name, "level": lv, "split": a.split, "n_pages_per_doc": LEVEL_PAGES[lv],
               "export_dir": os.path.relpath(export, ROOT), "heads": a.heads, "parameters": n_params,
               "decode_path": path, "amp": a.amp, "dtype": "fp16-autocast" if a.amp else "fp32",
               "max_tokens": budgets[lv], "max_lines": a.max_lines, "max_per_line": a.max_per_line,
               "warmup": a.warmup, "data": os.path.relpath(data[lv], ROOT),
               "metrics_definition": "hand.basic.metric_manager.MetricManager (cer, wer, loer, map_cer with READ post-processing)",
               "aggregate": agg, "identity_vs_ref": identity, "per_sample": rows,
               "raw_predictions": raws, "env": env, "notes": a.notes}
        res["env"]["gpu_snapshot_end"] = gpu_snapshot()
        jp = os.path.join(out_dir, "%s_%s%s.json" % (a.config_name, lv, a.tag))
        json.dump(res, open(jp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        cp = os.path.join(out_dir, "per_sample_%s_%s%s.csv" % (a.config_name, lv, a.tag))
        cols = ["name", "level", "cer", "wer", "loer", "map_cer", "gt_chars", "gt_lines", "pred_chars", "len_ratio",
                "tokens", "passes", "tokens_per_pass", "T_enc", "H_f", "W_f", "img_h", "img_w", "latency_s",
                "peak_mem_MiB", "truncated", "stop_reason", "failure", "n_page_open", "n_page_close",
                "mean_top1_conf", "forced_restarts", "loer_timeout", "edit_chars", "nb_chars", "edit_words", "nb_words"]
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
        summary[lv] = agg
        print("--- %s %s: CER %.4f WER %.4f LOER %s mAP %s  lat %.3fs  tok %.0f  T_enc %.0f  mem %.0fMiB  trunc %d/%d fail %d  identity %s"
              % (a.config_name, lv, agg["cer"], agg["wer"], agg["loer"], agg["map_cer"],
                 agg["latency_s_mean"], agg["tokens_mean"], agg["T_enc_mean"], agg["peak_mem_MiB_mean"],
                 agg["truncated_n"], agg["n"], agg["failed_n"],
                 None if identity is None else identity.get("identical_n")), flush=True)
        print("wrote", os.path.relpath(jp, ROOT), flush=True)
    base = summary.get("page")
    if base:
        for lv, agg in summary.items():
            agg["ratio_vs_page"] = {k: agg[k] / base[k] if base[k] else None
                                    for k in ("latency_s_mean", "tokens_mean", "T_enc_mean", "peak_mem_MiB_mean", "gt_chars_mean")}
    sp = os.path.join(out_dir, "summary_%s%s.json" % (a.config_name, a.tag))
    json.dump({"config": a.config_name, "decode_path": path, "amp": a.amp, "levels": summary, "env": env},
              open(sp, "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.relpath(sp, ROOT))


if __name__ == "__main__":
    main()
