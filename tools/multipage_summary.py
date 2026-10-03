#!/usr/bin/env python3
"""Collect the multi-page experiment JSONs into the final tables (JSON + Markdown).

Inputs (all produced by tools/multipage_eval.py unless stated):
  experiments/multipage/zero_shot/<CONFIG>_<level>[<tag>].json    zero-shot sweep
  experiments/multipage/adaptation/<CONFIG>_<level>.json           adapted checkpoints
  experiments/multipage/v1/read_double_page.json, read_triple_page.json   tools/evaluate_hand.py (V1)
Outputs:
  experiments/multipage/tables.json, experiments/multipage/tables.md

DPER / TPER. The document-level error rate of a double / triple page is the corpus CER over
the concatenated multi-page target (sum of edits / sum of characters over the split's
documents). Under this harness that is exactly the CER column of the double_page / triple_page
rows, so the tables print it as such instead of inventing a distinct number.
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP = os.path.join(ROOT, "experiments", "multipage")
LEVELS = ["page", "double_page", "triple_page"]
SCALE = {"page": "single", "double_page": "double", "triple_page": "triple"}
CONFIG_LABEL = {"BASE": "HAND page model (BASE, KV-cache greedy)", "E3": "+E3 (speculative m=5)",
                "E4": "+E4 (shared K/V, KV-cache greedy)", "E3E4": "+E3+E4 (E4 + its heads, m=5)"}


def load_dir(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(p).startswith(("summary_", "tables")):
            continue
        r = json.load(open(p, encoding="utf-8"))
        if "aggregate" not in r:
            continue
        tag = os.path.basename(p)[:-5]
        out[tag] = r
    return out


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


def first_page_diag(r):
    """Zero-shot diagnostic: CER of the whole prediction against the FIRST page of the multi-page
    ground truth, and how many documents end after exactly one ⓟ...Ⓟ block. Tells 'reads one page
    well, then stops' apart from 'reads badly'."""
    import pickle
    import editdistance
    strip = lambda t: "".join(c for c in t if not ("ⓐ" <= c <= "ⓩ" or "Ⓐ" <= c <= "Ⓩ"))
    lp = os.path.join(ROOT, r["data"], "labels.pkl")
    if not os.path.exists(lp) or r["level"] == "page":
        return None
    gt = pickle.load(open(lp, "rb"))["ground_truth"][r["split"]]
    e = n = 0
    one_block = 0
    closest = []
    for name, raw in r["raw_predictions"].items():
        pages = [p.lstrip("ⓟ") for p in gt[name]["text"].split("Ⓟ") if p]
        pred = strip(raw)
        ds = [editdistance.eval(pred, strip(p)) for p in pages]
        closest.append(int(min(range(len(ds)), key=lambda i: ds[i])))
        e += ds[0]
        n += len(strip(pages[0]))
        one_block += int(raw.count("ⓟ") == 1 and raw.count("Ⓟ") == 1)
    return {"cer_vs_first_page": e / n if n else None, "docs_with_exactly_one_page_block": one_block,
            "closest_gt_page_histogram": {str(k): closest.count(k) for k in sorted(set(closest))}, "n": len(closest)}


def row_of(r):
    a = r["aggregate"]
    return {"first_page_diag": first_page_diag(r),"method": CONFIG_LABEL.get(r["config"], r["config"]), "config": r["config"], "level": r["level"],
            "scale": SCALE[r["level"]], "n": a["n"], "cer": a["cer"], "wer": a["wer"],
            "dper_tper": a["cer"] if r["level"] != "page" else None,
            "loer": a["loer"], "map_cer": a["map_cer"], "T_enc_mean": a["T_enc_mean"],
            "tokens_mean": a["tokens_mean"], "passes_mean": a["passes_mean"],
            "latency_s_mean": a["latency_s_mean"], "latency_s_median": a["latency_s_median"],
            "peak_mem_MiB_mean": a["peak_mem_MiB_mean"], "peak_mem_MiB_max": a["peak_mem_MiB_max"],
            "truncated_n": a["truncated_n"], "failed_n": a["failed_n"], "stop_reasons": a["stop_reasons"],
            "len_ratio_mean": a.get("len_ratio_mean"), "decode_path": r["decode_path"], "dtype": r["dtype"],
            "max_tokens": r["max_tokens"], "parameters": r["parameters"], "export_dir": r["export_dir"],
            "identity": (r.get("identity_vs_ref") or {}).get("identical_n"),
            "json": os.path.relpath(r["_path"], ROOT)}


def md_table(rows, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) if not isinstance(r.get(c), dict) else json.dumps(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main():
    zs = load_dir(os.path.join(MP, "zero_shot"))
    ad = load_dir(os.path.join(MP, "adaptation"))
    for d, src in ((zs, "zero_shot"), (ad, "adaptation")):
        for k, r in d.items():
            r["_path"] = os.path.join(MP, src, k + ".json")
    tables = {"definitions": {
        "CER/WER": "corpus-level, hand.basic.metric_manager (WER tokeniser format_string_for_wer)",
        "DPER/TPER": "document-level CER over the concatenated double/triple-page target = the CER of the double/triple rows (identical by definition under this harness)",
        "LOER/mAP-CER": "hand.basic.metric_manager with the READ post-processing module, five-token scheme",
        "T_enc": "encoder grid H_f x W_f (mean over the split)", "tokens": "emitted characters incl. layout tokens (mean)",
        "latency": "wall-clock s/document, CUDA-synchronised, batch 1, warm-up 2, within one process per configuration",
        "peak mem": "torch.cuda.max_memory_allocated reset per document (MiB)",
        "truncation": "documents that hit the token budget (3000/6000/9000)"}}

    # (a) scale table: every zero-shot row, AMP only (fp32 rows listed separately)
    zrows = [row_of(r) for k, r in zs.items() if not k.endswith("_fp32")]
    for r in zrows:
        d = r.get("first_page_diag") or {}
        r["cer_first_page"] = d.get("cer_vs_first_page")
        r["one_block"] = d.get("docs_with_exactly_one_page_block")
    zrows.sort(key=lambda r: (list(CONFIG_LABEL).index(r["config"]) if r["config"] in CONFIG_LABEL else 9, LEVELS.index(r["level"])))
    tables["a_scale"] = zrows
    fp32 = [row_of(r) for k, r in zs.items() if k.endswith("_fp32")]
    tables["a_scale_fp32"] = fp32

    # (b) scaling ratios single -> double -> triple per configuration
    scaling = {}
    for cfg in sorted({r["config"] for r in zrows}):
        per = {r["level"]: r for r in zrows if r["config"] == cfg}
        if "page" not in per:
            continue
        base = per["page"]
        scaling[cfg] = {}
        for lv in LEVELS:
            if lv not in per:
                continue
            r = per[lv]
            scaling[cfg][lv] = {
                "cer": r["cer"], "wer": r["wer"], "loer": r["loer"], "map_cer": r["map_cer"],
                "cer_x_page": r["cer"] / base["cer"] if base["cer"] else None,
                "T_enc_x_page": r["T_enc_mean"] / base["T_enc_mean"],
                "tokens_x_page": r["tokens_mean"] / base["tokens_mean"],
                "latency_x_page": r["latency_s_mean"] / base["latency_s_mean"],
                "latency_per_token_ms": 1000.0 * r["latency_s_mean"] / r["tokens_mean"],
                "peak_mem_x_page": r["peak_mem_MiB_mean"] / base["peak_mem_MiB_mean"],
                "truncated_n": r["truncated_n"], "failed_n": r["failed_n"], "n": r["n"]}
    tables["b_scaling"] = scaling

    # (c) adaptation rows
    arows = [row_of(r) for r in ad.values()]
    arows.sort(key=lambda r: (r["config"], LEVELS.index(r["level"])))
    tables["c_adaptation"] = arows

    # V1 baselines (evaluate_hand.py records)
    v1 = {}
    for name in ("read_double_page", "read_triple_page"):
        p = os.path.join(MP, "v1", name + ".json")
        if os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            m = r["splits"].get("test", {}).get("metrics", {})
            v1[name] = {"cer": m.get("cer"), "wer": m.get("wer"), "nb_chars": m.get("nb_chars"), "nb_words": m.get("nb_words"),
                        "nb_samples": m.get("nb_samples"), "sample_time_s": m.get("sample_time"),
                        "loer_v1_scheme_not_comparable": m.get("loer"), "checkpoint": r["checkpoint"],
                        "level": r["level"], "json": os.path.relpath(p, ROOT),
                        "note": "V1 three-token scheme (ⓟⓐⓑ), consecutive-scan construction, HAND V1 encoder/decoder; not the main line"}
    tables["d_v1_baselines"] = v1

    json.dump(tables, open(os.path.join(MP, "tables.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    md = ["# Multi-page experiment tables", "",
          "Generated by tools/multipage_summary.py from experiments/multipage/{zero_shot,adaptation,v1}/*.json.", "",
          "DPER/TPER = document-level CER over the concatenated double/triple-page target; under this harness that is "
          "exactly the CER of the double/triple rows (sum of edits / sum of characters over the split), so no distinct "
          "number is printed.", "",
          "## (a) Scale table (zero-shot, AMP fp16, batch 1, GPU 0)", "",
          md_table(zrows, ["method", "scale", "n", "cer", "wer", "dper_tper", "loer", "map_cer", "T_enc_mean", "tokens_mean",
                           "passes_mean", "latency_s_mean", "peak_mem_MiB_mean", "truncated_n", "failed_n", "identity", "cer_first_page", "one_block"],
                   ["Method", "Scale", "n", "CER", "WER", "DPER/TPER", "LOER", "mAP-CER", "T_enc", "tokens", "passes",
                    "latency s", "peak MiB", "trunc.", "fail", "identical to greedy", "CER vs 1st page", "docs w/ 1 page block"]), ""]
    if fp32:
        md += ["### fp32 rows", "", md_table(fp32, ["method", "scale", "n", "cer", "wer", "loer", "map_cer", "latency_s_mean", "peak_mem_MiB_mean"],
                                              ["Method", "Scale", "n", "CER", "WER", "LOER", "mAP-CER", "latency s", "peak MiB"]), ""]
    md += ["## (b) Scaling single -> double -> triple (ratios vs the single-page row of the same process)", ""]
    for cfg, per in scaling.items():
        md += ["### %s" % CONFIG_LABEL.get(cfg, cfg), "",
               "| level | CER | WER | LOER | mAP-CER | CER x | T_enc x | tokens x | latency x | ms/token | peak mem x | trunc | fail |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for lv, s in per.items():
            md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d/%d | %d |" % (
                lv, fmt(s["cer"]), fmt(s["wer"]), fmt(s["loer"]), fmt(s["map_cer"]), fmt(s["cer_x_page"], 2),
                fmt(s["T_enc_x_page"], 2), fmt(s["tokens_x_page"], 2), fmt(s["latency_x_page"], 2),
                fmt(s["latency_per_token_ms"], 2), fmt(s["peak_mem_x_page"], 2), s["truncated_n"], s["n"], s["failed_n"]))
        md.append("")
    md += ["## (c) Adaptation (fine-tuned checkpoints)", "",
           md_table(arows, ["config", "scale", "n", "cer", "wer", "loer", "map_cer", "tokens_mean", "latency_s_mean", "truncated_n", "failed_n", "export_dir"],
                    ["Config", "Scale", "n", "CER", "WER", "LOER", "mAP-CER", "tokens", "latency s", "trunc.", "fail", "export"]) if arows else "(none)", "",
           "## (d) V1 baselines (three-token scheme, own test sets; secondary evidence)", ""]
    for k, v in v1.items():
        md.append("- %s: CER %s WER %s (n=%s, %s chars) checkpoint %s -> %s" % (k, fmt(v["cer"]), fmt(v["wer"]), v["nb_samples"], v["nb_chars"], v["checkpoint"], v["json"]))
    open(os.path.join(MP, "tables.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
