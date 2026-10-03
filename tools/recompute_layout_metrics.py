"""
Three-way validation of the layout metrics on real predictions.

Input: a prediction dump written by tools/dump_predictions.py (private/, git-ignored).
For every document the prediction is post-processed with the READ PostProcessingModule
(as both DAN and this repository do before LOER/mAP), then LOER is computed by

  (a) this repository's `hand/basic/metric_manager.edit_and_num_items_for_ged_from_str`
      (the code behind the 29.7-35.2 % figures in results_recovery/),
  (b) the DAN-faithful port in `hand/basic/layout_metrics.py`,
  (c) the official DAN implementation itself (`third_party/DAN/basic/metric_manager.py`,
      imported with TensorFlow blocked), which is the ground truth for "DAN's LOER".

(b) must equal (c) on every document. mAP_CER is recomputed the same three ways -- the
repository's `compute_layout_mAP_per_class` / `compute_global_mAP` against DAN's own copies of
those functions -- from the post-processed prediction and its post-processed per-token
confidences, and compared with the number the MetricManager reported during evaluation. The
order-invariant metrics (bWER, hWER, ΔWER, NSFD) are reported alongside CER/WER.

Output: a JSON summary with per-document values (no transcriptions) under
experiments/benchmark_suite/profiling/ and a printed table.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hand.basic import metric_manager as repo_mm  # noqa: E402
from hand.basic.post_pocessing_layout import PostProcessingModuleREAD  # noqa: E402
from hand.basic import layout_metrics as v2  # noqa: E402


def load_dan_metric_manager():
    sys.modules["tensorflow"] = None
    # DAN was written against numpy < 1.24, where `np.int` was an alias for the builtin `int`
    # (removed in numpy 1.24). Restoring the alias here keeps third_party/DAN unmodified and
    # is behaviour-preserving: `np.cumsum(..., dtype=np.int)` == `dtype=int`.
    if not hasattr(np, "int"):
        np.int = int
    dan = os.path.join(ROOT, "third_party", "DAN")
    sys.path.insert(0, dan)
    # DAN's module imports `Datasets.dataset_formatters...` and `basic...` from its own root;
    # our package is `hand.basic`, so no clash as long as DAN's root is first on the path.
    import importlib
    for m in [k for k in sys.modules if k == "basic" or k.startswith("basic.") or k == "Datasets" or k.startswith("Datasets.")]:
        del sys.modules[m]
    mod = importlib.import_module("basic.metric_manager")
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="private/research_audit/predictions/<model>__<dataset>__<split>.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "benchmark_suite", "profiling"))
    a = ap.parse_args()
    d = json.load(open(a.dump))
    samples = d["samples"]
    layout_tokens = "".join(v2.READ_MATCHING_TOKENS.keys()) + "".join(v2.READ_MATCHING_TOKENS.values())
    dan_mm = load_dan_metric_manager()

    rows = []
    tot = {"repo_edit": 0, "repo_norm": 0, "v2_edit": 0, "v2_norm": 0, "dan_edit": 0, "dan_norm": 0,
           "cer_edit": 0, "cer_n": 0, "pp_ops": 0, "gt_layout_tokens": 0}
    pairs_pp = []
    ap_repo, ap_dan = [], []          # per-document AP-per-class dicts, for the global mAP_CER
    n_map_docs = 0
    for s in samples:
        gt, pred = s["ground_truth"], s["prediction"]
        conf = s.get("confidence")
        pp = PostProcessingModuleREAD()
        if conf is not None and len(conf) == len(pred):
            pred_pp, conf_pp = pp.post_process(pred, conf)
        else:
            pred_pp, conf_pp = pp.post_process(pred), None
        tot["pp_ops"] += pp.num_op
        tot["gt_layout_tokens"] += len(v2.keep_only_tokens(gt, layout_tokens))
        pairs_pp.append((gt, pred_pp))
        # (a) repository implementation, exactly as MetricManager calls it
        page_token = "ⓟ"
        e_repo, n_repo = repo_mm.edit_and_num_items_for_ged_from_str(gt, pred_pp, layout_tokens, page_token)
        # (b) port and (c) official DAN. Both assume DAN's five-token scheme (ⓟ ⓝ ⓢ ⓐ ⓑ); the V1
        # datasets carry only ⓟ ⓐ ⓑ, on which DAN's graph builder raises (a body without a
        # section) -- then DAN's LOER is simply undefined for that document and is recorded as None.
        try:
            e_v2, n_v2 = v2.loer_items_read(gt, pred_pp)
            e_dan, n_dan = dan_mm.edit_and_num_items_for_ged_from_str_read(gt, pred_pp)
        except ValueError:
            e_v2 = n_v2 = e_dan = n_dan = None
        # mAP_CER: the repository's function and DAN's own copy, on the same post-processed
        # prediction and confidences the MetricManager would have used.
        map_doc = None
        if conf_pp is not None:
            a_repo = repo_mm.compute_layout_mAP_per_class(gt, pred_pp, conf_pp, v2.READ_MATCHING_TOKENS)
            a_dan = dan_mm.compute_layout_mAP_per_class(gt, pred_pp, conf_pp, v2.READ_MATCHING_TOKENS)
            ap_repo.append(a_repo)
            ap_dan.append(a_dan)
            n_map_docs += 1
            map_doc = repo_mm.compute_global_mAP([a_repo]) if a_repo else None
        e_cer = repo_mm.edit_cer_from_string(gt, pred, layout_tokens)
        n_cer = repo_mm.nb_chars_cer_from_string(gt, layout_tokens)
        rows.append({"name": s.get("name"), "repo": [e_repo, n_repo], "v2": [e_v2, n_v2], "dan": [e_dan, n_dan],
                     "cer": e_cer / max(n_cer, 1), "pp_ops": pp.num_op, "map_cer": map_doc})
        for k, (e, n) in (("repo", (e_repo, n_repo)), ("v2", (e_v2, n_v2)), ("dan", (e_dan, n_dan))):
            if e is not None:
                tot[k + "_edit"] += e
                tot[k + "_norm"] += n
        tot["cer_edit"] += e_cer
        tot["cer_n"] += n_cer

    loer_repo = tot["repo_edit"] / tot["repo_norm"]
    n_defined = sum(1 for r in rows if r["dan"][0] is not None)
    loer_v2 = tot["v2_edit"] / tot["v2_norm"] if tot["v2_norm"] else None
    loer_dan = tot["dan_edit"] / tot["dan_norm"] if tot["dan_norm"] else None
    agree_v2_dan = all(r["v2"] == r["dan"] for r in rows)
    scheme = sorted(set(c for s in samples for c in s["ground_truth"] if c in "ⓟⓝⓢⓐⓑ"))
    map_repo = repo_mm.compute_global_mAP(ap_repo) if ap_repo else None
    map_dan = dan_mm.compute_global_mAP(ap_dan) if ap_dan else None
    map_repo_cls = {k: float(v) for k, v in repo_mm.compute_global_mAP_per_class(ap_repo).items()} if ap_repo else {}
    map_dan_cls = {k: float(v) for k, v in dan_mm.compute_global_mAP_per_class(ap_dan).items()} if ap_dan else {}
    map_agree = (map_repo is not None and map_dan is not None
                 and abs(map_repo - map_dan) < 1e-12 and map_repo_cls.keys() == map_dan_cls.keys()
                 and all(abs(map_repo_cls[k] - map_dan_cls[k]) < 1e-12 for k in map_repo_cls))
    order = v2.order_invariant_metrics([(s["ground_truth"], s["prediction"]) for s in samples], layout_tokens)
    order_ws = v2.order_invariant_metrics([(s["ground_truth"], s["prediction"]) for s in samples], layout_tokens, tokenizer="whitespace")

    summary = {"dump": os.path.relpath(a.dump, ROOT), "model": d["model"], "dataset": d["dataset"], "split": d["split"],
               "n_documents": len(samples), "manager_metrics": d.get("metrics_from_manager"),
               "cer_recomputed": tot["cer_edit"] / tot["cer_n"],
               "gt_layout_token_scheme": scheme, "documents_where_dan_loer_is_defined": n_defined,
               "loer_repository_impl": loer_repo, "loer_dan_port": loer_v2, "loer_dan_official": loer_dan,
               "port_equals_official_on_every_document": agree_v2_dan,
               "pper": tot["pp_ops"] / max(tot["gt_layout_tokens"], 1),
               "map_cer_documents": n_map_docs,
               "map_cer_repository_impl": map_repo, "map_cer_dan_official": map_dan,
               "map_cer_repo_equals_official": map_agree,
               "map_cer_per_class_repository_impl": map_repo_cls,
               "map_cer_per_class_dan_official": map_dan_cls,
               "order_invariant_dan_tokenisation": order, "order_invariant_whitespace_tokenisation": order_ws,
               "per_document": rows}
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "layout_metrics_{}_{}_{}.json".format(d["model"], d["dataset"], d["split"]))
    json.dump(summary, open(out, "w"), indent=1)
    print("documents", len(samples), "| CER recomputed {:.4f} (manager {})".format(summary["cer_recomputed"], (d.get("metrics_from_manager") or {}).get("cer")))
    print("GT layout-token scheme:", "".join(scheme), "| documents where DAN's LOER is defined: {}/{}".format(n_defined, len(rows)))
    print("LOER  repository impl {:.4f} | DAN port {} | DAN official {} | port==official on every doc: {}".format(
        loer_repo, "{:.4f}".format(loer_v2) if loer_v2 is not None else "undefined", "{:.4f}".format(loer_dan) if loer_dan is not None else "undefined", agree_v2_dan))
    print("PPER {:.4f}".format(summary["pper"]))
    if map_repo is not None:
        print("mAP_CER  repository impl {:.4f} | DAN official {:.4f} | identical: {} | documents {}/{} | manager {}".format(
            map_repo, map_dan, map_agree, n_map_docs, len(rows), (d.get("metrics_from_manager") or {}).get("map_cer")))
        print("mAP_CER per class (repo): " + ", ".join("{} {:.4f}".format(k, v) for k, v in sorted(map_repo_cls.items())))
    else:
        print("mAP_CER  not computed: the dump carries no per-token confidences")
    print("order-invariant (DAN tokens): " + ", ".join("{} {:.4f}".format(k, v) for k, v in order.items()))
    print("order-invariant (whitespace): " + ", ".join("{} {:.4f}".format(k, v) for k, v in order_ws.items()))
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
