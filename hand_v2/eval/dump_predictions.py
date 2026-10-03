"""
Dump per-sample predictions (with layout tokens and per-token confidences) for a released
checkpoint, so that layout metrics can be recomputed offline by independent implementations.

`tools/reproduce_baseline.py` intended to store per-sample predictions but
`MetricManager.compute_metrics` never keeps `str_x`; the dump there is empty. This script
reuses the exact same model construction and evaluation path and only intercepts the
values handed to the metric manager.

The output embeds ground-truth transcriptions of a research-licensed dataset: write it
under `private/` (git-ignored) only.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tools.evaluate_hand import MODELS, build_params, stage_checkpoint, detect_additional_tokens, set_seed  # noqa: E402
from tools.reproduce_baseline import MAX_CHARS  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402
import hand.basic.metric_manager as mmod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="read_page", choices=sorted(MODELS))
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--dataset-override", default=None,
                    help="formatted dataset directory name to evaluate instead of the model's default "
                         "(e.g. READ_2016_double_page_sem_dan)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "private", "research_audit", "predictions"))
    a = ap.parse_args()

    spec = dict(MODELS[a.model])
    if a.dataset_override:
        # dataset name is "<dataset>_<level><variant>"; keep dataset/level, swap the variant
        assert a.dataset_override.startswith("{}_{}".format(spec["dataset"], spec["level"])), a.dataset_override
        spec["variant"] = a.dataset_override[len("{}_{}".format(spec["dataset"], spec["level"])):]

    run_dir = "dump_{}".format(a.model)
    stage_checkpoint(spec["ckpt"], run_dir)
    set_seed(0)
    additional_tokens = detect_additional_tokens(os.path.join(ROOT, spec["ckpt"]))
    params = build_params(spec, run_dir, a.batch_size, MAX_CHARS[spec["level"]], GlobalHTADecoder, additional_tokens)
    params["training_params"]["eval_metrics"] = ["cer", "wer", "time"]
    if not torch.cuda.is_available():
        # CPU fallback (used when both GPUs are held by other projects): identical numerics
        # apart from AMP, which is disabled on CPU by the manager.
        params["training_params"]["force_cpu"] = True
        params["training_params"]["use_amp"] = False
        params["training_params"]["nb_gpu"] = 0

    captured = []
    original = mmod.MetricManager.compute_metrics

    def capturing(self, values, metric_names):
        captured.append({
            "names": list(values.get("names", [])),
            "str_x": list(values["str_x"]),
            "str_y": list(values["str_y"]),
            "confidence": [[float(c) for c in cs] for cs in values.get("confidence_score", [])],
        })
        return original(self, values, metric_names)

    mmod.MetricManager.compute_metrics = capturing

    manager = Manager(params)
    manager.load_model()
    os.makedirs(a.out, exist_ok=True)
    for split in a.split:
        captured.clear()
        custom_name = "{}-{}".format(spec["dataset"], split)
        t0 = time.time()
        manager.predict(custom_name, [(spec["dataset"], split)], ["cer", "wer", "time"], output=True)
        elapsed = time.time() - t0
        samples = []
        for batch in captured:
            for i in range(len(batch["str_x"])):
                samples.append({"name": batch["names"][i] if batch["names"] else None,
                                "prediction": batch["str_x"][i], "ground_truth": batch["str_y"][i],
                                "confidence": batch["confidence"][i] if batch["confidence"] else None})
        mm = manager.metric_manager[custom_name]
        values = {k: (v.item() if hasattr(v, "item") else v) for k, v in mm.get_display_values(output=True).items()}
        dataset_dir = "{}_{}{}".format(spec["dataset"], spec["level"], spec["variant"])
        out = {"model": a.model, "checkpoint": spec["ckpt"], "dataset": dataset_dir, "split": split,
               "batch_size": a.batch_size, "n_samples": len(samples), "wall_clock_s": round(elapsed, 2),
               "metrics_from_manager": values, "samples": samples}
        path = os.path.join(a.out, "{}__{}__{}.json".format(a.model, dataset_dir, split))
        with open(path, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("wrote", path, "samples", len(samples), "manager metrics", {k: values[k] for k in ("cer", "wer") if k in values})


if __name__ == "__main__":
    main()
