"""
Equivalence test: the DAN-faithful LOER port (hand/basic/layout_metrics.py) against the
official DAN implementation (third_party/DAN/basic/metric_manager.py) on real READ 2016
ground-truth strings in DAN's five-token scheme, each perturbed at random (dropped, inserted,
swapped and duplicated layout tokens, then repaired by the READ post-processing module as
the evaluation protocol does).

Run: python tests/test_layout_metrics_vs_dan.py [--n 200] [--dataset READ_2016_double_page_sem_dan]
Requires third_party/DAN (see third_party/README.md). Ground-truth text never leaves the
process.
"""
import argparse
import os
import pickle
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hand.basic import layout_metrics as v2  # noqa: E402
from hand.basic.post_pocessing_layout import PostProcessingModuleREAD  # noqa: E402


def load_dan():
    sys.modules["tensorflow"] = None
    sys.path.insert(0, os.path.join(ROOT, "third_party", "DAN"))
    import importlib
    return importlib.import_module("basic.metric_manager")


def perturb(s, rng):
    begin = "ⓟⓝⓢⓐⓑ"
    end = "ⓅⓃⓈⒶⒷ"
    idx = [i for i, c in enumerate(s) if c in begin + end]
    ops = rng.randint(0, 4)
    s = list(s)
    for _ in range(ops):
        if not idx:
            break
        kind = rng.choice(["drop", "dup", "swap", "insert"])
        i = rng.choice(idx)
        if kind == "drop":
            s[i] = ""
        elif kind == "dup":
            s[i] = s[i] + s[i]
        elif kind == "swap":
            j = rng.choice(idx)
            s[i], s[j] = s[j], s[i]
        else:
            s.insert(i, rng.choice(begin + end))
        idx = [k for k, c in enumerate(s) if c and c[0] in begin + end]
    return "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dataset", default="READ_2016_double_page_sem_dan")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dan = load_dan()
    rng = random.Random(a.seed)
    labels = pickle.load(open(os.path.join(ROOT, "formatted", a.dataset, "labels.pkl"), "rb"))
    gts = [v["text"] for split in labels["ground_truth"].values() for v in split.values()]
    n_ok = n_pp_fail = 0
    for k in range(a.n):
        gt = rng.choice(gts)
        pred = perturb(gt, rng)
        try:
            pred_pp = PostProcessingModuleREAD().post_process(pred)
        except Exception:
            n_pp_fail += 1
            continue
        try:
            e_dan, n_dan = dan.edit_and_num_items_for_ged_from_str_read(gt, pred_pp)
        except ValueError as exc:
            # DAN's graph builder assumes a body/annotation is preceded by a section; the
            # post-processing normally guarantees it. Both implementations must agree on the error.
            try:
                v2.loer_items_read(gt, pred_pp)
                raise AssertionError("official DAN raised {} but the port did not".format(exc))
            except ValueError:
                n_ok += 1
                continue
        e_v2, n_v2 = v2.loer_items_read(gt, pred_pp)
        assert (e_v2, n_v2) == (e_dan, n_dan), (k, e_v2, n_v2, e_dan, n_dan)
        n_ok += 1
    print("port == official DAN on {} / {} perturbed documents ({} post-processing failures skipped)".format(n_ok, a.n, n_pp_fail))
    # identity and a fixed sanity case
    gt = gts[0]
    assert v2.loer_items_read(gt, gt)[0] == 0
    print("identity LOER 0 on a real document; |V|+|E| =", v2.loer_items_read(gt, gt)[1])


if __name__ == "__main__":
    main()
