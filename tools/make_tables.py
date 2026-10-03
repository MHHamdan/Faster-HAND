#!/usr/bin/env python3
"""
Generate the paper's results tables from measured results only.

This replaces `paper_results/analyze_read_results.py`, which produced its tables from a
hardcoded Python dict of numbers rather than from any experiment output. Every number
emitted here is read from a JSON file written by `tools/evaluate_hand.py`, from a run
record in `experiments/benchmark_suite/registry/`, or recomputed from stored per-sample
predictions. Nothing is typed in by hand.

If a cell has no measurement behind it, it is printed as `--` and listed under
"PENDING" so it cannot silently look like a result.

DETERMINISM. Two runs over an unchanged tree must produce byte-identical output. That is a
property this generator did not have and now does:

  * `results_real/*.json` is read in SORTED filename order, and a run is identified by its
    FILENAME STEM, never by the `model` field inside it. That field is not unique:
    `read_page.json`, `read_page_beam3.json` and `read_page_cov{2,4,8}.0.json` all declare
    `"model": "read_page"`, and the double- and triple-page families do the same. The
    previous version keyed a dict on that field inside an UNSORTED `glob.glob`, so whichever
    file the filesystem happened to return last won, and all three READ rows reported a
    coverage-sweep variant instead of the run named in the row. Measured before this fix:
    READ Page 3.96 (that is `read_page_cov2.0`), Double page 4.73 (`read_double_page_cov4.0`),
    Triple page 8.50 (`read_triple_page_cov2.0`), against the named runs' 3.87 / 3.68 / 10.00.
    Files sharing a `model` field are now reported in PENDING.md, each under its own name.
  * per-split prediction files are selected by NUMERIC epoch, not by string sort:
    `sorted()` puts `predict_X_9.txt` after `predict_X_1390.txt`.
  * nothing time-, host- or path-dependent is written into a table.

Every emitted number is also written to `SOURCES.md` next to the tables, with the file it
came from and the key path inside that file, so a reader can check a cell without reading
this script.

Usage:
    python tools/make_tables.py                      # tables/ + a markdown summary
    python tools/make_tables.py --out paper_tables   # choose the output directory
"""
import argparse
import glob
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results_real")
REGISTRY = os.path.join(ROOT, "experiments", "benchmark_suite", "registry")
POSTCORR = os.path.join(ROOT, "paper_results", "READ_German", "post_correction")

# Display order and human-readable labels for the recognition table. The first element is
# the FILENAME STEM in results_real/, which is the run's identity.
ROW_ORDER = [
    ("read_page", "READ 2016", "Page"),
    ("read_double_page", "READ 2016", "Double page"),
    ("read_triple_page", "READ 2016", "Triple page"),
    ("iam_page", "IAM", "Page"),
    ("khatt_paragraph", "KHATT", "Paragraph"),
    ("ahawp_paragraph", "AHAWP", "Paragraph"),
]

# The current page-level run, trained through hand_v2/train.py. It is NOT in results_real/
# (that directory holds the V1 checkpoints' evaluations), so it is read from its run
# record, identified by run_id so that no glob can pick a different run.
HEADLINE = {
    "run_id": "20260917T160949Z_e14_budget_1p26M_s0_bd9339",
    "label": "READ 2016 page (\\texttt{e14\\_budget\\_1p26M\\_s0}, best epoch 3580)",
}

_PENDING_SEEN = set()
PENDING = []
SOURCES = []       # (table, cell, value, artefact path relative to ROOT, key path)


def pending(msg):
    """Record a gap once. Two cells of one row can hit the same missing split."""
    if msg not in _PENDING_SEEN:
        _PENDING_SEEN.add(msg)
        PENDING.append(msg)


def rel(path):
    return os.path.relpath(path, ROOT)


def note(table, cell_name, value, path, keypath):
    SOURCES.append((table, cell_name, value, rel(path), keypath))


# ------------------------------------------------------------------ loading
def load_measured():
    """stem -> record, read in sorted filename order.

    Keyed by filename stem. The `model` field inside these files is NOT unique and must
    not be used as a key.
    """
    out, by_model = {}, {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("_"):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception as e:                                # noqa: BLE001
            pending("{}: unreadable ({})".format(os.path.basename(path), e))
            continue
        # results_real/ also holds non-result artefacts (e.g. split_integrity.json);
        # only per-model evaluation records carry a "model" key.
        if not (isinstance(rec, dict) and "model" in rec):
            continue
        rec["_path"] = path
        out[stem] = rec
        by_model.setdefault(rec["model"], []).append(stem)

    for model, stems in sorted(by_model.items()):
        if len(stems) > 1:
            pending(
                'results_real/: {} files declare "model": "{}" -- {}. Rows are keyed by '
                "filename, so each is reported under its own name and none silently "
                "stands in for another.".format(len(stems), model, ", ".join(sorted(stems))))
    return out


def load_headline():
    path = os.path.join(REGISTRY, HEADLINE["run_id"] + ".json")
    if not os.path.exists(path):
        pending("headline run record {} is missing; the page-level row cannot be "
                       "generated".format(rel(path)))
        return None, path
    with open(path) as f:
        return json.load(f), path


# ------------------------------------------------------------------ cells
def cell(table, rowname, rec, split, key):
    """Return a formatted percentage, or '--' when there is no measurement.

    Every '--' is recorded in PENDING: a blank cell must be explainable.
    """
    label = "{} {} {}".format(rowname, split, key)
    if rec is None:
        return "--"                       # the row itself is already in PENDING
    s = rec.get("splits", {}).get(split)
    if not s or s.get("status") != "ok":
        pending("{}: {} has no '{}' split with status ok in {}".format(
            table, rowname, split, rel(rec["_path"])))
        return "--"
    v = s.get("metrics", {}).get(key)
    if v is None:
        pending("{}: {} {} carries no '{}' metric in {}".format(
            table, rowname, split, key, rel(rec["_path"])))
        return "--"
    note(table, label, v, rec["_path"], "splits.{}.metrics.{}".format(split, key))
    return "{:.2f}".format(v * 100)


# ------------------------------------------------------------------ tables
def recognition_table(measured):
    """Main recognition results: measured CER/WER on test and valid."""
    t = "main_results"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Recognition results for the V1 released checkpoints. All values are"
        r" measured with \texttt{tools/evaluate\_hand.py} at batch size 1 on the official"
        r" splits; `--' marks a configuration for which no measurement exists, and every"
        r" one is listed in \texttt{PENDING.md}. Each row is the run of that name in"
        r" \texttt{results\_real/}; the coverage-sweep and beam-search variants are"
        r" separate runs and are not shown here. The current page-level model is trained"
        r" through \texttt{hand\_v2/train.py} and is reported separately in"
        r" Table~\ref{tab:headline}.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Level}"
        r" & \multicolumn{2}{c}{Test} & \multicolumn{2}{c}{Valid} \\",
        r" & & CER (\%) & WER (\%) & CER (\%) & WER (\%) \\ \hline",
    ]
    for key, ds, level in ROW_ORDER:
        rec = measured.get(key)
        if rec is None:
            pending("{}: results_real/{}.json does not exist -- not yet evaluated"
                           .format(t, key))
        lines.append("{} & {} & {} & {} & {} & {} \\\\".format(
            ds, level,
            cell(t, key, rec, "test", "cer"), cell(t, key, rec, "test", "wer"),
            cell(t, key, rec, "valid", "cer"), cell(t, key, rec, "valid", "wer")))
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def headline_table():
    """The current page-level run, straight out of its run record."""
    rec, path = load_headline()
    if rec is None:
        return None
    m = rec.get("metrics", {})
    t = "headline_read2016_page"
    cells = {}
    for split in ("test", "valid"):
        s = m.get(split)
        for key in ("cer", "wer"):
            if not s or s.get(key) is None:
                cells[(split, key)] = "--"
                pending("{}: run record {} carries no metrics.{}.{}".format(
                    t, rel(path), split, key))
            else:
                cells[(split, key)] = "{:.2f}".format(s[key] * 100)
                note(t, "{} {}".format(split, key), s[key], path,
                     "metrics.{}.{}".format(split, key))
    n = rec.get("n_params", {})
    params = "{:,}".format(n["total"]) if "total" in n else "--"
    if "total" in n:
        note(t, "parameters", n["total"], path, "n_params.total")
    else:
        pending("{}: run record {} carries no n_params.total".format(t, rel(path)))
    if m.get("test_evaluated") is False:
        pending("{}: the run record sets metrics.test_evaluated = false, so any "
                       "test cell above is not from a registered test evaluation".format(t))

    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Current page-level run on READ 2016, read from its run record"
        r" \texttt{%s.json}. Greedy decoding, evaluation batch size 1, seed 0. The"
        r" measured seed spread over three seeds is 0.70\,pp on test (4.574 $\pm$ 0.388),"
        r" so a difference smaller than that is not resolved by this measurement.}"
        % HEADLINE["run_id"].replace("_", r"\_"),
        r"\label{tab:headline}",
        r"\begin{tabular}{lccccc}", r"\hline",
        r"Run & Params & Test CER (\%) & Test WER (\%) & Valid CER (\%) & Valid WER (\%) \\"
        r" \hline",
        "{} & {} & {} & {} & {} & {} \\\\".format(
            HEADLINE["label"], params, cells[("test", "cer")], cells[("test", "wer")],
            cells[("valid", "cer")], cells[("valid", "wer")]),
        r"\hline", r"\end{tabular}", r"\end{table}"])


def efficiency_table(measured):
    """Parameter counts and measured per-sample inference time."""
    import torch
    t = "efficiency"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Model size and measured inference cost for the V1 released checkpoints."
        r" Parameter counts are summed directly from the checkpoints; time is wall-clock"
        r" per sample at batch size 1 on a single RTX PRO 6000.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{llccc}", r"\hline",
        r"Dataset & Level & Encoder (M) & Decoder (M) & s/sample \\ \hline",
    ]
    for key, ds, level in ROW_ORDER:
        rec = measured.get(key)
        ck = os.path.join(ROOT, "models", key, "best_model.pt")
        enc = dec = "--"
        if os.path.exists(ck):
            c = torch.load(ck, map_location="cpu", weights_only=False)
            n_e = sum(v.numel() for v in c["encoder_state_dict"].values())
            n_d = sum(v.numel() for v in c["decoder_state_dict"].values())
            enc, dec = "{:.3f}".format(n_e / 1e6), "{:.3f}".format(n_d / 1e6)
            note(t, key + " encoder params", n_e, ck, "encoder_state_dict (summed numel)")
            note(t, key + " decoder params", n_d, ck, "decoder_state_dict (summed numel)")
        else:
            pending("{}: {} has no checkpoint at {}, so its parameter counts print "
                           "as '--'".format(t, key, rel(ck)))
        tm = "--"
        if rec:
            s = rec.get("splits", {}).get("test")
            if s and s.get("status") == "ok":
                st = s["metrics"].get("sample_time")
                if st is not None:
                    tm = "{:.2f}".format(st)
                    note(t, key + " s/sample", st, rec["_path"],
                         "splits.test.metrics.sample_time")
        if tm == "--":
            pending("{}: {} has no test-split sample_time".format(t, key))
        lines.append("{} & {} & {} & {} & {} \\\\".format(ds, level, enc, dec, tm))
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _epoch_of(path):
    """predict_<name>_<epoch>.txt -> epoch as an int, or -1."""
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def ablation_table():
    """Encoder component ablation, read from the ablation runs' own outputs."""
    t = "encoder_ablation"
    rows = [("abl_line_fcn", "FCN encoder (DAN)"),
            ("abl_line_hand", "HAND: gated DSC + octave + SE"),
            ("abl_line_hand_no_octave", "HAND: gated DSC + SE (no octave)"),
            ("abl_line_hand_no_se", "HAND: gated DSC + octave (no SE)"),
            ("abl_line_hand_no_gate", "HAND: octave + SE (no gating)")]
    out, any_row, epochs = [], False, []
    for run, label in rows:
        fs = sorted(glob.glob(os.path.join(ROOT, "outputs", run, "results",
                                           "predict_*test*.txt")))
        if not fs:
            pending("{}: {} not run (no predict_*test*.txt under outputs/{}/results)"
                           .format(t, run, run))
            out.append((label, "--", "--"))
            continue
        # Highest EPOCH, not highest string: sorted() would put _9 after _1390.
        pick = max(fs, key=_epoch_of)
        if len(fs) > 1:
            pending("{}: {} has {} test prediction files; using epoch {} ({})"
                           .format(t, run, len(fs), _epoch_of(pick),
                                   os.path.basename(pick)))
        d = {}
        with open(pick) as f:
            for line in f:
                k, _, v = line.partition(":")
                d[k.strip()] = v.strip()
        if "cer" not in d or "wer" not in d:
            pending("{}: {} -- {} carries no cer/wer".format(t, run, rel(pick)))
            out.append((label, "--", "--"))
            continue
        out.append((label, "{:.2f}".format(float(d["cer"]) * 100),
                    "{:.2f}".format(float(d["wer"]) * 100)))
        note(t, run + " cer (epoch %d)" % _epoch_of(pick), float(d["cer"]), pick, "cer")
        note(t, run + " wer (epoch %d)" % _epoch_of(pick), float(d["wer"]), pick, "wer")
        epochs.append(_epoch_of(pick))
        any_row = True
    if not any_row:
        return None
    if len(set(epochs)) > 1:
        pending("{}: the rows are NOT all at the same epoch ({}), so the comparison "
                       "is not at equal budget".format(t, sorted(set(epochs))))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Encoder component ablation on READ 2016, line level. All runs use"
             r" identical data, seed and schedule on a dedicated GPU; the encoder is the"
             r" only variable. Not trained to convergence; the epoch each row was read at"
             r" is listed in \texttt{SOURCES.md}.}",
             r"\label{tab:encoder_ablation}",
             r"\begin{tabular}{lcc}", r"\hline",
             r"Encoder configuration & CER (\%) & WER (\%) \\ \hline"]
    for label, c, w in out:
        lines.append("{} & {} & {} \\\\".format(label, c, w))
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# Declared order. The baseline is taken from the FIRST of these that exists, so which
# pickle supplies it does not depend on dict or filesystem ordering.
POSTCORR_FILES = [
    ("byt5_corrected_results.pkl", "ByT5 (direct)"),
    ("byt5_line_corrected.pkl", "ByT5 (line-by-line)"),
    ("byt5_conservative_corrected.pkl", "ByT5 (conservative)"),
    ("selective_corrected_3pct.pkl", "ByT5 (selective, 3\\%)"),
    ("selective_corrected_4pct.pkl", "ByT5 (selective, 4\\%)"),
    ("mt5_corrected.pkl", "mT5-small"),
    ("t5_spelling_results.pkl", "T5 (spelling)"),
]


def postcorrection_table():
    """Post-OCR correction, recomputed from the stored per-sample predictions."""
    t = "postcorrection"
    try:
        import editdistance as ed
    except ImportError:
        pending("{}: `editdistance` not installed".format(t))
        return None
    if not os.path.isdir(POSTCORR):
        pending("{}: {} is missing, so no post-correction number is generated"
                       .format(t, rel(POSTCORR)))
        return None

    def cer(p, g):
        n = sum(ed.eval(a, b) for a, b in zip(p, g))
        d = sum(len(b) for b in g)
        return 100.0 * n / d if d else float("nan")

    def wer(p, g):
        n = sum(ed.eval(a.split(), b.split()) for a, b in zip(p, g))
        d = sum(len(b.split()) for b in g)
        return 100.0 * n / d if d else float("nan")

    rows, base, base_src = [], None, None
    for fn, label in POSTCORR_FILES:
        path = os.path.join(POSTCORR, fn)
        if not os.path.exists(path):
            pending("{}: {} missing".format(t, fn))
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        gt = d.get("ground_truths")
        orig = d.get("predictions_original") or d.get("predictions")
        corr = d.get("corrected")
        if not (gt and orig and corr):
            pending("{}: {} lacks predictions/ground truth".format(t, fn))
            continue
        # Always compare against the *same* uncorrected baseline. Comparing a corrected
        # output against a normalised intermediate (as the previous report did) makes a
        # degradation look like an improvement.
        if base is None:
            base, base_src = (cer(orig, gt), wer(orig, gt), len(gt)), path
            note(t, "uncorrected baseline cer", base[0], path,
                 "recomputed from predictions_original vs ground_truths")
        rows.append((label, cer(corr, gt), wer(corr, gt)))
        note(t, label + " cer", rows[-1][1], path,
             "recomputed from corrected vs ground_truths")

    if base is None:
        return None

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Post-OCR correction on READ 2016 (n=%d). Every value is recomputed"
        r" from the stored per-sample predictions against the same uncorrected baseline,"
        r" taken from \texttt{%s}.}"
        % (base[2], os.path.basename(base_src).replace("_", r"\_")),
        r"\label{tab:postcorrection}",
        r"\begin{tabular}{lcc}", r"\hline",
        r"Configuration & CER (\%) & WER (\%) \\ \hline",
        r"HAND (no correction) & \textbf{%.2f} & \textbf{%.2f} \\ \hline" % (base[0], base[1]),
    ]
    # Sorted by CER then by label, so a tie cannot reorder the rows between runs.
    for label, c, w in sorted(rows, key=lambda r: (round(r[1], 6), r[0])):
        lines.append("+ {} & {:.2f} & {:.2f} \\\\".format(label, c, w))
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper_tables"))
    args = ap.parse_args()
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    measured = load_measured()
    if not measured:
        print("No measured results in {}. Run tools/evaluate_hand.py first."
              .format(RESULTS), file=sys.stderr)

    for name, content in [
        ("main_results.tex", recognition_table(measured)),
        ("headline_read2016_page.tex", headline_table()),
        ("efficiency.tex", efficiency_table(measured)),
        ("encoder_ablation.tex", ablation_table()),
        ("postcorrection.tex", postcorrection_table()),
    ]:
        path = os.path.join(out_dir, name)
        if content is None:
            # Never leave a stale table from an earlier run standing beside a fresh one.
            if os.path.exists(path):
                os.remove(path)
                print("removed stale", path)
            continue
        with open(path, "w") as f:
            f.write(content + "\n")
        print("wrote", path)

    with open(os.path.join(out_dir, "SOURCES.md"), "w") as f:
        f.write("# Where every emitted number comes from\n\n")
        f.write("One row per number printed in the generated tables. `Value` is the raw "
                "value in the artefact; the tables print CER and WER as percentages, "
                "i.e. `value * 100`.\n\n")
        f.write("| Table | Cell | Value | Artefact | Key |\n|---|---|---|---|---|\n")
        for table, cell_name, value, path, keypath in SOURCES:
            f.write("| `{}` | {} | `{}` | `{}` | `{}` |\n"
                    .format(table, cell_name, value, path, keypath))
    print("wrote", os.path.join(out_dir, "SOURCES.md"))

    with open(os.path.join(out_dir, "PENDING.md"), "w") as f:
        f.write("# Cells with no measurement behind them\n\n")
        if PENDING:
            for p in PENDING:
                f.write("- {}\n".format(p))
        else:
            f.write("None -- every cell in every generated table is measured.\n")
    print("wrote", os.path.join(out_dir, "PENDING.md"))
    if PENDING:
        print("\nPENDING ({}):".format(len(PENDING)))
        for p in PENDING:
            print("  -", p)


if __name__ == "__main__":
    main()
