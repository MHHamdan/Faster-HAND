#!/usr/bin/env python3
"""One CPU-only command that tells a newcomer whether their install is correct.

    python tools/validate_install_cpu.py

No GPU, no CUDA context, no network. The script runs four stages and stops at the first
one whose *inputs* are missing, reporting SKIP with the exact reason and where to get the
input. It never writes into an existing artefact and never writes into `release/`.

    S1  environment      interpreter, torch, Pillow, numpy; the two load-bearing install
                         constraints (docs/reproducibility.md section 1)
    S2  architecture     builds the page model from config.json with NO weights and NO
                         dataset and checks the parameter counts and the tensor shapes.
                         This stage runs on a clean checkout with nothing downloaded.
    S3  weights          loads an exported checkpoint strictly and re-checks the counts
    S4  parity           transcribes the first N pages of the READ 2016 test split on CPU
                         and compares the per-page edit distances against a measurement

EXPECTED VALUES AND TOLERANCE, both from artefacts, never from taste:

  S2  parameter counts come from the `expected_parameters` / `parameters_encoder` /
      `parameters_decoder` fields of the config.json being validated. Those fields were
      written by release/tools/export_release_checkpoint.py from the checkpoint itself and
      agree with the training record
      experiments/benchmark_suite/registry/20260917T160949Z_e14_budget_1p26M_s0_bd9339.json
      ("n_params": {"encoder": 1706240, "decoder": 5327460, "total": 7033700}).

  S4  per-page edit distances come from release/PARITY_CPU.json `.per_page`, which is a
      CPU fp32 run of release/tools/evaluate_release.py over all 50 test pages: 827 edits
      over 23,262 characters, CER 0.035552, ~12.55 s/page.

      TOLERANCE --tol-edits 2 (default). Derivation, from the only divergence that has
      actually been measured on this model: that CPU fp32 run differs from the published
      AMP fp16 GPU artefact (seed_variance_test.json .per_page.e14_seed0_1p26M, 826 edits)
      by ONE edit on ONE page out of fifty. Two edits is that measured divergence rounded
      up. It is tight enough that a wrong charset, a wrong `additional_tokens`, a wrong
      positional-encoding mode or a wrong normalisation fail it by orders of magnitude
      (release/tools/evaluate_release.py documents the 0.5x-resize failure as ~an entire
      dropped line), and loose enough to survive fp32 arithmetic reordering.

      There is NO measurement of CPU-to-CPU divergence across different machines. A
      failure at tolerance 2 therefore means "investigate", not automatically "broken";
      the per-page table printed below says which page moved.

WHAT THIS COMMAND DOES NOT PROVE. It validates the inference path of the page model only.
It does not train anything, it does not touch the GPU, and passing it says nothing about
whether a training run will reproduce -- for that see docs/reproducibility.md section 5.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG_PATH = os.path.join("release", "hand-read2016-page", "config.json")
DEFAULT_MODEL_DIR = os.path.join("release", "hand-read2016-page")
DEFAULT_DATA = os.path.join("formatted", "READ_2016_page_sem_dan")
PARITY_ARTEFACT = os.path.join("release", "PARITY_CPU.json")

# Where each missing input comes from. Printed verbatim on SKIP so the message is
# actionable rather than a bare "not found".
ACQUIRE = {
    "config": "release/tools/export_release_checkpoint.py writes config.json; the tracked "
              "copy is release/hand-read2016-page/config.json. See docs/reproducibility.md section 7.",
    "weights": "export one:  python release/tools/export_release_checkpoint.py --ckpt "
               "outputs/e14_budget_1p26M_s0/checkpoints/best_3580.pt --out <dir>\n"
               "                 The checkpoint is produced by docs/reproducibility.md section 5 and "
               "is not redistributed in this repository.",
    "data": "formatted/READ_2016_page_sem_dan is git-ignored and is rebuilt from the raw "
            "corpus:\n                 python hand_v2/data/format_read_dan_splits.py "
            "--levels page\n                 Raw READ 2016 (CC BY 4.0) comes from Zenodo "
            "10.5281/zenodo.1297399. See docs/reproducibility.md section 2.",
    "parity": "release/PARITY_CPU.json carries the reference per-page edit distances.",
}

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


class Report:
    def __init__(self):
        self.stages = []

    def add(self, name, status, detail, **extra):
        rec = {"stage": name, "status": status, "detail": detail}
        rec.update(extra)
        self.stages.append(rec)
        print("[%-4s] %-14s %s" % (status, name, detail), flush=True)
        return status

    @property
    def failed(self):
        return any(s["status"] == FAIL for s in self.stages)

    @property
    def skipped(self):
        return [s["stage"] for s in self.stages if s["status"] == SKIP]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


# ------------------------------------------------------------------ S1 environment
def stage_environment(rep):
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        import numpy
        import PIL
    except Exception as e:                                    # noqa: BLE001
        return rep.add("S1-env", FAIL, "cannot import torch/numpy/PIL: %s" % e), info
    info.update(torch=torch.__version__, numpy=numpy.__version__, pillow=PIL.__version__,
                cuda_available=bool(torch.cuda.is_available()))

    try:
        from hand.models.baseline import FCN_Encoder, GlobalHTADecoder   # noqa: F401
    except Exception as e:                                    # noqa: BLE001
        return rep.add("S1-env", FAIL,
                       "`from hand.models.baseline import ...` failed: %s" % e), info

    # TRAINING-ONLY constraints. Neither blocks inference, so neither is a FAIL here.
    # docs/reproducibility.md section 1: a TensorFlow built against NumPy 1.x aborts
    # `import torch.utils.tensorboard` under NumPy 2.x. Detect TF WITHOUT importing it --
    # importing it is the very thing that can abort -- then test the import that actually
    # has to work for a training run to write logs. TensorBoard's lazy loader prints its
    # own traceback on failure, so the probe runs with stdout/stderr captured.
    import contextlib
    import importlib.util
    import io
    warnings = []
    info["tensorflow_present"] = importlib.util.find_spec("tensorflow") is not None
    if info["tensorflow_present"]:
        warnings.append("TensorFlow is importable here; docs/reproducibility.md section 1 says not to "
                        "install it alongside this project")
    buf = io.StringIO()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # TF logs from C++, not Python
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            import torch.utils.tensorboard  # noqa: F401
        info["tensorboard_import"] = "ok"
    except Exception as e:                                    # noqa: BLE001
        info["tensorboard_import"] = "failed: %s" % type(e).__name__
        warnings.append("`import torch.utils.tensorboard` fails (%s), so tools/train_hand.py "
                        "and hand_v2/train.py cannot write TensorBoard logs. Inference, "
                        "including every stage below, is unaffected" % type(e).__name__)
    # Pillow >= 10 removed FreeTypeFont.getsize(); the synthetic-page generator needs the
    # shim in hand/OCR/ocr_dataset_manager.py. Training only.
    if int(info["pillow"].split(".")[0]) < 10:
        warnings.append("Pillow %s < 10; the pinned requirement is >= 10" % info["pillow"])

    detail = "python %s, torch %s, numpy %s, Pillow %s, cuda_available=%s" % (
        info["python"], info["torch"], info["numpy"], info["pillow"], info["cuda_available"])
    if warnings:
        info["warnings"] = warnings
        pad = "\n                 training-only: "
        return rep.add("S1-env", WARN, detail + pad + pad.join(warnings)), info
    return rep.add("S1-env", PASS, detail), info


# ------------------------------------------------------------------ S2 architecture
DEC_KEYS = ("enc_dim", "l_max", "dec_pred_dropout", "attention_win", "vocab_size",
            "additional_tokens", "use_line_indices", "two_step_pos_enc_mode", "pe_h_max",
            "pe_w_max", "dec_num_layers", "dec_num_heads", "dec_res_dropout",
            "dec_att_dropout", "dec_dim_feedforward", "use_tokens_from_all_lines",
            "use_first_pass_tokens", "dropout")


def build_page_model(config, device):
    """Same construction as release/hand_release/inference.py:from_pretrained, with no
    weights loaded. Kept here so this stage runs on a checkout that has no release payload
    beyond config.json."""
    import torch
    from hand.models.baseline.fcn_encoder import FCN_Encoder
    from hand.models.baseline.dan_decoder import GlobalHTADecoder
    enc = FCN_Encoder({"input_channels": config["input_channels"],
                       "dropout": config["dropout"], "enc_dim": config["enc_dim"],
                       "nb_layers": config["nb_layers"], "device": device})
    dec_params = {k: config[k] for k in DEC_KEYS}
    dec_params["device"] = device
    dec = GlobalHTADecoder(dec_params)
    return enc.to(device).eval(), dec.to(device).eval(), torch


def stage_architecture(rep, config_path, height, width):
    import torch
    cfg_p = resolve(config_path)
    if not os.path.exists(cfg_p):
        return rep.add("S2-arch", SKIP, "no %s\n                 %s"
                       % (config_path, ACQUIRE["config"]))
    config = json.load(open(cfg_p, encoding="utf-8"))
    missing = [k for k in ("expected_parameters", "parameters_encoder", "parameters_decoder")
               if k not in config]
    if missing:
        return rep.add("S2-arch", FAIL,
                       "%s carries no %s, so there is nothing to check the built model "
                       "against" % (config_path, "/".join(missing)))

    enc, dec, _ = build_page_model(config, torch.device("cpu"))
    n_enc = sum(p.numel() for p in enc.parameters())
    n_dec = sum(p.numel() for p in dec.parameters())
    bad = []
    for got, want, what in ((n_enc, config["parameters_encoder"], "encoder"),
                            (n_dec, config["parameters_decoder"], "decoder"),
                            (n_enc + n_dec, config["expected_parameters"], "total")):
        if got != want:
            bad.append("%s %d != %d declared in %s" % (what, got, want, config_path))

    # A deterministic forward. The values are irrelevant (the weights are random); the
    # SHAPES are the contract: the encoder reduces H by 32 and W by 8, and the decoder's
    # output layer is exactly vocab_size + additional_tokens wide.
    torch.manual_seed(0)
    x = torch.rand(1, config["input_channels"], height, width)
    with torch.no_grad():
        feats = enc(x)
        want_shape = (1, config["enc_dim"], -(-height // 32), -(-width // 8))
        if tuple(feats.shape) != want_shape:
            bad.append("encoder output %s != expected %s for a %dx%d page"
                       % (tuple(feats.shape), want_shape, height, width))
        pf = dec.features_updater.get_pos_features(feats)
        pf = torch.flatten(pf, start_dim=2, end_dim=3).permute(2, 0, 1)
        vocab = config["vocab_size"]
        toks = torch.tensor([[vocab + 1]], dtype=torch.long)       # start token
        plen = torch.tensor([1], dtype=torch.int)
        reduced = [[feats.shape[2], feats.shape[3]]]
        _, pred, _, _ = dec(pf, pf, toks, reduced, plen, feats.size(), start=0, cache=None,
                            num_pred=1, padding_value=vocab + 2, use_mem_cache=False)
        want_width = vocab + config["additional_tokens"]
        if pred.shape[1] != want_width:
            bad.append("decoder output layer is %d wide, expected vocab_size + "
                       "additional_tokens = %d" % (pred.shape[1], want_width))

    detail = ("encoder %d + decoder %d = %d parameters; %dx%d page -> %dx%d visual "
              "positions; output layer %d wide"
              % (n_enc, n_dec, n_enc + n_dec, height, width, feats.shape[2], feats.shape[3],
                 pred.shape[1]))
    if bad:
        return rep.add("S2-arch", FAIL, detail + " | " + "; ".join(bad))
    return rep.add("S2-arch", PASS, detail, n_encoder=n_enc, n_decoder=n_dec,
                   n_total=n_enc + n_dec)


# ------------------------------------------------------------------ S3 weights
def stage_weights(rep, model_dir):
    md = resolve(model_dir)
    weight_file = None
    for cand in ("model.safetensors", "model.pt"):
        if os.path.exists(os.path.join(md, cand)):
            weight_file = os.path.join(md, cand)
            break
    if weight_file is None:
        return rep.add("S3-weights", SKIP, "no model weights in %s\n                 %s"
                       % (model_dir, ACQUIRE["weights"])), None

    sys.path.insert(0, os.path.join(ROOT, "release"))
    try:
        from hand_release.inference import HANDRecognizer
    except Exception as e:                                    # noqa: BLE001
        return rep.add("S3-weights", FAIL,
                       "cannot import release/hand_release/inference.py: %s" % e), None
    try:
        r = HANDRecognizer.from_pretrained(md, device="cpu")
    except Exception as e:                                    # noqa: BLE001
        return rep.add("S3-weights", FAIL, "strict weight load failed: %s" % e), None

    n = (sum(p.numel() for p in r.encoder.parameters())
         + sum(p.numel() for p in r.decoder.parameters()))
    digest = sha256(weight_file)
    detail = ("loaded %s strictly on CPU: %d parameters, charset %d, sha256 %s"
              % (os.path.basename(weight_file), n, len(r.charset), digest[:16]))
    rep.add("S3-weights", PASS, detail, parameters=n, charset=len(r.charset),
            weights_sha256=digest)
    return PASS, r


# ------------------------------------------------------------------ S4 parity
def stage_parity(rep, recog, data, split, pages, tol_edits, parity_path):
    if recog is None:
        return rep.add("S4-parity", SKIP, "no model loaded (S3 did not run)")
    dp = resolve(data)
    labels_p = os.path.join(dp, "labels.pkl")
    if not os.path.exists(labels_p):
        return rep.add("S4-parity", SKIP, "no %s\n                 %s"
                       % (os.path.join(data, "labels.pkl"), ACQUIRE["data"]))
    pp = resolve(parity_path)
    if not os.path.exists(pp):
        return rep.add("S4-parity", SKIP, "no %s\n                 %s"
                       % (parity_path, ACQUIRE["parity"]))

    import editdistance
    from hand_release.inference import strip_layout

    ref = json.load(open(pp, encoding="utf-8"))
    if ref.get("split") != split or ref.get("device") != "cpu":
        return rep.add("S4-parity", FAIL,
                       "%s is a %s/%s measurement; this stage compares a cpu/%s run"
                       % (parity_path, ref.get("device"), ref.get("split"), split))
    by_name = {p["name"]: p for p in ref["per_page"]}

    gts = pickle.load(open(labels_p, "rb"))["ground_truth"][split]
    names = sorted(gts, key=lambda n: int(n.split("_")[1].split(".")[0]))[:pages]
    unknown = [n for n in names if n not in by_name]
    if unknown:
        return rep.add("S4-parity", FAIL,
                       "%s has no reference entry for %s" % (parity_path, unknown[:3]))

    exp_edits = sum(by_name[n]["edit"] for n in names)
    exp_chars = sum(by_name[n]["nb"] for n in names)
    print("         comparing %d of %d pages against %s (CPU fp32, reference decode path)"
          % (len(names), ref["n_pages"], parity_path))
    print("         %-16s %6s %6s %8s %8s" % ("page", "edits", "ref", "chars", "s"))

    edits = chars = 0
    per_page, moved = [], []
    t0 = time.time()
    for name in names:
        gt = strip_layout(gts[name]["text"])
        out = recog.read(os.path.join(dp, split, name))
        e = editdistance.eval(out.text, gt)
        r_e, r_n = by_name[name]["edit"], by_name[name]["nb"]
        edits += e
        chars += len(gt)
        per_page.append({"name": name, "edit": e, "ref_edit": r_e, "nb": len(gt),
                         "ref_nb": r_n, "wall_s": out.latency_s})
        if e != r_e:
            moved.append("%s %d vs %d" % (name, e, r_e))
        if len(gt) != r_n:
            moved.append("%s character count %d vs %d" % (name, len(gt), r_n))
        print("         %-16s %6d %6d %8d %8.1f"
              % (name, e, r_e, len(gt), out.latency_s), flush=True)

    delta = abs(edits - exp_edits)
    ok = (delta <= tol_edits) and chars == exp_chars
    if len(names) < ref["n_pages"]:
        print("         NOTE: this is the first %d of %d pages. Its CER is a property of "
              "those pages, not\n               of the model -- the corpus figure is %d "
              "edits / %d chars, CER %.6f\n               (%s)."
              % (len(names), ref["n_pages"], ref["edits"], ref["chars"], ref["cer"],
                 parity_path))
    detail = ("%d edits / %d chars, CER %.6f; reference %d / %d, CER %.6f; |delta| = %d, "
              "tolerance %d; %.1f s total (%.1f s/page)"
              % (edits, chars, edits / chars if chars else float("nan"),
                 exp_edits, exp_chars, exp_edits / exp_chars if exp_chars else float("nan"),
                 delta, tol_edits, time.time() - t0,
                 (time.time() - t0) / max(len(names), 1)))
    if chars != exp_chars:
        detail += (" | CHARACTER COUNT MISMATCH: the ground truth differs from the one the "
                   "reference was measured on, so the CER is not comparable")
    if moved:
        detail += " | pages that moved: " + ", ".join(moved)
    return rep.add("S4-parity", PASS if ok else FAIL, detail,
                   edits=edits, chars=chars, ref_edits=exp_edits, ref_chars=exp_chars,
                   delta=delta, tol_edits=tol_edits, per_page=per_page)


# ------------------------------------------------------------------ main
PROTECTED = ("release", "results_real", "experiments", "outputs")


def out_path_ok(out_path):
    """True if --out may be written. Checked BEFORE any work is done, so a bad path costs
    no inference, and again before the write."""
    rel = os.path.relpath(resolve(out_path), ROOT)
    top = rel.split(os.sep)[0]
    if top in PROTECTED:
        print("refusing to write a validation report into %s/ -- that directory holds "
              "measured artefacts, and this command must never overwrite one. Choose "
              "another --out." % top, file=sys.stderr)
        return False
    return True


def write_report(out_path, payload):
    """Atomic, and never on top of a tracked artefact directory."""
    op = resolve(out_path)
    rel = os.path.relpath(op, ROOT)
    if not out_path_ok(out_path):
        return False
    os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
    tmp = op + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, op)
    print("wrote %s" % rel)
    return True


def main():
    ap = argparse.ArgumentParser(
        description="CPU-only install validation for the HAND page model.")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help="config.json to build and check the architecture against "
                         "(default: %s)" % DEFAULT_CONFIG_PATH)
    ap.add_argument("--model", default=DEFAULT_MODEL_DIR,
                    help="directory holding config.json, charset.json and the weights "
                         "(default: %s)" % DEFAULT_MODEL_DIR)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="formatted dataset root; its pages are already at 150 dpi")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--pages", type=int, default=3,
                    help="pages for the parity stage (default 3; pass 50 for the full "
                         "split). release/PARITY_CPU.json measured 12.55 s/page on an "
                         "unloaded host. The FIRST page costs much more than the rest "
                         "(202 s against 13-17 s in one measurement here) because of "
                         "thread and allocator warm-up, and CPU contention adds more, so "
                         "budget minutes for the default and tens of minutes for 50")
    ap.add_argument("--tol-edits", type=int, default=2,
                    help="parity tolerance in edit operations; see the module docstring "
                         "for where 2 comes from")
    ap.add_argument("--page-size", type=int, nargs=2, default=[1755, 1161],
                    metavar=("H", "W"),
                    help="synthetic page size for the architecture stage (default is the "
                         "size of test_0 in formatted/READ_2016_page_sem_dan)")
    ap.add_argument("--parity-artefact", default=PARITY_ARTEFACT)
    ap.add_argument("--require-parity", action="store_true",
                    help="exit non-zero if the parity stage cannot run")
    ap.add_argument("--out", default=None, help="write the report as JSON (atomically)")
    a = ap.parse_args()

    # No CUDA context is created by this script, and none is wanted: both GPUs may be in
    # use by someone else. Make that structural rather than a promise in prose.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

    # Fail before doing any work if --out is unusable: a run that cannot record its result
    # should not first spend ten minutes decoding pages.
    if a.out and not out_path_ok(a.out):
        return 2

    print("HAND CPU install validation -- repository %s" % ROOT)
    print("=" * 78)
    rep = Report()
    _, env = stage_environment(rep)
    if not rep.failed:
        stage_architecture(rep, a.config, a.page_size[0], a.page_size[1])
    recog = None
    if not rep.failed:
        _, recog = stage_weights(rep, a.model)
    if not rep.failed:
        stage_parity(rep, recog, a.data, a.split, a.pages, a.tol_edits, a.parity_artefact)

    print("=" * 78)
    verdict = "FAIL" if rep.failed else ("PASS" if not rep.skipped else "PASS (partial)")
    print("VERDICT  %s" % verdict)
    if rep.skipped:
        print("         stages skipped for missing inputs: %s" % ", ".join(rep.skipped))
        print("         a skipped stage is not a pass. Re-run it once the input is in "
              "place.")

    payload = {"verdict": verdict, "environment": env, "stages": rep.stages,
               "argv": sys.argv,
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.out and not write_report(a.out, payload):
        return 2
    if rep.failed:
        return 1
    if a.require_parity and "S4-parity" in rep.skipped:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
