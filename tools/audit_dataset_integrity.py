#!/usr/bin/env python3
"""
Phase 2, Task 5 — dataset integrity audit, with a non-zero exit for CI.

`tools/check_splits.py` compares transcription strings. That is necessary but not
sufficient: two splits can share no *string* and still share the same physical document, or
the same writer, which inflates any reported accuracy just as effectively. This tool adds
those two dimensions and turns the result into a gate.

For every dataset under `formatted/` it reports:

  * sample count and distinct-transcription count per split
  * transcriptions duplicated within a split, and shared across splits — both as DISTINCT
    strings and as held-out SAMPLES, which are different numbers (evaluation scores samples)
  * **identical images across splits**, by SHA-1 over raw file bytes — this is the direct
    test for "test contains identical documents from train", and it works even though the
    formatters renamed everything to `<split>_<n>.png` and discarded provenance
  * **writer overlap**, where writer identity is recoverable (see PROVENANCE below)

Exit status
    0  no blocking violation
    1  a blocking violation: an identical document, or an entire transcription set,
       shared between train and test

CI usage:
    python tools/audit_dataset_integrity.py --json dataset_audit.json --fail-on-violation

PROVENANCE
    The formatters discard writer and document identity when they write
    `formatted/<dataset>/<split>/<split>_<n>.png`. Writer overlap can therefore only be
    checked where a sidecar `provenance.json` exists, mapping formatted filename -> writer
    and source document. `tools/build_provenance.py` regenerates it for AHAWP from the raw
    `paragraphs_per_user/userNNN/` layout. Datasets without a sidecar report
    `writer_overlap: "provenance_unavailable"` — which is a finding, not a pass.
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMATTED = os.path.join(ROOT, "formatted")
SPLITS = ("train", "valid", "test")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def text_of(v):
    if isinstance(v, dict):
        for k in ("text", "label", "gt"):
            if k in v:
                return v[k]
    return str(v)


def load_labels(name):
    lp = os.path.join(FORMATTED, name, "labels.pkl")
    if not os.path.exists(lp):
        return None
    with open(lp, "rb") as f:
        labels = pickle.load(f)
    gt = labels.get("ground_truth", labels)
    return gt if isinstance(gt, dict) else None


def image_hashes(dataset, split, limit=None):
    """SHA-1 over raw file bytes, keyed by filename.

    File bytes rather than decoded pixels: the formatters write every split through the
    same PIL save path, so a document copied into two splits is byte-identical. This costs
    one read per file instead of a decode, which matters at 5 GB.
    """
    d = os.path.join(FORMATTED, dataset, split)
    if not os.path.isdir(d):
        return {}
    out = {}
    names = sorted(n for n in os.listdir(d) if n.lower().endswith(IMAGE_EXT))
    if limit:
        names = names[:limit]
    for n in names:
        h = hashlib.sha1()
        with open(os.path.join(d, n), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[n] = h.hexdigest()
    return out


def load_provenance(dataset):
    p = os.path.join(FORMATTED, dataset, "provenance.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def audit(dataset, hash_images=True, limit=None):
    gt = load_labels(dataset)
    if gt is None:
        return {"dataset": dataset, "error": "no readable labels.pkl"}

    texts, names = {}, {}
    for s in SPLITS:
        d = gt.get(s) if isinstance(gt.get(s), dict) else {}
        names[s] = list(d.keys())
        texts[s] = [text_of(v) for v in d.values()]

    uniq = {s: set(texts[s]) for s in SPLITS}
    rec = {
        "dataset": dataset,
        "n_samples": {s: len(texts[s]) for s in SPLITS},
        "n_unique_transcripts": {s: len(uniq[s]) for s in SPLITS},
        "violations": [],
        "warnings": [],
    }

    # --- duplicated transcriptions within a split -----------------------------------
    rec["duplicated_transcripts_within_split"] = {}
    for s in SPLITS:
        c = Counter(texts[s])
        dups = {t: n for t, n in c.items() if n > 1}
        rec["duplicated_transcripts_within_split"][s] = {
            "n_transcripts_duplicated": len(dups),
            "n_samples_affected": sum(dups.values()),
            "max_repeats_of_one_transcript": max(dups.values()) if dups else 0,
        }

    # --- transcription overlap across splits ----------------------------------------
    rec["transcript_overlap"] = {
        "valid_in_train": len(uniq["valid"] & uniq["train"]),
        "test_in_train": len(uniq["test"] & uniq["train"]),
        "test_in_valid": len(uniq["test"] & uniq["valid"]),
    }
    # The same overlap counted over SAMPLES rather than distinct strings. The two diverge
    # sharply where held-out text repeats: KHATT_line is 230/744 distinct (30.9 %) but
    # 470/999 instances (47.0 %). Evaluation scores samples, so the instance figure is the
    # one that bounds a reported CER -- report both so a document cannot quote the wrong one.
    rec["instance_overlap"] = {
        "valid_in_train": sum(t in uniq["train"] for t in texts["valid"]),
        "test_in_train": sum(t in uniq["train"] for t in texts["test"]),
        "test_in_valid": sum(t in uniq["valid"] for t in texts["test"]),
        "note": "held-out SAMPLES whose transcription occurs in train; transcript_overlap "
                "counts DISTINCT strings. Evaluation scores samples.",
    }
    for s in ("valid", "test"):
        if uniq[s] and uniq[s] <= uniq["train"]:
            msg = "{}: every distinct transcription also occurs in train".format(s)
            (rec["violations"] if s == "test" else rec["warnings"]).append(msg)
    if uniq["train"] and len(uniq["train"]) < 20:
        rec["warnings"].append(
            "only {} distinct transcriptions in train -- this is a memorisation task, "
            "not a recognition benchmark".format(len(uniq["train"])))

    # --- identical documents across splits ------------------------------------------
    if hash_images:
        h = {s: image_hashes(dataset, s, limit) for s in SPLITS}
        rev = {s: {} for s in SPLITS}
        for s in SPLITS:
            for fn, digest in h[s].items():
                rev[s].setdefault(digest, []).append(fn)
        rec["n_images_hashed"] = {s: len(h[s]) for s in SPLITS}
        rec["duplicate_images_within_split"] = {
            s: sum(len(v) - 1 for v in rev[s].values() if len(v) > 1) for s in SPLITS}
        shared = {
            "valid_in_train": sorted(set(rev["valid"]) & set(rev["train"])),
            "test_in_train": sorted(set(rev["test"]) & set(rev["train"])),
            "test_in_valid": sorted(set(rev["test"]) & set(rev["valid"])),
        }
        rec["identical_documents_across_splits"] = {k: len(v) for k, v in shared.items()}
        rec["identical_document_examples"] = {
            k: [{"hash": d[:12],
                 "train": rev["train"].get(d, []),
                 "valid": rev["valid"].get(d, []),
                 "test": rev["test"].get(d, [])}
                for d in v[:3]]
            for k, v in shared.items() if v}
        if shared["test_in_train"]:
            rec["violations"].append(
                "{} identical document(s) appear in both train and test".format(
                    len(shared["test_in_train"])))
        if shared["valid_in_train"]:
            rec["warnings"].append(
                "{} identical document(s) appear in both train and valid".format(
                    len(shared["valid_in_train"])))
    else:
        rec["identical_documents_across_splits"] = "not_checked"

    # --- writer overlap --------------------------------------------------------------
    prov = load_provenance(dataset)
    if prov is None:
        rec["writer_overlap"] = "provenance_unavailable"
        rec["warnings"].append(
            "writer identity was discarded by the formatter; writer-independence "
            "cannot be verified for this dataset")
    else:
        w = {s: {prov[n]["writer"] for n in names[s] if n in prov and "writer" in prov[n]}
             for s in SPLITS}
        docs = {s: {prov[n]["document"] for n in names[s]
                    if n in prov and "document" in prov[n]} for s in SPLITS}
        rec["writer_overlap"] = {
            "n_writers": {s: len(w[s]) for s in SPLITS},
            "valid_in_train": len(w["valid"] & w["train"]),
            "test_in_train": len(w["test"] & w["train"]),
        }
        rec["document_overlap"] = {
            "n_documents": {s: len(docs[s]) for s in SPLITS},
            "test_in_train": len(docs["test"] & docs["train"]),
        }
        if w["test"] & w["train"]:
            rec["violations"].append(
                "{} writer(s) appear in both train and test -- the split is not "
                "writer-independent".format(len(w["test"] & w["train"])))
        if docs["test"] & docs["train"]:
            rec["violations"].append(
                "{} source document(s) appear in both train and test".format(
                    len(docs["test"] & docs["train"])))

    rec["verdict"] = "VIOLATION" if rec["violations"] else (
        "WARN" if rec["warnings"] else "OK")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=None,
                    help="dataset directory name(s) under formatted/; default all")
    ap.add_argument("--json", default=os.path.join(ROOT, "dataset_audit.json"))
    ap.add_argument("--no-hash", action="store_true",
                    help="skip image hashing (much faster, loses the document check)")
    ap.add_argument("--limit", type=int, default=None,
                    help="hash at most N images per split (for a quick pass)")
    ap.add_argument("--fail-on-violation", action="store_true",
                    help="exit 1 if any dataset reports a violation (for CI)")
    a = ap.parse_args()

    if not os.path.isdir(FORMATTED):
        raise SystemExit("no formatted/ directory at {}".format(FORMATTED))
    datasets = a.dataset or sorted(
        d for d in os.listdir(FORMATTED)
        if os.path.exists(os.path.join(FORMATTED, d, "labels.pkl")))

    results = []
    for d in datasets:
        print("auditing {} ...".format(d), flush=True)
        try:
            results.append(audit(d, hash_images=not a.no_hash, limit=a.limit))
        except Exception as exc:
            results.append({"dataset": d, "error": repr(exc), "verdict": "ERROR"})

    # A provenance header, so a stale audit is visible from the file itself rather than
    # only from its mtime. The shipped dataset_audit.json covered 13 of the 26 corpora on
    # disk -- including READ_2016_page_sem_dan, the corpus the headline result is trained
    # on -- and carried nothing to say so.
    payload = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "tool": "tools/audit_dataset_integrity.py",
        "code_version": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip() or None,
        "n_corpora_on_disk": len([d for d in os.listdir(FORMATTED)
                                  if os.path.exists(os.path.join(FORMATTED, d, "labels.pkl"))]),
        "n_corpora_audited": len(results),
        "datasets": results,
    }
    with open(a.json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n{:<28} {:>7} {:>7} {:>8} {:>9} {:>9} {:>7}".format(
        "dataset", "train", "uniq", "test⊂train", "inst⊂train", "same-doc", "verdict"))
    print("-" * 84)
    for r in results:
        if "error" in r:
            print("{:<28} {:>54}".format(r["dataset"], "ERROR " + r["error"][:32]))
            continue
        same = r.get("identical_documents_across_splits")
        same = same.get("test_in_train") if isinstance(same, dict) else "-"
        inst = r["instance_overlap"]["test_in_train"]
        print("{:<28} {:>7} {:>7} {:>8} {:>9} {:>9} {:>7}".format(
            r["dataset"],
            r["n_samples"]["train"],
            r["n_unique_transcripts"]["train"],
            "{}/{}".format(r["transcript_overlap"]["test_in_train"],
                           r["n_unique_transcripts"]["test"]),
            "{}/{}".format(inst, r["n_samples"]["test"]),
            same,
            r["verdict"]))

    viol = [r for r in results if r.get("verdict") == "VIOLATION"]
    if viol:
        print("\n{} dataset(s) with blocking violations:".format(len(viol)))
        for r in viol:
            for v in r["violations"]:
                print("  {}: {}".format(r["dataset"], v))
    print("\nwrote", a.json)

    if a.fail_on_violation and viol:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
