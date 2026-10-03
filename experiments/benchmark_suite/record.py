"""
Run registry for HAND V2.

Every experiment writes one JSON record here -- config, commit, dataset, seed, GPU and
metrics -- so that no table cell exists without a file behind it. Records are
append-only: a run is written when it starts (status "running") and rewritten when it
finishes. The registry is `experiments/benchmark_suite/registry/<run_id>.json`.

    from experiments.benchmark_suite.record import RunRecord
    rec = RunRecord.start(name="exp3_memory_s0", kind="train", dataset=..., seed=0, config=...)
    ...
    rec.finish(metrics={"test": {...}, "valid": {...}}, artifacts={...})

The record is deliberately plain JSON with no external dependency, so it can be read by
anything and diffed in git.
"""
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY = os.path.join(ROOT, "experiments", "benchmark_suite", "registry")
SCHEMA_VERSION = 1


def _run(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# Training-relevant source roots. The content hash below identifies the code that ACTUALLY
# ran, independently of the commit recorded at launch -- see experiments/PROVENANCE_AUDIT.md
# section 2, where a dirty-tree launch made the recorded commit point at the PARENT of the
# commit containing the code the run used.
CODE_ROOTS = ("hand", "hand_v2", "tools")


def effective_code_state(roots=CODE_ROOTS):
    """Content hash over the Python sources on disk at launch, dirty tree included.

    This is the identifier to cite for a manuscript-facing result: unlike `commit` it cannot
    be made wrong by an uncommitted edit, and unlike `dirty_files` it pins the content rather
    than only naming the files that differed.
    """
    import hashlib
    h = hashlib.sha256()
    per_file, n = {}, 0
    for base in roots:
        base_dir = os.path.join(ROOT, base)
        if not os.path.isdir(base_dir):
            continue
        for root, dirs, fnames in os.walk(base_dir):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for fn in sorted(fnames):
                if not fn.endswith(".py"):
                    continue
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, ROOT)
                try:
                    with open(fp, "rb") as fh:
                        digest = hashlib.sha256(fh.read()).hexdigest()
                except OSError:
                    continue
                h.update(rel.encode()); h.update(digest.encode())
                per_file[rel] = digest
                n += 1
    return {"effective_code_sha256": h.hexdigest(), "n_files": n,
            "roots": list(roots), "per_file_sha256": per_file}


def git_provenance():
    head = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    diff = _run(["git", "diff", "HEAD", "--"] + list(CODE_ROOTS))
    import hashlib
    return {
        # the commit HEAD pointed at when the job was launched -- NOT necessarily the code that ran
        "commit": head,
        "branch": branch,
        "dirty": bool(dirty) if dirty is not None else None,
        "dirty_files": dirty.splitlines() if dirty else [],
        "describe": _run(["git", "describe", "--always", "--dirty"]),
        # the uncommitted delta against that commit, over training-relevant roots only
        "code_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "code_diff_bytes": len(diff.encode()) if diff else 0,
        "code_diff": diff if (diff and len(diff) <= 200000) else None,
        "provenance_note": ("Cite `effective_code_state.effective_code_sha256` for any "
                            "manuscript-facing result. `commit` records the launch state only."),
    }


def gpu_provenance():
    import torch
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "devices": []}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["devices"].append({"index": i, "name": p.name, "total_mem_MiB": p.total_memory // 2 ** 20,
                                    "capability": "{}.{}".format(p.major, p.minor)})
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    info["driver"] = drv.splitlines()[0] if drv else None
    return info


def env_provenance():
    return {"host": socket.gethostname(), "user": getpass.getuser(), "python": sys.version.split()[0],
            "platform": platform.platform(), "cpu_count": os.cpu_count(), "argv": sys.argv,
            "cwd": os.getcwd()}


def _jsonable(o):
    """Config dicts hold classes and functions; record their names, not their reprs."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if hasattr(o, "__name__"):
        return o.__name__
    return str(o)


class RunRecord:
    def __init__(self, d):
        self.d = d

    @property
    def run_id(self):
        return self.d["run_id"]

    @property
    def path(self):
        return os.path.join(REGISTRY, self.run_id + ".json")

    @classmethod
    def start(cls, name, kind, dataset, seed, config, notes=None, experiment=None, output_dir=None):
        # (effective_code_state is attached alongside git provenance; see the record body)
        """kind: train | eval | profile | analysis. dataset: dict with at least name/level/path."""
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        d = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "{}_{}_{}".format(ts, name, uuid.uuid4().hex[:6]),
            "name": name,
            "experiment": experiment,
            "kind": kind,
            "status": "running",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_utc": None,
            "dataset": dataset,
            "seed": seed,
            "git": git_provenance(),
            "effective_code_state": effective_code_state(),
            "gpu": gpu_provenance(),
            "env": env_provenance(),
            "config": _jsonable(config),
            "output_dir": output_dir,
            "metrics": {},
            "artifacts": {},
            "notes": notes,
        }
        rec = cls(d)
        rec.write()
        return rec

    def update(self, **fields):
        self.d.update(_jsonable(fields))
        self.write()

    def finish(self, metrics, status="complete", artifacts=None, **extra):
        self.d["metrics"] = _jsonable(metrics)
        if artifacts:
            self.d["artifacts"] = _jsonable(artifacts)
        self.d.update(_jsonable(extra))
        self.d["status"] = status
        # Validity sanity flag (integrity audit item C4): a "complete" run whose
        # validation or test CER never came below 0.5 is a collapsed run, not a result.
        try:
            m = self.d["metrics"]
            cers = [float(m[s]["cer"]) for s in ("valid", "test") if isinstance(m.get(s), dict) and "cer" in m[s]]
            if status == "complete" and cers and min(cers) > 0.5:
                self.d["validity"] = "SUSPECT: CER > 0.5 on every evaluated split (collapsed run?)"
        except Exception:  # noqa: BLE001 - never let the flag break a record
            pass
        self.d["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.write()

    def write(self):
        os.makedirs(REGISTRY, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f, indent=2, sort_keys=False)
        os.replace(tmp, self.path)


def load_registry():
    recs = []
    if not os.path.isdir(REGISTRY):
        return recs
    for fn in sorted(os.listdir(REGISTRY)):
        if fn.endswith(".json"):
            with open(os.path.join(REGISTRY, fn)) as f:
                recs.append(json.load(f))
    return recs


def parse_predict_file(path):
    """outputs/<run>/results/predict_<set>_<epoch>.txt -> dict of floats."""
    out = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    out[k.strip()] = v.strip()
    return out
