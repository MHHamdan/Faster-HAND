"""Download the released HAND weights from the GitHub release and verify them.

    from hand_release.hub import download
    path = download("hand-read2016-page")          # -> weights/hand-read2016-page/

    python -m hand_release.hub --list
    python -m hand_release.hub hand-read2016-page [--dest weights]

Each package is a zip with the exported model (model.safetensors, config.json, charset.json,
preprocessor_config.json), the speculative draft heads where they exist, and a README with the
reported results and the CC BY 4.0 attribution. The SHA-256 of every zip is pinned below; a
download that does not match is deleted and refused.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

RELEASE = "https://github.com/DocumentRecognitionModels/HAND-Decoding/releases/download/v1.1.0/"

MODELS = {
    "hand-read2016-page": {
        "sha256": "877246ec2503da82636a820d99091c9a99c234d5396b8cb8e9bc9ab8fc4f645c",
        "pages": 1, "heads": True,
        "about": "single-page model reported in the paper (READ 2016 test CER 3.55 %)",
    },
    "hand-read2016-page-compact": {
        "sha256": "c53984243256c9e90aee7aea887fd8706034a39cf31ddc81631251f58d6d7791",
        "pages": 1, "heads": True,
        "about": "compact single-page model, shared visual key/value projection (CER 4.00 %)",
    },
    "hand-read2016-double-page": {
        "sha256": "42928757e0d561343db7a7ef7452886ef002310770ea8ffa7164361728fbe9ea",
        "pages": 2, "heads": True,
        "about": "double-page adapted model (READ 2016 double-page test CER 3.60 %)",
    },
    "hand-read2016-triple-page": {
        "sha256": "1c09c18ecd3aac0a99011dee05b593832b9e4a1ec818efc2779e3c742262d58d",
        "pages": 3, "heads": False,
        "about": "triple-page adapted model (READ 2016 triple-page test CER 3.48 %)",
    },
}

DEFAULT_DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "weights")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name, dest=DEFAULT_DEST, force=False):
    """Return the directory holding the extracted model, downloading it if needed."""
    if name not in MODELS:
        raise KeyError("unknown model %r; choose one of %s" % (name, ", ".join(MODELS)))
    target = os.path.join(dest, name)
    if os.path.isfile(os.path.join(target, "model.safetensors")) and not force:
        return target
    os.makedirs(dest, exist_ok=True)
    url = RELEASE + name + ".zip"
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=dest)
    os.close(fd)
    try:
        print("downloading %s" % url, file=sys.stderr)
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        got = _sha256(tmp)
        if got != MODELS[name]["sha256"]:
            raise RuntimeError("checksum mismatch for %s: expected %s, got %s"
                               % (name, MODELS[name]["sha256"], got))
        with zipfile.ZipFile(tmp) as z:
            for member in z.namelist():
                if not member.startswith(name + "/") or ".." in member:
                    raise RuntimeError("unexpected path in archive: %s" % member)
            z.extractall(dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="model to download")
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true", help="download again even if present")
    ap.add_argument("--list", action="store_true", help="list the released models")
    a = ap.parse_args()
    if a.list or not a.name:
        for k, v in MODELS.items():
            print("%-28s %s" % (k, v["about"]))
        return
    print(download(a.name, a.dest, a.force))


if __name__ == "__main__":
    main()
