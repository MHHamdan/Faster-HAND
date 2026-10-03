#!/usr/bin/env bash
# Recover DAN's exact READ 2016 font inventory (41 files) from primary sources.
#
# DAN does not ship a font list as an input: OCR/ocr_dataset_manager.py:get_valid_fonts()
# walks a Fonts/ tree and keeps every .ttf whose cmap covers the dataset charset.
# third_party/DAN/Fonts/list_fonts_read_2016.txt records the RESULT for READ 2016:
#   21 DejaVu 2.370 + 18 Lato 2.006/2.007 + 2 Gentium Plus 5.000.
#
# Sources (primary / official only):
#   Lato 2.0 (files report 2.007, and 2.006 for the four Medium/Semibold faces)
#     upstream Lato2OFL release, redistributed byte-identically by Debian:
#     https://snapshot.debian.org/file/df0d7b9bc379b4617e86c87dadb6f2145116a734
#     (= fonts-lato_2.0.orig.tar.xz; SIL OFL 1.1)
#   Gentium Plus 5.000
#     SIL official download server:
#     https://software.sil.org/downloads/r/gentium/GentiumPlus-5.000.zip  (SIL OFL 1.1)
#   DejaVu 2.370
#     already present in this repository at Fonts/dejavu (DejaVu Fonts License, Bitstream Vera derived)
#
# Output: fonts_dan_read/ containing exactly the 41 files, plus CHECKSUMS.sha256.
# Font binaries are NOT committed (see .gitignore); this script plus the hashes reproduces them.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/fonts_dan_read"
WORK="${TMPDIR:-/tmp}/dan_fonts_$$"
LATO_URL="https://snapshot.debian.org/file/df0d7b9bc379b4617e86c87dadb6f2145116a734"
LATO_SHA256="160e7969c5d369496d13396f1d3ea90917c466c1625d9afe49003b92fbfafa91"
GENTIUM_URL="https://software.sil.org/downloads/r/gentium/GentiumPlus-5.000.zip"
GENTIUM_SHA256="335911f17bd2de4e43742e1d0367cfeff19a90abf7ed604f100a42705042e154"

mkdir -p "$WORK" "$OUT"

echo "== downloading Lato 2.0 (upstream Lato2OFL, via Debian snapshot) =="
curl -sSL -o "$WORK/fonts-lato_2.0.orig.tar.xz" "$LATO_URL"
echo "$LATO_SHA256  $WORK/fonts-lato_2.0.orig.tar.xz" | sha256sum -c -

echo "== downloading Gentium Plus 5.000 (SIL) =="
curl -sSL -o "$WORK/GentiumPlus-5.000.zip" "$GENTIUM_URL"
echo "$GENTIUM_SHA256  $WORK/GentiumPlus-5.000.zip" | sha256sum -c -

echo "== extracting the 41 files named by DAN =="
python3 - "$REPO" "$WORK" "$OUT" <<'PY'
import sys, os, tarfile, zipfile, shutil
repo, work, out = sys.argv[1:4]
wanted = [l.split('"')[1] for l in open(os.path.join(repo, "third_party/DAN/Fonts/list_fonts_read_2016.txt")) if '"' in l]
by_base = {os.path.basename(p): p.split('/')[-2] for p in wanted}
written = set()

t = tarfile.open(os.path.join(work, "fonts-lato_2.0.orig.tar.xz"))
for m in t.getmembers():
    b = os.path.basename(m.name)
    if m.isfile() and by_base.get(b) == "lato":
        open(os.path.join(out, b), "wb").write(t.extractfile(m).read()); written.add(b)

z = zipfile.ZipFile(os.path.join(work, "GentiumPlus-5.000.zip"))
for n in z.namelist():
    b = os.path.basename(n)
    if by_base.get(b) == "gentiumplus":
        open(os.path.join(out, b), "wb").write(z.read(n)); written.add(b)

for b, fam in by_base.items():
    if fam == "dejavu":
        src = os.path.join(repo, "Fonts/dejavu", b)
        if not os.path.exists(src):
            raise SystemExit(f"missing local DejaVu file: {src}")
        if os.path.islink(os.path.join(out, b)):
            os.unlink(os.path.join(out, b))
        shutil.copyfile(src, os.path.join(out, b)); written.add(b)

missing = set(by_base) - written
if missing:
    raise SystemExit(f"MISSING {len(missing)} files: {sorted(missing)}")
print(f"wrote {len(written)} font files to {out}")
PY

echo "== verifying versions, charset coverage and DAN's own selector =="
python3 "$REPO/scripts/verify_dan_fonts.py"

( cd "$OUT" && sha256sum *.ttf > CHECKSUMS.sha256 )
echo "== wrote $OUT/CHECKSUMS.sha256 =="
rm -rf "$WORK"
