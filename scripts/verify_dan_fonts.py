#!/usr/bin/env python3
"""Verify that fonts_dan_read/ reproduces DAN's exact READ 2016 font inventory.

Checks, in order:
  1. the 41 filenames of third_party/DAN/Fonts/list_fonts_read_2016.txt are all present;
  2. each file's internal version string matches the version DAN recorded beside it;
  3. every file covers the READ 2016 charset (this is what DAN's selector actually tests);
  4. DAN's own get_valid_fonts() logic over fonts_dan_read/ returns exactly 41 paths.

Exit code 0 only if all four hold. Writes a JSON report next to the fonts.
"""
import json
import os
import pickle
import sys
import hashlib

from fontTools.ttLib import TTFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(REPO, "fonts_dan_read")
LIST = os.path.join(REPO, "third_party/DAN/Fonts/list_fonts_read_2016.txt")
LABELS = os.path.join(REPO, "formatted/READ_2016_page_sem_dan/labels.pkl")
REPORT = os.path.join(FONT_DIR, "VERIFICATION.json")


def dan_list():
    """(basename, family, recorded_version) for each of DAN's 41 entries."""
    out = []
    for line in open(LIST):
        if '"' not in line:
            continue
        path = line.split('"')[1]
        version = line.split("# version")[1].strip() if "# version" in line else None
        out.append((os.path.basename(path), path.split("/")[-2], version))
    return out


def read_charset():
    charset = pickle.load(open(LABELS, "rb"))["charset"]
    return "".join(sorted({c for c in charset if len(c) == 1 and c.isprintable() and ord(c) < 0x2460}))


def char_in_font(char, font):
    return any(ord(char) in table.cmap for table in font["cmap"].tables if table.isUnicode())


def main():
    entries = dan_list()
    charset = read_charset()
    problems, records = [], []

    for base, family, recorded in entries:
        path = os.path.join(FONT_DIR, base)
        if not os.path.exists(path):
            problems.append(f"missing file: {base}")
            continue
        blob = open(path, "rb").read()
        font = TTFont(path, lazy=True)
        version = next((r.toUnicode() for r in font["name"].names if r.nameID == 5), "")
        missing = [c for c in charset if not char_in_font(c, font)]
        # DAN records e.g. "2.370"; the name table says "Version 2.37" / "Version 2.007; 2014-02-27"
        norm = version.replace("Version", "").split(";")[0].strip()
        ok_version = recorded is None or norm == recorded or norm == recorded.rstrip("0") or recorded.startswith(norm)
        if not ok_version:
            problems.append(f"{base}: version {norm!r} != recorded {recorded!r}")
        if missing:
            problems.append(f"{base}: does not cover {missing}")
        records.append({
            "file": base, "family": family, "recorded_version": recorded,
            "actual_version": norm, "covers_charset": not missing,
            "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
        })

    # DAN's selector, applied to fonts_dan_read/
    valid = []
    for dirpath, _, filenames in os.walk(FONT_DIR):
        for name in filenames:
            if not name.endswith(".ttf"):
                continue
            path = os.path.join(dirpath, name)
            try:
                font = TTFont(path, lazy=True)
                if all(char_in_font(c, font) for c in charset):
                    valid.append(path)
            except Exception as exc:                       # noqa: BLE001
                problems.append(f"{name}: unreadable ({exc})")

    report = {
        "charset_size": len(charset),
        "expected_files": len(entries),
        "present_files": len(records),
        "get_valid_fonts_count": len(valid),
        "problems": problems,
        "fonts": sorted(records, key=lambda r: r["file"]),
    }
    if os.path.isdir(FONT_DIR):
        json.dump(report, open(REPORT, "w"), indent=2)

    print(f"charset            : {len(charset)} characters")
    print(f"DAN list           : {len(entries)} files")
    print(f"present            : {len(records)} files")
    print(f"get_valid_fonts()  : {len(valid)} files")
    for fam in ("lato", "gentiumplus", "dejavu"):
        n = sum(1 for r in records if r["family"] == fam)
        vers = sorted({r["actual_version"] for r in records if r["family"] == fam})
        print(f"  {fam:12s} {n:2d} files, versions {vers}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    if len(valid) != len(entries):
        print(f"\nFAIL: selector returned {len(valid)}, expected {len(entries)}")
        return 1
    print("\nOK: fonts_dan_read reproduces DAN's exact 41-font inventory")
    return 0


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.exit(main())
