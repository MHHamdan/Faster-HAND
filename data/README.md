# Datasets

**This directory is empty in a clone, by design.** Everything in it except this file is
git-ignored. No dataset — no image, no transcription, no annotation — is redistributed in
this repository.

Locally it holds `data/<NAME>` symlinks created by `scripts/setup_dataset_links.sh`.

```bash
export HAND_EXTERNAL_DATA=/path/to/your/corpora      # corpora kept outside the repo

bash scripts/setup_dataset_links.sh --check          # report only, change nothing
bash scripts/setup_dataset_links.sh                  # create/refresh the symlinks
bash scripts/setup_dataset_links.sh --clean          # remove the links, never the targets
```

`configs/datasets_registry.yaml` is the machine-readable registry. Corpora outside the
repository are written there as `${HAND_EXTERNAL_DATA}/<corpus>`; corpora inside it are
repo-relative. Absolute machine paths are deliberately not published — they are meaningless
in your checkout. With the variable unset, external entries report `NOTARGET` and no link is
created. The script never copies, never downloads and never writes into a dataset.

---

## READ 2016 — the primary benchmark

| | |
|---|---|
| **Name** | READ 2016 / Ratsprotokolle (Bozen) — digitised council minutes, 1470–1805 |
| **Official source** | Zenodo DOI [10.5281/zenodo.1297399](https://doi.org/10.5281/zenodo.1297399), which supersedes [10.5281/zenodo.218236](https://doi.org/10.5281/zenodo.218236) and adds the ICFHR 2016 competition test set |
| **Licence / access** | **CC BY 4.0.** Freely downloadable. Attribution required if you redistribute any of it; this repository redistributes none |
| **Subset the paper uses** | page level, `_sem_dan` layout-token scheme — **350 train / 50 valid / 50 test**, charset 99 |

**Expected raw structure.** The formatter reads exactly these paths:

```
raw_READ2016/
├── PublicData/Training/      { *.JPG, page/*.xml }
├── PublicData/Validation/    { *.JPG, page/*.xml }
└── Test-ICFHR-2016/          { *.JPG, page/*.xml }
```

**Preprocessing.**

```bash
python hand_v2/data/format_read_dan_splits.py --levels page
```

This symlinks the raw scans into a temporary view and runs the unchanged formatter
`hand/Datasets/dataset_formatters/read2016_formatter.py`; nothing is copied or downloaded.
It asserts its own split counts and fails loudly rather than write a wrong dataset.

```
formatted/READ_2016_page_sem_dan/     350 / 50 / 50, charset 99     <- what the paper uses
formatted/READ_2016_page_sem/         350 / 50 / 50, charset 95     <- V1 scheme
```

**The two are not interchangeable.** `_sem_dan` carries DAN's five layout tokens
(`ⓟ ⓝ ⓢ ⓐ ⓑ` with closing forms), `_sem` carries three. A checkpoint trained on one will not
load against the other's charset, and the charset sizes above are the check.

`--levels double_page paragraph` rebuilds the other construction levels (169/24/24 and
1,602/182/199); neither is used by the page model.

**Resolution, and a failure mode.** Pages in `formatted/` are already at the model's 150 dpi
working resolution; the raw scans are 300 dpi and the formatter halved them once. **Anything
that resizes a formatted page again is wrong** — the docstring of
`release/hand_release/inference.py` records that failure dropping an entire line of
transcription while the model still looked confident.

## The initialisation checkpoint

Training does not start from random weights.

| | |
|---|---|
| **Source** | Zenodo record 7244382, DOI [10.5281/zenodo.7244382](https://doi.org/10.5281/zenodo.7244382) |
| **File** | `fcn_read_2016_line_syn.pt`, 20,855,873 bytes |
| **md5** | `191a4c5659be189b2160f56f3759850c` |
| **sha256** | `557d6c349131b113c1e316effdb028eff9b37bf39a48d0689125a3783b86a7c3` |
| **Licence** | **CC BY 4.0** — §3(a) attribution is the binding constraint on any weights derived from it |

Place it anywhere and pass that path to `--init-from`; the documented commands use
`weights/dan/fcn_read_2016_line_syn.pt`. **Do not re-host the file** — link to the Zenodo
record. This checkpoint, not the surrounding CeCILL-C code, is why the trained weights are
CC BY 4.0; see [`release/NOTICE.md`](../release/NOTICE.md) §3.

## The 41 synthetic-curriculum fonts

The synthetic-page curriculum draws from a specific font inventory. DAN does not publish a
font list as an input — its selector walks a tree and keeps every `.ttf` whose cmap covers
the dataset charset — so the inventory is reconstructed from primary sources.

```bash
bash scripts/setup_dan_fonts.sh        # Lato 2.0, Gentium Plus 5.000, DejaVu 2.370
python scripts/verify_dan_fonts.py     # versions, charset coverage, DAN's own selector
sha256sum -c fonts_dan_read/CHECKSUMS.sha256
```

No font binary is committed. The script downloads from Debian's snapshot mirror and SIL's
download server and verifies an archive hash before extracting. All 41 files (21 DejaVu
2.370, 18 Lato 2.006/2.007, 2 Gentium Plus 5.000) report `covers_charset: true` over the
88-character synthetic charset.

## The other corpora

| Corpus | Access | Preprocessing | Redistributable |
|---|---|---|---|
| **IAM** | register at <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>; place `lines.tgz` / `forms*.tgz` and the XML under `IAM/` | `hand/Datasets/dataset_formatters/iam_formatter.py` | **no** — research licence, registration required |
| **KHATT** | request from <https://khatt.ideas2serve.net/>; unpack under `KHATT/` | `khatt_formatter.py` | **no** — research licence, request required |
| **AHAWP** | public | `ahawp_formatter.py` | no. **Read the warning below before reporting any AHAWP number** |
| **RIMES**, **MAURDOR** | licensed access required; neither is present here | `rimes_formatter.py`, `maurdor_formatter.py` | no |
| **Bentham**, **Saint Gall**, **Washington** | per-corpus terms; formatters not built | — | no |

A reader who has not registered with the IAM and KHATT providers **cannot check those
numbers**. That limit is real and is stated in
[`docs/reproducibility.md`](../docs/reproducibility.md) section 9.

## Split manifests — verify you built the same splits

`experiments/benchmark_suite/manifests/` carries hash-only manifests for the corpora the
paper reports on: per-sample `image_sha256`, `text_sha256`, `n_chars` and the formatted
sample name. **No transcription text and no image data is present in any of them**, so they
can be published while the corpora cannot. Compare yours against them before quoting a
number.

## Check before you quote a number

```bash
python tools/audit_dataset_integrity.py --json dataset_audit.json --fail-on-violation
python tools/audit_corpus_text_overlap.py
```

The first checks transcription overlap, duplicate documents by content hash and writer
overlap; it exits non-zero on a blocking violation, so it can gate CI. The second measures
how much held-out *text* is already present in training as character 20-grams — a different
question, and the companion to the first, not a substitute.

Three verdicts you must know before any cross-corpus comparison
([`docs/ablations.md`](../docs/ablations.md) §4):

- **`KHATT_line` is contaminated** — 470 of 999 test instances (**47.0 %**) have a
  transcription that occurs in training.
- **AHAWP measures memorisation, not recognition.** Its entire transcription set — 65 / 10 /
  3 distinct strings for character / word / paragraph — is shared across every split. AHAWP
  numbers must never be reported as recognition or generalisation, and they appear in no
  results table here.
- **RIMES is not text-disjoint** — 0.394 mean 20-gram test-in-train coverage against a READ
  2016 control of 0.065. Its unseen-text role was withdrawn. **IAM Aachen (0.0007) is the
  only corpus here that supports an unseen-text claim.**
