# NOTICE

This file discharges **CeCILL-C Art. 6.4(3)** — the obligation to indicate, *in a text that
is easily accessible*, that the Software is used, its intellectual-property notices, and the
fact that it is governed by CeCILL-C. The README links here, and every CeCILL-C-governed
source file in this repository links here from its own header.

**The top-level `LICENSE` is MIT and does not apply to the files named in §1.** Read this
file before redistributing anything from this tree.

Licence text: [`licenses/LICENSE-CeCILL-C.md`](licenses/LICENSE-CeCILL-C.md)
(CeCILL-C v1.0, 2006-09-05; sha256 `405e0890e5997f766bfe0adcfad381077749b6c615d41dc4effb0baf271ada9f`,
md5 `4b685776510b47a6fd125c568e4493bb`) — **byte-identical to the copy distributed with the
upstream repositories**, verified against the pinned clone at commit `deab478a`.
MIT text: [`licenses/LICENSE-MIT.txt`](licenses/LICENSE-MIT.txt).

---

## 1. Denis Coquenet — the encoder and much of the training harness

> The convolutional feature extractor, the layout post-processing, the dataset and training
> managers, the metric manager, the dataset formatters and the transforms in this project are
> **Denis Coquenet's work**, distributed under the **CeCILL-C** licence, and are used here
> under that licence.
>
> Copyright Université de Rouen Normandie (1), INSA Rouen (2), tutelles du laboratoire LITIS
> (1 et 2). Contributor: Denis Coquenet.
>
> Upstream: <https://github.com/FactoDeepLearning/DAN> and
> <https://github.com/FactoDeepLearning/VerticalAttentionOCR>.

**Upstream pins.** The line-identity measurements in §1.1 and §1.2 were taken against these
exact commits. Re-clone them to reproduce the licence audit; neither tree is redistributed
here, because both are CeCILL-C code and the VerticalAttentionOCR clone additionally ships
IAM annotation XML (research-licensed transcriptions).

```bash
git clone https://github.com/FactoDeepLearning/DAN.git
git -C DAN checkout deab478a3eece2f53cdf83095be0d90cea46d156

git clone https://github.com/FactoDeepLearning/VerticalAttentionOCR.git
git -C VerticalAttentionOCR checkout 153cdc2cb055bdc7e28558e1bd23f49d9c681b24
```

| Repository | Commit | What this project uses from it |
|---|---|---|
| `FactoDeepLearning/DAN` | `deab478a3eece2f53cdf83095be0d90cea46d156` | the READ 2016 formatter (double-page pairing, paragraph enumeration), the metric manager (LOER graph construction), the `main_dan.py` training recipe, and the published READ 2016 page checkpoint used as the DAN comparison row |
| `FactoDeepLearning/VerticalAttentionOCR` | `153cdc2cb055bdc7e28558e1bd23f49d9c681b24` | the IAM Aachen/RWTH split lists (747 / 116 / 336 forms) |

The decoder architecture, the five-class layout-token scheme, the page protocol, the two-step
token positional code and the injected-error teacher forcing are also his (DAN and
Faster-DAN). The trained weights released here initialise from **his** released line
checkpoint (§3).

### 1.1 Files that retained the upstream notice — 18

```
hand/basic/generic_dataset_manager.py
hand/basic/generic_training_manager.py
hand/basic/__init__.py
hand/basic/metric_manager.py
hand/basic/post_pocessing_layout.py
hand/basic/scheduler.py
hand/basic/transforms.py
hand/basic/utils.py
hand/Datasets/__init__.py
hand/Datasets/dataset_formatters/generic_dataset_formatter.py
hand/Datasets/dataset_formatters/__init__.py
hand/Datasets/dataset_formatters/maurdor_formatter.py
hand/Datasets/dataset_formatters/read2016_formatter.py
hand/Datasets/dataset_formatters/rimes_formatter.py
hand/Datasets/dataset_formatters/utils_dataset.py
hand/models/baseline/fcn_encoder.py
hand/Datasets/format_read_dan_splits.py
hand/basic/layout_metrics.py
```

These are **Modified Software** in the sense of CeCILL-C Art. 5.3.2 wherever they were
changed here. They stay governed by CeCILL-C. They may not be shipped under MIT, and the
repository's top-level `LICENSE` (MIT) does not and cannot apply to them.

Fifteen of the eighteen are measurably modified against the pinned upstream clone and now
carry the Art. 5.2 modification statement reproduced in §1.3. The remaining three
(`hand/basic/__init__.py`, `hand/Datasets/__init__.py`,
`hand/Datasets/dataset_formatters/__init__.py`) contain the notice and no code, so there is
nothing modified to declare. Two files in §1.1 — `maurdor_formatter.py` and
`hand/Datasets/format_read_dan_splits.py` — have **no counterpart in the pinned upstream
clone**; they carry the notice because they reuse CeCILL-C-governed logic, and their
modification status against any earlier upstream release is therefore unverified. That is
recorded rather than glossed.

### 1.2 DAN derivatives whose notice has been RESTORED — 9

These nine files had had the upstream notice removed, which **CeCILL-C Art. 6.4(1) forbids**.
**The notice has been restored, byte-identical to upstream, in every one of them**, together
with the Art. 5.2 modification statement in §1.3. This was a release blocker and it is
closed; the restoration is comment-only and changes no behaviour.

Measured against the pinned upstream clone at `deab478a`, comment lines excluded, difflib
line identity:

| File in this project | Upstream | Lines identical |
|---|---|---|
| `hand/OCR/line_OCR/ctc/models_line_ctc.py` | `DAN/OCR/line_OCR/ctc/models_line_ctc.py` | 100.0 % (13/13) |
| `hand/OCR/line_OCR/ctc/trainer_line_ctc.py` | same path upstream | 84.2 % (64/76) |
| `hand/OCR/document_OCR/hand/main_hand.py` | `DAN/OCR/document_OCR/dan/main_dan.py` | 79.4 % (143/180) |
| `hand/OCR/line_OCR/ctc/main_syn_line.py` | same path upstream | 69.6 % (96/138) |
| `hand/models/baseline/attention.py` | `DAN/OCR/document_OCR/dan/models_dan.py` | 67.4 % (176/261) |
| `hand/OCR/ocr_utils.py` | `DAN/OCR/ocr_utils.py` | 65.0 % (13/20) |
| `hand/OCR/ocr_manager.py` | `DAN/OCR/ocr_manager.py` | 55.6 % (40/72) |
| `hand/OCR/ocr_dataset_manager.py` | `DAN/OCR/ocr_dataset_manager.py` | 55.3 % (766/1386) |
| `hand/models/baseline/dan_decoder.py` | `DAN/OCR/document_OCR/dan/models_dan.py` | 34.5 % (58/168) |

`attention.py` and `dan_decoder.py` are jointly a split of upstream `models_dan.py`; read
together their coverage is higher than either row alone. All 27 upstream DAN `.py` files
carry the notice.

**Verification.** Every one of the 27 files in §1.1 and §1.2 now carries the CeCILL-C notice
in its header, confirmed by `grep -rl CeCILL --include=*.py` over this repository. Two
further files match that grep — `release/hand_release/inference.py` and
`release/tools/generate_manifest.py` — because they *mention* CeCILL-C in prose; they are the
author's own work and are MIT (§2).

### 1.3 Art. 5.2 — modification statements

CeCILL-C Art. 5.2 requires each modification to carry an explicit notice naming the author of
the modification and the date it was created. **All twenty-four modified files now carry
one**, appended below the original notice. For the nine restored files (§1.2):

```
#  Modified by Mohammed Hamdan, 2025-2026. This file is a derivative of the upstream
#  file named in release/NOTICE.md section 1.2, which also records the measured line
#  identity against the pinned upstream commit. The modifications remain governed by
#  CeCILL-C; see release/NOTICE.md and release/licenses/LICENSE-CeCILL-C.md.
```

For the fifteen modified files that retained their notice (§1.1):

```
#  Modified by Mohammed Hamdan, 2025-2026, under CeCILL-C Article 5.3.2. The measured
#  divergence from the pinned upstream commit is recorded in release/NOTICE.md
#  section 1.1. This file remains governed by CeCILL-C; the repository's top-level
#  MIT LICENSE does not apply to it.
```

Both statements are comment-only. They changed the SHA-256 of the files they were added to,
which is why the `effective_code_state` digests inside the published run records do not match
the files as published; see `docs/REPRODUCIBILITY_ARTIFACTS.md` §5.

### 1.4 Art. 8 / 9 — the warranty and liability notice that must accompany distribution

> The Software is provided "as is" without any warranty of any kind, express or implied,
> including without limitation warranties of merchantability or fitness for a particular
> purpose. The Licensee acknowledges that the scientific and technical state of the art when
> the Software was distributed did not enable all possible uses to be tested and verified, nor
> the presence of defects to be detected. The Licensor shall not be liable for any direct or
> indirect damage arising from the use of the Software. See
> [`licenses/LICENSE-CeCILL-C.md`](licenses/LICENSE-CeCILL-C.md) Articles 8 and 9 for the
> operative text.

---

## 2. What in this release is *not* Coquenet's

CeCILL-C Art. 5.3.3 and 6.3 permit a **Related Module** — new source files that add functions
without modifying the Software — to be distributed under a different licence. On that basis
the following are the author's own work and are offered under
[MIT](licenses/LICENSE-MIT.txt):

```
release/hand_release/inference.py           the inference contract
release/tools/*.py                          export, evaluation, manifest, claim-language check
hand/Datasets/bucketing.py                  shape bucketing
tools/train.py, tests/
hand/models/baseline/spec_heads.py          speculative draft heads
hand/models/experimental/                   the untrained proposed architecture
hand/OCR/document_OCR/hand/trainer_std_hand.py, main_std_hand.py, metrics.py
tools/                                      every entry point
experiments/benchmark_suite/record.py
```

The boundary is written down rather than assumed, because CeCILL-C Art. 5.3.3 makes the
different-licence permission conditional on the module genuinely not modifying the Software.
Any file above that is later found to modify a CeCILL-C file moves to §1.1.

---

## 3. The initialisation checkpoint — CC BY 4.0

The released weights are trained starting from **Denis Coquenet's released line checkpoint**,
not from random initialisation. Attribution under **CC BY 4.0 §3(a)(1)** and the
modification statement under **§3(a)(1)(B)**:

> **Title:** Pretrained Document Attention Network for Handwritten Text Recognition
> **Creator:** Denis Coquenet (ORCID 0000-0001-5203-9423)
> **Source:** Zenodo, DOI [10.5281/zenodo.7244382](https://doi.org/10.5281/zenodo.7244382)
> **File used:** `fcn_read_2016_line_syn.pt`
> (md5 `191a4c5659be189b2160f56f3759850c`,
> sha256 `557d6c349131b113c1e316effdb028eff9b37bf39a48d0689125a3783b86a7c3`)
> **Licence:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
> **Modified:** yes. The encoder weights from that checkpoint were used to initialise a
> page-level model, which was then trained further on READ 2016 page images and synthetic
> pages. The released weights are therefore **Adapted Material** in the sense of CC BY 4.0.
> **Disclaimer:** the Licensed Material is provided as-is; see the CC BY 4.0 legal code,
> Section 5, Disclaimer of Warranties and Limitation of Liability.

Verified against the Zenodo REST record (`metadata.license` = `cc-by-4.0`,
`access_right` = open). The digests above are recorded in `data/README.md` and were computed
from the local copy; check yours against them before training.

CC BY 4.0 is **not** share-alike, and §3(a)(4) permits a different licence on the Adapted
Material provided attribution survives and no additional restriction bites the upstream
material. The released weights (GitHub release v1.1.0, outside the git history) are licensed
under **CC BY 4.0**, and each package carries the attribution above in its README.

**Coquenet's own weights are not re-hosted here.** Link to the Zenodo DOI instead: the
canonical deposit should stay canonical.

---

## 4. Training corpora

No corpus is redistributed in this release, in whole or in part, and no corpus image or
transcription is embedded in any released file. The corpora that the released weights were
trained on are named, with their licences, in each model card's Training Data section.

### 4.1 What the split manifests actually contain — enumerated, and corrected

An earlier version of this section claimed, under the heading "verified, not assumed", that
every manifest entry carries *exactly* seven fields. **That was wrong**, and four manifests
falsify it. The correction is recorded here rather than quietly replaced, because a notice
that overstates its own verification is worse than one that claims less.

Measured by enumerating every per-sample record in all 16 manifest files of the private
archive. **Five of those sixteen are published here**, under
`experiments/benchmark_suite/manifests/`, for the corpora the paper reports on. The **union
of fields across all sixteen is twelve**, and no single manifest carries all twelve:

| field | in how many of the 16 | what it is |
|---|---:|---|
| `image_sha256` | 14 | digest of the formatted image the pipeline saw |
| `name` | 14 | the formatted sample name, e.g. `test_0.png` |
| `text_sha256` | 14 | digest of the transcription — the digest, never the text |
| `n_chars` | 9 | transcription length |
| `source_sha256` | 4 | digest of the original scan |
| `n_lines` | 3 | count |
| `source_scan` | 3 | original scan filename — READ 2016 only |
| `n_regions` | 2 | count |
| `source_scans` | 2 | original scan filenames — READ 2016 multi-page builds only |
| `region_bbox_300dpi` | 1 | text-region bounding box from the source PAGE-XML |
| `region_index` | 1 | index of the region within its page |
| `written_page_number` | 1 | the page number **written on the manuscript** |

Two of the sixteen files carry no per-sample records at all: `Bentham_page_sequences.json`
and `READ_2016_page_sequences.json` describe consecutive-page windows and carry counts,
run lengths and window lists instead.

### 4.2 The three entries that are more than a hash or a count

1. **`READ_2016_double_page_sem_dan.json` — `written_page_number`** (217 entries). The page
   number written on the manuscript. That is transcribed content, three digits per double
   page. READ 2016 is CC BY 4.0, so this is an attribution obligation (§5) and not a licence
   breach.
2. **`READ_2016_paragraph_dan.json` — `region_bbox_300dpi`** (1,983 entries) with
   `region_index`. Text-region geometry from the corpus's own PAGE-XML. Layout, not text,
   and again READ 2016 only.
3. **`Bentham_page_sequences.json` — `name_by_document`** (433 entries). Bentham R0
   catalogue identifiers mapped to formatted sample names. Bentham's **image rights are
   unconfirmed**, so **no Bentham manifest is published here at all**; the corpus is outside
   the reported results and the identifiers carried the one rights risk in the set.

The READ 2016 manifests published here also carry `source_scan` / `source_scans`
(e.g. `Seite0405.JPG`), which name files of a CC BY 4.0 corpus.

With those named, the accurate summary is: **the manifests carry digests, opaque sample
names, counts, and — for READ 2016 only — the source-scan filenames, one manuscript page
number field and one region-geometry field.** No transcription text and no image data is
present in any of them.

---

## 5. READ 2016

> **READ 2016 / Ratsprotokolle (Bozen).** Digitised council minutes, 1470–1805.
> Deposited by Toselli, Romero, Villegas, Vidal and Sánchez, Zenodo DOI
> [10.5281/zenodo.1297399](https://doi.org/10.5281/zenodo.1297399) (which supersedes and
> extends [10.5281/zenodo.218236](https://doi.org/10.5281/zenodo.218236), deposited by
> Sánchez, Romero, Toselli and Vidal), produced in the context of the READ project
> (EU Horizon 2020). Both deposits declare **CC BY 4.0**.

An earlier dataset card in the development archive contradicted itself on this point, calling
READ 2016 "competition terms" in one place and CC BY 4.0 in another. **The licence audit
resolved it from the primary Zenodo records in favour of CC BY 4.0**, and `data/README.md` —
the only dataset document published here — states that and nothing else. The residual caveat
stands and is stated rather than hidden: the Zenodo licence field is the depositors'
assertion about a digitised archival collection. It settles what a redistributor may rely
on, not the underlying archive's rights. This repository redistributes none of the corpus.

---

## 6. Fonts

No font binary is shipped in this release. Synthetic training pages for the released page
model were rendered with the 41 files in `fonts_dan_read/` (DejaVu, Lato OFL, Gentium Plus
OFL), reproducible byte-exactly from upstream with `scripts/setup_dan_fonts.sh` and
`fonts_dan_read/CHECKSUMS.sha256`. Font data does not enter a checkpoint, and images rendered
with a font are not Modified Versions of it under OFL 1.1, so the released weights are
unencumbered by the font choice regardless. The 154 files under `Fonts/` have **no recorded
provenance** and at least one family (`tangerine`) carries no licence grant in its binary;
they are not shipped and should not be.
