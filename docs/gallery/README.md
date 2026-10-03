# Qualitative gallery

Figures that supplement the paper but are not included in either PDF. Every prediction shown
was read from the stored result files (`experiments/qualitative/predictions/*.json`,
`experiments/multipage/*.json`); nothing was decoded or edited for display. The colour key is
the one used throughout the paper: page grey, page number green, section purple, annotation
orange, body blue; red marks a character or layout-token error.

## READ 2016 single pages (page model, greedy decoding)

| File | What it shows |
|---|---|
| `read2016_page_test_test_35.pdf` | Lowest-error page of the test split (CER 0.63 %, three edits in 474 characters). Twenty-eight lines in three sections with two marginal annotations are emitted in the annotation-before-body order of the reference, and all layout tokens are correct. The three residual errors are a diacritic (*Gräben*), an inserted vowel and a two-letter substitution. |
| `read2016_page_test_test_25.pdf` | A page whose three sections each open with a marginal note. The decoder emits the annotation region before the body it refers to in every section, as the label scheme requires; the layout sequence is exact and the CER is 2.97 %. |
| `read2016_page_test_test_13.pdf` | A difficult hand. The layout sequence is exact but the CER rises to 7.78 %, with errors clustered in the cramped final lines; the segmentation-free baseline evaluated under the same protocol reaches 8.67 % on this page. |

## Multi-page inputs

| File | What it shows |
|---|---|
| `read2016_double_page_zeroshot_vs_adapted_test_test_23.pdf` | Double-page test image `test_23` decoded by the page model without adaptation (a) and by the adapted double-page model (b). Without adaptation the model reads the left page (2.5 % CER on that page), closes it and stops, so the transition the reference requires is never taken and the image-level CER is 48.4 %. After adaptation the decoder takes the transition and reads the right page (3.1 %), giving 3.17 % for the image. |
| `read2016_triple_page_zeroshot_test_test_0.pdf` | The page model applied without adaptation to triple-page test image `test_0`: only the first page is transcribed (3.0 % CER on that page, 71.7 % overall); the page number *203* is decoded as *207* and the boundary between the second and third sections is omitted. |

## Cross-script composite

| File | What it shows |
|---|---|
| `cross_script_comparison.pdf` | A READ 2016 page with layout tokens, an IAM form and a KHATT paragraph (right to left) side by side; each is decoded by a model trained on its own corpus. The panels duplicate Fig. 4 of the paper and row (a) of the IAM and KHATT figures of the supplementary. `iam_representative.pdf` and `khatt_representative.pdf` are its two source panels. |

## Animations (planned)

The repository ships `docs/assets/hand_decoding_test_11.gif` (greedy decoding of
test page 11 seen through the decoder's cross-attention; static equivalent:
`docs/assets/decoding_test_11.png`).
Two further animations can be rendered from stored data without any new decode and are planned
for this directory: the first 40 verification passes of speculative decoding on the same page
(from `experiments/qualitative/decoding_test_11_spec_steps.json`; static equivalent:
`docs/assets/decoding_test_11_speculative.png`), and a
line-by-line reveal of the adapted triple-page decode of `test_5` in reference reading order
(from `experiments/multipage/adaptation/FT_TRIPLE_triple_page.json`; static equivalent:
`docs/assets/qualitative_triple_page.png`).
