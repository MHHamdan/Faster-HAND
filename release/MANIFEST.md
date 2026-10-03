# MANIFEST

Every file published in this repository, with its SHA-256 **as published here**. Nothing
else: no private path, no internal document, no staging artefact. This file is the only
exclusion, since it cannot carry its own digest.

**The digests below are the ones a reader can check.** Where a published evidence artefact
was redacted before release (paths, a hostname, a user name, see below) its
digest differs from the development archive's, and the digest recorded here is the
post-redaction one. `docs/REPRODUCIBILITY_ARTIFACTS.md` section 5 lists every redaction and
what it touched; each is metadata-only and changes no measured value.

Two digests in this repository deliberately do **not** agree, and that is explained rather
than reconciled: the `effective_code_state.per_file_sha256` block inside each published run
record is the digest of the source **as it executed**, before the CeCILL-C notices of
`release/NOTICE.md` section 1 were restored. Those restorations are comment-only, so the
behaviour each record describes is unchanged. See `docs/REPRODUCIBILITY_ARTIFACTS.md`
section 5.

Verify the whole tree from the repository root:

```bash
sed -nE 's/^\| `([^`]+)` \| `([0-9a-f]{64})`.*/\2  \1/p' release/MANIFEST.md > /tmp/hand.sha256
sha256sum -c /tmp/hand.sha256
```

**323 files, 13,234,108 bytes.** Generated 2026-10-02. Regenerate with this script after any change to the tree.


## Root — project metadata, environment, licence

| File | SHA-256 | Bytes |
|---|---|---:|
| `.gitignore` | `88798a5ad20ebcf40b719505682fa3077ca889ca6f0469a592891fbd20aa1546` | 4,921 |
| `CITATION.cff` | `04933b518eb9acd8f1928bab10046b276edb69e6553aa88bcc436870d14020f0` | 3,186 |
| `LICENSE` | `04a0056cb7d15f3c3824ac1901f1c678d2e98b02721ab80f2d7dd4e93de20702` | 1,072 |
| `README.md` | `37943f4efad84a80ec1b236b66a6fe1be4a588ebcc9255bd287c61e962e5bc4a` | 14,091 |
| `dataset_audit.json` | `2581dc1b6ce607973071edfc554ba1b359be8e27e9c49619fe2a20d877ecbf7d` | 49,093 |
| `environment.yml` | `3ad9240a921c022c70d124320a772c8475a15c1772411a72e61cb168186c3017` | 1,328 |
| `requirements-pinned.txt` | `ca57b6cd97efd9137c1510e4e875cd75d683a63433b8ec997fc043ed15ae930f` | 1,351 |
| `requirements.txt` | `b20bd88243ed85937e3e42f64a6a56c05dc5ad9fc78c1f1d877f5f7cae42d57a` | 1,133 |
| `setup.py` | `3807cb10f2d48839442668b67da9a67ff9295c74ae7e09bb2ec37ef728b4aca9` | 1,492 |

## Documentation

| File | SHA-256 | Bytes |
|---|---|---:|
| `docs/REPRODUCIBILITY_ARTIFACTS.md` | `f5af51a934eff930f33bd5a9dd7ef3c1c579197772e7bf2523e2ae5114fdeae6` | 19,906 |
| `docs/ablations.md` | `58b17e6d7358e81a6e22b3006bee87180a735ea743b4799a09d00c619e4089ba` | 13,380 |
| `docs/assets/decoding_test_11.png` | `aa37d283e10676ffa1478685a90e74582f12b7a1430f77739dac416a9879ccc3` | 429,165 |
| `docs/assets/decoding_test_11_speculative.png` | `5059ef4c7d14de17a92c40dd25af1cbb86c1a2670846aca7a4d46dc41a391045` | 66,882 |
| `docs/assets/hand_architecture.png` | `7aa5a2f0aa8a94d1f86e0239f72c44e8770f64c022e27d226cf6681170a907aa` | 206,151 |
| `docs/assets/hand_architecture.tex` | `4e8cdcc127e1ea22bcfda74e1f977d20f044359e33352971b47b71196cd771f2` | 13,367 |
| `docs/assets/hand_decoding_test_11.gif` | `f80805110149f837394fb79c6c6ed0b8272e1479e27968b81d0805918fb2c84e` | 1,147,175 |
| `docs/assets/qualitative_double_page.png` | `23f479eafb2d53b78b61d27fb1306f1e8ee2685e7032371f38ab822d8cdf66c2` | 980,676 |
| `docs/assets/qualitative_triple_page.png` | `13485c499763bba75cd185d61c484a7563022fe7403d8709410d763848200ecd` | 802,507 |
| `docs/changelog.md` | `150811871ee9c53c44b8bf85d6b094204bda474f2ae2c8abf00ac81e5965dcc4` | 1,963 |
| `docs/gallery/README.md` | `9993e72e00d04ad9d549cee7c8189242d42587e59297522820140c0bc10362d5` | 3,544 |
| `docs/gallery/cross_script_comparison.pdf` | `9f005c98e1dd0e62d61c989f7d5429042ec3c8967324023454a7c24395cf8e99` | 341,608 |
| `docs/gallery/iam_representative.pdf` | `b2e4485dec9ba998e456ffa333aef0fde9d6e5b07bc00b1ab9d214b88f1a3f87` | 55,102 |
| `docs/gallery/khatt_representative.pdf` | `e76e1b34eb61b3002cacbfc26d9ded85e0c7513636c1518a02a73f36d808b0ea` | 69,741 |
| `docs/gallery/read2016_double_page_zeroshot_vs_adapted_test_test_23.pdf` | `b8784f51072ac0d53be959cda1ccf7363af2a1986fb0f3f02b4ecfc0a2534b2e` | 403,253 |
| `docs/gallery/read2016_page_test_test_13.pdf` | `31a6ff747253191f4fd2cffa4017d3dcceb4f0b6348482f8780e3be8adbac4d9` | 211,286 |
| `docs/gallery/read2016_page_test_test_25.pdf` | `92eb5d3a68c306cb2e54df4c4e8d4229921b28377638abc67d77ec901bc9bdba` | 200,137 |
| `docs/gallery/read2016_page_test_test_35.pdf` | `28c7e7bc64142a23a0ebc0bbf1525acd4fe2e542a9b17b825c8b03e699e74dbe` | 230,974 |
| `docs/gallery/read2016_triple_page_zeroshot_test_test_0.pdf` | `751d1686e73f327299000f3357fecf4d662947312f954e38da4acbf56deb2e5a` | 568,380 |
| `docs/model_card.md` | `99e90be745df171440a4b896c2ee46ce2165000139e79158cafd2ebc598223dc` | 8,100 |
| `docs/reproducibility.md` | `398b2859f997837d17faaa774cc31aa3e82fea90b2f6b0062e3ac93c0d0eb175` | 41,958 |

## Library — hand/

| File | SHA-256 | Bytes |
|---|---|---:|
| `hand/Datasets/__init__.py` | `0feeb664b42b08e4ec6cfcdf7af2006243a4403b66c7581c0926bff2e293457b` | 1,810 |
| `hand/Datasets/dataset_formatters/__init__.py` | `0feeb664b42b08e4ec6cfcdf7af2006243a4403b66c7581c0926bff2e293457b` | 1,810 |
| `hand/Datasets/dataset_formatters/ahawp_formatter.py` | `6c760fac75eeafa9aeae106cdfa77a684a4ecc1fcf29a03fd2135f16f2daa3e0` | 18,271 |
| `hand/Datasets/dataset_formatters/bentham_formatter.py` | `50332968b1d7fc569b2a5e559953c1070fdd42ca8f3dbfe9fc3e2a517078eb61` | 16,034 |
| `hand/Datasets/dataset_formatters/generic_dataset_formatter.py` | `1dc5d1472ec1302cb8e4def83fa4010629d7c7682f243980ac7bb203f2315f1e` | 7,329 |
| `hand/Datasets/dataset_formatters/iam_aachen_formatter.py` | `765f4bd0c09a3d634bbd55ad4137f99f352ade4b77a9c0262c5dce6a1ce39338` | 16,573 |
| `hand/Datasets/dataset_formatters/iam_formatter.py` | `545c1bf7b49df5007a355fba50c19c5c1dd0cca5065246eb9afa60682e072471` | 31,280 |
| `hand/Datasets/dataset_formatters/iam_histdb_formatter.py` | `e18d5bee3505ebf3e28d8b8ad5cd791adfbcd556f06e1ab908355f5c9a505eb4` | 16,157 |
| `hand/Datasets/dataset_formatters/khatt_formatter.py` | `68124b0a19c10b214c38d99eef4bfbe8897fa2dcbf77679f0213e457a947456e` | 16,135 |
| `hand/Datasets/dataset_formatters/maurdor_formatter.py` | `7e8cce7180c72718510e5373f6323a4267a46fa79997595ec110773d20faa135` | 30,728 |
| `hand/Datasets/dataset_formatters/read2016_formatter.py` | `90bb01cb30b5bcaa522943e86815d516ff4dab66fc3cfee36f767fb8cfa5573c` | 39,600 |
| `hand/Datasets/dataset_formatters/rimes_formatter.py` | `33ba08410132df0c6c39d92a2c0082a53344939e3005b8fdef6f428a475ee514` | 18,008 |
| `hand/Datasets/dataset_formatters/utils_dataset.py` | `92f5e582cd84960b57e8518d398d5242d5fefc935ba7399e65e78bcc8701fd9b` | 2,307 |
| `hand/OCR/__init__.py` | `1fb420fb97268c8de8d7833844b357cf68c6b72ccc24c1347793b774f71b123f` | 18 |
| `hand/OCR/document_OCR/__init__.py` | `225e50f404dd5fcc97a95c6d3ff9d8a94844c73c4f0bfaa4fd65d0ff88747b18` | 16 |
| `hand/OCR/document_OCR/hand/__init__.py` | `a0b2905e90214235d2b456240fff2556a90944254307c4f2c0effe103a21b097` | 17 |
| `hand/OCR/document_OCR/hand/main_hand.py` | `d0186cc6268d6761d423300981c3ba64a5c583a4a28adaebb33c50d482e3d320` | 13,312 |
| `hand/OCR/document_OCR/hand/main_std_hand.py` | `162cdf9b0228c8cde900a073bcf079a1e250b93a21b3e01cf4b3f9686a6742dd` | 10,491 |
| `hand/OCR/document_OCR/hand/metrics.py` | `196c75b049d390bbfcaf3352428f39df61b01e9a98755df5b68fab57c1064e77` | 7,228 |
| `hand/OCR/document_OCR/hand/trainer_hand.py` | `d2b7a51b903ad24ba3db67738bae2a6c1b694e3c1a292721056f5e8f6cb6b083` | 8,611 |
| `hand/OCR/document_OCR/hand/trainer_std_hand.py` | `6fe1ecbb99782cc0cac9b9e6665533172cc4b44fefdc7162e4b9b8ee11644f2a` | 29,104 |
| `hand/OCR/line_OCR/__init__.py` | `225e50f404dd5fcc97a95c6d3ff9d8a94844c73c4f0bfaa4fd65d0ff88747b18` | 16 |
| `hand/OCR/line_OCR/ctc/__init__.py` | `a0b2905e90214235d2b456240fff2556a90944254307c4f2c0effe103a21b097` | 17 |
| `hand/OCR/line_OCR/ctc/main_line_ctc_syn.py` | `9a8d03c772b7a3bd3e92be9ca761793ab13a3e37c3422a9f65bea681027f59d4` | 7,151 |
| `hand/OCR/line_OCR/ctc/main_syn_line.py` | `484d8292b5c924246061598440e005269b757a9028381676942a53b16b78e4d5` | 8,857 |
| `hand/OCR/line_OCR/ctc/models_line_ctc.py` | `d825fdcf27b0275992728b04eeb5253862472e89de832cc4eb88863d3cc930d3` | 2,690 |
| `hand/OCR/line_OCR/ctc/trainer_line_ctc.py` | `661d5027af9aa6abbe3cfa1a6e884610adf1fb77cc6c8e8b61f7c7024038c6cb` | 5,293 |
| `hand/OCR/ocr_dataset_manager.py` | `514d6ed27fb1c37ea2e91673f9d750f23345e0e7d669e80bbfa609f1842b7b63` | 85,108 |
| `hand/OCR/ocr_manager.py` | `a0420e7eabb63304aff388e26d07a7b54a82fb3c2d76b1575161cd815e8a3a09` | 5,866 |
| `hand/OCR/ocr_utils.py` | `4544bbf157e27aa17ce99d7099448476c3143a654adc869c1aa84968574b8a7f` | 2,696 |
| `hand/__init__.py` | `a2f82e3f8c334040cd82e733c2266784c9b875145572f9f3987adf42d95108d4` | 15 |
| `hand/basic/__init__.py` | `0feeb664b42b08e4ec6cfcdf7af2006243a4403b66c7581c0926bff2e293457b` | 1,810 |
| `hand/basic/generic_dataset_manager.py` | `913bb39ffbcc6552b9ded10eea7eb4851f30077c3fa98ab38b5f769348c1536e` | 19,460 |
| `hand/basic/generic_training_manager.py` | `9fdcfbc45f33a4de688a4d842fb384211ecad2be77723c0301876f3eb522677f` | 38,898 |
| `hand/basic/metric_manager.py` | `d99ed6ebf2a1a7e3bba2136f4eeea289b17cc5fb2d03675b81d424e91642011f` | 22,991 |
| `hand/basic/post_pocessing_layout.py` | `2080640e80767b29b621ea5e4c689ae46e866ab09c5d4270f117be3d91473981` | 10,820 |
| `hand/basic/scheduler.py` | `17948748876b6d934bf9395b8ab921cc6c05ea59e004d5ae2d0f60a5678f5877` | 3,359 |
| `hand/basic/t5_post_processing.py` | `073a420ed466d3ed947bf5f9f56b29d0a986e008864d3f05bba2f64dea5c2ac9` | 17,214 |
| `hand/basic/transforms.py` | `f023352f5ae3e284742e761485d04f6b05d6759b93a7ad93fa7dc6d3713d610b` | 17,180 |
| `hand/basic/utils.py` | `f3e830eaeeff09eed28c9cc0c874d6485fab184ad4f4e50cbb8c2d56dce35376` | 8,726 |
| `hand/models/__init__.py` | `1fee96d97219c96c95e75ad7a05083d7310fac6bd3fb1ea482d6a50dc94f9f84` | 365 |
| `hand/models/baseline/__init__.py` | `386e45a99523392bb898de03ada1f7b3964683c796490b9e9316fca573258053` | 480 |
| `hand/models/baseline/attention.py` | `68d726cbf5c4603d12d7225d1227f6d9b8d37876cafb2969cfd2d6afa7787287` | 20,549 |
| `hand/models/baseline/dan_decoder.py` | `576b42ff13fc5ea40ace1c7ccd70e7dfe7328c6df1b75dcd77d7a4802a9d7660` | 16,697 |
| `hand/models/baseline/fcn_encoder.py` | `6010644e1a1549a76793c36f2dfc5e660e0fafd9aed3de01e30ed6ca1450b086` | 7,228 |
| `hand/models/baseline/spec_heads.py` | `175bd247256898b528e0546681bbac181c982c81f00a1132cad1b1789e1e2b17` | 2,208 |
| `hand/models/experimental/README.md` | `5eb7f834b1c2ce87526738a116a368ead35e8b0701199a52ab91327fa24bbd96` | 2,746 |
| `hand/models/experimental/__init__.py` | `2f43c152ad202be91fa45fb99f0af19209cdb51e41a402f9c2f41f59aeb7d132` | 296 |
| `hand/models/experimental/advanced_decoder.py` | `58651ab69fe4e8f914cf3537473c1be0b071a15da0dbf71cf20f151c99525ebe` | 8,312 |
| `hand/models/experimental/complete_hand_model.py` | `7da74351c39512a5daa3150156a296fceb22097039e564fa596717a8db8640c7` | 11,121 |
| `hand/models/experimental/compute_metrics.py` | `a18dd0e410b0f64343e8871cffebd3e75815e878bcfff6d93eb1c2a4d133dbe4` | 11,781 |
| `hand/models/experimental/encoder_components.py` | `dcd268aa6f151af0acccfe7e2b7cd6e1ae98bd2b2beaf1776eccc495ab86bfea` | 11,999 |
| `hand/models/experimental/hand_decoder.py` | `94881a8965090be9a1090331165487d1433871f2d620f1ef0a01c01613ebf807` | 10,891 |
| `hand/models/experimental/hand_encoder.py` | `7267bab9f085c4497a2e4aac03b9375d1dbe7335c4d91845deacfee61cb10700` | 11,182 |
| `hand/models/experimental/memory_attention.py` | `0853b2e23c66327b0cd6b2d39ed5639016e7502cbeadcde149c0095cd58256d8` | 6,914 |
| `hand/models/experimental/msap.py` | `a9bf8c970de50bbfaecb28689bb0ad9abe764842511b5bffd806787000b0b0b6` | 15,028 |
| `hand/models/experimental/train_curriculum.py` | `9be01e0f08912d3cd2c92ee8645b5020d96ddf3484232dfe24e9e2aa00b55695` | 15,248 |
| `hand/models/experimental/train_multigpu.py` | `c6a68d7eea686d7b5f1a071e354c30bed4e54762fc62de3684eb7c39c5f3c1f2` | 23,719 |
| `hand/models/experimental/trainer_advanced.py` | `ef57ec108eb4ded3cb7cb50f0ce57eeeca3691a724e5eefcbe363e2b325290c5` | 21,077 |

## Library — hand_v2/

| File | SHA-256 | Bytes |
|---|---|---:|
| `hand_v2/README.md` | `2ea3c417a380cc82d27a924b5341c664a153110dbdbae925406993088537858b` | 1,663 |
| `hand_v2/__init__.py` | `c9c17566620c8d174b49052f1ea8de0c21924673f2c58944c69ba4b235ecdb97` | 587 |
| `hand_v2/data/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| `hand_v2/data/bucketing.py` | `88bd1ec6a1a66eb96e1b00a4a9fd5caf146c07c0babd267e6b6d3681eeab9672` | 3,063 |
| `hand_v2/data/format_new_datasets.py` | `72b44acfb8f814e51605e48c066b132490ed87483c7756865a835dc16c051b59` | 4,733 |
| `hand_v2/data/format_read_dan_splits.py` | `208a0dfa9a5062977b485558a7959a1e6dbbd1b2e9d62507a62103e312aad724` | 6,032 |
| `hand_v2/data/format_rimes_local.py` | `20a5a2a37c4eb0502e0fb2750872bc9e9c2745760f963d44316768c27f8b1749` | 7,528 |
| `hand_v2/data/manifests.py` | `2a0bdd17515c6363bdd778d3f2119e1b57ed7b6e6068854c2830db0ff2a7d53b` | 15,655 |
| `hand_v2/eval/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| `hand_v2/eval/dump_predictions.py` | `8617aa1376ab7ef90f58e5348a3b1c870beb0a9cd86ece7890ec043fc78ec683` | 4,999 |
| `hand_v2/eval/recompute_layout_metrics.py` | `7d539b06f03d1536cbf376fb5675ace339d43eff8d9d180f074f272aadade8bb` | 9,898 |
| `hand_v2/metrics/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| `hand_v2/metrics/layout_metrics.py` | `617763e271075fc133c02801aa575bcd2b93f087da7db3f1fd2ac7bf98bc17b9` | 12,788 |
| `hand_v2/models/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| `hand_v2/models/dancer_encoder.py` | `b1431871e9abe8ed22995937f46bdfa6531402e7fd437dcdb75ff7af845cb400` | 13,936 |
| `hand_v2/tests/test_charset_tokens.py` | `9e6f9f4183bc5f90bd1a3273024a9f9b2a6f33b327d78e6ad5e8665509738fc8` | 6,680 |
| `hand_v2/tests/test_dancer_encoder.py` | `4fb45d9c1bde6ac6407ee9abfe0155e90cf2fdba1838e1c4c18adffb52c3e1ae` | 6,571 |
| `hand_v2/tests/test_fast_decode_paths.py` | `66fee6be30b3d6c83922e71ff03c2ee8180a076ad413d46153426ca487c91581` | 7,991 |
| `hand_v2/tests/test_layout_metrics.py` | `ff9620172beb9dd26d42dd331269cfafaa8aad762d3097dd6d8fafa88bd0547b` | 6,737 |
| `hand_v2/tests/test_layout_metrics_vs_dan.py` | `8962a191b29a11781478e68a6b37beb773190d049c4ca0f61a9695cd669c1135` | 3,661 |
| `hand_v2/tests/test_sync_free_equivalence.py` | `219a7c297e7e3a150f044dc6ef0c62bd51a0eedd3b814d7fbcd93e1878240615` | 8,457 |
| `hand_v2/train.py` | `9e07e3a45832ae2276ead0d028a8bb9c8abdd71b6de8ab14e25ee46a2fea8595` | 10,187 |

## Entry points — tools/

| File | SHA-256 | Bytes |
|---|---|---:|
| `tools/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| `tools/audit_corpus_text_overlap.py` | `6ccd2bfee49c62ca8eb368f97549987e3f67db09ed11c8b564221d880d2f5c2c` | 5,991 |
| `tools/audit_dataset_integrity.py` | `44c80fbb5d546eb3866ba3c058d26b2c32a2e244f07fa058ce5c70521c8107b0` | 13,247 |
| `tools/audit_khatt_fixed_text.py` | `c23aed9eb9b6114bca77b918a411b07fc5bde25e2a487ca21f06961e80988c93` | 11,103 |
| `tools/build_read_triple_page_dan_manifest.py` | `dbc39133eeb906d3242171848abe1af2d7f92a38cd341da968a1604e0423198c` | 4,305 |
| `tools/decoding_animation.py` | `62ece5837e7ce167dc14d2ead7eabf4cbc31898d679de788dea160623d578ef5` | 27,426 |
| `tools/efficiency_bench.py` | `668be3fca4d0320cffc901568eff0e43b2df24271af5fa6f5f1a083baea12c92` | 17,190 |
| `tools/efficiency_combined.py` | `744727655973203b6cc188458f094968155d2679f3c36be41f2203a4c6ed5033` | 14,912 |
| `tools/evaluate_hand.py` | `7d599f859d3fe7cd5e7d0e170c0f8968c1f14617a916867ea5e1fcfa389a4e75` | 18,980 |
| `tools/graph_decode.py` | `1850cfd43c5441f27a86e006d9ed730e5e26031edf15baa94ec9b1c66955f8ba` | 14,201 |
| `tools/make_tables.py` | `f9ef45b8dcb984d21fa2a967095e96cea6c271be0661416e8285b9fe39c010ed` | 21,893 |
| `tools/multipage_eval.py` | `79907aecbf500b0e1ef0ef14799f02f557b3b862a6613b1d58fa8a06ab52e58e` | 25,075 |
| `tools/multipage_summary.py` | `79f6cfb01a0a26d0c8d19251e25cffce3256665bc08981987b0175190a85ceda` | 11,876 |
| `tools/qualitative_figures.py` | `ddc84e43aa5ad4e092cdd0b5fea714aa6fdf835e82ee34f2e46cb7c541d0fa4f` | 32,380 |
| `tools/qualitative_iam_khatt.py` | `06637cf93f5227ad601b78fa511903406aaa70a81c9772f1a3b684a4a4adf6dd` | 13,836 |
| `tools/qualitative_khatt.py` | `a868fd0156bed52bcbb395f24c4880a07e59e8bc1746a1856f10309a8b80284b` | 24,120 |
| `tools/qualitative_multipage.py` | `46bc16f709a56500dfb30285853cae80ae7b47e9a86be965e78a41eee83d0d0b` | 17,315 |
| `tools/reproduce_baseline.py` | `db6af53db7915eeffbebc2c550820101a1e0c4d0c3239527855d250f9238e65e` | 10,396 |
| `tools/seed_variance_analysis.py` | `ca2fc903eaf1e67a75e7d19b5c8056f0460f5ad212eb9152abb7e29300acba10` | 11,279 |
| `tools/spec_decode.py` | `206e2d1e16f7c5b18d320ef49a73d3ac4f3d64403da5d7e915bde4f67c772123` | 13,136 |
| `tools/train_hand.py` | `ee1db95f8d3d5c0ba7b35556c67f2ec285c233cb21af5e216b2105b7daaa4944` | 22,370 |
| `tools/train_spec_heads.py` | `610080dc6dc5634d950858263bd9f937ed99783e3543d79555135bd57f3889e2` | 7,285 |
| `tools/validate_install_cpu.py` | `0be2744cf38b10bdf1cd919425be73cc75c26787b902430bad0e26a97215c4e4` | 23,637 |
| `tools/verify_exact_decoding.py` | `7772b81305112d27f749c7f084e33b46c24fb815a383ad6476808e0ec64276a8` | 5,066 |

## Reproducibility scripts — scripts/

| File | SHA-256 | Bytes |
|---|---|---:|
| `scripts/setup_dan_fonts.sh` | `d08a7b2157204271e649d815188341731e6f60add22473bb769eba6db2069a6e` | 3,703 |
| `scripts/setup_dataset_links.sh` | `bf5e16b76f63490417f74634abfac2e306078918b284fceb2adbcd508eb89507` | 3,687 |
| `scripts/verify_dan_fonts.py` | `98aafe6ab7bb3cf5889b9cf0a8b9d5bb244bf384f5a6c550e8101aa141cb402d` | 4,903 |

## Configuration — configs/

| File | SHA-256 | Bytes |
|---|---|---:|
| `configs/README.md` | `1277153781a28b7766a347b1e0bca860b9caf6ac542f80d4bdddb90f39e6093c` | 1,491 |
| `configs/datasets_registry.yaml` | `a4e496d05ed98f8e0415f0b2d2bf4b29df97e6906acd8cdbb023d1352b86d8ac` | 9,117 |

## Release payload — licences, notices, inference contract, model configuration

| File | SHA-256 | Bytes |
|---|---|---:|
| `release/NOTICE.md` | `ac5fa42a3466993bc1ea58bd84f75d4012c9ed3dffa05a9a2744e3840ebc4a3b` | 17,326 |
| `release/PARITY_CPU.json` | `db5a3f043b181efac0e0ec758822245671d09ccf4492f2219ade9bfa0c10bf54` | 10,108 |
| `release/hand-read2016-page/.gitattributes` | `7e63903f3514a8ffc7463e309944bc0a70f1b6dc21f3741773b51915e490aad5` | 150 |
| `release/hand-read2016-page/charset.json` | `f54b3f596418572315e72e262cf57e3088ee429e36ea532db1fdf8bbf0a21af9` | 1,240 |
| `release/hand-read2016-page/config.json` | `c7e554cd5d5db02aa0633917f8d3bd3752d52d431c1cc4286c1571b043479531` | 1,353 |
| `release/hand-read2016-page/preprocessor_config.json` | `cb9153533612ae17492474171e160e26e5ef455600d8d85ac959980730150d01` | 764 |
| `release/hand-read2016-page/spec_heads_m5.json` | `691ccac0c761740c4c74507087957fc0211178ce4778887f6c2b79de6c78ae70` | 355 |
| `release/hand_release/__init__.py` | `7e4768d9280b68a0768adb95ca7aa52dbff488a2224d7eadfb2815b856f9b447` | 449 |
| `release/hand_release/inference.py` | `1eaf04a7f3d83c79a13a95029568079a73725c0877a00a02d09d2fbf8ed115eb` | 28,382 |
| `release/licenses/LICENSE-CeCILL-C.md` | `405e0890e5997f766bfe0adcfad381077749b6c615d41dc4effb0baf271ada9f` | 21,958 |
| `release/licenses/LICENSE-MIT.txt` | `c1eff5cff0bd189a8f69dc0e262dcb7cf2b8b1bd639dfa14fe52162d995a707f` | 1,075 |
| `release/tools/evaluate_release.py` | `5227bcfec08748dc38ee84a68a131c0295a6e8fbd92c7ede99ce493862c2e3f7` | 10,750 |
| `release/tools/export_release_checkpoint.py` | `ced4d7a010d16ea3b283cf64939147d4ac2df8131a3fe5640a06bbb413e802c1` | 11,300 |
| `release/tools/generate_manifest.py` | `a138aed15736c2fad42fdc8a3d67438587552acd234827ab9335aba4b038a538` | 5,175 |

## Reproducibility artefacts — run records, profiling, split manifests

| File | SHA-256 | Bytes |
|---|---|---:|
| `experiments/README.md` | `c1ee79adfeb4ffb9f1fb0b880ac1d02f357d25bbf714a56335ace192db2621cf` | 3,116 |
| `experiments/TEST_ACCESS_LOG.md` | `c690d6f131db351d284647a8cebcc5082c153c4efe5ad43e7e711afa31f54c99` | 3,196 |
| `experiments/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| `experiments/benchmark_suite/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| `experiments/benchmark_suite/manifests/IAM_page_aachen.json` | `65d4c99cd9a0d8f51c47a3d611410cf7c35555b8b35c79171f818d0148db6b2a` | 287,410 |
| `experiments/benchmark_suite/manifests/READ_2016_double_page_sem_dan.json` | `87507744bab31bf1e67374a69b854095573db4eb2999c9c4e6983514375b6e67` | 110,751 |
| `experiments/benchmark_suite/manifests/READ_2016_page_sem_dan.json` | `f86ef07305ed5d43b6d15901222839b5f61c592e26bc0ba4078bf90098c71f30` | 174,723 |
| `experiments/benchmark_suite/manifests/READ_2016_paragraph_dan.json` | `aaa2e7e1f5d00c24e7f4d93a83f76853be8a93a8f186c80ada2b5e4bc0185ee0` | 825,406 |
| `experiments/benchmark_suite/manifests/READ_2016_triple_page_sem.json` | `7356379c1c0062ea3b6ce1b4ff7d099f9cca1da3529b6e36eaa8fc2b24fed0f8` | 87,833 |
| `experiments/benchmark_suite/manifests/READ_2016_triple_page_sem_dan.json` | `0c60c434a17c005b0dee86e703f5517446dfd1c5011a71c975de9896123da15a` | 101,208 |
| `experiments/benchmark_suite/profiling/anchor_dan_official.json` | `2de9eb3155b2070c583373c7dc1b6352685870d1a754d6547d0641fe75c0ecac` | 1,288 |
| `experiments/benchmark_suite/profiling/b3a_vs_anchor_valid.json` | `e46b7d18bebf2f3da99dfbdb591c64b89490f3e1700d9a379cafbae4668eb519` | 12,696 |
| `experiments/benchmark_suite/profiling/e4_spec_decode_test.json` | `cb4c2effedb5e11d378d92fd4b01ea88f8cf9e88ffb61d1e1e06f831fbbda82a` | 336,124 |
| `experiments/benchmark_suite/profiling/e4_spec_decode_test_fp32.json` | `b70629459e89bd87d52a3dc4e5ce71ad05eae61af225a57b7aef878ce6269f37` | 336,101 |
| `experiments/benchmark_suite/profiling/e4_vs_anchor_efficiency.json` | `dfd9ca018bf9aba32c01b34f9b964f993f69b4267832a5eeef1e3a2ec85fa4f1` | 65,288 |
| `experiments/benchmark_suite/profiling/e4_vs_anchor_test.json` | `c3a8ecc47ae02ec4700b5e33aa95689c943da2a9fc77fc57478bcf4fd4ed7c32` | 12,469 |
| `experiments/benchmark_suite/profiling/e4_vs_anchor_valid.json` | `3941dc90cf21c40c35795a8a17c1e3ce06a4908c32b9d87f8985a5285b7517a4` | 12,660 |
| `experiments/benchmark_suite/profiling/e5_vs_anchor_valid.json` | `56d96dae8e3a15ff1e9b972e1fc38d78a40547f00bb0cb70951dfad0c44e3abf` | 12,656 |
| `experiments/benchmark_suite/profiling/efficiency_combined_test.json` | `35f799daabf9b148317bf22107f88dac27780434c8a7c93dc5a5f745d0042e2c` | 25,190 |
| `experiments/benchmark_suite/profiling/efficiency_hand_vs_dan.json` | `b789cc3e1ce6dc8a6d81376ba2433f8a4444bd14a93e0f26fafb2a45b2ef81ea` | 66,452 |
| `experiments/benchmark_suite/profiling/efficiency_hand_vs_dan_uncontended.json` | `6f80d7dccbba34ef1d585f3f11576ce29a47ccc86b0e7ab2591b775dbe4d6842` | 66,534 |
| `experiments/benchmark_suite/profiling/exact_decoding_equivalence.json` | `9d43ec1c73fa2e6c22d5f6732ee72a77d45ac8993b61bf88f1fac3ae9175f0a9` | 895 |
| `experiments/benchmark_suite/profiling/seed_variance_test.json` | `a85d08c297052c80058c2fca03ed58e07e648367623e234ec04fe79bc307b2b1` | 34,953 |
| `experiments/benchmark_suite/profiling/seed_variance_valid.json` | `850b8407cf43347f67892098be1ee14de4d8799e172d9c59435d64daf53b46ec` | 25,979 |
| `experiments/benchmark_suite/profiling/spec_decode_test.json` | `ba5b8c876075d8480f1d797dd18a5ff15e40346bdf71ec02e9c84eaba27c4cda` | 338,510 |
| `experiments/benchmark_suite/profiling/spec_decode_test_fp32.json` | `f7c514275e8040d1723f10b6426ffeef18270f7dc7e02c8f453d78a8e2e9df02` | 162,069 |
| `experiments/benchmark_suite/profiling/spec_heads_e4_m5_history.json` | `e5aa0bf075b8b284ad3854295618a13c9846c05d72a5018a95a2fb8d80bf8df3` | 86,048 |
| `experiments/benchmark_suite/profiling/spec_heads_m5_history.json` | `bd3d7fd6f622998e253591844afa1c81360dd2b1ba51d1acfa1f6896a370709e` | 86,238 |
| `experiments/benchmark_suite/record.py` | `48b536eb419eb289416b6290e7da038aec9874ab7253522278188adda4db61f1` | 8,947 |
| `experiments/benchmark_suite/registry/20260911T032022Z_s1_A1fixedR1_s0_b09a58.json` | `9efac58f0712fd2a88791884d7635322fbedcc0438d98194338b25771f07fa0d` | 6,223 |
| `experiments/benchmark_suite/registry/20260917T160949Z_e14_budget_1p26M_s0_bd9339.json` | `12c7de24eb1142cf505ce3cf5063709bc4ba6ba904f3a83e1e7c7abec260ec00` | 4,897 |
| `experiments/benchmark_suite/registry/20260917T160951Z_s1_A1fixedR1_s1_af4532.json` | `ac2c1884b33ba4b3a8bc7dff591368254d76c1284275db23445297b80027d9e7` | 4,718 |
| `experiments/benchmark_suite/registry/20260919T101946Z_s1_A1fixedR1_s2_1baf26.json` | `d33ee01b3b4f5fe0e45b3851b5a76acea9c0cab80cdba1ce61f3e9595d32daf4` | 4,964 |
| `experiments/benchmark_suite/registry/20260924T030209Z_b1_budget_cont_s0_1485fa.json` | `723afec5a5d4d1a4650a9686f49f9c261097287041c7a9cf0e136d1dc2bdc857` | 26,857 |
| `experiments/benchmark_suite/registry/20260924T135940Z_e4_sharekv_s0_4c50d4.json` | `d86c85bb53f09b3e40bf9fc23f729f209f9a24c9a32fb1f2570a4b5b9e41b7f6` | 51,823 |
| `experiments/benchmark_suite/registry/20260924T135940Z_e5_depth6_s0_e554ad.json` | `1df7d71fac9aad84e0633d35b81aa81a436619c7abb2182084c69588e0a93b2b` | 51,823 |
| `experiments/benchmark_suite/registry/20260924T150342Z_b3a_labelsmooth_s0_a61f5e.json` | `5e842ff1070fe85237933e6cfab24fffa029c26cb252a23dcad65ad0067a609e` | 52,271 |
| `experiments/benchmark_suite/registry/20260925T201240Z_e4_sharekv_s0_d0408c.json` | `e0a415dd2e86897009283cf1b5b48f4b6e67e0afdd0c9734e355df43a58e6ff5` | 78,898 |
| `experiments/benchmark_suite/registry/20260925T201240Z_e5_depth6_s0_3583cf.json` | `3149c22ece5e75785244bf7ef3c0af2986d609fcd87345441187d755dfcfbc25` | 78,897 |
| `experiments/benchmark_suite/registry/20260925T211803Z_b3a_labelsmooth_s0_efff0c.json` | `da925cb159cd8f1376e5169dc62e1d2641ad8c6c1fd450258c3715e14c5a069d` | 79,399 |
| `experiments/benchmark_suite/registry/20261002T031036Z_ft_double_page_smoke_d6d930.json` | `9fafc76c9d0b3a35d5a09d9e08e1358f27b5b8f4a08023b7b9352b3c0b7bbc0c` | 15,670 |
| `experiments/benchmark_suite/registry/20261002T031727Z_ft_double_page_from_e14_s0_f44211.json` | `90b1c77f4f03f4631566b1f09138d899f7c8f2de07526a399e435f177d90a23d` | 28,194 |
| `experiments/benchmark_suite/registry/20261002T072130Z_ft_triple_page_from_double_s0_f29b66.json` | `3ef001fc2bdcb83466b972858c2319d9e43f6721c3ea3710dd69a3aecae99ce5` | 30,779 |
| `experiments/multipage/README.md` | `48c8210420e6629028dd8310b85bf0e3f8ccf41ea9e9fbe86505373fc721dc79` | 12,360 |
| `experiments/multipage/adaptation/FT_DOUBLE_E3_double_page.json` | `a35a52baa4f55dc6ddb1d451ce0cabed84429ada3022a78603842ae4df4cf891` | 50,690 |
| `experiments/multipage/adaptation/FT_DOUBLE_E3_page.json` | `fcdc2b536454746f5ea90cfc7f04b1cb7194130ee038a68fc41c9953c4ec3792` | 158,456 |
| `experiments/multipage/adaptation/FT_DOUBLE_E3_triple_page.json` | `20513b56ad5640d6decd4953b29d8eea0af2e56156416dfb0a1ccdf97e589d99` | 36,315 |
| `experiments/multipage/adaptation/FT_DOUBLE_double_page.json` | `52d73dd0d84bcd538884aa476ddfe7e04611ecee54c07d2625dfde52268f70d9` | 50,345 |
| `experiments/multipage/adaptation/FT_DOUBLE_page.json` | `3475771fbd78454d9fc7668505d25bf173890f410f8708860cc5594202b91fe9` | 156,606 |
| `experiments/multipage/adaptation/FT_DOUBLE_triple_page.json` | `9530d885b401e137f540634a326c5ebe7abee323f441fe13667ea8bd4cd222e6` | 35,976 |
| `experiments/multipage/adaptation/FT_TRIPLE_double_page.json` | `5386ee3aaf9715a8364ae4ffa0008f8332c4cbdf995869e6fabe4131cc9fe456` | 171,829 |
| `experiments/multipage/adaptation/FT_TRIPLE_page.json` | `bd67bb7d2a304773109678b58f32e922923d35aff30b3b919c5e8a0e50fa331c` | 215,809 |
| `experiments/multipage/adaptation/FT_TRIPLE_triple_page.json` | `7ad2d31cdfa1cafab98ec21c738c00b3f05f369a466980f7f808f4b5a08b24ed` | 43,205 |
| `experiments/multipage/adaptation/per_sample_FT_DOUBLE_E3_double_page.csv` | `156d97dd99c375b1cccb4f8642492aabd4655a5851756fe7eeb947c205f6b9d9` | 6,626 |
| `experiments/multipage/adaptation/per_sample_FT_DOUBLE_E3_page.csv` | `13895f1cf6384495ddc34fd7a88ef343204ae37b61635c3d1b2b62c2bd06fa76` | 13,219 |
| `experiments/multipage/adaptation/per_sample_FT_DOUBLE_E3_triple_page.csv` | `6d7a1b34888387130a9566f087c989e8bdb6014bf38c2ca8db8945d1545eb619` | 5,189 |
| `experiments/multipage/adaptation/per_sample_FT_DOUBLE_double_page.csv` | `4a12f9c6841fa583f54780c3a2301f41c6299c69a0ac755a9288a661e4ffb287` | 6,624 |
| `experiments/multipage/adaptation/per_sample_FT_DOUBLE_page.csv` | `5a6bf1fce35c247e144a8cf68c804570d52b70084a4d986cdb65030c9e7f4f0b` | 12,886 |
| `experiments/multipage/adaptation/per_sample_FT_DOUBLE_triple_page.csv` | `abf1ac104c1e523e13b7cb6e6887798c7b6116f00a386db2293d2a4909ba4603` | 5,217 |
| `experiments/multipage/adaptation/per_sample_FT_TRIPLE_double_page.csv` | `4c3c96befef6b4c5d72b380e14326726636acb6fdfdbf0619cd893869b96eec8` | 6,789 |
| `experiments/multipage/adaptation/per_sample_FT_TRIPLE_page.csv` | `73a46319b8cb10840f204d2136ee3df66b4c6443ac977a7ef4cd72548d78f15a` | 12,475 |
| `experiments/multipage/adaptation/per_sample_FT_TRIPLE_triple_page.csv` | `01c3637a42a5a3ef9dca27d1341879ece06f433017281479900b335cced0dbce` | 4,609 |
| `experiments/multipage/adaptation/summary_FT_DOUBLE.json` | `f6d3ba1cce3c2aea994c8531e68e5cf4346b35afe415a618f8f000f418237661` | 4,974 |
| `experiments/multipage/adaptation/summary_FT_DOUBLE_E3.json` | `05e7293bf71a9761debd4ee5f62767ee673a50406de72fb97779d95a0c1be9ad` | 5,115 |
| `experiments/multipage/adaptation/summary_FT_TRIPLE.json` | `e46c2835099d392b1d53c2b8762db81d2349c65fabf72c288b5b538ea7b317ee` | 4,823 |
| `experiments/multipage/run_adaptation.sh` | `05a2a1d87ae25249a911cd547fdce3251e2b69ef57c3df0695ef3bec390e6583` | 3,387 |
| `experiments/multipage/run_adaptation_double.sh` | `cc2bfb142e85d4760bbfc04cc90ac4fa7637454cfc4dae15a213fc8dec4c8acf` | 1,438 |
| `experiments/multipage/run_zero_shot.sh` | `a498026360511bd689d68db543af56966e633e3213c59bb90eaed68eb5949ed5` | 1,269 |
| `experiments/multipage/tables.json` | `4c85b48b61ebeb22483c5b2c1e170b71969f68907da491dc6a7c49b1a0fd5760` | 33,222 |
| `experiments/multipage/tables.md` | `6d43001bb3cdcfd914ab5bda2dd3f8951f7b536f86563380c192f3548b2c934a` | 7,893 |
| `experiments/multipage/v1/_summary.json` | `5c9c7a6edd36002bc5b4413d4f77767ab5cb719a8a21af0b386bb4ba7db26215` | 1,648 |
| `experiments/multipage/v1/read_double_page.json` | `8eb6ad13f1502670c9e468b74e34cd45c36ee0e9b317a3048bdaca8f5074a0f9` | 741 |
| `experiments/multipage/v1/read_triple_page.json` | `5bbda0683a99e749ba4bec118c0361502e47a8e69b24826c072d69925756f8ad` | 741 |
| `experiments/multipage/zero_shot/BASE_FORCED_double_page.json` | `484a1f90ec6ea964b9cc2364a9362930d226b85a935d47e5e84bbb7d5dd3117f` | 38,326 |
| `experiments/multipage/zero_shot/BASE_FORCED_page.json` | `fbda653566bd81811c87fb52589cd41b6271dec22fcd6d199d67a99e3c426f1c` | 72,098 |
| `experiments/multipage/zero_shot/BASE_FORCED_triple_page.json` | `892ced2da99e7bdd4697455e76612c21fbf7a261ac4748341981b7df65e72acf` | 27,157 |
| `experiments/multipage/zero_shot/BASE_double_page.json` | `41015f1f58d31939e2ee93fc462ddccf4d4e478653ba90bce0c1818aff1e5929` | 36,233 |
| `experiments/multipage/zero_shot/BASE_page.json` | `8d17c15904a488a7ea29c8bbca1f5a8d64eb8b83868134fc593429e28d894b34` | 70,497 |
| `experiments/multipage/zero_shot/BASE_page_fp32.json` | `03d49a45b3646b055fd74fe3a44c356b3502ab59ed7982d3f0514c89f7527043` | 70,483 |
| `experiments/multipage/zero_shot/BASE_triple_page.json` | `8db25b893b6cd206ae4349364758379d38a8a1b1b89945c9822c342cc31d02e6` | 25,333 |
| `experiments/multipage/zero_shot/E3E4_double_page.json` | `a3f6f40df076402a31ac34ed440952fc300db7e788bc4468ca61da72c801263d` | 36,618 |
| `experiments/multipage/zero_shot/E3E4_page.json` | `ea6f1c2180bb898dbef4a70712ffbb007cc3e26316afacb5760e2db092d557ea` | 70,911 |
| `experiments/multipage/zero_shot/E3E4_triple_page.json` | `bd5010d5488be37238b578a28be576b0c8ca20a65442165593071e6029900f73` | 25,742 |
| `experiments/multipage/zero_shot/E3_double_page.json` | `3084235e2911c1b5c550fe146422288cd4ceb4878ff7da741a3c3df843b658d8` | 36,500 |
| `experiments/multipage/zero_shot/E3_page.json` | `c03d317c076967abd255cb2eae26168063e2dc432d6eb7083f4970ae56be978f` | 70,785 |
| `experiments/multipage/zero_shot/E3_triple_page.json` | `084fb4b4fef79f60ba45c31906bfbf40314c66b7365bb5fe8d24789495fadf89` | 25,583 |
| `experiments/multipage/zero_shot/E4_double_page.json` | `99b039e4fba3bb47c7d99e8aa1a149980d1595a615d90fde3c6f5369e33e4bef` | 36,334 |
| `experiments/multipage/zero_shot/E4_page.json` | `47940e241527f085853e3eb481615ad9995bbcd3d2c4411e097f4c4a7f61e002` | 70,673 |
| `experiments/multipage/zero_shot/E4_triple_page.json` | `b6c37acbf152cc42e2d0db07ba92db7c0a64f606fc703d2e613856e08b49c70a` | 25,439 |
| `experiments/multipage/zero_shot/per_sample_BASE_FORCED_double_page.csv` | `2d4e7e2be2864037f59c5e9e88bcb3252c5fdce01377be28b1aa95fc4a18cd44` | 6,676 |
| `experiments/multipage/zero_shot/per_sample_BASE_FORCED_page.csv` | `8b954238b2e6ad3d5d13e811c0879cd91fa72ec83129d3636621872f5401b310` | 12,097 |
| `experiments/multipage/zero_shot/per_sample_BASE_FORCED_triple_page.csv` | `bbdcae0a85299f7134c043063664bf24d9c2b962053451bd4265b05daff28512` | 4,619 |
| `experiments/multipage/zero_shot/per_sample_BASE_double_page.csv` | `22474fedb9374f72a9ecde62a4bdc6c164fef0a7d0d841312312b32cc8f3810d` | 7,426 |
| `experiments/multipage/zero_shot/per_sample_BASE_page.csv` | `745f5c5252bfc4bcb583528e746a4fb7b2cbee85aeb745831b365302e6dec828` | 11,988 |
| `experiments/multipage/zero_shot/per_sample_BASE_page_fp32.csv` | `51e0526e8b30b295b61c4b61477ef6b3e422c36da63636cb2f52bb5db40046e5` | 12,006 |
| `experiments/multipage/zero_shot/per_sample_BASE_triple_page.csv` | `a806b9f5c19ee5016463377b4cb33a7d071eda73cd1c168fa3e5b0b909fc2c40` | 5,095 |
| `experiments/multipage/zero_shot/per_sample_E3E4_double_page.csv` | `8cbaf146626d55af0bf4198d58680f023733c46b2f9bdb9b1004fafdc8fab114` | 7,404 |
| `experiments/multipage/zero_shot/per_sample_E3E4_page.csv` | `5ce5e3b112d4bf16468b47125d2ad2d0655db9bdb65997b793ce3a02a8f2156d` | 11,936 |
| `experiments/multipage/zero_shot/per_sample_E3E4_triple_page.csv` | `762d9550797a803a9d76ab147b83bbc5184d9ca512ab48679e8b8ea814d19628` | 5,090 |
| `experiments/multipage/zero_shot/per_sample_E3_double_page.csv` | `571ecb571451f78921620f7fadf4948ba9f095b2af4a1a96347c9114c2231df6` | 7,381 |
| `experiments/multipage/zero_shot/per_sample_E3_page.csv` | `57dcb1c32154621d0ca5225bc7ea5f9bd81711d7939fca511d501f94d21453f6` | 11,951 |
| `experiments/multipage/zero_shot/per_sample_E3_triple_page.csv` | `c4ae131a765fff5ffc26f95b2e6d12fcf84d7284a705a7aab4764e5001bdd126` | 5,033 |
| `experiments/multipage/zero_shot/per_sample_E4_double_page.csv` | `4ad41074c9064f1338594d74b485242ade0839f28584a2edf165d5fea59c23e4` | 7,439 |
| `experiments/multipage/zero_shot/per_sample_E4_page.csv` | `420ee4168fadeba1b514dc0a4507a415bd6bf2530da34718a25f9edc877a3bb0` | 12,025 |
| `experiments/multipage/zero_shot/per_sample_E4_triple_page.csv` | `6074b522370e0a240937931b20fff9aebbe3233b64516c9a49c533c2fb10b73e` | 5,103 |
| `experiments/multipage/zero_shot/summary_BASE.json` | `d5f84a2ff3563a9ca6f08723ccea3f39831b83547219c99855e0297877a8a653` | 4,541 |
| `experiments/multipage/zero_shot/summary_BASE_FORCED.json` | `bd41a6da9cddd972d298647bce5c2d38b8661d9921e671f9c0de6d2c1f4ffe03` | 4,729 |
| `experiments/multipage/zero_shot/summary_BASE_fp32.json` | `124d2289cdeae4e26b40a741f9677b58b099939b0c4feb3c0981d68ca0a6c8a1` | 2,443 |
| `experiments/multipage/zero_shot/summary_E3.json` | `889dd8b482fe6422f01862c55fc0ca9e9293dd396f5755c88f10c02939ed5cf2` | 4,662 |
| `experiments/multipage/zero_shot/summary_E3E4.json` | `1e3bae3d468a74bf289e22c5abdbbb7b9dd4cd4c67b4cd1ef68ab6cfe218f148` | 4,670 |
| `experiments/multipage/zero_shot/summary_E4.json` | `c5e5a83f323d147b81523091c2eae6e73a9f6466ef7267894b99638dbbef7f37` | 4,541 |
| `experiments/p0_3_khatt_l6_audit/corpus_text_overlap.json` | `66d89c992ecbcbd35518ee4528e6b63589c81b31f545c34b61ebc0fd47a2097a` | 10,415 |
| `experiments/qualitative/decoding_test_11_prediction.json` | `20a2c573a1657bf8a569bad14ad97a93c8a41adc82d252cefb463cb0dde41023` | 1,337 |
| `experiments/qualitative/decoding_test_11_spec_steps.json` | `dd43a0517b8a7302f538182340abbf6ee284b315cb126b07eb42212e012c2c79` | 10,901 |
| `experiments/qualitative/khatt/cer_distribution_test.json` | `28b56493b7100236e1b65f536c7ecf101e12f52535a159a74d1cc2dd1f6954a9` | 6,052 |
| `experiments/qualitative/predictions/read2016_double_page_adapted_test_test_23.json` | `e1395f60477689b709cafb51be3ec7470b7dfa7b121da7a7a73cc8de8e443a57` | 3,687 |
| `experiments/qualitative/predictions/read2016_double_page_zeroshot_test_test_0.json` | `1953f48e5f2a1befe0651bf6def1575ed3d3ac3a359ddb510d2225e20937e50b` | 3,628 |
| `experiments/qualitative/predictions/read2016_double_page_zeroshot_vs_adapted_test_test_23.json` | `f11905d477af84d1e07a75352de61cea212cf7bf0ecd5009094d93856a29d60d` | 6,270 |
| `experiments/qualitative/predictions/read2016_page_test_test_10.json` | `b10d4f59f49531e68513111a97c59ea2c21350830df41fcf81b8f92b09eac44c` | 12,713 |
| `experiments/qualitative/predictions/read2016_page_test_test_11.json` | `e8b47330d6b0f14854bf792ae509ad189e5fe5622f4d2a3553986dde5ff3d3d4` | 12,181 |
| `experiments/qualitative/predictions/read2016_page_test_test_13.json` | `c6606f283136590f694040b7b14c5355dbacc9d3fc0fcd0d21f3f55ad9305cbc` | 11,333 |
| `experiments/qualitative/predictions/read2016_page_test_test_25.json` | `41866ed3be32030a05692d42c8b618b0c5355d91b3bf0d69939c7b6919c88342` | 11,593 |
| `experiments/qualitative/predictions/read2016_page_test_test_35.json` | `6887c9876fc58ddddd89880941688a46fd6f5d79bc86cbecf4042655cd61777c` | 12,420 |
| `experiments/qualitative/predictions/read2016_page_test_test_4.json` | `5a299989e7b5cefac5de97dea99255f4a860cb5d10f2f21f71a163b8b7666570` | 13,917 |
| `experiments/qualitative/predictions/read2016_page_test_test_48.json` | `1898c9d571c3578812c4774e12144ef3b7ac8b73d6a0d86a144ca735568fe81b` | 10,422 |
| `experiments/qualitative/predictions/read2016_triple_page_adapted_test_test_5.json` | `303a5abf66bee078005149174c5ab93bf6b2b41bc9b10121a30ffc6e8601df81` | 5,122 |
| `experiments/qualitative/predictions/read2016_triple_page_zeroshot_test_test_0.json` | `8c782d43be087320090fe61d91b97ab8c61769da99bb7cce5c41892de6571dac` | 4,915 |

## Reproducibility artefacts — measured metrics

| File | SHA-256 | Bytes |
|---|---|---:|
| `results_real/_summary.json` | `2b224734902feb5f59a7aa846160ac15fd4cb10522d66f4a2ffd4e63b7078fed` | 802 |
| `results_real/ahawp_paragraph.json` | `808c13605e99f07446af758b72261229535fb0f8eba18f50a76b9cf5bc340b72` | 859 |
| `results_real/iam_page.json` | `30e3b74e94eaf417072aa15b11f79152eb0e83c0a7b0406c0adf4c0fc531d030` | 1,002 |
| `results_real/khatt_paragraph.json` | `c39c7d7f11261a1225f22f5f53e680efca28efd15e9e1f3161545673174f2225` | 1,022 |
| `results_real/read_double_page.json` | `61f088d2756b6190a720a22317ab6eae2db8c54186e50d4bcb82736d7d6284e0` | 745 |
| `results_real/read_double_page_cov2.0.json` | `67c9450dab26748f2e1840c6c2f95e245ae699915fa588c86a16740d24013940` | 747 |
| `results_real/read_double_page_cov4.0.json` | `e9e3b68db16175ae6c80de11f4936a97b7be63b3c5dd98000caeca7db2430d54` | 747 |
| `results_real/read_double_page_cov8.0.json` | `e9889f5009e568635ee6f6d08d8eb82f999f4a461b029067ed29699bfc4221e8` | 746 |
| `results_real/read_page.json` | `3bcfff87da403e9b540c33d59dc44d261ef60c62b54b492aa8fcb31039980cb4` | 724 |
| `results_real/read_page_beam3.json` | `fdf02bfaed804bbc8a7729a1f75156a88596ed175590e6ee980a2401c62981ee` | 1,007 |
| `results_real/read_page_cov2.0.json` | `d354768384834c3f17f14cc96a75caf90b8ec278ddfe0a30f29f01f027fb8ea8` | 723 |
| `results_real/read_page_cov4.0.json` | `a7fa5b99b1e4a7ee4dcfbf4b43a2fc81ce321cea422f23b9c3ddac4d4e48bded` | 724 |
| `results_real/read_page_cov8.0.json` | `f2795a629cc943b9dd800754ef27082d09038d5b478b0eb39bf99e99cedd7401` | 725 |
| `results_real/read_triple_page.json` | `f27c3b0dbf09e531ee6e9192055408eddf2f1c5c3b81fb2aa50da23b0ce9a5a5` | 746 |
| `results_real/read_triple_page_cov2.0.json` | `d993e8e9546c1cd9edf9fbde5cfd269e9b26df6d73dadcd3e5e15835c71ef693` | 745 |
| `results_real/read_triple_page_cov4.0.json` | `eb2822f0a44808b5845fd20ee57c770fea7ef302b4fb470debdf23149c0124a4` | 747 |
| `results_real/read_triple_page_cov8.0.json` | `95484c66da5b805e7a2bcfa422a828fdeed0d459424326b4072f4afc7aeb69de` | 747 |
| `results_real/run_abl_line_fcn.json` | `52cac41eec17b5710b165396f6af422e1056506905267b38c12f65f997a54361` | 623 |
| `results_real/run_pg_control.json` | `383185510afac4dccb3c2ff4a2c8c68d151ad8b148f84f4a5c8d8ccd20e582ce` | 931 |
| `results_real/run_pg_repaired.json` | `d80ad28859a5e13bf7dd02d5e2fc7ce88844fe6aeb1b9b555222037a41f9d625` | 930 |
| `results_real/run_pg_tf04.json` | `efce0f99adfb77af9194740b53fa1c1e2190aed11481c0135d1543a01e7ea8d5` | 926 |
| `results_real/run_pg_tf06.json` | `1396af3de9825eb9b04615320777b0683d0b8d27bf9083e0e0fa979cd79a694d` | 928 |
| `results_real/run_read_page_ft_control.json` | `6a8683720884b9edd81cf6c45ee6e7e867ec006f86ef10c8c48f6088f1e23749` | 941 |
| `results_real/split_integrity.json` | `eeb12fd9f622539ddfc51baef0dc16b570fb9121f86d8b61dc4c2533b160ff93` | 3,970 |
| `results_recovery/_summary.json` | `e6d9710c32010ddfca9cd90580f73f4f61185fb9c47fae19f9ddde33869c68dd` | 10,469 |
| `results_recovery/ahawp_paragraph/metrics.json` | `fe15113f31301a6bfe161b70b802fc92caf2514a2b631998785603d502f43306` | 1,328 |
| `results_recovery/ahawp_paragraph/runtime.json` | `15ded8032a7a3e08f01666843bb30cfbe5ccd21aef73b9a8f3373dfb1f7878d3` | 715 |
| `results_recovery/experiment2/analysis.json` | `9a9917b14a80191bc7feed773fcc385a56b823bbcd8868a4cf7c5ac46878ce67` | 6,171 |
| `results_recovery/iam_page/metrics.json` | `3db0d6f88653b4560da67acd7759b25e874da1aea18dbc9175325ec0adeeafd7` | 1,323 |
| `results_recovery/iam_page/runtime.json` | `f565b245fb390e01fe295d9627a618d7ead2d30680fc93870e200cd91b1586d2` | 712 |
| `results_recovery/khatt_paragraph/metrics.json` | `ff1a7b31b41fdea71e1123d48a5f2a8679e65a4118563fb31a977a074f3b91c8` | 1,342 |
| `results_recovery/khatt_paragraph/runtime.json` | `dd4528a288fa081b99c26d689af313e9174b330ccbc71f9afe677aa486fa08f2` | 718 |
| `results_recovery/read_double_page/metrics.json` | `1386782dda47ef894ffb955a28f448c70cf61cdf4c9d51963d3a320997b682a3` | 1,422 |
| `results_recovery/read_double_page/runtime.json` | `8568eb57aaf8b8a9b8ff9c30077ec8caecf52ad5ff66d244be69494f3dd58b03` | 720 |
| `results_recovery/read_line/metrics.json` | `51d94aba55458adf55eb161c904d5a5a39d412b0d7adf945444a3b61cab9e6bb` | 1,340 |
| `results_recovery/read_line/runtime.json` | `bb1cc3522bc26e00960e9c5f478e39e3627199b77d1728119747a7061982b2a7` | 713 |
| `results_recovery/read_page/metrics.json` | `66a075bc51cf4ceec2a39d0c5fdc5cc82c975fb3335ce81afe693c22f0e0d5e9` | 1,400 |
| `results_recovery/read_page/runtime.json` | `38bf9367cfad5054c3f76d3640b1cb5360d7a18abcb96fcc1f1a57d9e4854b1a` | 708 |
| `results_recovery/read_triple_page/metrics.json` | `57947445e89aab2699152d9a50830c1c0b887db66bb4e3f3657ea903537eb408` | 1,418 |
| `results_recovery/read_triple_page/runtime.json` | `6c3f395598bed7791ac5757d89809ecfcef9510555fa91dfcba4edd24e1c4e74` | 724 |

## Stub directories — README only; contents are git-ignored

| File | SHA-256 | Bytes |
|---|---|---:|
| `data/README.md` | `f5e4a616de0f44765881d53fd2f9dbb82fb0db062f4c2528a50574bb274d903e` | 7,796 |
| `models/README.md` | `9a880852fd29a8eb374522e649de7a0edbb6f90e8941e28d7c0a54eca29f4ef2` | 1,940 |
