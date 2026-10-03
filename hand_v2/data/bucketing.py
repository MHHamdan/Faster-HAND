"""
Shape bucketing for the input batch.

cuDNN 9's default (v8 API, heuristic mode) builds and compiles an execution plan the first
time it sees a convolution input shape: ~1.3 s per new shape for FCN_Encoder at line
scale, ~1.8 s at page scale, against 14 / 75 ms once cached
(experiments/benchmark_suite/profiling/cudnn_shape_bench*.json). Handwriting batches are
padded to their largest sample, so nearly every batch is a new shape.

`BucketedOCRCollate` pads the batch image tensor (bottom/right, same padding value the
baseline collate uses) up to a multiple of (h_mult, w_mult), so the number of distinct
shapes a run sees is bounded and the plan cache hits after warm-up. Per-sample
`imgs_shape` / `imgs_reduced_shape` / `imgs_position` are untouched: the encoder mask
already excludes padded positions for every sample, exactly as it does for the batch
padding the baseline applies. The only model-visible difference is slightly more zero
padding in the instance-normalisation statistics, of the same kind the baseline already
introduces through batch composition.

Off by default (`--shape-bucket 0 0`). Alternative with no numerical side effect:
`--cudnn-api v7` (see hand_v2/train.py).
"""
import torch
import torch.nn.functional as F

from hand.OCR.ocr_dataset_manager import OCRCollateFunction, OCRDataset


def _round_up(n, m):
    return n if m <= 1 else -(-n // m) * m


class BucketedOCRCollate(OCRCollateFunction):
    def __init__(self, config):
        super().__init__(config)
        self.h_mult, self.w_mult = config.get("shape_bucket", (0, 0))
        self.train_only = config.get("shape_bucket_train_only", True)

    def __call__(self, batch_data):
        out = super().__call__(batch_data)
        if self.h_mult <= 1 and self.w_mult <= 1:
            return out
        imgs = out["imgs"]                                   # (B, C, H, W), float32
        b, c, h, w = imgs.shape
        hb, wb = _round_up(h, self.h_mult), _round_up(w, self.w_mult)
        if (hb, wb) != (h, w):
            out["imgs"] = F.pad(imgs, (0, wb - w, 0, hb - h), value=self.img_padding_value)
            out["shape_bucketed_from"] = [h, w]
        return out


class BucketedOCRDataset(OCRDataset):
    """OCRDataset whose collate pads batches to shape buckets (training split only by default)."""

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        super().__init__(params, set_name, custom_name, paths_and_sets)
        cfg = params["config"]
        if set_name == "train" or not cfg.get("shape_bucket_train_only", True):
            self.collate_function = BucketedOCRCollate


def install_shape_bucketing(params, h_mult, w_mult, train_only=True):
    """Route the dataset manager through the bucketed collate. No-op when both multiples are 0."""
    cfg = params["dataset_params"]["config"]
    cfg["shape_bucket"] = (int(h_mult), int(w_mult))
    cfg["shape_bucket_train_only"] = bool(train_only)
    if h_mult > 1 or w_mult > 1:
        params["dataset_params"]["dataset_class"] = BucketedOCRDataset
    return params
