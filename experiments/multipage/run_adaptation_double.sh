#!/bin/bash
# Double branch of the adaptation pipeline, relaunched after the LOER guard was added to tools/multipage_eval.py
# (the first attempt hung in the exact graph edit distance on a single page the adapted model read as 8 page blocks).
cd "${HAND_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PYTHON:-python3}"
OUT=experiments/multipage/adaptation
CK=outputs/ft_double_page_from_e14_s0/checkpoints/best_220.pt
log() { echo "### $(date -u +%FT%TZ) $*"; }
log "double branch restart, checkpoint $CK"
env CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock $PY tools/multipage_eval.py --export-dir outputs/export_ft_double --config-name FT_DOUBLE --kv-cache --amp --level page double_page triple_page --out-dir $OUT
env CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock $PY tools/train_spec_heads.py --ckpt $CK --m 5 --epochs 60 --batch-size 2 --level double_page --max-hours 0.5 --out outputs/spec_heads_ft_double_m5
$PY release/tools/export_release_checkpoint.py --ckpt $CK --out outputs/export_ft_double --expect-params 7033700 --expect-charset 99 --heads outputs/spec_heads_ft_double_m5/heads.pt
env CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock $PY tools/multipage_eval.py --export-dir outputs/export_ft_double --heads outputs/export_ft_double/spec_heads_m5.safetensors --speculative --m 5 --config-name FT_DOUBLE_E3 --identity-ref FT_DOUBLE --amp --level page double_page triple_page --out-dir $OUT
log "DOUBLE BRANCH DONE"
