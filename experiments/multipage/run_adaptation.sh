#!/bin/bash
# After the double-page fine-tune ends: export its best checkpoint, evaluate it on all three levels
# (GPU 0), train m=5 draft heads on it (reduced: 60 epochs on double pages) and evaluate the
# speculative row with the identity check. In parallel (GPU 1) a reduced 1.5 h triple-page
# adaptation is started from the double-page checkpoint and evaluated the same way.
cd "${HAND_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PYTHON:-python3}"
OUT=experiments/multipage/adaptation
log() { echo "### $(date -u +%FT%TZ) $*"; }
best() { ls $1/checkpoints/best_*.pt | sort -t_ -k2 -n | tail -1; }

CK=$(best outputs/ft_double_page_from_e14_s0); log "double best checkpoint: $CK"
$PY release/tools/export_release_checkpoint.py --ckpt $CK --out outputs/export_ft_double --expect-params 7033700 --expect-charset 99

# --- GPU 1: triple-page adaptation from the double-page checkpoint (reduced budget: 1.0 h) ---
mkdir -p outputs/ft_triple_page_from_double_s0
( env CUDA_VISIBLE_DEVICES=1 flock /tmp/hand-gpu1.lock $PY tools/train.py --dataset READ_2016 --level triple_page --variant _sem_dan --encoder fcn \
    --init-from $CK --no-hand-encoding --additional-tokens 1 --batch-size 1 --lr 1e-4 --fonts-dir fonts_dan_read \
    --output ft_triple_page_from_double_s0 --seed 0 --eval-interval 10 --experiment multipage_adaptation \
    --no-synthetic --start-valid-from-steps 0 --workers 4 --max-hours 1.0 --max-epochs 10000 \
    --notes "Multi-page adaptation, triple pages: continue the 4 h double-page checkpoint ($CK) on READ_2016_triple_page_sem_dan for 1.0 h at batch 1, real triples only (the synthetic generator produces single pages at this level), reduced budget." \
    > outputs/ft_triple_page_from_double_s0/train.log 2>&1
  CK3=$(best outputs/ft_triple_page_from_double_s0); log "triple best checkpoint: $CK3"
  $PY release/tools/export_release_checkpoint.py --ckpt $CK3 --out outputs/export_ft_triple --expect-params 7033700 --expect-charset 99
  env CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock $PY tools/multipage_eval.py --export-dir outputs/export_ft_triple --config-name FT_TRIPLE --kv-cache --amp --level page double_page triple_page --out-dir $OUT
  log "TRIPLE BRANCH DONE"
) > $OUT/triple_branch.log 2>&1 &

# --- GPU 0: evaluate the adapted double-page model, zero-shot on page and triple as well ---
env CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock $PY tools/multipage_eval.py --export-dir outputs/export_ft_double --config-name FT_DOUBLE --kv-cache --amp --level page double_page triple_page --out-dir $OUT
# --- GPU 0: draft heads on the adapted base (reduced: 100 epochs, double pages), then +E3 row ---
env CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock $PY tools/train_spec_heads.py --ckpt $CK --m 5 --epochs 60 --batch-size 2 --level double_page --max-hours 0.5 --out outputs/spec_heads_ft_double_m5
$PY release/tools/export_release_checkpoint.py --ckpt $CK --out outputs/export_ft_double --expect-params 7033700 --expect-charset 99 --heads outputs/spec_heads_ft_double_m5/heads.pt
env CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock $PY tools/multipage_eval.py --export-dir outputs/export_ft_double --heads outputs/export_ft_double/spec_heads_m5.safetensors --speculative --m 5 --config-name FT_DOUBLE_E3 --identity-ref FT_DOUBLE --amp --level page double_page triple_page --out-dir $OUT
log "DOUBLE BRANCH DONE"
wait
log "ALL DONE"
