#!/bin/bash
# T2 zero-shot scaling sweep: four configurations x three levels, AMP fp16, plus an fp32 BASE single-page row.
# Every CUDA process goes through the GPU-0 lock.
cd "${HAND_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PYTHON:-python3}"
L="CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock"
OUT=experiments/multipage/zero_shot
run() { echo "### $(date -u +%FT%TZ) $*"; env CUDA_VISIBLE_DEVICES=0 flock /tmp/hand-gpu0.lock $PY tools/multipage_eval.py "$@"; }
run --export-dir outputs/export_e14 --config-name BASE --kv-cache --amp --level page double_page triple_page --out-dir $OUT
run --export-dir outputs/export_e14 --heads outputs/export_e14/spec_heads_m5.safetensors --speculative --m 5 --config-name E3 --identity-ref BASE --amp --level page double_page triple_page --out-dir $OUT
run --export-dir outputs/export_e4 --config-name E4 --kv-cache --amp --level page double_page triple_page --out-dir $OUT
run --export-dir outputs/export_e4 --heads outputs/export_e4/spec_heads_m5.safetensors --speculative --m 5 --config-name E3E4 --identity-ref E4 --amp --level page double_page triple_page --out-dir $OUT
run --export-dir outputs/export_e14 --config-name BASE --kv-cache --level page --tag _fp32 --out-dir $OUT
echo "### $(date -u +%FT%TZ) ALL DONE"
