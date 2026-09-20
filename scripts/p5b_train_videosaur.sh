#!/usr/bin/env bash
# P5b: train VideoSAUR with the STOCK trainer of the C-JEPA repo (nothing in that repo is modified).
#   bash scripts/p5b_train_videosaur.sh [NUM_SLOTS] [extra videosaur overrides ...]
#   e.g. CUDA_VISIBLE_DEVICES=2 bash scripts/p5b_train_videosaur.sh 12 trainer.max_steps=100000
# env: SHARD_DIR (default $OWM_CACHE/videosaur_shards), LOG_DIR (default $OWM_OUT/videosaur)
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=${1:-12}; shift || true
SHARD_DIR=$(realpath "${SHARD_DIR:-$OWM_CACHE/videosaur_shards}")
LOG_DIR=$(realpath -m "${LOG_DIR:-$OWM_OUT/videosaur}")
INFO="$SHARD_DIR/info.json"
NTR=$($OWM_PY -c "import json;print(json.load(open('$INFO'))['train']['shards']-1)")
NVA=$($OWM_PY -c "import json;print(json.load(open('$INFO'))['val']['shards']-1)")
VSZ=$($OWM_PY -c "import json;print(json.load(open('$INFO'))['val']['samples'])")
TR=$(printf "%s/train/robomme-train-{000000..%06d}.tar" "$SHARD_DIR" "$NTR")
VA=$(printf "%s/val/robomme-val-{000000..%06d}.tar" "$SHARD_DIR" "$NVA")
NW=$(( NTR + 1 < 8 ? NTR + 1 : 8 ))
cd "$CJEPA_ROOT"
PYTHONPATH="$CJEPA_ROOT:$OWM_ROOT/owm/shims:$OWM_ROOT" $OWM_PY src/third_party/videosaur/videosaur/train.py \
  --log-dir "$LOG_DIR" --no-interactive \
  "$OWM_ROOT/configs/videosaur_robomme.yml" \
  globals.NUM_SLOTS=$N "dataset.train_shards=$TR" "dataset.val_shards=$VA" dataset.val_size=$VSZ \
  dataset.num_workers=$NW trainer.devices=1 "$@"
