#!/usr/bin/env bash
# P5b: train VideoSAUR with the STOCK trainer of the C-JEPA repo (nothing in that repo is modified).
#   bash scripts/p5b_train_videosaur.sh [NUM_SLOTS] [extra videosaur overrides ...]
#   NUM_SLOTS 省略时取 configs/experiment.yaml -> cjepa.num_slots
#   单卡:  CUDA_VISIBLE_DEVICES=2     bash scripts/p5b_train_videosaur.sh 20
#   多卡:  CUDA_VISIBLE_DEVICES=2,3,4 bash scripts/p5b_train_videosaur.sh 20     # 自动 DDP，卡数由 CUDA_VISIBLE_DEVICES 决定
# env: SHARD_DIR (default $OWM_CACHE/videosaur_shards), LOG_DIR (default $OWM_OUT/videosaur)
set -euo pipefail
source "$(dirname "$0")/env.sh"
# slot 数 N：不带参数时取 configs/experiment.yaml -> cjepa.num_slots（P2 实测 n_max + 3）
N=${1:-$(PYTHONPATH="$OWM_ROOT" $OWM_PY -c "from owm.config import load_cfg; print(load_cfg().cjepa.num_slots)")}
shift 2>/dev/null || true
SHARD_DIR=$(realpath "${SHARD_DIR:-$OWM_CACHE/videosaur_shards}")
LOG_DIR=$(realpath -m "${LOG_DIR:-$OWM_OUT/videosaur}")
INFO="$SHARD_DIR/info.json"
NTR=$($OWM_PY -c "import json;print(json.load(open('$INFO'))['train']['shards']-1)")
NVA=$($OWM_PY -c "import json;print(json.load(open('$INFO'))['val']['shards']-1)")
VSZ=$($OWM_PY -c "import json;print(json.load(open('$INFO'))['val']['samples'])")
TR=$(printf "%s/train/robomme-train-{000000..%06d}.tar" "$SHARD_DIR" "$NTR")
VA=$(printf "%s/val/robomme-val-{000000..%06d}.tar" "$SHARD_DIR" "$NVA")
# 卡数 = CUDA_VISIBLE_DEVICES 里的个数。NUM_GPUS 必须跟着改：官方配置按总 batch 缩放学习率
# (lr = NUM_GPUS * BATCH_SIZE_PER_GPU / 32 * BASE_LR)，只改 trainer.devices 会让 lr 偏小。
NGPU=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-0}")
NW=$(( NTR + 1 < 8 ? NTR + 1 : 8 ))
# webdataset 要求每张卡至少 1 个 shard，且 num_workers <= 每卡 shard 数（datamodules.py:171-185）
NW=$(( NW < (NTR + 1) / NGPU ? NW : (NTR + 1) / NGPU ))
if (( (NVA + 1) < NGPU )); then
  echo "验证 shard 只有 $((NVA + 1)) 个，放不下 $NGPU 张卡。减少卡数，或用 --per-shard 调小重造 shard。" >&2
  exit 1
fi
# 多卡默认把每卡 batch 除以卡数，保持总 batch（以及由它推出的学习率）和单卡一致 —— 多卡纯粹是提速，不改训练配方。
# 若不除：每卡仍是 64，总 batch = 64 x 卡数，单步耗时不变 => 100k 步墙钟时间和单卡一样，白白多花算力。
# KEEP_TOTAL_BATCH=0 可关闭（此时学习率会按官方线性缩放规则自动变大）。
BATCH_OVERRIDE=""
DDP_OVERRIDE=""
BPG=$($OWM_PY -c "import yaml;print(yaml.safe_load(open('$OWM_ROOT/configs/videosaur_robomme.yml'))['globals']['BATCH_SIZE_PER_GPU'])")
if (( NGPU > 1 )); then
  # 冻结的 DINOv2 backbone 仍然带着 requires_grad=True（modules/encoders.py 硬编码 AutoModel.from_pretrained，
  # 配置里的 frozen: true 走的是被绕过的 timm 分支），前向又在 no_grad 里 => 223 个参数永远拿不到梯度。
  # 单卡无所谓，DDP 默认的 ddp_find_unused_parameters_false 会直接报错，所以多卡时必须放开这个检测。
  DDP_OVERRIDE="trainer.strategy=ddp_find_unused_parameters_true"
  if [[ "${KEEP_TOTAL_BATCH:-1}" == "1" ]]; then
    PER_GPU=$(( (BPG + NGPU - 1) / NGPU ))
    BATCH_OVERRIDE="globals.BATCH_SIZE_PER_GPU=$PER_GPU"
    echo "[owm] $NGPU 卡：每卡 batch $PER_GPU，总 $((PER_GPU * NGPU))（单卡为 $BPG）；KEEP_TOTAL_BATCH=0 可保留每卡 $BPG"
  fi
fi
# wandb settings come from configs/experiment.yaml -> wandb, same as every other training here
WB=$(PYTHONPATH="$OWM_ROOT" $OWM_PY -c "
from owm.wandb_utils import wandb_config
w = wandb_config()
print(' '.join(['wandb.enable=%s' % str(bool(w['enable'])).lower(), 'wandb.project=%s' % w['project'],
                'wandb.mode=%s' % w['mode']] + (['wandb.entity=%s' % w['entity']] if w['entity'] else [])))")
cd "$CJEPA_ROOT"
PYTHONPATH="$CJEPA_ROOT:$OWM_ROOT/owm/shims:$OWM_ROOT" $OWM_PY src/third_party/videosaur/videosaur/train.py \
  --log-dir "$LOG_DIR" --no-interactive \
  "$OWM_ROOT/configs/videosaur_robomme.yml" \
  globals.NUM_SLOTS=$N "dataset.train_shards=$TR" "dataset.val_shards=$VA" dataset.val_size=$VSZ \
  dataset.num_workers=$NW trainer.devices=$NGPU globals.NUM_GPUS=$NGPU $BATCH_OVERRIDE $DDP_OVERRIDE $WB "$@"
