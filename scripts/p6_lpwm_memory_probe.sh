#!/usr/bin/env bash
# P6 (spec 17 item 12): how many frames fit? Runs the real LPWM trainer for one tiny epoch per setting and reports
# either the peak CUDA memory of a SUCCESSFUL epoch, or OOM. The context length must then be shared by BOTH world
# models:  timestep_horizon + 1 = context_frames + n_future  ->  set temporal.context_frames in configs/experiment.yaml.
#   CUDA_VISIBLE_DEVICES=4 bash scripts/p6_lpwm_memory_probe.sh configs/lpwm_robomme.json 23 27 31
# Run it on an IDLE GPU: another process on the same card eats into the budget and the answer comes out too small.
set -uo pipefail
source "$(dirname "$0")/env.sh"
CONF=${1:-configs/lpwm_robomme.json}; shift || true
export OWM_LPWM_MAX_EPISODES=4 OWM_NO_WANDB=1
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[[ "${used:-0}" -gt 1000 ]] && echo "警告：本卡已被占用 ${used} MiB，探测结果会偏小" >&2
for T in "${@:-23 27}"; do
  log=$(mktemp)
  $OWM_PY -m owm.wm.lpwm_train --config "$CONF" --probe --set timestep_horizon=$T num_epochs=1 warmup_epoch=0 \
      > "$log" 2>&1
  peak=$(tr '\r' '\n' < "$log" | grep -oE "peak CUDA memory: [0-9.]+ GiB" | tail -1 | grep -oE "[0-9.]+")
  # the atexit hook prints the peak even when the run crashed, so decide on the validation line, not on that number
  if tr '\r' '\n' < "$log" | grep -q "validation loss:"; then
    echo "config=$CONF timestep_horizon=$T (clip = $((T+1)) frames): 通过，峰值 ${peak:-?} GiB"
  elif tr '\r' '\n' < "$log" | grep -qiE "out of memory|OutOfMemoryError"; then
    echo "config=$CONF timestep_horizon=$T (clip = $((T+1)) frames): 显存不足（OOM，崩溃前 ${peak:-?} GiB）"
  else
    echo "config=$CONF timestep_horizon=$T (clip = $((T+1)) frames): 失败，日志 $log"
    continue
  fi
  rm -f "$log"
done
rm -rf "$OWM_OUT/lpwm_probe"   # probe checkpoints are disposable (several GB)
