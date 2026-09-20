#!/usr/bin/env bash
# P6 (spec 17 item 12): how many frames fit? Runs the real LPWM trainer for one tiny epoch per setting and prints the
# peak CUDA memory (batch_size 1, fp32). The context length must then be shared by BOTH world models:
#   timestep_horizon + 1 = context_frames + n_future   ->   set temporal.context_frames in configs/experiment.yaml.
#   CUDA_VISIBLE_DEVICES=4 bash scripts/p6_lpwm_memory_probe.sh configs/lpwm_robomme.json 11 23 35
set -uo pipefail
source "$(dirname "$0")/env.sh"
CONF=${1:-configs/lpwm_robomme.json}; shift || true
export OWM_LPWM_MAX_EPISODES=4 OWM_CONFIG_OUTPUT_SUFFIX=probe
for T in "${@:-11 23}"; do
  out=$($OWM_PY -m owm.wm.lpwm_train --config "$CONF" --probe --set timestep_horizon=$T num_epochs=1 warmup_epoch=0 2>&1 | tr '\r' '\n' | grep -E "peak CUDA memory|OutOfMemory|out of memory" | tail -1)
  echo "config=$CONF timestep_horizon=$T (clip = $((T+1)) frames): ${out:-failed (see log)}"
done
rm -rf "$OWM_OUT/lpwm_probe"   # probe checkpoints are disposable (several GB)
