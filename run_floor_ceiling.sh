#!/usr/bin/env bash
cd /common/home/zl1308/Projects/owm_memory
source scripts/env.sh
mkdir -p "$OWM_OUT/logs"

TASKS=(PickXtimes VideoRepick ButtonUnmaskSwap VideoUnmaskSwap MoveCube VideoUnmask)
GPUS=(0 1 2 3)
for i in "${!TASKS[@]}"; do
  t=${TASKS[$i]}; g=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "$t -> GPU $g"
  CUDA_VISIBLE_DEVICES=$g $OWM_PY scripts/p4_train_readout.py \
      --conditions floor ceiling --tasks "$t" \
      > "$OWM_OUT/logs/p4_$t.log" 2>&1 &
done
wait
echo "All readout trainings finished."
grep -hE "seed[0-9] test" "$OWM_OUT"/logs/p4_*.log | awk '{print $1,$2,$3,$6,$7}' | column -t