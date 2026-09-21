# Source this file first:  source scripts/env.sh
# It pins the interpreter and builds a clean PYTHONPATH (the login shell's PYTHONPATH contains a package named
# `utils` that shadows LPWM's `utils/`, so it must not leak into these runs).
export OWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# the Docker image sets OWM_PY to its own copy of the env (docker/Dockerfile); on the host the default is used
export OWM_PY="${OWM_PY:-/common/home/zl1308/data/conda/envs/memory_owm/bin/python}"
export CJEPA_ROOT=/common/home/zl1308/Projects/cjepa
export LPWM_ROOT=/common/home/zl1308/Projects/lpwm
export PYTHONPATH="$OWM_ROOT"
export PATH="$(dirname "$OWM_PY"):$PATH"
export HF_HUB_DISABLE_TELEMETRY=1
# LPWM 在 24 帧上下文下峰值约 41 GiB / 44 GiB 可用，余量很小；减少显存碎片以免长训练中途 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# bulk storage for cache / data / outputs (configs/experiment.yaml -> paths.storage_root); the home volume has a quota
export OWM_STORAGE="$($OWM_PY -c "import yaml;print(yaml.safe_load(open('$OWM_ROOT/configs/experiment.yaml'))['paths']['storage_root'])")"
export OWM_CACHE="$OWM_STORAGE/cache" OWM_OUT="$OWM_STORAGE/outputs"
