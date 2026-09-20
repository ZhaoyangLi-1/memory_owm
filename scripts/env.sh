# Source this file first:  source scripts/env.sh
# It pins the interpreter and builds a clean PYTHONPATH (the login shell's PYTHONPATH contains a package named
# `utils` that shadows LPWM's `utils/`, so it must not leak into these runs).
export OWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OWM_PY=/common/home/zl1308/data/conda/envs/memory_owm/bin/python
export CJEPA_ROOT=/common/home/zl1308/Projects/cjepa
export LPWM_ROOT=/common/home/zl1308/Projects/lpwm
export PYTHONPATH="$OWM_ROOT"
export PATH="$(dirname "$OWM_PY"):$PATH"
export HF_HUB_DISABLE_TELEMETRY=1
# bulk storage for cache / data / outputs (configs/experiment.yaml -> paths.storage_root); the home volume has a quota
export OWM_STORAGE="$($OWM_PY -c "import yaml;print(yaml.safe_load(open('$OWM_ROOT/configs/experiment.yaml'))['paths']['storage_root'])")"
export OWM_CACHE="$OWM_STORAGE/cache" OWM_OUT="$OWM_STORAGE/outputs"
