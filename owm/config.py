"""Experiment configuration and path helpers."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

ALL_TASKS = [
    "BinFill", "ButtonUnmask", "ButtonUnmaskSwap", "InsertPeg", "MoveCube", "PatternLock",
    "PickHighlight", "PickXtimes", "RouteStick", "StopCube", "SwingXtimes", "VideoPlaceButton",
    "VideoPlaceOrder", "VideoRepick", "VideoUnmask", "VideoUnmaskSwap",
]
SOURCES = ("train", "test")  # train = official h5; test = generated h5 (plan A)


class Cfg(dict):
    """dict with attribute access."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


@lru_cache(maxsize=None)
def load_cfg(path: str | None = None) -> Cfg:
    path = path or os.environ.get("OWM_CONFIG", str(ROOT / "configs" / "experiment.yaml"))
    with open(path) as f:
        return Cfg(yaml.safe_load(f))


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def data_dir() -> Path:
    return resolve(load_cfg().paths.data_dir)


def cache_dir() -> Path:
    return resolve(load_cfg().paths.cache_dir)


def output_dir() -> Path:
    return resolve(load_cfg().paths.output_dir)


def h5_dir(source: str) -> Path:
    cfg = load_cfg()
    if source == "test" and os.environ.get("OWM_TEST_H5_DIR"):
        return Path(os.environ["OWM_TEST_H5_DIR"])
    return resolve(cfg.paths.train_h5_dir if source == "train" else cfg.paths.test_h5_dir)


def episode_h5(source: str, task: str, idx: int) -> Path:
    """The generated test set keeps one file per episode (<h5_dir>/episodes/<Task>_ep<i>.h5, group episode_<i>);
    the official data has one file per task."""
    single = h5_dir(source) / "episodes" / f"{task}_ep{idx}.h5"
    return single if single.exists() else h5_path(source, task)


def h5_path(source: str, task: str) -> Path:
    return h5_dir(source) / f"record_dataset_{task}.h5"


def wm_tasks() -> list[str]:
    cfg = load_cfg()
    return list(ALL_TASKS) if cfg.wm_train_tasks == "all_16" else list(cfg.wm_train_tasks)


def future_offsets() -> list[int]:
    return list(load_cfg().temporal.future_offsets)


def add_repo_to_path(name: str) -> Path:
    """Make a third-party repo importable (cjepa / lpwm). Nothing in those repos is modified."""
    repo = resolve(load_cfg().paths[f"{name}_repo"])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo
