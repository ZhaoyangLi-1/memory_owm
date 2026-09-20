"""Accuracy, memory score S and confidence intervals (spec 2.2, 2.5, 16.6): paired bootstrap over episodes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from owm.config import load_cfg


def seed_mean(pred: pd.DataFrame, condition: str, decision_type: str) -> pd.DataFrame:
    """Per decision: correctness averaged over the readout seeds (0, 1/3, 2/3, 1)."""
    p = pred[(pred.condition == condition) & (pred.decision_type == decision_type)]
    return p.groupby(["key", "t"], as_index=False)["correct"].mean()


def _episode_sums(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby("key")
    return g[cols].sum().to_numpy(dtype=np.float64), g.size().to_numpy(dtype=np.float64)


def _resample(n_ep: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_ep, size=(n, n_ep))


def bootstrap_accuracy(df: pd.DataFrame, col: str = "correct", n: int | None = None, seed: int = 0) -> dict:
    n = n or load_cfg().stats.bootstrap_samples
    sums, cnt = _episode_sums(df, [col])
    idx = _resample(len(cnt), n, seed)
    acc = sums[idx, 0].sum(1) / cnt[idx].sum(1)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return dict(acc=float(df[col].mean()), ci_low=float(lo), ci_high=float(hi), n=int(len(df)), n_episodes=int(len(cnt)))


def bootstrap_scores(df: pd.DataFrame, n: int | None = None, seed: int = 0) -> dict:
    """df: one row per memory-critical decision with columns key, floor, wm, ceil (seed-averaged correctness).
    Every resample recomputes the three accuracies and S together (paired)."""
    n = n or load_cfg().stats.bootstrap_samples
    cols = ["floor", "wm", "ceil"]
    sums, cnt = _episode_sums(df, cols)
    idx = _resample(len(cnt), n, seed)
    acc = sums[idx].sum(1) / cnt[idx].sum(1, keepdims=True)                     # [n, 3]
    f, w, c = acc[:, 0], acc[:, 1], acc[:, 2]
    s = (w - f) / np.maximum(c - f, 1e-6)
    stats = np.stack([f, w, c, s], 1)
    point = df[cols].mean().to_numpy()
    s_point = (point[1] - point[0]) / max(point[2] - point[0], 1e-6)
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    names = cols + ["S"]
    return dict(acc_floor=float(point[0]), acc_wm=float(point[1]), acc_ceil=float(point[2]), S=float(s_point),
                ci_low=dict(zip(names, map(float, lo))), ci_high=dict(zip(names, map(float, hi))),
                n=int(len(df)), n_episodes=int(len(cnt)))


def bootstrap_difference(df: pd.DataFrame, a: str, b: str, n: int | None = None, seed: int = 0) -> dict:
    """Paired difference acc(a) - acc(b), e.g. the action gain (spec 14.5) or floor+actions vs floor (14.4)."""
    n = n or load_cfg().stats.bootstrap_samples
    sums, cnt = _episode_sums(df, [a, b])
    idx = _resample(len(cnt), n, seed)
    acc = sums[idx].sum(1) / cnt[idx].sum(1, keepdims=True)
    d = acc[:, 0] - acc[:, 1]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=float(df[a].mean() - df[b].mean()), ci_low=float(lo), ci_high=float(hi))


def paired_table(pred: pd.DataFrame, task: str, conditions: dict[str, str],
                 decision_type: str = "memory_critical", split: str = "test") -> pd.DataFrame:
    """conditions: column name -> condition name, e.g. {"floor": "floor", "wm": "cjepa", "ceil": "ceiling"}.
    Only decisions present in every condition are kept (they are the same set by construction)."""
    p = pred[(pred.task == task) & (pred.split == split)]
    out = None
    for col, cond in conditions.items():
        m = seed_mean(p, cond, decision_type).rename(columns={"correct": col})
        out = m if out is None else out.merge(m, on=["key", "t"], how="inner")
    return out if out is not None else pd.DataFrame()
