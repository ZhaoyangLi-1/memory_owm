"""Pre-registered reading rules (spec 2.4). Applied in order; the first hit wins."""
from __future__ import annotations

from owm.config import load_cfg

VERDICTS = {
    "ceiling_fail": "rule 1: ceiling < 95% — two future steps do not reveal the answer / label rule problem",
    "floor_leak": "rule 2: floor significantly above chance — the current frame leaks the answer (exclude)",
    "cannot_see_or_predict": "rule 3: S CI contains 0 and prediction check < 70% — cannot see / cannot predict",
    "cannot_remember": "rule 4: S CI contains 0 and prediction check >= 70% — sees and predicts, does not remember",
    "no_memory_signal_check_insufficient": "rule 3/4: S CI contains 0; prediction check has too few samples to split 3 vs 4",
    "partial": "rule 5: S significantly > 0 but < 0.5 — partially remembers",
    "remembers": "rule 6: S >= 0.5 — remembers",
}


def verdict(scores: dict, chance: float, check_acc: float | None) -> str:
    """scores: output of bootstrap_scores; check_acc: prediction-check accuracy or None (insufficient samples)."""
    th = load_cfg().stats.thresholds
    if scores["acc_ceil"] < th.ceiling_min:
        return "ceiling_fail"
    if scores["ci_low"]["floor"] > chance:
        return "floor_leak"
    s_significant = scores["ci_low"]["S"] > 0
    if not s_significant:
        if check_acc is None:
            return "no_memory_signal_check_insufficient"
        return "cannot_see_or_predict" if check_acc < th.prediction_check_min else "cannot_remember"
    return "partial" if scores["S"] < th.s_partial else "remembers"


def needs_second_wm_seed(scores: dict) -> bool:
    """Spec 2.5: train a 2nd world-model seed if the CI of S straddles a reading threshold (0 or 0.5)."""
    th = load_cfg().stats.thresholds
    lo, hi = scores["ci_low"]["S"], scores["ci_high"]["S"]
    return (lo <= 0 <= hi) or (lo <= th.s_partial <= hi)
