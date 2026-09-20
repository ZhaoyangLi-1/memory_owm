"""Unified evidence format (spec 8.3).

cache/evidence/{name}/{source}/{task}/{episode}_{t}.npz
  tokens [S, K, n, D]   S samples (C-JEPA 1, LPWM 5, GT 1), K future steps, n tokens
  uv     [S, K, n, 2]   image position of every token, (u, v) in [0, 1]
  valid  [S, K, n]      False = padding
  ids    [S, K, n]      GT identity (gt_future only; NO_ID elsewhere)
  step   [K]            1..K
name: gt_future | cjepa | lpwm | cjepa_act | lpwm_act | action_history
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from owm.config import cache_dir, future_offsets
from owm.data.gt_state import GT, OBJ_TYPES
from owm.data.sampling import future_indices

NO_ID, ARM_ID = 33, 32
D_GT = len(OBJ_TYPES) + 4 + 3 + 2  # type(6) + colour(4) + xyz(3) + grip width + grip closed = 15


def evidence_path(name: str, source: str, task: str, episode: int, t: int) -> Path:
    return cache_dir() / "evidence" / name / source / task / f"{episode}_{t}.npz"


def save_evidence(path: Path, tokens, uv, valid, ids=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    S, K, n, _ = tokens.shape
    ids = np.full((S, K, n), NO_ID, np.int64) if ids is None else ids
    np.savez_compressed(path, tokens=tokens.astype(np.float32), uv=uv.astype(np.float32),
                        valid=valid.astype(bool), ids=ids.astype(np.int64), step=np.arange(1, K + 1))


def load_evidence(name: str, source: str, task: str, episode: int, t: int) -> dict:
    with np.load(evidence_path(name, source, task, episode, t)) as z:
        return {k: z[k] for k in z.files}


def gt_token_features(tok: dict, blind: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GT tokens of one frame -> (features [n+1, 15], uv [n+1, 2], ids [n+1], is_arm [n+1]); the arm is last.
    blind=True zeroes type and colour (prediction check: identity and position only)."""
    n = len(tok["id"])
    feat = np.zeros((n + 1, D_GT), np.float32)
    if not blind:
        feat[np.arange(n), tok["type"]] = 1.0
        feat[np.arange(n), len(OBJ_TYPES) + tok["color"]] = 1.0
    feat[:n, 10:13] = tok["pos"]
    feat[n, 10:13] = tok["arm_pos"]
    feat[n, 13], feat[n, 14] = tok["grip_width"], tok["grip_closed"]
    uv = np.concatenate([tok["uv"], tok["arm_uv"][None]], 0).astype(np.float32)
    ids = np.concatenate([tok["id"], [ARM_ID]]).astype(np.int64)
    is_arm = np.zeros(n + 1, bool)
    is_arm[n] = True
    return feat, np.clip(uv, 0.0, 1.0), ids, is_arm


def build_gt_future(gt: GT, t: int, n_frames: int) -> dict:
    """Ceiling evidence: the real future frames t+16, t+32 as GT tokens (clamped at the episode end)."""
    idx, clamped = future_indices(t, n_frames, future_offsets())
    idx = [min(i, gt.frame0 + gt.Te - 1) for i in idx]
    per_step = [gt_token_features(gt.tokens(i)) for i in idx]
    K, n = len(per_step), max(len(p[0]) for p in per_step)
    tokens, uv = np.zeros((1, K, n, D_GT), np.float32), np.zeros((1, K, n, 2), np.float32)
    valid, ids = np.zeros((1, K, n), bool), np.full((1, K, n), NO_ID, np.int64)
    for k, (f, u, i, _) in enumerate(per_step):
        tokens[0, k, :len(f)], uv[0, k, :len(f)], valid[0, k, :len(f)], ids[0, k, :len(f)] = f, u, True, i
    return dict(tokens=tokens, uv=uv, valid=valid, ids=ids, clamped=np.array(clamped))
