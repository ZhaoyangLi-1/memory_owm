"""Action chunks for the optional action-conditioned experiment (spec 14.3).

h5 frame k stores the observation AFTER executing joint_action[k], so the 16 actions that lead to sampled frame f
are joint_action[f-15 .. f]. They are concatenated into one 128-d vector (16 x 8, absolute joint targets + gripper).
Two flag channels follow:  [128] = "no action" (conditioning-video step without a real action, zero vector),
                           [129] = "unknown action" (future step, or dropped during training).
Verified (spec 17 item 14): video-phase joint_action is CONSTANT for VideoUnmask / VideoUnmaskSwap (no information),
but it is the real demonstration motion for VideoRepick and MoveCube (leak risk -> see the leak check 14.4).
"""
from __future__ import annotations

import numpy as np

from owm.config import load_cfg
from owm.data.h5_reader import Episode, parse_key

ACTION_DIM = 128
ACTION_TOKEN_DIM = ACTION_DIM + 2


def action_chunks(ep: Episode, frame_idx) -> np.ndarray:
    """[len(frame_idx), 130] action tokens for the given sampled frames."""
    chunk = load_cfg().actions.chunk
    ja = np.nan_to_num(ep.meta["joint_action"].astype(np.float32))
    video = ep.meta["is_video_demo"]
    out = np.zeros((len(frame_idx), ACTION_TOKEN_DIM), np.float32)
    for i, f in enumerate(frame_idx):
        lo = f - chunk + 1
        a = ja[max(lo, 0): f + 1]
        if lo < 0:  # episode start: repeat the first action
            a = np.concatenate([np.repeat(a[:1], -lo, 0), a])
        if video[f] and np.ptp(a, axis=0).max() < 1e-6:
            out[i, ACTION_DIM] = 1.0            # scripted video without robot actions
        else:
            out[i, :ACTION_DIM] = a.reshape(-1)
    return out


def sequence_actions(slot_key: str) -> np.ndarray:
    """Actions aligned with a slot sequence saved by p5c ("{source}/{task}/episode_{i}@{offset}")."""
    key, off = slot_key.rsplit("@", 1)
    ep = parse_key(key)
    stride = load_cfg().temporal.stride
    return action_chunks(ep, np.arange(int(off), ep.T, stride))


def unknown_actions(n: int) -> np.ndarray:
    out = np.zeros((n, ACTION_TOKEN_DIM), np.float32)
    out[:, ACTION_DIM + 1] = 1.0
    return out
