"""Future-perturbation test (spec 8.4, 16.7) — must pass 100% before any world-model result is used.

Every frame after t (and, for the action variants, every action after t) is replaced by noise and the WHOLE pipeline
(frame encoding -> slots / particles -> prediction -> positions) is run again; outputs must be identical."""
from __future__ import annotations

import numpy as np

from owm.data.h5_reader import Episode
from owm.data.sampling import history_indices
from owm.wm.actions import action_chunks


def history_actions(ep: Episode, t: int, noise_after_t: bool = False) -> np.ndarray:
    ep_meta = ep.meta
    if noise_after_t:   # corrupt every action after t in a copy of the episode, then chunk as usual
        ja = ep_meta["joint_action"].copy()
        ja[t + 1:] = np.random.default_rng(4321).normal(size=ja[t + 1:].shape)
        ep = Episode(ep.source, ep.task, ep.idx)
        ep.__dict__["meta"] = dict(ep_meta, joint_action=ja)
    return action_chunks(ep, history_indices(t))


def future_perturbation_test(adapter, ep: Episode, t: int, seed: int = 0, atol: float = 1e-5, control: bool = False) -> dict:
    use_act = getattr(adapter, "with_actions", False)
    frames = np.asarray(ep.frames)
    kw = dict(actions=history_actions(ep, t)) if use_act else {}
    tok_a, uv_a, _ = adapter.predict(frames, t, seed=seed, **kw)
    noisy = frames.copy()
    rng = np.random.default_rng(1234)
    noisy[t + 1:] = rng.integers(0, 256, size=noisy[t + 1:].shape, dtype=np.uint8)
    kw = dict(actions=history_actions(ep, t, noise_after_t=True)) if use_act else {}
    tok_b, uv_b, _ = adapter.predict(noisy, t, seed=seed, **kw)
    d_tok, d_uv = float(np.abs(tok_a - tok_b).max()), float(np.abs(uv_a - uv_b).max())
    extra = {}
    if control:   # negative control: corrupting the decision frame itself MUST change the prediction
        noisy[t] = rng.integers(0, 256, size=noisy[t].shape, dtype=np.uint8)
        tok_c, _, _ = adapter.predict(noisy, t, seed=seed, **kw)
        extra["control_changed"] = bool(np.abs(tok_a - tok_c).max() > atol)
    return dict(**extra, key=ep.key, t=t, max_abs_token_diff=d_tok, max_abs_uv_diff=d_uv,
                passed=bool(d_tok <= atol and d_uv <= atol), n_future_frames=int(len(frames) - t - 1))
