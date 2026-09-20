"""RoboMME video dataset with the interface LPWM's training / evaluation code expects.

__getitem__ -> (frames [T, 3, H, W] float32 in [0, 1], actions [T, A], 0-tensor, 0-tensor, 0-tensor); an unconditional
run only reads element 0, an action-conditioned run also reads element 1.

Clips are stride-16 sequences (spec 5.4): training draws a random offset in [0, 15] and a random start; validation uses
offset 0 from the episode start. Sequences shorter than the clip are padded at the END by repeating the last frame:
LPWM's context / dynamics transformers are causal, so every un-padded prefix — which is what the adapter feeds at
evaluation time — is unaffected by the padding.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from owm.config import load_cfg, wm_tasks
from owm.data.h5_reader import list_cached
from owm.data.splits import wm_split
from owm.wm.actions import ACTION_TOKEN_DIM, action_chunks

TRAIN_CLIPS_PER_EPISODE = 8


def to_lpwm_frames(frames: np.ndarray, image_size: int) -> torch.Tensor:
    """uint8 [T, 256, 256, 3] -> float [T, 3, S, S] in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", antialias=True, align_corners=False)
    return x


class RoboMMEVideoDataset(Dataset):
    def __init__(self, root=None, mode="train", sample_length=25, image_size=256, with_actions=False):
        cfg = load_cfg()
        self.mode = "train" if mode == "train" else "val"
        self.T, self.image_size, self.stride = int(sample_length), int(image_size), cfg.temporal.stride
        self.with_actions = with_actions
        self.episodes = [e for t in wm_tasks() for e in list_cached("train", t)
                         if wm_split(e.source, e.task, e.idx) == self.mode]
        import os
        if os.environ.get("OWM_LPWM_MAX_EPISODES"):   # debugging / memory probe only
            self.episodes = self.episodes[: int(os.environ["OWM_LPWM_MAX_EPISODES"])]
        if not self.episodes:
            raise RuntimeError("no cached episodes — run scripts/p0_extract_cache.py first")

    def __len__(self):
        return len(self.episodes) * (TRAIN_CLIPS_PER_EPISODE if self.mode == "train" else 1)

    def __getitem__(self, i):
        ep = self.episodes[i % len(self.episodes)]
        if self.mode == "train":
            off = np.random.randint(self.stride)
            n = len(range(off, ep.T, self.stride))
            start = np.random.randint(max(n - self.T, 0) + 1)
        else:
            off, start = 0, 0
        idx = np.arange(off, ep.T, self.stride)[start:start + self.T]
        if len(idx) < self.T:
            idx = np.concatenate([idx, np.repeat(idx[-1:], self.T - len(idx))])
        frames = to_lpwm_frames(ep.frames[idx], self.image_size)
        if self.with_actions:
            act = torch.from_numpy(action_chunks(ep, idx))
            if self.mode == "train":  # spec 14.3: drop actions with probability 0.5 so "unknown" is in-distribution
                drop = torch.rand(len(idx)) < load_cfg().actions.train_drop_prob
                act[drop] = 0.0
                act[drop, ACTION_TOKEN_DIM - 1] = 1.0
        else:
            act = torch.zeros(len(idx), 0)
        z = torch.zeros(0)
        return frames, act, z, z, z
