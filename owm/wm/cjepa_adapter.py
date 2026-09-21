"""C-JEPA adapter (spec 8.1): frames <= t -> VideoSAUR slots -> predictor -> future slots -> decoded centroids.

Nothing is trained here. The adapter slices the history itself (`history_indices`), so no frame after t can reach
either network; `future_perturb.py` verifies this end to end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from owm.config import load_cfg, output_dir
from owm.data.sampling import history_indices
from owm.wm.cjepa_predictor_ext import build_predictor
from owm.wm.videosaur_utils import VideoSaur


class CJepaAdapter:
    name = "cjepa"
    n_samples = 1  # the predictor is deterministic

    def __init__(self, predictor_ckpt: str | Path | None = None, videosaur_ckpt: str | Path | None = None,
                 device: str = "cuda", with_actions: bool = False):
        cfg = load_cfg()
        self.cfg, self.device, self.with_actions = cfg, device, with_actions
        self.vs = VideoSaur(videosaur_ckpt, device=device)
        tag = "actions" if with_actions else "main"
        # best = lowest validation future-MSE; the last epoch can be over-fitted (it was, 3x worse, in the first run)
        ck = Path(predictor_ckpt) if predictor_ckpt else output_dir() / "cjepa_predictor" / tag / "best_predictor.ckpt"
        if with_actions:
            from owm.wm.cjepa_action import build_action_predictor
            self.model = build_action_predictor(self.vs.num_slots, cfg, device)
            self.name = "cjepa_act"
        else:
            self.model = build_predictor(self.vs.num_slots, cfg, device)
        self.model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        self.model.eval().requires_grad_(False)
        self.ckpt = ck

    @torch.no_grad()
    def predict(self, frames: np.ndarray, t: int, seed: int = 0, actions: np.ndarray | None = None):
        """frames: uint8 [T, 256, 256, 3] (may contain frames after t — they are never touched).
        actions: [len(history), 130] action tokens of the history frames (action variant only).
        Returns tokens [1, F, N, D], uv [1, F, N, 2], valid [1, F, N]."""
        idx = history_indices(t)
        slots = self.vs.encode(np.asarray(frames[idx]), seed=seed)                 # [T_h, N, D], causal
        x = torch.from_numpy(slots)[None].to(self.device)
        if self.with_actions:
            from owm.wm.actions import unknown_actions
            F_ = self.model.pred_frames
            act = np.concatenate([actions, unknown_actions(F_)], 0)               # future actions are never given
            pred = self.model.inference(x, torch.from_numpy(act)[None].to(self.device))
        else:
            pred = self.model.inference(x)
        pred = pred[0].float().cpu().numpy()                                        # [F, N, D]
        uv, _ = self.vs.centroids(pred)
        return pred[None], uv[None], np.ones(pred.shape[:2], bool)[None]

    def describe(self) -> str:
        return json.dumps(dict(videosaur=str(self.vs.ckpt), predictor=str(self.ckpt), num_slots=self.vs.num_slots))
