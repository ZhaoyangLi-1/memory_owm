"""LPWM adapter (spec 8.2): history frames -> particles -> 2-step autoregressive rollout with latent actions
sampled from the PRIOR only, 5 samples with fixed seeds.

Verified on the LPWM code (spec 17 item 11):
  * `DLP.sample_from_x(x, cond_steps=T_c, use_all_ctx=False, ...)` encodes only x[:, :T_c]; the latent actions of the
    history transitions come from the posterior (inverse model over history frames, causal), and every rollout step
    draws its latent action from the context PRIOR head (`z_context_dyn`), which sees past particles only.
  * the single leak path is `use_all_ctx=True` (posterior contexts from ground-truth future frames; it is the default of
    the repo's animation / metric code). It is never used here — and we pass ONLY frames <= t, so a leak is impossible
    by construction.
  * particle encoding is per frame (no temporal interaction, GroupNorm / RMSNorm only); context and dynamics
    transformers are causal. `deterministic=False` samples the latent-action prior with the global torch RNG, so the
    seed is set immediately before each call.
  * keypoints `z_pos` are (y, x) in [-1, 1]  ->  u = (x + 1) / 2, v = (y + 1) / 2.
  * the model can condition on at most timestep_horizon + 1 frames; the dynamics window is timestep_horizon.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from owm.config import load_cfg, output_dir
from owm.data.sampling import history_indices
from owm.wm.lpwm_dataset import to_lpwm_frames
from owm.wm.lpwm_train import lpwm_sys_path


def find_run(run: str | Path | None, with_actions: bool) -> Path:
    root = Path(run) if run else output_dir() / ("lpwm_act" if with_actions else "lpwm")
    if (root / "hparams.json").exists():
        return root
    runs = sorted((p.parent for p in root.glob("*/hparams.json")), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no LPWM run (hparams.json) under {root}")
    return runs[-1]


def load_lpwm(run_dir: Path, ckpt: str | Path | None, device: str):
    lpwm_sys_path()
    from generate_lpwm_video_prediction import load_dlp_from_config
    if ckpt is None:
        cands = [p for p in (run_dir / "saves").glob("*.pth") if "best" not in p.name]
        ckpt = sorted(cands, key=lambda p: p.stat().st_mtime)[-1]   # latest checkpoint, not picked by any downstream score
    model = load_dlp_from_config(str(run_dir / "hparams.json"), str(ckpt))
    # the repo's loader swallows load errors; load again strictly so a mismatch cannot go unnoticed
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=True)
    return model.to(device).eval().requires_grad_(False), Path(ckpt)


class LpwmAdapter:
    name = "lpwm"

    def __init__(self, run: str | Path | None = None, ckpt: str | Path | None = None, device: str = "cuda",
                 with_actions: bool = False, batched_samples: bool = False):
        cfg = load_cfg()
        self.cfg, self.device, self.with_actions = cfg, device, with_actions
        self.run_dir = find_run(run, with_actions)
        self.model, self.ckpt = load_lpwm(self.run_dir, ckpt, device)
        self.hp = json.loads((self.run_dir / "hparams.json").read_text())
        self.image_size = int(self.hp["image_size"])
        self.max_cond = int(self.hp["timestep_horizon"]) + 1
        self.seeds = list(cfg.lpwm.rollout_seeds)[: cfg.lpwm.rollout_samples]
        self.n_samples = len(self.seeds)
        self.batched_samples = batched_samples
        self.n_future = len(cfg.temporal.future_offsets)
        if with_actions:
            self.name = "lpwm_act"

    @torch.no_grad()
    def _rollout(self, hist: torch.Tensor, actions, seed: int):
        kw = {}
        if self.with_actions:
            kw["actions"] = actions
        torch.manual_seed(seed)   # nothing may consume RNG between this line and the call
        _, z = self.model.sample_from_x(hist, num_steps=self.n_future, deterministic=False, cond_steps=hist.shape[1],
                                        use_all_ctx=False, return_z=True, decode=False, n_pred_eq_gt=False, **kw)
        return z

    @torch.no_grad()
    def predict(self, frames: np.ndarray, t: int, seed: int = 0, actions: np.ndarray | None = None):
        """frames uint8 [T, 256, 256, 3]; only frames <= t are read.
        Returns tokens [S, F, M, D_p], uv [S, F, M, 2], valid [S, F, M]; sample s uses torch seed seeds[s] + seed."""
        idx = history_indices(t)[-self.max_cond:]
        hist = to_lpwm_frames(np.asarray(frames[idx]), self.image_size)[None].to(self.device)
        act = None
        if self.with_actions:
            from owm.wm.actions import unknown_actions
            a = np.concatenate([actions[-len(idx):], unknown_actions(self.n_future)], 0)   # future actions never given
            act = torch.from_numpy(a)[None].to(self.device)
        if self.batched_samples:
            S = self.n_samples
            zs = [self._rollout(hist.expand(S, *hist.shape[1:]).contiguous(),
                                None if act is None else act.expand(S, *act.shape[1:]).contiguous(), self.seeds[0] + seed)]
        else:
            zs = [self._rollout(hist, act, s + seed) for s in self.seeds]
        toks, uvs = [], []
        for z in zs:
            F_ = self.n_future
            pos = z["z_pos"][:, -F_:].float()                                     # (y, x) in [-1, 1]
            parts = [pos, z["z_scale"][:, -F_:], z["z_depth"][:, -F_:], z["z_obj_on"][:, -F_:], z["z_features"][:, -F_:]]
            parts = [p.float() if p.dim() == 4 else p.float()[..., None] for p in parts]
            toks.append(torch.cat(parts, -1))
            uvs.append(torch.stack([(pos[..., 1] + 1) / 2, (pos[..., 0] + 1) / 2], -1))
        tokens = torch.cat(toks, 0).cpu().numpy()
        uv = np.clip(torch.cat(uvs, 0).cpu().numpy(), 0.0, 1.0)
        return tokens, uv, np.ones(tokens.shape[:3], bool)

    def describe(self) -> str:
        return json.dumps(dict(run=str(self.run_dir), ckpt=str(self.ckpt), image_size=self.image_size,
                               timestep_horizon=self.max_cond - 1, seeds=self.seeds))
