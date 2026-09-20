"""Thin wrappers around the stock VideoSAUR model of the C-JEPA repo (nothing there is modified).

Verified (spec 17 items 8, 9):
  * slots are strictly causal: encoder = per-frame DINOv2 (MapOverTime), processor = ScanOverTime with
    slots_t = SlotAttention(Predictor(slots_{t-1}), features_t); no temporal attention, no batch-norm.
  * the SlotMixer decoder can be called on arbitrary slots [B, S, D] and returns per-slot alpha masks over
    the 18 x 18 patch grid (`masks` [B, S, P], softmax over slots), so predicted slots can be decoded.
  * the slot initialiser samples noise even in eval mode -> we seed torch before every sequence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from owm.config import ROOT, add_repo_to_path, load_cfg, resolve


def _imports():
    add_repo_to_path("cjepa")
    shim = str(ROOT / "owm" / "shims")
    if shim not in sys.path:
        sys.path.append(shim)  # only used if the real torchcodec is absent
    from src.third_party.videosaur.videosaur import configuration, models
    from src.third_party.videosaur.videosaur.data import transforms
    return configuration, models, transforms


def find_checkpoint(path: str | Path | None = None) -> Path:
    """A .ckpt file, or a videosaur log dir (latest checkpoint), or None -> newest run in outputs/videosaur."""
    from owm.config import output_dir
    p = Path(path) if path else output_dir() / "videosaur"
    if p.is_file():
        return p
    cks = sorted(p.rglob("*.ckpt"), key=lambda f: f.stat().st_mtime)
    if not cks:
        raise FileNotFoundError(f"no VideoSAUR checkpoint under {p}")
    return cks[-1]


class VideoSaur:
    def __init__(self, ckpt: str | Path | None = None, config: str | Path | None = None, device: str = "cuda"):
        configuration, models, transforms = _imports()
        self.ckpt = find_checkpoint(ckpt)
        settings = self.ckpt.parent.parent / "settings.yaml"       # written by the stock trainer
        cfg_path = config or (settings if settings.exists() else resolve(load_cfg().cjepa.videosaur_config))
        state = torch.load(self.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        conf = configuration.load_config(str(cfg_path))
        self.conf = conf
        self.model = models.build(conf.model, conf.optimizer)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        bad = [k for k in missing if "backbone" not in k]   # the frozen DINOv2 backbone is re-loaded from HF anyway
        assert not bad, f"checkpoint does not match the config, missing keys: {bad[:5]}"
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.transform = transforms.build(conf.dataset.val_pipeline.transforms)["video"]
        self.num_slots = int(conf.globals.NUM_SLOTS)
        self.grid = int(round(float(conf.globals.NUM_PATCHES) ** 0.5))

    @torch.no_grad()
    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """uint8 [T, 256, 256, 3] -> float [T, 3, 252, 252], stock val transform (bicubic resize + ImageNet norm)."""
        return self.transform(np.ascontiguousarray(frames))

    @torch.no_grad()
    def encode(self, frames: np.ndarray, seed: int = 0, chunk: int = 64) -> np.ndarray:
        """Slots of a frame sequence, [T, N, D]. Frame t's slots depend on frames <= t only."""
        video = self.preprocess(frames)
        feats = []
        for i in range(0, len(video), chunk):  # the frame encoder is per-frame, chunking changes nothing
            out = self.model.encoder(video[None, i:i + chunk].to(self.device))
            feats.append(out["features"])
        feats = torch.cat(feats, 1)
        gen_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        slots0 = self.model.initializer(batch_size=1).to(self.device)
        torch.random.set_rng_state(gen_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        out = self.model.processor(slots0, feats)
        return out["state"][0].float().cpu().numpy()

    @torch.no_grad()
    def decode_masks(self, slots: np.ndarray | torch.Tensor) -> np.ndarray:
        """slots [..., N, D] -> alpha masks [..., N, g, g] (softmax over the N slots per patch)."""
        s = torch.as_tensor(slots, dtype=torch.float32, device=self.device)
        lead = s.shape[:-2]
        out = self.model.decoder.module(s.reshape(-1, *s.shape[-2:]))
        m = out["masks"].reshape(*lead, s.shape[-2], self.grid, self.grid)
        return m.cpu().numpy()

    def centroids(self, slots) -> tuple[np.ndarray, np.ndarray]:
        """Mask-weighted centroid (u, v) in [0, 1] of every slot [..., N, 2] and its mask mass [..., N]."""
        m = self.decode_masks(slots)
        g = self.grid
        c = (np.arange(g, dtype=np.float32) + 0.5) / g
        mass = m.sum((-1, -2))
        u = (m * c[None, :]).sum((-1, -2)) / np.maximum(mass, 1e-8)
        v = (m * c[:, None]).sum((-1, -2)) / np.maximum(mass, 1e-8)
        return np.stack([u, v], -1).astype(np.float32), (mass / (g * g)).astype(np.float32)
