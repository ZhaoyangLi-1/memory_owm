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
        raise SystemExit(f"{p} 下没有 VideoSAUR checkpoint。先跑 5b：\n"
                         f"  CUDA_VISIBLE_DEVICES=2,3,4 bash scripts/p5b_train_videosaur.sh")
    return cks[-1]


def align_slots(slots: np.ndarray) -> np.ndarray:
    """Re-order the slots of every frame so that index i is the same object over time, [T, N, D] -> [T, N, D].

    The trained model permutes its slot indices at every step (measured on the val slots: slot i of frame t is the
    nearest neighbour of slot i of frame t+1 in 7% of the cases, of frame t+2 in 87%; after re-matching the
    same-slot cosine is 0.99), and the C-JEPA predictor masks / identifies objects BY INDEX. Frame t is matched to
    the already aligned frame t-1 (Hungarian on cosine similarity), so it still depends on frames <= t only, and
    frame 0 keeps its order. Only the order changes, never a value."""
    from scipy.optimize import linear_sum_assignment
    out = np.array(slots, copy=True)
    prev = out[0].astype(np.float32)
    prev /= np.maximum(np.linalg.norm(prev, axis=-1, keepdims=True), 1e-8)
    for t in range(1, len(out)):
        cur = out[t].astype(np.float32)
        cur /= np.maximum(np.linalg.norm(cur, axis=-1, keepdims=True), 1e-8)
        _, col = linear_sum_assignment(-(prev @ cur.T))
        out[t], prev = out[t][col], cur[col]
    return out


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
        """Slots of a frame sequence, [T, N, D], index-aligned over time (`align_slots`). Frame t's slots depend on
        frames <= t only."""
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
        return align_slots(out["state"][0].float().cpu().numpy())

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
