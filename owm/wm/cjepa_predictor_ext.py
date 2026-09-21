"""Variable-length history for the STOCK C-JEPA predictor (spec 17 item 10).

Findings on the stock code (cjepa/src/cjepa_predictor.py):
  * `time_pos_embed` is a learned table of length history_frames + pred_frames -> W = 80, F = 2 just works
    (82 x N tokens, full attention); there is no other max-length constant.
  * there is NO padding-mask support, and the stock from-slot trainer drops every video shorter than W + F
    sub-sampled frames. This was asked for the spec's W = 80: at stride 16 only 0.8% of the 24128 training
    sequences reach 82 frames (max 89), i.e. ~500 clips — effectively nothing. `context_frames` later dropped
    to 32 (LPWM memory, spec 7.2), where the stock sampler would keep 26% of the sequences / 124k clips, so the
    argument is no longer "zero samples" but "74% of the data thrown away and short histories never trained".
  * the stock `inference()` already accepts a shorter history by RIGHT-ALIGNING it on the time table
    (`time_pos_embed[:, -T_total:]`) with the earliest given frame as the identity anchor.

So instead of front-padding + masking we train exactly the way the stock model infers: histories of variable
length T_h <= W, right-aligned, anchor = earliest real frame. A batch holds clips of one length, hence no padding
token ever exists. Parameters and state_dict are identical to `MaskedSlotPredictor` (this class only adds methods).

Also: the stock attention call uses need_weights=True (materialises B x L x L maps; ~1k tokens here). We call the
same nn.MultiheadAttention modules with need_weights=False — same maths, fused kernel.

Object-level masking follows the official code: `num_masked_slots` slot indices are hidden at every history frame
except the anchor. The official `get_mask_indices` re-creates its RNG from a fixed seed on every call, i.e. the
masked indices never change (slot order is random per video, so they are still random objects). `mask_sampling:
random` in experiment.yaml resamples them every step instead.
"""
from __future__ import annotations

import numpy as np
import torch
from einops import rearrange

from owm.config import add_repo_to_path

add_repo_to_path("cjepa")
from src.cjepa_predictor import MaskedSlotPredictor  # noqa: E402


class VarLenMaskedSlotPredictor(MaskedSlotPredictor):
    def __init__(self, *args, mask_sampling: str = "fixed", **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_sampling = mask_sampling
        self._rng = np.random.RandomState(self.seed)

    def get_mask_indices(self, batch_size, device):
        if self.mask_sampling == "fixed":
            return super().get_mask_indices(batch_size, device)
        idx = self._rng.choice(self.num_slots, self.num_masked_slots, replace=False)
        is_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        is_masked[idx] = True
        return is_masked, torch.from_numpy(idx).to(device)

    def _transformer(self, x):
        for attn, ff in self.transformer.layers:
            x = x + attn(x, x, x, need_weights=False)[0]
            x = x + ff(x)
        return self.transformer.norm(x)

    def _run(self, x, masked_indices):
        """x [B, T_h, S, D]; right-aligned on the time table; returns [B, T_h + F, S, D]."""
        B, T_h, S, D = x.shape
        T = T_h + self.pred_frames
        assert T <= self.total_frames, f"history {T_h} + {self.pred_frames} exceeds the time table {self.total_frames}"
        pos = self.time_pos_embed[:, -T:]
        query = self.mask_token + pos + self.id_projector(x[:, 0])[:, None]      # [B, T, S, D]
        tokens = query.clone()
        tokens[:, :T_h] = x + pos[:, :T_h]
        if len(masked_indices) and T_h > 1:
            tokens[:, 1:T_h, masked_indices] = query[:, 1:T_h, masked_indices]   # anchor (t = 0) stays visible
        out = self._transformer(rearrange(tokens, "b t s d -> b (t s) d"))
        return self.to_out(rearrange(out, "b (t s) d -> b t s d", t=T, s=S))

    def forward(self, x):
        """Training pass with object-level masking. Returns (prediction over history+future, masked slot indices)."""
        if self.num_masked_slots > 0:
            _, masked = self.get_mask_indices(x.shape[0], x.device)
        else:
            masked = torch.tensor([], dtype=torch.long, device=x.device)
        return self._run(x, masked), masked

    @torch.no_grad()
    def inference(self, x):
        """Full history visible, future = mask tokens -> [B, F, S, D]. Equals the stock inference()."""
        out = self._run(x, torch.tensor([], dtype=torch.long, device=x.device))
        return out[:, x.shape[1]:]


def unwrap(model):
    """The underlying module — `model` may be a DistributedDataParallel wrapper, which does not proxy attributes."""
    return getattr(model, "module", model)


def cjepa_loss(model: VarLenMaskedSlotPredictor, clip: torch.Tensor) -> dict:
    """Official loss (train_causalwm_from_clevrer_slot.compute_loss): MSE on the masked history slots
    (anchor frame included, as in the official code) + MSE on all future slots."""
    F_ = unwrap(model).pred_frames
    hist, target = clip[:, :-F_], clip[:, -F_:]
    pred, masked = model(hist)
    T_h = hist.shape[1]
    loss_future = torch.nn.functional.mse_loss(pred[:, T_h:], target)
    if len(masked):
        loss_hist = torch.nn.functional.mse_loss(pred[:, :T_h][:, :, masked], hist[:, :, masked])
    else:
        loss_hist = pred.new_zeros(())
    return dict(loss=loss_hist + loss_future, loss_future=loss_future, loss_masked_history=loss_hist)


def build_predictor(num_slots: int, cfg, device="cuda") -> VarLenMaskedSlotPredictor:
    c = cfg.cjepa
    m = c.num_masked_slots if c.num_masked_slots is not None else num_slots // 4
    p = c.predictor
    W, F_ = cfg.temporal.context_frames, len(cfg.temporal.future_offsets)   # single source of truth (spec section 6)
    return VarLenMaskedSlotPredictor(
        num_slots=num_slots, slot_dim=c.slot_dim, history_frames=W, pred_frames=F_,
        num_masked_slots=m, seed=42, depth=p.depth, heads=p.heads, dim_head=p.dim_head, mlp_dim=p.mlp_dim,
        dropout=p.dropout, mask_sampling=c.mask_sampling).to(device)
