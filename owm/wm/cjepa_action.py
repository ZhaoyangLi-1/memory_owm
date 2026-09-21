"""Optional experiment (spec 14.3): C-JEPA with actions as an auxiliary variable.

The official Push-T variant (MaskedSlot_AP_Predictor) appends action / proprio as extra slot tokens per frame and never
masks them. We do the same with ONE action token per frame (no proprio): slot index N holds a linear embedding of the
130-d action token. History action tokens are inputs; future action tokens are the "unknown action" embedding, never
the real future actions (the decision at t IS the future action). The loss ignores the action slot.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from owm.wm.actions import ACTION_TOKEN_DIM
from owm.wm.cjepa_predictor_ext import VarLenMaskedSlotPredictor, unwrap


class ActionVarLenPredictor(VarLenMaskedSlotPredictor):
    def __init__(self, num_object_slots: int, **kw):
        super().__init__(num_slots=num_object_slots + 1, **kw)
        self.num_object_slots = num_object_slots
        self.action_proj = nn.Linear(ACTION_TOKEN_DIM, self.slot_dim)

    def get_mask_indices(self, batch_size, device):
        # only object slots can be masked (the official AP predictor also excludes the auxiliary tokens)
        n = self.num_slots
        self.num_slots = self.num_object_slots
        try:
            return super().get_mask_indices(batch_size, device)
        finally:
            self.num_slots = n

    def _with_actions(self, x, act):
        """x [B, T_h, N, D], act [B, T_h + F, 130] -> tokens [B, T_h + F, N + 1, D] with the action slot filled."""
        B, T_h, N, D = x.shape
        T = T_h + self.pred_frames
        pos = self.time_pos_embed[:, -T:]
        a = self.action_proj(act)[:, :, None]                                      # [B, T, 1, D]
        anchor = torch.cat([x[:, 0], a[:, 0]], 1)                                  # [B, N + 1, D]
        query = self.mask_token + pos + self.id_projector(anchor)[:, None]
        tokens = query.clone()
        tokens[:, :T_h, :N] = x + pos[:, :T_h]
        tokens[:, :, N:] = a + pos                                                 # action tokens are always inputs
        return tokens, query

    def _run_act(self, x, act, masked):
        from einops import rearrange
        B, T_h, N, D = x.shape
        T = T_h + self.pred_frames
        tokens, query = self._with_actions(x, act)
        if len(masked) and T_h > 1:
            tokens[:, 1:T_h, masked] = query[:, 1:T_h, masked]
        out = self._transformer(rearrange(tokens, "b t s d -> b (t s) d"))
        return self.to_out(rearrange(out, "b (t s) d -> b t s d", t=T, s=N + 1))[:, :, :N]

    def forward(self, x, act):
        if self.num_masked_slots > 0:
            _, masked = self.get_mask_indices(x.shape[0], x.device)
        else:
            masked = torch.tensor([], dtype=torch.long, device=x.device)
        return self._run_act(x, act, masked), masked

    @torch.no_grad()
    def inference(self, x, act):
        return self._run_act(x, act, torch.tensor([], dtype=torch.long, device=x.device))[:, x.shape[1]:]


def cjepa_action_loss(model: ActionVarLenPredictor, clip, act) -> dict:
    F_ = unwrap(model).pred_frames
    hist, target = clip[:, :-F_], clip[:, -F_:]
    pred, masked = model(hist, act)
    T_h = hist.shape[1]
    lf = torch.nn.functional.mse_loss(pred[:, T_h:], target)
    lh = torch.nn.functional.mse_loss(pred[:, :T_h][:, :, masked], hist[:, :, masked]) if len(masked) else pred.new_zeros(())
    return dict(loss=lf + lh, loss_future=lf, loss_masked_history=lh)


def build_action_predictor(num_slots: int, cfg, device="cuda") -> ActionVarLenPredictor:
    c, p = cfg.cjepa, cfg.cjepa.predictor
    m = c.num_masked_slots if c.num_masked_slots is not None else num_slots // 4
    return ActionVarLenPredictor(
        num_object_slots=num_slots, slot_dim=c.slot_dim, history_frames=cfg.temporal.context_frames,
        pred_frames=len(cfg.temporal.future_offsets), num_masked_slots=m, seed=42, depth=p.depth, heads=p.heads,
        dim_head=p.dim_head, mlp_dim=p.mlp_dim, dropout=p.dropout, mask_sampling=c.mask_sampling).to(device)
