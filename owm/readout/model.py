"""Readout head (spec 9.10). Same code for every condition; only the evidence projection differs.

Differences from the reference listing in the spec (all additive):
  * the identity table has 33 rows: 0..31 are GT object ids, 32 is the arm;
  * evidence tokens may carry an identity (the GT-future evidence of the ceiling does, so future objects
    can be tied to the current candidates); predicted slots / particles use the "no id" row (index 33).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ARM_ID, NO_ID = 32, 33


def fourier(uv: torch.Tensor, n_freq: int = 16) -> torch.Tensor:
    """uv [..., 2] in [0, 1] -> [..., 4 * n_freq]."""
    freqs = (2.0 ** torch.arange(n_freq, device=uv.device, dtype=uv.dtype)) * math.pi
    x = uv[..., None] * freqs
    return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)


class Readout(nn.Module):
    # token types: 0 candidate object, 1 arm, 2 goal, 3 evidence
    def __init__(self, d_cand, d_goal, d_evid, n_options, d=256, layers=2, heads=4, ffn=1024,
                 dropout=0.1, n_freq=16, max_steps=8):
        super().__init__()
        self.n_freq = n_freq
        self.id_emb = nn.Embedding(34, 16)
        self.proj_cand = nn.Linear(d_cand + 16 + 4 * n_freq, d)
        self.proj_goal = nn.Linear(d_goal, d)
        self.proj_evid = nn.Linear(d_evid + 16 + 4 * n_freq, d) if d_evid else None
        self.type_emb = nn.Embedding(4, d)
        self.step_emb = nn.Embedding(max_steps + 1, d)  # 1, 2, ... = future step
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.label_mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, n_options))
        self.ptr_mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, b: SimpleNamespace):
        """b.cand [B,Nc,d_cand], b.cand_id [B,Nc], b.cand_uv [B,Nc,2], b.cand_type [B,Nc] (0 object / 1 arm),
        b.cand_pad [B,Nc] (True = padding), b.goal [B,d_goal], b.option_avail [B,n_options] bool,
        evidence (optional): b.evid [B,Ne,d_evid], b.evid_id [B,Ne], b.evid_uv [B,Ne,2], b.evid_step [B,Ne], b.evid_pad."""
        x_c = torch.cat([b.cand, self.id_emb(b.cand_id), fourier(b.cand_uv, self.n_freq)], -1)
        h_c = self.proj_cand(x_c) + self.type_emb(b.cand_type)
        h_g = (self.proj_goal(b.goal) + self.type_emb.weight[2])[:, None]
        tokens, pads = [h_g, h_c], [torch.zeros_like(b.cand_pad[:, :1]), b.cand_pad]
        if self.proj_evid is not None:
            x_e = torch.cat([b.evid, self.id_emb(b.evid_id), fourier(b.evid_uv, self.n_freq)], -1)
            h_e = self.proj_evid(x_e) + self.type_emb.weight[3] + self.step_emb(b.evid_step)
            tokens.append(h_e)
            pads.append(b.evid_pad)
        pad = torch.cat(pads, 1)
        h = self.encoder(torch.cat(tokens, 1), src_key_padding_mask=pad)
        pooled, _ = self.pool(h[:, :1], h, h, key_padding_mask=pad)
        label_logits = self.label_mlp(pooled[:, 0]).masked_fill(~b.option_avail, -1e9)
        h_cand = h[:, 1:1 + b.cand.shape[1]]
        ptr_logits = self.ptr_mlp(h_cand).squeeze(-1)
        ptr_logits = ptr_logits.masked_fill(b.cand_pad | (b.cand_type == 1), -1e9)
        return label_logits, ptr_logits


def readout_loss(label_logits, ptr_logits, y_label, y_obj, need_param_table, pointer_only=False):
    need = need_param_table[y_label].float()
    l_ptr = F.cross_entropy(ptr_logits, y_obj.clamp(min=0), reduction="none")
    l_ptr = (l_ptr * need).sum() / need.sum().clamp(min=1.0)
    if pointer_only:  # prediction check: the label is given
        return l_ptr
    return F.cross_entropy(label_logits, y_label) + l_ptr


@torch.no_grad()
def decode(label_probs, ptr_probs, need_param_table):
    """Inputs may be probabilities averaged over samples / seeds."""
    label = label_probs.argmax(-1)
    obj = ptr_probs.argmax(-1)
    obj = torch.where(need_param_table[label], obj, torch.full_like(obj, -1))
    return label, obj


def is_correct(pred_label, pred_obj, y_label, y_obj, need_param_table):
    need = need_param_table[y_label]
    return (pred_label == y_label) & (~need | (pred_obj == y_obj))


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
