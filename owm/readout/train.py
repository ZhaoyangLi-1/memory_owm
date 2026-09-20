"""Train one readout head (spec 9.5): one task x one condition x one seed. World models stay frozen —
they only appear here through the cached evidence files."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from owm.config import load_cfg, output_dir
from owm.readout.dataset import DecisionData, is_check
from owm.readout.model import Readout, count_parameters, readout_loss
from owm.wm.evidence import D_GT


def ckpt_path(task: str, condition: str, seed: int) -> Path:
    return output_dir() / "readout" / task / condition / f"seed{seed}.pt"


def build_model(data: DecisionData) -> Readout:
    r = load_cfg().readout
    return Readout(d_cand=D_GT, d_goal=data.d_goal, d_evid=data.d_evid, n_options=len(data.labels), d=r.d_model,
                   layers=r.layers, heads=r.heads, ffn=r.ffn, dropout=r.dropout, n_freq=r.fourier_freqs,
                   max_steps=max(load_cfg().temporal.context_frames, 8))


@torch.no_grad()
def eval_loss(model, data, idx, need, device, pointer_only, bs=256):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        S = data.n_samples_evidence()
        losses = []
        for s in range(S):
            b, y = data.collate(chunk, sample_idx=s, device=device)
            ll, pl = model(b)
            losses.append(readout_loss(ll, pl, y.label, y.cand, need, pointer_only).item())
        tot += float(np.mean(losses)) * len(chunk)
        n += len(chunk)
    return tot / max(n, 1)


def train_one(task: str, condition: str, seed: int, data: DecisionData | None = None, device: str = "cuda",
              verbose: bool = True) -> dict:
    cfg = load_cfg().readout
    data = data or DecisionData(task, condition)
    tr, va = data.split_indices("train"), data.split_indices("val")
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(f"{task}/{condition}: no training or validation decisions (train={len(tr)}, val={len(va)})")
    data.fit_stats(tr)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(data).to(device)
    need = data.need_param.to(device)
    pointer_only = is_check(condition)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    warm, total = cfg.schedule.warmup_steps, cfg.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))

    best, best_step, best_state = float("inf"), 0, None
    for step in range(1, total + 1):
        model.train()
        idx = rng.choice(tr, size=min(cfg.batch_size, len(tr)), replace=len(tr) < cfg.batch_size)
        b, y = data.collate(idx, rng=rng, device=device)
        ll, pl = model(b)
        loss = readout_loss(ll, pl, y.label, y.cand, need, pointer_only)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0:
            v = eval_loss(model, data, va, need, device, pointer_only)
            if v < best - 1e-6:
                best, best_step = v, step
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            if verbose and step % 500 == 0:
                print(f"  [{task}/{condition}/s{seed}] step {step} train {loss.item():.4f} val {v:.4f} (best {best:.4f} @ {best_step})")
            if step - best_step >= cfg.early_stop.patience_steps:
                break
    out = ckpt_path(task, condition, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=best_state, stats=data.stats, labels=data.labels, d_goal=data.d_goal, d_evid=data.d_evid,
                    task=task, condition=condition, seed=seed, best_val_loss=best, best_step=best_step,
                    n_params=count_parameters(model), n_train=len(tr), n_val=len(va)), out)
    return dict(task=task, condition=condition, seed=seed, best_val_loss=best, best_step=best_step,
                n_params=count_parameters(model), n_train=len(tr), n_val=len(va))


def load_head(task: str, condition: str, seed: int, data: DecisionData, device: str = "cuda") -> Readout:
    ck = torch.load(ckpt_path(task, condition, seed), map_location="cpu", weights_only=False)
    data.stats = ck["stats"]
    model = build_model(data)
    model.load_state_dict(ck["state"])
    return model.to(device).eval()
