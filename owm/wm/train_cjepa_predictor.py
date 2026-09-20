"""P5d: train the C-JEPA predictor from saved slots (W = context_frames, F = len(future_offsets)).

The stock trainer (src/train/train_causalwm_from_clevrer_slot.py) cannot be used as-is: it needs hydra /
stable-pretraining / stable-worldmodel, and it silently drops every video shorter than W + F sub-sampled frames —
at stride 16 that is every RoboMME episode. This trainer keeps the official model, loss (masked-history MSE +
future MSE), optimiser (AdamW, lr 5e-4, no scheduler) and checkpoint format (plain state_dict), and only changes
the clip sampling: a clip is the first T_h + F frames of a stride-16 sequence (T_h = 1 .. min(W, L - F)), exactly
the histories the model sees at evaluation time. All clips of a batch share T_h, so no padding is needed.

  python -m owm.wm.train_cjepa_predictor [--slots cache/slots/wm_train_slots.pkl] [--actions] [--tag main]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from owm.config import cache_dir, load_cfg, output_dir
from owm.wm.actions import ACTION_DIM, sequence_actions
from owm.wm.cjepa_predictor_ext import build_predictor, cjepa_loss


class SlotClips:
    def __init__(self, seqs: dict, W: int, F: int, with_actions: bool, rng: np.random.Generator):
        self.keys = [k for k, v in seqs.items() if len(v) >= F + 1]
        self.seqs = [torch.from_numpy(np.asarray(seqs[k], dtype=np.float32)) for k in self.keys]
        self.len = np.array([len(s) for s in self.seqs])
        self.W, self.F, self.rng = W, F, rng
        self.max_th = np.minimum(self.len - F, W)
        # number of clips per history length: a sequence of length L offers T_h = 1..min(W, L-F)
        self.th_weights = np.array([(self.max_th >= th).sum() for th in range(1, W + 1)], dtype=np.float64)
        self.actions = None
        if with_actions:
            self.actions = [torch.from_numpy(sequence_actions(k)) for k in self.keys]

    def batch(self, B: int, th: int | None = None, drop_prob: float = 0.0):
        if th is None:
            th = int(self.rng.choice(np.arange(1, self.W + 1), p=self.th_weights / self.th_weights.sum()))
        ok = np.where(self.max_th >= th)[0]
        idx = self.rng.choice(ok, size=B, replace=len(ok) < B)
        clip = torch.stack([self.seqs[i][: th + self.F] for i in idx])
        if self.actions is None:
            return clip, None
        act = torch.stack([self.actions[i][: th + self.F].clone() for i in idx])      # [B, T, 128 + 2]
        act[:, th:] = 0.0
        act[:, th:, ACTION_DIM + 1] = 1.0                                          # future actions are unknown
        if drop_prob > 0:
            drop = torch.from_numpy(self.rng.random((B, th)) < drop_prob)
            a = act[:, :th]
            a[drop] = 0.0
            a[..., ACTION_DIM + 1][drop] = 1.0
        return clip, act

    def val_batches(self, B: int, every: int = 4):
        for th in range(1, self.W + 1, every):
            ok = np.where(self.max_th >= th)[0]
            for i in range(0, len(ok), B):
                idx = ok[i:i + B]
                clip = torch.stack([self.seqs[j][: th + self.F] for j in idx])
                act = None
                if self.actions is not None:
                    act = torch.stack([self.actions[j][: th + self.F].clone() for j in idx])
                    act[:, th:] = 0.0
                    act[:, th:, ACTION_DIM + 1] = 1.0
                yield clip, act


def run_dir(tag: str) -> Path:
    return output_dir() / "cjepa_predictor" / tag


def main():
    cfg = load_cfg()
    tc = cfg.cjepa.train
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default=str(cache_dir() / "slots" / "wm_train_slots.pkl"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--actions", action="store_true", help="optional experiment (spec 14): action token per frame")
    ap.add_argument("--epochs", type=int, default=tc.max_epochs)
    ap.add_argument("--steps-per-epoch", type=int, default=tc.steps_per_epoch)
    ap.add_argument("--batch-size", type=int, default=tc.batch_size)
    ap.add_argument("--lr", type=float, default=tc.lr)
    ap.add_argument("--seed", type=int, default=tc.seed)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    with open(a.slots, "rb") as f:
        data = pickle.load(f)
    N = int(next(iter(data["train"].values())).shape[1])
    W, F = cfg.temporal.context_frames, len(cfg.temporal.future_offsets)
    train = SlotClips(data["train"], W, F, a.actions, rng)
    val = SlotClips(data["val"], W, F, a.actions, np.random.default_rng(0))
    print(f"slots N={N}; train sequences={len(train.keys)} val sequences={len(val.keys)}; "
          f"history lengths 1..{int(train.max_th.max())} (W={W}, F={F})")

    if a.actions:
        from owm.wm.cjepa_action import build_action_predictor, cjepa_action_loss
        model, loss_fn = build_predictor_with_actions(N, cfg, a.device, build_action_predictor, cjepa_action_loss)
    else:
        model = build_predictor(N, cfg, a.device)
        loss_fn = lambda m, clip, act: cjepa_loss(m, clip)  # noqa: E731
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    out = run_dir(a.tag)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(dict(vars(a), num_slots=N, W=W, F=F, slot_meta=data.get("meta", {})), indent=2, default=str))
    log, best, step = [], float("inf"), 0
    for epoch in range(a.epochs):
        model.train()
        t0, run = time.time(), []
        for _ in range(a.steps_per_epoch):
            clip, act = train.batch(a.batch_size, drop_prob=cfg.actions.train_drop_prob if a.actions else 0.0)
            l = loss_fn(model, clip.to(a.device), None if act is None else act.to(a.device))
            opt.zero_grad(set_to_none=True)
            l["loss"].backward()
            opt.step()
            run.append([l["loss"].item(), l["loss_future"].item(), l["loss_masked_history"].item()])
            step += 1
        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for clip, act in val.val_batches(a.batch_size):
                clip = clip.to(a.device)
                act = None if act is None else act.to(a.device)
                pred = model.inference(clip[:, :-F]) if act is None else model.inference(clip[:, :-F], act)
                vs += torch.nn.functional.mse_loss(pred, clip[:, -F:], reduction="sum").item()
                vn += pred.numel()
        v = vs / max(vn, 1)
        tr = np.mean(run, 0)
        log.append(dict(epoch=epoch, step=step, train_loss=tr[0], train_future=tr[1], train_masked=tr[2], val_future_mse=v))
        print(f"epoch {epoch} step {step} train {tr[0]:.5f} (future {tr[1]:.5f} masked {tr[2]:.5f}) val_future {v:.5f} {time.time() - t0:.0f}s")
        torch.save(model.state_dict(), out / f"epoch_{epoch}_predictor.ckpt")
        if v < best:
            best = v
            torch.save(model.state_dict(), out / "best_predictor.ckpt")
        (out / "log.json").write_text(json.dumps(log, indent=2))
    torch.save(model.state_dict(), out / "final_predictor.ckpt")


def build_predictor_with_actions(N, cfg, device, builder, loss):
    model = builder(N, cfg, device)
    return model, lambda m, clip, act: loss(m, clip, act)


if __name__ == "__main__":
    main()
