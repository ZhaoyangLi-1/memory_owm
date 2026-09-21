"""P5d: train the C-JEPA predictor from saved slots (W = context_frames, F = len(future_offsets)).

The stock trainer (src/train/train_causalwm_from_clevrer_slot.py) cannot be used as-is: it needs hydra /
stable-pretraining / stable-worldmodel, and its dataset (ClevrerSlotDataset) is a sliding window of FIXED length
W + F that silently drops every sequence shorter than that. Measured on the 24128 stride-16 training sequences
(length: median 27, mean 30.3, max 89), at W = 32 / F = 4 that keeps 26% of them and yields 124k clips — and it
would only ever train a full-length W history, while at evaluation the median history is 5 .. 21 frames
depending on the task.

This trainer keeps the official model, loss (masked-history MSE + future MSE), optimiser (AdamW, lr 5e-4, no
scheduler) and checkpoint format (plain state_dict), and only changes the clip sampling: a clip is the first
T_h + F frames of a stride-16 sequence (T_h = 1 .. min(W, L - F)) — 516k clips covering every history length.
All clips of a batch share T_h, so no padding is needed.

The clip is a PREFIX, which is what `history_indices(t)` returns for 94% of the memory-critical decisions; the
other 6% (all in PickXtimes) have more than W frames before t, so `[-W:]` truncates them to a window that does
not start at the episode beginning — those are trained slightly off-distribution. Deliberate: a mid-episode
window does not contain the conditioning video this benchmark asks the model to remember.

An epoch is one full pass over the clips (every sequence x every history length T_h, shuffled, each clip once);
`--steps-per-epoch N` switches back to N randomly sampled batches.

Progress: a tqdm bar per epoch and the per-step losses in wandb (step_loss / step_future_mse /
step_masked_history_mse), both on rank 0 only; `--log-every N` thins the wandb points.

  单卡:  python -m owm.wm.train_cjepa_predictor [--slots ...] [--actions] [--tag main]
  多卡:  CUDA_VISIBLE_DEVICES=2,3,4 torchrun --nproc_per_node=3 -m owm.wm.train_cjepa_predictor --tag main
         (DDP：每张卡各取 batch_size 个片段，有效 batch = batch_size x 卡数；学习率按需自行调整)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from owm.config import cache_dir, load_cfg, output_dir
from owm.wandb_utils import Run, flat_config
from owm.wandb_utils import init as wandb_init
from owm.wm.actions import ACTION_DIM, sequence_actions
from owm.wm.cjepa_predictor_ext import build_predictor, cjepa_loss


class SlotClips:
    def __init__(self, seqs: dict, W: int, F: int, with_actions: bool, rng: np.random.Generator,
                 th_rng: np.random.Generator | None = None):
        """`rng` picks which sequences go into a batch (per rank), `th_rng` picks the history length.
        Under DDP every rank must use the SAME history length, otherwise the tensors have different shapes and
        the gradient all-reduce mismatches — so `th_rng` is seeded identically on all ranks."""
        self.keys = [k for k, v in seqs.items() if len(v) >= F + 1]
        self.seqs = [torch.from_numpy(np.asarray(seqs[k], dtype=np.float32)) for k in self.keys]
        self.len = np.array([len(s) for s in self.seqs])
        self.W, self.F, self.rng = W, F, rng
        self.th_rng = th_rng if th_rng is not None else rng
        self.max_th = np.minimum(self.len - F, W)
        # number of clips per history length: a sequence of length L offers T_h = 1..min(W, L-F)
        self.th_weights = np.array([(self.max_th >= th).sum() for th in range(1, W + 1)], dtype=np.float64)
        self.actions = None
        if with_actions:
            self.actions = [torch.from_numpy(sequence_actions(k)) for k in self.keys]

    def make(self, idx, th: int, drop_prob: float = 0.0):
        """The clips (sequence `i`, history length `th`) as one padding-free batch."""
        clip = torch.stack([self.seqs[i][: th + self.F] for i in idx])
        if self.actions is None:
            return clip, None
        act = torch.stack([self.actions[i][: th + self.F].clone() for i in idx])      # [B, T, 128 + 2]
        act[:, th:] = 0.0
        act[:, th:, ACTION_DIM + 1] = 1.0                                          # future actions are unknown
        if drop_prob > 0:
            drop = torch.from_numpy(self.rng.random((len(idx), th)) < drop_prob)
            a = act[:, :th]
            a[drop] = 0.0
            a[..., ACTION_DIM + 1][drop] = 1.0
        return clip, act

    def batch(self, B: int, th: int | None = None, drop_prob: float = 0.0):
        """One randomly drawn batch (the `--steps-per-epoch N` mode: sampling with replacement)."""
        if th is None:
            th = int(self.th_rng.choice(np.arange(1, self.W + 1), p=self.th_weights / self.th_weights.sum()))
        ok = np.where(self.max_th >= th)[0]
        idx = self.rng.choice(ok, size=B, replace=len(ok) < B)
        return self.make(idx, th, drop_prob)

    def epoch_steps(self, B: int, world: int = 1) -> int:
        """Batches in one full pass. Batches never mix history lengths, so each T_h group rounds up on its own."""
        G = B * max(world, 1)
        return int(sum(-(-int(n) // G) for n in self.th_weights if n > 0))

    def epoch_batches(self, B: int, world: int = 1, rank: int = 0, drop_prob: float = 0.0):
        """One full pass: every (sequence, history length T_h) clip exactly once, shuffled.

        The schedule is drawn from `th_rng`, which is seeded identically on every rank, so all ranks walk the
        same T_h in the same order — the DDP shape requirement — and each rank then takes its own slice of a
        world-sized batch. A T_h group whose size is not a multiple of B x world is filled up by wrap-around
        (like torch's DistributedSampler), so no rank ever gets an empty batch and all ranks run the same
        number of steps."""
        G = B * max(world, 1)
        plan = []
        for th in range(1, self.W + 1):
            ok = np.where(self.max_th >= th)[0]
            if len(ok) == 0:
                continue
            ok = self.th_rng.permutation(ok)
            pad = (-len(ok)) % G
            if pad:
                ok = np.concatenate([ok, np.resize(ok, pad)])
            plan += [(th, ok[i:i + G]) for i in range(0, len(ok), G)]
        for j in self.th_rng.permutation(len(plan)):
            th, idx = plan[j]
            yield self.make(idx[rank * B:(rank + 1) * B], th, drop_prob)

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


def setup_distributed() -> tuple[int, int, int]:
    """(rank, world_size, local_rank). Multi-GPU is opt-in through torchrun; a plain `python -m ...` stays single-GPU."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world == 1:
        return 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    return int(os.environ.get("RANK", 0)), world, local_rank


def main():
    cfg = load_cfg()
    tc = cfg.cjepa.train
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default=str(cache_dir() / "slots" / "wm_train_slots.pkl"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--actions", action="store_true", help="optional experiment (spec 14): action token per frame")
    ap.add_argument("--epochs", type=int, default=tc.max_epochs)
    ap.add_argument("--steps-per-epoch", type=int, default=tc.steps_per_epoch,
                    help="fixed number of randomly sampled steps; omit (config null) = one full pass over the clips")
    ap.add_argument("--batch-size", type=int, default=tc.batch_size)
    ap.add_argument("--lr", type=float, default=tc.lr)
    ap.add_argument("--seed", type=int, default=tc.seed)
    ap.add_argument("--log-every", type=int, default=1,
                    help="wandb: log the per-step losses every N steps (1 = every step)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    is_main = rank == 0
    device = a.device if world == 1 else f"cuda:{local_rank}"
    torch.manual_seed(a.seed + rank)
    rng = np.random.default_rng([a.seed, rank])      # each rank draws different sequences -> larger effective batch
    if not Path(a.slots).exists():
        raise SystemExit(
            f"{a.slots} 不存在。P5 的四步是严格串行的，5d 需要先完成 5b 和 5c：\n"
            f"  5b  CUDA_VISIBLE_DEVICES=2,3,4 bash scripts/p5b_train_videosaur.sh      # 训练 VideoSAUR（最耗时）\n"
            f"  5c  CUDA_VISIBLE_DEVICES=2 python scripts/p5c_extract_slots.py          # 冻结后提取 slot\n"
            f"  5d  python -m owm.wm.train_cjepa_predictor --tag {a.tag}")
    with open(a.slots, "rb") as f:
        data = pickle.load(f)
    if not data.get("meta", {}).get("aligned"):
        # the adapter feeds index-aligned slots (VideoSaur.encode); object masking is by index, so training must match
        raise SystemExit(f"{a.slots} 的 slot 编号未按时间对齐（旧缓存）。先跑（纯 CPU，几分钟）：\n"
                         f"  python scripts/p5c_align_slots.py --slots {a.slots}")
    N = int(next(iter(data["train"].values())).shape[1])
    W, F = cfg.temporal.context_frames, len(cfg.temporal.future_offsets)
    train = SlotClips(data["train"], W, F, a.actions, rng, th_rng=np.random.default_rng(a.seed))
    val = SlotClips(data["val"], W, F, a.actions, np.random.default_rng(0))
    full_pass = a.steps_per_epoch is None
    steps_per_epoch = train.epoch_steps(a.batch_size, world) if full_pass else a.steps_per_epoch
    if is_main:
        print(f"slots N={N}; train sequences={len(train.keys)} val sequences={len(val.keys)}; "
              f"history lengths 1..{int(train.max_th.max())} (W={W}, F={F}); "
              f"{world} GPU(s), effective batch {a.batch_size * world}")
        print(f"epoch = {steps_per_epoch} steps "
              f"({'full pass over ' + str(int(train.max_th.sum())) + ' clips' if full_pass else 'random sampling'}), "
              f"{a.epochs} epochs -> {steps_per_epoch * a.epochs} steps total")

    if a.actions:
        from owm.wm.cjepa_action import build_action_predictor, cjepa_action_loss
        model, loss_fn = build_predictor_with_actions(N, cfg, device, build_action_predictor, cjepa_action_loss)
    else:
        model = build_predictor(N, cfg, device)
        loss_fn = lambda m, clip, act: cjepa_loss(m, clip)  # noqa: E731
    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    out = run_dir(a.tag)
    wb = Run()
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        args_cfg = dict(vars(a), steps_per_epoch=steps_per_epoch, full_pass=full_pass)
        (out / "args.json").write_text(json.dumps(dict(args_cfg, num_slots=N, W=W, F=F, world_size=world,
                                                       slot_meta=data.get("meta", {})), indent=2, default=str))
        wb = wandb_init(f"cjepa_predictor_{a.tag}", "cjepa_predictor", group=f"cjepa/{a.tag}",
                        tags=["actions"] if a.actions else [],
                        config=dict(flat_config(dict(cjepa=dict(cfg.cjepa), temporal=dict(cfg.temporal))), **args_cfg,
                                    num_slots=N, W=W, F=F, world_size=world, n_params=sum(p.numel() for p in core.parameters()),
                                    n_train_sequences=len(train.keys), n_val_sequences=len(val.keys)))
    log, best, step, v = [], float("inf"), 0, float("nan")
    for epoch in range(a.epochs):
        model.train()
        t0, run = time.time(), []
        drop_prob = cfg.actions.train_drop_prob if a.actions else 0.0
        it = (train.epoch_batches(a.batch_size, world, rank, drop_prob) if full_pass else
              (train.batch(a.batch_size, drop_prob=drop_prob) for _ in range(steps_per_epoch)))
        # only rank 0 draws the bar / logs steps; the other ranks would fight over the terminal and over wandb
        bar = tqdm(it, total=steps_per_epoch, desc=f"epoch {epoch}/{a.epochs - 1}", disable=not is_main,
                   dynamic_ncols=True, leave=False)
        for clip, act in bar:
            l = loss_fn(model, clip.to(device), None if act is None else act.to(device))
            opt.zero_grad(set_to_none=True)
            l["loss"].backward()
            opt.step()
            s = [l["loss"].item(), l["loss_future"].item(), l["loss_masked_history"].item()]
            run.append(s)
            step += 1
            if is_main:
                if step % a.log_every == 0:   # this rank's batches only; the epoch row below is the all-rank mean
                    wb.log(dict(step_loss=s[0], step_future_mse=s[1], step_masked_history_mse=s[2]), step=step)
                m = np.mean(run[-50:], 0)     # last 50 steps, so the bar reacts to what the model is doing now
                bar.set_postfix(loss=f"{s[0]:.4f}", avg50=f"{m[0]:.4f}", fut=f"{m[1]:.4f}", msk=f"{m[2]:.4f}")
        bar.close()
        tr = np.mean(run, 0)
        if world > 1:   # report the mean over ranks, not just this rank's batches
            t = torch.tensor(tr, device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
            tr = t.cpu().numpy()
        if not is_main:     # validation and checkpointing happen on rank 0 only (no collectives inside)
            continue
        core.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for clip, act in tqdm(val.val_batches(a.batch_size), desc="val", dynamic_ncols=True, leave=False):
                clip = clip.to(device)
                act = None if act is None else act.to(device)
                pred = core.inference(clip[:, :-F]) if act is None else core.inference(clip[:, :-F], act)
                vs += torch.nn.functional.mse_loss(pred, clip[:, -F:], reduction="sum").item()
                vn += pred.numel()
        v = vs / max(vn, 1)
        log.append(dict(epoch=epoch, step=step, train_loss=tr[0], train_future=tr[1], train_masked=tr[2], val_future_mse=v))
        tqdm.write(f"epoch {epoch} step {step} train {tr[0]:.5f} (future {tr[1]:.5f} masked {tr[2]:.5f}) val_future {v:.5f} {time.time() - t0:.0f}s")
        wb.log(dict(epoch=epoch, train_loss=tr[0], train_future_mse=tr[1], train_masked_history_mse=tr[2],
                    val_future_mse=v, best_val_future_mse=min(best, v), lr=a.lr, epoch_seconds=time.time() - t0), step=step)
        # always save the unwrapped state_dict, so checkpoints are identical whether or not DDP was used
        torch.save(core.state_dict(), out / f"epoch_{epoch}_predictor.ckpt")
        if v < best:
            best = v
            torch.save(core.state_dict(), out / "best_predictor.ckpt")
        (out / "log.json").write_text(json.dumps(log, indent=2))
    if is_main:
        torch.save(core.state_dict(), out / "final_predictor.ckpt")
        wb.summary(dict(best_val_future_mse=best, final_val_future_mse=v, epochs=a.epochs,
                        steps_per_epoch=steps_per_epoch, total_steps=step, run_dir=str(out)))
        wb.finish()
    if world > 1:
        torch.distributed.destroy_process_group()


def build_predictor_with_actions(N, cfg, device, builder, loss):
    model = builder(N, cfg, device)
    return model, lambda m, clip, act: loss(m, clip, act)


if __name__ == "__main__":
    main()
