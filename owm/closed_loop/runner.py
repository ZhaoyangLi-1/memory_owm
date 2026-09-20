"""Closed-loop evaluation under RoboMME's native `multi_choice` protocol (spec 11).

Verified interface (spec 17 item 15, robomme doc/env_format.md + scripts/evaluation.py):
  * BenchmarkEnvBuilder(env_id, dataset="test", action_space="multi_choice", max_steps=1300).make_env_for_episode(i)
  * reset() plays the conditioning video internally: obs["front_rgb_list"] = video frames + current frame
  * every env.step({"choice": label, "point": [y, x]}) IS one decision: the oracle planner executes the option and
    returns all dense frames; there is no separate "decision request" flag
  * the point is resolved to the nearest candidate actor (no distance threshold); an unknown label or a missing point
    for a need_parameter option ends the episode with info["status"] == "error" (obs is None)
  * success <=> info["status"] == "success" when terminated/truncated; timeout after max_steps executed env steps
  * GT for the candidates is read live from env.unwrapped (same code as the offline replay)
The ceiling cannot be run in closed loop (the real future has not happened yet).

  python -m owm.closed_loop.runner --condition floor
  python -m owm.closed_loop.runner --condition cjepa --tasks VideoUnmask --episodes 5
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from owm.config import load_cfg, output_dir
from owm.data import goal_parser
from owm.data.gt_state import _front_seg, assign_ids, live_tokens, scene_objects

MAX_DECISIONS = 40


def run_episode(task: str, episode: int, policy, dataset: str | None = None, max_steps: int | None = None) -> dict:
    """policy(ctx) -> {"choice": str, "point": [y, x] | None}. ctx: frames [T,256,256,3], t, tok, options, task_goal, objs."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    cl = load_cfg().closed_loop
    builder = BenchmarkEnvBuilder(env_id=task, dataset=dataset or cl.dataset, action_space="multi_choice",
                                  max_steps=max_steps or cl.max_steps)
    env = builder.make_env_for_episode(episode, include_maniskill_obs=True)
    log, status = [], "error"
    try:
        obs, info = env.reset()
        u = env.unwrapped
        objs = scene_objects(u, task)
        ids = assign_ids(f"closed_loop/{dataset or cl.dataset}/{task}/{episode}", len(objs))
        K = np.asarray(info["front_camera_intrinsic"], np.float64).reshape(3, 3)
        options = [dict(o) for o in info["available_multi_choices"]]
        task_goal = info["task_goal"][0] if isinstance(info["task_goal"], (list, tuple)) else str(info["task_goal"])
        frames = [np.asarray(f) for f in obs["front_rgb_list"]]
        for n in range(MAX_DECISIONS):
            E = np.asarray(obs["front_camera_extrinsic_list"][-1], np.float64).reshape(3, 4)
            tok = live_tokens(u, objs, ids, _front_seg(obs["maniskill_obs"][-1]), K, E)
            ctx = dict(frames=np.stack(frames), t=len(frames) - 1, tok=tok, options=options, task_goal=task_goal,
                       objs=objs, task=task, episode=episode, n_decision=n)
            action = policy(ctx)
            log.append(dict(t=ctx["t"], action=action, n_cand=int(len(tok["id"]))))
            if action is None or action.get("choice") not in [o["label"] for o in options]:
                status = "invalid_action"
                break
            obs, _, term, trunc, info = env.step(action)
            status = info.get("status", "error")
            if obs is None or status == "error":
                log[-1]["error"] = info.get("error_message")
                break
            frames += [np.asarray(f) for f in obs["front_rgb_list"]]
            if bool(term) or bool(trunc):
                break
        else:
            status = "too_many_decisions"
    finally:
        env.close()
    return dict(task=task, episode=episode, status=status, success=status == "success", decisions=log)


class ReadoutPolicy:
    """GT candidates (live) + goal + world-model prediction -> readout heads (3 seeds averaged; LPWM: 5 samples averaged)."""

    def __init__(self, task: str, condition: str, adapter=None, device: str = "cuda"):
        from owm.readout.dataset import DecisionData
        from owm.readout.model import Readout
        from owm.readout.train import ckpt_path
        from owm.wm.evidence import D_GT
        cfg = load_cfg()
        self.task, self.adapter, self.device = task, adapter, device
        self.heads = []
        for seed in cfg.readout.seeds:
            ck = torch.load(ckpt_path(task, condition, seed), map_location="cpu", weights_only=False)
            r = cfg.readout
            opts = [dict(label=l) for l in ck["labels"]]
            model = Readout(d_cand=D_GT, d_goal=ck["d_goal"], d_evid=ck["d_evid"], n_options=len(opts), d=r.d_model,
                            layers=r.layers, heads=r.heads, ffn=r.ffn, dropout=r.dropout, n_freq=r.fourier_freqs,
                            max_steps=max(cfg.temporal.context_frames, 8))
            model.load_state_dict(ck["state"])
            self.heads.append((model.to(device).eval(), ck))
        self.DecisionData, self.condition = DecisionData, condition

    @torch.no_grad()
    def __call__(self, ctx: dict) -> dict:
        from owm.readout.model import decode
        goal = goal_parser.goal_vector(self.task, ctx["task_goal"])
        evidence = self.adapter.predict(ctx["frames"], ctx["t"], seed=0) if self.adapter is not None else None
        S = evidence[0].shape[0] if evidence is not None else 1
        lp, pp, data = 0.0, 0.0, None
        for model, ck in self.heads:
            data = self.DecisionData.live(self.task, self.condition, ck, ctx["options"])
            sample = data.live_sample(ctx["tok"], goal, evidence)
            for s in range(S):
                b, _ = data.collate_items([sample], sample_idx=s, device=self.device)
                ll, pl = model(b)
                lp, pp = lp + ll.softmax(-1), pp + pl.softmax(-1)
        label, cand = decode(lp, pp, data.need_param.to(self.device))
        choice = data.labels[int(label[0])]
        point = None
        if int(cand[0]) >= 0:
            u_, v_ = ctx["tok"]["uv"][int(cand[0])]
            point = [int(round(float(v_) * 256)), int(round(float(u_) * 256))]    # [y, x] = [row, col]
        return {"choice": choice, "point": point}


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, help="floor | cjepa | lpwm")
    ap.add_argument("--tasks", nargs="+", default=list(cfg.tasks))
    ap.add_argument("--episodes", type=int, default=cfg.closed_loop.episodes)
    ap.add_argument("--dataset", default=cfg.closed_loop.dataset)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    adapter = None
    if a.condition.startswith("cjepa"):
        from owm.wm.cjepa_adapter import CJepaAdapter
        adapter = CJepaAdapter(device=a.device)
    elif a.condition.startswith("lpwm"):
        from owm.wm.lpwm_adapter import LpwmAdapter
        adapter = LpwmAdapter(device=a.device)
    out = output_dir() / "closed_loop" / a.condition
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in a.tasks:
        policy = ReadoutPolicy(task, a.condition, adapter, a.device)
        res = []
        for ep in range(a.episodes):
            t0 = time.time()
            r = run_episode(task, ep, policy, a.dataset)
            res.append(r)
            print(f"{task} ep{ep}: {r['status']} ({len(r['decisions'])} decisions, {time.time() - t0:.0f}s)")
            (out / f"{task}.json").write_text(json.dumps(res, indent=2, default=str))
        summary[task] = dict(success_rate=float(np.mean([r["success"] for r in res])), n=len(res))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
