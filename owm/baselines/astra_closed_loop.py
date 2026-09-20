"""GPT-6 Astra, setting B (spec 12.3): RoboMME's native closed-loop protocol (choose the high-level action, the oracle
planner executes it) — comparable with Table 11 of the RoboMME paper. No markers, raw pixels only.

  export OPENAI_API_KEY=...
  python -m owm.baselines.astra_closed_loop --tasks VideoUnmask --episodes 2     # trial
  python -m owm.baselines.astra_closed_loop
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from owm.baselines.astra_client import call_astra
from owm.closed_loop.runner import run_episode
from owm.config import load_cfg, output_dir
from owm.data.sampling import history_indices


class AstraPolicy:
    def __init__(self):
        from openai import OpenAI
        self.client, self.calls = OpenAI(), []

    def __call__(self, ctx: dict):
        cfg = load_cfg().astra
        idx = history_indices(ctx["t"])[-cfg.max_images:]
        res = call_astra(ctx["task_goal"], ctx["options"], [ctx["frames"][i] for i in idx], "B", self.client)
        self.calls.append(dict(task=ctx["task"], episode=ctx["episode"], t=ctx["t"], frames=idx, **res))
        p = res["parsed"]
        if p is None:
            return None   # invalid output -> the episode counts as failed
        pt = p.get("point")
        return {"choice": p["choice"], "point": [int(np.clip(pt[0], 0, 255)), int(np.clip(pt[1], 0, 255))] if pt else None}


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(cfg.tasks))
    ap.add_argument("--episodes", type=int, default=cfg.closed_loop.episodes)
    a = ap.parse_args()
    out = output_dir() / "astra" / "closed_loop"
    out.mkdir(parents=True, exist_ok=True)
    policy, summary = AstraPolicy(), {}
    for task in a.tasks:
        res = []
        for ep in range(a.episodes):
            r = run_episode(task, ep, policy)
            res.append(r)
            print(f"{task} ep{ep}: {r['status']}")
            (out / f"{task}.json").write_text(json.dumps(res, indent=2, default=str))
            (out / "calls.jsonl").write_text("\n".join(json.dumps(c, default=str) for c in policy.calls))
        summary[task] = dict(success_rate=float(np.mean([r["success"] for r in res])), n=len(res))
    summary["cost_usd"] = float(sum(c.get("cost_usd") or 0 for c in policy.calls))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
