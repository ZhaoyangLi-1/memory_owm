"""P7: run the frozen world models on every decision and cache the predicted future tokens (spec 8), then run the
future-perturbation test.

  python scripts/p7_extract_evidence.py --model cjepa            [--actions]
  python scripts/p7_extract_evidence.py --model lpwm             [--actions] [--batched-samples]
  python scripts/p7_extract_evidence.py --model action_history   # leak check of the action experiment (spec 14.4)
  python scripts/p7_extract_evidence.py --model cjepa --perturb-only --n-perturb 100
"""
import argparse
import json

import numpy as np
from tqdm import tqdm

from owm.config import load_cfg, output_dir
from owm.data.decision_index import load_index, usable
from owm.data.h5_reader import Episode
from owm.wm.evidence import evidence_path, save_evidence
from owm.wm.future_perturb import future_perturbation_test, history_actions


def build_adapter(a):
    if a.model == "cjepa":
        from owm.wm.cjepa_adapter import CJepaAdapter
        return CJepaAdapter(a.predictor_ckpt, a.videosaur_ckpt, a.device, with_actions=a.actions)
    from owm.wm.lpwm_adapter import LpwmAdapter
    return LpwmAdapter(a.lpwm_run, a.lpwm_ckpt, a.device, with_actions=a.actions, batched_samples=a.batched_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cjepa", "lpwm", "action_history"])
    ap.add_argument("--actions", action="store_true")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--predictor-ckpt", default=None)
    ap.add_argument("--videosaur-ckpt", default=None)
    ap.add_argument("--lpwm-run", default=None)
    ap.add_argument("--lpwm-ckpt", default=None)
    ap.add_argument("--batched-samples", action="store_true")
    ap.add_argument("--n-perturb", type=int, default=100)
    ap.add_argument("--perturb-only", action="store_true")
    ap.add_argument("--skip-perturb", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N decisions")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = load_cfg()
    df = usable(load_index())
    df = df[df.task.isin(a.tasks or cfg.tasks)].sort_values(["source", "task", "episode", "t"])
    if a.limit:
        df = df.head(a.limit)

    if a.model == "action_history":   # no network: the evidence is the executed action chunks up to t, newest first
        for r in tqdm(df.to_dict("records")):
            ep = Episode(r["source"], r["task"], r["episode"])
            act = history_actions(ep, r["t"])[::-1].copy()
            save_evidence(evidence_path("action_history", r["source"], r["task"], r["episode"], r["t"]),
                          act[None, :, None, :], np.zeros((1, len(act), 1, 2)), np.ones((1, len(act), 1), bool))
        return

    adapter = build_adapter(a)
    print("adapter:", adapter.describe())
    if not a.perturb_only:
        for r in tqdm(df.to_dict("records")):
            p = evidence_path(adapter.name, r["source"], r["task"], r["episode"], r["t"])
            if p.exists() and not a.overwrite:
                continue
            ep = Episode(r["source"], r["task"], r["episode"])
            kw = dict(actions=history_actions(ep, r["t"])) if a.actions else {}
            tokens, uv, valid = adapter.predict(ep.frames, r["t"], seed=0, **kw)
            save_evidence(p, tokens, uv, valid)
    if not a.skip_perturb:
        sub = df.sample(n=min(a.n_perturb, len(df)), random_state=0)
        res = []
        for r in tqdm(sub.to_dict("records"), desc="future perturbation"):
            res.append(future_perturbation_test(adapter, Episode(r["source"], r["task"], r["episode"]), r["t"], control=len(res) < 5))
        n_ok = sum(x["passed"] for x in res)
        assert all(x.get("control_changed", True) for x in res), "negative control failed: the test cannot detect changes"
        out = output_dir() / "reports"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"future_perturbation_{adapter.name}.json").write_text(json.dumps(dict(adapter=adapter.describe(), passed=n_ok, total=len(res), results=res), indent=2))
        print(f"future perturbation test [{adapter.name}]: {n_ok}/{len(res)} passed")
        if n_ok != len(res):
            raise SystemExit("FUTURE INFORMATION LEAK — fix the adapter before using any result")


if __name__ == "__main__":
    main()
