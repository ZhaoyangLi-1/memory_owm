"""P4 / P8: train + evaluate readout heads.

  python scripts/p4_train_readout.py --conditions floor ceiling                 # P4 reference heads
  python scripts/p4_train_readout.py --conditions cjepa lpwm check_cjepa check_lpwm   # P8
  python scripts/p4_train_readout.py --conditions floor --tasks VideoUnmask --seeds 0 --eval-split val
"""
import argparse

import pandas as pd

from owm.config import load_cfg, output_dir
from owm.readout.dataset import DecisionData, is_check
from owm.readout.evaluate import evaluate_head, merge_predictions
from owm.readout.train import train_one


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(cfg.tasks))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(cfg.readout.seeds))
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--min-check", type=int, default=cfg.stats.min_prediction_check_decisions, help="debug only")
    a = ap.parse_args()
    logs, preds = [], []
    min_check = a.min_check
    for task in a.tasks:
        for cond in a.conditions:
            try:
                data = DecisionData(task, cond)
            except FileNotFoundError as e:
                print(f"[skip] {task}/{cond}: evidence missing ({e})")
                continue
            n_tr, n_te = len(data.split_indices("train")), len(data.split_indices(a.eval_split))
            if is_check(cond) and (n_te < min_check or n_tr == 0):
                print(f"[insufficient] {task}/{cond}: {n_te} {a.eval_split} decisions (< {min_check}) — prediction check not run")
                continue
            for seed in a.seeds:
                if not a.eval_only:
                    logs.append(train_one(task, cond, seed, data, a.device))
                p = evaluate_head(task, cond, seed, a.eval_split, data, a.device)
                if len(p):
                    merge_predictions(p)      # saved as we go: predictions/<task>__<cond>__<split>.parquet
                if len(p):
                    crit = p[p.decision_type == ("non_critical" if is_check(cond) else "memory_critical")]
                    print(f"{task:17s} {cond:14s} seed{seed} {a.eval_split}: acc(all)={p.correct.mean():.3f} "
                          f"acc(main)={crit.correct.mean():.3f} n={len(crit)}")
    if logs:
        out = output_dir() / "reports" / "readout_training"
        out.mkdir(parents=True, exist_ok=True)
        for (task, cond), g in pd.DataFrame(logs).groupby(["task", "condition"]):
            g.to_csv(out / f"{task}__{cond}.csv", index=False)

if __name__ == "__main__":
    main()
