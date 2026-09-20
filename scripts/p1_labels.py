"""P1: label/rule checks (spec 4.5): timelines, option vocabulary, goal-parser coverage, decision index.

  python scripts/p1_labels.py                 # all checks on whatever is cached
  python scripts/p1_labels.py --timelines 5   # also print 5 episode timelines per task
"""
import argparse
import json
from collections import Counter

import numpy as np
import yaml

from owm.config import ROOT, load_cfg, output_dir
from owm.data import goal_parser
from owm.data.decision_index import build_index, coverage_report, index_path
from owm.data.h5_reader import list_cached


def timelines(task, n):
    for ep in list_cached("train", task)[:n]:
        m = ep.meta
        print(f"--- {ep.key} T={ep.T} n_video={ep.n_video} diff={ep.difficulty} goal={ep.task_goal[0]!r}")
        for t in np.where(m["is_boundary"])[0]:
            print(f"   t={t:4d} video={int(m['is_video_demo'][t])} choice={m['choice'][t]!r:4} point={m['point'][t].tolist()} "
                  f"| {m['grounded_subgoal'][t]} | {m['simple_subgoal'][t]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelines", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg()
    report = []

    vocab = {}
    for task in cfg.tasks:
        eps = [e for s in ("train", "test") for e in list_cached(s, task)]
        if not eps:
            print(f"[skip] {task}: nothing cached")
            continue
        if a.timelines:
            timelines(task, a.timelines)
        # 1) option vocabulary fixed within a task?
        variants = Counter(str(e.meta["options"]) for e in eps)
        vocab[task] = dict(fixed=len(variants) == 1, n_episodes=len(eps), options=json.loads(variants.most_common(1)[0][0]))
        report.append(f"{task}: {len(eps)} episodes, option variants={len(variants)}")
        # 2) goal parser coverage (must be 100%)
        goals, fails = Counter(g for e in eps for g in e.task_goal), []
        for g in goals:
            try:
                goal_parser.goal_vector(task, g)
            except goal_parser.GoalParseError as err:
                fails.append(str(err))
        report.append(f"   goal parser: {len(goals) - len(fails)}/{len(goals)} distinct instructions parsed")
        report += [f"   FAIL {f}" for f in fails]
        # 3) are conditioning-video actions informative? (spec 17 item 14)
        stds = [float(np.nanstd(e.meta["joint_action"][e.meta["is_video_demo"]], axis=0).max()) for e in eps if e.n_video]
        if stds:
            report.append(f"   video joint_action max-std: median={np.median(stds):.4f} -> "
                          f"{'informative (leak risk)' if np.median(stds) > 1e-3 else 'constant (no action info in the video)'}")
    (ROOT / "configs" / "option_vocab.yaml").write_text(yaml.safe_dump(vocab, sort_keys=False, allow_unicode=True))

    df = build_index()
    index_path().parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(index_path())
    report.append(f"\ndecision index: {len(df)} rows -> {index_path()}")
    report.append(f"skipped boundaries: {df.attrs['stats']}")
    report.append(coverage_report(df))
    text = "\n".join(report)
    print(text)
    out = output_dir() / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "p1_labels.txt").write_text(text)


if __name__ == "__main__":
    main()
