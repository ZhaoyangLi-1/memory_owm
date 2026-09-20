"""GPT-6 Astra, setting A (spec 12.2): the SAME memory-critical test decisions as the main table, same answer
interface (label + candidate number). History frames exactly as the world models get them (history_indices), the
decision frame carries numbered GT-candidate markers.

  export OPENAI_API_KEY=...
  python -m owm.baselines.astra_offline --limit 10          # trial run: check format, coordinates, cost
  python -m owm.baselines.astra_offline                     # full run   (condition name: astra_A)
  python -m owm.baselines.astra_offline --frame-only        # control: decision frame only (astra_A_frame_only)
  python -m owm.baselines.astra_offline --dry-run --limit 3 # build requests, no API call
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from owm.baselines import astra_prompts as P
from owm.baselines.astra_client import call_astra
from owm.config import ROOT, load_cfg, output_dir
from owm.data.decision_index import load_index, usable
from owm.data.gt_state import GT
from owm.data.h5_reader import Episode
from owm.data.sampling import history_indices
from owm.readout.evaluate import merge_predictions


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(cfg.tasks))
    ap.add_argument("--split", default="test")
    ap.add_argument("--frame-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cond = "astra_A_frame_only" if a.frame_only else "astra_A"
    vocab = yaml.safe_load((ROOT / "configs" / "option_vocab.yaml").read_text())
    df = usable(load_index())
    df = df[(df.split == a.split) & (df.decision_type == "memory_critical") & df.task.isin(a.tasks)]
    df = df.sort_values(["task", "episode", "t"])
    if a.limit:
        df = df.groupby("task").head(a.limit)
    out_dir = output_dir() / "astra"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{cond}_{a.split}.jsonl"
    done = {}
    if log_path.exists():   # resume: never pay twice for the same decision
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            done[(r["key"], r["t"])] = r
    client = None
    if not a.dry_run:
        from openai import OpenAI
        client = OpenAI()
    rows, cost = [], 0.0
    with open(log_path, "a") as log:
        for r in df.to_dict("records"):
            ep = Episode(r["source"], r["task"], r["episode"])
            options = vocab[r["task"]]["options"]
            tok = GT(r["source"], r["task"], r["episode"]).tokens(r["t"])
            hist = [r["t"]] if a.frame_only else history_indices(r["t"])[-cfg.astra.max_images:]
            marked, mapping = P.draw_candidates(ep.frames[r["t"]], tok["uv"])
            images = [ep.frames[i] for i in hist[:-1]] + [marked]
            rec = done.get((r["key"], r["t"]))
            if rec is None:
                if a.dry_run:
                    inp = P.build_input(ep.task_goal[0], options, images, "A")
                    print(r["key"], r["t"], f"{len(images)} images, {len(mapping)} candidates;", inp[0]["content"][-1]["text"][:160].replace("\n", " | "))
                    marked.save(out_dir / f"dryrun_{r['task']}_{r['episode']}_{r['t']}.png")
                    continue
                res = call_astra(ep.task_goal[0], options, images, "A", client)
                rec = dict(key=r["key"], t=r["t"], task=r["task"], frames=hist, n_candidates=len(mapping),
                           task_goal=ep.task_goal[0], model=cfg.astra.model, reasoning_effort=cfg.astra.reasoning_effort, **res)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            cost += rec.get("cost_usd") or 0.0
            parsed = rec.get("parsed")
            need = {o["label"]: o["need_parameter"] for o in options}
            pred_label, pred_obj = (parsed or {}).get("choice"), -1
            cid = (parsed or {}).get("candidate_id")
            if parsed and need.get(pred_label) and cid in mapping:
                pred_obj = int(tok["obj_index"][mapping[cid]])
            label_ok = parsed is not None and pred_label == r["label"]
            correct = label_ok and (not r["need_param"] or pred_obj == r["target_obj"])   # invalid output = wrong
            rows.append(dict(source=r["source"], task=r["task"], episode=r["episode"], key=r["key"], t=r["t"], split=a.split,
                             difficulty=r["difficulty"], decision_type=r["decision_type"], n_cand=r["n_cand"],
                             condition=cond, seed=0, label=r["label"], target_obj=r["target_obj"],
                             pred_label=str(pred_label), pred_obj=pred_obj, label_correct=bool(label_ok), correct=bool(correct)))
    if rows:
        p = pd.DataFrame(rows)
        merge_predictions(p)
        invalid = sum(1 for k in done.values() if k.get("parsed") is None)
        print(p.groupby("task")["correct"].agg(["mean", "size"]))
        print(f"total cost so far: ${cost:.2f}; invalid outputs (counted as wrong): {invalid}")


if __name__ == "__main__":
    main()
