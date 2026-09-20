"""Result tables (spec 13, 14.5): outputs/tables/main_table.md|json, action_table.md, appendix tables."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from owm.analysis.bootstrap import bootstrap_accuracy, bootstrap_difference, bootstrap_scores, paired_table, seed_mean
from owm.analysis.verdict import needs_second_wm_seed, verdict
from owm.config import load_cfg, output_dir
from owm.data.decision_index import load_index, usable
from owm.data.decision_rules import MEMORY_IN
from owm.readout.evaluate import load_predictions


def chance_level(index: pd.DataFrame, task: str, split: str = "test") -> float:
    """Pointer decisions: mean of 1 / number of candidates.
    Label decisions: accuracy on `split` of the best predictor that sees ONLY the goal ("most frequent training label
    given the goal vector"). For tasks without a goal variable this is the majority-class share of spec 2.5; for
    PickXtimes it accounts for the fact that the requested repeat count alone is informative (count = 1 -> always
    "stop"), which is a prior, not a leak from the current frame."""
    d = index[(index.task == task) & (index.decision_type == "memory_critical")]
    te, tr = d[d.split == split], d[d.split == "train"]
    if not len(te) or not len(tr):
        return float("nan")
    if MEMORY_IN[task] == "point":
        # 1 / number of candidates OF THE TARGET'S TYPE: the option text fixes the object type ("pick up the container"),
        # which the floor learns from the labels; buttons / target discs are not real alternatives.
        n = te["n_cand_type"] if "n_cand_type" in te and (te["n_cand_type"] > 0).all() else te["n_cand"]
        return float((1.0 / n.clip(lower=1)).mean())
    overall = tr["label"].value_counts().idxmax()
    by_goal = tr.groupby("goal_vec")["label"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    return float((te["goal_vec"].map(by_goal).fillna(overall) == te["label"]).mean())


def check_chance(index: pd.DataFrame, task: str, split: str = "test") -> float:
    d = index[(index.task == task) & (index.decision_type == "non_critical") & index.need_param & (index.split == split)]
    return float((1.0 / d["n_cand"].clip(lower=1)).mean()) if len(d) else float("nan")


def _fmt(a: dict | None) -> str:
    if not a or a.get("n", 1) == 0 or np.isnan(a["acc"]):
        return "—"
    return f"{100 * a['acc']:.1f} [{100 * a['ci_low']:.1f}, {100 * a['ci_high']:.1f}]"


def _acc(pred, task, cond, dtype, split):
    p = pred[(pred.task == task) & (pred.split == split)]
    m = seed_mean(p, cond, dtype)
    return bootstrap_accuracy(m) if len(m) else None


OPTIONAL_ROWS = {"astra_A": "GPT-6 Astra (setting A)", "astra_A_frame_only": "GPT-6 Astra (setting A, decision frame only)"}


def _apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """stats.exclude = {task: [difficulty, ...]} (reading rule 1 / 2 exclusions). The appendix still shows every stratum."""
    for task, diffs in (load_cfg().stats.get("exclude") or {}).items():
        df = df[~((df.task == task) & df.difficulty.isin(list(diffs)))]
    return df


def main_table(models=("cjepa", "lpwm"), split: str = "test") -> dict:
    cfg = load_cfg()
    pred_all = load_predictions()
    pred = _apply_exclusions(pred_all)
    index = _apply_exclusions(usable(load_index()))
    res = {}
    for task in cfg.tasks:
        r = dict(chance=chance_level(index, task, split), check_chance=check_chance(index, task, split))
        for cond in ("floor", "ceiling", *models, *OPTIONAL_ROWS):
            r[cond] = _acc(pred, task, cond, "memory_critical", split)
        for m in models:
            tab = paired_table(pred, task, {"floor": "floor", "wm": m, "ceil": "ceiling"}, split=split)
            chk = _acc(pred, task, f"check_{m}", "non_critical", split)
            r[f"check_{m}"] = chk
            if len(tab):
                sc = bootstrap_scores(tab)
                r[f"S_{m}"] = sc
                r[f"verdict_{m}"] = verdict(sc, r["chance"], chk["acc"] if chk else None)
                r[f"second_seed_{m}"] = needs_second_wm_seed(sc)
        # sanity (spec 10 step 7): main heads on non-critical decisions
        r["sanity_noncritical"] = {c: _acc(pred, task, c, "non_critical", split) for c in ("floor", "ceiling")}
        r["by_difficulty"] = {
            m: {d: _acc(pred_all[pred_all.difficulty == d], task, m, "memory_critical", split) for d in ("easy", "medium", "hard")}
            for m in ("floor", "ceiling", *models)}
        res[task] = r

    tasks = list(cfg.tasks)
    head = "| | " + " | ".join(tasks) + " |\n|---|" + "---|" * len(tasks)
    rows = [("机会水平 chance", lambda r: f"{100 * r['chance']:.1f}"),
            ("下限 floor: GT candidates, no future", lambda r: _fmt(r["floor"]))]
    for m in models:
        rows.append((f"**{m}**: GT candidates + predicted future", lambda r, m=m: _fmt(r[m])))
    rows.append(("上限 ceiling: GT candidates + real future", lambda r: _fmt(r["ceiling"])))
    for cond, name in OPTIONAL_ROWS.items():
        if any(res[t][cond] for t in tasks):
            rows.append((name, lambda r, cond=cond: _fmt(r[cond])))
    for m in models:
        rows.append((f"{m} memory score S", lambda r, m=m: "—" if f"S_{m}" not in r else
                     f"{r[f'S_{m}']['S']:.2f} [{r[f'S_{m}']['ci_low']['S']:.2f}, {r[f'S_{m}']['ci_high']['S']:.2f}]"))
    for m in models:
        rows.append((f"{m} prediction check (chance {{c}})", lambda r, m=m: _fmt(r[f"check_{m}"]) if r[f"check_{m}"] else "样本不足 insufficient"))
    for m in models:
        rows.append((f"{m} verdict", lambda r, m=m: r.get(f"verdict_{m}", "—")))
    lines = [head]
    for name, fn in rows:
        cells = [fn(res[t]) for t in tasks]
        lines.append(f"| {name.replace(' (chance {c})', '')} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    md += "\n\nAccuracy in %, mean over 3 readout seeds, [95% CI] = paired bootstrap over episodes.\n"
    if cfg.stats.get("exclude"):
        md += f"Excluded from this table (reading rule 1, see experiment.yaml -> stats.exclude): {dict(cfg.stats.exclude)}\n"
    md += "Prediction-check chance levels: " + ", ".join(f"{t} {100 * res[t]['check_chance']:.1f}" for t in tasks if not np.isnan(res[t]["check_chance"])) + "\n"

    out = output_dir() / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main_table.md").write_text(md)
    (out / "main_table.json").write_text(json.dumps(res, indent=2, default=float))
    app = ["| task | cond | easy | medium | hard |", "|---|---|---|---|---|"]
    for t in tasks:
        for m in ("floor", "ceiling", *models):
            app.append(f"| {t} | {m} | " + " | ".join(_fmt(res[t]["by_difficulty"][m][d]) for d in ("easy", "medium", "hard")) + " |")
    app += ["", "| task | floor (non-critical) | ceiling (non-critical) |", "|---|---|---|"]
    app += [f"| {t} | {_fmt(res[t]['sanity_noncritical']['floor'])} | {_fmt(res[t]['sanity_noncritical']['ceiling'])} |" for t in tasks]
    (out / "appendix.md").write_text("\n".join(app))
    return res


def action_table(models=("cjepa", "lpwm"), split: str = "test") -> str:
    """Spec 14.5: leak check (floor + action history vs floor) and the action gain ΔA per model."""
    cfg = load_cfg()
    pred = _apply_exclusions(load_predictions())
    tasks = list(cfg.tasks)
    rows = {k: [] for k in ["floor", "leak check: floor + action history", "leaks?"]}
    for m in models:
        rows.update({f"{m} no actions": [], f"{m} with actions": [], f"{m} ΔA": []})
    rows["conclusion"] = []
    for t in tasks:
        rows["floor"].append(_fmt(_acc(pred, t, "floor", "memory_critical", split)))
        rows["leak check: floor + action history"].append(_fmt(_acc(pred, t, "action_history", "memory_critical", split)))
        tab = paired_table(pred, t, {"a": "action_history", "b": "floor"}, split=split)
        leak = None
        if len(tab):
            d = bootstrap_difference(tab, "a", "b")
            leak = d["ci_low"] > 0
        rows["leaks?"].append("—" if leak is None else ("yes" if leak else "no"))
        concl = []
        for m in models:
            rows[f"{m} no actions"].append(_fmt(_acc(pred, t, m, "memory_critical", split)))
            rows[f"{m} with actions"].append(_fmt(_acc(pred, t, f"{m}_act", "memory_critical", split)))
            tab = paired_table(pred, t, {"a": f"{m}_act", "b": m}, split=split)
            if len(tab):
                d = bootstrap_difference(tab, "a", "b")
                rows[f"{m} ΔA"].append(f"{100 * d['diff']:+.1f} [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}]")
                if leak:
                    concl.append(f"{m}: leak — not interpretable as visual memory")
                elif leak is not None:
                    concl.append(f"{m}: " + ("actions help" if d["ci_low"] > 0 else "actions hurt" if d["ci_high"] < 0 else "no effect"))
            else:
                rows[f"{m} ΔA"].append("—")
        rows["conclusion"].append("; ".join(concl) or "—")
    md = "| | " + " | ".join(tasks) + " |\n|---|" + "---|" * len(tasks) + "\n"
    md += "\n".join(f"| {k} | " + " | ".join(v) + " |" for k, v in rows.items())
    out = output_dir() / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "action_table.md").write_text(md)
    return md
