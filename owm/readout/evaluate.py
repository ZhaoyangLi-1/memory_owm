"""Inference + scoring of readout heads on a split (spec 9.3, 9.6, 10) -> outputs/predictions.parquet."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from owm.config import output_dir
from owm.readout.dataset import DecisionData, is_check
from owm.readout.model import decode
from owm.readout.train import load_head


def predictions_dir():
    return output_dir() / "predictions"


def load_predictions() -> pd.DataFrame:
    """All saved predictions (one parquet per task / condition / split, so parallel runs never collide)."""
    files = sorted(predictions_dir().glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no predictions under {predictions_dir()} — run scripts/p4_train_readout.py first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@torch.no_grad()
def predict_probs(model, data: DecisionData, idx, device="cuda", bs=256):
    """Label / pointer probabilities averaged over the evidence samples (LPWM: 5 rollouts)."""
    S = data.n_samples_evidence()
    lab, ptr, cobj = [], [], []
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        pl, pp = 0.0, 0.0
        for s in range(S):
            b, y = data.collate(chunk, sample_idx=s, device=device)
            ll, ptl = model(b)
            pl, pp = pl + ll.softmax(-1) / S, pp + ptl.softmax(-1) / S
        lab.append(pl.cpu())
        ptr.append(pp.cpu())
        cobj.append(y.cand_obj)
    n = max(p.shape[1] for p in ptr)
    ptr = [torch.nn.functional.pad(p, (0, n - p.shape[1])) for p in ptr]
    cobj = [np.pad(c, ((0, 0), (0, n - c.shape[1])), constant_values=-1) for c in cobj]
    return torch.cat(lab), torch.cat(ptr), np.concatenate(cobj)


def evaluate_head(task: str, condition: str, seed: int, split: str = "test", data: DecisionData | None = None,
                  device: str = "cuda") -> pd.DataFrame:
    data = data or DecisionData(task, condition)
    model = load_head(task, condition, seed, data, device)
    idx = data.split_indices(split)
    if len(idx) == 0:
        return pd.DataFrame()
    label_p, ptr_p, cand_obj = predict_probs(model, data, idx, device)
    rows = data.df.iloc[idx]
    y_label = torch.tensor([data.labels.index(l) for l in rows["label"]])
    y_obj = rows["target_obj"].to_numpy()
    pred_label, pred_cand = decode(label_p, ptr_p, data.need_param)
    ptr_obj = cand_obj[np.arange(len(idx)), ptr_p.argmax(-1).numpy()]          # pointer argmax as GT object index
    pred_obj = np.where(pred_cand.numpy() >= 0, ptr_obj, -1)
    need = data.need_param[y_label].numpy()
    if is_check(condition):   # label is given: score the pointer only
        correct = ptr_obj == y_obj
        pred_label = y_label
        pred_obj = ptr_obj
    else:
        correct = (pred_label.numpy() == y_label.numpy()) & (~need | (pred_obj == y_obj))
    return pd.DataFrame(dict(
        source=rows["source"].to_numpy(), task=task, episode=rows["episode"].to_numpy(), key=rows["key"].to_numpy(),
        t=rows["t"].to_numpy(), split=split, difficulty=rows["difficulty"].to_numpy(),
        decision_type=rows["decision_type"].to_numpy(), n_cand=rows["n_cand"].to_numpy(), condition=condition, seed=seed,
        label=rows["label"].to_numpy(), target_obj=y_obj, pred_label=[data.labels[i] for i in pred_label.tolist()],
        pred_obj=pred_obj, label_correct=pred_label.numpy() == y_label.numpy(), correct=correct))


def merge_predictions(new: pd.DataFrame) -> None:
    """Store predictions as predictions/<task>__<condition>__<split>.parquet, replacing the seeds that were re-run."""
    predictions_dir().mkdir(parents=True, exist_ok=True)
    for (task, cond, split), g in new.groupby(["task", "condition", "split"]):
        f = predictions_dir() / f"{task}__{cond}__{split}.parquet"
        if f.exists():
            old = pd.read_parquet(f)
            g = pd.concat([old[~old.seed.isin(g.seed.unique())], g], ignore_index=True)
        g.to_parquet(f)
