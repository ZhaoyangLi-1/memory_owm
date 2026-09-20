"""Assemble candidate / goal / evidence tokens and labels for the readout head (spec 9.1, 9.7).

Conditions
  floor                 no evidence
  ceiling               evidence = gt_future
  cjepa / lpwm / ...    evidence = cache/evidence/<name>
  check_<name>          prediction check: candidates without type/colour, non-critical pointer decisions only,
                        pointer loss only
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from owm.config import load_cfg
from owm.data.decision_index import load_index, usable
from owm.data.gt_state import GT
from owm.wm.evidence import D_GT, NO_ID, build_gt_future, gt_token_features, load_evidence


def evidence_name(condition: str) -> str | None:
    c = condition.removeprefix("check_")
    return {"floor": None, "ceiling": "gt_future"}.get(c, c)


def is_check(condition: str) -> bool:
    return condition.startswith("check_")


def ck_get(ckpt: dict, key: str):
    return ckpt[key]


@dataclass
class Sample:
    row: dict
    cand: np.ndarray        # [Nc, 15] (arm last)
    cand_id: np.ndarray
    cand_uv: np.ndarray
    cand_type: np.ndarray   # 0 object / 1 arm
    cand_obj: np.ndarray    # GT object index per candidate (-1 for the arm)
    goal: np.ndarray
    y_label: int
    y_cand: int             # index into the candidate list (-1 if no parameter)
    evid: np.ndarray | None  # [S, K, n, D]
    evid_uv: np.ndarray | None
    evid_valid: np.ndarray | None
    evid_id: np.ndarray | None


class DecisionData:
    """All decisions of one task for one condition, held in memory."""

    def __init__(self, task: str, condition: str, df: pd.DataFrame | None = None, options: list[dict] | None = None):
        self.task, self.condition = task, condition
        self.evid_name = evidence_name(condition)
        self.blind = is_check(condition)
        df = usable(load_index()) if df is None else df
        df = df[df.task == task]
        if self.blind:
            parent = df["parent_type"] if "parent_type" in df else df["decision_type"]
            df = df[(parent == "non_critical") & df.need_param]   # incl. the training-only pseudo-decisions of those
        self.df = df.reset_index(drop=True)
        if options is None:
            import yaml
            from owm.config import ROOT
            options = yaml.safe_load((ROOT / "configs" / "option_vocab.yaml").read_text())[task]["options"]
        self.labels = [o["label"] for o in options]
        self.need_param = torch.tensor([bool(o["need_parameter"]) for o in options])
        self.samples = [self._build(r) for r in self.df.to_dict("records")]
        self.d_goal = len(self.samples[0].goal) if self.samples else 1
        self.d_evid = self.samples[0].evid.shape[-1] if self.samples and self.samples[0].evid is not None else 0
        self.stats: dict = {}

    _gt_cache: dict = {}

    @classmethod
    def live(cls, task: str, condition: str, ckpt: dict, options: list[dict]) -> "DecisionData":
        """Shell without stored decisions, for closed-loop use: normalisation statistics and dims come from a
        trained head's checkpoint; samples are built on the fly with `live_sample` and batched with `collate_items`."""
        self = cls.__new__(cls)
        self.task, self.condition = task, condition
        self.evid_name, self.blind = evidence_name(condition), is_check(condition)
        self.labels = [o["label"] for o in options]
        self.need_param = torch.tensor([bool(o["need_parameter"]) for o in options])
        self.samples, self.df = [], pd.DataFrame()
        self.d_goal, self.d_evid, self.stats = ck_get(ckpt, "d_goal"), ck_get(ckpt, "d_evid"), ckpt["stats"]
        return self

    def live_sample(self, tok: dict, goal: np.ndarray, evidence: tuple | None) -> Sample:
        feat, uv, ids, is_arm = gt_token_features(tok, blind=self.blind)
        ev = (None, None, None, None)
        if evidence is not None:
            tokens, euv, valid = evidence
            ev = (tokens, euv, valid, np.full(tokens.shape[:3], NO_ID, np.int64))
        return Sample(row={}, cand=feat, cand_id=ids, cand_uv=uv, cand_type=is_arm.astype(np.int64),
                      cand_obj=np.concatenate([tok["obj_index"], [-1]]), goal=np.asarray(goal, np.float32), y_label=-1,
                      y_cand=-1, evid=ev[0], evid_uv=ev[1], evid_valid=ev[2], evid_id=ev[3])

    def _gt(self, source, task, episode) -> GT:
        k = (source, task, episode)
        if k not in self._gt_cache:
            self._gt_cache[k] = GT(*k)
        return self._gt_cache[k]

    def _build(self, r: dict) -> Sample:
        gt = self._gt(r["source"], r["task"], r["episode"])
        tok = gt.tokens(r["t"])
        feat, uv, ids, is_arm = gt_token_features(tok, blind=self.blind)
        cand_obj = np.concatenate([tok["obj_index"], [-1]])
        y_cand = int(np.where(cand_obj == r["target_obj"])[0][0]) if r["need_param"] else -1
        ev = dict(tokens=None, uv=None, valid=None, ids=None)
        if self.evid_name == "gt_future":
            ev = build_gt_future(gt, r["t"], r["n_frames"])
        elif self.evid_name is not None:
            ev = load_evidence(self.evid_name, r["source"], r["task"], r["episode"], r["t"])
        return Sample(row=r, cand=feat, cand_id=ids, cand_uv=uv, cand_type=is_arm.astype(np.int64), cand_obj=cand_obj,
                      goal=np.asarray(json.loads(r["goal_vec"]), np.float32), y_label=self.labels.index(r["label"]),
                      y_cand=y_cand, evid=ev["tokens"], evid_uv=ev["uv"], evid_valid=ev["valid"], evid_id=ev["ids"])

    # ------------------------------------------------------------------ normalisation (train statistics only)
    def fit_stats(self, idx: np.ndarray) -> dict:
        pos = np.concatenate([self.samples[i].cand[:, 10:13] for i in idx])
        st = dict(pos_mean=pos.mean(0), pos_std=pos.std(0) + 1e-6)
        if self.d_evid and self.evid_name != "gt_future":
            ev = np.concatenate([self.samples[i].evid[self.samples[i].evid_valid] for i in idx])
            st.update(evid_mean=ev.mean(0), evid_std=ev.std(0) + 1e-6)
        self.stats = {k: v.astype(np.float32) for k, v in st.items()}
        return self.stats

    def _norm_gt(self, f: np.ndarray) -> np.ndarray:
        f = f.copy()
        f[..., 10:13] = (f[..., 10:13] - self.stats["pos_mean"]) / self.stats["pos_std"]
        return f

    def _norm_evid(self, e: np.ndarray) -> np.ndarray:
        if self.evid_name == "gt_future":
            return self._norm_gt(e)
        return (e - self.stats["evid_mean"]) / self.stats["evid_std"]

    # ------------------------------------------------------------------ batching
    def split_indices(self, split: str) -> np.ndarray:
        return np.where(self.df["split"].to_numpy() == split)[0]

    def collate(self, idx, sample_idx: int | None = None, rng: np.random.Generator | None = None, device="cpu"):
        """sample_idx: which evidence sample (LPWM has 5) to use; None = random (training).
        rng given -> the candidate order is shuffled (training), so list position carries no information."""
        return self.collate_items([self.samples[i] for i in idx], sample_idx, rng, device)

    def collate_items(self, items: list, sample_idx: int | None = None, rng: np.random.Generator | None = None, device="cpu"):
        B, Nc = len(items), max(len(s.cand) for s in items)
        cand = np.zeros((B, Nc, D_GT), np.float32)
        cand_id = np.full((B, Nc), NO_ID, np.int64)
        cand_uv = np.zeros((B, Nc, 2), np.float32)
        cand_type = np.zeros((B, Nc), np.int64)
        cand_pad = np.ones((B, Nc), bool)
        cand_obj = np.full((B, Nc), -1, np.int64)
        y_cand = np.full(B, -1, np.int64)
        resample = rng is not None and load_cfg().readout.get("resample_ids", True)
        id_maps = []
        for b, s in enumerate(items):
            n = len(s.cand)
            perm = rng.permutation(n) if rng is not None else np.arange(n)
            # Training only: re-draw the random identity numbers on every draw (consistently for the candidates and the
            # GT-future evidence of this sample). With numbers fixed per episode the target's number is a perfect
            # memorisation key for ~100 training episodes and the ceiling head never learns to read the evidence.
            id_map = np.concatenate([rng.permutation(32), [32, 33]]) if resample else np.arange(34)
            id_maps.append(id_map)
            cand[b, :n], cand_id[b, :n], cand_uv[b, :n] = self._norm_gt(s.cand)[perm], id_map[s.cand_id[perm]], s.cand_uv[perm]
            cand_type[b, :n], cand_obj[b, :n], cand_pad[b, :n] = s.cand_type[perm], s.cand_obj[perm], False
            if s.y_cand >= 0:
                y_cand[b] = int(np.where(perm == s.y_cand)[0][0])
        out = dict(cand=cand, cand_id=cand_id, cand_uv=cand_uv, cand_type=cand_type, cand_pad=cand_pad,
                   goal=np.stack([s.goal for s in items]), option_avail=np.ones((B, len(self.labels)), bool))
        if self.d_evid:
            Ne = max(s.evid.shape[1] * s.evid.shape[2] for s in items)
            evid = np.zeros((B, Ne, self.d_evid), np.float32)
            evid_uv, evid_id = np.zeros((B, Ne, 2), np.float32), np.full((B, Ne), NO_ID, np.int64)
            evid_step, evid_pad = np.ones((B, Ne), np.int64), np.ones((B, Ne), bool)
            for b, s in enumerate(items):
                S, K, n, _ = s.evid.shape
                si = (rng.integers(S) if rng is not None else 0) if sample_idx is None else min(sample_idx, S - 1)
                m = K * n
                evid[b, :m] = self._norm_evid(s.evid[si]).reshape(m, -1)
                evid_uv[b, :m] = np.clip(s.evid_uv[si].reshape(m, 2), 0.0, 1.0)
                evid_id[b, :m] = id_maps[b][s.evid_id[si].reshape(m)]
                evid_step[b, :m] = np.repeat(np.arange(1, K + 1), n)
                evid_pad[b, :m] = ~s.evid_valid[si].reshape(m)
            evid[evid_pad] = 0.0
            out.update(evid=evid, evid_uv=evid_uv, evid_id=evid_id, evid_step=evid_step, evid_pad=evid_pad)
        batch = SimpleNamespace(**{k: torch.from_numpy(v).to(device) for k, v in out.items()})
        y = SimpleNamespace(label=torch.tensor([max(s.y_label, 0) for s in items], device=device),
                            cand=torch.from_numpy(y_cand).to(device), cand_obj=cand_obj)
        return batch, y

    def n_samples_evidence(self) -> int:
        return max((s.evid.shape[0] for s in self.samples if s.evid is not None), default=1)
