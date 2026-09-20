"""Single place where Weights & Biases is configured for every training in this project.

Settings live in configs/experiment.yaml -> wandb. Every trainer calls `init(...)`; if wandb is disabled,
not installed, or fails to start, `init` returns None and all logging calls become no-ops, so a training
run never dies because of logging.

Runs are organised as:
  project   configs/experiment.yaml -> wandb.project
  job_type  videosaur | cjepa_predictor | lpwm | readout
  group     one group per logical unit (e.g. "PickXtimes/ceiling"), so seeds are averaged in the UI
"""
from __future__ import annotations

import os
from typing import Any

from owm.config import load_cfg


def wandb_config() -> dict:
    cfg = load_cfg()
    w = dict(cfg.get("wandb") or {})
    w.setdefault("enable", False)
    w.setdefault("project", "owm_memory")
    w.setdefault("entity", None)
    w.setdefault("mode", "online")
    w.setdefault("tags", [])
    return w


def enabled() -> bool:
    if os.environ.get("OWM_NO_WANDB"):
        return False
    return bool(wandb_config()["enable"])


class Run:
    """Thin wrapper so callers never have to check for None or catch logging errors."""

    def __init__(self, run=None, module=None):
        self._run, self._wandb = run, module

    def __bool__(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict, step: int | None = None, commit: bool = True) -> None:
        if self._run is None:
            return
        clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool)) and v == v}
        if not clean:
            return
        try:
            self._wandb.log(clean, step=step, commit=commit)
        except Exception as e:  # never let logging kill a training run
            print(f"[owm.wandb] log failed: {type(e).__name__}: {e}")

    def summary(self, values: dict) -> None:
        if self._run is None:
            return
        try:
            self._run.summary.update(values)
        except Exception as e:
            print(f"[owm.wandb] summary update failed: {type(e).__name__}: {e}")

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            pass
        self._run = None


def init(name: str, job_type: str, config: dict | None = None, group: str | None = None,
         tags: list[str] | None = None, reinit: bool = True) -> Run:
    """Start a run, or return a disabled Run if wandb is off / unavailable."""
    if not enabled():
        return Run()
    try:
        import wandb
    except ImportError:
        print("[owm.wandb] wandb is not installed — logging disabled")
        return Run()
    w = wandb_config()
    try:
        run = wandb.init(project=w["project"], entity=w["entity"], name=name, group=group, job_type=job_type,
                         mode=os.environ.get("WANDB_MODE", w["mode"]), tags=list(w["tags"]) + list(tags or []),
                         config=config or {}, reinit=reinit)
    except Exception as e:
        print(f"[owm.wandb] init failed ({type(e).__name__}: {e}) — continuing without logging")
        return Run()
    print(f"[owm.wandb] {run.url if hasattr(run, 'url') else name}")
    return Run(run, wandb)


def flat_config(obj: Any, prefix: str = "") -> dict:
    """Nested dict / config -> flat {'a.b': value} suitable for wandb config."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat_config(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        out[prefix.rstrip(".")] = list(obj)
    else:
        out[prefix.rstrip(".")] = obj
    return out
