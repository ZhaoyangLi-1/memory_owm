"""P6: train LPWM on RoboMME with the STOCK training function of the LPWM repo.

Nothing in the LPWM repo is edited. The repo selects datasets through `get_video_dataset(ds, ...)`; this launcher
wraps that factory in-process so that ds == "robomme" returns our dataset, then calls the repo's `train_ddlp`.

  single GPU : CUDA_VISIBLE_DEVICES=2 python -m owm.wm.lpwm_train [--config configs/lpwm_robomme.json] [--actions]
  multi GPU  : CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes 3 [--mixed_precision bf16] \
                   -m owm.wm.lpwm_train --accelerate

timestep_horizon is derived from configs/experiment.yaml (context_frames + n_future - 1) so that LPWM and C-JEPA
share the context length; use scripts/p6_lpwm_memory_probe.py to find what fits in GPU memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from owm.config import ROOT, cache_dir, load_cfg, output_dir, resolve
from owm.wandb_utils import Run
from owm.wandb_utils import init as wandb_init


def lpwm_sys_path() -> Path:
    """Import LPWM's top-level modules (models, modules, utils, datasets, eval). A foreign package called `utils` or
    `datasets` earlier on sys.path would shadow them, so the repo goes first and stale modules are dropped."""
    repo = resolve(load_cfg().paths.lpwm_repo)
    sys.path[:] = [str(repo)] + [p for p in sys.path if p != str(repo)]
    for name in ("utils", "datasets", "eval", "modules", "models"):
        mod = sys.modules.get(name)
        if mod is not None and not str(getattr(mod, "__file__", "") or getattr(mod, "__path__", [""])[0]).startswith(str(repo)):
            del sys.modules[name]
    return repo


def patch_wandb_logging(trainer, wb):
    """Log the repo's own per-epoch numbers to wandb without editing the LPWM repo.

    `train_ddlp` builds its epoch summary via `format_epoch_summary(epoch=..., loss=..., ...)` — every metric
    arrives as a keyword argument, so wrapping that function gives clean numbers (no log parsing). The validation
    loss is computed after the summary, so it is picked up from the `validation loss: X` line that goes to
    `log_line`. Both names are module-level imports in the trainer, so patching the trainer module is enough."""
    import re

    state = {"epoch": 0}
    original_summary, original_log_line = trainer.format_epoch_summary, trainer.log_line

    def format_epoch_summary(**kw):
        state["epoch"] = int(kw.get("epoch", state["epoch"]))
        metrics = {f"train/{k}": v for k, v in kw.items() if k != "epoch" and isinstance(v, (int, float))}
        if "obj_on" in kw:
            metrics["train/on_l1"] = kw["obj_on"]      # mean number of visible particles (spec 7.3 acceptance)
        wb.log(dict(metrics, epoch=state["epoch"]), step=state["epoch"])
        return original_summary(**kw)

    def log_line(log_dir, line, *args, **kwargs):
        m = re.search(r"validation loss: ([\d.eE+-]+)", str(line))
        if m:
            wb.log({"val/loss": float(m.group(1)), "epoch": state["epoch"]}, step=state["epoch"])
        return original_log_line(log_dir, line, *args, **kwargs)

    trainer.format_epoch_summary, trainer.log_line = format_epoch_summary, log_line


def patch_dataset_factory(with_actions: bool):
    import datasets.get_dataset as gd
    from owm.wm.lpwm_dataset import RoboMMEVideoDataset
    original = gd.get_video_dataset

    def get_video_dataset(ds, root, seq_len=1, mode="train", image_size=128):
        if ds == "robomme":
            return RoboMMEVideoDataset(root, mode=mode, sample_length=seq_len, image_size=image_size,
                                       with_actions=with_actions)
        return original(ds, root, seq_len=seq_len, mode=mode, image_size=image_size)

    gd.get_video_dataset = get_video_dataset
    return get_video_dataset


def resolved_config(base: str, with_actions: bool, overrides: dict) -> dict:
    cfg = load_cfg()
    conf = json.loads(resolve(base).read_text())
    conf.pop("_comment", None)
    conf["root"] = str(cache_dir() / "episodes")
    conf["timestep_horizon"] = cfg.temporal.context_frames + len(cfg.temporal.future_offsets) - 1
    if with_actions:
        from owm.wm.actions import ACTION_TOKEN_DIM
        conf.update(action_condition=True, action_dim=ACTION_TOKEN_DIM, run_prefix=conf.get("run_prefix", "") + "_act")
    conf.update(overrides)
    conf["cond_steps"] = min(conf["cond_steps"], conf["timestep_horizon"])
    conf["animation_horizon"] = max(conf["cond_steps"] + len(cfg.temporal.future_offsets), min(conf["animation_horizon"], conf["timestep_horizon"]))
    return conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=load_cfg().lpwm.base_config)
    ap.add_argument("--actions", action="store_true", help="optional experiment (spec 14): official action conditioning")
    ap.add_argument("--accelerate", action="store_true", help="use the repo's multi-GPU trainer (run under `accelerate launch`)")
    ap.add_argument("--probe", action="store_true", help="memory probe: write to outputs/lpwm_probe instead")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON", help="override config fields, e.g. batch_size=2 num_epochs=50")
    a = ap.parse_args()
    overrides = {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in a.set}

    run_root = output_dir() / ("lpwm_probe" if a.probe else "lpwm_act" if a.actions else "lpwm")
    run_root.mkdir(parents=True, exist_ok=True)
    conf = resolved_config(a.config, a.actions, overrides)
    conf_path = run_root / "resolved_config.json"
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        conf_path.write_text(json.dumps(conf, indent=2))

    repo = lpwm_sys_path()
    # the repo resolves "eval/lpips/vgg.pth" and its log directory relative to the working directory
    (run_root / "eval").mkdir(exist_ok=True)
    link = run_root / "eval" / "lpips"
    if not link.exists():
        try:
            link.symlink_to(repo / "eval" / "lpips")
        except FileExistsError:
            pass
    os.chdir(run_root)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if a.accelerate:
        import train_lpwm_accelerate as trainer   # this module overwrites CUDA_VISIBLE_DEVICES at import time
        if visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        import train_lpwm as trainer
    factory = patch_dataset_factory(a.actions)
    import eval.eval_gen_metrics
    import eval.eval_model
    for mod in (trainer, eval.eval_model, eval.eval_gen_metrics):
        mod.get_video_dataset = factory
    import atexit
    import torch
    atexit.register(lambda: print(f"[owm] peak CUDA memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB"))
    # only rank 0 gets a wandb run, otherwise every DDP process would create its own
    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0 and not a.probe
    wb = wandb_init(f"lpwm{'_act' if a.actions else ''}", "lpwm", group="lpwm",
                    tags=["actions"] if a.actions else [], config=conf) if is_main else Run()
    patch_wandb_logging(trainer, wb)
    atexit.register(wb.finish)
    print(f"[owm] LPWM run dir: {run_root}  T={conf['timestep_horizon']}  image={conf['image_size']}  actions={a.actions}")
    trainer.train_ddlp(str(conf_path))


if __name__ == "__main__":
    main()
