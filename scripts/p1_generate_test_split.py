"""P1 (plan A, spec 5.2): generate the offline TEST demonstrations with RoboMME's built-in planner for the official
val-split seeds (or for brand-new seeds), in exactly the HDF5 format of the training data.

Verified (spec 17 item 4): the repo ships no generation script any more, but the recipe survives in
tests/_shared/dataset_generation.py and in git history; scenes are a pure function of (seed, difficulty), and
RobommeRecordWrapper writes the same h5 layout (it only buffers timesteps when save_video=True).

  python scripts/p1_generate_test_split.py --gpus 2 3 4 --workers-per-gpu 2                 # official val seeds (50 / task)
  python scripts/p1_generate_test_split.py --extra 200 --seed-base 2000000 --gpus 2 3 4     # +200 new seeds / task
Output: <test_h5_dir>/episodes/<Task>_ep<i>.h5 (one file per episode, group episode_<i>) + generation_meta.json. Failure-recovery injection (used for
train episodes 0-5) is NOT applied to the test demonstrations.
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

MAX_SEED_ATTEMPTS = 20
_DIFF = ["easy", "medium", "hard"]


def _init(q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(q.get())


def _tensor_to_bool(v) -> bool:
    import numpy as np
    import torch
    if v is None:
        return False
    if isinstance(v, torch.Tensor):
        return bool(v.detach().cpu().bool().any().item())
    if isinstance(v, np.ndarray):
        return bool(np.any(v))
    return bool(v)


def run_one(env_id: str, episode: int, seed: int, difficulty: str, work: Path) -> bool:
    import gymnasium as gym
    import torch
    from robomme.env_record_wrapper import FailsafeTimeout, RobommeRecordWrapper
    import robomme.robomme_env  # noqa: F401  (registers the envs)
    from robomme.robomme_env.utils.planner_fail_safe import (FailAwarePandaArmMotionPlanningSolver,
                                                             FailAwarePandaStickMotionPlanningSolver, ScrewPlanFailure)
    from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError

    env = gym.make(env_id, obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos", render_mode="rgb_array",
                   reward_mode="dense", seed=seed, difficulty=difficulty)
    env = RobommeRecordWrapper(env, dataset=str(work), env_id=env_id, episode=episode, seed=seed, save_video=True)
    ok = False
    try:
        env.reset()
        kw = dict(debug=False, vis=False, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=False,
                  print_env_info=False)
        if env_id in ("PatternLock", "RouteStick"):
            planner = FailAwarePandaStickMotionPlanningSolver(env, joint_vel_limits=0.3, **kw)
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(env, **kw)
        screw, rrt = planner.move_to_pose_with_screw, planner.move_to_pose_with_RRTStar

        def screw_then_rrt(*args, **kwargs):
            for fn, exc in ((screw, ScrewPlanFailure), (rrt, Exception)):
                for _ in range(3):
                    try:
                        r = fn(*args, **kwargs)
                    except exc:
                        continue
                    if not (isinstance(r, int) and r == -1):
                        return r
            return -1

        planner.move_to_pose_with_screw = screw_then_rrt
        env.unwrapped.evaluate()   # MoveCube builds its task_list inside evaluate()
        for task in list(getattr(env.unwrapped, "task_list", []) or []):
            solve = task.get("solve")
            if not callable(solve):
                continue
            env.unwrapped.evaluate(solve_complete_eval=True)
            failed = False
            try:
                r = solve(env, planner)
                failed = isinstance(r, int) and r == -1
            except ScrewPlanFailure:
                failed = True
            except FailsafeTimeout:
                break
            if failed:
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            ev = env.unwrapped.evaluate(solve_complete_eval=True)
            if _tensor_to_bool(ev.get("success", False)):
                ok = True
                break
            if failed or _tensor_to_bool(ev.get("fail", False)):
                break
        else:
            ok = _tensor_to_bool(env.unwrapped.evaluate(solve_complete_eval=True).get("success", False))
        ok = ok or _tensor_to_bool(getattr(env, "episode_success", False))
    except SceneGenerationError:
        ok = False
    finally:
        try:
            env.close()   # the wrapper writes the h5 here, only for successful episodes
        except Exception:
            pass
    return ok


def job(args):
    env_id, episode, base_seed, difficulty, out_root = args
    out_root = Path(out_root)
    final = out_root / "episodes" / f"{env_id}_ep{episode}.h5"
    if final.exists():
        return env_id, episode, json.loads(final.with_suffix(".json").read_text())["seed"], "cached"
    for attempt in range(MAX_SEED_ATTEMPTS):
        seed = base_seed + attempt
        work = out_root / "work" / f"{env_id}_ep{episode}_s{seed}"
        shutil.rmtree(work, ignore_errors=True)
        try:
            ok = run_one(env_id, episode, seed, difficulty, work)
        except Exception as e:  # planner / sim hiccup: try the next seed like the official generator
            print(f"[{env_id} ep{episode}] seed {seed}: {type(e).__name__}: {e}")
            ok = False
        h5 = work / "hdf5_files" / f"{env_id}_ep{episode}_seed{seed}.h5"
        if ok and h5.exists():
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(h5), final)
            final.with_suffix(".json").write_text(json.dumps(dict(env_id=env_id, episode=episode, seed=seed,
                                                                  base_seed=base_seed, difficulty=difficulty)))
            shutil.rmtree(work, ignore_errors=True)
            return env_id, episode, seed, "ok"
        shutil.rmtree(work, ignore_errors=True)
    return env_id, episode, None, "failed"


def merge(out_root: Path, tasks):
    """No physical merge (it would duplicate ~45 GB): the reader accepts the per-episode files. Only the meta is collected."""
    meta = {}
    for task in tasks:
        files = sorted((out_root / "episodes").glob(f"{task}_ep*.h5"), key=lambda p: int(p.stem.split("_ep")[1]))
        meta[task] = [json.loads(f.with_suffix(".json").read_text()) for f in files]
        bumped = sum(m["seed"] != m["base_seed"] for m in meta[task])
        print(f"{task}: {len(files)} episodes ({bumped} with a bumped seed)")
    (out_root / "generation_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    from owm.config import h5_dir, load_cfg
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(cfg.tasks))
    ap.add_argument("--episodes", type=int, default=50, help="official val episodes per task")
    ap.add_argument("--extra", type=int, default=0, help="additional episodes with new seeds (spec 5.2 sample size)")
    ap.add_argument("--seed-base", type=int, default=2_000_000, help="new seeds: base + 10000 * task_index + 100 * i")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    out_root = h5_dir("test")
    out_root.mkdir(parents=True, exist_ok=True)
    if not a.merge_only:
        from robomme.env_record_wrapper import BenchmarkEnvBuilder
        jobs = []
        for ti, task in enumerate(a.tasks):
            b = BenchmarkEnvBuilder(env_id=task, dataset="val")
            for ep in range(min(a.episodes, b.get_episode_num())):
                seed, diff = b.resolve_episode(ep)
                jobs.append((task, ep, seed, diff, str(out_root)))
            for i in range(a.extra):   # difficulty mix as in the official splits (~50/25/25)
                diff = _DIFF[0] if i % 4 < 2 else _DIFF[1 + (i % 4 == 3)]
                jobs.append((task, a.episodes + i, a.seed_base + 10000 * ti + 100 * i, diff, str(out_root)))
        ctx = get_context("spawn")
        q = ctx.Queue()
        n = len(a.gpus) * a.workers_per_gpu
        for i in range(n):
            q.put(a.gpus[i % len(a.gpus)])
        with ProcessPoolExecutor(n, mp_context=ctx, initializer=_init, initargs=(q,)) as ex:
            for f in as_completed([ex.submit(job, j) for j in jobs]):
                print(*f.result())
    merge(out_root, a.tasks)


if __name__ == "__main__":
    main()
