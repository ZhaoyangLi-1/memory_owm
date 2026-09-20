"""P0: dump front_rgb + per-frame labels of every episode from the h5 files into cache/episodes.

  python scripts/p0_extract_cache.py --source train --tasks all --workers 16
  python scripts/p0_extract_cache.py --source test  --tasks eval            # after p1_generate_test_split.py
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from owm.config import ALL_TASKS, h5_path, load_cfg
from owm.data.h5_reader import episode_ids, extract_episode


def _one(args):
    source, task, idx, overwrite = args
    extract_episode(source, task, idx, overwrite)
    return task, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--tasks", nargs="+", default=["all"], help="'all' (16 tasks), 'eval' (6 tasks) or task names")
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    tasks = ALL_TASKS if a.tasks == ["all"] else list(load_cfg().tasks) if a.tasks == ["eval"] else a.tasks
    jobs = []
    for task in tasks:
        ids = episode_ids(a.source, task)
        if not ids:
            print(f"[skip] no h5 episodes for {a.source}/{task}")
            continue
        jobs += [(a.source, task, i, a.overwrite) for i in ids if a.episodes is None or i in a.episodes]
    print(f"{len(jobs)} episodes")
    with ProcessPoolExecutor(a.workers) as ex:
        for _ in tqdm(as_completed([ex.submit(_one, j) for j in jobs]), total=len(jobs)):
            pass


if __name__ == "__main__":
    main()
