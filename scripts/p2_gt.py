"""P2: replay every cached episode of the evaluated tasks to extract GT state, then report
replay consistency, target matching and n_max (-> number of VideoSAUR slots N = n_max + 3).

  python scripts/p2_gt.py --gpus 2 3 4 --workers-per-gpu 3
  python scripts/p2_gt.py --report-only
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
from tqdm import tqdm


def _init(gpu_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get())


def _one(job):
    from owm.data.gt_state import replay_episode
    from owm.data.h5_reader import Episode
    source, task, idx, overwrite = job
    try:
        replay_episode(Episode(source, task, idx), overwrite)
        return job, None
    except Exception as e:  # keep going; failures are listed in the report
        return job, f"{type(e).__name__}: {e}"


def report():
    from owm.config import load_cfg, output_dir
    from owm.data.decision_index import build_index, coverage_report, index_path
    from owm.data.gt_state import GT, gt_path
    from owm.data.h5_reader import list_cached
    cfg = load_cfg()
    lines, n_max_hard = [], 0
    for task in cfg.tasks:
        for source in ("train", "test"):
            eps = [e for e in list_cached(source, task) if gt_path(source, task, e.idx).exists()]
            if not eps:
                continue
            diffs, oks, nmax = [], [], []
            for e in eps:
                g = GT(source, task, e.idx)
                diffs.append(float(np.nanmean(g.d["rgb_diff"])))
                oks.append(float(g.d["replay_ok"].mean()))
                n_vis = max(int(g.visible(g.frame0 + j).sum()) for j in range(0, g.Te, 16)) + 1  # + arm
                nmax.append((e.difficulty, n_vis))
            hard = [n for d, n in nmax if d == "hard"] or [n for _, n in nmax]
            n_max_hard = max(n_max_hard, max(hard))
            lines.append(f"{source:5s} {task:17s} episodes={len(eps):3d} replay_ok={np.mean(oks):.3f} "
                         f"rgb_diff mean={np.mean(diffs):.3f} worst={np.max(diffs):.3f} "
                         f"n_visible(max, hard)={max(hard)} n_visible(max, all)={max(n for _, n in nmax)}")
    lines.append(f"\nn_max (hard episodes, incl. arm) = {n_max_hard}  ->  suggested num_slots N = n_max + 3 = {n_max_hard + 3}")
    df = build_index()
    df.to_parquet(index_path())
    lines.append(coverage_report(df))
    text = "\n".join(lines)
    print(text)
    out = output_dir() / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "p2_gt.txt").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["train", "test"])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--workers-per-gpu", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        from owm.config import load_cfg
        from owm.data.gt_state import gt_path
        from owm.data.h5_reader import list_cached
        jobs = [(s, t, e.idx, a.overwrite) for s in a.sources for t in (a.tasks or load_cfg().tasks)
                for e in list_cached(s, t) if a.overwrite or not gt_path(s, t, e.idx).exists()]
        print(f"{len(jobs)} episodes to replay")
        ctx = get_context("spawn")
        q = ctx.Queue()
        n_workers = len(a.gpus) * a.workers_per_gpu
        for i in range(n_workers):
            q.put(a.gpus[i % len(a.gpus)])
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init, initargs=(q,)) as ex:
            for f in tqdm(as_completed([ex.submit(_one, j) for j in jobs]), total=len(jobs)):
                job, err = f.result()
                if err:
                    print(f"[fail] {job[:3]}: {err}")
    report()


if __name__ == "__main__":
    main()
