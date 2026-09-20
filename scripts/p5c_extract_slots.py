"""P5c: run the frozen VideoSAUR over every world-model training / validation episode (stride-16 sequences at
several offsets) and save the slots in the C-JEPA repo's format {split: {key: [T, N, D]}}.

  python scripts/p5c_extract_slots.py [--ckpt outputs/videosaur/...ckpt] [--offsets 0 4 8 12]
Evaluation-time slots are NOT taken from this file: the adapter re-encodes the truncated history of each decision.
"""
import argparse
import pickle
import zlib

import numpy as np
from tqdm import tqdm

from owm.config import cache_dir, load_cfg, wm_tasks
from owm.data.h5_reader import list_cached
from owm.data.sampling import train_sequence_indices
from owm.data.splits import wm_split
from owm.wm.videosaur_utils import VideoSaur


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--offsets", nargs="+", type=int, default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    stride = cfg.temporal.stride
    offsets = a.offsets or (list(range(stride)) if cfg.cjepa.slot_offsets == "all" else list(cfg.cjepa.slot_offsets))
    vs = VideoSaur(a.ckpt, device=a.device)
    data = {"train": {}, "val": {}}
    eps = [e for t in (a.tasks or wm_tasks()) for e in list_cached("train", t)]
    for ep in tqdm(eps):
        split = wm_split(ep.source, ep.task, ep.idx)
        if split not in data:
            continue
        for off in offsets:
            idx = train_sequence_indices(ep.T, off, stride)
            if len(idx) < 2:
                continue
            slots = vs.encode(ep.frames[idx], seed=zlib.crc32(f"{ep.key}@{off}".encode()))
            data[split][f"{ep.key}@{off}"] = slots.astype(np.float16)
    from pathlib import Path
    out = Path(a.out) if a.out else cache_dir() / "slots" / "wm_train_slots.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(dict(data, meta=dict(ckpt=str(vs.ckpt), num_slots=vs.num_slots, stride=stride, offsets=offsets)), f)
    print({k: len(v) for k, v in data.items()}, "->", out)


if __name__ == "__main__":
    main()
