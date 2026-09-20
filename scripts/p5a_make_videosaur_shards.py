"""P5a: webdataset shards for the stock VideoSAUR trainer.

One sample per (episode, offset): {"video.npy": uint8 [T_s, 256, 256, 3]} — the stride-16 sub-sampled sequence.
The stock pipeline accepts ndarray videos directly (pipelines.ensure_video_array), so no mp4 / torchcodec
is involved; the 256 -> 252 resize and ImageNet normalisation are done by the stock transforms.

  python scripts/p5a_make_videosaur_shards.py --offsets 0 8
"""
import argparse
import io
import json
import tarfile

import numpy as np
from tqdm import tqdm

from owm.config import cache_dir, load_cfg, wm_tasks
from owm.data.h5_reader import list_cached
from owm.data.sampling import train_sequence_indices
from owm.data.splits import wm_split


def _add(tar, name, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", nargs="+", type=int, default=[0, 8])
    ap.add_argument("--per-shard", type=int, default=32)
    ap.add_argument("--min-frames", type=int, default=6, help="chunk_size of the VideoSAUR config")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: episodes per task")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stride = load_cfg().temporal.stride
    from pathlib import Path
    out = Path(a.out) if a.out else cache_dir() / "videosaur_shards"
    samples = {"train": [], "val": []}
    for task in (a.tasks or wm_tasks()):
        for ep in list_cached("train", task)[: a.limit]:
            s = wm_split(ep.source, ep.task, ep.idx)
            if s in samples:
                samples[s] += [(ep, o) for o in (a.offsets if s == "train" else a.offsets[:1])]
    rng = np.random.default_rng(0)
    counts = {}
    for split, items in samples.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        order = rng.permutation(len(items))
        n_shards, tar, n_written = 0, None, 0
        for n, i in enumerate(tqdm(order, desc=split)):
            ep, off = items[i]
            idx = train_sequence_indices(ep.T, off, stride)
            if len(idx) < a.min_frames:
                continue
            if n_written % a.per_shard == 0:
                if tar:
                    tar.close()
                tar = tarfile.open(out / split / f"robomme-{split}-{n_shards:06d}.tar", "w")
                n_shards += 1
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(ep.frames[idx]))
            key = f"{ep.task}_ep{ep.idx:03d}_o{off:02d}"
            _add(tar, f"{key}.video.npy", buf.getvalue())
            _add(tar, f"{key}.meta.json", json.dumps(dict(key=ep.key, offset=off, n=len(idx))).encode())
            n_written += 1
        if tar:
            tar.close()
        counts[split] = dict(samples=n_written, shards=n_shards)
    (out / "info.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))
    print(f"train_shards: {out}/train/robomme-train-{{000000..{counts['train']['shards'] - 1:06d}}}.tar")
    print(f"val_shards:   {out}/val/robomme-val-{{000000..{counts['val']['shards'] - 1:06d}}}.tar   val_size={counts['val']['samples']}")


if __name__ == "__main__":
    main()
