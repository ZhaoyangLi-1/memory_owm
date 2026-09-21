"""P5c (repair): index-align an existing slot cache in place, CPU only — no re-extraction needed.

Caches written before `VideoSaur.encode()` aligned its output have slot indices that are permuted at every step
(see `owm.wm.videosaur_utils.align_slots`). This applies the same alignment to every sequence, keeps the old file
as <name>_unaligned.pkl and marks the new one with meta["aligned"] = True (the predictor trainer checks it).

  python scripts/p5c_align_slots.py [--slots cache/slots/wm_train_slots.pkl]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from owm.config import cache_dir
from owm.wm.videosaur_utils import align_slots


def self_match(seqs: list[np.ndarray]) -> float:
    """Fraction of slots whose nearest neighbour in the next frame has the same index."""
    hit = []
    for s in seqs:
        s = s.astype(np.float32)
        s /= np.maximum(np.linalg.norm(s, axis=-1, keepdims=True), 1e-8)
        hit.append((np.einsum("tid,tjd->tij", s[:-1], s[1:]).argmax(-1) == np.arange(s.shape[1])).mean())
    return float(np.mean(hit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default=str(cache_dir() / "slots" / "wm_train_slots.pkl"))
    a = ap.parse_args()
    path = Path(a.slots)
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data.get("meta", {}).get("aligned"):
        raise SystemExit(f"{path} 已经是对齐过的（meta.aligned = True），无需再跑。")
    probe = list(data["val"])[:200]
    before = self_match([data["val"][k] for k in probe])
    for split in ("train", "val"):
        for k in tqdm(data[split], desc=split):
            data[split][k] = align_slots(data[split][k])
    after = self_match([data["val"][k] for k in probe])
    data["meta"] = dict(data.get("meta", {}), aligned=True)
    tmp, old = path.with_suffix(".pkl.tmp"), path.with_name(path.stem + "_unaligned.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    path.rename(old)
    tmp.rename(path)
    print(f"slot i -> slot i at the next frame: {100 * before:.1f}% -> {100 * after:.1f}%  ({len(probe)} val sequences)")
    print(f"aligned: {path}\noriginal kept as: {old}")


if __name__ == "__main__":
    main()
