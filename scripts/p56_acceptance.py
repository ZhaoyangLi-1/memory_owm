"""P5 / P6 acceptance checks of the trained world models (spec 7.3).

  python scripts/p56_acceptance.py cjepa [--episodes 20]    # val-loss plateau + slot-mask overlays + predicted-mask panels
  python scripts/p56_acceptance.py lpwm  [--episodes 20]    # val-loss plateau + on_l1 + keypoint overlays + rendered rollouts
Outputs: outputs/acceptance/<model>/...  (the visual checks are for a human; the plateau test is automatic).
The future-perturbation test is part of scripts/p7_extract_evidence.py.
"""
import argparse
import json
import re

import numpy as np
import torch
from PIL import Image

from owm.config import load_cfg, output_dir
from owm.data.decision_index import load_index, usable
from owm.data.h5_reader import Episode
from owm.data.sampling import future_indices, history_indices

PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
                    [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40],
                    [255, 250, 200], [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
                    [255, 255, 255], [0, 0, 0]] * 4, np.uint8)


def plateau(values: list[float], name: str) -> str:
    """Spec 7.3: relative change of the validation loss over the last 10% of training < 2%."""
    v = np.asarray(values, float)
    if len(v) < 10:
        return f"{name}: only {len(v)} validation points — keep training"
    k = max(int(round(0.1 * len(v))), 2)
    rel = abs(v[-k:].mean() - v[-2 * k:-k].mean()) / max(abs(v[-2 * k:-k].mean()), 1e-12)
    return f"{name}: last-10% relative change = {100 * rel:.2f}%  ->  {'PLATEAU (pass)' if rel < 0.02 else 'still changing (keep training)'}"


def overlay(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """frame [256,256,3], masks [N,g,g] soft -> colour overlay of the arg-max slot."""
    seg = masks.argmax(0)
    seg = np.asarray(Image.fromarray(seg.astype(np.uint8)).resize((256, 256), Image.NEAREST))
    return (0.5 * frame + 0.5 * PALETTE[seg]).astype(np.uint8)


def pick_decisions(n_per_task: int):
    df = usable(load_index())
    df = df[(df.decision_type == "memory_critical") & (df.split != "test")]   # never look at test episodes here
    return df.groupby("task").head(n_per_task).to_dict("records")


def run_cjepa(a):
    from owm.wm.cjepa_adapter import CJepaAdapter
    out = output_dir() / "acceptance" / "cjepa"
    out.mkdir(parents=True, exist_ok=True)
    ad = CJepaAdapter(a.predictor_ckpt, a.videosaur_ckpt, a.device)
    log = ad.ckpt.parent / "log.json"
    lines = []
    if log.exists():
        lines.append(plateau([e["val_future_mse"] for e in json.loads(log.read_text())], "C-JEPA predictor val future-MSE"))
    metrics = sorted(ad.vs.ckpt.parent.parent.glob("metrics/*/metrics.csv"))
    if metrics:
        import pandas as pd
        m = pd.read_csv(metrics[-1])
        col = [c for c in m.columns if c.startswith("val/") and "loss" in c]
        if col:
            lines.append(plateau(m[col[0]].dropna().tolist(), f"VideoSAUR {col[0]}"))
    for r in pick_decisions(a.episodes):
        ep = Episode(r["source"], r["task"], r["episode"])
        idx = history_indices(r["t"])
        slots = ad.vs.encode(np.asarray(ep.frames[idx]), seed=0)
        show = idx[-6:]
        row1 = [overlay(np.asarray(ep.frames[f]), m) for f, m in zip(show, ad.vs.decode_masks(slots[-len(show):]))]
        tokens, uv, _ = ad.predict(ep.frames, r["t"])
        fut, _ = future_indices(r["t"], ep.T)
        row2 = [overlay(np.asarray(ep.frames[f]), m) for f, m in zip(fut, ad.vs.decode_masks(tokens[0]))]   # predicted masks on the REAL future frame
        row2 += [np.zeros((256, 256, 3), np.uint8)] * (len(row1) - len(row2))
        Image.fromarray(np.concatenate([np.concatenate(row1, 1), np.concatenate(row2, 1)], 0)).save(
            out / f"{r['task']}_ep{r['episode']}_t{r['t']}.jpg", quality=85)
    lines.append(f"panels: {out}  (top: slot masks of the last history frames; bottom: masks decoded from the PREDICTED slots, "
                 f"drawn over the real t+16 / t+32 frames)")
    (out / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def run_lpwm(a):
    from owm.wm.lpwm_adapter import LpwmAdapter
    from owm.wm.lpwm_dataset import to_lpwm_frames
    out = output_dir() / "acceptance" / "lpwm"
    out.mkdir(parents=True, exist_ok=True)
    ad = LpwmAdapter(a.lpwm_run, a.lpwm_ckpt, a.device)
    lines = []
    log = ad.run_dir / "log.txt"
    if log.exists():
        txt = log.read_text()
        val = [float(v) for v in re.findall(r"^validation loss: ([\-\d\.eE+]+)", txt, flags=re.M)]
        if val:
            lines.append(plateau(val, "LPWM validation loss"))
    on_l1 = []
    for r in pick_decisions(a.episodes):
        ep = Episode(r["source"], r["task"], r["episode"])
        idx = history_indices(r["t"])[-ad.max_cond:]
        hist = to_lpwm_frames(np.asarray(ep.frames[idx]), ad.image_size)[None].to(a.device)
        with torch.no_grad():
            torch.manual_seed(0)
            rec, z = ad.model.sample_from_x(hist, num_steps=ad.n_future, deterministic=False, cond_steps=hist.shape[1],
                                            use_all_ctx=False, return_z=True, decode=True, n_pred_eq_gt=False)
        rec = (rec[0, -ad.n_future:].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        fut, _ = future_indices(r["t"], ep.T)
        S = ad.image_size
        last = np.asarray(Image.fromarray(np.asarray(ep.frames[r["t"]])).resize((S, S))).copy()
        on_l1.append(float(z["z_obj_on"][0, :hist.shape[1]].reshape(hist.shape[1], -1).sum(-1).mean()))
        pos, on = z["z_pos"][0, hist.shape[1] - 1].cpu().numpy(), z["z_obj_on"][0, hist.shape[1] - 1].reshape(-1).cpu().numpy()
        for (y, x), o in zip(pos, on):   # z_pos is (y, x) in [-1, 1]
            if o > 0.5:
                cy, cx = int((y + 1) / 2 * (S - 1)), int((x + 1) / 2 * (S - 1))
                last[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3] = (255, 255, 0)
        real = [np.asarray(Image.fromarray(np.asarray(ep.frames[f])).resize((S, S))) for f in fut]
        Image.fromarray(np.concatenate([last, *rec, *real], 1)).save(out / f"{r['task']}_ep{r['episode']}_t{r['t']}.jpg", quality=85)
    M = int(ad.hp["n_kp_prior"])
    lines.append(f"LPWM on_l1 (mean number of visible particles per frame) = {np.mean(on_l1):.1f} of M = {M}  ->  "
                 f"{'OK' if 0.05 * M < np.mean(on_l1) < 0.95 * M else 'near 0 or near M: ADJUST M (spec 7.2)'}")
    lines.append(f"panels: {out}  (decision frame with visible particles | rendered prediction t+16, t+32 | real t+16, t+32)")
    (out / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["cjepa", "lpwm"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--predictor-ckpt", default=None)
    ap.add_argument("--videosaur-ckpt", default=None)
    ap.add_argument("--lpwm-run", default=None)
    ap.add_argument("--lpwm-ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    (run_cjepa if a.model == "cjepa" else run_lpwm)(a)
