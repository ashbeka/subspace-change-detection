"""
Qualitative panel: where the DS-fusion prior fixes the baseline CNN's errors.

Trains M0 (bands only) and M4 (bands + DS + smoothed-PCA + IR-MAD), one seed,
then for selected OSCD test cities renders, per city:
  [RGB pre | RGB post | GT | M0 pred | M4 pred | fix map]
with TP=white, TN=black, FP=red, FN=blue; the fix map shows where M4 corrects M0
(green = M4 fixes an M0 error, red = M4 introduces a new error).

This is the interpretability figure for the seminar -- the "what does the DS-fusion
prior actually change?" slide.

Run:
  .\.venv\Scripts\python.exe -m phase1.experiments.ds_prior_panel \
      --cities norcia,lasvegas,montpellier,brasilia --steps 2500 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics as skm

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data.oscd_dataset import OFFICIAL_TRAIN, OFFICIAL_TEST
from phase1.data.preprocessing import load_band_stats
from phase1.experiments.unet_ds_prior import (
    BAND_ORDER, CONFIG_CHANNELS, OUT_ROOT, assemble_input, infer_city,
    precompute_city, train_one,
)

RGB = (3, 2, 1)  # B04,B03,B02 in BAND_ORDER


def rgb_disp(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    img = np.stack([x[i] for i in RGB], axis=-1).astype(np.float32)
    out = np.zeros_like(img)
    for c in range(3):
        v = img[..., c][valid]
        if v.size == 0:
            continue
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        if hi > lo:
            out[..., c] = np.clip((img[..., c] - lo) / (hi - lo), 0, 1)
    out[~valid] = 0
    return out


def confusion_rgb(pred: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((*y.shape, 3), np.float32)
    tp = valid & (pred == 1) & (y == 1)
    tn = valid & (pred == 0) & (y == 0)
    fp = valid & (pred == 1) & (y == 0)
    fn = valid & (pred == 0) & (y == 1)
    out[tp] = [1, 1, 1]      # white
    out[tn] = [0, 0, 0]      # black
    out[fp] = [1, 0, 0]      # red
    out[fn] = [0, 0.4, 1]    # blue
    return out


def fix_rgb(p0, p4, y, valid) -> np.ndarray:
    out = np.full((*y.shape, 3), 0.15, np.float32)
    out[~valid] = 0
    m0_correct = (p0 == y)
    m4_correct = (p4 == y)
    out[valid & ~m0_correct & m4_correct] = [0, 1, 0]   # green: M4 fixes M0 error
    out[valid & m0_correct & ~m4_correct] = [1, 0, 0]    # red: M4 regression
    return out


def best_thr(prob, y, valid):
    s = prob[valid].ravel(); yy = y[valid].ravel().astype(np.uint8)
    if np.unique(yy).size < 2:
        return 0.5
    p, r, t = skm.precision_recall_curve(yy, s)
    f1 = (2 * p * r) / (p + r + 1e-12)
    bi = int(np.nanargmax(f1))
    return float(t[min(bi, t.size - 1)]) if t.size else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oscd_root", type=Path, default=ROOT / "data" / "OSCD")
    ap.add_argument("--stats_path", type=Path, default=ROOT / "phase1" / "data" / "oscd_band_stats.json")
    ap.add_argument("--cities", default="norcia,lasvegas,montpellier,brasilia")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = load_band_stats(args.stats_path)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    train_recs = [precompute_city(c, "train", args.oscd_root, stats, 1234) for c in OFFICIAL_TRAIN]
    test_recs = {c: precompute_city(c, "test", args.oscd_root, stats, 1234) for c in cities}

    targs = SimpleNamespace(steps=args.steps, batch=16, patch=96, base=args.base, lr=1e-3, p_change=0.6)
    models = {}
    for cfg in ["bands", "bands_fusion"]:
        chans = CONFIG_CHANNELS[cfg]
        tr_in = [assemble_input(r, chans) for r in train_recs]
        print(f"[train] {cfg} ...", flush=True)
        models[cfg] = (train_one(cfg, args.seed, train_recs, tr_in, targs, device), chans)

    outdir = OUT_ROOT / "panels"
    outdir.mkdir(parents=True, exist_ok=True)
    for city, rec in test_recs.items():
        valid = rec["valid"].astype(bool); y = rec["y"].astype(np.uint8)
        probs = {}
        for cfg, (model, chans) in models.items():
            probs[cfg] = infer_city(model, assemble_input(rec, chans), device)
        t0 = best_thr(probs["bands"], y, valid)
        t4 = best_thr(probs["bands_fusion"], y, valid)
        p0 = (probs["bands"] >= t0).astype(np.uint8)
        p4 = (probs["bands_fusion"] >= t4).astype(np.uint8)
        ap0 = skm.average_precision_score(y[valid].ravel(), probs["bands"][valid].ravel())
        ap4 = skm.average_precision_score(y[valid].ravel(), probs["bands_fusion"][valid].ravel())

        panels = [
            (rgb_disp(rec["x1"], valid), "pre (RGB)"),
            (rgb_disp(rec["x2"], valid), "post (RGB)"),
            (confusion_rgb(y, y, valid), "ground truth"),
            (confusion_rgb(p0, y, valid), f"M0 bands  AP={ap0:.3f}"),
            (confusion_rgb(p4, y, valid), f"M4 +DS-fusion  AP={ap4:.3f}"),
            (fix_rgb(p0, p4, y, valid), "fix map (green=fixed)"),
        ]
        fig, axes = plt.subplots(1, 6, figsize=(22, 4))
        for axx, (im, title) in zip(axes, panels):
            axx.imshow(im); axx.set_title(title, fontsize=10); axx.axis("off")
        fig.suptitle(f"{city}  (TP=white TN=black FP=red FN=blue)", fontsize=12)
        fig.tight_layout()
        fp = outdir / f"panel_{city}.png"
        fig.savefig(fp, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {fp}  (M0 AP {ap0:.3f} -> M4 AP {ap4:.3f})", flush=True)
    print(f"\npanels -> {outdir}")


if __name__ == "__main__":
    main()
