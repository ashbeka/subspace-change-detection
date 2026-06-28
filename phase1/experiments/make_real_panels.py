"""Render REAL xBD-S12 disaster panels: pre / post / damage GT / projector candidate map / IR-MAD."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phase1.data.xbd_s12 import load_normalization, read_s2_pair, load_metadata
from phase1.scripts.compare_oscd_spatial_subspaces import band_image_spatial_control_values
from phase1.baselines.ir_mad import ir_mad_score

XBD = ROOT / "data" / "xbd_s12"
MASKS = XBD / "project_damage_masks_majority_v1"
OUT = ROOT / "phase1" / "outputs" / "seminar_figures" / "real_panels"
RGB = (3, 2, 1)  # B04,B03,B02 in the 12-band order
DMG_CMAP = ListedColormap(["#1a1a1a", "#2c7", "#fd0", "#f80", "#e22"])  # bg, intact, minor, major, destroyed


def stretch(img):
    out = np.zeros_like(img)
    for c in range(3):
        v = img[..., c]; lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        out[..., c] = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    return out


def projector_map(pre, post, valid, rank=11):
    rows, cols = np.where(valid)
    if rows.size < 20:
        return np.zeros(valid.shape, np.float32)
    A = pre[:, rows, cols].T.astype(np.float32); B = post[:, rows, cols].T.astype(np.float32)
    vals = band_image_spatial_control_values(A, B, rank=rank, seed=1234, mode="projector_distance")
    out = np.zeros(valid.shape, np.float32); out[rows, cols] = vals
    return out


def norm01(m, valid):
    out = np.zeros_like(m); v = m[valid]
    if v.size:
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        out[valid] = np.clip((m[valid] - lo) / (hi - lo + 1e-9), 0, 1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    low, high = load_normalization(XBD)
    records = load_metadata(XBD)
    by_uid = {r.uid: r for r in records}
    # rank patches by damaged-pixel count (classes 2-4)
    cand = []
    for fp in sorted(MASKS.glob("*_mask.npy")):
        uid = fp.name.replace("_mask.npy", "")
        m = np.load(fp)
        dmg = int(((m >= 2) & (m <= 4)).sum())
        if dmg > 250:
            cand.append((dmg, uid, fp))
    cand.sort(reverse=True)
    # pick a few from different events
    picked, seen = [], set()
    for dmg, uid, fp in cand:
        ev = uid.split("_")[0]
        if ev in seen and len(seen) >= 4:
            continue
        seen.add(ev); picked.append((uid, fp))
        if len(picked) >= 4:
            break
    print(f"picked patches: {[u for u, _ in picked]}", flush=True)

    for uid, fp in picked:
        try:
            pre, post, valid = read_s2_pair(XBD, uid, low, high)
        except Exception as e:
            print(f"  skip {uid}: {e}"); continue
        mask = np.load(fp)
        dmg_disp = np.clip(mask, 0, 4)  # 0..4 for cmap; 5,6 fold into bg visually
        proj = norm01(projector_map(pre, post, valid), valid)
        try:
            irm = norm01(ir_mad_score(pre, post, valid, iters=10), valid)
        except Exception:
            irm = np.zeros(valid.shape, np.float32)
        panels = [
            (stretch(np.stack([pre[i] for i in RGB], -1)), "before (RGB)", None),
            (stretch(np.stack([post[i] for i in RGB], -1)), "after (RGB)", None),
            (dmg_disp, "damage (minor/major/destroyed)", DMG_CMAP),
            (proj, "projector candidate map (ours)", "inferno"),
            (irm, "IR-MAD (classical)", "inferno"),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.3))
        for ax, (im, title, cm) in zip(axes, panels):
            ax.imshow(im, cmap=cm) if cm else ax.imshow(im)
            ax.set_title(title, fontsize=12); ax.axis("off")
        fig.suptitle(f"xBD-S12  {uid}   ({uid.split('_')[0].replace('-', ' ')})", fontsize=13, y=1.02)
        fig.tight_layout()
        out = OUT / f"xbd_{uid}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out.name}", flush=True)
    print(f"\nreal xBD panels -> {OUT}")


if __name__ == "__main__":
    main()
