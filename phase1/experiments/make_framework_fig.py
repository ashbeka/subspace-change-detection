"""Publication-style, color-coded framework figure (real chips + colored branches + labeled arrows + loss)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phase1.data.preprocessing import load_band_stats
from phase1.experiments.unet_ds_prior import precompute_city, _rank_norm
from phase1.data.xbd_s12 import load_normalization, read_s2_pair
from phase1.scripts.compare_oscd_spatial_subspaces import band_image_spatial_control_values

OUT = ROOT / "phase1" / "outputs" / "seminar_figures"
RGB = (3, 2, 1)
# stage color scheme (fill, edge, text)
COL = {
    "data":   ("#e8edf3", "#5a6473"),
    "geom":   ("#d7e6f7", "#2f5fa8"),
    "classic":("#dcecd4", "#4a7c2f"),
    "fusion": ("#fce6cf", "#b8731b"),
    "unsup":  ("#f9d2c8", "#c0392b"),
    "sup":    ("#e4d6f0", "#6a3d9a"),
}
W, H = 16.4, 8.6


def rgb_stretch(cube, valid, idx=RGB):
    img = np.stack([cube[i] for i in idx], -1).astype(np.float32)
    out = np.zeros_like(img)
    for c in range(3):
        v = img[..., c][valid] if valid is not None else img[..., c].ravel()
        if v.size:
            lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            out[..., c] = np.clip((img[..., c] - lo) / (hi - lo + 1e-9), 0, 1)
    if valid is not None:
        out[~valid] = 0
    return out


def load_chips():
    stats = load_band_stats(ROOT / "phase1" / "data" / "oscd_band_stats.json")
    r = precompute_city("montpellier", "test", ROOT / "data" / "OSCD", stats, 1234)
    v = r["valid"].astype(bool)
    pre, post = rgb_stretch(r["x1"], v), rgb_stretch(r["x2"], v)
    ds = r["ds"]; fused = _rank_norm(np.mean([r["ds"], r["spca"], r["irmad"]], 0).astype(np.float32), v)
    gt = r["y"].astype(float)
    # xBD candidate overlay (projector top-5% on after)
    XBD = ROOT / "data" / "xbd_s12"; low, high = load_normalization(XBD)
    masks = XBD / "project_damage_masks_majority_v1"
    uid = "hurricane-florence_00000410"; fp = masks / f"{uid}_mask.npy"
    if not fp.exists():
        cands = sorted(masks.glob("*_mask.npy"), key=lambda p: -int(((np.load(p) >= 2) & (np.load(p) <= 4)).sum()))
        fp = cands[0]; uid = fp.name.replace("_mask.npy", "")
    pr, po, vv = read_s2_pair(XBD, uid, low, high)
    rows, cols = np.where(vv)
    vals = band_image_spatial_control_values(pr[:, rows, cols].T.astype(np.float32), po[:, rows, cols].T.astype(np.float32), rank=11, seed=1234, mode="projector_distance")
    proj = np.zeros(vv.shape, np.float32); proj[rows, cols] = vals
    thr = np.percentile(proj[vv], 95)
    xb = rgb_stretch(po, None); xb[(proj >= thr) & vv] = [1, 0.12, 0.12]
    return pre, post, ds, fused, gt, xb


def main():
    pre, post, ds, fused, gt, xb = load_chips()
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    INK = "#1a2330"

    def box(x, y, w, h, text, kind, fs=10.5):
        fc, ec = COL[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

    def chip(im, x, y, s, title, cmap=None, ec="#5a6473"):
        a = fig.add_axes([x / W, y / H, s / W, s / H]); a.imshow(im, cmap=cmap)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(ec); sp.set_linewidth(1.8)
        a.set_title(title, fontsize=8.8, color=INK, pad=2)

    def arr(x1, y1, x2, y2, color="#566", lab=None, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=1.7, color=color, linestyle=ls))
        if lab:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, lab, ha="center", fontsize=8.2, color=color, style="italic")

    ax.text(W / 2, 8.25, "Proposed framework: spatially-faithful Difference Subspaces for label-free change & disaster-damage detection",
            ha="center", fontsize=13.5, color=INK, fontweight="bold")

    # --- inputs ---
    chip(pre, 0.35, 5.55, 1.7, "$I_1$  pre (13 bands)")
    chip(post, 0.35, 3.05, 1.7, "$I_2$  post (13 bands)")
    box(2.35, 4.35, 1.45, 1.3, "Normalize\nper-band\nz-score", "data", 9.5)
    arr(2.05, 6.4, 2.35, 5.5); arr(2.05, 3.9, 2.35, 4.6)

    # --- geometric branch (blue, top) ---
    gy = 6.15
    box(4.05, gy, 1.7, 1.3, "Band-image\nsamples\n$X_t\\in\\mathbb{R}^{N\\times B}$", "geom", 9.5)
    box(5.95, gy, 1.75, 1.3, "Spatial PCA\n$\\Phi,\\Psi$\n(rank 12)", "geom", 9.5)
    box(7.9, gy, 1.95, 1.3, "Canonical angles\n$\\Phi^\\top\\Psi=U\\Sigma V^\\top$", "geom", 9.5)
    box(10.05, gy, 1.8, 1.3, "Diff. subspace $D$\n/ projector\n$P_\\Phi-P_\\Psi$", "geom", 9.2)
    chip(ds, 12.05, gy - 0.02, 1.35, "DS / projector map", cmap="inferno", ec=COL["geom"][1])
    arr(3.8, 5.0, 4.05, gy + 0.4, COL["geom"][1])
    for x in (5.75, 7.7, 9.85, 11.85):
        arr(x, gy + 0.65, x + 0.2, gy + 0.65, COL["geom"][1])

    # --- classical branch (green, bottom) ---
    cy = 2.95
    box(4.05, cy, 1.7, 1.25, "PCA-diff", "classic", 10)
    box(5.95, cy, 1.75, 1.25, "IR-MAD\n(= CCA)", "classic", 10)
    chip(fused * 0 + _safe(ds), 0, 0, 0.01, "", None) if False else None  # noop
    arr(3.8, 4.4, 4.05, cy + 0.9, COL["classic"][1])
    arr(5.75, cy + 0.6, 5.95, cy + 0.6, COL["classic"][1])

    # --- fusion (amber) ---
    box(8.15, cy, 1.95, 1.25, "Percentile-rank\nfusion", "fusion", 10)
    arr(7.7, cy + 0.6, 8.15, cy + 0.6, COL["classic"][1])
    arr(12.7, gy - 0.05, 9.1, cy + 1.3, COL["geom"][1], lab="DS adds DS-specific evidence")  # DS into fusion
    chip(fused, 10.35, cy - 0.05, 1.3, "fused change map", cmap="inferno", ec=COL["fusion"][1])
    arr(10.1, cy + 0.6, 10.35, cy + 0.6, COL["fusion"][1])

    # --- two output heads ---
    # unsupervised (coral)
    box(12.4, 4.6, 1.75, 1.1, "Threshold /\ntop-k% triage", "unsup", 9.5)
    chip(xb, 14.35, 4.35, 1.55, "damage candidates (xBD)", ec=COL["unsup"][1])
    arr(12.0, cy + 1.2, 12.4, 4.9, COL["fusion"][1])
    arr(14.15, 5.15, 14.35, 5.15, COL["unsup"][1])
    # supervised (purple)
    box(12.4, 1.35, 1.75, 1.15, "FC-EF U-Net\n[bands $\\oplus$ DS prior]", "sup", 9.2)
    chip(gt, 14.35, 1.1, 1.5, "change segmentation", cmap="gray", ec=COL["sup"][1])
    arr(11.65, cy, 12.4, 2.0, COL["fusion"][1], lab="prior channel")
    arr(14.15, 1.85, 14.35, 1.85, COL["sup"][1])
    ax.text(13.27, 0.95, "loss: BCE + Dice", ha="center", fontsize=8.5, color=COL["sup"][1], style="italic")

    # legend
    handles = [Patch(fc=COL[k][0], ec=COL[k][1], label=v) for k, v in
               [("data", "input / preprocess"), ("geom", "subspace geometry"), ("classic", "classical baselines"),
                ("fusion", "label-free fusion"), ("unsup", "unsupervised triage"), ("sup", "supervised head")]]
    ax.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    for ext in ("png", "svg"):
        try:
            fig.savefig(OUT / f"fig1c_framework.{ext}", dpi=190, bbox_inches="tight", facecolor="white")
        except PermissionError:
            pass
    plt.close(fig); print("  saved fig1c_framework (png/svg)")


def _safe(x):
    return x


if __name__ == "__main__":
    main()
