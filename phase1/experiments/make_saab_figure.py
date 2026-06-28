"""Saab-DS seminar figures: the 3-step construction journey + the detailed pipeline + results bar (PNG + editable SVG)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

OUT = Path(__file__).resolve().parents[2] / "phase1" / "outputs" / "seminar_figures"
OUT.mkdir(parents=True, exist_ok=True)
INK = "#16202e"
C = {"data": ("#e9edf2", "#5a6473"), "naive": ("#eceff2", "#8a93a0"),
     "band": ("#d6e6f7", "#2f5fa8"), "saab": ("#d3ece2", "#1d8f6e"),
     "ds": ("#fce6cf", "#b8731b"), "out": ("#f6d3c9", "#c0392b")}


def _save(fig, name):
    for ext in ("png", "svg"):
        try:
            fig.savefig(OUT / f"{name}.{ext}", dpi=190, bbox_inches="tight", facecolor="white")
        except PermissionError:
            print(f"  [skip {name}.{ext} locked]")
    plt.close(fig); print(f"  saved {name}")


def box(ax, x, y, w, h, text, kind, fs=10):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x1, y1, x2, y2, c="#5a6473", lab=None, lw=1.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=lw, color=c))
    if lab:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, lab, ha="center", fontsize=8.4, color=c, style="italic")


def fig_journey():
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(7.5, 5.05, "The construction journey: how you define the 'subspace' decides everything",
            ha="center", fontsize=13.5, color=INK, fontweight="bold")
    panels = [
        (0.4, "Step 1 — Global pixel DS", "naive", ["sample = one 13-band pixel", "subspace in $\\mathbb{R}^{13}$ (colour)",
         "position is LOST"], "AP 0.06", "#8a93a0"),
        (5.3, "Step 2 — Band-image DS", "band", ["sample = one band-image", "subspace in $\\mathbb{R}^{N}$ (spatial)",
         "position is KEPT"], "AP 0.24", "#2f5fa8"),
        (10.2, "Step 3 — Successive Saab-DS", "saab", ["+ Green-Learning Saab features", "local 3$\\times$3 patterns, 2 hops",
         "+ Difference Subspace"], "AP 0.34", "#1d8f6e"),
    ]
    for x, title, kind, lines, ap, col in panels:
        box(ax, x, 1.3, 4.4, 3.0, "", kind)
        ax.text(x + 2.2, 3.95, title, ha="center", fontsize=11.5, color=col, fontweight="bold")
        for i, ln in enumerate(lines):
            ax.text(x + 2.2, 3.35 - i * 0.55, ln, ha="center", fontsize=10, color=INK)
        ax.text(x + 2.2, 1.62, ap, ha="center", fontsize=15, color=col, fontweight="bold")
    arrow(ax, 4.8, 2.8, 5.3, 2.8, "#2f5fa8", "keep spatial layout")
    arrow(ax, 9.7, 2.8, 10.2, 2.8, "#1d8f6e", "richer local features")
    ax.text(7.5, 0.6, "Frozen held-out OSCD (10 unseen cities) — Successive Saab-DS beats every classical baseline; "
            "DS also beats L2/PCA on the same features.", ha="center", fontsize=9.5, color="#5a6473")
    _save(fig, "fig_saab_journey")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16.5, 7.2)); ax.set_xlim(0, 16.5); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(8.25, 6.85, "Successive Saab-DS pipeline (label-free, training-free feature extraction $\\rightarrow$ Difference Subspace)",
            ha="center", fontsize=13, color=INK, fontweight="bold")
    # Row 1 — feature extraction (Saab hops)
    y1 = 4.7
    box(ax, 0.3, y1, 1.7, 1.2, "Pre / post\nimages\n13 bands", "data", 9.5)
    box(ax, 2.4, y1, 1.9, 1.2, "3$\\times$3\nneighborhood\n$z\\in\\mathbb{R}^{9K}$", "saab", 9.5)
    box(ax, 4.7, y1, 2.2, 1.2, "Saab transform\nDC (mean) + AC (PCA)\n$\\leq$16 channels", "saab", 9.5)
    box(ax, 7.3, y1, 1.5, 1.2, "2$\\times$2\nmax-pool", "saab", 9.5)
    box(ax, 9.2, y1, 2.1, 1.2, "Hop 2\n(repeat Saab)\ncoarser context", "saab", 9.5)
    for x1, x2 in [(2.0, 2.4), (4.3, 4.7), (6.9, 7.3), (8.8, 9.2)]:
        arrow(ax, x1, y1 + 0.6, x2, y1 + 0.6, C["saab"][1])
    ax.text(8.25, y1 + 1.45, "feature extraction — same transform for both dates, no labels (Green Learning / PixelHop)",
            ha="center", fontsize=9, color=C["saab"][1], style="italic")
    # down to row 2
    arrow(ax, 10.25, y1, 10.25, 3.55, C["saab"][1])
    # Row 2 — DS scoring per hop + fusion
    y2 = 2.2
    box(ax, 0.3, y2, 2.2, 1.2, "Per-hop band-image\nsubspaces\n$\\Phi_h,\\Psi_h$ (rank 12)", "band", 9.3)
    box(ax, 2.9, y2, 2.4, 1.2, "Canonical angles\n$\\Phi_h^\\top\\Psi_h=U\\Sigma V^\\top$", "band", 9.3)
    box(ax, 5.7, y2, 2.2, 1.2, "Difference subspace\n$D_h$", "ds", 9.5)
    box(ax, 8.3, y2, 2.2, 1.2, "Hop score\n$s_h=\\|D_hD_h^\\top\\Delta_h\\|$", "ds", 9.3)
    box(ax, 10.9, y2, 1.7, 1.2, "Average\nthe hops", "ds", 9.5)
    box(ax, 13.0, y2, 2.0, 1.2, "Change map\n(AP 0.34)", "out", 9.8)
    arrow(ax, 10.25, 3.55, 1.4, y2 + 1.2, C["band"][1])  # hop features -> subspaces
    for x1, x2 in [(2.5, 2.9), (5.3, 5.7), (7.9, 8.3), (10.5, 10.9), (12.6, 13.0)]:
        arrow(ax, x1, y2 + 0.6, x2, y2 + 0.6, C["ds"][1] if x1 > 5 else C["band"][1])
    # matched controls branch
    box(ax, 8.3, 0.45, 4.3, 1.0, "Matched controls on the SAME features: L2 (0.31) · PCA (0.31) · cross-recon (0.33)  <  DS (0.34)",
        "data", 9)
    arrow(ax, 9.4, y2, 9.4, 1.45, "#8a93a0", "DS beats them all")
    # legend
    handles = [Patch(fc=C[k][0], ec=C[k][1], label=v) for k, v in
               [("data", "data / control"), ("saab", "Saab features (Green Learning)"),
                ("band", "spatial subspace"), ("ds", "Difference Subspace"), ("out", "change map")]]
    ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    _save(fig, "fig_saab_pipeline")


def fig_results():
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ["raw CVA", "IR-MAD", "PCA-diff", "smoothed\nPCA", "global\npixel DS", "band-image\nDS",
             "wavelet\nLL+DS", "Successive\nSaab-DS"]
    ap = [0.279, 0.265, 0.307, 0.314, 0.060, 0.295, 0.288, 0.342]
    cols = ["#8a93a0", "#4c78a8", "#4c78a8", "#4c78a8", "#cfcfcf", "#2f5fa8", "#9c6", "#1d8f6e"]
    b = ax.bar(names, ap, color=cols, edgecolor="black", lw=0.7)
    for r, v in zip(b, ap):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.6)
    ax.set_ylabel("accuracy (Average Precision)")
    ax.set_title("Frozen held-out OSCD (10 unseen cities): Successive Saab-DS wins")
    ax.tick_params(axis="x", rotation=15); ax.set_ylim(0, 0.39)
    ax.axhline(0.314, ls="--", c="#999", lw=1); ax.text(0.1, 0.322, "best classical (smoothed PCA)", fontsize=8, color="#777")
    _save(fig, "fig_saab_results")


if __name__ == "__main__":
    print("building Saab figures ...")
    fig_journey(); fig_pipeline(); fig_results()
    print(f"-> {OUT}")
