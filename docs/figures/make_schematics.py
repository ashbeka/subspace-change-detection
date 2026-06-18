"""Generate committed method schematics for docs/METHOD.md.

Run: cd <worktree> && <main-venv>/python.exe docs/figures/make_schematics.py
Produces: fig_mssa_construction.png, fig_velocity_acceleration.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))


def _box(ax, x, y, w, h, text, fc="#eef3fb", ec="#3b5b92", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=5)


def _arrow(ax, p1, p2, color="#333", lw=1.6, style="-|>", mut=12):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=mut, lw=lw, color=color))


# ---------------------------------------------------------------- Fig 1: M-SSA construction
def mssa_construction():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.05]})
    # Panel A: multi-band time series + window + event
    ax = axes[0]; t = np.arange(96); t0 = 48
    for b, (off, P) in enumerate([(0, 16), (1.6, 16), (3.2, 16)]):
        x = np.sin(2 * np.pi * t / P) + 0.7 * np.sin(4 * np.pi * t / P)
        x[t0:] = np.sin(2 * np.pi * t[t0:] / P) * 1.18  # 2nd harmonic drops (event)
        ax.plot(t, x + off * 1.0, lw=1.4, color=f"C{b}")
        ax.text(-2, off, f"band {b+1}", ha="right", va="center", fontsize=8, color=f"C{b}")
    ax.axvspan(24, 48, color="#cfe0f5", alpha=.6); ax.axvspan(48, 72, color="#f5d9cf", alpha=.6)
    ax.text(36, 4.0, "past W", ha="center", fontsize=8); ax.text(60, 4.0, "present W", ha="center", fontsize=8)
    ax.axvline(t0, color="k", ls="--", lw=1); ax.text(t0 + 1, -1.6, "event", fontsize=8)
    ax.set_title("A. per-region multi-band index series", fontsize=10)
    ax.set_xlabel("date"); ax.set_yticks([]); ax.set_xlim(-6, 96)

    # Panel B: Hankel stack
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("B. stack per-band Hankel matrices", fontsize=10)
    cols = ["#cfe0f5", "#bcd3ef", "#a9c6ea"]
    for b in range(3):
        y0 = 7.0 - b * 2.4
        for j in range(4):
            ax.add_patch(plt.Rectangle((1.6 + j * 0.9, y0), 0.85, 1.9, fc=cols[b], ec="#3b5b92", lw=.8))
        ax.text(1.1, y0 + 0.95, f"H{b+1}", ha="right", va="center", fontsize=9, color="C%d" % b)
    ax.text(5.0, 9.4, "L delays x (W-L+1)", ha="center", fontsize=8)
    _arrow(ax, (5.0, 0.4), (5.0, 1.1), color="#3b5b92")
    ax.text(5.0, 0.1, "H = [H1; H2; H3]  in  R^{BL x (W-L+1)}", ha="center", fontsize=8.5)

    # Panel C: SVD -> subspace point on Grassmann
    ax = axes[2]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("C. joint signal subspace S(t)", fontsize=10)
    _box(ax, 1.0, 7.2, 8.0, 1.6, "S(t) = top-k left singular vectors of H\n(energy-based rank)", fs=9)
    # mini grassmann with a trajectory
    th = np.linspace(0.15, 2.6, 100)
    gx, gy = 5 + 3.0 * np.cos(th), 3.4 + 1.7 * np.sin(th)
    ax.plot(gx, gy, color="#999", lw=1.2)
    pts = [10, 45, 70, 92]
    for i, p in enumerate(pts):
        ax.plot(gx[p], gy[p], "o", ms=8, color="C3" if i == 2 else "#3b5b92")
        ax.text(gx[p], gy[p] - 0.5, f"S({['t-2','t-1','t','t+1'][i]})", ha="center", fontsize=7.5)
    ax.text(5, 0.7, "trajectory on Grassmann manifold Gr(k, BL)", ha="center", fontsize=8.5, style="italic")
    fig.suptitle("M-SSA construction: a temporal window of dates is the genuine 'set' -> a multi-dimensional subspace",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, "fig_mssa_construction.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: velocity/acceleration on geodesic
def velocity_acceleration():
    fig, ax = plt.subplots(figsize=(8.8, 5.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    # manifold curve (the actual trajectory)
    th = np.linspace(0.2, 2.4, 200)
    cx, cy = 5 + 3.4 * np.cos(th), 2.2 + 2.6 * np.sin(th)
    ax.plot(cx, cy, color="#bbb", lw=2, zorder=1)
    ax.text(cx[5] + 0.1, cy[5] - 0.4, "subspace trajectory", color="#888", fontsize=9)
    # three points S(t), S(t+tau), S(t+2tau)
    i1, i2, i3 = 20, 100, 185
    P1, P2, P3 = (cx[i1], cy[i1]), (cx[i2], cy[i2]), (cx[i3], cy[i3])
    for P, lab, c in [(P1, "S(t)", "#3b5b92"), (P2, "S(t+τ)", "C3"), (P3, "S(t+2τ)", "#3b5b92")]:
        ax.plot(*P, "o", ms=11, color=c, zorder=5)
        ax.text(P[0], P[1] + 0.35, lab, ha="center", fontsize=10, zorder=6)
    # geodesic chord S(t)->S(t+2tau) and Karcher midpoint M
    ax.plot([P1[0], P3[0]], [P1[1], P3[1]], ls="--", color="#2a8f5a", lw=1.8, zorder=2)
    M = ((P1[0] + P3[0]) / 2, (P1[1] + P3[1]) / 2)
    ax.plot(*M, "s", ms=9, color="#2a8f5a", zorder=5)
    ax.text(M[0] - 0.1, M[1] - 0.5, "M = Karcher mean\n(constant-velocity midpoint)",
            ha="center", fontsize=8.5, color="#2a8f5a")
    # velocity arrow (first difference)
    _arrow(ax, P1, P2, color="C0", lw=2.4, mut=16)
    ax.text((P1[0] + P2[0]) / 2 - 0.2, (P1[1] + P2[1]) / 2 + 0.45,
            "velocity  d1 = ||DS(S(t),S(t+τ))||", color="C0", fontsize=9.5, rotation=0)
    # acceleration arrow (M -> S(t+tau)) = deviation from midpoint
    _arrow(ax, M, P2, color="C1", lw=2.4, mut=16)
    ax.text(P2[0] + 0.15, (M[1] + P2[1]) / 2, "acceleration\nd2 = ||DS(S(t+τ), M)||",
            color="C1", fontsize=9.5, va="center")
    # along/orth annotation
    ax.text(0.2, 7.4, "first-order DS  = velocity of the subspace (change onset)", fontsize=9.5, color="C0")
    ax.text(0.2, 6.9, "second-order DS = acceleration; deviation of S(t+τ) from the geodesic midpoint M",
            fontsize=9.5, color="C1")
    ax.text(0.2, 6.4, "geodesic split: along-path (trend) vs off-path 'orth' (abrupt regime break)",
            fontsize=9.5, color="#2a8f5a")
    ax.set_title("First / second-order Difference Subspace = velocity / acceleration on the Grassmann manifold",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_velocity_acceleration.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    mssa_construction()
    velocity_acceleration()
    print("wrote fig_mssa_construction.png, fig_velocity_acceleration.png to", HERE)
