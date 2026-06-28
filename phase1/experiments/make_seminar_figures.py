"""Generate professional-grade seminar figures (PNG + PDF, LaTeX-style math via mathtext)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "axes.titlesize": 13, "axes.titleweight": "bold",
})
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase1" / "outputs" / "seminar_figures"
OUT.mkdir(parents=True, exist_ok=True)
C = {"naive": "#9aa0a6", "classical": "#4c78a8", "geom": "#e45756", "ours": "#54a24b", "cnn": "#b279a2"}


def save(fig, name):
    ok = []
    for ext in ("png", "svg", "pdf"):  # png+svg first (svg = editable); pdf optional
        try:
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight"); ok.append(ext)
        except PermissionError:
            print(f"  [skip {name}.{ext}: file locked/open]")
    plt.close(fig)
    print(f"  saved {name}: {'/'.join(ok)}")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2)
    def box(x, y, w, h, text, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=col, ec="black", alpha=0.85, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=1.6, color="#333"))
    box(0.2, 3.4, 2.3, 1.2, "Pre image\n$X_{pre}\\in\\mathbb{R}^{B\\times H\\times W}$", "#cfe2f3")
    box(0.2, 0.6, 2.3, 1.2, "Post image\n$X_{post}\\in\\mathbb{R}^{B\\times H\\times W}$", "#cfe2f3")
    box(3.0, 2.0, 2.6, 1.2, "Band-image samples\n$X_t\\in\\mathbb{R}^{N\\times B}$\n(each band = 1 vector)", "#fff2cc")
    box(6.1, 3.4, 2.5, 1.2, "Spatial PCA\n$\\Phi=U_{pre},\\ \\Psi=U_{post}$\nrank $r{=}12$", "#d9ead3")
    box(6.1, 0.6, 2.5, 1.2, "Difference Subspace\n$D=(\\Phi U-\\Psi V)[2(I-\\Sigma)]^{-1/2}$", "#f4cccc")
    box(9.1, 2.0, 2.0, 1.2, "Score map\n$s_i=\\sum_b\\|P_D\\delta_b\\|^2$", "#f9cb9c")
    box(11.3, 3.4, 1.5, 1.2, "Fusion /\nU-Net prior", "#d5a6bd")
    box(11.3, 0.6, 1.5, 1.2, "Candidate\ndamage map", "#d5a6bd")
    arrow(2.5, 3.8, 3.0, 2.9); arrow(2.5, 1.2, 3.0, 2.3)
    arrow(5.6, 2.8, 6.1, 3.7); arrow(5.6, 2.4, 6.1, 1.2)
    arrow(8.6, 3.7, 9.6, 3.2); arrow(8.6, 1.2, 9.6, 2.3)
    arrow(11.1, 2.9, 11.3, 3.7); arrow(11.1, 2.3, 11.3, 1.2)
    ax.text(6.5, 4.9, "Spatially-faithful Difference-Subspace pipeline", ha="center", fontsize=14, fontweight="bold")
    save(fig, "fig1_pipeline_schematic")


def bar(ax, names, vals, cols, title, ylab, fmt="{:.3f}", rot=25):
    b = ax.bar(names, vals, color=cols, edgecolor="black", lw=0.7)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(title); ax.set_ylabel(ylab); ax.tick_params(axis="x", rotation=rot)


def fig_ladder():
    fig, ax = plt.subplots(figsize=(11, 5))
    names = ["Raw CVA\n(naive)", "PCA-diff", "smoothed\nPCA", "IR-MAD", "Band-Image\nDS", "rank\nfusion", "U-Net\n(bands)", "U-Net\n+DS-fusion"]
    vals = [0.226, 0.254, 0.268, 0.214, 0.241, 0.278, 0.483, 0.513]
    cols = [C["naive"], C["classical"], C["classical"], C["classical"], C["geom"], C["ours"], C["cnn"], C["ours"]]
    bar(ax, names, vals, cols, "How methods compare on OSCD urban change (higher = better)", "accuracy (Average Precision)")
    ax.axhline(0.226, ls="--", c="gray", alpha=0.6); ax.text(0.1, 0.232, "naive floor", color="gray", fontsize=8)
    save(fig, "fig2_ladder")


def fig_oscd_fusion():
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ["bands\n(M0)", "+DS\n(M1)", "+cross\n(M2)", "+sPCA\n(M3)", "+sPCA\n+IR-MAD\n(M5)", "+cross+sPCA\n+IR-MAD (M6)", "+DS+sPCA\n+IR-MAD (M4)"]
    vals = [0.4825, 0.4788, 0.4808, 0.4780, 0.4804, 0.4870, 0.5128]
    cols = [C["cnn"], C["geom"], C["classical"], C["classical"], C["classical"], C["classical"], C["ours"]]
    bar(ax, names, vals, cols, "Adding the subspace map to a CNN helps only in combination\n(M4 with DS beats M5 without DS and M6 with a look-alike control)", "accuracy (test AP)", rot=0)
    ax.set_ylim(0.45, 0.53)
    save(fig, "fig3_oscd_fusion_dsspecific")


def fig_xbd():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = ["Raw L2\n(naive)", "PCA-diff", "Band-Image\nDS", "IR-MAD", "Projector\n(ours)"]
    auroc = [0.443, 0.591, 0.626, 0.730, 0.734]
    cols = [C["naive"], C["classical"], C["geom"], C["classical"], C["ours"]]
    bar(axes[0], names, auroc, cols, "Finding disaster damage on xBD (5 unseen disasters) — higher is better", "ranking quality (AUROC)", rot=20)
    axes[0].set_ylim(0.4, 0.78)
    pct = [1, 5, 10]
    axes[1].plot(pct, [0.078, 0.247, 0.373], "o-", color=C["ours"], lw=2.2, label="Projector (ours)")
    axes[1].plot(pct, [0.049, 0.178, 0.321], "s--", color=C["classical"], lw=2, label="IR-MAD")
    axes[1].plot(pct, [0.01, 0.05, 0.10], ":", color="gray", label="random")
    for p, r in zip(pct, [0.078, 0.247, 0.373]):
        axes[1].annotate(f"{r:.0%}", (p, r), textcoords="offset points", xytext=(4, 6), fontsize=9)
    axes[1].set_title("Operational triage: damaged-pixel recall vs review budget")
    axes[1].set_xlabel("review budget (top % of ranked pixels)"); axes[1].set_ylabel("recall of damaged pixels")
    axes[1].legend(); axes[1].set_xticks(pct)
    save(fig, "fig4_xbd_headline")


def fig_cca():
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["CVA\n(naive)", "IR-MAD\n(iter. CCA)", "SAM", "sparse-CCA\n(S3CCA)", "plain\nCCA", "DS"]
    vals = [0.978, 0.958, 0.919, 0.906, 0.707, 0.508]
    cols = [C["naive"], C["classical"], C["naive"], C["classical"], C["classical"], C["geom"]]
    bar(ax, names, vals, cols, "The CCA family we were asked to test (Benton HSI) — higher is better", "accuracy (AUC)", rot=15)
    ax.set_ylim(0.45, 1.0)
    save(fig, "fig5_cca_family")


def fig_otsu():
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = ["DS", "smoothed-PCA", "fusion"]
    free = [0.337, 0.358, 0.369]; otsu = [0.142, 0.146, 0.147]
    x = np.arange(len(groups)); w = 0.34
    ax.bar(x - w / 2, free, w, label="threshold-free best-F1", color=C["classical"], edgecolor="black")
    ax.bar(x + w / 2, otsu, w, label="Otsu hard-threshold F1", color=C["naive"], edgecolor="black")
    ax.axhline(0.513, ls="--", color=C["ours"], lw=2); ax.text(1.4, 0.52, "U-Net (raw, NO threshold) F1=0.51", color=C["ours"], fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("F1"); ax.legend()
    ax.set_title("Hard Otsu thresholding hurts; training with no threshold wins")
    save(fig, "fig6_otsu_ablation")


def fig_changetype():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    names = ["polar-CVA\ndirection", "raw change\nvector", "DS canonical\ndirections", "PCA-of-diff"]
    vals = [0.832, 0.787, 0.467, 0.422]
    cols = [C["classical"], C["naive"], C["geom"], C["classical"]]
    bar(ax, names, vals, cols, "Telling 6 change types apart (Benton): the simple direction wins\n(higher = better match to ground-truth types)", "agreement with truth (NMI)", rot=15)
    save(fig, "fig7_changetype_interpretability")


def fig_diagnostic():
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off")
    ax.text(0.5, 0.93, "The diagnostic: REPRESENTATION $\\times$ OPERATOR $\\times$ DECISION",
            ha="center", transform=ax.transAxes, fontsize=14, fontweight="bold")
    rows = [
        ["change family", "representation", "comparison operator", "what beats DS"],
        ["amplitude (fire, flood)", "raw spectrum", "magnitude / CVA", "CVA / raw L2"],
        ["low-dim spectral", "raw spectrum", "spectral angle (SAM)", "SAM (DS$\\equiv$SAM)"],
        ["distributed / full-rank", "covariance", "correlation / IR-MAD", "covariance, IR-MAD"],
        ["seasonal vs abrupt", "harmonic model", "deseasonalized residual", "BFAST / CCDC / DRMAT"],
        ["change TYPE", "change direction", "direction clustering", "polar-CVA"],
        ["where DS helps", "spatial subspace", "fusion + localization", "DS-specific (OSCD, xBD)"],
    ]
    nr, nc = len(rows), 4
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            fc = "#34495e" if i == 0 else ("#d6eadf" if i == nr - 1 else "white")
            tc = "white" if i == 0 else "black"
            ax.add_patch(plt.Rectangle((j / nc, (nr - 1 - i) / nr * 0.8), 1 / nc, 0.8 / nr,
                         transform=ax.transAxes, fc=fc, ec="#888", lw=0.7))
            ax.text((j + 0.5) / nc, ((nr - 1 - i) + 0.5) / nr * 0.8, cell, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.6, color=tc, fontweight="bold" if i in (0, nr - 1) else "normal")
    save(fig, "fig8_diagnostic_matrix")


def fig_math():
    fig, ax = plt.subplots(figsize=(10.5, 6)); ax.axis("off")
    # (label, equation) — labels are plain text, equations are mathtext-safe
    rows = [
        ("1. Samples (band-image, spatial-faithful):", r"$X_t=[\,\mathrm{vec}(b_1)\,|\,\cdots\,|\,\mathrm{vec}(b_B)\,]\in\mathbb{R}^{N\times B},\ \ N=H\times W$"),
        ("    each band-image = ONE sample; the date = ONE subspace", None),
        ("2. Per-date subspaces (rank r):", r"$\Phi=U_{pre}^{1:r},\ \ \Psi=U_{post}^{1:r}\in\mathbb{R}^{N\times r}$"),
        ("3. Canonical angles:", r"$\Phi^\top\Psi=U\Sigma V^\top,\ \ \Sigma=\mathrm{diag}(\cos\theta_i)$"),
        ("4. Difference Subspace:", r"$D=(\Phi U-\Psi V)\,[\,2(I-\Sigma)\,]^{-1/2}$"),
        ("    magnitude:", r"$\sum_i 2\,(1-\cos\theta_i)$"),
        ("5. Change score (pixel i):", r"$s_i=\sum_{b=1}^{B}\,\|P_D\,\delta_b\|^2,\ \ \delta_b=X_{post,b}-X_{pre,b},\ \ P_D=DD^\top$"),
        ("6. Projector distance (xBD):", r"$d_i=\|(P_\Phi-P_\Psi)\|_{\mathrm{row}\,i},\ \ P_\Phi=\Phi\Phi^\top$  (subspace rotation)"),
    ]
    ax.text(0.5, 0.98, "Spatially-faithful Difference Subspace — the math",
            ha="center", transform=ax.transAxes, fontsize=14, fontweight="bold")
    y = 0.88
    for lab, eq in rows:
        ax.text(0.03, y, lab, transform=ax.transAxes, fontsize=11.5, va="top", fontweight="bold")
        if eq is not None:
            ax.text(0.07, y - 0.052, eq, transform=ax.transAxes, fontsize=13, va="top", color="#1a3c6e")
            y -= 0.115
        else:
            y -= 0.055
    save(fig, "fig9_math_reference")


if __name__ == "__main__":
    print("generating seminar figures ...")
    fig_pipeline(); fig_ladder(); fig_oscd_fusion(); fig_xbd(); fig_cca(); fig_otsu(); fig_changetype(); fig_diagnostic(); fig_math()
    print(f"\nAll figures -> {OUT}")
