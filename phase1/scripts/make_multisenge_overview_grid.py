from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import rasterio


PHASE1_ROOT = Path(__file__).resolve().parent
MULTISENGE_ROOT = PHASE1_ROOT / "data" / "raw" / "MultiSenGE" / "s2"
MULTISENGE_VIZ_ROOT = PHASE1_ROOT / "outputs" / "multisenge_viz"


def _load_rgb(path: Path):
    """Load MultiSenGE S2 patch as RGB (B04,B03,B02) with simple contrast stretch."""
    with rasterio.open(path) as src:
        b2 = src.read(1).astype(np.float32)  # B02
        b3 = src.read(2).astype(np.float32)  # B03
        b4 = src.read(3).astype(np.float32)  # B04
    rgb = np.stack([b4, b3, b2], axis=-1)

    valid = np.isfinite(rgb)
    if not np.any(valid):
        return np.zeros_like(rgb, dtype=np.float32)

    lo, hi = np.percentile(rgb[valid], (2, 98))
    if hi <= lo:
        hi = lo + 1.0
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return rgb


def make_multisenge_overview_grid(n_patches: int, output_path: Path):
    """Create a grid showing pre/post RGB and DS projection for a few MultiSenGE patches."""
    log_path = MULTISENGE_VIZ_ROOT / "multisenge_viz_log.json"
    with open(log_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    entries = entries[:n_patches]
    n_rows = len(entries)
    n_cols = 3  # pre, post, DS proj

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.0),
        squeeze=False,
    )

    for i, e in enumerate(entries):
        patch_id = e["patch_id"]
        t1_rel = e["t1"]
        t2_rel = e["t2"]

        t1_path = PHASE1_ROOT / t1_rel
        t2_path = PHASE1_ROOT / t2_rel
        ds_proj_path = MULTISENGE_VIZ_ROOT / f"{patch_id}_proj.png"

        pre_rgb = _load_rgb(t1_path)
        post_rgb = _load_rgb(t2_path)
        ds_proj_img = plt.imread(ds_proj_path)

        # Pre
        ax = axes[i, 0]
        ax.imshow(pre_rgb)
        ax.set_axis_off()
        if i == 0:
            ax.set_title("Pre RGB", fontsize=10)
        ax.set_ylabel(patch_id, fontsize=8)

        # Post
        ax = axes[i, 1]
        ax.imshow(post_rgb)
        ax.set_axis_off()
        if i == 0:
            ax.set_title("Post RGB", fontsize=10)

        # DS projection
        ax = axes[i, 2]
        ax.imshow(ds_proj_img)
        ax.set_axis_off()
        if i == 0:
            ax.set_title("DS projection", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    out = PHASE1_ROOT / "docs" / "figs" / "multisenge_overview_6x3.png"
    make_multisenge_overview_grid(6, out)
    print(f"Saved MultiSenGE overview grid to {out}")

