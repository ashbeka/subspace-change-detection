"""
GAP A: time-sequential first/second Difference-Subspace magnitudes + geodesic
decomposition on a REAL multi-date Sentinel-2 sequence.

This is Sensei's most-repeated ask (Slack, ~4x): treat each multi-band satellite
image as a subspace, and compute how the FIRST DS (change "velocity") and SECOND
DS (change "acceleration"/abruptness, with along- vs orthogonal-geodesic split)
evolve across a time sequence. Framed as a first/unique ANALYSIS trial (not a
SOTA detector), exactly as Sensei stated for ACCV.

Construction = the spatially-faithful BAND-IMAGE subspace (each band-image is a
sample; ambient space = spatial positions), so this simultaneously answers
Sensei's spatial-information concern. Paper-faithful code:
phase1/subspace/temporal_band_images.py (first/second DS, geodesic).

Data: tmp/ipol416_sequences/<area> with dated GeoTIFFs (RGBI). al_wakrah (Qatar,
11 real S2 dates 2016-2018), beijing_airport, piraeus, + controlled synthetic_*.

Run:
  .\.venv\Scripts\python.exe -m phase1.experiments.temporal_ds_sequence \
      --sequences al_wakrah,beijing_airport,piraeus,synthetic_a
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.subspace.temporal_band_images import (
    build_band_image_subspace, sequence_common_valid_mask, temporal_difference_measurements,
)

SEQ_ROOT = ROOT / "tmp" / "ipol416_sequences"
OUT = ROOT / "phase1" / "outputs" / "temporal_ds_sequence"
DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def load_sequence(seq_dir: Path):
    tifs = sorted(seq_dir.glob("*.tif"))
    cubes, dates = [], []
    for fp in tifs:
        m = DATE_RE.search(fp.name)
        if not m:
            continue
        with rasterio.open(fp) as src:
            cube = src.read().astype(np.float32)  # (bands,H,W)
        cubes.append(cube)
        dates.append("".join(m.groups()))  # YYYYMMDD
    return cubes, dates


def mean_spectrum_l2(cubes, mask):
    """Trivial null: L2 distance between consecutive dates' mean spectra over valid pixels."""
    means = [c[:, mask].mean(axis=1) for c in cubes]
    return [float(np.linalg.norm(means[i + 1] - means[i])) for i in range(len(means) - 1)]


def process(seq_dir: Path, rank: int) -> dict:
    cubes, dates = load_sequence(seq_dir)
    if len(cubes) < 3:
        return {}
    # align shapes (use the smallest common HxW)
    h = min(c.shape[1] for c in cubes); w = min(c.shape[2] for c in cubes)
    cubes = [c[:, :h, :w] for c in cubes]
    mask = sequence_common_valid_mask(cubes, nodata_value=0.0, require_all_bands=True)
    if int(mask.sum()) < 50:
        mask = np.ones((h, w), bool)
    bases = [build_band_image_subspace(c, mask, rank=rank, preprocessing="centered_band_l2").basis
             for c in cubes]
    meas = temporal_difference_measurements(bases, dates)
    meas["mean_spectrum_l2"] = mean_spectrum_l2(cubes, mask)
    meas["dates"] = dates
    meas["n_valid"] = int(mask.sum())
    meas["rank"] = int(bases[0].shape[1])
    return meas


def plot_sequence(name: str, meas: dict, outdir: Path):
    dates = meas["dates"]
    adj_mid = [f"{dates[i][4:6]}/{dates[i][2:4]}->{dates[i+1][4:6]}/{dates[i+1][2:4]}"
               for i in range(len(dates) - 1)]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    # first DS magnitude (velocity) vs trivial mean-spectrum L2
    f1 = np.array(meas["first_adjacent_magnitude"])
    triv = np.array(meas["mean_spectrum_l2"])
    ax[0].plot(range(len(f1)), f1 / (f1.max() + 1e-9), "o-", label="first-DS magnitude (normalized)")
    ax[0].plot(range(len(triv)), triv / (triv.max() + 1e-9), "s--", alpha=0.7,
               label="trivial mean-spectrum L2 (normalized)")
    ax[0].set_xticks(range(len(adj_mid))); ax[0].set_xticklabels(adj_mid, rotation=60, fontsize=7)
    ax[0].set_title(f"{name}: first-DS change 'velocity' across the sequence"); ax[0].legend(fontsize=8)
    # second DS magnitude (acceleration) with along/orthogonal geodesic split
    s_total = np.array(meas["second_magnitude"])
    s_along = np.array(meas["second_along_geodesic"])
    s_orth = np.array(meas["second_orthogonal_geodesic"])
    xs = range(1, 1 + len(s_total))
    ax[1].plot(xs, s_total, "o-", label="2nd-DS total (abruptness)")
    ax[1].plot(xs, s_along, "^--", alpha=0.7, label="along-geodesic (smooth drift)")
    ax[1].plot(xs, s_orth, "v--", alpha=0.7, label="orthogonal (off-geodesic = abrupt)")
    mid = [f"{dates[i][4:6]}/{dates[i][2:4]}" for i in range(1, len(dates) - 1)]
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(mid, rotation=60, fontsize=7)
    ax[1].set_title(f"{name}: second-DS 'acceleration' + geodesic decomposition"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fp = outdir / f"seq_{name}.png"
    fig.savefig(fp, dpi=130, bbox_inches="tight"); plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="al_wakrah,beijing_airport,piraeus,synthetic_a")
    ap.add_argument("--rank", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = [s.strip() for s in args.sequences.split(",") if s.strip()]
    allres = {}
    for name in names:
        seq_dir = SEQ_ROOT / name
        if not seq_dir.exists():
            print(f"  [skip] {name} (no dir)"); continue
        meas = process(seq_dir, args.rank)
        if not meas:
            print(f"  [skip] {name} (<3 dates)"); continue
        fp = plot_sequence(name, meas, OUT)
        # correlation of DS 'velocity' with the trivial null -> is DS just the mean shift?
        f1 = np.array(meas["first_adjacent_magnitude"]); triv = np.array(meas["mean_spectrum_l2"])
        corr = float(np.corrcoef(f1, triv)[0, 1]) if f1.size > 1 and triv.std() > 0 else float("nan")
        allres[name] = {k: v for k, v in meas.items()}
        allres[name]["first_ds_vs_trivial_corr"] = corr
        print(f"==> {name}: {len(meas['dates'])} dates, rank {meas['rank']}, n_valid {meas['n_valid']}; "
              f"first-DS vs trivial-mean-L2 corr = {corr:.3f}; peak 2nd-DS at idx "
              f"{int(np.argmax(meas['second_magnitude']))+1}; saved {fp.name}", flush=True)
    with (OUT / "measurements.json").open("w") as f:
        json.dump(allres, f, indent=2, default=float)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
