"""L1 experiment: controlled synthetic injection (the zero-circularity rigor anchor).

Tests the core hypotheses on fully-labeled synthetic data with realistic radiometric confounds
(global seasonal gain, haze offset, gradual spectral drift, sensor noise):
  H1 sensitivity:  a temporal-DS velocity spikes at the injected onset t0.
  H2 invariance:   DS is less responsive than raw differencing to radiometric confounds
                   (higher event/confound contrast).
  H3 localization: per-tile DS map at t0 lights up the event region.
  H4 scaling:      detection grows with injected magnitude.
  H5 recovery:     distance-to-baseline rises then falls; off-geodesic 2nd-order marks abrupt onset.

Because the per-date vs window construction and the centering choice are unsettled (the design's open
question), this script SWEEPS construction variants and lets the metrics decide, against two mandatory
nulls: trivial raw-reflectance difference and SSA minimum-angle.

Run: ./.venv/Scripts/python.exe -m temporal.experiments.synth_injection
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth import SynthConfig, make_scene

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "L1_synth")
os.makedirs(OUT, exist_ok=True)


def tile_grid(H, W, tile):
    return [(r, c) for r in range(0, H - tile + 1, tile) for c in range(0, W - tile + 1, tile)]


def event_overlap(r, c, tile, box):
    r0, r1, c0, c1 = box
    return (r < r1 and r + tile > r0 and c < c1 and c + tile > c0)


# ----------------------------------------------------------------- subspace series builders
def build_series(cube, tile, mode, rank, T_w, center):
    """Return {(r,c): (centers, [basis])}.

    mode "spatial":  one subspace per DATE from the tile's (bands x pixels) matrix.
    mode "temporal": sliding-window subspace from the tile's (bands*pixels x T_w) date-stack,
                     attributed to the window center date.
    `center` toggles mean-centering (autocorrelation-PCA / lab-faithful when False).
    """
    T, B, H, W = cube.shape
    out = {}
    for (r, c) in tile_grid(H, W, tile):
        blk = cube[:, :, r:r + tile, c:c + tile]  # (T, B, tile, tile)
        if mode == "spatial":
            centers = np.arange(T)
            bases = [ss.pca_subspace(blk[t].reshape(B, -1), rank, center=center, energy=0.99) for t in range(T)]
        elif mode == "temporal":
            feats = blk.reshape(T, -1).T            # (B*tile*tile, T)
            centers, bases = [], []
            for start in range(0, T - T_w + 1):
                X = feats[:, start:start + T_w]      # (ambient, T_w)
                bases.append(ss.pca_subspace(X, rank, center=center, energy=0.99))
                centers.append(start + T_w // 2)
            centers = np.array(centers)
        else:
            raise ValueError(mode)
        out[(r, c)] = (centers, bases)
    return out


# ----------------------------------------------------------------- curves
def velocity(centers, bases, lag=1):
    d = np.array([ss.magnitude(bases[i], bases[i + lag]) for i in range(len(bases) - lag)])
    return centers[:len(d)], d


def minangle(centers, bases, lag=1):
    d = np.array([1.0 - np.max(ss.canonical_cosines(bases[i], bases[i + lag])) for i in range(len(bases) - lag)])
    return centers[:len(d)], d


def dist_to_ref(centers, bases, ref_idx=0):
    ref = bases[ref_idx]
    return centers, np.array([ss.magnitude(b, ref) for b in bases])


def trivial_velocity(cube, r, c, tile, lag=1):
    T = cube.shape[0]
    blk = cube[:, :, r:r + tile, c:c + tile].reshape(T, -1)
    d = np.array([np.mean(np.abs(blk[t + lag] - blk[t])) for t in range(T - lag)])
    return np.arange(T - lag) + lag // 2, d  # roughly center-aligned


def trivial_distance(cube, r, c, tile, ref_t=0):
    T = cube.shape[0]
    blk = cube[:, :, r:r + tile, c:c + tile].reshape(T, -1)
    ref = blk[ref_t]
    return np.arange(T), np.array([np.mean(np.abs(blk[t] - ref)) for t in range(T)])


# ----------------------------------------------------------------- metrics
def loc_err(centers, curve, t0):
    return int(abs(centers[int(np.argmax(curve))] - t0))


def contrast(centers, curve, t0):
    win = curve[(centers >= t0 - 1) & (centers <= t0 + 2)]
    peak = float(np.max(win)) if win.size else 0.0
    pre = curve[centers < t0 - 2]
    conf = float(np.percentile(pre, 95)) if pre.size else 1e-9
    return peak / (conf + 1e-9)


def evaluate(cube, scene, cfg, tile, mode, rank, T_w, center, lag=1):
    t0 = cfg.event_t0
    series = build_series(cube, tile, mode, rank, T_w, center)
    etile = next((rc for rc in series if event_overlap(*rc, tile, cfg.event_box)), (0, 0))
    cen, d1 = velocity(*series[etile], lag)
    _, ma = minangle(*series[etile], lag)
    tcen, tv = trivial_velocity(cube, *etile, tile, lag)

    # state-detection AUC across all tiles x dates (positive = event tile AND changed date)
    labels, sc_ds, sc_tv = [], [], []
    for (r, c), (centers, bases) in series.items():
        is_evt = event_overlap(r, c, tile, cfg.event_box)
        dc, dpre = dist_to_ref(centers, bases)
        _, tdist = trivial_distance(cube, r, c, tile)
        for k, t in enumerate(dc):
            labels.append(1 if (is_evt and scene.changed_dates[t]) else 0)
            sc_ds.append(dpre[k]); sc_tv.append(tdist[t])
    labels = np.array(labels)
    auc_ds = auc_tv = None
    if 0 < labels.sum() < len(labels):
        auc_ds = float(roc_auc_score(labels, sc_ds)); auc_tv = float(roc_auc_score(labels, sc_tv))

    # state localization: per-tile d_pre averaged over the post-event (changed) dates
    vals, regs = [], []
    for (r, c), (centers, bases) in series.items():
        _, dpre = dist_to_ref(centers, bases)
        post = dpre[centers >= t0]
        vals.append(float(np.mean(post)) if post.size else 0.0)
        regs.append(1 if event_overlap(r, c, tile, cfg.event_box) else 0)
    vals = np.array(vals); regs = np.array(regs)
    thr = np.percentile(vals, 100 * (1 - regs.mean()))
    pred = (vals >= thr).astype(int)
    union = np.sum((pred == 1) | (regs == 1))
    iou = float(np.sum((pred == 1) & (regs == 1)) / union) if union else 0.0

    return {
        "loc_err_ds": loc_err(cen, d1, t0), "loc_err_trivial": loc_err(tcen, tv, t0),
        "loc_err_minangle": loc_err(cen, ma, t0),
        "contrast_ds": round(contrast(cen, d1, t0), 3),
        "contrast_trivial": round(contrast(tcen, tv, t0), 3),
        "contrast_minangle": round(contrast(cen, ma, t0), 3),
        "state_auc_ds": round(auc_ds, 3) if auc_ds else None,
        "state_auc_trivial": round(auc_tv, 3) if auc_tv else None,
        "loc_iou_ds": round(iou, 3),
    }, {"cen": cen, "d1": d1, "ma": ma, "tcen": tcen, "tv": tv, "etile": etile, "series": series}


# ----------------------------------------------------------------- main
def gain_invariance_check(rank=4, T_w=8):
    """Direct check of H2 mechanism: a pure global-gain change must give DS~0 but raw-distance>0."""
    rng = np.random.default_rng(7)
    base = rng.uniform(0.1, 0.8, (10, 64))            # one tile: bands x pixels, one structure
    w0 = np.stack([base for _ in range(T_w)], axis=2).reshape(10 * 64, T_w)
    gains = np.linspace(1.0, 1.6, T_w)                 # pure multiplicative gain ramp, no structural change
    w1 = np.stack([g * base for g in gains], axis=2).reshape(10 * 64, T_w)
    S0 = ss.pca_subspace(w0 + rng.normal(0, 1e-3, w0.shape), rank, center=False, energy=0.99)
    S1 = ss.pca_subspace(w1 + rng.normal(0, 1e-3, w1.shape), rank, center=False, energy=0.99)
    ds = ss.magnitude(S0, S1)
    raw = float(np.mean(np.abs(w1 - w0)))
    return {"ds_under_pure_gain": round(ds, 5), "raw_dist_under_pure_gain": round(raw, 5)}


def confound_sweep(mode, center, rank, T_w, tile=8):
    """Vary radiometric-confound strength; show DS state-AUC stays robust while trivial degrades."""
    out = {"gain_amp": [], "ds_auc": [], "trivial_auc": []}
    for amp in [0.0, 0.05, 0.1, 0.2, 0.3, 0.45]:
        c = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0, event_magnitude=0.6,
                        season_gain_amp=amp, haze_amp=amp * 0.3, season_drift_amp=0.05)
        s = make_scene(c)
        m, _ = evaluate(s.cube, s, c, tile, mode, rank, T_w, center)
        out["gain_amp"].append(amp); out["ds_auc"].append(m["state_auc_ds"]); out["trivial_auc"].append(m["state_auc_trivial"])
    return out


def main():
    inv = gain_invariance_check()
    print("[gain-invariance]", inv, "(DS should be ~0, raw>0)")
    cfg = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0, event_magnitude=1.0)
    scene = make_scene(cfg); cube = scene.cube

    variants = [
        ("spatial",  True,  3, 6),
        ("spatial",  False, 3, 6),
        ("temporal", False, 4, 6),
        ("temporal", True,  4, 6),
        ("temporal", False, 4, 8),
    ]
    print(f"{'mode':9} {'cen':5} {'rk':2} {'Tw':2} | {'locErr(ds/triv/ma)':20} {'contrast(ds/triv/ma)':24} {'AUC(ds/triv)':14} {'IoU':5}")
    table = []
    best = None
    for mode, center, rank, T_w in variants:
        m, cur = evaluate(cube, scene, cfg, 8, mode, rank, T_w, center)
        m.update({"mode": mode, "center": center, "rank": rank, "T_w": T_w})
        table.append(m)
        print(f"{mode:9} {str(center):5} {rank:2d} {T_w:2d} | "
              f"{m['loc_err_ds']:>3}/{m['loc_err_trivial']:>3}/{m['loc_err_minangle']:>3}{'':9} "
              f"{m['contrast_ds']:>6}/{m['contrast_trivial']:>6}/{m['contrast_minangle']:>6}{'':4} "
              f"{str(m['state_auc_ds']):>6}/{str(m['state_auc_trivial']):>6} {m['loc_iou_ds']:>5}")
        score = (m["state_auc_ds"] or 0) + (1 if m["loc_err_ds"] <= 3 else 0)
        if best is None or score > best[0]:
            best = (score, mode, center, rank, T_w, cur)

    # magnitude sweep on the best DS variant
    _, mode, center, rank, T_w, _ = best
    sweep = {"magnitude": [], "ds_contrast": [], "trivial_contrast": [], "ds_auc": [], "trivial_auc": []}
    for mag in [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]:
        c2 = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0, event_magnitude=mag)
        s2 = make_scene(c2)
        m2, _ = evaluate(s2.cube, s2, c2, 8, mode, rank, T_w, center)
        sweep["magnitude"].append(mag)
        sweep["ds_contrast"].append(m2["contrast_ds"]); sweep["trivial_contrast"].append(m2["contrast_trivial"])
        sweep["ds_auc"].append(m2["state_auc_ds"]); sweep["trivial_auc"].append(m2["state_auc_trivial"])

    # recovery scenario on best variant
    cfgr = SynthConfig(event_box=(16, 32, 16, 32), event_t0=20, recovery_len=20, event_magnitude=1.0)
    sr = make_scene(cfgr)
    seriesr = build_series(sr.cube, 8, mode, rank, T_w, center)
    etile = next(rc for rc in seriesr if event_overlap(*rc, 8, cfgr.event_box))
    cen_r, bases_r = seriesr[etile]
    _, dpre = dist_to_ref(cen_r, bases_r)
    orth = np.array([ss.second_order_decomposed(bases_r[i], bases_r[i + 1], bases_r[i + 2])[1]
                     for i in range(len(bases_r) - 2)])

    # figures (best variant)
    _, _, _, _, _, cur = best
    _fig_curves(cur, cfg.event_t0, mode, center, os.path.join(OUT, "fig1_event_curves.png"))
    _fig_sweep(sweep, os.path.join(OUT, "fig2_magnitude_sweep.png"))
    _fig_localization(cur, scene, 8, cfg, os.path.join(OUT, "fig3_localization.png"))
    _fig_recovery(cen_r, dpre, orth, cfgr, os.path.join(OUT, "fig4_recovery.png"))

    # confound-robustness sweep on the best variant (the headline H2 demonstration)
    csweep = confound_sweep(mode, center, rank, T_w)
    print("confound sweep:", json.dumps(csweep))
    _fig_confound(csweep, os.path.join(OUT, "fig5_confound_robustness.png"))

    summary = {"variants": table, "best_variant": {"mode": mode, "center": center, "rank": rank, "T_w": T_w},
               "gain_invariance": inv, "magnitude_sweep": sweep, "confound_sweep": csweep}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest DS variant: mode={mode} center={center} rank={rank} T_w={T_w}")
    print("magnitude sweep:", json.dumps(sweep))
    print(f"Saved metrics + figures to {OUT}")


def _norm(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (np.ptp(x) + 1e-12)


def _fig_curves(cur, t0, mode, center, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(cur["cen"], _norm(cur["d1"]), label="DS velocity (event tile)", lw=2.3, color="C3")
    ax.plot(cur["tcen"], _norm(cur["tv"]), label="trivial raw-diff", lw=1.6, color="C0", ls="--")
    ax.plot(cur["cen"], _norm(cur["ma"]), label="SSA min-angle", lw=1.3, color="C2", ls=":")
    ax.axvline(t0, color="k", lw=1, alpha=0.6, label=f"true event t0={t0}")
    ax.set_xlabel("date index"); ax.set_ylabel("normalized response")
    ax.set_title(f"L1 event-onset detection (best DS: {mode}, center={center})")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_sweep(sweep, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); m = sweep["magnitude"]
    ax[0].plot(m, sweep["ds_contrast"], "o-", color="C3", label="DS")
    ax[0].plot(m, sweep["trivial_contrast"], "s--", color="C0", label="trivial")
    ax[0].set_title("H4 contrast vs magnitude"); ax[0].set_xlabel("magnitude"); ax[0].legend(); ax[0].grid(alpha=.3)
    if sweep["ds_auc"][0] is not None:
        ax[1].plot(m, sweep["ds_auc"], "o-", color="C3", label="DS")
        ax[1].plot(m, sweep["trivial_auc"], "s--", color="C0", label="trivial")
    ax[1].set_title("state AUC vs magnitude"); ax[1].set_xlabel("magnitude"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_localization(cur, scene, tile, cfg, path):
    H = W = scene.cube.shape[2]; grid = np.zeros((H // tile, W // tile))
    for (r, c), (centers, bases) in cur["series"].items():
        cc, v = velocity(centers, bases); j = int(np.argmin(np.abs(cc - cfg.event_t0)))
        grid[r // tile, c // tile] = v[j]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.3))
    im = ax[0].imshow(grid, cmap="inferno"); ax[0].set_title("DS velocity map at onset"); fig.colorbar(im, ax=ax[0], fraction=.046)
    ax[1].imshow(scene.region_mask, cmap="gray"); ax[1].set_title("true event region")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_confound(cs, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(cs["gain_amp"], cs["ds_auc"], "o-", color="C3", lw=2.2, label="DS distance-to-baseline")
    ax.plot(cs["gain_amp"], cs["trivial_auc"], "s--", color="C0", lw=1.8, label="trivial raw distance")
    ax.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.5, label="chance")
    ax.set_xlabel("radiometric confound strength (seasonal gain amplitude)")
    ax.set_ylabel("changed-state detection AUC")
    ax.set_title("H2: DS state detection is robust to confounds; raw distance degrades")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_recovery(cen, dpre, orth, cfg, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(cen, _norm(dpre), label="d_pre (distance to baseline)", lw=2.3, color="C3")
    ax.plot(cen[:len(orth)], _norm(orth), label="2nd-order off-geodesic (orth)", lw=1.6, color="C1")
    ax.axvline(cfg.event_t0, color="k", lw=1, alpha=.6, label=f"onset t0={cfg.event_t0}")
    ax.axvline(cfg.event_t0 + cfg.recovery_len, color="g", lw=1, ls="--", alpha=.6, label="recovery end")
    ax.set_xlabel("date index"); ax.set_ylabel("normalized")
    ax.set_title("H5 degradation->recovery (d_pre rises then falls)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
