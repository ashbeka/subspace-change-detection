"""ADVERSARIAL PROBE: construction robustness & artifact hunt for the temporal-DS claim.

Angle: is the DS confound-robustness win (state-AUC 1.0->0.86 vs trivial 1.0->0.49) a
real property of the descriptor, or an artifact of:
  (A) the unequal-dim magnitude generalization (max(k1,k2)) -> DS just detects "rank changed"?
  (B) the energy=0.99 threshold / rank-cap choice -> knife-edge hyperparameter?
  (C) the d_pre reference = first window -> leakage / lucky baseline?
  (D) tile size / window / lag interacting with the synthetic confound periods?

We reuse the EXACT builders/metrics from synth_injection.py (no reimplementation, no rigging),
and only swap the things we want to stress. We also add FAIR baselines the original omitted:
  - trivial distance with a robust (median-over-pre-window) reference instead of single date 0,
  - gain-normalized trivial distance (divide each date by its scene-mean) -- the obvious fix any
    practitioner applies, which should neutralize global gain for free.
  - SAM (spectral-angle) raw distance -- inherently gain-invariant, the real null DS must beat.

Outputs under temporal/outputs/_probe_construction/.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth import SynthConfig, make_scene
from temporal.experiments import synth_injection as si

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_probe_construction")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------- helpers
def build_series_eqrank(cube, tile, mode, rank, T_w, center, fixed_k):
    """Like si.build_series but FORCES every window to exactly fixed_k components
    (disable energy-driven variable rank and the max-dim padding advantage)."""
    T, B, H, W = cube.shape
    out = {}
    for (r, c) in si.tile_grid(H, W, tile):
        blk = cube[:, :, r:r + tile, c:c + tile]
        if mode == "spatial":
            centers = np.arange(T)
            bases = [_pca_fixed(blk[t].reshape(B, -1), fixed_k, center) for t in range(T)]
        elif mode == "temporal":
            feats = blk.reshape(T, -1).T
            centers, bases = [], []
            for start in range(0, T - T_w + 1):
                X = feats[:, start:start + T_w]
                bases.append(_pca_fixed(X, fixed_k, center))
                centers.append(start + T_w // 2)
            centers = np.array(centers)
        out[(r, c)] = (centers, bases)
    return out


def _pca_fixed(data, k, center):
    """PCA basis with EXACTLY k columns (energy=None so rank is fixed, never variable)."""
    return ss.pca_subspace(data, k, center=center, energy=None)


def state_auc_from_series(cube, scene, series, t0, ref_idx=0, ref_mode="firstwin"):
    """Replicate si.evaluate's state-AUC for DS and trivial, but with a pluggable DS reference
    and pluggable trivial baseline. Returns (auc_ds, auc_triv_raw, auc_triv_norm, auc_sam)."""
    labels, sc_ds, sc_traw, sc_tnorm, sc_sam = [], [], [], [], []
    T = cube.shape[0]
    # scene-mean per date for gain normalization (global radiometric estimate)
    scene_mean = cube.reshape(T, -1).mean(axis=1)  # (T,)
    for (r, c), (centers, bases) in series.items():
        is_evt = si.event_overlap(r, c, TILE, scene.cfg.event_box)
        # DS distance to reference
        if ref_mode == "firstwin":
            ref = bases[ref_idx]
        elif ref_mode == "premedian":
            # robust pre-event reference: median-angle representative among pre-t0 windows
            pre = [i for i, t in enumerate(centers) if t < t0 - 2]
            ref = bases[pre[len(pre) // 2]] if pre else bases[ref_idx]
        dpre = np.array([ss.magnitude(b, ref) for b in bases])
        blk = cube[:, :, r:r + TILE, c:c + TILE].reshape(T, -1)
        rawref = blk[0]
        tdist_raw = np.array([np.mean(np.abs(blk[t] - rawref)) for t in range(T)])
        # gain-normalized trivial: divide each date by its scene-mean (kills global multiplicative gain)
        bnorm = blk / scene_mean[:, None]
        tdist_norm = np.array([np.mean(np.abs(bnorm[t] - bnorm[0])) for t in range(T)])
        # SAM: spectral angle between mean spectrum at t and at ref (gain-invariant by construction)
        B = cube.shape[1]
        spec = blk.reshape(T, B, -1).mean(axis=2)  # (T, B) mean spectrum
        sref = spec[0]
        def sam(a, b):
            na = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            return float(np.arccos(np.clip(np.dot(a, b) / na, -1, 1)))
        tdist_sam = np.array([sam(spec[t], sref) for t in range(T)])
        for k, t in enumerate(centers):
            labels.append(1 if (is_evt and scene.changed_dates[t]) else 0)
            sc_ds.append(dpre[k]); sc_traw.append(tdist_raw[t])
            sc_tnorm.append(tdist_norm[t]); sc_sam.append(tdist_sam[t])
    labels = np.array(labels)
    if not (0 < labels.sum() < len(labels)):
        return None, None, None, None
    return (float(roc_auc_score(labels, sc_ds)),
            float(roc_auc_score(labels, sc_traw)),
            float(roc_auc_score(labels, sc_tnorm)),
            float(roc_auc_score(labels, sc_sam)))


TILE = 8  # module global so the helper above can read tile w/o threading it through


def tile_of(series):
    return TILE


def confound_sweep_custom(builder, **bkw):
    """Sweep gain amplitude; builder(cube)->series. Reports all 4 detectors."""
    rows = []
    for amp in [0.0, 0.05, 0.1, 0.2, 0.3, 0.45]:
        c = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0,
                        event_magnitude=0.6, season_gain_amp=amp, haze_amp=amp * 0.3,
                        season_drift_amp=0.05)
        s = make_scene(c)
        series = builder(s.cube)
        ds, traw, tnorm, sam = state_auc_from_series(s.cube, s, series, c.event_t0, **bkw)
        rows.append({"amp": amp, "ds": rnd(ds), "triv_raw": rnd(traw),
                     "triv_gainnorm": rnd(tnorm), "sam": rnd(sam)})
    return rows


def rnd(x):
    return round(x, 3) if x is not None else None


# ----------------------------------------------------------------------- probes
def probe_A_equal_rank():
    """Force every window to EXACTLY fixed_k. If DS-AUC survives, the win is NOT just
    'rank changed'. If it collapses to chance, the max-dim padding was doing the work."""
    res = {}
    for mode, center, T_w in [("spatial", False, 6), ("temporal", False, 6)]:
        for fixed_k in [1, 2, 3, 4]:
            def b(cube, m=mode, ct=center, tw=T_w, k=fixed_k):
                return build_series_eqrank(cube, TILE, m, 4, tw, ct, k)
            res[f"{mode}_c{center}_Tw{T_w}_k{fixed_k}"] = confound_sweep_custom(b)
    return res


def probe_B_energy_rankcap():
    """Sweep energy threshold and rank cap with the ORIGINAL variable-rank builder.
    Knife-edge if DS-AUC swings wildly across these."""
    res = {}
    for energy in [0.9, 0.95, 0.99, 0.999, 0.9999]:
        for rank in [2, 4, 6, 10]:
            def b(cube, e=energy, rk=rank):
                return build_series_energy(cube, TILE, "spatial", rk, 6, False, e)
            res[f"E{energy}_rk{rank}"] = confound_sweep_custom(b)
    return res


def build_series_energy(cube, tile, mode, rank, T_w, center, energy):
    """si.build_series but with explicit energy (original hardcodes 0.99)."""
    T, B, H, W = cube.shape
    out = {}
    for (r, c) in si.tile_grid(H, W, tile):
        blk = cube[:, :, r:r + tile, c:c + tile]
        if mode == "spatial":
            centers = np.arange(T)
            bases = [ss.pca_subspace(blk[t].reshape(B, -1), rank, center=center, energy=energy)
                     for t in range(T)]
        else:
            feats = blk.reshape(T, -1).T
            centers, bases = [], []
            for start in range(0, T - T_w + 1):
                X = feats[:, start:start + T_w]
                bases.append(ss.pca_subspace(X, rank, center=center, energy=energy))
                centers.append(start + T_w // 2)
            centers = np.array(centers)
        out[(r, c)] = (centers, bases)
    return out


def probe_C_reference_leakage():
    """Compare first-window ref vs robust pre-median ref vs a LATE (post-event) ref.
    Also report rank of the chosen reference to see if ref is rank-1 (then everything looks far)."""
    res = {}
    for ref_mode in ["firstwin", "premedian"]:
        def b(cube):
            return build_series_energy(cube, TILE, "spatial", 3, 6, False, 0.99)
        res[ref_mode] = confound_sweep_custom(b, ref_mode=ref_mode)
    return res


def probe_D_tile_window_lag():
    """Sweep tile size and window. Tile size changes ambient dim (B*tile*tile) and pixel count."""
    res = {}
    for tile in [4, 8, 16]:
        for T_w in [4, 6, 8, 10]:
            global TILE
            TILE = tile
            def b(cube, t=tile, tw=T_w):
                return build_series_energy(cube, t, "temporal", 4, tw, False, 0.99)
            res[f"tile{tile}_Tw{T_w}"] = confound_sweep_custom(b)
    TILE = 8
    return res


def probe_rank_of_windows():
    """Diagnostic: what rank does energy=0.99 actually select for event vs non-event windows,
    pre vs post event? If post-event event-tile rank jumps, the max-dim term explains the AUC."""
    c = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0,
                    event_magnitude=0.6, season_gain_amp=0.45, haze_amp=0.135, season_drift_amp=0.05)
    s = make_scene(c)
    cube = s.cube
    diag = {}
    for mode, T_w in [("spatial", 6), ("temporal", 6)]:
        series = build_series_energy(cube, 8, mode, 4, T_w, False, 0.99)
        evt = next(rc for rc in series if si.event_overlap(*rc, 8, c.event_box))
        non = next(rc for rc in series if not si.event_overlap(*rc, 8, c.event_box))
        for name, rc in [("event_tile", evt), ("nonevent_tile", non)]:
            centers, bases = series[rc]
            ranks = [b.shape[1] for b in bases]
            pre = [r for ce, r in zip(centers, ranks) if ce < c.event_t0]
            post = [r for ce, r in zip(centers, ranks) if ce >= c.event_t0]
            diag[f"{mode}_{name}"] = {"pre_rank": pre, "post_rank": post,
                                      "pre_mean": round(float(np.mean(pre)), 2),
                                      "post_mean": round(float(np.mean(post)), 2)}
    return diag


def probe_L0_invariants():
    """Confirm the L0 algebra invariants still hold under stress."""
    rng = np.random.default_rng(3)
    res = {}
    # identical -> 0
    B = ss.pca_subspace(rng.standard_normal((20, 8)), 3, center=False)
    res["identical_zero"] = round(ss.magnitude(B, B), 9)
    # fully orthogonal k-dim -> 2k
    Q, _ = np.linalg.qr(rng.standard_normal((20, 6)))
    A, C = Q[:, :3], Q[:, 3:6]
    res["orthogonal_2k"] = round(ss.magnitude(A, C), 6)  # expect 6.0
    # pure gain invariance (the headline mechanism), with center=False
    base = rng.uniform(0.1, 0.8, (10, 64))
    S0 = ss.pca_subspace(base, 4, center=False, energy=0.99)
    S1 = ss.pca_subspace(2.7 * base, 4, center=False, energy=0.99)
    res["pure_gain_ds"] = round(ss.magnitude(S0, S1), 9)
    # additive haze (constant offset added to every pixel) -- is DS invariant to that too?
    S2 = ss.pca_subspace(base + 0.2, 4, center=False, energy=0.99)
    res["pure_haze_ds_centerFalse"] = round(ss.magnitude(S0, S2), 6)
    S0c = ss.pca_subspace(base, 4, center=True, energy=0.99)
    S2c = ss.pca_subspace(base + 0.2, 4, center=True, energy=0.99)
    res["pure_haze_ds_centerTrue"] = round(ss.magnitude(S0c, S2c), 6)
    # unequal-dim sanity: 3-dim subspace vs same 3-dim PLUS one extra orthogonal dim -> 2*1
    extra = np.concatenate([A, C[:, :1]], axis=1)  # 4 cols, contains A
    res["unequaldim_extra_ortho"] = round(ss.magnitude(A, extra), 6)  # expect 2.0
    return res


def main():
    out = {}
    out["L0_invariants"] = probe_L0_invariants()
    print("[L0]", json.dumps(out["L0_invariants"]))
    out["rank_diag"] = probe_rank_of_windows()
    print("[rank-diag]", json.dumps(out["rank_diag"], indent=1))
    out["A_equal_rank"] = probe_A_equal_rank()
    out["B_energy_rankcap"] = probe_B_energy_rankcap()
    out["C_reference"] = probe_C_reference_leakage()
    out["D_tile_window"] = probe_D_tile_window_lag()
    with open(os.path.join(OUT, "probe_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- digest printing
    print("\n=== PROBE A: equal-rank (disable max-dim padding) -- amp 0.0 and 0.45 ===")
    for k, rows in out["A_equal_rank"].items():
        print(f"  {k:28} ds@0={rows[0]['ds']} ds@.45={rows[-1]['ds']} "
              f"sam@.45={rows[-1]['sam']} gainnorm@.45={rows[-1]['triv_gainnorm']}")
    print("\n=== PROBE B: energy/rank cap (variable-rank original) -- ds@0.0 / ds@0.45 ===")
    for k, rows in out["B_energy_rankcap"].items():
        print(f"  {k:14} ds: {rows[0]['ds']} -> {rows[-1]['ds']}")
    print("\n=== PROBE C: reference choice -- ds@0.0 / ds@0.45 ===")
    for k, rows in out["C_reference"].items():
        print(f"  {k:12} ds: {rows[0]['ds']} -> {rows[-1]['ds']}")
    print("\n=== PROBE D: tile/window -- ds@0.0 / ds@0.45 ===")
    for k, rows in out["D_tile_window"].items():
        print(f"  {k:16} ds: {rows[0]['ds']} -> {rows[-1]['ds']}")
    print("\n=== FAIR BASELINES at amp=0.45 (spatial,k=3,E.99 original) ===")
    fair = out["C_reference"]["firstwin"]
    print(f"  ds={fair[-1]['ds']} triv_raw={fair[-1]['triv_raw']} "
          f"triv_GAINNORM={fair[-1]['triv_gainnorm']} SAM={fair[-1]['sam']}")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
