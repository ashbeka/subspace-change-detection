"""Synthetic multispectral image time series with controlled, fully-labeled change events.

Used by L1 (docs/DESIGN_TEMPORAL_DS_ACCV2026.md s11.3): inject a known change at known time and
magnitude into an otherwise-stable scene that ALSO contains realistic radiometric confounds
(global seasonal gain, per-band haze offset, gradual spectral seasonal drift, sensor noise).
Ground truth (event time, region, magnitude) is known by construction => zero circularity.

The confounds are the point: a good change descriptor must spike on the structural event and stay
quiet during the radiometric swings that fool raw differencing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SynthConfig:
    H: int = 64
    W: int = 64
    B: int = 10
    T: int = 60
    n_materials: int = 4
    event_t0: int = 30           # date of abrupt event onset
    event_box: tuple = (20, 44, 20, 44)  # r0, r1, c0, c1 of the changed region
    event_magnitude: float = 1.0  # 1.0 = full conversion to the "burn" endmember
    recovery_len: int = 0        # 0 = permanent step; >0 = linear recovery back toward original over N dates
    season_gain_amp: float = 0.15   # global multiplicative seasonal swing
    season_drift_amp: float = 0.08   # gradual per-band spectral seasonal drift (a structural-ish confound)
    haze_amp: float = 0.03       # per-date additive offset
    noise_sigma: float = 0.02
    seed: int = 0


@dataclass
class SynthScene:
    cube: np.ndarray             # (T, B, H, W) reflectance
    cfg: SynthConfig
    region_mask: np.ndarray      # (H, W) bool, the changed region
    changed_dates: np.ndarray    # (T,) bool, dates where region differs from its pre-event state
    meta: dict = field(default_factory=dict)


def _endmembers(rng, B, n):
    """n distinct, smooth-ish spectral signatures in [0.05, 0.9]."""
    base = np.linspace(0, 1, B)
    sigs = []
    for _ in range(n):
        center = rng.uniform(0, 1)
        width = rng.uniform(0.15, 0.5)
        amp = rng.uniform(0.3, 0.8)
        bias = rng.uniform(0.05, 0.3)
        s = bias + amp * np.exp(-0.5 * ((base - center) / width) ** 2)
        sigs.append(np.clip(s, 0.02, 0.95))
    return np.array(sigs)  # (n, B)


def make_scene(cfg: SynthConfig) -> SynthScene:
    rng = np.random.default_rng(cfg.seed)
    H, W, B, T = cfg.H, cfg.W, cfg.B, cfg.T
    endm = _endmembers(rng, B, cfg.n_materials + 1)  # +1 = "burn" endmember
    burn = endm[-1]
    mats = endm[:-1]

    # smooth-ish spatial label map: blocky quadrant layout + a little noise
    label = np.zeros((H, W), dtype=int)
    label[: H // 2, : W // 2] = 0
    label[: H // 2, W // 2 :] = 1
    label[H // 2 :, : W // 2] = 2 % cfg.n_materials
    label[H // 2 :, W // 2 :] = 3 % cfg.n_materials
    base = mats[label]                       # (H, W, B)
    base = np.moveaxis(base, 2, 0)           # (B, H, W)

    r0, r1, c0, c1 = cfg.event_box
    region_mask = np.zeros((H, W), dtype=bool)
    region_mask[r0:r1, c0:c1] = True

    cube = np.zeros((T, B, H, W), dtype=np.float64)
    changed_dates = np.zeros(T, dtype=bool)

    # gradual spectral seasonal drift direction (per-band), applied to the WHOLE scene smoothly
    drift_dir = rng.standard_normal(B)
    drift_dir = drift_dir / np.linalg.norm(drift_dir)

    for t in range(T):
        struct = base.copy()
        # --- event: convert region to a blend toward the burn endmember ---
        if t >= cfg.event_t0:
            if cfg.recovery_len > 0:
                # linear recovery: full burn at t0, back to original by t0+recovery_len
                frac = max(0.0, 1.0 - (t - cfg.event_t0) / cfg.recovery_len)
            else:
                frac = 1.0
            alpha = cfg.event_magnitude * frac
            if alpha > 1e-6:
                changed_dates[t] = True
                region_spec = (1 - alpha) * struct[:, region_mask] + alpha * burn[:, None]
                struct[:, region_mask] = region_spec
        # --- gradual seasonal spectral drift (structural-ish confound, smooth in time) ---
        season_phase = np.sin(2 * np.pi * t / max(2, T // 1.5))
        struct = struct + cfg.season_drift_amp * season_phase * drift_dir[:, None, None]
        # --- global multiplicative seasonal gain (radiometric confound) ---
        gain = 1.0 + cfg.season_gain_amp * np.sin(2 * np.pi * t / max(2, T // 1.2) + 0.7)
        # --- per-date additive haze offset (radiometric confound) ---
        haze = cfg.haze_amp * np.sin(2 * np.pi * t / max(2, T // 0.9) + 1.3)
        img = gain * struct + haze
        img = img + rng.normal(0, cfg.noise_sigma, img.shape)
        cube[t] = np.clip(img, 0.0, None)

    return SynthScene(
        cube=cube, cfg=cfg, region_mask=region_mask, changed_dates=changed_dates,
        meta={"endmembers": endm, "label": label},
    )
