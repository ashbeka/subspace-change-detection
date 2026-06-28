"""Synthetic 1-D index time series with a DYNAMICS change (not a mean/level change).

The L1 adversarial verdict showed first-order DS on spectra == spectral angle (SAM). To justify the
subspace machinery at all, it must detect something SAM / mean-differencing / level-normalisation CANNOT:
a change in TEMPORAL DYNAMICS where the mean (and often the variance) is preserved. This is the SSA
setting of Kanai et al. 2023 (DS between signal subspaces), the construction Sensei pointed to.

Each region is a scalar index series (think NDVI of a pixel/parcel). Event regions undergo a dynamics
change at t0; non-event regions keep their pre-dynamics. Ground truth (event/no-event, t0) is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("amp_collapse", "freq_shift", "noise_replace", "trend_onset", "recovery")


@dataclass
class DynConfig:
    T: int = 80
    t0: int = 40
    period: float = 12.0
    amp: float = 1.0
    baseline: float = 0.5
    noise: float = 0.15
    n_regions: int = 120
    event_frac: float = 0.5
    mode: str = "freq_shift"
    seed: int = 0


def _seasonal(t, period, amp, phase):
    return amp * (np.sin(2 * np.pi * t / period + phase) + 0.4 * np.sin(4 * np.pi * t / period + phase))


def _make_one(cfg: DynConfig, is_event: bool, rng):
    t = np.arange(cfg.T)
    phase = rng.uniform(0, 2 * np.pi)
    pre = cfg.baseline + _seasonal(t, cfg.period, cfg.amp, phase)
    x = pre.copy()
    if is_event:
        tt = t[cfg.t0:]
        ph2 = phase
        if cfg.mode == "amp_collapse":            # mean preserved, variance drops
            x[cfg.t0:] = cfg.baseline + 0.1 * _seasonal(tt, cfg.period, cfg.amp, ph2)
        elif cfg.mode == "freq_shift":            # mean & variance ~preserved, structure changes
            x[cfg.t0:] = cfg.baseline + _seasonal(tt, cfg.period / 2.2, cfg.amp, ph2)
        elif cfg.mode == "noise_replace":         # mean & variance ~preserved, deterministic->stochastic
            sd = np.std(_seasonal(t, cfg.period, cfg.amp, phase))
            x[cfg.t0:] = cfg.baseline + rng.normal(0, sd, tt.shape)
        elif cfg.mode == "trend_onset":           # gradual degradation: linear trend appears
            x[cfg.t0:] = pre[cfg.t0:] - 0.04 * (tt - cfg.t0)
        elif cfg.mode == "recovery":              # degrade then recover (for 2nd-order)
            half = (cfg.T - cfg.t0) // 2
            ramp = np.concatenate([np.linspace(0, -1.0, half), np.linspace(-1.0, 0, len(tt) - half)])
            x[cfg.t0:] = pre[cfg.t0:] + ramp
        else:
            raise ValueError(cfg.mode)
    x = x + rng.normal(0, cfg.noise, x.shape)
    return x


def make_panel(cfg: DynConfig):
    """Returns X (n_regions, T) and labels (n_regions,) bool event."""
    rng = np.random.default_rng(cfg.seed)
    n_evt = int(cfg.n_regions * cfg.event_frac)
    labels = np.array([True] * n_evt + [False] * (cfg.n_regions - n_evt))
    rng.shuffle(labels)
    X = np.stack([_make_one(cfg, bool(lb), rng) for lb in labels])
    return X, labels
