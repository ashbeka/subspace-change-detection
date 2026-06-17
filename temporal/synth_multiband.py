"""Multi-band index time series with MULTI-MODE structural dynamics changes.

The decisive question after L1: is there a regime where the FULL Difference Subspace (all canonical
angles) beats the single minimum-angle (conventional SSA) AND all simple scalars? Theory (Kanai et al.
2023) says yes when the change is distributed across MULTIPLE temporal modes and there is noise -- the
full DS aggregates evidence across angles, the min-angle uses only one.

So we construct changes that alter the temporal-mode subspace across several modes while keeping the
per-band mean and variance ~preserved (so mean/variance detectors are blind):
  harmonic_dropout : a harmonic disappears (its sin+cos modes vanish; >=2 canonical angles move)
  freq_split       : some bands shift to a new frequency (new modes appear)
  recovery         : multi-harmonic content degrades then recovers (for 2nd-order/geodesic)

This is satellite-relevant: phenological cycles are multi-harmonic; deforestation/abandonment/disturbance
change the harmonic STRUCTURE, not just the level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("harmonic_dropout", "freq_split", "recovery")


@dataclass
class MBConfig:
    B: int = 6              # bands
    T: int = 96
    t0: int = 48
    period: float = 16.0
    noise: float = 0.25
    n_regions: int = 120
    event_frac: float = 0.5
    mode: str = "harmonic_dropout"
    changed_band: int = 2   # for attribution: which band carries the structural change (freq_split)
    seed: int = 0


def _two_harmonic(t, P, a1, a2, ph):
    return a1 * np.sin(2 * np.pi * t / P + ph) + a2 * np.sin(4 * np.pi * t / P + ph)


def _var_preserving_single(t, P, target_var, ph):
    """Single harmonic whose variance matches target_var (a1^2/2)."""
    a = np.sqrt(2 * target_var)
    return a * np.sin(2 * np.pi * t / P + ph)


def _make_region(cfg: MBConfig, is_event: bool, rng):
    t = np.arange(cfg.T)
    X = np.zeros((cfg.B, cfg.T))
    a1, a2 = 1.0, 0.7
    base_var = (a1 ** 2 + a2 ** 2) / 2.0
    for b in range(cfg.B):
        ph = rng.uniform(0, 2 * np.pi)
        pre = _two_harmonic(t, cfg.period, a1, a2, ph)
        x = pre.copy()
        if is_event:
            tt = t[cfg.t0:]
            if cfg.mode == "harmonic_dropout":          # 2nd harmonic vanishes; variance preserved
                x[cfg.t0:] = _var_preserving_single(tt, cfg.period, base_var, ph)
            elif cfg.mode == "freq_split":              # changed band shifts frequency; others unchanged
                if b == cfg.changed_band:
                    x[cfg.t0:] = _two_harmonic(tt, cfg.period / 1.7, a1, a2, ph)
            elif cfg.mode == "recovery":                # 2nd harmonic fades then returns
                half = (cfg.T - cfg.t0) // 2
                w = np.concatenate([np.linspace(1, 0, half), np.linspace(0, 1, len(tt) - half)])
                x[cfg.t0:] = a1 * np.sin(2 * np.pi * tt / cfg.period + ph) + \
                             (a2 * w) * np.sin(4 * np.pi * tt / cfg.period + ph)
        X[b] = x
    X = X + rng.normal(0, cfg.noise, X.shape)
    return X


def make_panel(cfg: MBConfig):
    """Returns X (n_regions, B, T) and labels (n_regions,) bool."""
    rng = np.random.default_rng(cfg.seed)
    n_evt = int(cfg.n_regions * cfg.event_frac)
    labels = np.array([True] * n_evt + [False] * (cfg.n_regions - n_evt))
    rng.shuffle(labels)
    X = np.stack([_make_region(cfg, bool(lb), rng) for lb in labels])
    return X, labels
