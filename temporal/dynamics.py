"""Temporal dynamics: turn a sequence of per-date subspaces into velocity / acceleration / recovery curves.

A subspace trajectory S(0), S(1), ..., S(T-1) on the Grassmann manifold (one subspace per date).
  velocity     d1(t)        = magnitude(S(t), S(t+lag))                      -> change onset
  acceleration d2(t)        = second_order_magnitude(S(t), S(t+lag), S(t+2*lag)) -> abrupt vs uniform
  geodesic     (along, orth)= second_order_decomposed(...)                   -> on-path vs regime-break
  recovery     d_pre(t)     = magnitude(S(t), S_pre)                         -> rising=degrading, falling=recovering
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import subspace as ss


def velocity_curve(bases: Sequence[np.ndarray], lag: int = 1) -> np.ndarray:
    """d1(t) = magnitude(S(t), S(t+lag)). Returned aligned to the *center* of the pair: index t."""
    T = len(bases)
    return np.array([ss.magnitude(bases[t], bases[t + lag]) for t in range(T - lag)])


def acceleration_curve(bases: Sequence[np.ndarray], lag: int = 1):
    """Returns dict with d2 (direct second-order magnitude), along, orth curves over valid t.

    Uses the triple (S(t), S(t+lag), S(t+2*lag)); the score is attributed to the middle index t+lag.
    """
    T = len(bases)
    d2, along, orth = [], [], []
    for t in range(T - 2 * lag):
        b1, b2, b3 = bases[t], bases[t + lag], bases[t + 2 * lag]
        d2.append(ss.second_order_magnitude(b1, b2, b3))
        a, o = ss.second_order_decomposed(b1, b2, b3)
        along.append(a)
        orth.append(o)
    return {
        "center_index": np.arange(lag, T - lag),
        "d2": np.array(d2),
        "along": np.array(along),
        "orth": np.array(orth),
    }


def distance_to_baseline(bases: Sequence[np.ndarray], ref: np.ndarray) -> np.ndarray:
    """d_pre(t) = magnitude(S(t), ref). Rising => degrading (away from baseline), falling => recovering."""
    return np.array([ss.magnitude(b, ref) for b in bases])


def numeric_derivative(curve: np.ndarray) -> np.ndarray:
    """Central finite difference |d/dt| of a curve, for the d2 ~ d/dt d1 self-consistency check."""
    return np.abs(np.gradient(np.asarray(curve, dtype=np.float64)))
