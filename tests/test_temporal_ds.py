"""L0 sanity / correctness tests for the temporal Difference Subspace core (temporal/subspace.py).

These are the label-free correctness invariants from docs/DESIGN_TEMPORAL_DS_ACCV2026.md s11.3 L0:
  - identical subspaces -> magnitude 0
  - rotation within the same span -> magnitude 0 (subspace, not basis, is what matters)
  - fully orthogonal k-dim -> magnitude 2k (max)
  - known canonical angle -> magnitude 2(1 - cos theta)
  - difference subspace of identical -> empty; karcher of identical -> the subspace
  - constant-velocity geodesic -> second-order (acceleration) ~ 0
  - accelerating geodesic -> second-order peaks at the speed change; d2 correlates with |d/dt d1|

Run:  ./.venv/Scripts/python.exe -m pytest tests/test_temporal_ds.py -q
 or:  ./.venv/Scripts/python.exe tests/test_temporal_ds.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal import subspace as ss
from temporal import dynamics as dyn

RNG = np.random.default_rng(0)


def _random_basis(d, k):
    return ss.orthonormalize(RNG.standard_normal((d, k)))


def test_identical_subspace_zero():
    B = _random_basis(13, 6)
    assert abs(ss.magnitude(B, B)) < 1e-9


def test_rotation_within_span_zero():
    """Same span, different (rotated) basis -> magnitude must be ~0 (DS is a function of the subspace)."""
    B = _random_basis(13, 6)
    R, _ = np.linalg.qr(RNG.standard_normal((6, 6)))  # in-subspace rotation
    B2 = B @ R
    assert ss.magnitude(B, B2) < 1e-9


def test_orthogonal_is_max():
    """Two orthogonal k-dim subspaces of R^2k -> magnitude = 2k (all cos theta = 0)."""
    k = 4
    Q = _random_basis(2 * k, 2 * k)
    B1, B2 = Q[:, :k], Q[:, k:]
    assert abs(ss.magnitude(B1, B2) - 2 * k) < 1e-6


def test_known_canonical_angle():
    """One-dim subspaces separated by theta -> magnitude = 2(1 - cos theta)."""
    for theta in [0.1, 0.5, 1.0, np.pi / 2]:
        v1 = np.array([[1.0], [0.0], [0.0]])
        v2 = np.array([[np.cos(theta)], [np.sin(theta)], [0.0]])
        assert abs(ss.magnitude(v1, v2) - 2 * (1 - np.cos(theta))) < 1e-9


def test_difference_subspace_identical_empty():
    B = _random_basis(13, 6)
    D = ss.difference_subspace(B, B)
    assert D.shape[1] == 0


def test_karcher_identical_is_subspace():
    B = _random_basis(13, 6)
    M = ss.karcher_subspace(B, B)
    assert M.shape[1] == 6
    assert ss.magnitude(M, B) < 1e-9


def _geodesic_trajectory(d, k, angles, seed=1):
    """Subspace trajectory S(t) = expm(angle(t) * A) @ B0 for a fixed skew-symmetric generator A.

    `angles` is the cumulative rotation angle at each time step. Constant step => constant velocity.
    Returns list of orthonormal bases.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    A = A - A.T  # skew-symmetric => expm(theta A) is a rotation
    A = A / np.linalg.norm(A)
    B0 = ss.orthonormalize(rng.standard_normal((d, k)))
    return [ss.orthonormalize(expm(a * A) @ B0) for a in angles]


def test_constant_velocity_zero_acceleration():
    """Constant-velocity geodesic: d1 ~ constant, second-order magnitude ~ 0 vs d1."""
    angles = np.linspace(0, 1.2, 40)  # uniform steps => constant velocity
    traj = _geodesic_trajectory(12, 3, angles)
    d1 = dyn.velocity_curve(traj, lag=1)
    acc = dyn.acceleration_curve(traj, lag=1)
    d2 = acc["d2"]
    # acceleration should be tiny relative to velocity on a constant-speed geodesic
    assert np.median(d2) < 0.05 * np.median(d1), (np.median(d2), np.median(d1))


def test_accelerating_geodesic_detects_speed_change():
    """A trajectory that is slow, then abruptly fast: acceleration must peak at the transition,
    and d2 should correlate positively with |d/dt d1|."""
    slow = np.linspace(0, 0.15, 25)
    fast = slow[-1] + np.linspace(0, 1.6, 25)[1:]
    angles = np.concatenate([slow, fast])  # speed jumps at index ~25
    traj = _geodesic_trajectory(12, 3, angles, seed=2)
    d1 = dyn.velocity_curve(traj, lag=1)
    acc = dyn.acceleration_curve(traj, lag=1)
    d2 = acc["d2"]
    # peak of acceleration near the speed transition
    transition = 25 - acc["center_index"][0]
    peak = int(np.argmax(d2))
    assert abs(peak - transition) <= 4, (peak, transition)
    # positive correlation between acceleration and |d/dt velocity|
    dd1 = dyn.numeric_derivative(d1)[acc["center_index"][0]:acc["center_index"][0] + len(d2)]
    r = np.corrcoef(d2, dd1)[0, 1]
    assert r > 0.5, r


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} L0 sanity tests passed.")
    return passed == len(fns)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
