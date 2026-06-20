"""Faithful implementation of Kanai, Sogi, Maki, Fukui (2023) "Time-series Anomaly Detection based on
Difference Subspace between Signal Subspaces" (arXiv:2303.17802) — the lab's actual change/anomaly method,
which my earlier 'temporal DS' work never built. See docs/SUBSPACE_CONSTRUCTION_LEDGER.md §1.

Construction (what each subspace IS):
  - trajectory (Hankel) matrix H_t in R^{w x M} from a 1D window of a series  (SSA).
  - SIGNAL SUBSPACE P_t = top-r left singular vectors of H_t  -> dominant temporal/oscillatory structure.
  - DIFFERENCE SUBSPACE D between past P_{t-tau} and present P_t (they overlap -> generalized DS handles it):
    the directions in which the present temporal structure DIFFERS from the past.
  - NON-ANOMALOUS REFERENCE D_N: principal subspace of the DSs seen during a NORMAL period -> the *normal*
    change-direction. Score = how far the current DS departs from D_N (direction) x magnitude departure.
This 'model-the-normal, flag-the-residual' reference is the invariance lever.
"""
from __future__ import annotations

import numpy as np

from . import subspace as ss


def trajectory_matrix(h: np.ndarray, w: int, M: int) -> np.ndarray:
    """Hankel trajectory matrix (w x M) from the LAST w+M-1 samples of h (Kanai Eq.1)."""
    h = np.asarray(h, float)
    base = len(h) - (w + M - 1)
    if base < 0:
        raise ValueError(f"series too short: need {w+M-1}, got {len(h)}")
    return np.stack([h[base + c: base + c + w] for c in range(M)], axis=1)


def signal_subspace(h: np.ndarray, w: int, M: int, r: int) -> np.ndarray:
    """SSA signal subspace: top-r left singular vectors of the trajectory matrix (w x r)."""
    H = trajectory_matrix(h, w, M)
    U, _, _ = np.linalg.svd(H, full_matrices=False)
    return np.ascontiguousarray(U[:, :min(r, U.shape[1])])


def mssa_signal_subspace(X: np.ndarray, w: int, M: int, r: int) -> np.ndarray:
    """Multivariate-SSA signal subspace from a multichannel series X (L x B): per-band Hankel matrices
    stacked vertically -> trajectory matrix (B*w x M); top-r left singular vectors. REPRESENTS: the joint
    spectral-temporal structure of the window (captures cross-band + temporal correlation simultaneously)."""
    X = np.asarray(X, float)
    base = len(X) - (w + M - 1)
    if base < 0:
        raise ValueError(f"series too short: need {w+M-1}, got {len(X)}")
    blocks = [np.stack([X[base + c: base + c + w, b] for c in range(M)], axis=1) for b in range(X.shape[1])]
    Hm = np.vstack(blocks)                                          # (B*w, M)
    U, _, _ = np.linalg.svd(Hm, full_matrices=False)
    return np.ascontiguousarray(U[:, :min(r, U.shape[1])])


def _mu(P_past: np.ndarray, P_pres: np.ndarray) -> float:
    """Log super-volume of the DS = sum log cos(theta_i) over past/present canonical angles (Kanai Eq.5)."""
    cos = np.clip(ss.canonical_cosines(P_past, P_pres), 1e-6, 1.0)
    return float(np.sum(np.log(cos)))


def learn_reference(normal: np.ndarray, w: int, M: int, r: int, tau: int,
                    nor_dims: int | None = None, stride: int = 1):
    """Learn the non-anomalous reference DS D_N and its mean log-volume mu_N from a NORMAL series."""
    span = w + M - 1
    Ds, mus = [], []
    for t in range(span + tau, len(normal) + 1, stride):
        Pp = signal_subspace(normal[t - tau - span: t - tau], w, M, r)
        Pc = signal_subspace(normal[t - span: t], w, M, r)
        D = ss.difference_subspace(Pp, Pc)
        if D.shape[1] == 0:
            continue
        Ds.append(D); mus.append(_mu(Pp, Pc))
    if not Ds:
        raise RuntimeError("no non-degenerate DS in the normal period (check w,M,r,tau)")
    G = sum(D @ D.T for D in Ds)                                   # principal subspace of normal DSs
    vals, vecs = np.linalg.eigh(G)
    if nor_dims is None:
        nor_dims = max(1, int(np.median([D.shape[1] for D in Ds])))
    D_N = np.ascontiguousarray(vecs[:, ::-1][:, :nor_dims])
    return D_N, float(np.mean(mus))


def change_degree(series: np.ndarray, w: int, M: int, r: int, tau: int,
                  D_N: np.ndarray, mu_N: float, c: int | None = None, stride: int = 1):
    """Slide over `series`; return (end_index, a_hat, baseline_min_angle, baseline_mean_angle).

    a_hat = beta * delta  (Kanai Eq.6):  delta = direction index = mean(1 - cos<D_in, D_N>);
                                         beta  = (mu(D_in) - mu_N)^2 = magnitude index.
    Baselines = conventional SSA dissimilarities: 1 - cos(theta_min) and mean(1 - cos theta_i).
    """
    span = w + M - 1
    ts, a_hat, b_min, b_mean = [], [], [], []
    for t in range(span + tau, len(series) + 1, stride):
        Pp = signal_subspace(series[t - tau - span: t - tau], w, M, r)
        Pc = signal_subspace(series[t - span: t], w, M, r)
        cab = ss.canonical_cosines(Pp, Pc)                        # past/present canonical cosines (desc)
        b_min.append(1.0 - float(np.max(cab)))                    # 1 - cos(theta_min)
        b_mean.append(float(np.mean(1.0 - cab)))                  # mean(1 - cos)
        D = ss.difference_subspace(Pp, Pc)
        if D.shape[1] == 0:
            a_hat.append(0.0); ts.append(t - 1); continue
        cref = ss.canonical_cosines(D, D_N)
        cc = c or len(cref)
        delta = float(np.mean(1.0 - cref[:cc]))                   # direction departure from normal DS
        beta = (_mu(Pp, Pc) - mu_N) ** 2                          # magnitude departure
        a_hat.append(beta * delta); ts.append(t - 1)
    return np.array(ts), np.array(a_hat), np.array(b_min), np.array(b_mean)
