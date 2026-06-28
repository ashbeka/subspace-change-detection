"""Slow Feature Analysis (SFA) and the Slow Feature Subspace, for satellite signal change detection.

Motivation: a PCA/M-SSA subspace is VARIANCE-dominated, so the seasonal cycle swamps it and a Difference
Subspace cannot tell 'the cycle structurally changed' from 'the cycle is oscillating' (this killed the
temporal-DS tests). SFA (Wiskott; PCA-SFA, Kobayashi 2017; Slow Feature Subspace, Beleza/Fukui 2023) is
ORDER-sensitive: it extracts the slowly-varying structure by minimizing the variance of the temporal
derivative. The slow-feature SUBSPACE is therefore sensitive to a change in the temporal dynamics where a
variance subspace is blind (Beleza et al. Fig.1: reversed-frame videos give identical PCA subspaces but
different SFA subspaces).

sfa_subspace(X): X is (d, T) signal-over-time (columns = time). Solves the generalized eigenproblem
    Cder w = lambda Cov w      (Cder = cov of temporal derivative, Cov = data cov)
slow features = SMALLEST lambda. The slow-feature subspace spans the k slowest weight vectors.
A "Slow-Feature Difference Subspace" (SFA-DS) = DS magnitude between two slow-feature subspaces.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from . import subspace as ss


def sfa_subspace(X: np.ndarray, n_slow: int = 4, eps: float = 1e-4):
    """(d, T) signal-over-time -> orthonormal (d, k) slow-feature subspace (k slowest directions)."""
    X = np.asarray(X, float)
    d, T = X.shape
    if T < 4:
        return None
    Xc = X - X.mean(1, keepdims=True)
    Xdot = np.diff(X, axis=1)
    Cov = Xc @ Xc.T / max(T - 1, 1) + eps * np.eye(d)
    Cder = Xdot @ Xdot.T / max(T - 2, 1) + eps * np.eye(d)
    lam, W = eigh(Cder, Cov)                 # ascending lambda; slow = smallest
    k = int(min(n_slow, W.shape[1]))
    Q, _ = np.linalg.qr(W[:, :k])            # orthonormalize for canonical-angle comparison
    return Q[:, :k]


def sfa_ds(Xa: np.ndarray, Xb: np.ndarray, n_slow: int = 4):
    """Slow-Feature Difference Subspace magnitude between two signal windows (d, T)."""
    Sa, Sb = sfa_subspace(Xa, n_slow), sfa_subspace(Xb, n_slow)
    if Sa is None or Sb is None:
        return 0.0
    return ss.magnitude(Sa, Sb)


def slowness(X: np.ndarray, eps: float = 1e-4):
    """Mean temporal-derivative variance ratio of the slowest component (small = very slow structure)."""
    sub = sfa_subspace(X, n_slow=1, eps=eps)
    if sub is None:
        return np.nan
    y = sub[:, 0] @ (X - X.mean(1, keepdims=True))
    return float(np.var(np.diff(y)) / (np.var(y) + 1e-9))
