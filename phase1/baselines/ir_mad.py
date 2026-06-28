"""IR-MAD baseline for multiband change detection.

Source/provenance:
- Nielsen 2007 regularized/iteratively reweighted MAD: CCA is used to build
  canonical variates for the two dates; their differences are MAD variates.
- Google Earth Engine iMAD tutorial: the iterative weights are chi-square
  survival probabilities of the standardized MAD statistic, so likely unchanged
  pixels dominate the next covariance estimate.

Project adaptation:
- Sentinel-2 OSCD tiles can contain more than one million valid pixels, so this
  implementation optionally subsamples pixels for estimating the CCA transforms
  and then applies the final transforms to all valid pixels.
- This is a compact baseline implementation, not a replacement for a full
  geospatial IR-MAD package. It is intended as fair classical comparison
  pressure for DS/PCA-diff experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from scipy.stats import chi2

Array = np.ndarray


@dataclass(frozen=True)
class IRMADResult:
    """Full iMAD result needed by downstream invariant-pixel methods."""

    score: Array
    no_change_probability: Array
    canonical_correlations: Array
    iterations: int


def _weighted_mean_cov(x: Array, w: Array, eps: float) -> tuple[Array, Array]:
    """Return weighted mean and covariance for samples shaped bands x pixels."""
    w = np.asarray(w, dtype=np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= eps:
        w = np.ones_like(w, dtype=np.float64)
        w_sum = float(w.size)
    wn = w / (w_sum + eps)
    mean = np.sum(x * wn[None, :], axis=1, keepdims=True)
    xc = x - mean
    cov = (xc * wn[None, :]) @ xc.T
    return mean, cov


def _weighted_cross_cov(x: Array, y: Array, mx: Array, my: Array, w: Array, eps: float) -> Array:
    w_sum = float(np.sum(w))
    if w_sum <= eps:
        w = np.ones_like(w, dtype=np.float64)
        w_sum = float(w.size)
    wn = w / (w_sum + eps)
    return ((x - mx) * wn[None, :]) @ (y - my).T


def _solve_cca(c11: Array, c22: Array, c12: Array, eps: float) -> tuple[Array, Array, Array]:
    """Solve paired CCA generalized eigenproblems.

    Returns A, B, rho where columns of A/B define canonical variates
    U=A^T X and V=B^T Y. Eigenvalues are rho^2.
    """
    p = c11.shape[0]
    reg11 = c11 + eps * np.eye(p)
    reg22 = c22 + eps * np.eye(p)

    inv22_c21 = la.solve(reg22, c12.T, assume_a="pos", check_finite=False)
    inv11_c12 = la.solve(reg11, c12, assume_a="pos", check_finite=False)
    mat_a = c12 @ inv22_c21
    mat_b = c12.T @ inv11_c12
    eig_a, a = la.eigh(mat_a, reg11, check_finite=False)
    eig_b, b = la.eigh(mat_b, reg22, check_finite=False)

    order_a = np.argsort(eig_a)[::-1]
    order_b = np.argsort(eig_b)[::-1]
    eig = np.clip(eig_a[order_a], 0.0, 1.0)
    a = a[:, order_a]
    b = b[:, order_b]
    rho = np.sqrt(eig)
    return a, b, rho


def _align_signs(a: Array, b: Array, c12: Array) -> tuple[Array, Array]:
    """Flip B signs so paired canonical variates have positive covariance."""
    diag = np.diag(a.T @ c12 @ b)
    signs = np.where(diag < 0.0, -1.0, 1.0)
    return a, b * signs[None, :]


def ir_mad_result(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    iters: int = 3,
    downsample_max_pixels: int = 200000,
    random_state: int = 1234,
    eps: float = 1e-6,
    convergence_tol: float = 1e-4,
) -> IRMADResult:
    """
    Compute IR-MAD chi-square change statistic. Returns min-max normalized score.
    """
    # Flatten valid pixels
    v = valid_mask.reshape(-1)
    mat1 = x1.reshape(x1.shape[0], -1)[:, v]
    mat2 = x2.reshape(x2.shape[0], -1)[:, v]
    n = mat1.shape[1]
    rng = np.random.default_rng(int(random_state))
    # Optional subsample for coefficient estimation
    if n > downsample_max_pixels:
        idx = rng.choice(n, size=downsample_max_pixels, replace=False)
        samp1 = mat1[:, idx]
        samp2 = mat2[:, idx]
    else:
        samp1 = mat1
        samp2 = mat2

    w = np.ones(samp1.shape[1], dtype=np.float64)
    A = np.eye(samp1.shape[0], dtype=np.float64)
    B = np.eye(samp2.shape[0], dtype=np.float64)
    rho = np.zeros(samp1.shape[0], dtype=np.float64)
    m1 = np.mean(samp1, axis=1, keepdims=True)
    m2 = np.mean(samp2, axis=1, keepdims=True)

    completed_iterations = 0
    for iteration in range(max(1, iters)):
        prev_rho = rho.copy()
        m1, c11 = _weighted_mean_cov(samp1, w, eps)
        m2, c22 = _weighted_mean_cov(samp2, w, eps)
        c12 = _weighted_cross_cov(samp1, samp2, m1, m2, w, eps)
        try:
            A, B, rho = _solve_cca(c11, c22, c12, eps)
            A, B = _align_signs(A, B, c12)
        except Exception:
            break

        completed_iterations = iteration + 1

        u = A.T @ (samp1 - m1)
        vv = B.T @ (samp2 - m2)
        mad = u - vv
        sigma2 = np.maximum(2.0 * (1.0 - rho), eps)
        z = np.sum((mad * mad) / sigma2[:, None], axis=0)
        # High p-values are more compatible with no-change and therefore get
        # higher weight in the next invariant-background estimate.
        w = chi2.sf(z, df=samp1.shape[0])
        if np.max(np.abs(rho - prev_rho)) < convergence_tol:
            break

    # Apply final transform to full matrices
    u_full = A.T @ (mat1 - m1)
    v_full = B.T @ (mat2 - m2)
    mad_full = u_full - v_full
    sigma2 = np.maximum(2.0 * (1.0 - rho), eps)
    mag = np.sum((mad_full * mad_full) / sigma2[:, None], axis=0)
    no_change = chi2.sf(mag, df=mat1.shape[0])
    score = np.zeros(valid_mask.size, dtype=np.float32)
    no_change_map = np.zeros(valid_mask.size, dtype=np.float32)
    score[v] = mag
    no_change_map[v] = no_change.astype(np.float32, copy=False)
    # Min-max normalize
    if np.any(score > 0):
        s_min, s_max = score[v].min(), score[v].max()
        if s_max > s_min:
            score[v] = (score[v] - s_min) / (s_max - s_min + eps)
        else:
            score[v] = 0.0
    return IRMADResult(
        score=score.reshape(valid_mask.shape),
        no_change_probability=no_change_map.reshape(valid_mask.shape),
        canonical_correlations=rho.astype(np.float32, copy=True),
        iterations=completed_iterations,
    )


def ir_mad_score(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    iters: int = 3,
    downsample_max_pixels: int = 200000,
    random_state: int = 1234,
    eps: float = 1e-6,
    convergence_tol: float = 1e-4,
) -> Array:
    """Compute the min-max normalized IR-MAD chi-square change statistic."""
    return ir_mad_result(
        x1=x1,
        x2=x2,
        valid_mask=valid_mask,
        iters=iters,
        downsample_max_pixels=downsample_max_pixels,
        random_state=random_state,
        eps=eps,
        convergence_tol=convergence_tol,
    ).score
