"""Linear anomalous-change detectors for paired multispectral imagery.

Source/provenance:
- Theiler and Perkins, "Proposed Framework for Anomalous Change Detection"
  (ICML workshop, 2006), equations 14-19, describes the chronochrome linear
  predictor and covariance-equalization detector.
- Schaum and Stocker's chronochrome/covariance-equalization family is treated
  here as established comparison pressure, not project novelty.

Project adaptation:
- A sample is one co-registered Sentinel-2 pixel pair. Covariances are fit on a
  seeded subset when a city contains too many pixels, then the frozen transform
  is applied to every valid pixel.
- Scores are min-max normalized only for map storage; AUROC/AP rankings are
  unchanged by that normalization.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la

Array = np.ndarray


def _paired_matrices(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    max_fit_pixels: int,
    random_state: int,
) -> tuple[Array, Array, Array, Array, Array]:
    valid = valid_mask.reshape(-1)
    first = x1.reshape(x1.shape[0], -1)[:, valid].astype(np.float64, copy=False)
    second = x2.reshape(x2.shape[0], -1)[:, valid].astype(np.float64, copy=False)
    if first.shape[1] == 0:
        raise ValueError("No valid paired pixels were supplied.")
    if first.shape[1] > int(max_fit_pixels):
        rng = np.random.default_rng(int(random_state))
        index = rng.choice(first.shape[1], size=int(max_fit_pixels), replace=False)
        fit_first = first[:, index]
        fit_second = second[:, index]
    else:
        fit_first = first
        fit_second = second
    return valid, first, second, fit_first, fit_second


def _cov(x: Array) -> Array:
    return (x @ x.T) / max(1, x.shape[1] - 1)


def _cross_cov(y: Array, x: Array) -> Array:
    return (y @ x.T) / max(1, x.shape[1] - 1)


def _ridge(cov: Array, regularization: float) -> Array:
    scale = float(np.trace(cov) / max(1, cov.shape[0]))
    ridge = max(float(regularization) * max(scale, 1e-12), 1e-10)
    return cov + ridge * np.eye(cov.shape[0], dtype=np.float64)


def _mahalanobis_columns(residual: Array, covariance: Array, regularization: float) -> Array:
    regularized = _ridge(covariance, regularization)
    solved = la.solve(regularized, residual, assume_a="pos", check_finite=False)
    return np.maximum(np.sum(residual * solved, axis=0), 0.0)


def _symmetric_inverse_sqrt(covariance: Array, regularization: float) -> Array:
    regularized = _ridge(covariance, regularization)
    values, vectors = la.eigh(regularized, check_finite=False)
    floor = max(float(np.max(values)) * float(regularization), 1e-10)
    values = np.maximum(values, floor)
    return (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T


def _score_map(values: Array, valid: Array, shape: tuple[int, int], eps: float = 1e-12) -> Array:
    output = np.zeros(valid.size, dtype=np.float32)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high > low:
        finite = (finite - low) / (high - low + eps)
    else:
        finite = np.zeros_like(finite)
    output[valid] = finite.astype(np.float32, copy=False)
    return output.reshape(shape)


def chronochrome_score(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    direction: str = "symmetric",
    max_fit_pixels: int = 200000,
    random_state: int = 1234,
    regularization: float = 1e-5,
) -> Array:
    """Score deviations from the learned cross-date linear relationship.

    The forward score implements y_hat = C X^-1 x and measures the prediction
    residual with its covariance. ``symmetric`` averages percentile-equivalent
    forward and reverse scores so neither acquisition is privileged.
    """
    valid, first, second, fit_first, fit_second = _paired_matrices(
        x1, x2, valid_mask, max_fit_pixels, random_state
    )
    mean_first = np.mean(fit_first, axis=1, keepdims=True)
    mean_second = np.mean(fit_second, axis=1, keepdims=True)
    fit_x = fit_first - mean_first
    fit_y = fit_second - mean_second
    full_x = first - mean_first
    full_y = second - mean_second
    cov_x = _cov(fit_x)
    cov_y = _cov(fit_y)
    cross_yx = _cross_cov(fit_y, fit_x)

    def forward_score() -> Array:
        transform = la.solve(_ridge(cov_x, regularization), cross_yx.T, assume_a="pos", check_finite=False).T
        residual_fit = fit_y - transform @ fit_x
        residual_full = full_y - transform @ full_x
        return _mahalanobis_columns(residual_full, _cov(residual_fit), regularization)

    def reverse_score() -> Array:
        transform = la.solve(_ridge(cov_y, regularization), cross_yx, assume_a="pos", check_finite=False).T
        residual_fit = fit_x - transform @ fit_y
        residual_full = full_x - transform @ full_y
        return _mahalanobis_columns(residual_full, _cov(residual_fit), regularization)

    key = str(direction).lower()
    if key == "forward":
        score = forward_score()
    elif key == "reverse":
        score = reverse_score()
    elif key == "symmetric":
        forward = forward_score()
        reverse = reverse_score()
        # Percentile ranks put the asymmetric scores on the same scale without labels.
        score = _percentile_ranks(forward) + _percentile_ranks(reverse)
    else:
        raise ValueError(f"Unknown chronochrome direction: {direction!r}")
    return _score_map(score, valid, valid_mask.shape)


def _percentile_ranks(values: Array) -> Array:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(1, values.size - 1)


def covariance_equalization_score(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    max_fit_pixels: int = 200000,
    random_state: int = 1234,
    regularization: float = 1e-5,
) -> Array:
    """Whiten each date separately, subtract, then score the residual covariance."""
    valid, first, second, fit_first, fit_second = _paired_matrices(
        x1, x2, valid_mask, max_fit_pixels, random_state
    )
    mean_first = np.mean(fit_first, axis=1, keepdims=True)
    mean_second = np.mean(fit_second, axis=1, keepdims=True)
    fit_x = fit_first - mean_first
    fit_y = fit_second - mean_second
    whiten_x = _symmetric_inverse_sqrt(_cov(fit_x), regularization)
    whiten_y = _symmetric_inverse_sqrt(_cov(fit_y), regularization)
    residual_fit = whiten_x @ fit_x - whiten_y @ fit_y
    residual_full = whiten_x @ (first - mean_first) - whiten_y @ (second - mean_second)
    score = _mahalanobis_columns(residual_full, _cov(residual_fit), regularization)
    return _score_map(score, valid, valid_mask.shape)
