"""Factor local hyperspectral change into mean, scale, shape, and orientation.

Research sources
----------------
- Principal angles and projector geometry are standard Grassmann/subspace
  comparisons and are the same geometric ingredients used by first-order DS.
- Bures covariance distance follows the Gaussian/covariance geometry identity
  ``tr(C1)+tr(C2)-2 tr((C1^1/2 C2 C1^1/2)^1/2)``.
- The experiment is pressure-tested against Wu, Du, and Zhang (JSTARS 2013,
  DOI 10.1109/JSTARS.2013.2241396) and covariance-equalization ACD.  This file
  does not claim those established HSI methods as project inventions.

Project hypothesis
------------------
For local pixel sets from two dates, separate four different changes instead of
calling all covariance change "DS": local mean, total covariance scale,
normalized eigenspectrum shape, and leading-eigenspace orientation.  The
orientation attribution is the row energy of ``P1-P2``.  It is invariant to
basis sign and within-subspace rotation, and its sum exactly equals the squared
projector distance, making it an auditable sensor-band decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la


Array = np.ndarray


@dataclass(frozen=True)
class LocalMomentGeometry:
    mean_distance: float
    log_scale_change: float
    eigenspectrum_hellinger: float
    orientation_chordal: float
    orientation_chordal_normalized: float
    first_order_ds_magnitude: float
    normalized_covariance_frobenius: float
    normalized_covariance_bures: float
    orientation_attribution: Array
    normalized_covariance_attribution: Array
    principal_cosines: Array
    first_eigengap: float
    second_eigengap: float
    effective_rank: int
    difference_basis: Array


@dataclass(frozen=True)
class SpectralSubspace:
    mean: Array
    centered: Array
    basis: Array
    eigen_profile: Array
    trace: float
    eigengap: float


def _top_spectral_subspace(samples: Array, rank: int, seed: int, eps: float) -> SpectralSubspace:
    """Fit an exact centered basis from rows=pixels, columns=bands.

    Local eigenspace orientation and projector attribution must be reproducible.
    We therefore solve the smaller of the sample and feature Gram eigenproblems
    exactly instead of using randomized SVD. ``seed`` remains in the signature
    for API compatibility but does not influence this linear-algebra step.
    """
    del seed
    if samples.ndim != 2 or samples.shape[0] < 3:
        raise ValueError(f"Expected at least three pixel samples, got {samples.shape}.")
    mean = np.mean(samples, axis=0)
    centered = samples.astype(np.float64, copy=False) - mean[None, :]
    trace = float(np.sum(centered * centered) / max(1, centered.shape[0] - 1))
    maximum_rank = min(centered.shape[0] - 1, centered.shape[1])
    effective_rank = max(1, min(int(rank), maximum_rank))
    requested = min(maximum_rank, effective_rank + 1)
    sample_count, feature_count = centered.shape
    if sample_count <= feature_count:
        gram = centered @ centered.T
        values, left_vectors = la.eigh(
            gram,
            subset_by_index=(sample_count - requested, sample_count - 1),
            check_finite=False,
            driver="evr",
        )
        order = np.argsort(values)[::-1]
        singular = np.sqrt(np.maximum(values[order], 0.0))
        left_vectors = left_vectors[:, order]
        if float(np.min(singular)) <= eps:
            _, singular, vt = la.svd(
                centered,
                full_matrices=False,
                check_finite=False,
                lapack_driver="gesdd",
            )
            singular = singular[:requested]
            vt = vt[:requested]
        else:
            right_vectors = centered.T @ left_vectors
            right_vectors /= singular[None, :]
            vt = right_vectors.T
    else:
        gram = centered.T @ centered
        values, right_vectors = la.eigh(
            gram,
            subset_by_index=(feature_count - requested, feature_count - 1),
            check_finite=False,
            driver="evr",
        )
        order = np.argsort(values)[::-1]
        singular = np.sqrt(np.maximum(values[order], 0.0))
        vt = right_vectors[:, order].T
    leading = np.square(singular[:effective_rank]) / max(centered.shape[0] - 1, 1)
    residual = max(trace - float(np.sum(leading)), 0.0)
    profile = np.concatenate([leading, np.asarray([residual])]) / max(trace, eps)
    if singular.size > effective_rank:
        lambda_r = float(singular[effective_rank - 1] ** 2)
        lambda_next = float(singular[effective_rank] ** 2)
        eigengap = max(lambda_r - lambda_next, 0.0) / max(lambda_r, eps)
    else:
        eigengap = 1.0
    return SpectralSubspace(
        mean=mean,
        centered=centered,
        basis=vt[:effective_rank].T,
        eigen_profile=profile,
        trace=trace,
        eigengap=eigengap,
    )


def projector_row_energy(first_basis: Array, second_basis: Array) -> Array:
    """Return per-feature contribution to ``||P1-P2||_F^2``.

    The result is unchanged by sign flips or right multiplication of either
    basis by an orthogonal matrix.  Summing the result gives twice the squared
    chordal distance between equal-rank subspaces.
    """
    if first_basis.shape != second_basis.shape:
        raise ValueError(f"Basis mismatch: {first_basis.shape} vs {second_basis.shape}.")
    difference = first_basis @ first_basis.T - second_basis @ second_basis.T
    return np.sum(difference * difference, axis=1)


def factor_local_moments(
    first_samples: Array,
    second_samples: Array,
    rank: int,
    seed: int = 1234,
    eps: float = 1e-10,
) -> LocalMomentGeometry:
    """Factor one pair of local hyperspectral pixel sets."""
    if first_samples.shape != second_samples.shape:
        raise ValueError(
            f"Local sample mismatch: {first_samples.shape} vs {second_samples.shape}."
        )
    first = _top_spectral_subspace(first_samples, rank, seed, eps)
    second = _top_spectral_subspace(second_samples, rank, seed + 1, eps)
    effective_rank = min(first.basis.shape[1], second.basis.shape[1])
    first_basis = first.basis[:, :effective_rank]
    second_basis = second.basis[:, :effective_rank]
    cosines = la.svdvals(first_basis.T @ second_basis)
    cosines = np.clip(cosines, 0.0, 1.0)
    chordal_squared = float(np.sum(1.0 - np.square(cosines)))
    chordal = float(np.sqrt(max(chordal_squared, 0.0)))
    principal_left, principal_cosines, principal_right_t = la.svd(
        first_basis.T @ second_basis, full_matrices=False, check_finite=False
    )
    principal_cosines = np.clip(principal_cosines, 0.0, 1.0)
    keep = (1.0 - principal_cosines) > 1e-7
    if np.any(keep):
        principal_difference = (
            first_basis @ principal_left[:, keep]
            - second_basis @ principal_right_t.T[:, keep]
        )
        squared_pair_magnitude = 2.0 * (1.0 - principal_cosines[keep])
        difference_basis = principal_difference / np.sqrt(squared_pair_magnitude)[None, :]
        difference_basis /= np.maximum(
            np.linalg.norm(difference_basis, axis=0, keepdims=True), eps
        )
    else:
        squared_pair_magnitude = np.zeros((0,), dtype=np.float64)
        difference_basis = np.zeros((first_basis.shape[0], 0), dtype=np.float64)

    # Unit-trace covariance distances can be computed from the centered sample
    # matrices without materializing one bands-by-bands covariance per window.
    y1 = first.centered / np.sqrt(max(float(np.sum(first.centered**2)), eps))
    y2 = second.centered / np.sqrt(max(float(np.sum(second.centered**2)), eps))
    gram1 = y1 @ y1.T
    gram2 = y2 @ y2.T
    cross = y1 @ y2.T
    cov_frob_sq = (
        float(np.sum(gram1 * gram1))
        + float(np.sum(gram2 * gram2))
        - 2.0 * float(np.sum(cross * cross))
    )
    fidelity = float(np.sum(la.svdvals(cross)))
    bures_sq = max(2.0 - 2.0 * min(fidelity, 1.0), 0.0)

    profile_first = first.eigen_profile
    profile_second = second.eigen_profile
    profile_size = max(profile_first.size, profile_second.size)
    profile_first = np.pad(profile_first, (0, profile_size - profile_first.size))
    profile_second = np.pad(profile_second, (0, profile_size - profile_second.size))
    hellinger = np.linalg.norm(np.sqrt(profile_first) - np.sqrt(profile_second)) / np.sqrt(2.0)

    projector_attribution = projector_row_energy(first_basis, second_basis)
    covariance_difference = y1.T @ y1 - y2.T @ y2
    covariance_attribution = np.sum(covariance_difference**2, axis=1)
    return LocalMomentGeometry(
        mean_distance=float(np.linalg.norm(second.mean - first.mean)),
        log_scale_change=float(abs(np.log(max(second.trace, eps) / max(first.trace, eps)))),
        eigenspectrum_hellinger=float(hellinger),
        orientation_chordal=chordal,
        orientation_chordal_normalized=chordal / np.sqrt(max(1, effective_rank)),
        first_order_ds_magnitude=float(
            np.sum(squared_pair_magnitude) / max(1, effective_rank)
        ),
        normalized_covariance_frobenius=float(np.sqrt(max(cov_frob_sq, 0.0))),
        normalized_covariance_bures=float(np.sqrt(bures_sq)),
        orientation_attribution=projector_attribution.astype(np.float32),
        normalized_covariance_attribution=covariance_attribution.astype(np.float32),
        principal_cosines=cosines.astype(np.float32),
        first_eigengap=float(first.eigengap),
        second_eigengap=float(second.eigengap),
        effective_rank=effective_rank,
        difference_basis=difference_basis.astype(np.float32),
    )


def shrinkage_log_euclidean_distance(
    first_samples: Array,
    second_samples: Array,
    shrinkage: float = 0.1,
    eps: float = 1e-8,
) -> float:
    """Log-Euclidean distance between fixed-shrinkage local covariances.

    This control operates in a caller-provided low-dimensional global PCA
    space.  Shrinkage toward each covariance's isotropic trace target makes the
    logarithm well-defined when a local window has fewer samples than bands.
    """
    if first_samples.shape != second_samples.shape:
        raise ValueError("Shrinkage-SPD samples must have matching shapes.")
    dimension = first_samples.shape[1]

    def covariance(samples: Array) -> Array:
        centered = samples - np.mean(samples, axis=0, keepdims=True)
        empirical = centered.T @ centered / max(1, samples.shape[0] - 1)
        scale = float(np.trace(empirical) / max(1, dimension))
        alpha = float(np.clip(shrinkage, 0.0, 1.0))
        return (1.0 - alpha) * empirical + alpha * max(scale, eps) * np.eye(dimension)

    first_cov = covariance(first_samples.astype(np.float64, copy=False))
    second_cov = covariance(second_samples.astype(np.float64, copy=False))

    def symmetric_log(matrix: Array) -> Array:
        values, vectors = la.eigh(matrix, check_finite=False)
        values = np.maximum(values, eps)
        return (vectors * np.log(values)[None, :]) @ vectors.T

    return float(la.norm(symmetric_log(first_cov) - symmetric_log(second_cov), ord="fro"))
