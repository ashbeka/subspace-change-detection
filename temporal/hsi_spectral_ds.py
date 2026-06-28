"""#1 Spectral-signal Difference Subspace for hyperspectral change detection.

Per pixel, treat the B-band spectrum as a 1-D signal and delay-embed it ALONG WAVELENGTH (Hankel) -> a
multi-dimensional 'spectral-shape subspace' (top-r SVD). Change score = DS magnitude between the two dates'
subspaces. This subspace is amplitude-invariant (scale-invariant), so it should resist illumination
pseudo-change that defeats CVA/raw-diff. The dimensionality hypothesis: at low band count the subspace is
rank~1 and DS == spectral angle (SAM); at 224 bands it is genuinely multi-dim and DS can beat SAM.

L0 must pass first: verify the spectral-SSA subspace has intrinsic rank > 1 on real data, else DS degenerates.
"""
from __future__ import annotations

import numpy as np

from . import subspace as ss


def hankel_wavelength(spec: np.ndarray, w: int) -> np.ndarray:
    """Spectrum (B,) -> Hankel trajectory matrix (w, B-w+1) along the wavelength axis."""
    B = spec.shape[0]
    K = B - w + 1
    return np.stack([spec[i:i + K] for i in range(w)])          # (w, K)


def spectral_ssa_subspace(spec: np.ndarray, w: int = 20, rank: int = 8, energy: float = 0.99):
    """Spectrum (B,) -> orthonormal (w, r) spectral-shape subspace (centered Hankel, top-r SVD)."""
    H = hankel_wavelength(np.asarray(spec, float), w)
    return ss.pca_subspace(H, rank, center=True, energy=energy)


def ds_score(s1: np.ndarray, s2: np.ndarray, w: int = 20, rank: int = 8) -> float:
    """DS magnitude between the spectral-shape subspaces of two spectra of the same pixel."""
    A = spectral_ssa_subspace(s1, w, rank)
    Bb = spectral_ssa_subspace(s2, w, rank)
    if A is None or Bb is None:
        return 0.0
    return ss.magnitude(A, Bb)


def sam(s1: np.ndarray, s2: np.ndarray) -> float:
    """Spectral Angle Mapper (amplitude-invariant scalar baseline) = the rank-1 / DS-degenerate limit."""
    a = np.asarray(s1, float); b = np.asarray(s2, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1)))


def cva(s1, s2):
    """Change Vector Analysis = L2 of the spectral difference (amplitude-SENSITIVE)."""
    return float(np.linalg.norm(np.asarray(s1, float) - np.asarray(s2, float)))


def intrinsic_rank(spec: np.ndarray, w: int = 20, energy: float = 0.95) -> int:
    """L0 check: how many SVD components of the wavelength-Hankel reach `energy` (rank>1 => not degenerate)."""
    H = hankel_wavelength(np.asarray(spec, float), w)
    Hc = H - H.mean(1, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    e = np.cumsum(s ** 2) / (np.sum(s ** 2) + 1e-12)
    return int(np.searchsorted(e, energy) + 1)


def score_image(img1, img2, method="ds", w=20, rank=8, sample_idx=None):
    """img1,img2: (N_pixels, B) standardized spectra. Returns per-pixel change score (N,).
    sample_idx: optional indices to score a subset (for speed)."""
    N = img1.shape[0]
    idx = np.arange(N) if sample_idx is None else sample_idx
    out = np.zeros(len(idx))
    for k, i in enumerate(idx):
        s1, s2 = img1[i], img2[i]
        if method == "ds":
            out[k] = ds_score(s1, s2, w, rank)
        elif method == "sam":
            out[k] = sam(s1, s2)
        elif method == "cva":
            out[k] = cva(s1, s2)
        elif method == "rawmean":
            out[k] = float(np.mean(np.abs(s1 - s2)))
        else:
            raise ValueError(method)
    return out
