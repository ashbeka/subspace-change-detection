# BET 1 — Registration-robust subtle distributional HSI change in the N<<B regime: rigorous design + pre-registration

Created 2026-06-21. Full decomposition + math + pre-registered nulls/metrics/falsifier BEFORE coding. This is the
one cell not yet fairly tested: my earlier N<<B run failed only because Hermiston's change is huge + amplitude-
dominated (1-px misregistration negligible, subspace discards amplitude). Here we test the regime where the
set-subspace's three structural properties ALL matter at once.

## 1. Refined problem statement
For SUBTLE, DISTRIBUTIONAL, (near-)mean-preserving change in LOCAL hyperspectral patches where N (pixels/patch)
<< B (bands), detect/score the change with a low-rank SET-SUBSPACE comparison, robust to sub-pixel
misregistration and global illumination, where (a) per-pixel methods false-alarm on misregistration, (b) the
patch mean misses the distributional change, and (c) full/shrinkage covariance & correlation distances are
noise-dominated at N<<B.

## 2. Decomposition — what each competitor can and cannot do (the honest map)
| property needed | per-pixel CVA/SAM | patch-mean | sample cov/corr | shrinkage cov/corr | SET-SUBSPACE |
|---|---|---|---|---|---|
| registration-robust (permutation-invariant) | NO (aligns wrong px) | YES | YES | YES | YES |
| catches mean-PRESERVING distributional change | partial | NO | YES | YES | YES |
| well-posed at N<<B (rank ≤ N-1 << B) | n/a | YES | NO (rank-deficient/noisy) | YES (regularized) | YES (low-rank IS the estimand) |
| illumination/scale-invariant | SAM yes/CVA no | SAM yes | corr yes | corr yes | yes (angles) |
**Key consequence:** registration-robustness is SHARED by all patch methods → it only rules out per-pixel; it
does NOT favor the subspace over patch-mean/correlation. The genuine crux is: **does the SET-SUBSPACE beat
SHRINKAGE-COVARIANCE/CORRELATION (the regularized N<<B competitor) for SUBTLE distributional change?** That is
the falsifier; everything else (per-pixel, patch-mean, sample-correlation) is a weaker null.

## 3. Math — why N<<B is the subspace's only structural home
- Sample covariance Ĉ = (1/N)Σ(x-μ̄)(x-μ̄)ᵀ ∈ R^{B×B} has rank ≤ N-1. At N=25, B=200, 175+ eigen-directions are
  pure sampling noise. Correlation R̂ and any full-matrix distance (Frobenius, log-Euclidean/SPD) integrate that
  noise → high-variance estimator. log-Euclidean is UNDEFINED (Ĉ singular) without regularization.
- A change Δ = C2 − C1 that is LOW-RANK (a few new/rotated spectral directions — the signature of sub-pixel
  mixing / incipient stress / partial transition) lives in a few directions. The top-k subspace (k<N) is the
  consistently-estimable part; comparing the top-k subspaces (canonical angles) tracks the low-rank Δ while
  discarding the noise tail. Shrinkage (Ledoit-Wolf) instead pulls Ĉ toward a target — it regularizes but keeps
  all directions (incl. noise) and a scale (less scale-invariant). HYPOTHESIS: for a low-rank Δ at N<<B, the
  low-rank subspace is a lower-variance detector than shrinkage-cov/corr. THIS is the only place the prior
  evidence (subspace loses everywhere at N>>B) does not pre-determine the outcome.
- Estimators (pre-registered): set-subspace score = DS magnitude 2·Σ(1−cosθ_i) between the centered top-k
  subspaces of the two patches (also report Σ(1−cosθ_i)/k = mean principal angle). Centered ⇒ mean-removed ⇒
  fair vs patch-mean.

## 4. Pre-registered experimental program (run in order; each a falsifiable gate)
- **E1 — the crux sweep (non-spatial, random pixel sets from a real class).** Real Salinas-class spectra pool.
  Sweep N ∈ {9,16,25,49,100,200} at B∈{100,204}. Change = (a) sub-pixel mix (replace fraction f of pixels with
  material B), (b) mean-preserving new-direction loading. Subtle magnitude (tune so best AUC ≈ 0.8 at largest N).
  Nuisance: global illumination α∈{0,0.1}. Methods: patch-mean CVA/SAM, sample-corr-Frob, shrink-corr-Frob,
  shrink-cov-logEuclid, SET-SUBSPACE (DS, mean-angle). Metric: AUC(changed vs unchanged) vs N.
  **Falsifier:** if SET-SUBSPACE never beats shrink-corr/shrink-cov at small N (its predicted home), Bet 1 dies.
- **E2 — spatial + misregistration.** Real image patches (Hermiston/real); add sub-pixel misregistration; add
  per-pixel CVA/SAM/IR-MAD as the registration-fooled nulls. Confirms the SHARED registration advantage of patch
  methods and tests whether subspace adds over patch-mean/shrink-corr under misregistration too.
- **E3 — change-type & subtlety surface.** Sweep change magnitude × type (mean / scale / orientation / mix /
  heterogeneity) to map exactly which (subtle, distributional) changes the subspace uniquely catches.
- **E4 — real validation.** Inject subtle distributional change into real Hermiston no-change regions; + any real
  subtle-change labels available. Honest: show where it holds and where it doesn't.
- **E5 — adversarial verification (workflow):** independent skeptics attack the strongest result (is shrinkage
  tuned fairly? is the change rigged? is the small-N win a variance artifact that vanishes with more trials?).

## 5. Honest priors (no overclaiming)
- Registration robustness is real but SHARED → not the differentiator. The whole bet rides on the N<<B low-rank
  denoising edge over shrinkage-covariance. Prior probability of a clean win: ~25%.
- Most likely failure mode: shrinkage-correlation matches the subspace at all N (regularization suffices). If so,
  Bet 1 becomes another decisive diagnostic row ("even at N<<B, shrinkage-covariance subsumes the subspace").
- Decision: E1 is the gate. If the subspace shows a real small-N advantage over shrinkage, push E2-E5; else,
  document and pivot to Bet 2.
