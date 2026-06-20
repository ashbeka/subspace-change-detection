# THE TWIST (N<<B small-sample set-subspace CD) — also FAILS (2026-06-21)

Script: `temporal/experiments/small_sample_subspace_cd.py` · `temporal/outputs/small_sample_subspace_cd/`
The best-derived remaining shot: every prior loss was at N>>B (covariance beats low-rank subspace); the REAL
HSI regime is N<<B (5x5 patch = 25 px, 242 bands) where covariance is rank-deficient and the set-subspace
should be the well-posed, permutation-invariant (registration-robust), scale-invariant tool. Tested on real
Hermiston under 1-px misregistration + illumination.

## Result — AUC(change vs no-change)
| condition | per_pixel_CVA | patch_mean_CVA | patch_mean_SAM | SET_SUBSPACE | corr_Frob |
|---|---|---|---|---|---|
| clean (sh0,a0) | 0.991 | 0.994 | **0.997** | 0.673 | 0.880 |
| misreg+illum (sh1,a0.1) | 0.982 | 0.989 | **0.994** | 0.652 | 0.858 |

## Verdict — FAILS; the twist is dead too
- **SET_SUBSPACE is the WORST method (0.67)**, even in its supposed home regime. patch_mean_SAM (~the simplest
  thing) wins (0.997) and is fully robust to the nuisance.
- The registration-robustness hypothesis is FALSE here: per-pixel CVA barely drops under a 1-px shift
  (0.991→0.982) — because the change is so large a 1-px misregistration can't rival it.
- Root cause: Hermiston's change is AMPLITUDE-DOMINATED (CVA 0.98); the angle-based subspace DISCARDS amplitude
  → near-useless. The subspace would only matter for direction/structure-dominated change at controlled
  amplitude — but that regime is owned by SAM/IR-MAD/SFA, and the distributional regime by the correlation matrix.

## The complete, exhaustive picture (every regime tested)
| regime | winner | subspace |
|---|---|---|
| amplitude-dominated change | CVA | discards amplitude → worst |
| direction-dominated / illumination | SAM / IR-MAD / SFA | matched/owned |
| distributional (mean-preserving) | correlation matrix | redundant/worse |
| small-sample N<<B | patch-mean SAM | worst |
| temporal / warp | DTW/TWDTW / harmonic | owned |
**Subspace geometry is dominated in EVERY regime by a simpler standard statistic.** There is no open cell for
subspace-as-a-better-detector/comparator. The DIAGNOSTIC is the contribution. The only qualitatively-different
remaining angle is interpretability/ATTRIBUTION (Codex Ch2: contiguous-band naming via S3CCA) — a SECONDARY
contribution where the "win" is a structured output, not an AUC, and hard to validate without per-band GT.
