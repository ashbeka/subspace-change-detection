# Factorized local distribution change (mean | dispersion | covariance-ORIENTATION) — FIRST POSITIVE mechanism (2026-06-21)

Script: `temporal/experiments/orientation_factorization.py` · `temporal/outputs/orientation_factorization/`
Codex research-mining Challenge 1 (the user's favorite), reframed correctly as a CHARACTERIZATION /
interpretability contribution — NOT a detection-AUC claim. The geometry lever = the top-k eigenspace of the
CENTERED covariance (a Grassmann point), invariant to mean (centering) and to isotropic eigenvalue scale.

## Result — synthetic, 40 seeds (each descriptor's AUC for detecting each change-type vs stable)
| descriptor | mean | scale | orient |
|---|---|---|---|
| mean_shift | 1.00 | 0.81 | 0.55 |
| dispersion \|Δtr Σ\| | 0.58 | 1.00 | 0.63 |
| **ORIENTATION (Grassmann eigenspace angle)** | 0.53 | **0.52** | **0.94** |
| SPD (log-Euclidean) | 0.54 | 1.00 | 1.00 |
| Frobenius \|ΔΣ\| | 0.52 | 1.00 | 1.00 |

**The ORIENTATION descriptor fires ONLY on an orientation change (0.94), ignoring mean (0.53) and scale
(0.52) — scale-invariant by construction — while SPD/Frobenius CONFLATE scale and orientation (fire 1.0 on
both).** The three descriptors {mean_shift, dispersion, ORIENTATION} cleanly factorize the change into its
mean / dispersion / orientation components; the orientation component is the one geometry uniquely supplies.

## Verdict — FIRST POSITIVE (a characterization mechanism), but a LEAD not a result
What's real: geometry provides an interpretable, scale-invariant change component (covariance ORIENTATION)
that the standard covariance/SPD distance cannot disentangle from scale. This is a producible output a scalar
(and a full-covariance distance) structurally cannot give — the open "characterization" lever both research
minings converged on, and Sensei's "novel first/unique" bar (a factorized change descriptor for HSI).

Honest caveats (do NOT oversell):
- The scale-invariance is partly TRUE BY CONSTRUCTION (eigenvectors are unchanged by isotropic scaling); the
  synthetic confirms the MECHANISM, not real-world value. SPD-conflation is also expected.
- The disentangle metric is magnitude-confounded for Frobenius/dispersion; the clean evidence is the detection
  table (ORIENTATION isolates orient; SPD conflates).
- UNVALIDATED on real data: it is unknown whether REAL HSI change has meaningful orientation components, or
  whether the factorization is more USEFUL/specific than reporting SPD. Codex's gate: must beat full
  covariance/SPD + nonparametric tests at explaining orientation-only change MORE specifically/stably.

## Next (to turn the lead into a result)
1. REAL validation on Hermiston: per local patch, factorize the real 2004→2007 change into mean/dispersion/
   orientation; is there a non-trivial orientation component, and does it localize sensibly vs the 5 change types?
2. Contiguous-band ATTRIBUTION (Codex Challenge 2): which contiguous wavelength intervals carry the orientation
   change (DS-basis / S3CCA) — the "which bands" output, validated vs per-band diff / sparse PCA.
3. Robustness gate (Codex Challenge 3): orientation false-alarm under 1-pixel registration shift / missing bands
   — geometry's eigenspace can rotate under these; must be below true orientation change.
