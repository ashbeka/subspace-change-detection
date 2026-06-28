# Band ATTRIBUTION (Codex Ch2) — the last stone: REDUNDANT/WORSE (2026-06-21)

Script: `temporal/experiments/attribution_test.py` · `temporal/outputs/attribution_test/`
Validatable band-injection test: inject a change in a KNOWN contiguous interval into real Hermiston spectra,
add noise + a broadband illumination distractor; measure which method recovers the interval (Average Precision).

## Result (AP; random baseline 0.099; 40 trials)
| condition | perband_meandiff | perband_stddiff | perband_standardized | DS_basis | DS_smooth |
|---|---|---|---|---|---|
| clean | 1.000 | 1.000 | 1.000 | 0.242 | 0.257 |
| noise | 0.757 | 0.471 | **1.000** | 0.226 | 0.250 |
| noise+distractor | **0.600** | 0.324 | 0.214 | 0.177 | 0.186 |

## Verdict — DECISIVELY REDUNDANT/WORSE
The DS-basis attribution is NEAR-RANDOM (0.18–0.26) in every condition; per-band differencing recovers the
injected interval at AP 0.6–1.0. The difference-subspace directions spread across correlated bands rather than
localizing to the changed ones, so geometry is WORSE than the trivial null even for the interpretable-output
task it was supposed to uniquely own. Even under the broadband distractor (which confounds standardized diff),
per-band mean-diff (0.60) crushes DS (0.18). Attribution is closed.

## The positive hunt is COMPLETE and comprehensively NEGATIVE
Every avenue tested, each against the correct null, on real data where possible:
detection (DS≡SAM; CVA/IR-MAD/harmonic/correlation win across amplitude/direction/distributional/temporal/
small-sample regimes) · characterization (T1 redundant with Δ²m) · warp-invariance (owned by DTW/TWDTW) ·
material-subspace MSM/GDS (worse than SAM-to-mean) · orientation-factorization (worse than correlation matrix) ·
small-sample N<<B (subspace worst) · attribution (near-random, worse than per-band diff).
**Subspace geometry is dominated by a simpler standard statistic in EVERY tested cell.** No open positive cell
remains except H-C (geometry on deep/foundation features) — engineering-heavy, and any win would be attributable
to the features, not the geometry. The DIAGNOSTIC is the contribution: a rigorous, real-data-validated,
pre-registered-null map of when/why subspace geometry is redundant for satellite CD — Sensei's "novel first
trial" bar, corroborated by two independent research-minings.
