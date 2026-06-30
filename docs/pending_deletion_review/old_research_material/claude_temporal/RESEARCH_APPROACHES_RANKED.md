# Research approaches — ranked, with paper skeletons (seminar prep)  [2026-06-20]

Purpose: answer "what problem statements have we tackled, which is best, and how would each look as a paper."
Deep on the PROBLEM / WHY (the defensible core); brief on the rest. The single best one gets presented to Sensei.

## 1. All problem statements tackled — RANKED
Score = scientific coherence × defensibility × Sensei/senpai fit (Sensei rewards a NOVEL first/unique trial, not
SOTA; his + senpai methods are DS / 2nd-order DS / GDS / MSM / SSA / SLS / RTW / SFA on the Grassmannian).

| # | problem statement | status | rank |
|---|---|---|---|
| A | **DIAGNOSTIC: when/why does subspace geometry help/fail for satellite CD — and when is it just spectral-angle / mean-vector / harmonic-deseasonalization?** | STRONGLY SUPPORTED (full pre-registered ladder) | **1 (most defensible NOW)** |
| B | **Material-signature subspace (MSM/GDS) CD on hyperspectral** — classify/track materials by SUBSPACES (intra-material variation), detect material transitions | UNTESTED (the live positive bet) | **2 (best POSITIVE shot)** |
| C | Mean-preserving distributional change detection via subspaces (local spectral distribution shifts the mean misses) | MODEST, real but non-unique (covariance also catches it) | 3 |
| D | Spectral-subspace for ABRUPT damage (band-subspace change; Gaza motive) | PARTIAL (band-subspace strong on fire; NOT vs SAM/CVA) | 4 |
| E | Faithful signal-subspace DS + learned non-anomalous reference D_N (Kanai) for temporal CD | FALSIFIED on real data (abrupt/gradual/phenology) | 5 (best-built, negative) |
| F | H-A invariance/nuisance-robust CD via invariant-subspace residual (SFA/GDS-common) | detection CLOSED (IR-MAD owns it); attribution-only | 6 |
| G | H-B change-trajectory characterization via 2nd-order DS (velocity/accel/geodesic) | redundant with mean-vector; 2nd-order never fairly tested | 7 |
| H | Hyperspectral spectral-SIGNAL DS (Hankel along wavelength), dimensionality threshold | L0 viable, detection negative (lost to SAM/CVA) | 8 |
| I | Temporal DS on S2 (crude date-window subspace velocity) | failed 3× on real S2 (seasonality) | 9 |
| J | Bi-temporal DS on 13-band S2 / OSCD | DS≡SAM, lost to PCA-diff (the original method-forcing) | 10 |

## 2. The best / most coherent
- **Most DEFENSIBLE right now = A, the diagnostic.** It is rigorously supported, novel as a *first systematic
  characterization*, and squarely fits Sensei's "novel first trial" bar. Weakness: it is a NEGATIVE result; the
  lab culture prefers a positive method.
- **Best shot at a POSITIVE that fits Sensei = B, material-signature subspace (MSM/GDS) CD.** It is the lab's
  home method family (Fukui's MSM/GDS), is genuinely UNTESTED in clean form, and is the one place the geometry
  has a structural reason to win: hyperspectral (200+ bands) makes a material/region subspace genuinely
  MULTI-dimensional, so DS ≠ SAM (we already verified rank>1, DS⊥SAM on 204-band Salinas) — the exact condition
  the low-dim failures lacked. Recommended experimental focus now; A is the guaranteed failsafe.

## 3. Paper skeletons (deep on PROBLEM/WHY; brief elsewhere)

### Paper A — the diagnostic (failsafe)
- **Title:** "When does subspace geometry help change detection in satellite imagery? A diagnostic study."
- **Problem / WHY (the defensible core):** Subspace/Grassmann methods are widely assumed to be a distinct,
  powerful CD family. We show this is mostly an *illusion of generality*: the first-order Difference Subspace
  equals the spectral angle whenever the representation is low-dimensional, and on real satellite data subspace
  detectors are matched or beaten by trivial/standard baselines (spectral angle, mean-vector differences,
  IR-MAD, harmonic deseasonalization) across abrupt, gradual, and phenological change. The contribution is to
  delineate *precisely when geometry adds information and when it is redundant* — a map practitioners lack.
- Methodology: a pre-registered ladder of controlled + real experiments, each with the trivial/standard null;
  the REPRESENTATION × COMPARISON-OPERATOR × DECISION factorization; the construction ledger.
- Results: the full negative tally + the narrow positive conditions (genuine multi-dimensionality; invariance).
- Discussion/Conclusion/Future: geometry as a *mechanism* not a family; insert it only where multi-dim structure
  exists. Future: the positive niche (Paper B).

### Paper B — material-signature subspace CD (the positive bet)  [TO BE EXPERIMENTED NOW]
- **Title:** "Material-subspace change detection for hyperspectral imagery via mutual/generalized difference
  subspaces."
- **Problem / WHY (the defensible core):** Pixel-wise CD compares single spectra (SAM/CVA) and is fooled by
  intra-material variability (illumination, mixtures, sensor condition) — it cannot tell "the material changed"
  from "the same material looks different." A *material* is not a point but a SUBSPACE: the span of its spectral
  signatures captures that intra-material variation. Representing each material as a subspace (MSM/GDS) and
  detecting change as a *transition between material subspaces* (canonical-angle membership change) is invariant
  to intra-material variation by construction, and — because hyperspectral spectra are high-dimensional — the
  material subspace is genuinely multi-dimensional, so the difference subspace carries information SAM cannot.
  This reframes CD from "did this pixel's vector move" to "did this location leave its material subspace,"
  which is interpretable (which material → which) and robust. (Sensei/senpai fit: directly MSM/GDS, his core.)
- Methodology (brief): material-subspace dictionary from class spectra; per-pixel/region MSM membership; change
  = membership/DS shift; vs SAM-to-centroid, CVA, IR-MAD; swept over intra-material/illumination nuisance.
- Results / Discussion / Conclusion / Future: TBD by the experiment; if it beats SAM-centroid under nuisance
  with the subspace (not just a mean), that is the positive. Future: real bitemporal HSI benchmark.

### Paper C — distributional (mean-preserving) change detection  [secondary]
- **Title:** "Detecting distributional change the mean misses: subspace change detection for hyperspectral CD."
- **Problem / WHY:** mainstream CD is mean/per-pixel based and BLIND to changes that alter the local spectral
  *distribution* (composition, heterogeneity, fragmentation) while leaving the mean ≈ unchanged — exactly the
  signature of partial damage / mixed-pixel land-cover change. A subspace captures this distributional structure.
  WHY-caveat: ordinary covariance statistics also catch it, so the contribution must be the subspace's
  scale-invariance / sample-robustness, not mere detection. (Weaker than B; keep as fallback / part of A.)

## 4. Answers to the open questions (= next experiments)
- **(d) Did we exhaust spectral-subspace for abrupt damage (Gaza)? NO.** We only observed the crude band-subspace
  velocity localize a fire cleanly (margin 9.96), but NEVER compared it against SAM/CVA, nor on real damage. It
  is a live, quick thread (relevant to the damage motive). If band-subspace ≈ SAM/CVA → folds into A; if it
  beats them (plausible only if the local band-subspace is multi-dim) → a secondary positive. WORTH RUNNING.
- **(e) Material-signature subspaces — YES, this is Paper B, the live positive bet.** Running it now.
- **(f) Re-implementation fairness — YES, partially.** I already corrected two (2nd-order DS, Kanai's D_N).
  Others were crude/weak adaptations (my IR-MAD had an additive-change bug; the windowed SFA was a poor fit;
  the temporal DS was a crude date-window velocity). A SMART re-implementation — essence-of-the-paper but adapted
  to hyperspectral satellite CD, not a slavish copy — is the right spirit and is exactly what B and (d) are.
  HONEST CAVEAT: re-implementation can rescue the UNDER-TESTED methods (MSM/GDS/spectral-damage), but it will
  NOT overturn the ROBUST negatives (DS≡SAM for low-dim; IR-MAD owns affine-invariant detection) — those hold
  by construction, not by my coding.
