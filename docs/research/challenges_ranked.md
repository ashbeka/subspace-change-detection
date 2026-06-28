# Ranked Research Challenges: Hyperspectral Temporal Change

Created 2026-06-21 from the supplied papers, project notes/experiments, the 2019 HSI-CD reviews, the 2025 *Information Fusion* survey boundary, and closest-method searches. Ranking favors a real remote-sensing need, a structural fit to lab methods, available falsifiers, and a study feasible without inventing unavailable labels.

## Ranking Criteria

Scores are 1-5; higher is better except risk. “Open” means the *specific formulation* appears open, not that the broad topic has no prior work.

| Criterion | Weight | Question |
|---|---:|---|
| Problem importance | 25% | Does this matter for real HSI CD rather than for protecting DS? |
| Structural method fit | 25% | Does geometry capture information or invariance a simple null lacks? |
| Novelty headroom | 20% | Is the exact task/output absent from closest methods? |
| Falsifiability | 15% | Can a small decisive test kill the idea? |
| Feasibility | 15% | Can public/synthetic-grounded data and existing tools test it? |

## Ranked Table

| Rank | Challenge | Importance | Fit | Headroom | Falsifiable | Feasible | Risk | Weighted view |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Explain subtle local HSI distribution change after separating mean, dispersion, and orientation | 5 | 5 | 3 | 5 | 4 | 4 | Best surviving task; covariance/unmixing may subsume it. |
| 2 | Attribute change to stable, contiguous wavelength intervals | 5 | 4 | 3 | 5 | 4 | 4 | Strong interpretability need; sparse PCA/band selection are close. |
| 3 | Distinguish true spectral-structure change from registration and missing-sample artifacts | 5 | 3 | 3 | 5 | 4 | 5 | Essential validity gate; geometry is currently fragile. |
| 4 | Evaluate subtle/subpixel change with scarce or mechanism-poor labels | 5 | 3 | 3 | 3 | 2 | 5 | Critical but primarily a data/protocol contribution. |
| 5 | Handle endmember/spectral variability without confusing it with material change | 5 | 3 | 2 | 4 | 3 | 5 | Unmixing and ACD are mature; geometry has thin room. |
| 6 | Preserve spatial support without erasing within-window arrangement | 4 | 3 | 2 | 4 | 4 | 4 | Patch tensor/SiROC already occupy the broad claim. |
| 7 | Phase-invariant dense temporal HSI change via RTW-like representations | 4 | 5 | 4 | 5 | 1 | 5 | High novelty, blocked by dense labeled HSI and TWDTW pressure. |
| 8 | Reduce hundreds of bands/computation without losing subtle change | 4 | 3 | 1 | 4 | 5 | 2 | PCA, sketching, sparse methods, and ECDBS are mature. |
| 9 | Transfer interpretable HSI evidence to damage progression/recovery | 5 | 2 | 4 | 2 | 1 | 5 | Deep motive, not currently supported by suitable HSI labels. |

## 1. Factorized Local Spectral-Distribution Change

**Problem.** Binary HSI CD usually emphasizes a mean spectral displacement or a learned feature difference. A local area can instead change in heterogeneity/correlation structure while its mean remains similar. The project observed this controlled regime, but ordinary covariance detected it too.

**Why it matters.** Subpixel mixing, within-class spectral variability, crop/stress heterogeneity, and partially transformed land-cover patches can change local distributions before a strong mean shift. A factorized output can distinguish:

```text
mean displacement | total/relative dispersion change | covariance orientation change
```

**Why geometry may fit.** The top-r centered covariance eigenspace is a Grassmann point. Centering removes mean; orientation is invariant to positive isotropic covariance scale; DS/projector leverage can expose directions. This is a legitimate structural lever only for the orientation component.

**Closest pressure.** Covariance equalization, quadratic ACD, shrinkage covariance/SPD distances, MMD/energy tests, Wu's HSI SCD, SMSL, and unmixing already detect related changes.

**Decision gate.** Go only if orientation-only changes with matched mean/eigenvalues are detected and explained more specifically or stably than full covariance/SPD and nonparametric tests.

**Failure consequence.** If covariance/SPD matches every output, the research should report the diagnostic equivalence and remove DS from the method claim.

## 2. Contiguous Wavelength Attribution

**Problem.** A binary map does not say whether change is driven by red-edge, water/SWIR, mineral, pigment, or sensor-artifact regions. Hundreds of correlated bands make scattered importances hard to interpret.

**Why it matters.** Wavelength attribution is closer to physical explanation and can help reject atmospheric/noisy bands, compare sites, and audit a change score.

**Why geometry may fit.** The diagonal of a DS/projector is a basis-invariant wavelength leverage score. S3CCA supplies a principled smooth overlapping-group prior over the ordered spectral axis.

**Closest pressure.** Sparse PCA HSI-CD already identifies wavelength regions; unsupervised band selection and ECDBS select useful bands; HCV retains band patterns; unmixing gives material-specific directions.

**Decision gate.** Go only if interval attribution recovers known affected wavelengths more accurately and stably than per-band standardized differences, sparse PCA, group-sparse covariance difference, and unmixing residuals.

## 3. Registration And Missing-Sample Discrimination

**Problem.** The 2019 practical review calls subpixel registration a persistent issue. Project interventions show one-pixel translation and missing composites produce large geometric false alarms.

**Why it matters.** An elegant orientation score is unusable if spatial mixing or invalid samples rotate its eigenspace more than true change.

**Fit.** Geometry does not inherently solve registration. Its possible contribution is an eigengap/bootstrap uncertainty and multiscale consistency test, not a claim of robustness.

**Closest pressure.** SiROC, covariance equalization, context-sensitive registration-noise mitigation, local prediction, and explicit alignment.

**Decision gate.** Orientation false-alarm distributions under one-pixel shifts, blur, missing pixels/bands, and PSF mismatch must be below or calibratable against true orientation-only change.

## 4. Evaluation Under Label Scarcity

**Problem.** Public HSI-CD datasets are usually small, bitemporal, and labeled only as changed/unchanged. They do not label “mean,” “dispersion,” “orientation,” or affected wavelength intervals.

**Why it matters.** Binary accuracy cannot validate a mechanism explanation.

**Best protocol.** Combine:

1. exact moment-matched synthetic regimes with known wavelength support;
2. semi-synthetic interventions anchored in real HSI spectra/endmembers;
3. public real HSI-CD binary labels;
4. independent physical/endmember inspection for a small number of real transitions.

**Risk.** Synthetic-only success is not a remote-sensing result. Weak physical labels can become circular if derived from the same spectra.

**Decision gate.** The contribution requires at least one independently supported real transition and clear separation of mechanism-validation metrics from binary-map metrics.

## 5. Spectral Variability Versus Material Change

**Problem.** Illumination, atmosphere, BRDF, phenology, and within-class variability can alter spectra without a material transition.

**Why it matters.** This is the central false-alarm pressure in HSI CD.

**Fit.** Mean/scale/orientation factorization can describe which second-order component moved, but it does not identify a material by itself.

**Closest pressure.** IR-MAD/SFA/ACD model prevalent relations; coupled and multitemporal unmixing explicitly model spectral variability and abundance change.

**Decision gate.** Compare orientation evidence to abundance/endmember changes under controlled variability. If unmixing is more stable and more physical, use it as the primary explanation.

## 6. Spatial Structure Preservation

**Problem.** A local set of spectra gives spatial support but is permutation-invariant within the window. It preserves local distribution, not arrangement.

**Why it matters.** Boundaries, thin objects, and damage morphology are spatial phenomena.

**Closest pressure.** Patch tensor CD, SiROC, spectral-spatial networks, multiscale local methods, and Green Learning-style representations.

**Decision gate.** Compare shuffled-window, translated-window, patch-tensor, and raw local-statistic ablations. Do not call the method spatially aware if shuffling leaves it unchanged.

## 7. Phase-Invariant Dense Temporal HSI Change

**Problem.** Similar seasonal cycles occur at shifted times; a true change can alter cycle shape rather than phase.

**Why geometry may fit.** RTW builds a subspace from time-warped ordered tuples, supplying a phase/warp invariance not present in plain DS.

**Closest pressure.** TWDTW, phase-aware harmonic models, BFAST/CCDC, SFA/SSA. Project trajectory dynamics and SSA-DS natural-event detection already failed.

**Decision gate.** In a controlled pair, RTW must ignore pure phase shift and detect same-phase shape change when TWDTW/harmonic controls do not. It also needs a dense HSI series; multispectral success would not prove HSI-CD novelty.

## 8. Spectral Compression And Computation

**Problem.** Hundreds of bands and local covariances are expensive and statistically underdetermined.

**Existing solutions.** PCA/MNF, band selection, sparse PCA, random sketches, low-rank/tensor factorizations, small-Gram computations, shrinkage.

**Decision gate.** Runtime/memory and score stability across rank/band count must be reported. Efficiency is secondary unless it is substantially better at matched accuracy/attribution.

## 9. Damage Progression And Recovery

**Problem.** The motivating use case requires multi-date damage/recovery labels and often higher spatial resolution than satellite HSI supplies.

**Closest pressure.** PWTT with SAR stacks, CCDC/BFAST, high-resolution optical deep CD, and building-footprint methods.

**Decision gate.** Keep as motivation/future validation until a dataset connects HSI temporal observations to independently verified damage or recovery states.

## Rejected Challenges As Primary Contributions

- Better first-order DS detection: SAM-equivalent at rank 1 and empirically weaker at higher ranks.
- Generic invariant subspace detection: IR-MAD/SFA/ACD occupy the cell.
- Subspace trajectory curvature: vector second differences match it; second-order DS did not provide distinct evidence.
- Material-subspace membership: centroid SAM beat MSM/GDS in the controlled Salinas study.
- First spatial/tensor HSI subspace: prior spatial-spectral, patch-tensor, and subspace HSI-CD methods exist.

## Honest Novelty Verdict

The highest-fit open challenge is not “detect distribution change with a subspace.” It is to *separate and explain one specific distribution mechanism*: local covariance-orientation change after mean and scale quotienting, with contiguous wavelength attribution. The problem is important, but novelty is only moderate because covariance tests, unmixing, and sparse band attribution are close.

## Next Falsifiable Step

Write the four-regime moment-matched benchmark and its attribution ground truth, then verify on paper that every metric can distinguish mechanism accuracy from ordinary binary detection before implementing the method.
