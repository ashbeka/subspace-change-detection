# Synthesis: Specific Research Tasks And Top-Task Plan

Created 2026-06-21. This synthesis obeys the project's hard findings: rank-1 DS is SAM; plain subspace detectors lose to simple/classical baselines; invariance is occupied by IR-MAD/SFA/ACD; trajectory dynamics is largely closed; and mean-preserving change is not uniquely geometric because covariance detects it.

## Convergence In One Sentence

Pursue **local, moment-factorized hyperspectral change characterization**: separate mean shift, total dispersion, normalized eigenspectrum, and scale-quotiented eigenspace orientation in bitemporal satellite HSI, then test whether the orientation component supports stable contiguous wavelength attribution beyond covariance/SPD, two-sample, unmixing, and per-band/sparse controls.

This is a change-*mechanism and explanation* task. It is not another promise that DS will improve binary detection.

## Ranked Specific Tasks

| Rank | Specific task | Structural lever | Novelty status | Feasibility | Main falsifier |
|---:|---|---|---|---|---|
| 1 | Mean/dispersion/eigenspectrum/orientation decomposition with contiguous wavelength attribution for local bitemporal satellite HSI change | Captures second-order orientation after quotienting mean and isotropic scale; basis-level explanation | Provisional combination/new output | High for formula/semi-synthetic; medium for satellite labels | Full covariance/SPD + simple attribution matches it |
| 2 | A rigorous diagnostic benchmark of when subspace geometry adds information over sufficient scalar/statistical nulls | Equivalence and invariance map | Strong negative-study framing, not a new detector | High | No coherent theorem or positive output beyond a comparison table |
| 3 | RTW warp-invariant detection of seasonal-cycle *shape* change independent of phase | Genuine temporal-warp invariance | High application novelty | Low for HSI; medium for Sentinel-2 | TWDTW/phase-aware harmonic matches it |
| 4 | Mode-specific product-Grassmann attribution for spectral versus spatial HSI patch change | Preserves tensor modes | Combination, crowded | Medium | Tucker/patch-tensor residual provides same mode output |
| 5 | Structured contiguous wavelength attribution for conventional HSI change maps | Ordered spectral support | Thin; sparse PCA/band selection close | High | Per-band fused lasso/sparse PCA matches it |
| 6 | DS/Grassmann head over frozen remote-sensing foundation features | Rich learned set plus label-free geometry | Modern combination | Medium/high engineering | Same encoder cosine/L2 explains all performance |

Task 2 is the fallback spine of Task 1: if geometry fails, the equivalence/failure result remains useful. Task 3 has a cleaner invariance story but does not satisfy the immediate satellite-HSI data constraint.

## Top Task: Precise Problem Statement

Given two co-registered satellite hyperspectral images of the same scene, identify local changes and explain *which statistical mechanism and wavelength intervals changed*:

1. local mean spectrum;
2. total spectral dispersion;
3. relative eigenspectrum/covariance shape;
4. orientation of the dominant centered spectral-variation subspace.

The central research question is:

```text
After mean and isotropic scale are removed, does local spectral-distribution
orientation change provide stable, physically useful wavelength-level evidence
that is not already supplied by full covariance/SPD tests, nonparametric
two-sample tests, spectral unmixing, or simple bandwise attribution?
```

The positive claim is conditional. If the answer is no, the result closes the final mean-preserving subspace hypothesis cleanly.

## Why This Task, Not The Rejected Alternatives

### What survives

- HSI has enough bands for nontrivial multidimensional variation.
- Centered local sets can represent a distribution whose mean is deliberately excluded.
- Orientation is invariant to positive isotropic covariance scale, fixing the scale-confound in the earlier uncentered test.
- DS supplies a basis/projector whose row leverage can be attributed to wavelengths.
- S3CCA motivates smooth contiguous support on an ordered spectral axis.

### What is not revived

- No first-order rank-1 DS claim.
- No second-order/geodesic trajectory claim.
- No generic illumination-invariance claim.
- No material-subspace membership claim.
- No first spatial/tensor/subspace HSI-CD claim.
- No assumption that a binary AUC increase is the contribution.

## Provisional Novelty Claim

The exact predecessor search found:

- HSI target/background subspace CD;
- covariance equalization and quadratic covariance ACD;
- multiview subspace learning for anomalous HSI CD;
- coupled/multitemporal unmixing with material-specific change output;
- sparse PCA and learned band selection for HSI CD;
- covariance/Wishart change testing in PolSAR;
- tensor and deep spectral-spatial HSI CD.

It did **not** find the exact combination of:

```text
local symmetric pre/post HSI distributions
+ explicit mean/total-scale/normalized-spectrum/orientation factorization
+ scale-quotiented Grassmann/DS orientation
+ basis-invariant contiguous wavelength attribution
+ direct covariance/SPD, MMD, unmixing, and sparse-band falsifiers.
```

Therefore the safe wording is: **“We investigate a moment-factorized and wavelength-attributed local HSI change task.”** “First” is not authorized until an IEEE/Scopus/Web of Science full-text search and advisor confirmation close the remaining literature gap.

## Method And Math

### 1. Local sample construction

For a spatial support `R` with `n` valid pixels and `B` wavelengths at date `t`:

```text
X_t = [x_(t,1), ..., x_(t,n)] in R^(B x n).
```

Use the same registered support and validity mask at both dates. Windows/superpixels must not cross invalid pixels. The primary method is permutation-invariant inside `R`; it preserves local spectral distribution but **not spatial arrangement**.

### 2. Mean and centered covariance

```text
mu_t = X_t 1 / n
Z_t = X_t - mu_t 1^T
C_t = Z_t Z_t^T/(n-1) + epsilon I.
```

For `n << B`, use shrinkage or a dual SVD rather than a numerically singular empirical covariance. `epsilon`, shrinkage, sample count, and masks are reported.

### 3. Factorization

Let `C_t = U_t diag(lambda_t) U_t^T`. Define:

```text
s_mean  = ||Sigma_pool^(-1/2) (mu_2 - mu_1)||
s_scale = |log trace(C_2) - log trace(C_1)|
p_t     = lambda_t / sum(lambda_t)
s_shape = ||log(p_2 + delta) - log(p_1 + delta)||_2
```

Choose a top-r orientation block `U_tr` by a pre-registered energy/eigengap rule. Its principal angles are `theta_i` from `svd(U_1r^T U_2r)`:

```text
s_orient_grass = sqrt(sum_i sin(theta_i)^2)
s_orient_DS    = 2 sum_i (1 - cos(theta_i)).
```

These are related but not identical Grassmann/DS metrics. Both must be compared to the projector Frobenius score and full SPD distances; metric choice is an ablation, not novelty.

Trace normalization does not change eigenvectors, so orientation is invariant to `C_2 = alpha C_1`, `alpha > 0`. Near-repeated eigenvalues make individual directions unidentifiable; only the full degenerate block may be compared. Report eigengaps and bootstrap uncertainty.

### 4. Difference-direction attribution

Construct the first-order DS `D` between `span(U_1r)` and `span(U_2r)` using the verified canonical-pair/projector formulation. Use the projector

```text
P_D = D D^T,
a_b = (P_D)_(b,b), b=1,...,B.
```

`a_b` is invariant to rotations of the chosen DS basis. It attributes the *orientation discrepancy*, not mean or scale change.

Turn noisy leverage values into contiguous wavelength intervals with a standard fused/structured penalty, for example:

```text
z* = argmin_(z >= 0) 0.5 ||z-a||_2^2
     + lambda_1 ||z||_1
     + lambda_TV sum_b |z_(b+1)-z_b|.
```

Use wavelength spacing rather than raw band index when gaps are irregular, and exclude known invalid/absorption bands before fitting. This structured postprocessing is inspired by S3CCA's smooth overlapping support but is not S3CCA and must not be named as such.

### 5. Decision output

Do not hide the components in one tuned score initially. Produce four component maps, uncertainty/eigengap maps, and interval attributions. A combined binary score is secondary and may use a fixed norm or validation-only calibration. Mechanism labels are assigned only in controlled data where ground truth exists.

## Construction Card

| Field | Definition |
|---|---|
| Variant | centered, trace-quotiented local spectral-distribution orientation with DS leverage attribution |
| Source | covariance eigenspaces + Fukui/Maki DS + structured spectral support inspired by S3CCA |
| Sample unit | one valid pixel spectrum within the same local support/date |
| Matrix | `X_t in R^(B x n)`, then centered `Z_t` |
| Subspace count | one per date per local support/scale |
| Basis | top-r left singular/eigenvectors with shrinkage and eigengap uncertainty |
| Comparison | principal-angle/Grassmann and DS metrics; full covariance/SPD controls |
| Preserved | wavelength identity, local distribution, dominant correlation orientation |
| Lost | mean in orientation branch, isotropic scale, within-window spatial arrangement, eigenvector sign |
| Outputs | mean, scale, shape, orientation, uncertainty, wavelength intervals |
| Verification | moment-matched formula tests before real data |

## Fair Comparison Baselines

### Traditional and invariant

- raw L2/CVA and per-band standardized difference;
- SAM;
- PCA-diff/PCA-Kmeans;
- IR-MAD;
- SFA/ISFA;
- Wu 2013 HSI SCD.

### Distributional and covariance

- per-band variance change and covariance Frobenius;
- Ledoit-Wolf/other shrinkage covariance difference;
- log-Euclidean and affine-invariant SPD distances;
- generalized eigenvalue/covariance-equalization ACD;
- Box-style/permutation covariance test where assumptions permit;
- MMD and energy distance.

### Interpretation and subpixel

- sparse PCA loadings on the difference image;
- fused-lasso per-band effect sizes;
- group-sparse covariance-difference attribution;
- HCV/band-selection output;
- coupled or sparse multitemporal unmixing and abundance/endmember changes.

### Spatial and learned

- raw local statistics at identical support;
- SiROC or an established contextual residual;
- patch-tensor HSI CD;
- one competitive supervised/self-supervised HSI-CD model when labels permit.

No weak baseline may stand in for a stronger member of the same family.

## Research-Only Experiment Plan

No experiment was run in this mining task. The following plan is pre-implementation.

### Gate 0: Algebra and implementation truth

1. Equal subspaces produce zero orientation/DS scores.
2. Basis rotations within a subspace leave score and `diag(P_D)` unchanged.
3. `C_2 = alpha C_1` changes scale only; orientation remains zero.
4. Equal eigenvalues expose non-identifiability; uncertainty must rise rather than produce a confident direction.
5. Verify DS canonical-pair, sum-projector, and projector-distance relations on toy matrices.

### Gate 1: Exact moment-matched regimes

Generate at least five regimes with matched sample count and repeated seeds:

| Regime | Mean | Trace | Normalized eigenvalues | Orientation | Expected active output |
|---|---|---|---|---|---|
| Null/resampling | same | same | same | same | none |
| Mean-only | changed | same | same | same | mean |
| Scale-only | same | changed | same | same | scale |
| Shape-only | same | same | changed | same | shape |
| Orientation-only | same | same | same | changed | orientation |

For attribution, constrain the orientation change to a known contiguous wavelength block. The synthetic covariance must remain positive definite. This is a formula test, not remote-sensing evidence.

### Gate 2: Real-spectrum semi-synthetic test

Anchor distributions in real HSI spectra/endmembers (for example Salinas only as a spectral source), then impose controlled mean/scale/shape/orientation interventions and spatial mixtures. Add nuisance strata:

- global gain and offset;
- diagonal per-band gain;
- sensor noise and band dropout;
- varying local sample count;
- one-pixel translation, blur, and PSF mismatch;
- class boundary/mixed support.

This gate asks whether the components remain identifiable under realistic covariance and sample scarcity.

### Gate 3: Public bitemporal HSI benchmark

Use at least one established HSI-CD dataset for binary-map pressure, clearly stating whether it is airborne or satellite. Binary labels test localization only, not mechanism truth. Keep train/validation/test regions separate and avoid calibrating thresholds on the test map.

### Gate 4: Satellite HSI evidence

The publication-level satellite claim requires a real bitemporal pair from Hyperion, PRISMA, EnMAP, GF-5/ZY-1, or another documented satellite HSI source with:

- acquisition and preprocessing provenance;
- co-registration/PSF audit;
- independent change evidence;
- sufficient local valid samples;
- a physically interpretable transition or endmember comparison.

Do not treat Salinas or an airborne pair as satellite evidence. Do not acquire a large new corpus until Gates 0-2 pass.

### Gate 5: Generalization

Repeat on a second sensor/site or hold out a land-cover transition. Report rank/window/smoothing sensitivity and whether attributed intervals transfer in wavelength coordinates.

## Metrics

### Mechanism identification

- macro-F1 and confusion matrix across null/mean/scale/shape/orientation;
- one-vs-rest AUROC/AP for each component;
- leakage: response of every inactive component in each matched regime;
- calibration/error bars across seeds and sample sizes.

### Wavelength attribution

- interval IoU, precision, recall, and boundary error;
- rank correlation with known band-effect support;
- bootstrap interval-selection frequency/Jaccard stability;
- physical consistency with independent endmember/absorption evidence on real cases.

### Binary map and robustness

- AP/PR-AUC as primary under imbalance, plus AUROC, F1, IoU, precision/recall;
- false-alarm rate under each nuisance stratum;
- change-to-translation response ratio;
- city/site/transition-stratified results, not only pooled scores;
- runtime and peak memory.

## Pre-Registered Falsifiers

The top task loses its geometric contribution if any of these hold robustly:

1. A shrinkage full-covariance/SPD decomposition matches orientation detection, mechanism classification, and uncertainty.
2. MMD/energy detects orientation-only change equally well and simple per-band/group covariance attribution recovers the same intervals.
3. Sparse PCA or fused per-band effect sizes match interval accuracy/stability.
4. Unmixing gives a more accurate and more physical explanation on real transitions.
5. Eigenspace orientation is unstable under realistic eigengaps/sample counts.
6. One-pixel translation or boundary mixing produces responses indistinguishable from true orientation change.
7. Results exist only on arbitrary Gaussian rotations and fail on real-spectrum semi-synthetic or satellite evidence.

If detection survives but attribution does not, report a covariance change detector, not an interpretable geometric method. If attribution survives but binary detection is weaker, a characterization contribution remains possible. If neither survives, close the direction and retain the diagnostic equivalence result.

## What A Publishable Result Would Look Like

### Positive outcome

- exact decomposition verified;
- orientation-only regime isolated with low leakage;
- contiguous wavelength intervals recovered more stably than sparse/per-band/covariance nulls;
- complementary real satellite HSI evidence with honest limits;
- no SOTA claim required.

### Negative but useful outcome

- proof/experiments show DS orientation is completely subsumed by regularized covariance/SPD;
- a clear map of sample-size/eigengap/registration failure boundaries;
- recommendation to use simpler covariance or unmixing methods;
- this becomes Task 2, the diagnostic study.

### Insufficient outcome

- synthetic AUC only;
- no covariance/unmixing baseline;
- one visually attractive map;
- “first use of DS on HSI” framing;
- airborne evidence described as satellite evidence;
- selected wavelength intervals evaluated against the same score that generated them.

## Remaining Literature Gap

General scholarly APIs and DOI indexes found no exact predecessor, but this is not proof. The 2025 survey full text and subscription full-text search were not available in this session. Before “first” wording, search IEEE Xplore, Scopus, and Web of Science full text for combinations of:

```text
hyperspectral change detection + Grassmann
hyperspectral change detection + eigenspace orientation
hyperspectral covariance change + spectral attribution
local covariance change + wavelength interval
subspace angle change + band selection/attribution
```

Record zero-result queries and read every plausible hit.

## Honest Novelty Verdict

The top task is specific, falsifiable, and provisionally novel as a *factorized characterization and attribution output*. It is not novel as DS, covariance change, subspace HSI CD, subpixel CD, or band selection. Its publication value depends on orientation and wavelength attribution surviving much stronger nulls than the project used in the original mean-preserving experiment.

## Next Falsifiable Step

Without running a long experiment, first write the algebra/formula test for the five moment-matched regimes and the equivalence table relating DS, principal angles, projector distance, and SPD/covariance decompositions. That document decides whether a distinct geometric output exists before any dataset or model investment.
