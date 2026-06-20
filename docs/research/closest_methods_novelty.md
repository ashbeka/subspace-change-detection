# Closest Methods And Novelty Boundaries

Created 2026-06-21. The purpose is adversarial: for each candidate, identify the nearest prior work, copy its *standard of justification*, and state what would make our proposal redundant.

## Novelty Standard

Use four levels:

| Level | Meaning | Allowed wording |
|---|---|---|
| Established | The exact method/task exists. | Baseline or adaptation only. |
| Combination | Components exist, exact combination not found. | “We investigate the combination...” pending full-text search. |
| New task/output | Operator exists, but evaluated output/question appears new. | “First study of...” only after database verification. |
| New method | Mathematical operator itself is new and non-equivalent. | Requires proof and stronger prior search. |

The top candidate is currently **combination + possible new task/output**, not a new geometry.

## Direction A: Moment-Quotiented Local Spectral-Distribution Change With Wavelength Attribution

### Exact proposed object

For local co-registered HSI spectra `X_t in R^(B x n)`:

```text
mu_t = mean(X_t, columns)
Z_t  = X_t - mu_t 1^T
C_t  = Z_t Z_t^T/(n-1) + epsilon I
Cbar_t = C_t / trace(C_t)
Cbar_t = U_t Lambda_t U_t^T
```

Report three separate components:

```text
s_mean   = robust/whitened distance(mu_1, mu_2)
s_scale  = |log trace(C_2) - log trace(C_1)|
s_shape  = ||log lambda_2 - log lambda_1|| on normalized eigenvalues
s_orient = ||sin Theta(U_1r,U_2r)|| or an equivalent projector/DS score
```

Attribute only the orientation component with the basis-invariant difference-projector leverage `a_b = (P_D)_{bb}` (or an algebraically equivalent measure), then impose fused/overlapping-group structure along wavelength to obtain contiguous intervals.

### Closest 1: Wu et al. 2013 HSI SCD

**What it does.** Date-2 pixel as target; date-1 pixel plus neighborhood/undesired spectra as background span; orthogonal residual is change.

**How it justifies novelty.** High spectral dimension is difficult; a background subspace uses abundant spectral information and accepts spatial/semantic side information. Hyperion/HJ-1A maps and false alarms support it.

**Overlap.** Local spectra, subspace, hyperspectral, bitemporal, unsupervised.

**Difference.** Asymmetric target-background residual versus symmetric pre/post local centered-distribution factorization; no explicit mean/scale/orientation split or contiguous wavelength attribution.

**Killer baseline from it.** Orthogonal-subspace residual using identical neighborhoods.

### Closest 2: Covariance equalization and quadratic ACD

**What it does.** Fits/equalizes first- and second-order cross-date relations and scores quadratic residuals; spatial/local variants exist.

**How it justifies novelty.** Model prevalent changes/nuisance statistically, then identify deviations. Theiler compares quadratic detectors quantitatively, so covariance is a mature method family rather than a weak baseline.

**Overlap.** Means/covariances, HSI, distribution change, unsupervised detection.

**Difference.** The proposed output explicitly decomposes and names orientation, scale, and mean and seeks spectral intervals rather than only an anomaly score.

**Killer baseline.** Shrinkage full covariance with AIRM/log-Euclidean/Bures or generalized-eigen decomposition. If it yields the same component scores and attribution, geometry adds nothing.

### Closest 3: SMSL anomalous HSI CD

**What it does.** Sketched multiview self-representation learns common/specific temporal structure and uses representation-matrix differences.

**How it justifies novelty.** Preserves major information with lower complexity and extracts view-specific change.

**Overlap.** Modern subspace learning, HSI anomalous CD, temporal views.

**Difference.** Learned global/self-representation versus local closed-form Grassmann orientation and physical wavelength output.

**Killer baseline.** SMSL or a reproducible self-representation method on an anomalous-change dataset.

### Closest 4: Coupled and multitemporal unmixing

**What it does.** Estimates endmembers and abundances over time; reports material-specific change direction/intensity and handles spectral variability.

**How it justifies novelty.** Subpixel composition is physically meaningful; temporal coupling improves accuracy/efficiency; theory or auxiliary imagery validates conditions.

**Overlap.** Subpixel/local distribution, subtle change, explanation.

**Difference.** The proposed method is library-free and describes statistical orientation/wavelength intervals, not named abundance changes.

**Killer baseline.** If unmixing is stable and produces the correct material explanation, its semantics are stronger. The geometric method must not relabel an endmember problem as an eigenspace problem.

### Closest 5: Sparse PCA and band-selection HSI CD

**What they do.** Sparse PCA identifies wavelength regions on a difference image; unsupervised band selection and ECDBS select discriminative bands for CD.

**How they justify novelty.** Reduce redundancy/computation and improve discrimination while exposing selected wavelengths.

**Overlap.** Spectral attribution/selection and interpretability.

**Difference.** They primarily explain overall/mean difference or prediction utility. The proposed intervals explain only scale-quotiented covariance orientation.

**Killer baselines.** Per-band standardized mean/variance difference, sparse PCA loadings, fused-lasso covariance-difference energy, ECDBS band importance.

### Closest 6: PolSAR covariance change tests

**What they do.** Wishart/determinant/omnibus/Wilks/Hotelling tests detect changes in complex covariance matrices, including time series and multifrequency/block settings.

**Overlap.** Remote-sensing covariance change and sometimes matrix geometry.

**Difference.** Sensor model, real optical HSI spectra, local wavelength attribution, and moment-factorized output.

**Novelty boundary.** Never claim first remote-sensing covariance-manifold change detector.

### Direction-A verdict

**Defensible claim if the search and experiments hold:** first systematic satellite-HSI study of local change decomposed into mean, dispersion, and scale-quotiented eigenspace orientation, with contiguous wavelength attribution and explicit covariance/unmixing nulls.

**Not defensible:** new DS, new covariance CD, uniquely detects mean-preserving change, first interpretable HSI CD, or expected SOTA detector.

## Direction B: Structured Contiguous-Band Change Attribution Without Orientation Factorization

### Closest methods

- S3CCA: smooth overlapping-group weights for partial pattern matching.
- Sparse PCA HSI-CD: sparse wavelength regions on the difference image.
- Unsupervised band selection HSI-CD: quantitative band-subset evaluation.
- ECDBS: learned band correlation/attention for end-to-end CD.
- HCV/hierarchical spectral analysis: band-pattern multiple change.

### Novelty pressure

Direct S3CCA transfer is invalid because it maximizes commonality, not discrepancy. A new contrastive formulation must state whether weights act on mean difference, covariance difference, DS basis, or learned features. Once this is stated, sparse PCA/group-lasso may already be the simpler operator.

### Fair baselines

Raw per-band effect size, contiguous fused lasso on the difference, sparse PCA, ECDBS band importance, HCV, and unmixing/endmember residuals.

### Verdict

Potentially publishable as an attribution method, but weaker than Direction A because “contiguous change bands” alone is already close to sparse PCA and band selection.

## Direction C: RTW Phase-Invariant Satellite/HSI Change

### Closest methods

- RTW/eGDA: time-warp-tolerant subspaces for action recognition.
- TWDTW/dtwSat and multiannual DTW CD: temporal alignment in satellite image time series.
- BFAST/CCDC and harmonic regression: seasonal trend/break models.
- SFA/SSA: invariant or low-rank temporal representations.

### How closest work justifies itself

RTW argues that randomized ordered tuples preserve temporal variation while tolerating speed; TWDTW uses phenological alignment; BFAST/CCDC explicitly separate season/trend/break.

### Exact possible novelty

First use of an RTW *subspace* as a phase-invariant change representation for satellite series, distinguishing cycle-shape change from cycle timing. This is a new application/task, not a new RTW method.

### Killer nulls

TWDTW distance, phase-aware harmonic residual, phenological metric differences, BFAST/CCDC, and SFA.

### Verdict

Higher application novelty than Direction A, but poor immediate fit to labeled dense HSI. Keep behind a controlled phase gate and do not substitute Sentinel-2 evidence for HSI evidence.

## Direction D: n-Mode/Product-Grassmann HSI Patch Change

### Closest methods

- n-mode GDS and temporal/product-Grassmann tensor classification.
- Patch tensor HSI CD.
- Low-rank/sparse tensor HSI CD and restoration.
- Spectral-spatial CNN/transformer HSI CD.

### Exact possible novelty

Mode-specific Grassmann change attribution: separately report which spatial or spectral factor changed. The novelty is not tensor use, unfolding, or product manifolds.

### Killer nulls

Tucker factor differences, tensor reconstruction residual, patch tensor CD, separable PCA, raw local spectral/spatial scores.

### Verdict

Conceptually aligned but more complex and crowded than Direction A. It also risks poor identifiability: tensor mode rotations and rank choices can change explanations.

## Direction E: SLS/DS On Frozen Remote-Sensing Features

### Closest methods

- SLS product-Grassmann latent subspaces for sound.
- Deep/foundation-feature remote-sensing CD.
- Siamese cosine/L2 and linear-probe heads.
- Subspace/self-attention modules within CD networks.

### Exact possible novelty

An unsupervised, closed-form DS/Grassmann head over frozen remote-sensing foundation features with an attribution/uncertainty output.

### Killer nulls

Same frozen encoder with cosine, Euclidean, mean-pooled difference, Mahalanobis, and a linear probe.

### Verdict

Modern and potentially label-efficient, but the encoder can explain nearly all benefit and wavelength interpretability is lost. Future work.

## Direction F: Diagnostic “When Geometry Adds Information”

### Closest work

Method-comparison/review papers, Theiler's quadratic ACD comparison, HSI-CD surveys, and this project's negative reports.

### Exact possible novelty

A pre-registered equivalence/invariance/structure ladder across MS and HSI that identifies when DS collapses to SAM, when covariance subsumes eigenspace change, and when structured attribution remains distinct.

### Fair baselines

Every row's simplest sufficient statistic: SAM, CVA, IR-MAD/SFA, covariance/SPD, MMD, unmixing, sparse PCA, and same-encoder feature distance.

### Verdict

The most defensible negative/falsification contribution, but a paper needs a coherent theoretical result or a positive task output rather than a list of failed methods. It is the fallback spine for Direction A.

## Baseline Matrix For The Top Task

| Claim component | Minimum baseline | Strong baseline | What a win must mean |
|---|---|---|---|
| Mean change | raw L2/CVA/SAM | IR-MAD/ISFA | Geometry is not expected to win. |
| Scale change | trace/variance ratio | likelihood/Box-style test | Correct component isolation. |
| Orientation change | top eigenvector angle | shrinkage SPD decomposition/full covariance | Better specificity/stability, not just detection. |
| General distribution shift | covariance Frobenius | MMD/energy | Explainability at competitive sensitivity. |
| Subpixel/material change | spectral angle/endmember residual | coupled multitemporal unmixing | Library-free value or complementary mechanism. |
| Wavelength attribution | per-band effect size | sparse PCA/group covariance/ECDBS/unmixing | Correct contiguous support and stability. |
| Spatial robustness | raw local window | SiROC/patch tensor | Lower artifact response at matched true-change sensitivity. |
| Binary HSI CD | PCA-diff/Kmeans | Wu SCD, ACD, deep HSI-CD | Secondary; no SOTA requirement. |

## Honest Novelty Verdict

Direction A has the best balance of specificity, lab fit, and falsifiability, but its novelty margin is narrow. The defensible unit is the *factorized and attributed task/output*, not DS or covariance orientation itself. Directions B-E are either closer to prior art or blocked by data; Direction F is the honest fallback.

## Next Falsifiable Step

Complete a full-text novelty search for Direction A and write an algebra note that maps every proposed score to its covariance/SPD equivalent. If no non-equivalent output remains after that mapping, stop before implementation and publish only the diagnostic result if it is sufficiently coherent.
