# Methods Reference

## Quick Links

- [1. Purpose](#purpose)
- [2. Method Status Labels](#method-status-labels)
- [3. Core Geometry Formulas And Caveats](#core-geometry-formulas-and-caveats)
- [4. Subspace / Geometry Cards](#subspace--geometry-cards)
- [5. Classical Baseline Cards](#classical-baseline-cards)
- [6. Neural / Foundation Cards](#neural--foundation-cards)
- [7. Temporal / Structured Method Cards](#temporal--structured-method-cards)
- [8. HSI / Spectral Distribution Cards](#hsi--spectral-distribution-cards)
- [9. Dataset Cards](#dataset-cards)
- [10. Implementation Caveats](#implementation-caveats)
- [11. Source-To-Code Rule](#source-to-code-rule)

## Purpose

This file explains methods only enough to implement, verify, and defend them.
It is not limited to subspace methods. Different method families use different
cards.

## Method Status Labels

| Label | Meaning |
|---|---|
| verified | formula/shape tests and meaningful output exist |
| experimental | runs but claim is not settled |
| reference-only | useful source material, not active code |
| paused | implemented or discussed, but not current positive route |

## Core Geometry Formulas And Caveats

These are the durable geometry facts extracted from old notes, KB files, and
Claude temporal method ledgers. Keep them here once; other docs should refer
to this section instead of repeating the formulas.

### Difference Subspace Magnitude

For two subspaces with principal angles `theta_i`, the DS/geometric magnitude
used in the lab lineage is:

```text
Mag(D(U,V)) = 2 * sum_i (1 - cos(theta_i))
```

This is closely related to standard Grassmann distances. Before claiming a new
DS score is distinct, compare it against:

- Grassmann chordal distance;
- projector Frobenius distance;
- covariance/SPD orientation terms when covariance features are used.

### Rank-1 Warning

Rank-1 spectral DS can collapse into a monotone version of spectral-angle-like
behavior. If a method uses only one dominant spectral direction, compare it
against SAM/cosine angle before claiming a subspace contribution.

### GDS / gFDA Boundary

GDS is meaningful when there are real multiple subspaces/classes/regimes. Do
not create circular pseudo-classes merely so GDS can be applied. If the
classes are generated from the same score being evaluated, the experiment is
not a clean GDS test.

### Second-Order DS

For a trajectory of subspaces `(S1, S2, S3)`, the second-order DS compares the
middle subspace with the geodesic midpoint between its neighbors:

```text
D2(S1,S2,S3) = D(S2, M(S1,S3))
```

where `M(S1,S3)` is the midpoint, usually estimated by a Karcher or
principal-geodesic midpoint. A low second-order value can mean smooth motion
or no acceleration; it is not automatically a bug.

### Spatial Contribution Score

For two bases or projected directions, a per-coordinate contribution diagnostic
can be written as:

```text
c_j = sum_i (u_i[j] - v_i[j])^2
```

Use it only as an attribution/diagnostic, not as proof of causal changed
pixels.

## Subspace / Geometry Cards

### Global Pixel DS

Status: paused.

Purpose: compare pre/post spectral distributions.

Construction:

```text
one sample = one valid pixel spectrum
X_pre, X_post in R^(13 x N_pixels)
PCA basis per date: U_pre, U_post in R^(13 x r)
DS score from projection of pixel difference onto the difference subspace
```

Preserves:

- spectral band relationship at each pixel.

Loses:

- pixel location during subspace fitting.

Evidence:

- weak on OSCD compared with raw L2/PCA-diff.
- supports Sensei's criticism that spatial information is broken.
- early canonical/global DS evidence was weak: canonical DS AUROC `0.6246`,
  pixel spectral difference AUROC `0.7559`, PCA-diff AUROC `0.8134`.

Code trail:

- `phase1/ds/`
- `phase1/subspace/`
- `phase1/scripts/compare_oscd_spatial_subspaces.py`

### Patch / Local-Window DS

Status: experimental, not current best.

Purpose: preserve local spatial context by changing the sample unit.

Construction variants:

```text
patch DS: one sample = flattened local patch across bands
window DS: one subspace per local image window
```

Evidence:

- patch5 improved over global pixel DS on Beirut;
- multi-city results did not beat strong PCA-diff controls.

### Band-Image DS

Status: experimental, useful boundary result.

Purpose: test Jang-style channel flattening.

Construction:

```text
one sample = one full Sentinel-2 band image flattened to a spatial vector
X_pre, X_post in R^(N_pixels x B)
B = 13 for OSCD, 12 for xBD-S12 Sentinel-2
```

Preserves:

- spatial layout inside each band vector.

Risk:

- only 13 band samples, so rank and robustness are constrained.
- centered rank is at most `B-1`; for OSCD `B=13`, so rank `<=12`.
- ambient dimension is the valid-pixel grid; the score is spatial because each
  basis vector lives over the image grid.

Evidence:

- stronger than global pixel DS;
- useful for xBD-S12 candidate localization;
- not a universal detector.

Matched-control formulas:

```text
spatial Gram: normalized U_pre^T U_post style band/feature relation
projector distance: row norm of (U_pre U_pre^T - U_post U_post^T)
cross reconstruction: reconstruction error of one date features by the other date subspace
```

For OSCD Band-Image controls, computations are kept in the small
band/feature-space products where possible, rather than explicitly materializing
huge pixel-by-pixel projectors.

### Successive Saab-DS

Status: current strongest internal OSCD route.

Purpose: build richer label-free local feature maps before applying DS.

Construction:

```text
pre/post images -> shared 3x3 local Saab transform
DC response = local average-like component
AC responses = strongest local PCA-like pattern components
2x2 pooling -> second hop -> response maps
one response map = one band-image sample
canonical DS at each hop -> scaled/fused score map
```

Important boundary:

- this is PixelHop/Green-Learning-inspired successive Saab feature extraction;
- it is not the full PixelHop supervised pipeline.

Evidence:

- OSCD test AP `0.3420`;
- train-fitted filters AP `0.3381`;
- xBD-S12 transfer did not pass as a successive-feature detector.

Source trail:

- PixelHop / successive subspace learning;
- Green Learning overview;
- Fukui/Maki DS.

### Temporal First/Second DS And Geodesic Decomposition

Status: implemented characterization.

Purpose: answer Sensei's multi-date subspace request.

Construction:

```text
one date or date window -> one satellite subspace
first DS magnitude -> change velocity between adjacent subspaces
second DS magnitude -> abruptness/curvature over triples
geodesic along component -> smooth drift
orthogonal component -> off-geodesic abrupt behavior
d_pre(t) -> recovery/distance-to-pre-event diagnostic, not signed recovery proof
```

Evidence:

- works as a curve/descriptor;
- needs labeled sequence validation before detector claim.

Code trail:

- `temporal/subspace.py`
- `temporal/dynamics.py`
- `temporal/experiments/`

Provenance/correction:

- This implements first/second DS and geodesic-style descriptors.
- It does not implement Kanai's learned normal-reference `D_N` detector unless
  that construction is added explicitly.

## Classical Baseline Cards

### Raw L2 / CVA

Purpose: direct spectral change magnitude.

Why it matters:

- simplest baseline;
- often strong;
- if DS cannot beat or complement it, claims must narrow.

### PCA-Diff / Smoothed PCA

Purpose: low-rank difference or spatially smoothed classical pressure baseline.

Why it matters:

- repeatedly strong on OSCD;
- must be included in any standalone change-map comparison.

### Celik PCA-Kmeans

Purpose: established unsupervised remote-sensing change baseline.

Use:

- baseline pressure, not novelty.

### IR-MAD

Purpose: correlation/CCA-style multivariate alteration detection.

Why it matters:

- strong classical multispectral/hyperspectral baseline;
- close in spirit to finding stable/change directions between pre/post views.

## Neural / Foundation Cards

### U-Net / Siamese-Style Segmentation

Purpose: supervised segmentation pressure and optional prior-channel test.

Current role:

- not the main proof yet;
- rerun DS-prior fusion before using it as a claim.

Card fields for neural methods:

```text
input channels/features
training split
label budget
baseline without geometry
matched non-DS control
metric and seed count
```

### Foundation / Deep Feature Geometry

Purpose:

- not to replace foundation models;
- to ask whether subspace geometry explains or structures their embeddings.

Possible future object:

```text
pre/post patch embeddings -> PCA/DS/GDS or clustering in feature space
```

Minimum controls:

- raw embedding distance;
- PCA or cosine distance;
- linear classifier or shallow probe if labels exist.

## Temporal / Structured Method Cards

### SFA / Slow Feature Subspace

Status: reference-only / candidate.

Purpose:

- model slowly varying background or seasonal components;
- detect change as residual outside a learned invariant/slow subspace.
- preserve temporal order information that ordinary PCA frame-subspaces can
  discard.

Core object:

```text
solve a generalized eigenproblem that finds projections with low temporal
variation subject to variance/normalization constraints
```

Variants to know:

- USFA, SSFA, and ISFA-style remote-sensing variants;
- transformed residual scoring rather than raw post/pre subtraction.
- Slow Feature Subspace (SFS): compute several slowest SFA weight vectors,
  build a low-dimensional subspace from those vectors, then compare SFS objects
  with canonical angles/MSM-style methods.

SFS transfer idea:

```text
satellite sequence -> frame/date descriptors X_t
SFA -> slowest weight vectors W
SFS basis = PCA(W) without treating frames as unordered samples
compare regions/events by canonical angles or DS over SFS objects
```

Current boundary:

- useful conceptual pressure for temporal satellite sequences;
- not yet an active positive detector in this repo.
- the source paper is action-recognition, not remote sensing; adaptation must
  beat simple temporal summaries, PCA trajectory subspaces, and raw residuals.

Minimum controls:

- raw residuals;
- seasonal summaries;
- PCA reconstruction;
- SSA;
- documented event labels or independent weak labels.

### SSA / Trajectory Subspace

Status: reference-only / candidate.

Purpose:

- represent time-series windows as trajectory matrices/subspaces.

Project use:

- possible alternative to unordered date PCA for MultiSenGE/Harmonized
  Sentinel-2 sequences.

Minimum controls:

- snapshot PCA;
- global temporal shift;
- dynamic time warping;
- harmonic/Fourier summaries.

### Temporal Pairing And Seasonality Protocol

Status: required caution for sequence experiments.

Purpose:

- avoid treating arbitrary earliest/latest images as clean change evidence;
- separate true event or land-cover change from seasonal snow, vegetation,
  soil moisture, cloud, haze, or illumination effects.

Rules:

```text
do not trust earliest/latest pairing without reporting the date gap
prefer within-season pre/post windows when possible
mask obvious clouds/snow before DS/PCA/IR-MAD if claiming damage relevance
compare temporal DS against simple seasonal and raw residual controls
```

Project implication:

- MultiSenGE earliest/latest visualizations are exploratory;
- a detector claim needs controlled date selection or labels/proxies that
  distinguish pseudo-change from the target change.

### RTW

Status: paused.

Purpose:

- generate time-warped samples so sequences with different speeds can form
  comparable subspaces.

Construction:

```text
sequence -> random monotone time-warped tuples -> hypo-subspace
compare sequences by subspace distance under timing variation
```

Evidence:

- current satellite/crop tests did not beat simpler controls.

Reopen only if:

- a new timing-invariance task is defined and simple shift/RMS/PCA controls are
  beaten.
- the gate separates phase shift from cycle-shape/material change.

### CCA / KCCA / S3CCA / Temporally Regularized CCA

Status: reference-only / candidate.

Purpose:

- compare related views while preserving correlation or smooth structure.

Possible project roles:

- pre/post view matching;
- band-group attribution;
- temporal sequence alignment;
- structured partial-pattern matching.

Gate:

- define which two views are being correlated;
- compare against IR-MAD, PCA, and raw correlation controls.
- treat S3CCA/TRCCA primarily as attribution/structured-view tools unless a
  detector protocol is explicitly defined.

### Venus KDS / KGDS Construction

Status: reference-only / understanding aid.

Purpose:

- preserve what Sensei sent and clarify the original TPAMI image-set setting.

Construction:

```text
raw Venus data: approximately (480, 640, 1, 300)
downsampled view: 63 x 48 = 3024-D vector
KDS: compare two 100-D kernel subspaces
KGDS: compare three 150-D kernel subspaces to form a 300-D KGDS object
```

Boundary:

- this is image-set classification geometry, not a direct satellite change map;
- kernel preimage reconstruction is not part of the current project.

### Tensor / n-Mode GDS / Product Grassmann

Status: future lane.

Purpose:

- preserve multi-mode structure such as bands, space, time, object regions, or
  deep-feature factors.

Gate:

- show that mode-preserving geometry adds information beyond flattening.

### PCA Reconstruction / Subspace Residual Baseline

Status: baseline / diagnostic.

Purpose:

- test whether a post image, patch, object descriptor, or deep feature fits a
  learned "normal" or pre-event subspace;
- provide a simple anomaly/change baseline before DS claims.

Construction:

```text
fit PCA basis U on normal/pre-event features
project x: z = U^T (x - mu)
reconstruct x_hat = mu + U z
score = ||x - x_hat||^2
```

Possible sample units:

- pixel spectra;
- patch descriptors;
- building-level descriptors;
- U-Net/foundation-model encoder features.

Boundary:

- this is a baseline and explanation tool, not DS novelty.

### Sparse Subspace Clustering For Change Types

Status: candidate / baseline pressure.

Purpose:

- test whether changed regions form a union of simpler low-dimensional change
  patterns;
- provide a stronger unsupervised baseline than one global DS score;
- optionally generate pseudo-labels or auxiliary channels for a supervised
  model.

Construction:

```text
sample = pixel, patch, object, or temporal trajectory descriptor
X = [x_1, ..., x_N]
solve sparse self-expression: X approx X C, diag(C)=0
cluster the affinity |C| + |C|^T
interpret clusters as change types or trajectory groups
```

Possible project uses:

- cluster changed patches into no-change, mild spectral shift, severe surface
  change, vegetation-to-urban, urban-to-rubble, or recovery states;
- feed cluster ids or subspace coordinates into U-Net as extra channels;
- compare DS against a principled union-of-subspaces baseline.

Minimum controls:

- k-means on the same descriptors;
- spectral clustering without sparse self-expression;
- U-Net/no-cluster baseline if used as pseudo-labels;
- human/semantic inspection of cluster meaning.

Boundary:

- SSC is not automatically the proposed novelty. It is useful only if the
  clusters are interpretable, externally validated, or improve a downstream
  task without circular labels.

### Diffusion / Pseudotime Progression Features

Status: future route.

Purpose:

- represent disaster damage or recovery as a continuous progression rather
  than only discrete classes;
- borrow the "order samples along a trajectory" idea from diffusion-map
  pseudotime analysis.

Construction:

```text
tile/object descriptors -> k-NN graph
graph diffusion map -> low-dimensional manifold coordinates
pseudotime/order -> intact -> damaged -> recovering progression hypothesis
marker features -> bands/textures/subspace coordinates associated with stages
```

Possible project use:

- stage buildings or regions by recovery trajectory;
- add a progression score as a U-Net channel or object-level feature;
- discover marker bands/textures that explain a damage/recovery stage.

Minimum controls:

- simple severity score;
- PCA/UMAP/t-SNE visualization only as diagnostic;
- clustering with discrete labels;
- human or dataset labels that can validate stage ordering.

Boundary:

- pseudotime is an analogy, not evidence by itself. It needs an interpretable
  endpoint and validation that the ordering corresponds to real damage or
  recovery.

### Neural Prior Integration Variants

Status: candidate after standalone prior evidence.

Purpose:

- preserve feedback asking whether DS/PCA/IR-MAD priors can be used more
  carefully than plain channel concatenation.
- test whether geometry helps under few labels before claiming it helps fully
  supervised deep networks.

Possible variants:

```text
prior as extra input channel
prior as attention/gating map
prior as loss weighting map
prior used only in early curriculum epochs
prior as pseudo-label or teacher signal
```

Minimum controls:

- raw U-Net/Siamese model;
- same architecture with no-DS prior such as PCA/IR-MAD/L2;
- matched cross-reconstruction control;
- multiple seeds and fixed train/test split.

Boundary:

- do not claim DS helps neural segmentation until it beats both no-prior and
  non-DS-prior controls.

### MCDA / Disaster-Resilience Decision Layer

Status: application-level future route.

Purpose:

- turn change maps into planning evidence for reconstruction, evacuation,
  hospitals, roads, schools, or critical infrastructure;
- answer the repeated senpai question: "what do we do with the heatmap?"
- preserve DMaaS, IoT, edge device, and operational-dashboard ideas without
  letting them define the current thesis.

Construction:

```text
damage/change evidence + land-use + roads + terrain + population/context
-> weighted criteria or learned decision score
-> priority map / candidate infrastructure or evacuation support map
```

Possible inputs:

- DS/Saab/IR-MAD/PCA change scores;
- land-use/land-cover maps;
- building or road masks;
- DEM/slope/aspect/topography;
- IoT or UAV observations if a real deployment scenario exists;
- disaster-response constraints.

Boundary:

- this is not the current core method. It becomes viable only after a concrete
  decision task and validation criterion are defined.

### Band-Combination / Spectral Attribution Search

Status: candidate diagnostic.

Purpose:

- identify which Sentinel-2 band groups or indices carry useful change
  evidence;
- avoid treating all 13 bands as equally useful by default.

Possible groups:

- RGB;
- RGB + NIR;
- red-edge;
- NIR/SWIR moisture and burn-sensitive groups;
- vegetation/building/water indices.

Gate:

- run on held-out cities/events;
- report both performance and band-meaning;
- avoid exhaustive overfit unless nested validation is used.

### Sparse Modeling / Dictionary / Feature Selection

Status: future baseline or feature-engineering route.

Purpose:

- preserve the personal-note ideas around LASSO, sparse modeling, dictionary
  learning, and automatic feature selection;
- test whether a small interpretable feature subset explains change better than
  dense all-band/all-feature scores.

Possible uses:

```text
candidate descriptors -> LASSO / sparse logistic model -> selected bands/features
patch/object features -> dictionary atoms -> reconstruction or cluster score
DS/Saab/PCA scores -> sparse fusion model -> interpretable lightweight prior
```

Minimum controls:

- ordinary linear/logistic regression;
- random forest or shallow MLP if labels exist;
- PCA or band-group attribution without sparsity.

Boundary:

- sparse selection is useful only if it improves interpretability, label
  efficiency, or deployment cost. It is not automatically a subspace novelty.

### Object / Building-Level Descriptors

Status: application candidate.

Purpose:

- move from pixel-level maps to object-level change/damage reasoning.

Construction:

```text
one building/greenhouse/object -> one descriptor vector
descriptor = geometry + spectral stats + texture + context + temporal differences
```

Possible features:

- area, perimeter, compactness, elongation, orientation;
- mean/median/std per band or index inside the object mask;
- texture such as GLCM, edge density, local variance;
- neighborhood context such as road/water/building density;
- pre/post differences or concatenated pre/post descriptors.

Use:

- damage classification;
- greenhouse abandonment/state monitoring;
- object retrieval for human review;
- clustering of recovery/change trajectories.

Gate:

- requires object masks, polygons, or reliable object proposals;
- compare against ordinary object descriptors and classical classifiers before
  claiming subspace value.

### Post-Classification / Semantic Change

Status: candidate route, not current OSCD core.

Purpose:

- answer "what changed into what?" instead of only "changed or unchanged."

Construction:

```text
pre image -> class/object map
post image -> class/object map
change = from-class -> to-class or object-state transition
```

Possible bridge to this project:

- use SAM/CLIP/GeoAI/foundation models to propose objects or semantics;
- use DS/GDS/KDS/SSC to explain or cluster changed objects/regions;
- use subspace constraints as a light interpretable head, not as the whole
  semantic model.

Gate:

- needs semantic labels, object proposals, or credible pseudo-labels;
- compare against standard semantic/open-vocabulary change baselines.

### VLM Explanation Subspaces

Status: future-only.

Purpose:

- preserve Sensei's idea that textual explanations for image sets may be
  represented as subspaces, then analyzed by first/second DS and geodesic
  variation.

Candidate construction:

```text
satellite image set -> VLM captions/explanations -> text embeddings
embedding set per date/region -> subspace
first/second DS/geodesic -> explanation trajectory or semantic drift
```

Gate:

- define the VLM, prompt, embedding model, stability checks, and evaluation;
- compare against direct text-embedding distance and raw visual-feature
  distance.

Boundary:

- this is not current computer-vision evidence; it is a future
  foundation/semantic route.

## HSI / Spectral Distribution Cards

### Local Moment / Orientation HSI Geometry

Status: paused candidate.

Purpose:

- test whether many-band HSI data give a more natural subspace setting than
  13-band Sentinel-2 images;
- separate spectral mean, dispersion, and orientation changes.

Candidate construction:

```text
local window spectra -> mu_t
center spectra -> covariance C_t
dispersion features -> eigenvalues / trace / log spectrum
orientation features -> trace-normalized top-r eigenspace
DS/projector score -> orientation change
diag(P_D) -> wavelength leverage / attribution
```

Minimum falsifiers:

- covariance/SPD distances;
- SAM/CVA and IR-MAD;
- SFA/ISFA;
- MMD/energy tests;
- sparse PCA/band selection;
- spectral unmixing;
- at least one competitive deep HSI-CD model if the task becomes a detector
  claim.

Forbidden without evidence:

- first subspace HSI change detector;
- first covariance/distributional HSI change detector;
- superior to IR-MAD/SFA/unmixing/deep HSI baselines;
- DS uniquely detects mean-preserving change.

## Dataset Cards

### OSCD

Task: binary changed-area detection.

Use: current main benchmark.

Limitation:

- two dates only;
- binary labels do not capture all semantic/spectral changes.

### xBD-S12

Task: disaster damage/building labels with Sentinel-1/2.

Use: external score-map/candidate-localization pressure.

Limitation:

- damage classification is different from generic change detection;
- current geometry is candidate localization, not severity classification.

### Harmonized Sentinel-2 / MultiSenGE

Task: multi-date sequence analysis.

Use:

- candidate for temporal first/second DS, GDS, geodesic dynamics.

Limitation:

- labels/evaluation must be defined before detector claims.

## Implementation Caveats

These details are preserved once here so old runbooks and method notes can be
retired later.

| Topic | Durable caveat |
|---|---|
| Legacy residual priors | Old `residual` / `oscd_saved_priors_fast` maps are legacy residual-stack priors, not the repaired canonical DS route. A Beirut audit found old residual-like maps nearly raw-L2-correlated (`0.999990`), while eig/canonical variants had about `0.19037` raw-L2 correlation. |
| OSCD masks | Validity should use `valid_pre AND valid_post`; old notes used `min_valid_bands: 3`. Do not imply full cloud/SCL masking unless implemented. |
| Band statistics | OSCD normalization depends on the saved band-statistics path used by the relevant run. Record the path with each reproducible experiment. |
| Celik / IR-MAD | These are baseline pressure methods. Their prior folders and formulas require audit before strong claims. |
| City evaluation | Stitched city-level evaluation is different from patch-level sampling. Record which unit is used. |
| Augmentation | `RandomGaussianNoise` was raw-only in older phase2 notes. Verify before applying it to prior channels. |
| PriorsFusionUNet | Older branch assumptions are not current truth; reproduce before claiming DS-specific gain. |
| DamageDatasetAdapter | This was not a completed xBD-S12 training/evaluation pipeline. |
| Spatial wording | A method is not spatially faithful merely because it uses a window; it must preserve within-window arrangement or explicitly state what spatial structure is lost. |

## Source-To-Code Rule

For every new method, record:

```text
source material -> math object -> satellite adaptation -> code path -> test -> result -> claim
```

If any link is missing, the method is experimental only.
