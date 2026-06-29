# Methods And Implementation Reference

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

Evidence:

- stronger than global pixel DS;
- useful for xBD-S12 candidate localization;
- not a universal detector.

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
```

Evidence:

- works as a curve/descriptor;
- needs labeled sequence validation before detector claim.

Code trail:

- `temporal/subspace.py`
- `temporal/dynamics.py`
- `temporal/experiments/`

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

## Source-To-Code Rule

For every new method, record:

```text
source material -> math object -> satellite adaptation -> code path -> test -> result -> claim
```

If any link is missing, the method is experimental only.

