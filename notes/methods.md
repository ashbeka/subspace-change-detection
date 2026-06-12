# Method Notes

## Table Of Contents

- [1. Current Project Scope](#1-current-project-scope)
- [2. Data](#2-data)
- [3. Phase 1: Prior Maps](#3-phase-1-prior-maps)
- [4. Spectral Change Versus Semantic Change](#4-spectral-change-versus-semantic-change)
- [5. Phase 2: Supervised Segmentation](#5-phase-2-supervised-segmentation)
- [6. DS In The Current OSCD Adaptation](#6-ds-in-the-current-oscd-adaptation)
- [7. Corrected DS Implementation](#7-corrected-ds-implementation)
- [8. PCA Rank 6](#8-pca-rank-6)
- [9. DS, GDS, KDS, KGDS](#9-ds-gds-kds-kgds)
- [10. Future Method Hooks](#10-future-method-hooks)
- [11. Venus KDS/KGDS Audit](#11-venus-kdskgds-audit)
- [12. Spatial Information Problem](#12-spatial-information-problem)
  - [Spatial-subspace novelty boundary](#spatial-subspace-novelty-boundary)
  - [Tensor / n-mode GDS idea](#tensor--n-mode-gds-idea)
  - [Multiscale subspace pyramid / Green Learning lead](#multiscale-subspace-pyramid--green-learning-lead)
- [13. Projection Back To Image Space](#13-projection-back-to-image-space)
- [14. Method Caveats](#14-method-caveats)
  - [Subspace construction card](#subspace-construction-card)
  - [Source-linked implementation workflow](#source-linked-implementation-workflow)
  - [Subspace code reading path](#subspace-code-reading-path)
  - [Paper-to-code verification](#paper-to-code-verification)
  - [Prior folder naming](#prior-folder-naming)
  - [Baseline interpretation](#baseline-interpretation)
  - [Phase 1 thresholding vs Phase 2 priors](#phase-1-thresholding-vs-phase-2-priors)
  - [Phase 1 engineering details](#phase-1-engineering-details)
  - [Split and evaluation caveats](#split-and-evaluation-caveats)
  - [Phase 2 training and evaluation details](#phase-2-training-and-evaluation-details)
  - [Loader and model caveats](#loader-and-model-caveats)
  - [MultiSenGE caveat](#multisenge-caveat)
  - [KDS and CCA implementation gaps](#kds-and-cca-implementation-gaps)
  - [Deep-feature subspace gap](#deep-feature-subspace-gap)

## 1. Current Project Scope

Current implemented core:

```text
OSCD Sentinel-2 binary change detection
pre image + post image -> Phase 1 unsupervised priors -> Phase 2 supervised segmentation
```

Here `Phase 1` and `Phase 2` are workflow labels, not immutable theory. In current code they mean:

```text
phase1/ = geometric/classical change-prior generation
phase2/ = neural segmentation or downstream learning using raw bands and optional priors
```

This split can be renamed or reorganized if the research pivots to KDS/GDS/KGDS, clustering, semantic interpretation, or another dataset pipeline.

Current project is not yet:

- disaster damage segmentation;
- xBD/xBD-S12 damage classification;
- multi-class semantic change detection;
- operational disaster response mapping.

## 2. Data

OSCD uses pre/post Sentinel-2 image pairs. Each image has 13 aligned bands:

```text
B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
```

The active loader uses rectified `.tif` band files, not RGB `.png` previews.

For Beirut:

```text
image size = 1180 x 1070
N = 1,262,600 pixel locations
X_pre  shape = 13 x 1,262,600
X_post shape = 13 x 1,262,600
```

In code, `phase1/data/preprocessing.py::vectorize_cube` maps `(C,H,W)` to `(C,N)`.

## 3. Phase 1: Prior Maps

Phase 1 computes unsupervised change-score maps from pre/post imagery.

Implemented prior families:

- DS projection.
- DS cross-residual.
- raw pixel difference / CVA.
- PCA-diff.
- Celik PCA-kmeans.
- IR-MAD.
- geodesic / subspace-distance variants.

Saved prior maps follow this structure:

```text
phase1/outputs/<run>/oscd_change_maps/<split>/<method>/<city>_score.npy
```

Phase 2 can load those maps as extra input channels.

## 4. Spectral Change Versus Semantic Change

DS, PCA-diff, pixel difference, CVA, Celik, and IR-MAD are unsupervised spectral/radiometric change methods.

They answer:

```text
Did the measured Sentinel-2 band values change?
```

OSCD labels answer a narrower semantic question:

```text
Is this pixel part of the labeled changed area according to OSCD annotation policy?
```

Therefore a prior map can highlight vegetation seasonality, snow, haze, thin cloud, soil moisture, illumination shifts, or registration artifacts even when OSCD does not label them as target change. This is not automatically an implementation error, but it can hurt IoU/F1 against OSCD masks.

For supervised segmentation, current Phase 2 uses continuous prior score maps as extra channels. It does not use thresholded DS/PCA masks as labels by default.

## 5. Phase 2: Supervised Segmentation

Phase 2 trains binary OSCD segmentation models.

Raw input:

```text
pre 13 bands + post 13 bands = 26 channels
```

Raw plus priors:

```text
26 raw channels + selected Phase 1 score maps
```

Main model families:

- U-Net.
- Siamese U-Net.
- ResNet-backbone U-Net.
- prior-fusion U-Net.

Main active comparison:

```text
E0_raw_unet vs prior-channel variants
```

## 6. DS In The Current OSCD Adaptation

Current global OSCD DS:

```text
one sample = one valid pixel
sample vector = 13 Sentinel-2 band values at that pixel
pre date -> PCA -> Phi in R^(13 x r)
post date -> PCA -> Psi in R^(13 x r)
DS compares Phi and Psi
```

Current scalar score:

```text
score_i = || D^T (x_post_i - x_pre_i) ||^2
```

Then the score for each pixel is reshaped back to `(H,W)`.

Important limitation:

- Pixel coordinates are not used while fitting the global PCA subspace.
- Position is only restored after scoring.
- This is why spatially aware DS must be tested.

## 7. Corrected DS Implementation

Old unsafe behavior:

- `subspace_variant: residual` built a legacy residual-stack basis.
- With OSCD rank 6 in 13-D, it could produce a `(13,12)` basis.
- On Beirut, its score correlated about `0.99999` with raw spectral L2 difference.
- Treat old `oscd_saved_priors_fast` DS maps as legacy residual-stack priors unless proven otherwise.

Current corrected behavior:

- `legacy_residual_stack` preserves old behavior for reproducibility.
- `residual` remains an alias for old configs.
- `canonical` uses principal vectors and canonical angles.
- `eig` is repaired so equal/shared subspaces return empty/zero DS instead of arbitrary QR fallback.

Important OSCD audit result:

```text
X_pre/X_post shape: (13, 1262600)
Phi/Psi shape:      (13, 6)
legacy D shape:     (13, 12), corr with raw L2 = 0.999990
eig D shape:        (13, 6),  corr with raw L2 = 0.190370
canonical D shape:  (13, 6),  corr with raw L2 = 0.190364
```

## 8. PCA Rank 6

Rank 6 means PCA keeps 6 basis directions in the 13-band feature space.

It does not reduce 1,262,600 pixels to 6 pixels. The pixels remain samples. PCA learns 6 directions that summarize major variation among the 13-band pixel vectors.

This rank is currently a practical hyperparameter, not a proven theoretical optimum.

Action:

- Test ranks `2, 3, 4, 5, 6, 8, 10, 12`.
- Test variance thresholds such as `95%`, `99%`, and `99.5%`.

## 9. DS, GDS, KDS, KGDS

DS:

- Linear Difference Subspace.
- Compares two subspaces.
- Natural for OSCD pre/post if using two dates.

GDS:

- Generalized Difference Subspace.
- Compares more than two subspaces.
- More natural for multi-date data such as MultiSenGE.

KDS:

- Kernel Difference Subspace.
- Nonlinear DS using the kernel trick and KPCA.
- Natural for the TPAMI Venus setting.
- Not active in OSCD yet.

KGDS:

- Kernel Generalized Difference Subspace.
- Nonlinear GDS for three or more subspaces.
- Natural for Venus three-class audit or MultiSenGE multi-date exploration.

Signal Latent Subspace:

- Future bridge between subspace methods and deep models.
- Mahyub et al. 2024 builds subspaces from neural-network latent features rather than raw input vectors.
- Multiple latent-feature subspaces can be fused through a product Grassmann manifold.
- GDS projection is used there to improve between-class separation.
- For this project, the satellite analogue would be: extract features from a remote-sensing CNN/foundation model, build subspaces from patch/tile/date features, then compare raw-spectral DS against latent-feature DS.
- Do not treat this as implemented or as remote-sensing evidence yet.

## 10. Future Method Hooks

These are preserved as method hooks, not active evidence. They come from prior notes, advisor/senpai feedback, reference code, and the final organization source batch.

Multi-date subspace methods:

- First-order DS is the pairwise before/after setting.
- Second-order DS is the natural next subspace idea for progression/recovery, but it requires at least three meaningful time points or period subspaces.
- Period-subspace DS should use multiple images per side when available, for example early-month versus late-month, first half-year versus second half-year, or pre-event window versus post-event window.
- SSA, SFA, DMD, RTW, and Fourier time-series analysis are future temporal tools. They need enough temporal depth and a clear target before implementation.
- RTW/Deep RTW suggests a way to compare date sequences without assuming events happen at the same temporal speed: randomly sample ordered date subsequences, build a sequence subspace, and compare subspaces by canonical angles or DS/GDS. This is only meaningful for multi-date data, not OSCD's two dates.
- PCA-SFA and Slow Feature Subspace suggest a way to separate slow/background temporal variation from faster anomalous change. A satellite version would need same-season or dense time series so "slow" does not simply mean seasonal drift.
- Product Grassmann and Hankel-like temporal embeddings suggest treating satellite data as factors, for example spectral subspace, local spatial/patch subspace, and temporal/date subspace, instead of flattening everything into one unordered pixel matrix.
- G-LMSM and Signal Latent Subspace suggest a neural-subspace hybrid: extract CNN/foundation-model features, build subspaces from patch/date features, and either compare them geometrically or learn dictionary subspaces on the Grassmann manifold.
- Pseudo-whitened mutual subspace methods suggest a discriminative projection step before subspace comparison. For this project, that would require labels or reliable pseudo-labels and should not be mixed into unsupervised DS claims.
- Shape-subspace and superposed-shape-subspace papers are useful because they turn spatial structure into subspace geometry. The satellite analogy is patch-stack or key-date subspaces that preserve local geometry better than global unordered pixel PCA.

Spatial and interpretability variants:

- Sliding/local-window DS was already proposed in old specs and now directly matches Sensei's spatial-information critique.
- Band-group DS and per-band attribution can help explain whether DS is driven by visible, red-edge, NIR, SWIR, or atmospheric bands.
- PCA reconstruction error can be a baseline or diagnostic: fit a pre/no-change PCA model and use post reconstruction residuals as change/anomaly evidence.
- Coordinate- or patch-aware variants are method candidates only if they improve spatial meaning without turning into arbitrary feature engineering.

Green Learning / PixelHop / wavelet route:

- Green Learning and PixelHop/PixelHop++ were suggested as possible ways to preserve multi-scale spatial information through successive subspace-style local transforms.
- A satellite version would treat local patch responses or multiscale components as the samples/features for DS, KDS, CCA, or SSC rather than using unordered raw pixels.
- Wavelet and image-compression intuition is relevant because it decomposes image content into scale/frequency components. This may help separate local structure from broad radiometric change.
- This is a future feature-construction route, not implemented evidence. It should be tested only after the global-vs-spatial DS audit defines what spatial information is missing.
- Senpai's specific pyramid idea: build one subspace for the whole image, then subspaces for a `2x2` split, then `4x4`, then finer grids; combine or compare subspace responses across scales. This is better called a multiscale subspace pyramid until the exact Green Learning / PixelHop / wavelet reference is confirmed.

Fukui subspace-set overview:

- The 2024 "Geometry of subspace set and its application to machine learning" PDF is a lab overview tying together subspace method, mutual subspace method, canonical angles, DS, GDS, GDS projection, kernel variants, and Grassmann representation.
- Use it as a conceptual map for explaining how this project relates to the lab's broader subspace family.
- Do not cite it as the primary source for a claim when a peer-reviewed DS/GDS/KDS/MSM paper is the actual source.

SSC and sparse modeling:

- Sparse Subspace Clustering is a future baseline or pseudo-label source, not current core evidence.
- A defensible SSC role would be: cluster change-feature vectors into change types, compare against DS/PCA-diff, or provide auxiliary targets/channels for a neural model.
- Dictionary learning and LASSO are future sparse-modeling tools for feature extraction or band/feature selection.

Geometry and decision layers:

- Grassmann geodesic and SPD covariance-geodesic change scores are future local change-map candidates.
- GFK is a possible domain-shift bridge between pre/post subspaces, but it needs a clear experiment.
- Product Grassmann manifold is a possible way to fuse several subspace factors, such as spectral, spatial, temporal, or deep-feature subspaces. It is inspired by the SLS paper, but it is not active in this repo.
- MCDA, edge/server payload formulas, UAV deployment, DMaaS, and graph decision layers are application/decision-system ideas, not part of the current implemented method.

Archive details recovered in the second pass:

- The old Phase 1 spec used `rank_r: 6`, local DS windows such as `64` or `96`, and strides such as `32` or `48`. Current experiments should broaden that grid rather than treating the old defaults as final.
- Old PCA-diff design used PCA on `X_post - X_pre`, with `S` chosen either by an energy threshold around `95%` or by a small rank such as `3` to `5`.
- Old Celik baseline defaults were local windows `h in {7, 9, 11}`, PCA energy at least `90%`, `k=2`, `k-means++`, and runtime controls such as `downsample_max_side`.
- Old OSCD thresholding had two modes: per-tile Otsu with no labels, and train-calibrated global thresholds over `0.05..0.95` with step `0.05`, optimized by F1 or IoU. This thresholding is for Phase 1 binary maps, not for Phase 2 prior channels.
- Old MultiSenGE pair strategies included `earliest_latest`, `adjacent`, and `first_mid_last`; do not use earliest/latest as the only evidence because it maximizes seasonality and cloud/snow risk.
- Old quick-figure scripts preserved three paper/slide hooks: Delta1/Delta2 grids, toy MCDA heatmaps, and uplink payload curves. These are explanatory figure ideas, not active experiment code.
- The old edge payload equation was `embedded ~= T*H*W*m*4` bytes/s versus raw 16-bit imagery roughly `T*H*W*d*2` bytes/s. A deployment claim would need a measured target such as `5-10x` reduction, not only this formula.
- Old augmentation defaults were synchronized pre/post transforms: rotations, flips, scale `0.9-1.1`, translation up to `5%`, brightness/contrast about `+/-10-15%`, gamma `0.9-1.1`, Gaussian noise up to about `0.01`, and blur sigma `0.5-1.0`. These are historical training defaults, not current evidence.
- Old xView2 notes claimed about `22,068` images, `11,034` pre/post pairs, about `850,736` building polygons, and about `45,000 km^2`; verify from primary sources before citing.
- Old damage-extension metrics included quadratic-weighted kappa for ordinal damage and xBD-S12-style `F1loc`, `F1dmg`, `F1comp`. These belong only to a promoted damage task, not OSCD.

Historical conflict to preserve:

- Some old `research-notes/` files said residual/eig DS behaved almost identically on OSCD. The later subspace audit found the active residual-stack path was not paper-faithful enough and behaved almost like raw L2 on Beirut. Treat the old equivalence statement as historical conflict, not current truth.

## 11. Venus KDS/KGDS Audit

Sensei's Venus data:

```text
data/venus_tpami2015/venus_tpami2015_no_accessories.mat        -> (480, 640, 1, 300)
data/venus_tpami2015/venus_tpami2015_earrings.mat              -> (480, 640, 1, 300)
data/venus_tpami2015/venus_tpami2015_earrings_necklace.mat     -> (480, 640, 1, 300)
```

Demo representation:

```text
downsampled view = 63 x 48 = 3024-D vector
one sculpture set = 3024 x 300 matrix
KDS = two 100-D kernel subspaces
KGDS = three 150-D kernel subspaces, 300-D KGDS
```

Current status:

- Kernel coefficient and projection equations are implemented.
- Preimage reconstruction is not implemented.
- Therefore, current Venus outputs are projection-energy/coordinate diagnostics, not full TPAMI figure reproduction.

## 12. Spatial Information Problem

Current global pixel DS sees unordered 13-D samples:

```text
road pixel, sea pixel, building pixel, vegetation pixel -> all columns in one matrix
```

This is not automatically wrong, but it may be too weak for spatial change detection.

Spatial alternatives to test:

- Patch-vector DS: one sample is a `3x3x13` or `5x5x13` local patch.
- Local-window DS: one subspace is fit per `64`, `128`, or `256` pixel region.
- Coordinate-augmented DS: diagnostic only, not a main method.
- CCA/S3CCA-style structured matching: larger future research track.

### Spatial-subspace novelty boundary

The latest literature check changes the claim boundary:

- Spatially aware subspace ideas already exist in remote sensing and hyperspectral imaging.
- Therefore the thesis should not claim that "spatial subspaces for satellite images" are entirely new.
- The possible niche is narrower: adapt and evaluate DS/GDS-style subspace geometry for multispectral satellite change maps while comparing sample definitions that preserve different amounts of spatial support.

Useful related-work anchors:

- Wu, Du, and Zhang, 2013, "A Subspace-Based Change Detection Method for Hyperspectral Images" uses subspace reasoning for hyperspectral change detection. The MDPI integrated HSI CD paper lists it as a core classical HSI-CD reference: https://www.mdpi.com/2072-4292/14/11/2523
- Chen and Wang, 2017, LRSD_SS uses spectrally-spatially regularized low-rank/sparse decomposition for multitemporal hyperspectral change feature extraction: https://www.mdpi.com/2072-4292/9/10/1044
- Spectral-spatial sparse subspace clustering exists for hyperspectral remote-sensing image analysis, so clustering/spatial-subspace language is not novel by itself.

Safe current claim:

```text
Spatial support is the key unresolved adaptation problem for using DS-style subspace geometry on Sentinel-2 change maps.
```

Unsafe current claim:

```text
This project invents the first spatial subspace representation for satellite images.
```

The current method candidates should be treated as a hierarchy:

1. global pixel DS: easiest, least spatial;
2. local-window DS: still pixel-sample based, but region-specific;
3. patch-vector DS: preserves local layout inside each sample;
4. object/superpixel DS: preserves semantic/spatial units if reliable objects exist;
5. tensor/n-mode DS or GDS: preserves modes explicitly, but is a later method track.

### Tensor / n-mode GDS idea

Tensor or n-mode GDS is intriguing because satellite data are naturally multi-dimensional:

```text
bands x height x width
bands x patch_height x patch_width
bands x height x width x time
```

Ordinary global pixel DS flattens this structure into columns of 13-band samples. Tensor/n-mode methods instead try to keep separate modes such as spectral, spatial, and temporal axes.

The relevant reference is Gatto et al., "Tensor Analysis with n-Mode Generalized Difference Subspace," which proposes n-mode GDS for tensor data because ordinary GDS does not directly handle tensor structure: https://arxiv.org/abs/1909.01954

Project role:

- future method-development track, not immediate OSCD evidence;
- potentially strong if the thesis shifts from "better prior channel" to "subspace construction for spatial-spectral-temporal satellite tensors";
- should only be implemented after the simpler global/window/patch audit identifies what structure is actually missing.

### Multiscale subspace pyramid / Green Learning lead

Senpai's wavelet/JPEG-inspired idea can be written as a spatial hierarchy:

```text
level 0: whole image                         -> 1 subspace
level 1: split image into 2 x 2 cells        -> 4 subspaces
level 2: split image into 4 x 4 cells        -> 16 subspaces
level 3: split image into 8 x 8 cells        -> 64 subspaces
...
```

For each spatial cell, build a pre-date subspace and a post-date subspace from the valid 13-band pixels or patch vectors inside that cell, then compute a DS-style score for that cell or for pixels inside it.

Why this matters:

- coarse levels preserve global scene context;
- fine levels preserve local spatial context;
- this directly targets Sensei's concern that global pixel DS breaks spatial information;
- it resembles wavelet/multiresolution thinking because information is represented at several spatial scales;
- it also resembles Green Learning / PixelHop intuition because local transforms can be stacked from coarse-to-fine or low-to-high detail.

Important caveat:

- Do not call this exact method "Green Learning" until the source paper/notes are matched to the implementation.
- Safer working name: `multiscale_subspace_pyramid`.
- First implement it as an audit after global/window/patch DS, not as the first experiment.

Possible score aggregation:

```text
score_pixel = weighted_sum(level_0_score, level_1_cell_score, level_2_cell_score, ...)
```

Open implementation questions:

- Should each cell score be constant over the cell, or should each cell fit a local DS basis and then score each pixel inside the cell?
- What weights should be used across scales: equal, inverse level, validation-selected, or learned?
- Should cells overlap to reduce block artifacts?
- Should the pyramid operate on raw 13-band pixels or patch vectors?
- Should it compare DS projection energy, residual energy, or normalized projection ratio?

Implementation details to preserve:

- `3x3` patch vectors have `13 x 3 x 3 = 117` values.
- `5x5` patch vectors have `13 x 5 x 5 = 325` values.
- The DS score for patch-vector DS should be assigned to the center pixel of each patch.
- Patch borders and valid masks must be handled explicitly; do not silently drop border regions without reporting the count.
- Full-image patch extraction can be large, so sampled fitting is acceptable if the sampling seed, sample count, and city list are recorded.
- Local-window DS should save both per-city metrics and maps, and should inspect boundary artifacts from overlapping-window aggregation.
- Audit whether any existing `sliding_window_ds` code path uses corrected `canonical`/`eig` DS or still behaves like legacy residual-stack DS before using it as evidence.

## 13. Projection Back To Image Space

Current OSCD DS does not reconstruct a 13-band image.

It computes one scalar score per pixel and reshapes scores back to the image grid.

Possible interpretation task:

```text
delta_x      = x_post - x_pre
delta_x_ds   = D D^T delta_x
residual     = delta_x - delta_x_ds
```

This could show which band combinations DS emphasizes and may help explain projection to Sensei/senpais.

## 14. Method Caveats

These caveats were extracted from the old archive docs and should stay visible because they affect thesis claims and future code work.

### Subspace construction card

For every method variant, write a short construction card before treating its output as research evidence. This is the seminar-critical explanation.

Required fields:

```text
variant name:
source/reference:
sample unit:
input matrix/tensor:
subspace count:
subspace basis shape:
how basis is fit:
how pre/post are compared:
score map semantics:
spatial information preserved:
spatial information lost:
code path:
verification:
```

Current variant cards:

| variant | sample unit | input object | subspace construction | comparison / score | spatial status |
|---|---|---|---|---|---|
| global pixel DS | one valid pixel | `X_pre, X_post in R^(13 x N)` | PCA on all valid 13-band pixels per date, giving `Phi, Psi in R^(13 x r)` | canonical/eig DS basis `D`, score `||D^T(x_post - x_pre)||^2` | loses pixel position during fitting; restores scores to map afterward |
| local-window DS | one valid pixel inside a window | for each window, `X_pre_w, X_post_w in R^(13 x N_w)` | PCA per pre/post window | score pixels inside that window with the window's DS basis; aggregate overlaps | preserves regional context; may create boundary/block artifacts |
| patch-vector DS | one local patch centered on a pixel | `3x3x13 = 117-D` or `5x5x13 = 325-D` sample vectors | PCA on flattened patch samples per date | score center pixel using DS in patch-feature space | preserves local layout inside each sample; higher dimensional |
| multiscale subspace pyramid | one cell or pixel inside a pyramid cell | whole image, `2x2`, `4x4`, `8x8` grid cells | PCA/DS per spatial cell and date | combine per-scale DS scores into one map | preserves coarse and fine spatial support; weights and block artifacts must be audited |
| tensor / n-mode GDS | tensor sample or tensor mode | e.g. `bands x height x width x time` | mode-wise subspaces instead of full flattening | n-mode GDS / tensor comparison | preserves explicit modes; future method track |
| deep-feature subspace | encoder feature vector or patch embedding | feature matrix from CNN/foundation model | PCA/subspace over learned features | DS/GDS over latent feature subspaces | preserves whatever the encoder learned; source of interpretability is weaker unless audited |

Use this wording in seminar if asked:

```text
The main difference between variants is the sample unit used to construct the subspace. In global pixel DS, the sample is one 13-band pixel. In patch DS, the sample is a local spatial patch. In local-window DS, the subspace is fitted separately inside each region. The experiment is testing which sample definition preserves enough spatial information for a meaningful change map.
```

### Source-linked implementation workflow

For every new method or experiment, Codex must leave a clear trail from source material to code. This is now part of the research workflow because the previous project risk was implementing methods without the student understanding or verifying the source.

Before implementing a paper-derived or reference-code-derived method, write down:

1. Source material:
   - paper title, equation/section if known, URL or local path;
   - reference code path if used;
   - what is treated as theory, what is treated only as implementation guidance.
2. Satellite data object:
   - exact sample definition: pixel, patch, local window, object, date, or deep feature;
   - input shape before transformation;
   - output shape and map semantics.
3. Mathematical object:
   - basis/subspace dimension;
   - projection, residual, canonical-angle, kernel, or tensor operation;
   - scalar score definition if a map is produced.
4. Code object:
   - new or edited files;
   - function names;
   - config keys;
   - dependency choices.
5. Verification:
   - toy test or shape check;
   - equal/shared-subspace edge case if relevant;
   - reference-code comparison if available;
   - one-city smoke output before full experiments.

The expected discussion flow with the student is:

```text
source -> math object -> satellite adaptation -> code path -> test -> one-city output -> thesis claim
```

Do not jump from source directly to large experiments. Do not promote a method to "evidence" until this chain is clear.

### Subspace code reading path

Use this order when re-checking the DS/KDS implementation:

1. `phase1/scripts/audit_oscd_subspace.py`
   - Load one city, normalize, vectorize, fit PCA, compare legacy/eig/canonical DS.
2. `phase1/data/preprocessing.py`
   - Focus on `vectorize_cube` and `devectorize_cube`.
3. `phase1/ds/pca_utils.py`
   - Focus on `fit_pca_basis`, `difference_subspace_canonical`, `difference_subspace_eig`, and `legacy_residual_stack_difference_subspace`.
4. `phase1/ds/ds_scores.py`
   - Focus on `_compute_ds_matrix_scores`.
5. `phase1/scripts/venus_kds_demo.py`
   - Maps Sensei's Venus files into the TPAMI-style KDS/KGDS audit.
6. `phase1/subspace/kernel_difference_subspace.py`
   - Code version of the kernel coefficient/projection equations.

Read the TPAMI 2015 DS/GDS paper before interpreting the Venus KDS/KGDS code, and read S3CCA only after the current DS construction is clear.

### Paper-to-code verification

Current tests should be treated as method-verification guards, not as a claim that the codebase is final.

For each paper-derived method, keep a small verification path:

- restate the mathematical input/output object and dimensions;
- test toy cases from the paper logic, such as equal subspaces producing zero/empty DS;
- compare against trusted reference code when available;
- check map semantics, for example whether a DS score is a projection energy, residual energy, normalized ratio, or thresholded mask;
- record where implementation choices depart from the paper, such as global pixel samples versus image-set samples.

This applies first to DS/KDS and then to PCA-diff, Celik PCA-kmeans, IR-MAD, CVA, GDS/KGDS, CCA/KCCA/S3CCA, and future spatial/temporal variants.

### Prior folder naming

`phase1/outputs/oscd_saved_priors_fast` is a saved Phase 1 prior-map folder. It is not a trained model and "fast" does not mean optimized or best. It means the prior maps were generated with a faster Phase 1 configuration.

Because those DS maps came from the old residual-stack path, use this wording:

```text
legacy residual-stack DS priors
```

Do not call that folder clean paper-faithful DS evidence. Future output names should be more explicit, for example:

```text
priors_oscd_legacy_residual
priors_oscd_canonical_ds
priors_oscd_classical_full
```

### Baseline interpretation

- `pixel_diff` and CVA are effectively the same spectral L2 magnitude in the current code. Do not present them as independent methods unless the implementation is changed or the duplication is explicitly stated.
- Runtime claims from old Phase 1 summaries are unsafe unless timing is instrumented per method. Do not claim one prior is faster/slower from mixed old logs alone.
- Geodesic-prior smoke logs contain `NotGeoreferencedWarning`; do not assume real geospatial coordinates from those rasters unless metadata is separately verified.

### Phase 1 thresholding vs Phase 2 priors

Phase 1 can evaluate score maps with Otsu or train-calibrated thresholds. Phase 2 does not train on those thresholded masks by default. It loads continuous min-max-normalized prior score maps as additional channels.

This distinction matters:

```text
Phase 1: score map -> optional threshold -> unsupervised binary metric
Phase 2: continuous score map -> extra neural-network input channel
```

Therefore, a weak Phase 1 thresholded F1 does not automatically mean a prior is useless for Phase 2, but it is a warning sign.

### Phase 1 engineering details

- OSCD normalization uses bandwise z-score statistics from `phase1/data/oscd_band_stats.json` through `fit_or_load_band_stats` / `load_band_stats`.
- The stats path is configurable. If missing, the Phase 1 runner can create it, so record the generated stats file with any thesis-usable run.
- Valid masks are built from pre and post images and paired operations use `valid_pre AND valid_post`.
- Common configs use `min_valid_bands: 3`; this is a practical nodata/missing-band rule, not a mathematical DS requirement.
- Cloud/SCL masking is not currently an obvious part of the OSCD pipeline; do not imply cloud-aware masking without adding it.
- Phase 1 score normalization, such as percentile clipping or min-max scaling, is an engineering choice for comparable maps and neural-network inputs, not part of DS theory.
- DS eigen thresholds such as `eps=1e-6` are numerical defaults. Expose or record them if a result depends on them.
- IR-MAD subsampling is seeded in the current code, but still record the seed because old IR-MAD behavior and runtimes can otherwise be hard to reproduce.
- `Celik` and `IR-MAD` maps usually require the fuller Phase 1 prior folder, not only the fast/core prior folder.

### Split and evaluation caveats

- OSCD has train/test labels. The project creates its own validation split by holding out train cities. Do not call the validation set "official" unless that split is externally verified.
- If a document says "official split," check whether it means the OSCD train/test split or the project-defined train/val split.
- Current final Phase 2 evaluation should use stitched city-level masks, not averaged patch metrics, for thesis claims.
- Default Phase 2 evaluation threshold is often `0.5`; threshold tuning and probability-map analysis are separate tasks.

### Phase 2 training and evaluation details

- Raw Phase 2 input is `[pre(13), post(13)] = 26` channels.
- Prior maps are loaded from `phase1/outputs/.../oscd_change_maps/<split>/<method>/<city>_score.npy`.
- Prior channels are min-max normalized per tile before being appended to the raw channels.
- Patch lists are generated per city with configured `patch_size` and `patch_overlap`.
- Training is patch-wise because full OSCD city images are large.
- Final validation/evaluation should stitch overlapping patch probabilities back into one full-city probability map before computing city metrics.
- `best.ckpt` should be monitored using the same stitched validation style that will be used for final evaluation.
- `BCEDiceLoss` computes BCE plus soft Dice on valid pixels only, and supports `pos_weight`.
- Random augmentations/noise are train-only; validation/test should be deterministic.
- `RandomGaussianNoise` applies only to raw channels and valid pixels, not to priors or nodata.
- If a pretrained ResNet backbone is used with non-RGB inputs, the first convolution is inflated from RGB ImageNet weights to the configured channel count.
- In code terms, that first convolution is the ResNet `conv1`; record this when comparing pretrained and non-pretrained runs.
- Some model paths upsample logits to the ground-truth mask size during training/evaluation. This is acceptable, but record it if native model output resolution differs from target size.
- Class-imbalance options beyond current `pos_weight`, such as focal loss or change-heavy patch sampling, remain possible future experiments, not current evidence.
- Raw-only CLI paths can still require `--phase1_change_maps_root`; this is confusing but harmless if the config has no enabled priors.
- Saved priors are assumed spatially aligned with OSCD tiles; smoke checks only prove shape/channel loading for limited patches, not full alignment.

### Loader and model caveats

- `OSCDSegmentationDataset` uses the configured/default 13-band order. Future configs should make band order explicit to avoid silent assumptions.
- Archive wording called this an explicit `band_order` risk: if configs omit `band_order`, the loader falls back to the default 13 Sentinel-2 bands.
- `valid_mask = valid_pre AND valid_post` is meant to avoid nodata and border artifacts, but its impact on labeled changed pixels must be measured.
- `PriorsFusionUNet` is a lightweight early-fusion/reweighting variant, not a full attention or sophisticated fusion architecture.
- `PriorsFusionUNet` assumes positive raw and prior branch channel counts. A priors-only or raw-only fusion configuration is unsafe unless the model branch is audited.
- A priors-only configuration is unsafe unless the dataset/model path is audited; current code paths were mainly exercised with raw channels present.
- Dependency versions are not fully pinned. For old-result reproduction, record the active `.venv`, Torch build, CUDA availability, and package versions in run metadata.
- `DamageDatasetAdapter` is a generic CSV adapter only. The OSCD trainer/evaluator currently constructs `OSCDSegmentationDataset`; it does not switch into xBD/xBD-S12 damage training.
- There is no integrated `train_damage_seg.py`, xBD-S12 loader, multi-class damage loss, or damage-specific metric pipeline yet.
- `phase1.eval.run_oscd_eval.aggregate_metrics()` was flagged in the archive as a stub risk. Do not rely on that function unless the current code path is audited.

### MultiSenGE caveat

MultiSenGE remains exploratory. Before using it for GDS/KGDS, audit the manifest and path assumptions, especially Sentinel-1 paths. Old notes flagged a possible local-path mismatch around `s1/<name>.tif` versus nested `s1/s1/<name>.tif`.

Old MultiSenGE visual checks paired the earliest valid date with the latest valid date for each spatial patch. This is simple and maximizes the temporal baseline, but it makes seasonality and snow/cloud effects likely. For thesis-grade evidence, use controlled pair selection or multi-date sequence analysis rather than relying only on earliest/latest pairs.

### KDS and CCA implementation gaps

OSCD KDS is possible but not yet specified enough to claim:

- full-image pixel KPCA is too expensive for hundreds of thousands to millions of pixels;
- a sampled global KDS would need controlled pixel sampling and out-of-sample projection for every pixel;
- a local/windowed KDS would need local model fitting and runtime control;
- a Nyström/prototype KDS would need representative pixel prototypes;
- a patch-vector KDS would need explicit patch extraction and border/mask handling.

CCA/S3CCA and KCCA are separate future routes. They matter because they can preserve or compare structured sample relationships, while current global PCA treats pixels as exchangeable columns. Do not present CCA, KCCA, or S3CCA as implemented in the active OSCD pipeline.

Possible satellite CCA views:

- pre-date features versus post-date features;
- spectral-band vectors versus patch/context vectors;
- raw Sentinel-2 features versus encoder/deep features;
- earlier date-window subspaces versus later date-window subspaces;
- DS/PCA-diff features versus semantic or pseudo-label features.

CCA/KCCA/S3CCA/TRCCA should be considered if the chosen problem is structured matching, partial correspondence, or temporal alignment. They should not be added just because reference code exists.

Reference-code method expansion backlog:

- The bundled reference code contains more than DS/KDS. It includes PCA, KPCA, CCA, Kernel CCA, kernel basis construction, canonical-angle similarity, mutual-subspace variants, RTW-style temporal matching, Grassmann utilities, and subspace-magnitude/decomposition utilities.
- Treat these implementations as reliable starting points and cross-check material, not ground truth. Each candidate method still needs a paper-to-code verification note, toy tests, dimensional checks, and an OSCD/MultiSenGE-specific adaptation plan.
- Candidate data objects for our satellite setting:
  - global pixel samples: one column is one 13-band pixel;
  - patch samples: one column is one flattened `k x k x 13` patch;
  - local-window subspaces: one subspace is built per image region;
  - date subspaces: one subspace is built per Sentinel-2 date in a multi-temporal sequence;
  - object-level samples: one column is a building, greenhouse, connected component, polygon, or object-proposal descriptor;
  - deep-feature samples: one column is an encoder feature, patch embedding, or tile/date embedding.
- Candidate method families to screen:
  - KPCA/KDS: nonlinear version of the current PCA/DS idea, likely requiring sampling, Nyström prototypes, or local windows;
  - CCA/KCCA: compare two views of the data, such as pre/post, spectral/spatial features, raw/deep features, or date-pair representations;
  - S3CCA/TRCCA/KOTRCCA: structured or temporal CCA routes if the project shifts toward patch sequences or multi-date Sentinel-2;
  - MSM/KMSM/CMSM/KCMSM: image-set/subspace classifiers that may become useful if we define pseudo-classes, change types, or labeled object/date groups;
  - RTW/SFA/temporal tensor/Product-Grassmann methods: future multi-date routes for preserving time ordering rather than treating dates as unordered samples;
  - Grassmann magnitude/decomposition tools: useful for attribution, region comparison, and first/second-order change analysis.
- Minimum evidence gate before any family becomes a real project method:
  - identify the exact reference files and paper equations;
  - define samples, features, labels/evaluation proxy, and output map semantics;
  - show a small city-level smoke result with runtime and memory;
  - compare against raw L2, PCA-diff, canonical/eig DS, and a modern neural baseline when relevant;
  - record whether the method preserves spatial position, temporal order, both, or neither.

Object-level descriptors are a possible pivot, not current OSCD evidence. They may fit greenhouse monitoring, building change, or damage mapping better than unordered global pixel DS because the sample unit carries semantic/spatial identity. They require object labels, polygons, or reliable object proposals before they can become a defensible experiment.

### Deep-feature subspace gap

The Signal Latent Subspace paper suggests a different adaptation: instead of fitting subspaces directly on raw Sentinel-2 band vectors, first extract latent features from a neural model and then build subspaces from those features.

This could answer a different question:

```text
Do subspace methods become more useful when their samples are spatial/deep feature vectors rather than raw per-pixel spectra?
```

Possible satellite variants:

- patch embeddings from a remote-sensing CNN or foundation model;
- U-Net encoder features from raw pre/post images;
- separate subspaces for spectral, spatial, temporal, and prior-map feature factors;
- product-Grassmann fusion across several feature factors;
- GDS projection to improve separation between known classes or pseudo-classes.

This should wait until the raw/global/local DS audit is clear.
