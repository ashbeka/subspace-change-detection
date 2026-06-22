# Method Notes

## Table Of Contents

- [1. Current Project Scope](#1-current-project-scope)
- [2. Data](#2-data)
- [3. Phase 1: Prior Maps](#3-phase-1-prior-maps)
  - [Feature-Extraction-Like Role](#feature-extraction-like-role)
  - [Subspace Feature Isolation Route](#subspace-feature-isolation-route)
  - [Spectral And Spatial-Band Subspace Candidate](#spectral-and-spatial-band-subspace-candidate)
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
  - [IR-MAD versus DS](#ir-mad-versus-ds)
  - [Phase 1 thresholding vs Phase 2 priors](#phase-1-thresholding-vs-phase-2-priors)
  - [Phase 1 engineering details](#phase-1-engineering-details)
  - [Split and evaluation caveats](#split-and-evaluation-caveats)
  - [Phase 2 training and evaluation details](#phase-2-training-and-evaluation-details)
  - [Loader and model caveats](#loader-and-model-caveats)
  - [MultiSenGE caveat](#multisenge-caveat)
  - [KDS and CCA implementation gaps](#kds-and-cca-implementation-gaps)
  - [Deep-feature subspace gap](#deep-feature-subspace-gap)
- [15. Temporal Band-Image Difference-Subspace Dynamics](#15-temporal-band-image-difference-subspace-dynamics)
- [16. Cross-Method Fidelity And Status](#16-cross-method-fidelity-and-status)
- [17. Cross-Sensor Band-Image Boundary](#17-cross-sensor-band-image-boundary)

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

### Feature-Extraction-Like Role

The project does not currently have one single learned feature extractor whose purpose is "understand the raw data" before change detection.

Instead, it has several feature-extraction-like transforms:

1. Raw spectral stack:
   - Input is still raw data: `pre 13 bands + post 13 bands`.
   - This is not feature extraction; it is the baseline representation given directly to the model or classical methods.

2. PCA/subspace construction:
   - This is the main classical feature-extraction-like step.
   - It converts many sample vectors into a lower-dimensional basis.
   - For global OSCD DS, samples are `13-D` pixel vectors and PCA produces a `13 x r` basis.
   - For patch DS, samples are `k x k x 13` patch vectors and PCA produces a basis in patch-feature space.

3. DS projection score:
   - This is not a generic feature extractor; it is a change-specific transform.
   - It uses the pre/post subspace relationship to turn each pixel or patch difference into a scalar change score.
   - The output is an interpretable prior/change map, not a learned representation.

4. Other classical transforms:
   - PCA-diff, Celik PCA-kmeans, raw L2/CVA, and IR-MAD also act like handcrafted feature/change extractors.
   - They summarize pre/post multispectral differences into score maps or transformed variables.

5. Neural feature extraction:
   - U-Net/Siamese U-Net learn internal convolutional features during Phase 2.
   - Those learned features are currently used for supervised segmentation, not for explaining the raw data.
   - A future deep-feature subspace experiment could explicitly extract CNN/remote-sensing encoder features and build subspaces from them, but that is not the current implemented core.

Short answer:

```text
The current "feature extraction" layer is classical PCA/subspace-derived prior-map generation, especially DS/PCA-diff/Celik/IR-MAD. It is used to turn raw 13-band pre/post Sentinel-2 data into interpretable change-score maps. The U-Net also learns features internally, but that is not the project's main explanatory feature-extraction method yet.
```

### Subspace Feature Isolation Route

A separate possible role for subspace methods is not to directly output the final change map. It is to isolate useful multispectral characteristics before another change method uses them.

In this route, the subspace basis is treated as a feature-isolation tool:

- PCA/subspace directions can separate dominant variation patterns in the 13 Sentinel-2 bands.
- DS/KDS/GDS projections can isolate directions where pre/post or multi-date subspaces differ.
- Band-group subspaces can separate visible, red-edge, NIR, SWIR, and atmospheric-band behavior.
- Local, patch, object, or date-window subspaces can isolate spatially or temporally localized characteristics.
- Latent-feature subspaces can isolate learned characteristics from a U-Net, Siamese network, or remote-sensing encoder.

The potential research question is:

```text
Can subspace projections isolate multispectral characteristics that make downstream change detection more interpretable, robust, or label-efficient than using raw bands alone?
```

Important caution:

- An unsupervised subspace direction is not automatically a semantic feature like "building," "vegetation," or "water."
- It is first a direction of variance, difference, correlation, or discriminative separation depending on the method.
- To call it a meaningful characteristic, verify it through band attribution, region examples, correlation with known indices/classes, downstream metrics, or expert/manual inspection.

Possible outputs:

- projection coefficient maps;
- reconstructed band-wise contribution maps;
- residual/background-separated maps;
- band-group attribution tables;
- feature channels passed to PCA-diff, IR-MAD, U-Net, clustering, or another change detector.

### Spectral And Spatial-Band Subspace Candidate

The updated Apple Notes and Jang discussion introduced a second possible meaning of "spectral subspace." Keep these variants separate:

| Variant | Sample unit | Matrix shape idea | What it preserves | Main risk |
|---|---|---|---|---|
| global pixel spectral subspace | one pixel location | `13 x N` | joint 13-band values at many pixels | ignores pixel position during PCA fitting |
| patch-vector spectral-spatial subspace | one local patch | `(k*k*13) x N_patches` | local neighborhood around each pixel | high dimensional and slower |
| local-window spectral subspace | all pixels inside one window | `13 x N_window` per window | local distribution without mixing whole city | block/window artifacts and runtime |
| Band-Image DS | one Sentinel-2 band image | `N_pixels x 13` or local equivalent | spatial layout inside each band vector | only 13 band samples; rank and stability need checks |
| band-group/product subspace | VIS/red-edge/NIR/SWIR groups | multiple group-specific matrices | separates physical band families | needs careful fusion and attribution |

The Band-Image DS variant is a concrete response to the concern that pixel samples destroy spatial layout. In that version, each column can be one flattened band image, so the vector entries are pixel positions rather than spectral bands:

```text
current global pixel DS:
  one sample vector = one pixel location
  vector dimension  = 13 Sentinel-2 band values
  matrix shape      = 13 x N_pixels

Band-Image DS spatial candidate:
  one sample vector = one Sentinel-2 band image
  vector dimension  = H x W pixel positions
  matrix shape      = N_pixels x 13
```

For Beirut, this would be roughly:

```text
X_pre_flat_bands in R^(1,262,600 x 13)
```

where each of the 13 columns is one full flattened band image. This is closer to "one channel image as one high-dimensional spatial vector." It gives PCA a high-dimensional ambient space, but it also gives PCA only 13 samples. Therefore the maximum useful centered PCA rank is at most `12`, and practical rank may be unstable. This is not automatically more correct than pixel-vector DS; it is a different sample definition that should be tested.

Why Senpai's suggestion is worth testing:

- It preserves spatial layout inside each band vector instead of discarding pixel position during fitting.
- It is closer to classical image-set language: each sample is a whole image-like object, not one pixel.
- It may become more natural for hyperspectral imagery, where `B` could be hundreds of channels rather than 13.
- It can answer a seminar question clearly: "What happens if each band is the vector, instead of each pixel?"

Risks:

- With Sentinel-2, 13 samples is very small for robust PCA in a million-dimensional space.
- The learned subspace may describe band-to-band spatial similarity more than changed-area evidence.
- Projection/scoring back to a pixel map must be defined carefully; otherwise it may become a global image-level score, not a local change map.
- It must be compared against patch-vector, local-window, PCA-diff, raw L2/CVA, and IR-MAD before being promoted.

Do not use "spectral subspace" as a thesis term until the sample unit, matrix shape, rank rule, scoring equation, and expected output map are fixed.

2026-06-18 evidence update:

- `flatbands` was renamed to `band_image_ds`; `flatbands` remains a compatibility alias only.
- `band_image_ds` is implemented in `phase1/scripts/compare_oscd_spatial_subspaces.py::band_image_ds_score`.
- It was swept over all 24 local OSCD cities at ranks 6, 8, 10, and 12, with additional core-city checks at ranks 2-5.
- It became the strongest DS-family method by AP in 44 of 48 city/rank runs.
- It still did not beat PCA-diff on mean AP, so it is a promising sample-definition candidate, not a proven detector.
- The score map is produced by projecting the 13-column band-difference matrix into the spatial DS and summing projected energy per pixel across bands.
- A follow-up score ablation showed that `band_image_norm` keeps the same AUROC/AP as `band_image_ds` but improves all-city Otsu F1 from `0.1129` to `0.2007`.
- Per-band projected-energy attribution maps are available for selected cities to inspect which Sentinel-2 bands drive the projected score.
- Rank 12 is the strongest tested setting: mean AUROC `0.8477`, AP `0.2410`, and best F1 `0.3021`. Its AP remains below PCA-diff (`0.2541`), and using the maximum centered rank weakens the interpretation that a compact DS alone explains the useful signal.
- Against corrected pressure baselines, rank-8 Band-Image DS is significantly worse than PCA-diff by paired city AP, significantly better than the current Celik adaptation, and not reliably different from raw L2 or repaired IR-MAD.
- Equal-weight within-image rank fusion of PCA-diff, Band-Image DS, and IR-MAD raises mean AUROC to `0.8708` and wins 21/24 cities against PCA-diff, but the AP gain is not significant and Otsu F1 falls. This is complementarity evidence, not a finished detector.
- A fixed-grid `1x1 + 2x2 + 4x4` pixel-spectral DS pyramid did not improve core-city AP over global pixel DS. Stop that exact construction; it is not equivalent to Green Learning, PixelHop, or wavelet features.

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

### Post-Classification Change Detection

Post-classification change detection means:

```text
pre image -> classify each pixel/object/state
post image -> classify each pixel/object/state
change map = compare pre class/state with post class/state
```

This is different from the current direct-change setup:

```text
pre image + post image -> score or segment changed areas directly
```

Project interpretation:

- PCCD is useful when the research question is semantic or state-based, for example `vegetation -> urban`, `active greenhouse -> abandoned greenhouse`, or `undamaged building -> damaged building`.
- PCCD is less natural for current OSCD because OSCD provides binary changed-area labels, not full pre/post land-cover class maps.
- PCCD can compound classification errors: a wrong pre class or wrong post class can create a false change.
- PCCD could become important if the project pivots to semantic change, object-level greenhouse/building monitoring, or xBD/xBD-S12 damage states.

How it could connect to subspaces:

- Use subspace/DS/KDS/IR-MAD/Celik maps as features for a per-date classifier.
- Build per-object or per-patch subspace descriptors, classify each date state, then compare states.
- Use post-classification outputs as a semantic interpretation layer after geometric change scoring.

Near-term decision:

```text
Do not make PCCD the immediate OSCD core.
Keep it as a semantic/object-level pivot and as a possible baseline if we adopt a dataset with pre/post class or state labels.
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
- RTW generates many Time Elastic features by repeatedly selecting `R` elements
  from a sequence while retaining temporal order, concatenating them into
  vectors in `R^(dR)`, and fitting a PCA "hypo subspace" to those vectors. Two
  sequences are compared through the singular values `kappa_i` of `X^T Y`,
  commonly summarized as `mean(kappa_i^2)`. The bundled MATLAB `TEfeatures.m`
  confirms ordered random subsequence sampling; it is not arbitrary image
  warping. For satellite use, timing/tempo tolerance must be demonstrated
  against phase-aware harmonic, Fourier, DTW/TWDTW, and non-warped controls,
  not assumed from the method name. This requires multi-date data, not OSCD's
  two dates.
- Controlled MultiSenGE result, 2026-06-21: the selected raw RTW construction
  (`R=4`, `L=64`, rank `5`) was order-sensitive and nuisance-tolerant but did
  not add beyond an order-invariant snapshot subspace on held-out structural
  changes. The pooled positive signal was dominated by a relative-band-phase
  intervention that ordinary cross-band covariance also detected. Treat RTW as
  a closed detector route for the current data, not as evidence that temporal
  ordering never matters.
- Natural-label BreizhCrops follow-up, 2026-06-21: a `snapshot subspace` means
  the centered or uncentered PCA span of date-level ten-band spectra, compared
  through MSM-style canonical correlations. The name is descriptive; it is not
  DS or a distinct named Sensei method. After adding global-shift alignment,
  correlation, PCA cross-reconstruction, shift-orbit, bandwise, seasonal,
  DTW/TWDTW, and M-SSA controls, nested-selected RTW lost on two geographic
  holdouts. PCA cross-reconstruction was strongest. This demonstrates that the
  useful signal was largely spectral-distribution/phase alignment rather than
  a capability unique to random ordered-subsequence geometry.
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

Working definition:

```text
A change map is spatially meaningful if high scores form coherent image regions
that correspond to plausible changed objects or changed land-cover areas, while
avoiding scores that are only scattered pixel noise, window/block artifacts, or
global radiometric/seasonal shifts unrelated to the target change labels.
```

For this project, "spatially meaningful" is not a vague visual compliment. It means the method passes several checks:

- Metric check: the map ranks OSCD changed pixels above unchanged pixels using AUROC/AP and improves or complements raw L2/PCA-diff.
- Region check: high-score areas are spatially connected around actual changed regions, not isolated speckles.
- Boundary check: the map is not dominated by artificial patch/window/grid boundaries.
- Pseudo-change check: high scores are not mainly clouds, shadows, seasonal vegetation, water brightness, or registration artifacts.
- Interpretation check: we can explain what sample unit produced the score, such as pixel spectra, patches, windows, or flattened band images.

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
| Band-Image DS | one full Sentinel-2 band image | `X_pre_flat, X_post_flat in R^(N_valid_pixels x 13)` | PCA in pixel-position space from 13 band-image samples per date | project the `13` band-difference images into spatial DS, then reduce projected evidence per pixel | preserves global spatial layout inside each band vector; sample-limited on Sentinel-2 because there are only 13 bands |
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

1. `phase1/scripts/inspect_oscd_subspace.py`
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

### IR-MAD versus DS

IR-MAD should be treated as a required classical comparison baseline, not a side option.

Relationship to DS:

| point | DS / patch DS | IR-MAD |
|---|---|---|
| data object | pre/post multispectral images | pre/post multispectral images |
| linear algebra family | PCA subspaces, principal vectors, Difference Subspace | CCA, MAD variates, iteratively reweighted CCA |
| sample pairing | current score uses `x_post - x_pre`; global subspace fitting treats pixels as unordered samples | paired pre/post pixels are central to the MAD variates |
| background/no-change model | implicit through PCA basis and DS score | explicit through iterative reweighting of likely unchanged observations |
| spatial information | global version loses position during fitting; patch/window variants add local support | standard IR-MAD is also mostly spectral/global unless localized or patched |
| thesis role | proposed/adapted method family | established remote-sensing baseline and CCA pressure test |

Why it matters:

- IR-MAD is close enough to DS to pressure the project fairly: both are multivariate linear representations of pre/post change.
- It is different enough that beating or complementing it would mean something: IR-MAD uses CCA/MAD and iterative reweighting, while DS uses PCA subspaces and difference directions.
- If IR-MAD outperforms DS, the project should not hide that. It would push the thesis toward spatial DS, KDS/KPCA, temporal GDS/KGDS, or an interpretability/diagnostic framing.
- If patch DS and IR-MAD fail in different places, that may support a hybrid-prior or false-positive-diagnosis contribution.

Current verification status:

- `phase1/baselines/ir_mad.py` now uses paired CCA transforms, the two cross-covariance factors in the generalized eigenproblems, sign-aligned MAD variates, variance `2(1-rho)`, and chi-square survival weights that emphasize likely unchanged pixels.
- Formula guards cover equal images and a synthetic changed block. This makes the implementation suitable for current comparison pressure, but it remains a compact project implementation rather than an independently certified reproduction of every Nielsen implementation detail.
- On all 24 OSCD cities, repaired IR-MAD reaches mean AUROC `0.8471` and AP `0.2138`; its Otsu F1 is only `0.0547`. It ranks some changes well but is poorly calibrated by per-image Otsu and often responds to broad seasonal/agricultural variation.

### Celik PCA-k-means verification status

- The old code concatenated all bands inside every `9x9` patch, producing 1,053-dimensional vectors and excessive memory use. The corrected default follows the scalar difference-image family more closely: compute CVA/L2 magnitude, extract scalar patches, fit PCA and k-means on a seeded subset, and predict in chunks.
- `multiband_patch` remains an explicit project variant; it is not the default Celik comparison.
- Synthetic changed-block and reproducibility guards pass. The implementation is still an adaptation, not a line-by-line reproduction.
- Mean all-city AP is `0.1621`, below Band-Image DS (`0.2340`). Qualitative failures include dominant water/radiometric regions that do not cover the full OSCD target.

### Rank-fusion diagnostic

For score maps `s_m`, the diagnostic fusion converts valid-pixel values to within-image percentile ranks and averages them with equal weights:

```text
f(x) = mean_m percentile_rank(s_m(x))
```

This uses no OSCD labels and avoids incompatible score scales. It is useful for testing whether methods carry complementary rankings. It is not calibrated probability fusion, and its weak Otsu result means it cannot yet be treated as a deployable binary change map.

### Split-safe changed-area calibration

To test whether threshold scale is the main bottleneck, score maps are converted into exact top-ranked-pixel decisions. For method `m`, a changed-area fraction `q_m` is selected by macro F1 on official OSCD training cities only:

```text
q_m* = argmax_q mean_city F1(top_q(score_m), label), city in OSCD train
```

The chosen `q_m*` is frozen and applied to every official test city. This is supervised calibration because it uses training labels, but it is split-safe because test labels do not select the fraction. It is a project evaluation rule, not part of DS/IR-MAD theory.

Current evidence: three-way rank fusion improves held-out mean F1 from PCA-diff's `0.2452` to `0.2670`, but the 10-city paired test is not significant. Extensive false-positive regions remain, so the next issue is nuisance/pseudo-change representation rather than another threshold search.

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

## 15. Temporal Band-Image Difference-Subspace Dynamics

### 15.1 Construction Card

| Field | Current temporal construction |
|---|---|
| Variant | full/low-rank temporal band-image subspace |
| Source | Fukui et al. 2024 second-order DS; Jang/senpai channel-flattening advice; standard Grassmann geodesic interpolation for the time-aware diagnostic |
| Sample unit | one complete aligned band image at one date |
| Date matrix | `X_t in R^(N_common_pixels x B_bands)` |
| Subspace count | one `S_t` for every date and spatial region |
| Basis | leading left singular vectors, `S_t in R^(N_common_pixels x r)` |
| Full rank | `r=B` when all independent band-image directions are retained |
| Comparison | adjacent first DS; triple second DS; geodesic along/orthogonal decomposition; irregular-time geodesic deviation |
| Spatial information | fixed pixel coordinates are ambient dimensions; they are preserved only if all dates use the same mask/alignment |
| Lost information | exact band identity is not preserved after taking the span; full-rank span is invariant to invertible mixing/scaling of its band vectors |
| Code | `phase1/subspace/temporal_band_images.py`, `phase1/subspace/second_order_ds.py`, `phase1/subspace/geodesic.py` |
| Verification | `tests/test_temporal_subspace_dynamics.py` plus controlled gain/offset/translation/local-change injections |

This is not the old OSCD pixel-spectrum construction. The matrix orientation is
the opposite:

```text
global OSCD pixel DS:     X_t in R^(B bands x N pixel samples)
temporal band-image DS:   X_t in R^(N spatial coordinates x B band-image samples)
```

The temporal construction follows Sensei's requested first trial: a small
channel-dimensional subspace in a super-high-dimensional spatial vector space.

### 15.2 First And Second Magnitudes

For orthonormal bases `S_a` and `S_b`, let singular values of
`S_a^T S_b` be `cos(theta_i)`. The first DS magnitude is:

```text
Mag(D(S_a,S_b)) = 2 * sum_i (1 - cos(theta_i)).
```

Given equally spaced `S_{t-1}, S_t, S_{t+1}`, the paper defines:

```text
D2 = D(S_t, M(S_{t-1},S_{t+1})),
```

where `M` is the principal-component/Karcher-style mean subspace obtained from
the endpoint principal vectors. The implementation uses the mathematically
equivalent principal-vector form instead of allocating an infeasible
`N_pixels x N_pixels` projector matrix.

For the decomposition:

```text
W = span(S_{t-1}, S_{t+1})
omega(S_t) = projection of S_t onto W
orthogonal component = Mag(D(S_t, omega(S_t)))
along component = Mag(D(omega(S_t), M))
```

The paper states that total magnitude is approximately the sum of these two
components and explicitly leaves a proof for future work. The project therefore
records the decomposition residual instead of asserting exact equality.

### 15.3 Irregular-Cadence Diagnostic

The paper derives second-order DS from a central difference with equal
`Delta t` and then sets `Delta t=1`. Satellite acquisitions are often irregular.
The paper-faithful number is retained, but a separate diagnostic is added:

1. Let `h_left=t-t_left` and `h_right=t_right-t`.
2. Set `alpha=h_left/(h_left+h_right)`.
3. Interpolate the endpoint subspaces along their shortest Grassmann geodesic
   to `Gamma(alpha)`.
4. Measure `rho(S_t,Gamma(alpha))` and
   `Mag(D(S_t,Gamma(alpha)))`.
5. Report the small-angle acceleration proxy
   `2*rho/(h_left*h_right)` separately.

This asks whether the observed middle subspace departs from constant-speed
endpoint motion at its actual acquisition time. It is a project adaptation,
not a redefinition of Fukui et al.'s second-order DS.

Source for geodesic interpolation: Edelman, Arias, and Smith 1998, *The
Geometry of Algorithms with Orthogonality Constraints*.

### 15.4 Spatial Attribution

For each canonical principal-vector pair `u_i,v_i`, the DS magnitude contains
`||u_i-v_i||^2`. The spatial contribution at ambient coordinate `j` is:

```text
c_j = sum_i (u_i[j] - v_i[j])^2.
```

Therefore `sum_j c_j = Mag(D)`. Reshaping `c` through the common spatial mask
produces an attribution map without pretending that the DS is a pixel-wise
classifier. These maps currently respond broadly to scene structure on IPOL
Las Vegas, motivating local/multiscale subspaces.

### 15.5 Verified Invariance And Failure Boundary

With `r=B` and centered band images:

- an invertible per-band scaling changes the basis vectors but not their span;
- centering removes a constant per-band spatial offset;
- consequently, tested global gain/offset changes produce zero or near-zero DS;
- a spatial translation changes which ambient coordinate contains each value,
  so the span changes strongly.

This explains the controlled result: radiometric invariance and translation
sensitivity are consequences of the representation, not accidental metrics.
The next method question is how to reduce the latter while preserving the
former and retaining localized-change sensitivity.

### 15.6 Bidirectional Temporal-Context Construction

The one-date construction above asks how the spatial-band span moves from one
date to the next. A second construction, motivated by the backward/forward
reference sets in Dagobert et al. 2022, compares date windows at boundary `t`.
It is an adaptation, not a reproduction of their NNLS/NFA detector.

| Field | Bidirectional temporal-context construction |
|---|---|
| Source | Fukui-Maki canonical DS; Dagobert et al. backward/forward temporal context |
| Boundary | last pre-context date `t-1`; first post-context date `t` |
| Context | `V` dates before and `V` dates after, with endpoint replication only at sequence ends |
| Per-band matrix | `X^-_(t,c), X^+_(t,c) in R^(N_common_pixels x V_dates)` |
| Joint matrix | `X^-_t, X^+_t in R^(N_common_pixels x (B_bands V_dates))` |
| Basis | leading non-degenerate left singular vectors; rank is clipped to numerical rank |
| DS score | mean canonical spatial DS contribution over band factors, or one joint contribution map |
| Linear control | first post-date projected outside the backward basis plus last pre-date projected outside the forward basis |
| Raw control | adjacent per-pixel RMS band difference |
| Preserved | pixel coordinates, temporal side of the boundary, and per-band identity in the factorized version |
| Lost | date order inside each context span and coefficient sign constraints used by IPOL NNLS |
| Code | `phase1/subspace/temporal_context.py` |
| Verification | endpoint indexing, equal contexts, persistent local change, and global gain/offset invariance in `tests/test_temporal_subspace_dynamics.py` |

The tall matrices use the snapshot identity `X^T X v=sigma^2 v`, followed by
`u=Xv/sigma`. This is mathematically the same left singular subspace as a full
SVD but avoids decomposing an `N_pixels x N_pixels` object.

The linear projection novelty score is deliberately separate from DS. It is a
signed orthogonal-subspace residual; Dagobert et al. instead fit a non-negative
cone and apply an a-contrario NFA decision. Similar behavior does not make the
two methods equivalent.

### 15.7 Current Method Boundary

Four IPOL sequences reject the present temporal-context DS map as a competitive
local detector. Its best tested configuration has much lower macro AP than raw
adjacent RMS. The linear projection novelty control is stronger: rank-2,
`V=3`, per-band context improves mean AUROC on all four sequences and nearly
matches macro AP, but only beats raw AP on Piraeus. These are agreement results
against IPOL output, not accuracy against ground truth.

Controlled MultiSenGE interventions show a narrower useful property:

- centered/L2 context geometry is effectively invariant to tested global gain
  and offset;
- temporal-context scores can distinguish a persistent injected change from
  the same one-date transient, while adjacent raw difference cannot;
- raw difference still localizes the injected region best;
- one-pixel translation produces a much larger geometric response than the
  local event.

Therefore the viable hypotheses are now split:

1. first/second/geodesic quantities as sequence-level event descriptors;
2. projection novelty as a radiometrically invariant localization mechanism;
3. registration-robust spatial support as the missing method component.

Do not call the current temporal-context DS map a validated change detector.

Registration scale-space result:

- Gaussian low-pass support reduces translation response monotonically, but
  robustness depends strongly on event size and strength.
- Global phase correlation removes tested integer shifts, but its interpolation
  and natural-context alignment reduce local-event AP.
- The ratio of low-pass to native response separates the tested global shift
  from persistent injections, but raw difference separates them too. This is a
  generic scale-space cue, not evidence that DS uniquely solves registration.
- Local deformation, parallax, seasonal boundaries, and real labels remain
  untested. Those are required before a robustness claim.

### 15.8 Seasonal Observation Subspaces

The next temporal construction uses dates, not bands or pixels, as the sample
set. For one fixed spatial support and one season/year `y`, let

```text
x_(y,m) in R^(B * N)
```

be the consistently flattened multispectral patch at composite date `m`, with
`B` bands and `N` common valid spatial locations. Stack `M` date observations:

```text
X_y = [x_(y,1), ..., x_(y,M)] in R^((B*N) x M).
```

After per-feature temporal centering and optional per-observation L2
normalization, the leading left singular vectors form the seasonal observation
subspace:

```text
S_y in Gr(r, B*N),  r <= M - 1 after centering.
```

This is a genuine image-set-style construction: the columns are repeated
observations of the same aligned region through one seasonal cycle. A sequence
`S_(y-1), S_y, S_(y+1)` supports the paper-faithful first/second DS and geodesic
decomposition already implemented in this project.

Construction card:

| Field | Definition |
|---|---|
| Variant | seasonal observation subspace |
| Source | image-set PCA/MSM convention; Kanai et al. 2023 temporal signal subspaces; Fukui et al. 2024 first/second DS |
| Sample unit | one aligned multispectral date composite of the same patch/field |
| Matrix | `X_y in R^((B*N) x M)` |
| Subspace count | one per patch/field per season or year |
| Fitting | SVD/PCA over date columns; rank bounded by valid temporal samples |
| Comparison | adjacent first DS; triple second DS; along/orthogonal and irregular-time geodesic diagnostics |
| Preserved | spatial coordinates, band identity, and seasonal observation set |
| Lost | ordering inside the season unless time/Hankel embedding is added |
| Required controls | NDVI/amplitude, raw spectral mean, minimum principal angle, MOSUM/BFAST/JUST where data length permits |
| Verification | synthetic seasonal-regime tests, no-event cycles, gain/offset, missing dates, phase shifts, gradual transitions, and real labeled/verified events |

[gap] Ordinary PCA of `X_y` is order-invariant within the season.
[why it matters] Irrigation and phenology are trajectories, not only unordered
sets; two cycles with the same span but different ordering can look identical.
[next check] Compare the snapshot subspace with a Hankel/SSA or product-Grassmann
variant after the simpler construction passes the event-alignment gate.

[gap] IrrMapper annual classes are model outputs and can inherit Landsat,
growing-season, orchard/vineyard, wetland, and weak-NDVI errors.
[why it matters] Derived start/stop transitions are weak labels, not direct event
annotations.
[next check] Manually verify selected fields in independent imagery and repeat
the main result on DynamicEarthNet or another independently labeled sequence.

Controlled result, 2026-06-20:

- The rank-1 seasonal orientation score can identify the strongest boundary
  within a synthetic event sequence, but its absolute calibration is weak.
- Feature-centered geometry rejects tested global gain/offset and phase shifts,
  yet false-alarms heavily on missing composites, gradual drift, and one-pixel
  translation.
- Singular energy and normalized singular-spectrum changes outperform DS in the
  current synthetic event task. This supports an explicit orientation-plus-
  energy representation rather than claiming DS alone contains all change
  information.
- At high noise, rank-2 second-order along change becomes competitive with NDVI
  curvature, but bootstrap intervals overlap. Treat this as a real-data
  hypothesis only.
- Date permutation leaves the ordinary seasonal subspace unchanged. Order-aware
  Hankel/SSA/product-Grassmann construction remains a justified follow-up, not a
  current contribution.

### 15.9 Order-Aware And Local Temporal Representations

The unordered matrix `X=[x_1,...,x_T]` cannot encode date order because
`span(XP)=span(X)` for a permutation matrix `P`. The order-aware adaptation now
uses

```text
h_j = [x_j; ...; x_(j+L-1)]
H_L = [h_1,...,h_(T-L+1)] in R^((D*L) x (T-L+1)).
```

Kanai et al. use a scalar SSA trajectory matrix. Replacing each scalar by one
flattened multispectral observation is this project's multivariate satellite
adaptation. First temporal differences are a simpler order-aware control.

Construction card:

| Field | Definition |
|---|---|
| Variants | unordered observations, first differences, block trajectory |
| Sample unit | one aligned date observation of one fixed spatial cell |
| Feature dimension | `D=B*N_cell`; trajectory dimension `D*L` |
| Subspace | leading left singular vectors, usually rank 1-2 |
| First comparison | canonical-angle DS magnitude / Grassmann distance |
| Second comparison | Fukui second total, along, and orthogonal magnitudes |
| Non-DS diagnostics | representation energy, normalized singular spectrum, normalized covariance operator |
| Spatial extension | independent 2x2, 4x4, or 8x8 cell subspaces |
| Preserved | fixed cell layout; trajectory variants also preserve local date neighborhoods |
| Lost | unordered variant loses order; non-overlapping cells quantize boundaries |
| Code | `phase1/subspace/temporal_trajectory.py`; MultiSenGE intervention runners |
| Verification | eight invariance/formula tests plus controlled real-background nuisance studies |

The normalized covariance diagnostic compares

```text
C_X = X X^T / trace(X X^T)
d_cov(X,Y) = ||C_X-C_Y||_F.
```

It retains orientation and relative singular energy. It is not DS and must not
be presented as Fukui's method. The implementation uses small Gram matrices so
it does not form the huge ambient covariance.

Controlled MultiSenGE findings:

- trajectory/difference subspaces correctly respond to date permutations;
- they did not outperform unordered seasonal descriptors on the current
  localization task;
- pure DS orientation discards amplitude changes that remain in the same span;
- fine local support plus Gaussian sigma 2 raised unordered eigenspectrum AP to
  `0.688`, versus NDMI `0.680` and first DS `0.456`;
- eigenspectrum gain/offset false alarms were `0.050`, but missing-composite and
  translation false alarms remained `0.358` and `0.292`;
- these are controlled interventions, not natural-change accuracy.

[gap] Does local eigenspectrum performance survive independently labeled
temporal changes rather than injected spectral modes?
[why it matters] Without this, the finding is a behavior study, not a remote-
sensing result.
[next check] Acquire one labeled DynamicEarthNet AOI or manually verify an
IrrMapper/Sentinel-2 transition slice; compare DS, eigenspectrum, NDVI/NDMI/NBR,
raw multispectral controls, and one established temporal-break baseline.

### 15.10 Rolling Local Trajectory Subspaces On SpaceNet 7

| Construction-card field | Definition |
|---|---|
| Variant | rolling local RGB trajectory subspace |
| Source/reference | SpaceNet 7/MUDS data model; project trajectory construction; paper-guided first/second DS and geodesic decomposition |
| Sample unit | one aligned RGB observation of one fixed AOI cell at one month |
| Input | `x_t in R^(3*N_cell)`; lag-two `H_t in R^((6*N_cell) x 5)` over a six-month window |
| Subspace count | one rank-two subspace per cell and rolling window |
| Basis | two leading left singular vectors of feature-centered `H_t` in the frozen confirmation run |
| Comparison | first DS/Grassmann quantities; second total, along, and orthogonal magnitudes |
| Target | first appearance of a persistent SpaceNet 7 building ID, aggregated to fixed cells |
| Preserved | within-cell RGB layout and adjacent-month order in each trajectory column |
| Lost | object identity inside the subspace; detail below cell resolution; information outside RGB |
| Code | `phase1/data/spacenet7_dataset.py`; `phase1/scripts/evaluate_spacenet7_temporal_subspaces.py` |
| Verification | loader/rasterization tests, temporal formula tests, development AOIs, and four untouched confirmation AOIs |

`labels_match_pix` uses pixel-coordinate geometries. Rasterizing those labels
with the image's projected affine transform silently produces incorrect masks;
the correct operation uses the identity affine transform. A regression test
guards this dataset-specific fact.

SpaceNet UDM polygons declare CRS84 coordinates and must be reprojected to the
image CRS before rasterization. Validity is then intersected over the dates
required by each local first/second comparison; a sequence-wide intersection
is unnecessarily destructive.

The construction is mathematically executable and formula-tested, but it
failed as a detector: confirmation AP was `0.1127`, versus `0.1910` for the
two-radiometric rank fusion. This separates implementation correctness from task
utility. Do not infer that a correct DS equation automatically defines the
right satellite sample object or score.

The component behavior is itself informative. First-DS magnitude and first
Grassmann distance had mean within-AOI Spearman correlation `0.9999`; at the
tested small rotations they provide almost the same ranking. Second total and
along had correlation `0.9596`, so total magnitude was largely governed by
motion along the estimated geodesic. Orthogonal magnitude was distinct from
raw second difference (`rho=-0.0314`) but nearly random for new-building cells
(AUROC `0.5055`). Distinctness is not evidence of task relevance.

## 16. Cross-Method Fidelity And Status

The cross-branch evidence review corrected several method-category shortcuts:

| Label | Correct status |
|---|---|
| Band-Image DS | Pairwise linear canonical DS on flattened band-image samples; **not GDS**. |
| Material common-removal experiment | GDS-inspired proxy; not a complete formal multi-subspace GDS evaluation. |
| RFF nonlinear DS | Explicit-feature kernel proxy; **not paper-faithful KDS/KGDS**. |
| Autoencoder latent subspaces | Deep-feature proxy; **not Fukui's Signal Latent Subspace**. |
| Fixed-grid DS pyramid | Spatial-support proxy; **not Green Learning, PixelHop, or wavelets**. |
| Moving-average/slow features | Temporal smoothing/SFA proxy; **not S3CCA or TRCCA**. |

Method completion as of 2026-06-22:

- **Implemented and real-tested:** canonical first DS, second DS, geodesic
  along/orthogonal decomposition, RTW, SFA, SSA/signal-subspace DS, MSM,
  IR-MAD/CCA structure, spatial Band-Image/patch/window DS.
- **Reference verified only:** Venus KDS/KGDS construction.
- **Partial/proxy only:** GDS satellite adaptation, SLS, SFS, nonlinear kernel
  DS, multiscale Green Learning.
- **Untested faithfully:** satellite KDS/KGDS, standalone KCCA, S3CCA, TRCCA,
  KMSM, product-Grassmann SLS, true wavelet/PixelHop construction.

The local HSI moment experiment provides a construction example for method
verification. For a local `w x w` window, each pixel is a `B`-band sample,
`X_t in R^((w*w) x B)`, and exact dual PCA gives `U_t in R^(B x r)`. Mean,
covariance scale, normalized eigenspectrum shape, leading-eigenspace
orientation, canonical DS projection, and projector row-energy attribution
were separated and tested with five controlled mechanisms. On Hermiston,
Farmland, and Shenzhen, every frozen geometric hypothesis lost to the strongest
direct control. The mathematics is usable; the detector hypothesis is rejected
for this construction.

General rule: a method is not credited by family name. It must preserve the
exact mathematical object, sample definition, score, and evaluation protocol
from its source, with project adaptations named separately.

### 16.1 Band-Image Matched Spatial Controls

The Band-Image DS null experiment uses `X_t in R^(N x 13)`, with rows as fixed
spatial positions and columns as band-image samples. Because
`fit_pca_basis(X_t)` fits sklearn PCA on `X_t^T`, centering subtracts the mean
of the 13 band values at each spatial position. Every matched control uses the
same centering axis.

- **Normalized spatial Gram:** row norm of
  `A_pre A_pre^T-A_post A_post^T`, where `A=X/||X||_F`. This retains all
  singular modes and removes total matrix scale.
- **Projector distance:** row norm of
  `U_pre U_pre^T-U_post U_post^T`. This isolates rank-matched orientation and
  does not use the spectral-difference vector.
- **Cross-reconstruction novelty:** symmetric cross-subspace residual minus
  each date's self-reconstruction residual. This is zero for equal subspaces
  even when rank truncation leaves ordinary reconstruction error.

The row formulas are evaluated through `13 x 13` products without forming an
`N x N` matrix. Explicit small-matrix tests confirm the Gram/projector
identities and common-permutation equivariance.

Empirical boundary after correcting all controls to use the PCA sample-centering
axis: Band-Image DS beats Gram, projector, and cross-reconstruction AP, but
loses to spatially filtered PCA. Therefore spatial orientation alone is not
the useful statistic. A DS-specific ranking contribution is isolated relative
to the matched geometric nulls, but not as an improvement over the strongest
individual spatial-PCA map.

### 16.2 xBD-S12 Transfer And Task Decomposition

The 12-band xBD-S12 adaptation uses `X_t in R^(N_spatial x 12)` and centered
rank 11. Three targets must remain separate:

1. full-scene damaged-pixel retrieval (`2-4` vs `0-1`);
2. damage discrimination inside buildings (`2-4` vs `1`);
3. building localization diagnostic (`1-4` vs `0`).

The frozen external result shows that row-wise projector distance is strongest
for targets 1 and 3, while raw spectral L2 is strongest for target 2. Thus
projector distance should be interpreted as spatial candidate/localization
evidence. It is invariant to basis rotation because it evaluates the row norm
of `P_pre-P_post`, not individual PCA vector signs.

Canonical DS is weaker than projector distance as a stand-alone xBD-S12 map.
It beats a centering- and rank-matched cross-reconstruction control on all five
unseen test events, before and after the boundary stress test, but this
direction does not repeat consistently across 11 training disasters. Treat it
as event-dependent DS evidence, not a universal transferable advantage.

The first fixed-combination hypothesis was tested and rejected:

```text
projector rank map = candidate/localization evidence [supported]
raw L2 = conditional radiometric evidence [supported]
mean/product of both maps = improved candidate score [rejected]
```

High-rank centered PCA and dual uncentered autocorrelation projectors behave
similarly; ranks 8-11 form a plateau on training events. Keep the two signals
separate until a mechanism stronger than score averaging is defined. Centered
PCA and uncentered autocorrelation remain different mathematical objects even
when their high-rank empirical behavior is similar.

On an identical deterministic 100-patch sample from each of 11 training
events, centered rank-11 projector distance exceeds IR-MAD full-scene AP by
`+0.00814` on average, interval `[+0.00470,+0.01171]`, with 10/11 wins. At a
fixed 5% pixel-review budget it retrieves 38.2% of damaged pixels with 7.64x
prevalence lift, versus IR-MAD's 30.2% and 6.03x. The unseen-event values are
24.7%/4.93x versus 17.8%/3.55x. These are candidate-ranking statistics; they
do not turn projector distance into a damage-severity classifier.

The fixed-budget metric uses the top-scoring fraction of each patch. If a
score tie crosses the budget, expected positives from the tied group are used
instead of selecting pixels by flattened index. This matters because
percentile-rank maps can contain large tied groups.

Object-level evaluation samples original xBD polygons at Sentinel-2 pixel
centers and intersects each instance with the official downsampled class mask.
At a 5% scene threshold, projector geometry has the highest damaged-building
hit recall on both train (`0.452`) and test (`0.358`) events, but also a high
intact-building hit rate (`0.377` and `0.267`). PCA-diff is better for
damaged-versus-intact object classification. The projector is therefore a
high-coverage region proposal signal rather than an object damage score.

Maximum-within-object aggregation is size-sensitive. The projector advantage
over IR-MAD persists under p90 aggregation and within event-relative object
size tertiles, including small test objects (`0.1965` versus `0.1346` p90 hit
recall at 5%), but all object-recall claims must state the size sensitivity.

Controlled registration shifts on training events show a bounded sensitivity.
At one 4 m Sentinel pixel, projector AP drops by `-0.00233` and p90 object
recall by `-0.0236`, with both event-bootstrap intervals below zero. At two
pixels the drops are `-0.00414` and `-0.0358`. Projector remains the strongest
absolute candidate method through the tested range, but degrades more clearly
than low-performing PCA/raw controls. Use "registration-sensitive candidate
geometry," never "registration-invariant geometry."

## 17. Cross-Sensor Band-Image Boundary

The same input convention can produce materially different objects as the
number of band-image samples changes:

```text
xBD-S12:   X_t in R^(16384 x 12), centered rank 11
HSI:       X_t in R^(N x 149..198), fixed transfer rank 11
SpaceNet7: X_t in R^(tile_pixels x 3), centered rank 2
```

On HSI, canonical DS can isolate a useful spatial mode on Hermiston but not on
Benton or Shenzhen. On SpaceNet7, three RGB samples create a severe rank-two
bottleneck: cross-reconstruction is more useful than DS, while projector
distance loses to IR-MAD. Therefore "preserves spatial coordinates" is not
sufficient. The sample count, retained spectral diversity, rank, scene-change
mechanism, and score semantics determine whether geometry is useful.

The cross-sensor result forbids treating projector distance as a generic
building-localization method. Its xBD behavior is currently specific to that
12-band event task. Canonical DS and projector distance also remain distinct:
DS projects the observed date difference onto principal-vector difference
directions, while projector distance measures row-wise change of the fitted
subspace operators without using the observed difference vector.
