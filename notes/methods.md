# Method Notes

## Table Of Contents

- [Current Project Scope](#current-project-scope)
- [Data](#data)
- [Phase 1: Prior Maps](#phase-1-prior-maps)
- [Spectral Change Versus Semantic Change](#spectral-change-versus-semantic-change)
- [Phase 2: Supervised Segmentation](#phase-2-supervised-segmentation)
- [DS In The Current OSCD Adaptation](#ds-in-the-current-oscd-adaptation)
- [Corrected DS Implementation](#corrected-ds-implementation)
- [PCA Rank 6](#pca-rank-6)
- [DS, GDS, KDS, KGDS](#ds-gds-kds-kgds)
- [Future Method Hooks](#future-method-hooks)
- [Venus KDS/KGDS Audit](#venus-kdskgds-audit)
- [Spatial Information Problem](#spatial-information-problem)
- [Projection Back To Image Space](#projection-back-to-image-space)
- [Method Caveats](#method-caveats)
  - [Subspace code reading path](#subspace-code-reading-path)
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

This file explains how the current project works and what each method means. It is intentionally direct and code-facing.

## Current Project Scope

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

## Data

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

## Phase 1: Prior Maps

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

## Spectral Change Versus Semantic Change

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

## Phase 2: Supervised Segmentation

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

## DS In The Current OSCD Adaptation

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

## Corrected DS Implementation

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

## PCA Rank 6

Rank 6 means PCA keeps 6 basis directions in the 13-band feature space.

It does not reduce 1,262,600 pixels to 6 pixels. The pixels remain samples. PCA learns 6 directions that summarize major variation among the 13-band pixel vectors.

This rank is currently a practical hyperparameter, not a proven theoretical optimum.

Action:

- Test ranks `2, 3, 4, 5, 6, 8, 10, 12`.
- Test variance thresholds such as `95%`, `99%`, and `99.5%`.

## DS, GDS, KDS, KGDS

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

## Future Method Hooks

Source: `research-notes/` ingestion on 2026-06-07. These are preserved as method hooks, not active evidence.

Multi-date subspace methods:

- First-order DS is the pairwise before/after setting.
- Second-order DS is the natural next subspace idea for progression/recovery, but it requires at least three meaningful time points or period subspaces.
- Period-subspace DS should use multiple images per side when available, for example early-month versus late-month, first half-year versus second half-year, or pre-event window versus post-event window.
- SSA, SFA, DMD, RTW, and Fourier time-series analysis are future temporal tools. They need enough temporal depth and a clear target before implementation.

Spatial and interpretability variants:

- Sliding/local-window DS was already proposed in old specs and now directly matches Sensei's spatial-information critique.
- Band-group DS and per-band attribution can help explain whether DS is driven by visible, red-edge, NIR, SWIR, or atmospheric bands.
- PCA reconstruction error can be a baseline or diagnostic: fit a pre/no-change PCA model and use post reconstruction residuals as change/anomaly evidence.
- Coordinate- or patch-aware variants are method candidates only if they improve spatial meaning without turning into arbitrary feature engineering.

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

## Venus KDS/KGDS Audit

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

## Spatial Information Problem

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

Implementation details to preserve:

- `3x3` patch vectors have `13 x 3 x 3 = 117` values.
- `5x5` patch vectors have `13 x 5 x 5 = 325` values.
- The DS score for patch-vector DS should be assigned to the center pixel of each patch.
- Patch borders and valid masks must be handled explicitly; do not silently drop border regions without reporting the count.
- Full-image patch extraction can be large, so sampled fitting is acceptable if the sampling seed, sample count, and city list are recorded.
- Local-window DS should save both per-city metrics and maps, and should inspect boundary artifacts from overlapping-window aggregation.
- Audit whether any existing `sliding_window_ds` code path uses corrected `canonical`/`eig` DS or still behaves like legacy residual-stack DS before using it as evidence.

## Projection Back To Image Space

Current OSCD DS does not reconstruct a 13-band image.

It computes one scalar score per pixel and reshapes scores back to the image grid.

Possible interpretation task:

```text
delta_x      = x_post - x_pre
delta_x_ds   = D D^T delta_x
residual     = delta_x - delta_x_ds
```

This could show which band combinations DS emphasizes and may help explain projection to Sensei/senpais.

## Method Caveats

These caveats were extracted from the old archive docs and should stay visible because they affect thesis claims and future code work.

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
