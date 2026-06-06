# Method Notes

This file explains how the current project works and what each method means. It is intentionally direct and code-facing.

## Current Project Scope

Current implemented core:

```text
OSCD Sentinel-2 binary change detection
pre image + post image -> Phase 1 unsupervised priors -> Phase 2 supervised segmentation
```

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

## Archive-Ingested Method Caveats

These caveats were extracted from the old archive docs and should stay visible because they affect thesis claims and future code work.

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

### Split and evaluation caveats

- OSCD has train/test labels. The project creates its own validation split by holding out train cities. Do not call the validation set "official" unless that split is externally verified.
- Current final Phase 2 evaluation should use stitched city-level masks, not averaged patch metrics, for thesis claims.
- Default Phase 2 evaluation threshold is often `0.5`; threshold tuning and probability-map analysis are separate tasks.

### Loader and model caveats

- `OSCDSegmentationDataset` uses the configured/default 13-band order. Future configs should make band order explicit to avoid silent assumptions.
- `valid_mask = valid_pre AND valid_post` is meant to avoid nodata and border artifacts, but its impact on labeled changed pixels must be measured.
- `PriorsFusionUNet` is a lightweight early-fusion/reweighting variant, not a full attention or sophisticated fusion architecture.
- A priors-only configuration is unsafe unless the dataset/model path is audited; current code paths were mainly exercised with raw channels present.

### MultiSenGE caveat

MultiSenGE remains exploratory. Before using it for GDS/KGDS, audit the manifest and path assumptions, especially Sentinel-1 paths. Old notes flagged a possible local-path mismatch around `s1/<name>.tif` versus nested `s1/s1/<name>.tif`.
