# OSCD Spatial DS And Classical Baseline Pressure Test

## 1. Research Question

Can a spatially aware Difference Subspace (DS) construction produce useful pixel-level changed-area evidence on 13-band OSCD Sentinel-2 pairs, and where does it help or fail against established classical baselines?

This report tests ranking and binary-segmentation evidence. It does not claim disaster-damage recognition, semantic change understanding, or superiority to modern neural change detectors.

## 2. Source-To-Code Checks

### IR-MAD

The old local implementation was not acceptable for comparison: it used one transform for both dates, omitted one cross-covariance factor from the generalized eigenproblem, and assigned larger iterative weights to likely changed pixels.

The repaired implementation in `phase1/baselines/ir_mad.py` now uses paired CCA transforms, MAD variates, variance `2(1-rho)`, and chi-square survival weights so likely unchanged pixels dominate the next covariance estimate. Sources: [Nielsen 2007](https://doi.org/10.1109/TIP.2006.888195) and the [Google Earth Engine iMAD tutorial](https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt2).

### Celik PCA-KMeans

The old adaptation concatenated every band inside every `9x9` patch, producing 1,053-dimensional patch vectors and a very large matrix. The corrected default first computes the scalar CVA/L2 difference magnitude, extracts local scalar patches, fits PCA and k-means on a seeded subset, and predicts all valid patches in chunks. This is closer to the difference-image pipeline in [Celik 2009](https://ieeexplore.ieee.org/document/4802550). It remains a project adaptation, not a line-by-line reproduction.

Formula guards now cover equal-image IR-MAD, synthetic changed-block IR-MAD, synthetic changed-block Celik, deterministic Celik fitting, rank fusion, and equal-image pyramid DS.

## 3. Protocol

- Dataset: all 24 locally available OSCD cities.
- Input: aligned pre/post 13-band `.tif` stacks, globally normalized using the existing OSCD band statistics.
- Task: pixel-level binary change ranking and thresholded segmentation against OSCD labels.
- Fixed seed: `1234`.
- Main DS rank: `8`; Band-Image sensitivity additionally tested ranks `2,3,4,5,6,8,10,12` on core cities and ranks `10,12` on all cities.
- Metrics: AUROC, average precision (AP), best F1/IoU, Otsu F1/IoU, raw-L2 correlation, runtime, city-wise wins, bootstrap confidence intervals, and paired Wilcoxon tests.
- No per-city method tuning or label-fitted fusion weights.

Primary all-city output:

`phase1/outputs/spatial_ds_traditional_pressure_allcities_corrected_20260618_174228/`

## 4. Corrected All-City Results

| method | mean AUROC | mean AP | mean best F1 | mean Otsu F1 | mean raw-L2 corr. |
|---|---:|---:|---:|---:|---:|
| PCA-diff | 0.8392 | **0.2541** | **0.3076** | **0.2160** | 0.8588 |
| Band-Image DS norm, rank 8 | 0.8412 | 0.2340 | 0.2928 | 0.2007 | 0.7775 |
| raw L2/CVA | 0.7717 | 0.2261 | 0.2873 | 0.1802 | 1.0000 |
| repaired IR-MAD | **0.8471** | 0.2138 | 0.2846 | 0.0547 | 0.5089 |
| Celik PCA-k-means adaptation | 0.6867 | 0.1621 | 0.2380 | 0.1857 | 0.4283 |
| patch5 DS | 0.7119 | 0.1331 | 0.1952 | 0.0663 | 0.4614 |
| patch3 DS | 0.7021 | 0.1185 | 0.1797 | 0.0465 | 0.4682 |
| global pixel DS | 0.6270 | 0.0625 | 0.1093 | 0.0247 | 0.3438 |
| local window DS | 0.6302 | 0.0598 | 0.1055 | 0.0358 | 0.3278 |

Band-Image DS versus pressure baselines, paired over 24 cities by AP:

| comparison | mean AP delta | 95% bootstrap CI | wins/losses | Wilcoxon p | interpretation |
|---|---:|---:|---:|---:|---|
| Band-Image minus PCA-diff | -0.0201 | [-0.0365, -0.0047] | 6/18 | 0.0249 | significantly worse |
| Band-Image minus raw L2 | +0.0079 | [-0.0100, +0.0248] | 15/9 | 0.2076 | no reliable difference |
| Band-Image minus IR-MAD | +0.0202 | [-0.0206, +0.0569] | 16/8 | 0.1355 | no reliable difference |
| Band-Image minus Celik | +0.0720 | [+0.0389, +0.1073] | 21/3 | 0.0003 | significantly better |

PCA-diff is the strongest single method by AP in 11 cities, Band-Image DS in 4, IR-MAD in 4, Celik in 2, and raw L2, patch5, and local-window DS in one city each.

## 5. Band-Image Rank Sensitivity

Core-five AP increased from `0.2869` at rank 2 to `0.3987` at rank 12. On all 24 cities:

| rank | mean AUROC | mean AP | mean best F1 | mean Otsu F1 |
|---:|---:|---:|---:|---:|
| 8 | 0.8412 | 0.2340 | 0.2928 | 0.2007 |
| 10 | 0.8463 | 0.2392 | 0.3004 | 0.2036 |
| 12 | **0.8477** | **0.2410** | **0.3021** | **0.2036** |

Rank 12 narrows the PCA-diff AP gap to `-0.0130`, wins 8 cities and loses 16, with paired `p=0.0646`. It remains below PCA-diff on mean AP.

Interpretation: Band-Image DS is sensitive to retained rank and performs best near the maximum centered rank of 12. That weakens a compact-difference-subspace interpretation: much of its useful signal may come from retaining broad band-image variation.

Rank output:

`phase1/outputs/band_image_ds_rank10_12_allcities_20260618_175418/`

## 6. Label-Free Rank Fusion

To test complementarity without supervised fitting, each component map was converted to within-image percentile ranks and combined with equal weights.

| method | mean AUROC | mean AP | mean best F1 | mean Otsu F1 |
|---|---:|---:|---:|---:|
| PCA + Band-Image + IR-MAD rank fusion | **0.8708** | **0.2674** | **0.3257** | 0.1084 |
| PCA + Band-Image rank fusion | 0.8562 | 0.2592 | 0.3159 | 0.1065 |
| Band-Image + IR-MAD rank fusion | 0.8679 | 0.2569 | 0.3181 | 0.1026 |
| PCA + IR-MAD rank fusion | 0.8667 | 0.2564 | 0.3129 | 0.1076 |
| PCA-diff | 0.8406 | 0.2541 | 0.3074 | **0.2164** |

The three-way fusion improves AUROC over PCA-diff by `+0.0301`, wins 21/24 cities, and is significant by paired Wilcoxon test (`p=0.00024`). AP improves by `+0.0133`, wins 14/24 cities, but is not significant (`p=0.1780`; bootstrap CI crosses zero). Otsu F1 degrades strongly.

Allowed interpretation: the three score families contain complementary ranking information. Forbidden interpretation: the fusion solves binary segmentation. Score calibration and threshold selection remain unresolved.

Fusion output:

`phase1/outputs/spatial_score_rank_fusion_allcities_20260618_175956/`

## 7. Multiscale Fixed-Grid Pyramid Decision

The planned whole-image, `2x2`, `4x4`, and optional `8x8` hierarchy was implemented using canonical pixel-spectral DS in every grid cell. It is a spatial-support experiment inspired by the project note; it is not PixelHop, Green Learning, or a wavelet transform.

Core-five mean AP:

| method | mean AP |
|---|---:|
| global pixel DS | 0.0791 |
| pyramid `1,2,4` squared energy | 0.0765 |
| pyramid `1,2,4` norm | 0.0762 |
| pyramid `1,2,4,8` norm | 0.0761 |

Decision: stop this exact fixed-grid spectral-pyramid branch. It does not improve ranking over global pixel DS. A future Green Learning/PixelHop or wavelet experiment must change the feature representation, not merely repeat global pixel DS inside grid cells.

Pyramid output:

`phase1/outputs/spatial_pyramid_core5_decision_20260618_180652/`

## 8. Qualitative Failure Modes

- Beirut: Band-Image DS resembles PCA/raw spectral maps and still responds broadly to urban radiometric differences; it does not isolate only labeled objects.
- Norcia: IR-MAD ranks labeled changes strongly but also emphasizes broad agricultural/seasonal variation; Otsu calibration remains poor.
- Brasilia: patch5 can localize useful regions, while Celik can concentrate on a dominant water/river change that is not the complete annotation target.
- Across cities, high AUROC can coexist with very low AP or Otsu F1 because changed pixels are rare and radiometric pseudo-change is widespread.

Representative grids are under each run's `comparison_grid.png`. Aggregate figures:

- `phase1/outputs/spatial_ds_traditional_pressure_allcities_corrected_20260618_174228/sweep_mean_ap_auroc.png`
- `phase1/outputs/spatial_ds_traditional_pressure_allcities_corrected_20260618_174228/sweep_city_average_precision_heatmap.png`
- `phase1/outputs/spatial_score_rank_fusion_allcities_20260618_175956/sweep_mean_ap_auroc.png`

## 9. Research Decision

1. Do not claim Band-Image DS beats PCA-diff as a standalone detector.
2. Retain Band-Image DS rank 12 as the best current DS-family spatial construction and as an interpretable component map.
3. Retain repaired IR-MAD as required classical comparison pressure.
4. Treat equal-weight rank fusion as evidence of complementary ranking, not a finished segmentation method.
5. Stop fixed-grid pixel-spectral pyramid work in its current form.
6. Next experiments should target the observed real problem: pseudo-change suppression and score calibration, with split-safe thresholding and a feature representation that can distinguish radiometric nuisance from target change.

## 10. Reproduction Commands

Traditional pressure comparison:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank8_traditional_pressure:8:global_pixel+patch3+patch5+window128s64mean+band_image_norm+celik_pca_kmeans+ir_mad" --output-dir "phase1/outputs/spatial_ds_traditional_pressure_allcities_corrected_$tag" --continue-on-error --no-save-npy
```

Rank-12 fusion comparison:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank12_label_free_fusion:12:band_image_norm+ir_mad+rank_fusion_pca_band+rank_fusion_band_irmad+rank_fusion_pca_irmad+rank_fusion_pca_band_irmad" --output-dir "phase1/outputs/spatial_score_rank_fusion_allcities_$tag" --continue-on-error --no-save-npy
```
