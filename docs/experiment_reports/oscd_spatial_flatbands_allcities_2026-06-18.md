# OSCD Spatial DS Sweep: Flattened-Band Candidate Across All Cities

Date: 2026-06-18

Tracked interpretation of ignored output folder:

```text
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/
```

## 1. Purpose

Test Senpai/Jang's suggested opposite sample definition for subspace construction:

```text
current global pixel DS: X in R^(13 x N_pixels)
  one sample = one pixel's 13-band vector

flattened-band DS: X in R^(N_pixels x 13)
  one sample = one full Sentinel-2 band image flattened over valid pixels
```

This directly targets Sensei's concern that global pixel DS breaks spatial information during PCA fitting. It is still a Phase 1 score-map experiment only. It does not prove that a U-Net will improve if this map is used as a prior channel.

## 2. Command

One-line command used:

```powershell
$cities = 'abudhabi,aguasclaras,beihai,beirut,bercy,bordeaux,brasilia,chongqing,cupertino,dubai,hongkong,lasvegas,milano,montpellier,mumbai,nantes,norcia,paris,pisa,rennes,rio,saclay_e,saclay_w,valencia'; $tag = Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities $cities --configs "rank6_flatbands:6:global_pixel+patch3+patch5+window128s64mean+flatbands;rank8_flatbands:8:global_pixel+patch3+patch5+window128s64mean+flatbands" --output-dir "phase1/outputs/spatial_ds_allcities_flatbands_$tag" --continue-on-error
```

## 3. Sweep Design

Cities:

```text
abudhabi, aguasclaras, beihai, beirut, bercy, bordeaux, brasilia, chongqing,
cupertino, dubai, hongkong, lasvegas, milano, montpellier, mumbai, nantes,
norcia, paris, pisa, rennes, rio, saclay_e, saclay_w, valencia
```

Configurations:

| config | rank | methods |
|---|---:|---|
| rank6_flatbands | 6 | global_pixel, patch3, patch5, window128s64mean, flatbands |
| rank8_flatbands | 8 | global_pixel, patch3, patch5, window128s64mean, flatbands |

Baselines included in every run:

```text
raw_l2
pca_diff
```

## 4. Flattened-Band Construction

For each city/date:

```text
X_pre_flat, X_post_flat in R^(N_valid_pixels x 13)
```

Each column is one full Sentinel-2 band image flattened over valid pixel locations. PCA therefore learns a basis in pixel-position space from 13 band-image samples. With Sentinel-2, the practical centered PCA rank is capped at 12, so ranks 6 and 8 are valid but sample-limited.

Score definition:

```text
D = canonical Difference Subspace between pre/post flattened-band PCA bases
Delta = X_post_flat - X_pre_flat
projected = D D^T Delta
score(pixel i) = sum over 13 bands of projected(i, band)^2
```

Spatial status:

- Preserves spatial layout inside each band-image vector.
- Uses only 13 band samples, so the basis may describe band-to-band spatial similarity rather than changed-area evidence.
- Must be judged by labels, maps, baselines, and failure modes, not by the construction alone.

## 5. Mean Results Across 24 Cities

Sorted by mean average precision.

| config | rank | method | mean AUROC | mean AP | mean best F1 | mean Otsu F1 | mean raw-L2 corr |
|---|---:|---|---:|---:|---:|---:|---:|
| rank8_flatbands | 8 | pca_diff | 0.8392 | 0.2541 | 0.3076 | 0.2160 | 0.8588 |
| rank6_flatbands | 6 | pca_diff | 0.8370 | 0.2535 | 0.3068 | 0.2140 | 0.8564 |
| rank8_flatbands | 8 | flatbands | 0.8412 | 0.2340 | 0.2928 | 0.1129 | 0.6596 |
| rank6_flatbands | 6 | flatbands | 0.8313 | 0.2317 | 0.2913 | 0.1093 | 0.6634 |
| rank6_flatbands | 6 | raw_l2 | 0.7717 | 0.2261 | 0.2873 | 0.1802 | 1.0000 |
| rank8_flatbands | 8 | raw_l2 | 0.7717 | 0.2261 | 0.2873 | 0.1802 | 1.0000 |
| rank8_flatbands | 8 | patch5 | 0.7119 | 0.1331 | 0.1952 | 0.0663 | 0.4614 |
| rank6_flatbands | 6 | patch5 | 0.7002 | 0.1293 | 0.1884 | 0.0811 | 0.4686 |
| rank8_flatbands | 8 | patch3 | 0.7021 | 0.1185 | 0.1797 | 0.0465 | 0.4682 |
| rank6_flatbands | 6 | patch3 | 0.6896 | 0.1184 | 0.1818 | 0.0602 | 0.4639 |
| rank6_flatbands | 6 | global_pixel | 0.6310 | 0.0725 | 0.1251 | 0.0272 | 0.3717 |
| rank6_flatbands | 6 | window128s64mean | 0.6300 | 0.0658 | 0.1184 | 0.0345 | 0.3600 |
| rank8_flatbands | 8 | global_pixel | 0.6270 | 0.0625 | 0.1093 | 0.0247 | 0.3438 |
| rank8_flatbands | 8 | window128s64mean | 0.6302 | 0.0598 | 0.1055 | 0.0358 | 0.3278 |

## 6. Winner Counts

Best method by AP across all methods and all 48 city/rank runs:

| method | wins |
|---|---:|
| pca_diff | 30 |
| flatbands | 11 |
| raw_l2 | 3 |
| patch5 | 2 |
| patch3 | 1 |
| window128s64mean | 1 |

Best DS-family method by AP across all 48 city/rank runs:

| method | wins |
|---|---:|
| flatbands | 44 |
| patch5 | 2 |
| patch3 | 1 |
| window128s64mean | 1 |

## 7. Interpretation

[experiment-evidence] The flattened-band spatial construction is the strongest DS-family method in this sweep. It wins 44 of 48 DS-family city/rank comparisons and substantially improves over global pixel DS.

[experiment-evidence] It is not yet a performance winner over PCA-diff. Rank-8 flatbands has mean AP `0.2340`, while rank-8 PCA-diff has mean AP `0.2541`.

[experiment-evidence] Rank-8 flatbands has slightly higher mean AUROC than PCA-diff (`0.8412` vs `0.8392`) but lower AP and lower Otsu F1. This means it can rank changed pixels well in some settings, but its score calibration and high-precision threshold behavior are weaker.

[experiment-evidence] The mean raw-L2 correlation of flatbands is around `0.66`, lower than PCA-diff's `0.86`. This suggests the flattened-band map is not just a raw spectral L2 duplicate.

[risk] Because there are only 13 Sentinel-2 band samples, the flattened-band PCA basis is sample-limited. It may become more natural on hyperspectral imagery, but for Sentinel-2 it must stay framed as a tested adaptation, not a paper-faithful DS default.

[risk] Qualitative maps show that flatbands can still highlight seasonal, radiometric, or land-cover shifts that are not OSCD target changes. This keeps pseudo-change reduction as the next real bottleneck.

Safe claim:

```text
Changing the subspace sample unit from unordered pixel spectra to flattened band images makes DS much more competitive on OSCD score maps, but PCA-diff still remains the stronger average-precision baseline.
```

Forbidden claim:

```text
Flattened-band DS solves spatial change detection or should replace PCA-diff.
```

## 8. Immediate Next Decision

Do not run a long U-Net sweep yet.

Next method work should test whether flatbands is failing because of score calibration, pseudo-change sensitivity, or the sample-limited basis:

1. Add DS score-definition ablations:
   - squared projection energy;
   - unsquared projection norm;
   - normalized projection ratio `||D^T delta||^2 / ||delta||^2`;
   - robust/per-city score normalization.
2. Add fair classical pressure:
   - verify and run Celik PCA-kmeans;
   - verify and run IR-MAD on the same city set.
3. Inspect city-level qualitative maps:
   - cases where flatbands wins: aguasclaras, bordeaux, chongqing rank8, cupertino rank6, milano rank8, paris, saclay;
   - cases where PCA-diff still wins: Beirut, Dubai, Las Vegas, Montpellier, Mumbai, Nantes, Rio.
4. Only after that, decide whether a Phase 2 prior-channel rerun is justified.

## 9. Files

Ignored output files:

```text
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/sweep_metrics_all.csv
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/sweep_summary_by_config_method.csv
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/sweep_best_ap_by_city_config.csv
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/sweep_best_ds_ap_by_city_config.csv
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/sweep_report.md
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/runs/
```

Representative qualitative grids inspected:

```text
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/runs/rank8_flatbands__chongqing/comparison_grid.png
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/runs/rank6_flatbands__beihai/comparison_grid.png
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/runs/rank8_flatbands__norcia/comparison_grid.png
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/runs/rank8_flatbands__saclay_e/comparison_grid.png
```
