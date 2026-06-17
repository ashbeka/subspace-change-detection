# OSCD Band-Image DS Score Ablation

Date: 2026-06-18

## 1. Purpose

Test whether the weak threshold behavior of the previous flattened-band result was caused by the final score reduction, not by the Band-Image DS construction itself.

Renamed method:

```text
old alias: flatbands
active name: band_image_ds
formal description: Band-Image Difference Subspace
```

Construction:

```text
one sample = one Sentinel-2 band image flattened over valid pixel positions
X_pre, X_post in R^(N_valid_pixels x 13)
PCA basis lives in valid-pixel spatial position space
DS compares the pre-date and post-date band-image subspaces
```

## 2. Commands

Smoke test with attribution:

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city beirut --rank 8 --methods band_image_ds,band_image_norm,band_image_ratio,band_image_residual --output-dir "phase1/outputs/spatial_ds_band_image_beirut_smoke_$tag" --save-band-attribution --no-save-npy
```

Core5 sweep:

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --configs "rank8_band_image_scores:8:global_pixel+patch3+patch5+window128s64mean+band_image_ds+band_image_norm+band_image_ratio+band_image_residual" --output-dir "phase1/outputs/spatial_ds_band_image_score_ablation_core5_$tag" --continue-on-error --no-save-npy
```

All-city sweep:

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank8_band_image_scores:8:global_pixel+patch3+patch5+window128s64mean+band_image_ds+band_image_norm+band_image_ratio+band_image_residual" --output-dir "phase1/outputs/spatial_ds_band_image_score_ablation_allcities_$tag" --continue-on-error --no-save-npy
```

Targeted attribution maps:

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'; foreach ($city in @('chongqing','nantes','bordeaux','norcia')) { .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city $city --rank 8 --methods patch5,window128s64mean,band_image_ds,band_image_norm,band_image_ratio,band_image_residual --output-dir "phase1/outputs/spatial_ds_band_image_attribution_${city}_$tag" --save-band-attribution --no-save-npy }
```

## 3. Score Definitions

Let `delta` be the 13-column band-difference image matrix after valid pixels are flattened, and let `P_D` be projection onto the Band-Image DS basis.

| method | score |
|---|---|
| `band_image_ds` | `sum_bands (P_D delta_band)^2` |
| `band_image_norm` | `sqrt(sum_bands (P_D delta_band)^2)` |
| `band_image_ratio` | `sum_bands (P_D delta_band)^2 / (sum_bands delta_band^2 + 1e-8)` |
| `band_image_residual` | `sum_bands (delta_band - P_D delta_band)^2` |

`band_image_norm` is monotonic with `band_image_ds`, so AUROC, AP, best F1, and best IoU should be nearly identical. Otsu thresholding can change because Otsu uses the score histogram scale.

## 4. Core5 Results

Output:

```text
phase1/outputs/spatial_ds_band_image_score_ablation_core5_20260618_021813/
```

Mean over Beirut, Dubai, Las Vegas, Milano, and Norcia:

| method | mean AUROC | mean AP | mean best F1 | mean Otsu F1 |
|---|---:|---:|---:|---:|
| `pca_diff` | 0.8733 | 0.3953 | 0.4112 | 0.3403 |
| `band_image_ds` | 0.8761 | 0.3726 | 0.3961 | 0.1673 |
| `band_image_norm` | 0.8761 | 0.3726 | 0.3961 | 0.3122 |
| `raw_l2` | 0.8263 | 0.3550 | 0.3849 | 0.2707 |
| `patch5` | 0.7783 | 0.1966 | 0.2586 | 0.0998 |
| `patch3` | 0.7920 | 0.1897 | 0.2572 | 0.0815 |
| `band_image_residual` | 0.5879 | 0.0900 | 0.1557 | 0.0361 |
| `band_image_ratio` | 0.7303 | 0.0842 | 0.1630 | 0.0820 |
| `global_pixel` | 0.6471 | 0.0791 | 0.1236 | 0.0096 |
| `window128s64mean` | 0.6353 | 0.0729 | 0.1177 | 0.0458 |

Interpretation:

- `band_image_norm` keeps the same ranking metrics as `band_image_ds`.
- `band_image_norm` substantially improves Otsu F1 on core5.
- PCA-diff remains stronger on mean AP and mean Otsu F1.

## 5. All-City Results

Output:

```text
phase1/outputs/spatial_ds_band_image_score_ablation_allcities_20260618_022314/
```

Mean over all 24 local OSCD cities:

| method | mean AUROC | mean AP | mean best F1 | mean Otsu F1 | mean Otsu IoU | mean raw-L2 corr |
|---|---:|---:|---:|---:|---:|---:|
| `pca_diff` | 0.8392 | 0.2541 | 0.3076 | 0.2160 | 0.1375 | 0.8588 |
| `band_image_ds` | 0.8412 | 0.2340 | 0.2928 | 0.1129 | 0.0672 | 0.6596 |
| `band_image_norm` | 0.8412 | 0.2340 | 0.2928 | 0.2007 | 0.1242 | 0.7775 |
| `raw_l2` | 0.7717 | 0.2261 | 0.2873 | 0.1802 | 0.1141 | 1.0000 |
| `patch5` | 0.7119 | 0.1331 | 0.1952 | 0.0663 | 0.0365 | 0.4614 |
| `patch3` | 0.7021 | 0.1185 | 0.1797 | 0.0465 | 0.0251 | 0.4682 |
| `band_image_residual` | 0.5766 | 0.0712 | 0.1203 | 0.0555 | 0.0311 | 0.5147 |
| `global_pixel` | 0.6270 | 0.0625 | 0.1093 | 0.0247 | 0.0134 | 0.3438 |
| `window128s64mean` | 0.6302 | 0.0598 | 0.1055 | 0.0358 | 0.0192 | 0.3278 |
| `band_image_ratio` | 0.7037 | 0.0586 | 0.1170 | 0.0773 | 0.0426 | -0.0510 |

Best AP winner counts over 24 cities:

```text
all methods:
  pca_diff 15, band_image_ds 4, patch5 2, raw_l2 1, window128s64mean 1, band_image_ratio 1

DS-family only:
  band_image_ds 15, band_image_norm 5, patch5 2, window128s64mean 1, band_image_ratio 1
```

## 6. Main Finding

`band_image_norm` is the best active Band-Image DS score for thresholded maps:

- It keeps the same AUROC/AP as `band_image_ds`.
- It raises all-city Otsu F1 from `0.1129` to `0.2007`.
- It nearly reaches PCA-diff Otsu F1 (`0.2160`) and beats raw L2 Otsu F1 (`0.1802`).
- It still does not beat PCA-diff on mean AP (`0.2340` vs `0.2541`).

Therefore the score reduction was part of the problem, but not the whole problem.

## 7. Spatially Meaningful Interpretation

Supported:

- Band-Image DS is much stronger than global pixel DS, patch DS, and local-window DS on all-city mean AP.
- The construction is not just raw spectral L2; `band_image_ds` raw-L2 correlation is `0.6596`, below PCA-diff's `0.8588`.
- The norm score gives a more usable unsupervised threshold than squared energy.

Not supported:

- It is not an overall detector winner over PCA-diff.
- `band_image_ratio` and `band_image_residual` are weak as primary change scores.
- This does not justify Phase 2 U-Net training yet.

## 8. Visual Outputs

All-city comparison grids:

```text
phase1/outputs/spatial_ds_band_image_score_ablation_allcities_20260618_022314/runs/*/comparison_grid.png
```

Targeted attribution folders:

```text
phase1/outputs/spatial_ds_band_image_attribution_chongqing_20260618_022904/
phase1/outputs/spatial_ds_band_image_attribution_nantes_20260618_022904/
phase1/outputs/spatial_ds_band_image_attribution_bordeaux_20260618_022904/
phase1/outputs/spatial_ds_band_image_attribution_norcia_20260618_022904/
```

Each folder contains:

```text
comparison_grid.png
band_attribution_band_image_ds.png
band_attribution_band_image_ds.csv
band_attribution_band_image_norm.png
band_attribution_band_image_norm.csv
band_attribution_band_image_ratio.png
band_attribution_band_image_ratio.csv
band_attribution_band_image_residual.png
band_attribution_band_image_residual.csv
```

## 9. Next Decision

Do not run another neural sweep yet.

Next method step:

```text
Band-Image DS norm map -> qualitative failure review -> Celik/IR-MAD formula audit -> same 24-city comparison
```

If Celik or IR-MAD beats Band-Image DS clearly, the thesis should treat Band-Image DS as a useful diagnostic/adaptation study, not the final detector.

If Band-Image DS has clearer spatial interpretation in selected cities, continue with:

```text
band-group attribution -> local/multiscale band-image DS -> pseudo-change failure taxonomy
```
