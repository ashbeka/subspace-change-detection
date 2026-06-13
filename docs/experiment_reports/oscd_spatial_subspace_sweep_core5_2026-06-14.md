# OSCD Spatial Subspace Sweep: Core5 Cities

Date: 2026-06-14
Tracked interpretation of ignored output folder:

```text
phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/
```

## 1. Purpose

Test whether spatial support improves Difference Subspace (DS) score maps on OSCD, instead of relying on the one-city Beirut run.

This is a Phase 1 score-map experiment only. It does not prove that a prior will improve U-Net segmentation.

## 2. Command

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --output-dir phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823 --no-save-npy --continue-on-error
```

## 3. Sweep Design

Cities:

```text
beirut, dubai, lasvegas, milano, norcia
```

Configurations:

| config | rank | methods |
|---|---:|---|
| rank4_core | 4 | global_pixel, patch3, patch5 |
| rank6_spatial | 6 | global_pixel, window128, patch3, patch5 |
| rank8_core | 8 | global_pixel, patch3, patch5 |

Baselines included in every run:

```text
raw_l2
pca_diff
```

## 4. Mean Results Across 5 Cities

Sorted by mean average precision.

| config | rank | method | mean AUROC | mean AP | mean Otsu F1 | mean best F1 |
|---|---:|---|---:|---:|---:|---:|
| rank8_core | 8 | pca_diff | 0.8733 | 0.3953 | 0.3403 | 0.4112 |
| rank6_spatial | 6 | pca_diff | 0.8698 | 0.3936 | 0.3389 | 0.4091 |
| rank4_core | 4 | pca_diff | 0.8613 | 0.3897 | 0.3366 | 0.4053 |
| all | all | raw_l2 | 0.8263 | 0.3550 | 0.2707 | 0.3849 |
| rank8_core | 8 | patch5 | 0.7783 | 0.1966 | 0.0998 | 0.2586 |
| rank8_core | 8 | patch3 | 0.7920 | 0.1897 | 0.0815 | 0.2572 |
| rank4_core | 4 | patch3 | 0.7253 | 0.1677 | 0.1114 | 0.2354 |
| rank6_spatial | 6 | patch5 | 0.7447 | 0.1655 | 0.1158 | 0.2243 |
| rank4_core | 4 | patch5 | 0.7036 | 0.1650 | 0.1564 | 0.2138 |
| rank6_spatial | 6 | patch3 | 0.7372 | 0.1547 | 0.0859 | 0.2207 |
| rank4_core | 4 | global_pixel | 0.6633 | 0.1162 | 0.0762 | 0.1761 |
| rank6_spatial | 6 | global_pixel | 0.6694 | 0.1025 | 0.0250 | 0.1610 |
| rank6_spatial | 6 | window128s64mean | 0.6434 | 0.0802 | 0.0396 | 0.1431 |
| rank8_core | 8 | global_pixel | 0.6471 | 0.0791 | 0.0096 | 0.1236 |

## 5. City-Level Winners

Winner by average precision, including baselines:

| city | rank4_core | rank6_spatial | rank8_core |
|---|---|---|---|
| beirut | pca_diff AP 0.5277 | pca_diff AP 0.5284 | pca_diff AP 0.5283 |
| dubai | pca_diff AP 0.4260 | pca_diff AP 0.4291 | pca_diff AP 0.4299 |
| lasvegas | pca_diff AP 0.7498 | pca_diff AP 0.7536 | pca_diff AP 0.7545 |
| milano | raw_l2 AP 0.1655 | raw_l2 AP 0.1655 | pca_diff AP 0.1657 |
| norcia | pca_diff AP 0.0921 | pca_diff AP 0.0958 | patch5 AP 0.1696 |

Best DS-family method by city/config:

| city | rank4_core | rank6_spatial | rank8_core |
|---|---|---|---|
| beirut | patch5 AP 0.3359 | patch5 AP 0.2729 | patch5 AP 0.2192 |
| dubai | patch3 AP 0.2148 | global_pixel AP 0.1888 | patch3 AP 0.2886 |
| lasvegas | patch5 AP 0.2336 | patch3 AP 0.2827 | patch3 AP 0.3531 |
| milano | patch3 AP 0.0876 | patch3 AP 0.0738 | patch3 AP 0.0716 |
| norcia | global_pixel AP 0.0557 | global_pixel AP 0.0664 | patch5 AP 0.1696 |

## 6. Interpretation

[experiment-evidence] Spatial support helps DS relative to global pixel DS, but does not make DS competitive with PCA-diff/raw L2 in this sweep.

The best DS-family mean AP was:

```text
rank8_core / patch5: AP 0.1966
```

The strongest baseline was:

```text
rank8_core / pca_diff: AP 0.3953
```

So the strongest DS-family configuration achieved roughly half the mean AP of PCA-diff on this five-city sweep.

[experiment-evidence] Patch-vector DS is the only spatial variant worth keeping in the immediate path. It consistently improves over global pixel DS on mean AP:

```text
best patch DS mean AP: 0.1966
best global pixel DS mean AP: 0.1162
window128 mean AP: 0.0802
```

[experiment-evidence] Local-window DS with `128x128` windows is not promising as configured. Its mean AP was lower than global pixel DS and patch-vector DS.

[experiment-evidence] DS performance is city-dependent:

- Norcia rank8 patch5 beat pca_diff by AP in that city/config.
- Beirut, Dubai, Las Vegas, and Milan were still best explained by pca_diff or raw L2.

[risk] This weakens a thesis claim of "DS improves OSCD change detection." A safer claim is:

```text
Spatial sample construction substantially changes DS behavior, but current patch/window DS variants are not yet stronger than simple PCA-diff baselines on OSCD.
```

## 7. Decision

Do not run a long Phase 2 U-Net sweep with these spatial DS maps yet.

The next best step is one of:

1. Inspect comparison grids and failure modes for the five cities.
2. Expand the sweep to all official OSCD test cities with only the useful candidates:
   - raw_l2;
   - pca_diff;
   - global_pixel DS;
   - patch3;
   - patch5.
3. Add score-definition ablations for patch DS:
   - squared projection norm;
   - unsquared projection norm;
   - normalized projection ratio `||D^T delta||^2 / ||delta||^2`;
   - per-city vs global normalization.
4. Compare against Celik-style patch PCA-kmeans, because Celik is the closest classical spatial-patch pressure baseline.

## 8. Files

Ignored output files:

```text
phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/sweep_metrics_all.csv
phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/sweep_summary_by_config_method.csv
phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/sweep_best_ap_by_city_config.csv
phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/sweep_best_ds_ap_by_city_config.csv
phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/sweep_report.md
phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/runs/
```
