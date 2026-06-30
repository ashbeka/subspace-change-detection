# Reproducible Commands

## Quick Links

- [1. Purpose](#1-purpose)
- [2. Environment And Discovery](#2-environment-and-discovery)
- [3. Current Priority: Reproduce DS-Prior Fusion](#3-current-priority-reproduce-ds-prior-fusion)
- [4. Successive Saab-DS](#4-successive-saab-ds)
- [5. xBD-S12 External Evaluation](#5-xbd-s12-external-evaluation)
- [6. OSCD Spatial And Multiresolution Sweeps](#6-oscd-spatial-and-multiresolution-sweeps)
- [7. Temporal / MultiSenGE / RTW](#7-temporal--multisenge--rtw)
- [8. HSI And SpaceNet7 Transfer Gates](#8-hsi-and-spacenet7-transfer-gates)
- [9. Cleanup And Health Checks](#9-cleanup-and-health-checks)

## 1. Purpose

Commands only. Research explanation belongs in the other active docs. Use this
file when you need the shortest path from idea to rerunnable command.

## 2. Environment And Discovery

Health check:

```powershell
.\.venv\Scripts\python.exe project_cli.py doctor
```

Interactive command center:

```powershell
.\.venv\Scripts\python.exe project_cli.py
```

List commands:

```powershell
.\.venv\Scripts\python.exe project_cli.py list all
```

Compile the CLI before editing or running method commands:

```powershell
.\.venv\Scripts\python.exe -m py_compile project_cli.py
```

## 3. Current Priority: Reproduce DS-Prior Fusion

Status: command path must be confirmed from absorbed Claude code before running.

Goal:

```text
raw bands vs bands+DS vs no-DS priors vs DS priors vs matched-cross controls
```

Required comparison:

| Input/model | Purpose |
|---|---|
| raw bands | supervised floor |
| bands + DS | direct DS-prior test |
| bands + smoothed PCA + IR-MAD | strong no-DS prior control |
| bands + DS + smoothed PCA + IR-MAD | multi-prior fusion |
| bands + cross-reconstruction + smoothed PCA + IR-MAD | matched non-DS control |

Before running:

```powershell
.\.venv\Scripts\python.exe -m py_compile project_cli.py
```

## 4. Successive Saab-DS

Internal OSCD frozen train-fit gate:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-successive-transfer --fit-source oscd13 --target oscd --max-fit-samples 30000 --bootstrap 5000 --maps-per-unit 2 --device cuda --output-dir phase1/outputs/successive_transfer_oscd_trainfit_official_20260623_084003
```

xBD train-fit external gate:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-successive-transfer --fit-source xbd12 --target xbd --fit-patches-per-event 20 --max-fit-samples 30000 --bootstrap 5000 --maps-per-unit 1 --device cuda --output-dir phase1/outputs/successive_transfer_xbd_trainfit_official_20260623_084114
```

Best xBD sensitivity variant:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-successive-transfer --fit-source xbd12 --target xbd --input-normalization pair_band_zscore --fit-patches-per-event 100 --max-fit-samples 30000 --bootstrap 5000 --maps-per-unit 0 --device cuda --output-dir phase1/outputs/successive_transfer_xbd_trainfit_pairz100_20260623_091718
```

Summary figure/report generation:

```powershell
.\.venv\Scripts\python.exe phase1\scripts\summarize_successive_transfer_experiment.py --oscd-primary phase1/outputs/successive_transfer_oscd_trainfit_official_20260623_084003 --oscd-seed7 phase1/outputs/successive_transfer_oscd_trainfit_seed7_20260623_091518 --oscd-seed2026 phase1/outputs/successive_transfer_oscd_trainfit_seed2026_20260623_091613 --xbd-trainfit phase1/outputs/successive_transfer_xbd_trainfit_official_20260623_084114 --xbd-oscd-native phase1/outputs/successive_transfer_oscd12_to_xbd_native_20260623_084822 --xbd-oscd-pairz phase1/outputs/successive_transfer_oscd12_to_xbd_pairz_20260623_085743 --xbd-trainfit-pairz phase1/outputs/successive_transfer_xbd_trainfit_pairz_20260623_090651 --xbd-trainfit-pairz100 phase1/outputs/successive_transfer_xbd_trainfit_pairz100_20260623_091718 --output-dir docs/experiment_reports/assets/successive_transfer_2026-06-23
```

Reports:

```text
docs/experiment_reports/oscd_successive_subspace_learning_ds_2026-06-23.md
docs/experiment_reports/successive_saab_trainfit_external_gate_2026-06-23.md
```

## 5. xBD-S12 External Evaluation

Prepare data if the archive is already available locally:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-prepare --skip-label-extraction
```

Frozen test evaluation:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split test --bootstrap 5000 --maps-per-event 3 --boundary-buffer 0 --output-dir phase1/outputs/xbd_s12_frozen_test_unbuffered_complete_20260622_111613
```

Boundary-buffer stress:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split test --bootstrap 5000 --maps-per-event 0 --boundary-buffer 3 --event-only --output-dir phase1/outputs/xbd_s12_frozen_test_boundary3_stress_20260622_114715
```

Train/test classical confirmation:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split train --patches-per-event 100 --seed 24680 --event-only --maps-per-event 0 --rank 11 --bootstrap 5000 --metric-workers 4 --output-dir phase1/outputs/xbd_s12_train_classical_confirmation_20260622_130558
```

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split test --seed 1234 --event-only --maps-per-event 0 --rank 11 --bootstrap 5000 --metric-workers 4 --output-dir phase1/outputs/xbd_s12_frozen_test_budget_metrics_workers4_20260622_133735
```

Object retrieval:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-object-retrieval --split train --patches-per-event 100 --seed 24680 --workers 4 --bootstrap 5000 --output-dir phase1/outputs/xbd_s12_object_train100_20260622_140604
```

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-object-retrieval --split test --seed 1234 --workers 4 --bootstrap 5000 --output-dir phase1/outputs/xbd_s12_object_test_20260622_140133
```

Registration stress:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-registration-stress --patches-per-event 20 --magnitudes 0.25,0.5,1.0 --workers 4 --bootstrap 5000 --output-dir phase1/outputs/xbd_s12_registration_train20_blas1_20260622_142433
```

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-registration-stress --patches-per-event 20 --magnitudes 1.5,2.0 --workers 4 --bootstrap 5000 --output-dir phase1/outputs/xbd_s12_registration_train20_large_20260622_143254
```

Report asset summary:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-summarize --unbuffered phase1/outputs/xbd_s12_frozen_test_unbuffered_complete_20260622_111613 --boundary phase1/outputs/xbd_s12_frozen_test_boundary3_stress_20260622_114715 --train-sweep phase1/outputs/xbd_s12_train_geometry_radiometry_20260622_123321 --train-confirmation phase1/outputs/xbd_s12_train_geometry_confirmation_20260622_124000 --train-classical phase1/outputs/xbd_s12_train_classical_confirmation_20260622_130558 --test-budget phase1/outputs/xbd_s12_frozen_test_budget_metrics_workers4_20260622_133735 --object-train phase1/outputs/xbd_s12_object_train100_20260622_140604 --object-test phase1/outputs/xbd_s12_object_test_20260622_140133 --registration-near phase1/outputs/xbd_s12_registration_train20_blas1_20260622_142433 --registration-large phase1/outputs/xbd_s12_registration_train20_large_20260622_143254 --output-dir docs/experiment_reports/assets/xbd_s12_external_2026-06-22
```

## 6. OSCD Spatial And Multiresolution Sweeps

Spatial subspace smoke test:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city beirut --rank 6 --methods global_pixel,window128,patch3,patch5 --no-save-npy
```

Core spatial sweep:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --no-save-npy --continue-on-error
```

Frozen multiresolution reproduction:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities test --configs "frozen_rank12:12:global_pixel+patch3+window128s64mean+smoothed_pca_sigma1+celik_pca_kmeans+ir_mad+band_image_norm+wavelet_swt_db2_l2_ds_ll+successive_saab_h2_ds_hop1+successive_saab_h2_ds_hop2+successive_saab_h2_ds_fused+successive_saab_h2_l2_fused+successive_saab_h2_pca_fused+successive_saab_h2_cross_fused;senpai_pyramid_rank6:6:smoothed_pca_sigma1+band_image_norm+multiscale_band_image_l2+multiscale_band_image_l4+multiscale_band_image_1_2_4_fixed+multiscale_band_image_1_2_4_shifted+multiscale_band_image_product_1_2_4_shifted+multiscale_band_image_cross_1_2_4_shifted+multiscale_band_image_pca_1_2_4_shifted" --output-dir phase1/outputs/multiresolution_frozen_test10_reproduction --ssl-energy-threshold 0.95 --ssl-max-channels 16 --ssl-max-fit-samples 30000 --feature-device auto --continue-on-error --no-save-npy
```

## 7. Temporal / MultiSenGE / RTW

MultiSenGE manifest:

```powershell
.\.venv\Scripts\python.exe -m phase1.scripts.build_multisenge_manifest --multisenge_root data/MultiSenGE --output_path phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --patch_ids "32TLT_3855_0,32TLT_4369_257,32TLT_4883_514,32TLT_5397_1028,32TLT_5654_257" --min_s2_dates 20 --seed 1234
```

First/second DS and geodesic sequence dynamics:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-multisenge-temporal-dynamics --manifest phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --rank 10 --preprocessing centered --patch-ids "32TLT_3855_0,32TLT_4369_257,32TLT_4883_514,32TLT_5397_1028,32TLT_5654_257" --output-dir "phase1/outputs/multisenge_temporal_timeaware_core5_$tag"
```

Temporal injections:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-multisenge-temporal-injections --manifest phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --rank 10 --preprocessing centered --repeats 12 --output-dir "phase1/outputs/multisenge_temporal_injections_$tag"
```

Order-aware interventions:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-multisenge-order-aware-interventions --output-dir "phase1/outputs/multisenge_order_aware_interventions_$tag" --crop-size 32 --repeats 8 --max-patches 5 --ranks 1,2 --representations unordered,difference,trajectory2,trajectory3 --preprocessing feature_centered_observation_l2 --bootstrap 300
```

RTW invariance gate:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-rtw-invariance-gate --output-dir "phase1/outputs/multisenge_rtw_invariance_$tag"
```

BreizhCrops download:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-breizhcrops-download --regions frh01,frh02,frh03,frh04
```

BreizhCrops RTW natural-label transfer:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-rtw-breizhcrops-transfer --search-rtw --development-region frh01 --holdout-region frh04 --output-dir "phase1/outputs/breizhcrops_rtw_nested_search_frh01_frh04_$tag" --max-fields-per-class 80 --anchors-per-class 40 --rtw-replicates 3 --bootstrap 1000 --seed 2718
```

## 8. HSI And SpaceNet7 Transfer Gates

HSI local moment geometry:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-hsi-moment-geometry --datasets "benton,hermiston,farmland,shenzhen" --configs "joint_robust_zscore:5:3:3,joint_robust_zscore:9:5:3,joint_robust_zscore:13:7:5,per_date_zscore:9:5:5" --bootstrap 2000 --output-dir phase1/outputs/hsi_moment_geometry_final_stable_20260621_185316
```

HSI Band-Image transfer:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-hsi-band-image-transfer --datasets "benton,hermiston,farmland,shenzhen" --rank 11 --seed 1234 --bootstrap 500 --output-dir phase1/outputs/hsi_band_image_transfer_frozen_complete_20260622_185629
```

HSI rank sensitivity:

```powershell
foreach($rank in 3,6,11,24,48,96){ .\.venv\Scripts\python.exe project_cli.py phase1-hsi-band-image-transfer --datasets "benton,hermiston,farmland,shenzhen" --rank $rank --seed 1234 --bootstrap 0 --output-dir "phase1/outputs/hsi_band_image_rankdiag_r${rank}_20260622_190113" }
```

SpaceNet7 Band-Image transfer:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spacenet7-band-image-transfer --workers 4 --bootstrap 3000 --output-dir phase1/outputs/spacenet7_band_image_transfer_frozen_20260622_191204
```

SpaceNet7 temporal geometry controls:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spacenet7-temporal-subspaces --aoi-root "data/SpaceNet7_confirmation/L15-1691E-1211N_6764_3347_13" --representations trajectory2 --preprocessing feature_centered --window 6 --rank 2 --grids 8 --output-dir "phase1/outputs/spacenet7_geometry_$tag"
```

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spacenet7-temporal-subspaces --aoi-root "data/SpaceNet7_confirmation/L15-1691E-1211N_6764_3347_13" --window 6 --grids 8 --radiometric-normalization per_date_channel_standardize --controls-only --output-dir "phase1/outputs/spacenet7_controls_$tag"
```

## 9. Cleanup And Health Checks

Git:

```powershell
git status --short --branch
git worktree list
```

Disk overview:

```powershell
Get-ChildItem -Directory | ForEach-Object { $s=(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum; "{0,-24} {1,10:N1} MB" -f $_.Name, ($s/1MB) }
```

Markdown path sanity for active docs:

```powershell
rg -n "pending_deletion_review|old_notes|old_project_docs|source_records|experiment_reports" docs -g "*.md"
```
