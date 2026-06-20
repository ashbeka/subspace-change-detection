# Runbook

## Table Of Contents

- [0. Golden Rules](#0-golden-rules)
- [1. Git Workflow](#1-git-workflow)
- [2. Environment Setup](#2-environment-setup)
- [3. Data Layout](#3-data-layout)
- [4. One-Command Liveness Check](#4-one-command-liveness-check)
- [5. Phase 1: Generate Prior Maps](#5-phase-1-generate-prior-maps)
- [5A. Phase 1: Spatial Subspace Audit Workflow](#5a-phase-1-spatial-subspace-audit-workflow)
- [6. Phase 1: MultiSenGE Optional Exploration](#6-phase-1-multisenge-optional-exploration)
- [6A. Temporal First/Second DS Study](#6a-temporal-firstsecond-ds-study)
- [7. Phase 2 Config Matrix](#7-phase-2-config-matrix)
- [8. Phase 2: Short E0/E1 Smoke Run](#8-phase-2-short-e0e1-smoke-run)
- [9. Phase 2: Single Full Run](#9-phase-2-single-full-run)
- [10. Phase 2: Recommended Controlled Sweep](#10-phase-2-recommended-controlled-sweep)
- [11. Compare Experiments](#11-compare-experiments)
- [12. Visualize Predictions](#12-visualize-predictions)
- [13. Resume and Restart](#13-resume-and-restart)
- [14. Metadata to Record for Every Real Experiment](#14-metadata-to-record-for-every-real-experiment)
- [15. Cleanup and Retention](#15-cleanup-and-retention)
- [16. Troubleshooting](#16-troubleshooting)
- [17. Current Evidence Anchors](#17-current-evidence-anchors)

Generated: 2026-05-03
Workflow updated: 2026-06-06
Primary environment: Windows PowerShell, repo root `E:\research_projects\subspace-change-detection`

This is the operational cheat sheet for reproducing the current project. It is deliberately practical: commands, expected paths, outputs, and failure checks. The project's truthful current scope is OSCD Sentinel-2 binary change segmentation with optional unsupervised prior channels. It is not yet an end-to-end xBD/xBD-S12 damage segmentation system.

## 0. Golden Rules

- Run commands from the repo root unless a section says otherwise.
- Use timestamped output folders under ignored `phase1/outputs/` and `phase2/outputs/`.
- Keep `data/`, `phase1/outputs/`, and `phase2/outputs/` out of git.
- Commit code/docs/config changes, not datasets, checkpoints, or generated maps.
- Treat old outputs as evidence to audit, not proof.
- For thesis claims, do not use 1-epoch smoke results as performance evidence.
- Treat "Phase 1" and "Phase 2" here as command/workflow labels matching current folders, not as a fixed research structure.

## 1. Git Workflow

Check state:

```powershell
git status --short --branch
git log --oneline --decorate -5
```

Default project workflow:

```powershell
git switch main
git pull --ff-only
```

Work on `main` unless there is a deliberate reason to isolate a risky feature, restructuring pass, or long-running experiment branch.

Commit one coherent change:

```powershell
git status --short
git add <paths>
git commit -m "<type>: <short summary>"
```

Push so it appears on GitHub:

```powershell
git push origin main
```

## 2. Environment Setup

Activate the root virtual environment:

```powershell
cd E:\research_projects\subspace-change-detection
.\.venv\Scripts\Activate.ps1
$py=".\.venv\Scripts\python.exe"
```

Install dependencies if rebuilding the environment:

```powershell
& $py -m pip install -r phase1/requirements.txt
& $py -m pip install -r phase2/requirements.txt
& $py -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

GPU sanity check:

```powershell
& $py -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expected in the current environment:

```text
torch 2.9.1+cu126
cuda True
NVIDIA GeForce RTX 4070 SUPER
```

CPU fallback:

```powershell
$device="cpu"
```

GPU default:

```powershell
$device="cuda"
```

Central CLI:

```powershell
.\.venv\Scripts\python.exe project_cli.py
.\.venv\Scripts\python.exe project_cli.py doctor
.\.venv\Scripts\python.exe project_cli.py list all
.\.venv\Scripts\python.exe project_cli.py list commands
.\.venv\Scripts\python.exe project_cli.py list configs
```

Use the CLI as the main command surface for checks, inventories, prior generation, training, evaluation, sweeps, visualization, and cleanup previews. Running it with no subcommand opens the interactive command center; every menu action prints the exact command before execution. The explicit commands below remain documented for reproducibility and debugging.

## 3. Data Layout

OSCD must exist at:

```text
data/OSCD/
  onera_satellite_change_detection dataset__images/
  onera_satellite_change_detection dataset__train_labels/
  onera_satellite_change_detection dataset__test_labels/
```

Check OSCD counts:

```powershell
(Get-ChildItem "data/OSCD/onera_satellite_change_detection dataset__images" -Directory).Count
(Get-ChildItem "data/OSCD/onera_satellite_change_detection dataset__train_labels" -Directory).Count
(Get-ChildItem "data/OSCD/onera_satellite_change_detection dataset__test_labels" -Directory).Count
```

Expected current counts:

```text
24 image cities
14 train-label cities
10 test-label cities
```

MultiSenGE is optional for Phase 1 temporal exploration:

```text
data/MultiSenGE/
  ground_reference/
  labels/
  s1/
  s2/
```

xBD is currently future work only. The generic adapter expects CSV indexes that are not present:

```text
data/xbd/train.csv
data/xbd/val.csv
data/xbd/test.csv
```

If those CSVs are absent, do not claim xBD training/evaluation is implemented.

## 4. One-Command Liveness Check

This checks imports, OSCD shapes, raw/prior channels, forward passes, xBD CSV absence, and existing artifacts without training:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; @'
from pathlib import Path
import csv, importlib, yaml, torch

ROOT = Path.cwd()
modules = [
    "phase1.eval.run_oscd_eval",
    "phase1.eval.run_oscd_geodesic_priors",
    "phase2.train.train_oscd_seg",
    "phase2.eval.evaluate_oscd_seg",
    "phase2.viz.viz_seg_predictions",
    "phase2.viz.viz_oscd_combined",
]
for name in modules:
    importlib.import_module(name)
print("imports: ok", len(modules))
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())

from phase2.data.oscd_seg_dataset import OSCDSegmentationDataset
from phase2.models.unet2d import UNet2D

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["patch_size"] = 128
    cfg["dataset"]["patch_overlap"] = 0
    cfg["dataset"]["augmentations"] = {"flip": False, "rotate90": False, "noise": False}
    cfg["dataset"]["cache"] = {"cities": True, "max_cities": 1}
    return cfg

raw_cfg = load_cfg(ROOT / "phase2/configs/oscd/core/E0_raw_unet.yaml")
raw_ds = OSCDSegmentationDataset(ROOT / raw_cfg["dataset"]["root"], "train", raw_cfg, phase1_change_maps_root=None)
raw_item = raw_ds[0]
print("raw:", len(raw_ds), tuple(raw_item["x"].shape), tuple(raw_item["y"].shape), tuple(raw_item["valid"].shape))
model = UNet2D(in_channels=raw_item["x"].shape[0], base_channels=8, depth=3, num_classes=1)
with torch.no_grad():
    print("raw_forward:", tuple(model(raw_item["x"].unsqueeze(0)).shape))

prior_cfg = load_cfg(ROOT / "phase2/configs/oscd/core/E1_raw_ds_unet.yaml")
prior_root = ROOT / prior_cfg["phase1"]["change_maps_root"]
prior_ds = OSCDSegmentationDataset(ROOT / prior_cfg["dataset"]["root"], "train", prior_cfg, phase1_change_maps_root=prior_root)
prior_item = prior_ds[0]
print("raw+ds:", len(prior_ds), tuple(prior_item["x"].shape), tuple(prior_item["y"].shape), tuple(prior_item["valid"].shape))
model = UNet2D(in_channels=prior_item["x"].shape[0], base_channels=8, depth=3, num_classes=1)
with torch.no_grad():
    print("prior_forward:", tuple(model(prior_item["x"].unsqueeze(0)).shape))

xbd_root = ROOT / "data/xbd"
print("xbd_missing_csvs:", [p for p in ["train.csv", "val.csv", "test.csv"] if not (xbd_root / p).exists()])

artifact = ROOT / "phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv"
if artifact.exists():
    with artifact.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("artifact_rows:", len(rows), [r.get("experiment_tag") or r.get("tag") for r in rows])
'@ | & $py -
```

Expected important shapes:

```text
raw x = (26, 128, 128)
raw+DS x = (27, 128, 128)
raw+DS+PCA x = (28, 128, 128)
forward logits = (1, 1, 128, 128)
```

Note: `E3_raw_ds_pca_unet.yaml` is raw+DS+PCA. `E1_raw_ds_unet.yaml` is raw+DS only.

## 5. Phase 1: Generate Prior Maps

Set common paths:

```powershell
$oscd="data/OSCD"
$phase1Fast="phase1/outputs/oscd_saved_priors_fast"
$changeMaps="phase1/outputs/oscd_saved_priors_fast/oscd_change_maps"
$phase1Full="phase1/outputs/oscd_saved_full"
$changeMapsFull="phase1/outputs/oscd_saved_full/oscd_change_maps"
```

Fast OSCD priors for Phase 2 core experiments:

```powershell
& $py -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_priors_fast.yaml --oscd_root $oscd --output_dir $phase1Fast --save_change_maps
```

Full classical baseline priors:

```powershell
& $py -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_default.yaml --no_window --oscd_root $oscd --output_dir $phase1Full --save_change_maps
```

Geodesic priors:

```powershell
& $py -m phase1.eval.run_oscd_geodesic_priors --config phase1/configs/oscd_geodesic_priors.yaml --oscd_root $oscd --output_dir phase1/outputs/oscd_geodesic_priors
```

Expected Phase 1 output structure:

```text
phase1/outputs/<run>/
  oscd_eval_results.json
  oscd_eval_summary.csv
  oscd_change_maps/<split>/<method>/<city>_score.npy
  oscd_change_maps/<split>/<method>/<city>_mask.png
```

Important method folder names:

```text
ds_projection
ds_cross_residual
pixel_diff
cva
pca_diff
celik
ir_mad
geodesic_dist
```

Phase 2 prior loading expects exactly this structure.

## 5A. Phase 1: Spatial Subspace Comparison Workflow

Current research priority:

```text
global pixel DS vs patch-vector DS vs local-window DS vs Band-Image DS
```

Source-linked implementation rule:

```text
source material -> math object -> satellite adaptation -> code path -> toy check -> one-city map -> thesis claim
```

Existing command for current global subspace construction inspection:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-subspace-inspect --city beirut --rank 6 --split auto
```

Equivalent raw command:

```powershell
.\.venv\Scripts\python.exe phase1/scripts/inspect_oscd_subspace.py --city beirut --rank 6 --split auto
```

What this inspection proves:

- OSCD loads correctly for one city.
- The data matrix is `X_pre, X_post in R^(13 x N)`.
- PCA rank and explained variance can be inspected.
- Legacy residual-stack DS, repaired eig DS, and canonical DS can be compared against raw spectral L2.

What it does not yet prove:

- patch-vector DS;
- local-window DS as a controlled comparison;
- Band-Image DS, where each Sentinel-2 band image is one flattened spatial vector;
- multiscale subspace pyramid;
- city-level metric maps and visual outputs for the spatial variants.

Spatial comparison command:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city beirut --rank 6 --methods global_pixel,window128,patch3,patch5
```

Spatial comparison with the current Band-Image DS candidate:

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city beirut --rank 8 --methods global_pixel,patch3,patch5,window128s64mean,band_image_ds,band_image_norm --output-dir "phase1/outputs/spatial_ds_beirut_band_image_$tag" --no-save-npy
```

Equivalent raw command:

```powershell
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe phase1/scripts/compare_oscd_spatial_subspaces.py `
  --city beirut `
  --rank 6 `
  --methods global_pixel,window128,patch3,patch5 `
  --output_dir "phase1/outputs/oscd_spatial_subspace_compare_$tag"
```

Outputs:

```text
spatial_subspace_metrics.csv
run_metadata.json
score_maps/raw_l2.png
score_maps/pca_diff.png
score_maps/global_pixel.png
score_maps/window128s64mean.png
score_maps/patch3.png
score_maps/patch5.png
score_maps/band_image_ds.png
score_maps/band_image_norm.png
comparison_grid.png
```

Multi-city/multi-config sweep command:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --no-save-npy
```

Completed reference sweep:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --output-dir phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823 --resume --no-save-npy
```

Tracked report:

```text
docs/experiment_reports/oscd_spatial_subspace_sweep_core5_2026-06-14.md
```

Completed all-city Band-Image DS score-ablation sweep:

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank8_band_image_scores:8:global_pixel+patch3+patch5+window128s64mean+band_image_ds+band_image_norm+band_image_ratio+band_image_residual" --output-dir "phase1/outputs/spatial_ds_band_image_score_ablation_allcities_$tag" --continue-on-error --no-save-npy
```

Tracked report:

```text
docs/experiment_reports/oscd_band_image_ds_score_ablation_2026-06-18.md
```

Corrected all-city classical pressure comparison:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank8_traditional_pressure:8:global_pixel+patch3+patch5+window128s64mean+band_image_norm+celik_pca_kmeans+ir_mad" --output-dir "phase1/outputs/spatial_ds_traditional_pressure_allcities_corrected_$tag" --continue-on-error --no-save-npy
```

Rank-12 label-free fusion comparison:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank12_label_free_fusion:12:band_image_norm+ir_mad+rank_fusion_pca_band+rank_fusion_band_irmad+rank_fusion_pca_irmad+rank_fusion_pca_band_irmad" --output-dir "phase1/outputs/spatial_score_rank_fusion_allcities_$tag" --continue-on-error --no-save-npy
```

Tracked interpretation:

```text
docs/experiment_reports/oscd_spatial_ds_baseline_pressure_2026-06-18.md
```

Generate saved score arrays for split-safe calibration:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; $source="phase1/outputs/spatial_score_calibration_source_allcities_$tag"; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank12_calibration_source:12:band_image_norm+ir_mad+rank_fusion_pca_band_irmad" --output-dir $source --continue-on-error --save-npy
```

Fit changed-area fractions on official training cities and evaluate them unchanged on official test cities:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-score-calibration --sweep-root phase1/outputs/spatial_score_calibration_source_allcities_20260618_182915 --config rank12_calibration_source --output-dir "phase1/outputs/oscd_split_safe_calibration_$tag"
```

The second command writes train calibration curves, held-out test metrics, paired city tests, and TP/FP/FN diagnostic maps. It is supervised by training labels; it is not an unsupervised threshold.

Default sweep design:

```text
cities: beirut,dubai,lasvegas,milano,norcia
rank4_core: rank 4, methods global_pixel,patch3,patch5
rank6_spatial: rank 6, methods global_pixel,window128,patch3,patch5
rank8_core: rank 8, methods global_pixel,patch3,patch5
```

Sweep outputs:

```text
phase1/outputs/oscd_spatial_subspace_sweep_<timestamp>/
  sweep_manifest.json
  sweep_metrics_all.csv
  sweep_summary_by_config_method.csv
  sweep_best_ap_by_city_config.csv
  sweep_best_ds_ap_by_city_config.csv
  sweep_report.md
  logs/
  runs/<config>__<city>/
```

Validation meaning:

- `AUROC`: threshold-free ranking check; higher means OSCD changed pixels usually receive higher scores than unchanged pixels.
- `average_precision`: precision-recall ranking check; more informative than accuracy when changed pixels are rare.
- `best F1/IoU`: oracle diagnostic threshold chosen with labels. Useful for upper-bound diagnosis, not deployable evidence.
- `Otsu F1/IoU`: unsupervised threshold check based only on the score histogram.
- `raw_l2_corr`: redundancy check; if a DS score is almost perfectly correlated with raw spectral L2, it is probably not adding a distinct geometric signal.
- `raw_l2` and `pca_diff`: baseline pressure so DS is not compared only against itself.

Completed fixed-grid pyramid decision:

```text
spatial_pyramid_1_2_4_energy
spatial_pyramid_1_2_4_norm
spatial_pyramid_1_2_4_8_norm
```

This is the wavelet/JPEG/Green-Learning-inspired idea:

```text
1x1 whole image -> 1 subspace
2x2 grid       -> 4 subspaces
4x4 grid       -> 16 subspaces
8x8 grid       -> 64 subspaces
```

Core-five testing did not improve AP over global pixel DS. Stop this exact construction. Do not call it "Green Learning," PixelHop, or a wavelet implementation: those require a different, source-verified feature hierarchy.

## 6. Phase 1: MultiSenGE Optional Exploration

Create a small manifest:

```powershell
& $py -m phase1.scripts.build_multisenge_manifest --multisenge_root data/MultiSenGE --output_path phase1/outputs/multisenge_manifest_50p_5dates.json --min_s2_dates 5 --max_patches 50 --seed 1234
```

Run temporal/geodesic analysis:

```powershell
& $py -m phase1.eval.run_multisenge_temporal_geodesic --config phase1/configs/multisenge_temporal_geodesic.yaml --multisenge_root data/MultiSenGE --manifest phase1/outputs/multisenge_manifest_50p_5dates.json --output_dir phase1/outputs/multisenge_temporal_geodesic
```

Run MultiSenGE visualization:

```powershell
& $py -m phase1.eval.run_multisenge_viz --config phase1/configs/multisenge_default.yaml --multisenge_root data/MultiSenGE/s2 --output_dir phase1/outputs/multisenge_viz
```

MultiSenGE is exploratory. Do not use it as the main supervised segmentation benchmark without new work.

## 6A. Temporal First/Second DS Study

Formula tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_temporal_subspace_dynamics tests.test_seasonal_observation_subspaces -v
```

Build the current five-patch, 23-date MultiSenGE manifest:

```powershell
.\.venv\Scripts\python.exe -m phase1.scripts.build_multisenge_manifest --multisenge_root data/MultiSenGE --output_path phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --patch_ids "32TLT_3855_0,32TLT_4369_257,32TLT_4883_514,32TLT_5397_1028,32TLT_5654_257" --min_s2_dates 20 --seed 1234
```

Run full-rank band-image first/second DS on those patches:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-multisenge-temporal-dynamics --manifest phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --rank 10 --preprocessing centered --patch-ids "32TLT_3855_0,32TLT_4369_257,32TLT_4883_514,32TLT_5397_1028,32TLT_5654_257" --output-dir "phase1/outputs/multisenge_temporal_timeaware_core5_$tag"
```

Run controlled local-change/radiometric/translation injections:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-multisenge-temporal-injections --manifest phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --rank 10 --preprocessing centered --repeats 12 --output-dir "phase1/outputs/multisenge_temporal_injections_$tag"
```

Run any directory of registered date-prefixed multispectral TIFFs:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-registered-sequence-dynamics --sequence-dir tmp/ipol416_vegas --output-dir "phase1/outputs/registered_temporal_ds_$tag" --rank 0 --preprocessing centered --top-k 20
```

Compare backward/forward temporal contexts on a registered sequence:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-temporal-context-ds --sequence-dir tmp/ipol416_vegas_pseudogamma --context-sizes 3,5 --ranks 1,2 --factorizations per_band,joint --reference-lognfa-dir tmp/ipol416_reference_run_pseudogamma/lognfa_dir --output-dir "phase1/outputs/temporal_context_ds_vegas_$tag" --figure-config 5:2:per_band --figure-count 4
```

Run controlled persistent/transient/radiometric/translation interventions on
the current five-patch MultiSenGE manifest:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-temporal-context-injections --manifest phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --target-date 20200909 --context-size 3 --rank 2 --factorization per_band --repeats 4 --max-patches 5 --output-dir "phase1/outputs/temporal_context_injections_$tag"
```

Run the subpixel translation and low-frequency robustness curve:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-temporal-registration-curve --manifest phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --target-date 20200909 --context-size 3 --rank 2 --shifts 0.25,0.5,1,2 --strategies native,gaussian1,gaussian2,pool2,pool4,phase_align --local-strength 0.25 --window-size 32 --repeats 3 --max-patches 5 --output-dir "phase1/outputs/temporal_registration_curve_$tag"
```

Run the seasonal observation-subspace stress test before acquiring irrigation
data. This compares abrupt shape change, amplitude-only change, gradual drift,
phase shift, radiometric nuisance, missing composites, and translation:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-seasonal-regime-study --repeats 80 --ranks 1,2,4,8 --preprocessing uncentered,feature_centered,feature_centered_observation_l2 --bootstrap 200 --output-dir "phase1/outputs/seasonal_regime_subspace_study_$tag"
```

Compare unordered, first-difference, and block-trajectory subspaces on real
MultiSenGE backgrounds with controlled event and nuisance transformations:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-multisenge-order-aware-interventions --output-dir "phase1/outputs/multisenge_order_aware_interventions_$tag" --crop-size 32 --repeats 8 --max-patches 5 --ranks 1,2 --representations unordered,difference,trajectory2,trajectory3 --preprocessing feature_centered_observation_l2 --bootstrap 300
```

Run the RTW timing/tempo-invariance gate. The command screens all configurations
on a fixed development subset, re-estimates six finalists, freezes one, and
scores two untouched patches against Fourier, harmonic, DTW/TWDTW, M-SSA,
snapshot-subspace, and scalar controls:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-rtw-invariance-gate --output-dir "phase1/outputs/multisenge_rtw_invariance_$tag"
```

The completed 2026-06-21 gate was negative for incremental RTW value. Do not
advance to natural irrigation/crop-transition claims unless a new mechanism and
fresh preregistration justify reopening it.

Run the strongest current local/off-grid controlled protocol with fair
multispectral index controls:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-multiscale-order-aware-interventions --output-dir "phase1/outputs/multiscale_order_aware_fair_controls_$tag" --crop-size 32 --grids 8 --representations unordered --rank 1 --preprocessing feature_centered_observation_l2 --spatial-smoothing-sigma 2 --repeats 4 --max-patches 5 --bootstrap 300
```

Check public IrrMapper coverage and, after configuring an Earth Engine-enabled
Cloud project, query one candidate AOI without downloading imagery:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-irrigation-data-feasibility --ee-project <GOOGLE_CLOUD_PROJECT_ID> --bbox=-112.60,45.20,-112.40,45.35 --start 2017-01-01 --end 2025-01-01 --output-dir "phase1/outputs/irrigation_regime_data_feasibility_$tag"
```

Interpretation rules:

- `rank 0` means full band-image span (`r=B`) for the generic runner.
- The paper's second DS assumes equal time spacing. Use `gap_ratio` and the
  separately named time-aware geodesic deviation for irregular dates.
- MultiSenGE's static reference is not temporal ground truth.
- Contribution maps sum to DS magnitude but are attribution maps, not calibrated
  change probabilities.
- Do not claim superiority until external NFA/MOSUM/BFAST/JUST or equivalent
  baselines and an event-evaluation protocol are present.
- IPOL log-NFA comparisons are detector agreement, not ground-truth accuracy.
- `temporal_context_ds` is canonical DS between backward/forward context spans.
  `linear_projection_novelty` is a separate orthogonal-residual control and is
  not IPOL's NNLS/NFA algorithm.
- The seasonal-regime command is synthetic diagnostics, not irrigation
  accuracy. Real IrrMapper transitions are weak labels and require manual or
  independent verification.
- The order-aware and multiscale intervention commands use real backgrounds but
  injected transformations. They test behavior/localization, not natural event
  accuracy. Do not report their AP as DynamicEarthNet or IrrMapper performance.

## 7. Phase 2 Config Matrix

Core configs:

| tag | config | input |
|---|---|---|
| E0_raw | `phase2/configs/oscd/core/E0_raw_unet.yaml` | raw pre/post S2, 26 channels |
| E1_raw_ds | `phase2/configs/oscd/core/E1_raw_ds_unet.yaml` | raw + DS projection |
| E1b_raw_ds_cross | `phase2/configs/oscd/core/E1b_raw_ds_cross_unet.yaml` | raw + DS cross-residual |
| E2_raw_pca | `phase2/configs/oscd/core/E2_raw_pca_unet.yaml` | raw + PCA-diff |
| E3_raw_ds_pca | `phase2/configs/oscd/core/E3_raw_ds_pca_unet.yaml` | raw + DS projection + PCA-diff |
| E4_raw_pixel | `phase2/configs/oscd/extended/E4_raw_pixel_unet.yaml` | raw + pixel-diff |
| E5_raw_celik | `phase2/configs/oscd/extended/E5_raw_celik_unet.yaml` | raw + Celik prior; requires full Phase 1 maps |
| E6_raw_irmad | `phase2/configs/oscd/extended/E6_raw_irmad_unet.yaml` | raw + IR-MAD prior; requires full Phase 1 maps |
| S0_siamese | `phase2/configs/oscd/core/S0_raw_siamese.yaml` | Siamese raw baseline |
| E0_raw_resnet | `phase2/configs/oscd/extended/E0_raw_resnet.yaml` | ResNet-backbone raw baseline |
| E3_raw_ds_pca_resnet | `phase2/configs/oscd/extended/E3_raw_ds_pca_resnet.yaml` | ResNet-backbone raw+priors |
| E3_raw_ds_pca_fusion | `phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml` | fusion model raw+priors |

Current active thesis-critical comparison:

```text
E0_raw vs E1_raw_ds
```

## 8. Phase 2: Short E0/E1 Smoke Run

Use this to prove the train/eval path is alive. Do not use it as performance evidence.

```powershell
$ts=Get-Date -Format 'yyyyMMdd_HHmmss'
$smokeRoot="phase2/outputs/smoke_e0_e1_$ts"
New-Item -ItemType Directory -Force -Path $smokeRoot | Out-Null

$run=Join-Path $smokeRoot "E0_raw"
& $py -m phase2.train.train_oscd_seg --config phase2/configs/oscd/core/E0_raw_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run --device cuda --epochs 1
& $py -m phase2.eval.evaluate_oscd_seg --config phase2/configs/oscd/core/E0_raw_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval_best") --device cuda

$run=Join-Path $smokeRoot "E1_raw_ds"
& $py -m phase2.train.train_oscd_seg --config phase2/configs/oscd/core/E1_raw_ds_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run --device cuda --epochs 1
& $py -m phase2.eval.evaluate_oscd_seg --config phase2/configs/oscd/core/E1_raw_ds_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval_best") --device cuda
```

The 2026-05-03 smoke run lives at:

```text
phase2/outputs/smoke_e0_e1_20260503_040723
```

It proved liveness only. It did not prove DS helps.

## 9. Phase 2: Single Full Run

Set output root:

```powershell
$epochs=150
$outRoot="phase2/outputs/runs_gpu_${epochs}ep_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
```

E0 raw-only:

```powershell
$run=Join-Path $outRoot "E0_raw_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd/core/E0_raw_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd/core/E0_raw_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

E1 raw+DS:

```powershell
$run=Join-Path $outRoot "E1_raw_ds_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd/core/E1_raw_ds_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd/core/E1_raw_ds_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

Outputs per run:

```text
best.ckpt
last.ckpt
train_log.json
run_metadata.json
eval/oscd_seg_eval_results.json
eval/oscd_seg_eval_summary.csv
eval/run_metadata.json
```

## 10. Phase 2: Recommended Controlled Sweep

The sweep script writes patched `config_used.yaml` files into each output directory, avoiding tracked YAML edits.
The script runs Python unbuffered so epoch output should stream into the terminal and each run's `train_console.log.txt`.

Core 3-seed reproduction with durable logs:

```powershell
powershell -ExecutionPolicy Bypass -File phase2/scripts/run_phase2_sweep.ps1 -Preset core -Epochs 150 -Seeds "1234,1235,1236" -OutputTag "core_150ep_$(Get-Date -Format 'yyyyMMdd_HHmmss')" -Retention full
```

Interactive one-command version with live `tqdm` batch progress bars. Prefer PowerShell 7 (`pwsh`) for clean terminal rendering:

```powershell
cd E:\research_projects\subspace-change-detection; .\.venv\Scripts\Activate.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File phase2/scripts/run_phase2_sweep.ps1 -Preset core -Epochs 150 -Seeds "1234,1235,1236" -OutputTag "core_150ep_$(Get-Date -Format 'yyyyMMdd_HHmmss')" -Retention full -ProgressBars
```

`-ProgressBars` disables transcript/per-run console capture and lets Python render native `tqdm` bars directly in the terminal. Current training bars are labeled with the run folder, for example `E1_raw_ds__seed1234 ep 27/150`, and completed bars are cleared by default to reduce terminal overflow. The durable evidence is still `train_log.json`, `run_metadata.json`, and eval CSV/JSON.

Core only includes:

```text
E0_raw_unet
S0_siamese
E1_raw_ds
E1b_raw_ds_cross
E2_raw_pca
E3_raw_ds_pca
```

Full sweep with additional models:

```powershell
powershell -ExecutionPolicy Bypass -File phase2/scripts/run_phase2_sweep.ps1 -Preset full -Epochs 150 -Seeds "1234,1235,1236" -OutputTag "full_150ep_$(Get-Date -Format 'yyyyMMdd_HHmmss')" -Retention full
```

Do not run `full+eig` until eig priors exist and have been smoke-checked.

Watch a running sweep from another PowerShell window:

```powershell
cd E:\research_projects\subspace-change-detection
powershell -ExecutionPolicy Bypass -File phase2/scripts/watch_phase2_sweep.ps1
```

Watch a specific sweep folder:

```powershell
powershell -ExecutionPolicy Bypass -File phase2/scripts/watch_phase2_sweep.ps1 -RunRoot phase2/outputs/sweep_core_150ep_E0_E1_repro_20260503_042530 -RefreshSeconds 15
```

Print one status snapshot:

```powershell
powershell -ExecutionPolicy Bypass -File phase2/scripts/watch_phase2_sweep.ps1 -RunRoot phase2/outputs/sweep_core_150ep_E0_E1_repro_20260503_042530 -Once
```

Tail the raw transcript:

```powershell
Get-Content phase2/outputs/sweep_core_150ep_E0_E1_repro_20260503_042530/sweep_transcript.txt -Tail 60 -Wait
```

## 11. Compare Experiments

Manual E0/E1 comparison:

```powershell
& $py -m phase2.eval.compare_priors_effect --summaries (Join-Path $outRoot "E0_raw_unet\eval\oscd_seg_eval_results.json") (Join-Path $outRoot "E1_raw_ds_unet\eval\oscd_seg_eval_results.json") --tags E0_raw E1_raw_ds --output (Join-Path $outRoot "e0_e1_summary.csv")
```

Core comparison:

```powershell
& $py -m phase2.eval.compare_priors_effect --summaries (Join-Path $outRoot "E0_raw_unet\eval\oscd_seg_eval_results.json") (Join-Path $outRoot "E1_raw_ds_unet\eval\oscd_seg_eval_results.json") (Join-Path $outRoot "E1b_raw_ds_cross_unet\eval\oscd_seg_eval_results.json") (Join-Path $outRoot "E2_raw_pca_unet\eval\oscd_seg_eval_results.json") (Join-Path $outRoot "E3_raw_ds_pca_unet\eval\oscd_seg_eval_results.json") --tags E0_raw E1_raw_ds E1b_raw_ds_cross E2_raw_pca E3_raw_ds_pca --output (Join-Path $outRoot "oscd_priors_ablation_summary.csv")
```

The comparison script uses the `test.summary` block from each eval JSON.

Analyze a completed sweep per seed and per city:

```powershell
.\.venv\Scripts\python.exe -m phase2.eval.analyze_sweep_results --sweep_root phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422 --output_dir phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422/analysis_20260505
```

This writes `summary_by_tag.csv`, `pairwise_seed_deltas.csv`, `per_city_delta_summary.csv`, `threshold_proxy_candidates.csv`, and `analysis_report.md`. Threshold rows are proxy evidence only unless probability maps are saved or inference is rerun.

## 12. Visualize Predictions

Segmentation prediction figures:

```powershell
$run=Join-Path $outRoot "E1_raw_ds_unet"
& $py -m phase2.viz.viz_seg_predictions --device cuda --config phase2/configs/oscd/core/E1_raw_ds_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "figs_seg") --cities test
```

Combined priors and segmentation figures:

```powershell
$run=Join-Path $outRoot "E3_raw_ds_pca_unet"
& $py -m phase2.viz.viz_oscd_combined --device cuda --config phase2/configs/oscd/core/E3_raw_ds_pca_unet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "figs_combined") --cities test
```

Phase 1 OSCD method grid:

```powershell
& $py -m phase1.eval.visualize_oscd_examples --config phase1/configs/oscd_priors_fast.yaml --oscd_root $oscd --output_dir phase1/outputs/oscd_figs_all --cities chongqing --metrics_json phase1/outputs/oscd_saved_priors_fast/oscd_eval_results.json
```

## 13. Resume and Restart

Resume a run:

```powershell
& $py -m phase2.train.train_oscd_seg --device cuda --config <same_config> --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir <same_output_dir> --resume
```

Resume from explicit checkpoint:

```powershell
& $py -m phase2.train.train_oscd_seg --device cuda --config <same_config> --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir <same_output_dir> --resume_ckpt <same_output_dir>\best.ckpt
```

Overwrite an existing run directory:

```powershell
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config <config> --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir <same_output_dir> --overwrite_output_dir
```

Use overwrite only when you intentionally do not need the previous artifacts.

## 14. Metadata to Record for Every Real Experiment

For a run to be thesis-usable, record:

- Git commit hash.
- Branch name.
- Config path and copied config used.
- Seed.
- Epoch count.
- Device and GPU.
- OSCD split.
- Phase 1 prior map root.
- Checkpoint path.
- Eval JSON and CSV path.
- Notes on failed/interrupted/resumed runs.

Training and evaluation scripts already write `run_metadata.json`; keep those files with the outputs.

## 15. Cleanup and Retention

Ignored local artifacts:

```text
data/
phase1/outputs/
phase2/outputs/
research-notes/
prompt_stuff/
*.ckpt
```

Safe cleanup preview:

```powershell
.\.venv\Scripts\python.exe project_cli.py cleanup
```

Aggressive cleanup preview:

```powershell
.\.venv\Scripts\python.exe project_cli.py cleanup --aggressive
```

Apply cleanup only after reading the preview:

```powershell
.\.venv\Scripts\python.exe project_cli.py cleanup --apply
```

Actual aggressive cleanup requires an explicit keep-list decision first. Do not delete:

```text
phase1/outputs/oscd_saved_priors_fast
phase1/outputs/oscd_saved_priors_fast_eig
phase1/outputs/oscd_saved_full
phase2/outputs/runs_gpu_150ep_20251215_233309
phase2/outputs/smoke_e0_e1_20260503_040723
```

until they are either reproduced or intentionally archived elsewhere.

## 16. Troubleshooting

CUDA is false:

```powershell
& $py -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Missing prior map:

```text
FileNotFoundError: Missing prior score map
```

Fix: run Phase 1 with `--save_change_maps`, and make sure the Phase 2 config's enabled prior has a matching folder under `oscd_change_maps/<split>/<method>/`.

Output directory exists:

```text
Refusing to overwrite existing run artifacts
```

Fix: choose a new timestamped output directory, use `--resume`, or use `--overwrite_output_dir` deliberately.

State-dict mismatch:

```text
RuntimeError: Error(s) in loading state_dict
```

Fix: check that the evaluation config matches the training config, especially model type, raw/prior channel count, enabled priors, backbone, and checkpoint path.

xBD CSVs missing:

```text
data/xbd/train.csv, val.csv, test.csv absent
```

This is expected right now. xBD is future work.

Long command quoting on Windows:

- Prefer setting variables first.
- Avoid shell line continuations unless you know the shell behavior.
- In this repo, PowerShell examples use one complete command per line.

## 17. Current Evidence Anchors

Project brief:

```text
docs/PROJECT_BRIEF.md
```

Fresh liveness smoke:

```text
phase2/outputs/smoke_e0_e1_20260503_040723
```

Old full artifact to audit, not trust blindly:

```text
phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv
```

Most important next analysis:

```text
Targeted visualization plus true threshold tuning for the completed v5 core sweep. Current per-city/per-seed CSV analysis is done; true threshold tuning still needs probability maps or an inference threshold-sweep.
```

## 18. SpaceNet 7 Temporal Subspace Gate

Evaluate one SpaceNet 7 AOI with the frozen rolling trajectory construction:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spacenet7-temporal-subspaces --aoi-root "data/SpaceNet7_confirmation/L15-1691E-1211N_6764_3347_13" --representations trajectory2 --preprocessing feature_centered --window 6 --rank 2 --grids 8 --output-dir "phase1/outputs/spacenet7_geometry_$tag"
```

Compute fair standardized radiometric controls without repeating SVD fitting:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spacenet7-temporal-subspaces --aoi-root "data/SpaceNet7_confirmation/L15-1691E-1211N_6764_3347_13" --window 6 --grids 8 --radiometric-normalization per_date_channel_standardize --controls-only --output-dir "phase1/outputs/spacenet7_controls_$tag"
```

Analyze already paired confirmation runs with the fixed rank fusion:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spacenet7-hybrid-analysis --input-root phase1/outputs --geometry-glob "spacenet7_confirmation_validmask_*_geometry_20260621_012400" --controls-glob "spacenet7_confirmation_validmask_*_controls_20260621_012400" --bootstrap 3000 --output-dir "phase1/outputs/spacenet7_confirmation_hybrid_$tag"
```

These commands reproduce a negative gate, not a promoted baseline. Do not tune
this construction on the confirmation AOIs. See
`docs/experiment_reports/spacenet7_temporal_subspace_validation_2026-06-21.md`.
