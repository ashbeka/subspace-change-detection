# Roadmap

Status: active project-management document  
Updated: 2026-05-03

This roadmap is intentionally narrow. It prioritizes evidence that can support a thesis or paper over adding new storylines.

## Current Thesis Frame

The current defensible project is:

> Interpretable unsupervised multispectral change priors for supervised Sentinel-2 binary change segmentation.

The active benchmark is OSCD. xBD/xBD-S12 damage mapping is future work until there is an implemented dataset pipeline, model path, and damage-specific evaluation.

## Latest Controlled Experiment

Completed sweep:

```text
phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422
```

Command family:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File phase2/scripts/run_phase2_sweep.ps1 -Preset core -Epochs 150 -Seeds "1234,1235,1236" -OutputTag "core_150ep_repro_v5_<timestamp>" -Retention full -ProgressBars
```

Purpose:

- Reproduce the core OSCD Phase 2 comparison.
- Test whether raw+DS improves over raw-only across 3 seeds.
- Compare DS with Siamese raw, DS cross-residual, PCA-diff, and DS+PCA.

Runs expected:

```text
6 experiment tags x 3 seeds = 18 model trainings
150 epochs each
```

Core tags:

| tag | model | input |
|---|---|---|
| `E0_raw_unet` | U-Net | raw pre/post Sentinel-2 |
| `S0_siamese` | Siamese U-Net | raw pre/post Sentinel-2 |
| `E1_raw_ds` | U-Net | raw + DS projection |
| `E1b_raw_ds_cross` | U-Net | raw + DS cross-residual |
| `E2_raw_pca` | U-Net | raw + PCA-diff |
| `E3_raw_ds_pca` | U-Net | raw + DS projection + PCA-diff |

## Result Status

The sweep completed all 18 runs. Main result:

```text
E1_raw_ds did not beat E0_raw_unet across the three seeds.
```

Mean test IoU/F1:

```text
E0_raw_unet:    IoU 0.2396 +/- 0.0037, F1 0.3588 +/- 0.0086
E1_raw_ds:      IoU 0.2213 +/- 0.0087, F1 0.3282 +/- 0.0105
E3_raw_ds_pca:  IoU 0.2460 +/- 0.0062, F1 0.3663 +/- 0.0068
S0_siamese:     IoU 0.2501 +/- 0.0337, F1 0.3645 +/- 0.0384
```

Detailed result audit:

```text
docs/RESULTS_OSCD_CORE_SWEEP_20260503.md
```

## Immediate Next Tasks

1. Inspect per-city metrics for E0, E1, E3, and S0.
2. Generate qualitative figures for representative cities.
3. Decide whether to run threshold tuning.
4. Decide whether to run E4/E5/E6 classical priors in the same 3-seed controlled style.
5. Reframe thesis contribution away from "DS alone improves segmentation."

## Commands For Additional Result Audit

Find newest v5 sweep:

```powershell
cd E:\research_projects\DS_damage_segmentation
$runRoot = Get-ChildItem phase2/outputs -Directory | Where-Object { $_.Name -like "sweep_core_150ep_repro_v5_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$runRoot.FullName
```

Show summary:

```powershell
Import-Csv (Join-Path $runRoot.FullName "sweep_test_summary.csv") | Sort-Object seed,tag | Format-Table tag,seed,mean_iou,mean_f1,mean_auroc,mean_pr_auc
```

Compute E1 minus E0:

```powershell
$rows = Import-Csv (Join-Path $runRoot.FullName "sweep_test_summary.csv")
foreach ($seed in ($rows.seed | Sort-Object -Unique)) {
  $e0 = $rows | Where-Object { $_.seed -eq $seed -and $_.tag -eq "E0_raw_unet" } | Select-Object -First 1
  $e1 = $rows | Where-Object { $_.seed -eq $seed -and $_.tag -eq "E1_raw_ds" } | Select-Object -First 1
  [PSCustomObject]@{
    seed = $seed
    delta_iou = [double]$e1.mean_iou - [double]$e0.mean_iou
    delta_f1 = [double]$e1.mean_f1 - [double]$e0.mean_f1
    delta_auroc = [double]$e1.mean_auroc - [double]$e0.mean_auroc
    delta_pr_auc = [double]$e1.mean_pr_auc - [double]$e0.mean_pr_auc
  }
} | Format-Table
```

## Reading Path

Read only these first:

1. `README.md`
2. `docs/PROJECT_MASTER_BRIEF.md`
3. `docs/PIPELINE_EXPLAINED.md`
4. `docs/RESULTS_OSCD_CORE_SWEEP_20260503.md`
5. `docs/ROADMAP.md`
6. `docs/REPRODUCIBILITY_CHEATSHEET.md`
7. `docs/ARTIFACT_INDEX.md`

Then read code in this order:

1. `phase1/data/oscd_dataset.py`
2. `phase1/ds/pca_utils.py`
3. `phase1/ds/ds_scores.py`
4. `phase1/eval/run_oscd_eval.py`
5. `phase2/data/oscd_seg_dataset.py`
6. `phase2/models/unet2d.py`
7. `phase2/train/train_oscd_seg.py`
8. `phase2/eval/evaluate_oscd_seg.py`

Archived docs are historical context, not the first reading path.

## Paused Work

- xBD/xBD-S12 damage segmentation implementation.
- Multi-class damage labels.
- Building-instance damage metrics.
- Broad disaster-management platform framing.
- Foundation-model additions.
- Major code restructure.
- Output deletion.

## Next Development Tasks

After the sweep result is audited:

1. Add a small result-analysis script if manual PowerShell comparisons become repetitive.
2. Decide whether to run classical-prior baselines E4/E5/E6 in the same 3-seed controlled style.
3. Decide whether the thesis should emphasize DS specifically or the broader prior-channel framework.
4. Improve config naming only after the result interpretation is stable.
