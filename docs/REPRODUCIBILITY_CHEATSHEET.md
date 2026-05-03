# Reproducibility Cheat Sheet

Generated: 2026-05-03
Branch used when written: `audit/project-master-brief-20260503`
Primary environment: Windows PowerShell, repo root `E:\research_projects\DS_damage_segmentation`

This is the operational cheat sheet for reproducing the current project. It is deliberately practical: commands, expected paths, outputs, and failure checks. The project's truthful current scope is OSCD Sentinel-2 binary change segmentation with optional unsupervised prior channels. It is not yet an end-to-end xBD/xBD-S12 damage segmentation system.

## 0. Golden Rules

- Run commands from the repo root unless a section says otherwise.
- Use timestamped output folders under ignored `phase1/outputs/` and `phase2/outputs/`.
- Keep `data/`, `phase1/outputs/`, and `phase2/outputs/` out of git.
- Commit code/docs/config changes, not datasets, checkpoints, or generated maps.
- Treat old outputs as evidence to audit, not proof.
- For thesis claims, do not use 1-epoch smoke results as performance evidence.

## 1. Git Workflow

Check state:

```powershell
git status --short --branch
git log --oneline --decorate -5
```

Create a feature or audit branch:

```powershell
git switch main
git pull --ff-only
git switch -c <branch-name>
```

Commit one coherent change:

```powershell
git status --short
git add <paths>
git commit -m "<type>: <short summary>"
```

Push so it appears on GitHub:

```powershell
git push -u origin <branch-name>
```

Current audit branch already pushed:

```text
origin/audit/project-master-brief-20260503
```

## 2. Environment Setup

Activate the root virtual environment:

```powershell
cd E:\research_projects\DS_damage_segmentation
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

raw_cfg = load_cfg(ROOT / "phase2/configs/oscd_seg_baseline.yaml")
raw_ds = OSCDSegmentationDataset(ROOT / raw_cfg["dataset"]["root"], "train", raw_cfg, phase1_change_maps_root=None)
raw_item = raw_ds[0]
print("raw:", len(raw_ds), tuple(raw_item["x"].shape), tuple(raw_item["y"].shape), tuple(raw_item["valid"].shape))
model = UNet2D(in_channels=raw_item["x"].shape[0], base_channels=8, depth=3, num_classes=1)
with torch.no_grad():
    print("raw_forward:", tuple(model(raw_item["x"].unsqueeze(0)).shape))

prior_cfg = load_cfg(ROOT / "phase2/configs/oscd_seg_E1_raw_ds.yaml")
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

Note: `oscd_seg_priors.yaml` is raw+DS+PCA. `oscd_seg_E1_raw_ds.yaml` is raw+DS only.

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

## 7. Phase 2 Config Matrix

Core configs:

| tag | config | input |
|---|---|---|
| E0_raw | `phase2/configs/oscd_seg_baseline.yaml` | raw pre/post S2, 26 channels |
| E1_raw_ds | `phase2/configs/oscd_seg_E1_raw_ds.yaml` | raw + DS projection |
| E1b_raw_ds_cross | `phase2/configs/oscd_seg_E1b_raw_ds_cross.yaml` | raw + DS cross-residual |
| E2_raw_pca | `phase2/configs/oscd_seg_E2_raw_pca.yaml` | raw + PCA-diff |
| E3_raw_ds_pca | `phase2/configs/oscd_seg_priors.yaml` | raw + DS projection + PCA-diff |
| E4_raw_pixel | `phase2/configs/oscd_seg_E4_raw_pixel.yaml` | raw + pixel-diff |
| E5_raw_celik | `phase2/configs/oscd_seg_E5_raw_celik.yaml` | raw + Celik prior; requires full Phase 1 maps |
| E6_raw_irmad | `phase2/configs/oscd_seg_E6_raw_irmad.yaml` | raw + IR-MAD prior; requires full Phase 1 maps |
| S0_siamese | `phase2/configs/oscd_seg_siamese.yaml` | Siamese raw baseline |
| E0_raw_resnet | `phase2/configs/oscd_seg_baseline_resnet.yaml` | ResNet-backbone raw baseline |
| E3_raw_ds_pca_resnet | `phase2/configs/oscd_seg_priors_resnet.yaml` | ResNet-backbone raw+priors |
| E3_raw_ds_pca_fusion | `phase2/configs/oscd_seg_priors_fusion.yaml` | fusion model raw+priors |

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
& $py -m phase2.train.train_oscd_seg --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run --device cuda --epochs 1
& $py -m phase2.eval.evaluate_oscd_seg --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval_best") --device cuda

$run=Join-Path $smokeRoot "E1_raw_ds"
& $py -m phase2.train.train_oscd_seg --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run --device cuda --epochs 1
& $py -m phase2.eval.evaluate_oscd_seg --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval_best") --device cuda
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
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

E1 raw+DS:

```powershell
$run=Join-Path $outRoot "E1_raw_ds_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
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
cd E:\research_projects\DS_damage_segmentation; .\.venv\Scripts\Activate.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File phase2/scripts/run_phase2_sweep.ps1 -Preset core -Epochs 150 -Seeds "1234,1235,1236" -OutputTag "core_150ep_$(Get-Date -Format 'yyyyMMdd_HHmmss')" -Retention full -ProgressBars
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
cd E:\research_projects\DS_damage_segmentation
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

## 12. Visualize Predictions

Segmentation prediction figures:

```powershell
$run=Join-Path $outRoot "E1_raw_ds_unet"
& $py -m phase2.viz.viz_seg_predictions --device cuda --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "figs_seg") --cities test
```

Combined priors and segmentation figures:

```powershell
$run=Join-Path $outRoot "E3_raw_ds_pca_unet"
& $py -m phase2.viz.viz_oscd_combined --device cuda --config phase2/configs/oscd_seg_priors.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "figs_combined") --cities test
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
powershell -ExecutionPolicy Bypass -File clean_house.ps1 -WhatIf
```

Aggressive cleanup preview:

```powershell
powershell -ExecutionPolicy Bypass -File clean_house.ps1 -Aggressive -WhatIf
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

Master brief:

```text
docs/PROJECT_MASTER_BRIEF.md
```

Fresh liveness smoke:

```text
phase2/outputs/smoke_e0_e1_20260503_040723
```

Old full artifact to audit, not trust blindly:

```text
phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv
```

Most important next experiment:

```text
Fresh controlled E0_raw vs E1_raw_ds, ideally 3 seeds, 150 epochs, stitched city-level evaluation.
```
