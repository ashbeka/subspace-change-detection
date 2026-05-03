# DS Damage Segmentation

Current truthful scope: **Sentinel-2 OSCD binary change segmentation with unsupervised change priors**.

This repository studies whether interpretable unsupervised change maps, especially Difference Subspace (DS) priors, can improve supervised OSCD Sentinel-2 binary change segmentation. The codebase has two active phases:

- `phase1/`: generate unsupervised change-score maps from pre/post Sentinel-2 imagery.
- `phase2/`: train and evaluate supervised binary segmentation models using raw pre/post bands, optionally with Phase 1 prior channels.

The project does **not** currently implement end-to-end disaster damage segmentation, xBD/xBD-S12 training, multi-class building-damage labels, or building-instance damage metrics. Those are future-work directions until implemented and evaluated.

## Start Here

- [Project Master Brief](docs/PROJECT_MASTER_BRIEF.md): current truth-status document for scope, novelty, risks, and next steps.
- [Roadmap](docs/ROADMAP.md): active experiments, decision points, and next actions.
- [Pipeline Explained](docs/PIPELINE_EXPLAINED.md): plain-language walkthrough of data, priors, training, and evaluation.
- [Core Sweep Results](docs/RESULTS_OSCD_CORE_SWEEP_20260503.md): current 3-seed OSCD result audit.
- [Reproducibility Cheat Sheet](docs/REPRODUCIBILITY_CHEATSHEET.md): commands for environment setup, data checks, Phase 1 prior generation, Phase 2 training/evaluation, sweeps, visualization, and cleanup.
- [Artifact Index](docs/ARTIFACT_INDEX.md): generated-output inventory and cleanup gate.
- [Project Structure Review](docs/PROJECT_STRUCTURE_REVIEW.md): cleanup and structure improvement proposal.
- [Docs Index](docs/README.md): reading order and archive explanation.

## Repository Map

```text
phase1/
  baselines/      classical change detection baselines
  configs/        OSCD and MultiSenGE prior-generation configs
  data/           OSCD/MultiSenGE loaders and preprocessing
  ds/             Difference Subspace scoring utilities
  eval/           Phase 1 evaluation and visualization CLIs
  outputs/        ignored generated priors/results

phase2/
  configs/        OSCD segmentation configs and ablation configs
  data/           OSCD segmentation dataset and future damage adapter
  models/         U-Net, ResNet U-Net, fusion U-Net, Siamese U-Net
  train/          training loop, losses, callbacks
  eval/           stitched OSCD evaluation and comparison utilities
  viz/            prediction and prior visualization scripts
  outputs/        ignored checkpoints/evals/figures

docs/
  README.md
  PROJECT_MASTER_BRIEF.md
  ROADMAP.md
  PIPELINE_EXPLAINED.md
  RESULTS_OSCD_CORE_SWEEP_20260503.md
  REPRODUCIBILITY_CHEATSHEET.md
  ARTIFACT_INDEX.md
  PROJECT_STRUCTURE_REVIEW.md
  CLEANUP_TRACKER.md
  archive/
```

## Minimal Smoke Check

From the repo root on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$py=".\.venv\Scripts\python.exe"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then see [docs/REPRODUCIBILITY_CHEATSHEET.md](docs/REPRODUCIBILITY_CHEATSHEET.md) for the full liveness check and experiment commands.

## Main Research Question

How do interpretable unsupervised multispectral change priors, especially Difference Subspace and PCA-diff priors, affect supervised Sentinel-2 binary change segmentation on OSCD compared with raw pre/post and Siamese baselines under controlled training and stitched evaluation?

The fresh 2026-05-03 v5 core sweep did not reproduce the older raw+DS-over-raw artifact: DS alone underperformed raw-only across three seeds, while raw+DS+PCA showed a small mean IoU/F1 gain that still needs per-city and threshold analysis.
