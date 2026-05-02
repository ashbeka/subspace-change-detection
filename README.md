# DS Damage Segmentation

Current truthful scope: **Sentinel-2 OSCD binary change segmentation with unsupervised change priors**.

This repository studies whether interpretable unsupervised change maps, especially Difference Subspace (DS) priors, can improve supervised OSCD Sentinel-2 binary change segmentation. The codebase has two active phases:

- `phase1/`: generate unsupervised change-score maps from pre/post Sentinel-2 imagery.
- `phase2/`: train and evaluate supervised binary segmentation models using raw pre/post bands, optionally with Phase 1 prior channels.

The project does **not** currently implement end-to-end disaster damage segmentation, xBD/xBD-S12 training, multi-class building-damage labels, or building-instance damage metrics. Those are future-work directions until implemented and evaluated.

## Start Here

- [Project Master Brief](docs/PROJECT_MASTER_BRIEF.md): current truth-status document for scope, novelty, risks, and next steps.
- [Reproducibility Cheat Sheet](docs/REPRODUCIBILITY_CHEATSHEET.md): commands for environment setup, data checks, Phase 1 prior generation, Phase 2 training/evaluation, sweeps, visualization, and cleanup.
- [Run Pipeline](RUN_PIPELINE.md): older PowerShell run pipeline, still useful but superseded by the cheat sheet for current orientation.

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
  PROJECT_MASTER_BRIEF.md
  REPRODUCIBILITY_CHEATSHEET.md
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

Can interpretable unsupervised multispectral change priors, especially Difference Subspace priors, improve supervised Sentinel-2 binary change segmentation on OSCD compared with a raw pre/post Sentinel-2 baseline under controlled training and stitched evaluation?

The next decisive experiment is a fresh controlled E0 raw-only vs E1 raw+DS reproduction across multiple seeds.
