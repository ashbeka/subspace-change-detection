# DS Damage Segmentation

Current truthful scope: **Sentinel-2 OSCD binary change segmentation with unsupervised change priors**.

This repository studies whether interpretable unsupervised change maps, especially Difference Subspace (DS) priors, can improve supervised OSCD Sentinel-2 binary change segmentation. The codebase currently uses two workflow folders:

- `phase1/`: generate unsupervised change-score maps from pre/post Sentinel-2 imagery.
- `phase2/`: train and evaluate supervised binary segmentation models using raw pre/post bands, optionally with Phase 1 prior channels.

Those names are pragmatic code organization, not fixed thesis doctrine. The real conceptual split is geometric/classical prior generation versus neural segmentation or downstream learning.

The project does **not** currently implement end-to-end disaster damage segmentation, xBD/xBD-S12 training, multi-class building-damage labels, or building-instance damage metrics. Those are future-work directions until implemented and evaluated.

## Start Here

- [Docs Index](docs/README.md): reading order and active-vs-archive rules.
- [Project Brief](docs/PROJECT_BRIEF.md): short current truth-status for scope, pipeline, evidence, next decision, and forbidden overclaims.
- [Runbook](docs/RUNBOOK.md): commands for setup, checks, Phase 1 prior generation, Phase 2 training/evaluation, sweeps, visualization, and cleanup.
- [Research Notes](notes/README.md): feedback, method understanding, literature triage, and experiment planning.

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
  configs/        OSCD core/extended configs and future damage templates
  data/           OSCD segmentation dataset and future damage adapter
  models/         U-Net, ResNet U-Net, fusion U-Net, Siamese U-Net
  train/          training loop, losses, callbacks
  eval/           stitched OSCD evaluation and comparison utilities
  viz/            prediction and prior visualization scripts
  outputs/        ignored checkpoints/evals/figures

docs/
  README.md
  PROJECT_BRIEF.md
  RUNBOOK.md
  experiment_reports/

notes/
  README.md
  feedback.md
  methods.md
  literature.md
  experiments.md
```

## Minimal Smoke Check

From the repo root on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$py=".\.venv\Scripts\python.exe"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then see [docs/RUNBOOK.md](docs/RUNBOOK.md) for the full liveness check and experiment commands.

## Main Research Question

How do interpretable unsupervised multispectral change priors, especially Difference Subspace and PCA-diff priors, affect supervised Sentinel-2 binary change segmentation on OSCD compared with raw pre/post and Siamese baselines under controlled training and stitched evaluation?

The fresh 2026-05-03 v5 core sweep did not reproduce the older raw+DS-over-raw artifact: DS alone underperformed raw-only across three seeds, while raw+DS+PCA showed a small mean IoU/F1 gain. Per-city analysis shows heterogeneous behavior, especially on `dubai`, `lasvegas`, `milano`, `brasilia`, and `norcia`; true threshold tuning still requires saved probabilities or a threshold-sweep evaluator.
