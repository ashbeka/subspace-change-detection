# Reproducible Commands

## Purpose

Commands only. Research explanation belongs in the other active docs.

## Environment

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

## Current Priority: Reproduce DS-Prior Fusion

Status: command path must be confirmed from absorbed Claude code before running.

Goal:

```text
raw bands vs bands+DS vs no-DS priors vs DS priors vs matched-cross controls
```

Before running:

```powershell
.\.venv\Scripts\python.exe -m py_compile project_cli.py
```

## Successive Saab-DS Reports

Read:

```text
docs/experiment_reports/oscd_successive_subspace_learning_ds_2026-06-23.md
docs/experiment_reports/successive_saab_trainfit_external_gate_2026-06-23.md
```

## xBD-S12 External Evaluation

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split test --bootstrap 5000 --maps-per-event 3 --boundary-buffer 0 --output-dir phase1/outputs/xbd_s12_frozen_test_unbuffered_complete_20260622_111613
```

## Spatial Subspace Smoke Test

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city beirut --rank 6 --methods global_pixel,window128,patch3,patch5 --no-save-npy
```

## Spatial Sweep

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --no-save-npy --continue-on-error
```

## Temporal DS / Geodesic

Use the `temporal/` experiment scripts and reports until the CLI wrapper is
finalized. Main references:

```text
temporal/experiments/
docs/experiment_reports/multispectral_temporal_difference_subspaces_2026-06-19.md
docs/pending_deletion_review/old_research_material/claude_temporal/METHOD.md
```

## Cleanup Checks

Git:

```powershell
git status --short --branch
git worktree list
```

Disk overview:

```powershell
Get-ChildItem -Directory | ForEach-Object { $s=(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum; "{0,-24} {1,10:N1} MB" -f $_.Name, ($s/1MB) }
```

