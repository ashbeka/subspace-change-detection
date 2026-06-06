# Project Brief

This is the current project-facing summary. It should stay short. For research thinking, use `notes/`. For commands, use `docs/RUNBOOK.md`.

## Current Scope

The broader research direction is **interpretable subspace-based change detection for multispectral satellite imagery**.

The implemented project is **Sentinel-2 OSCD binary change detection with unsupervised prior maps**.

The active pipeline is:

```text
pre/post Sentinel-2 images -> Phase 1 prior maps -> Phase 2 supervised segmentation
```

`Phase 1` and `Phase 2` are current workflow labels and folder names. They should be read as "geometric/classical prior generation" and "neural segmentation/downstream learning," not as a fixed research structure.

OSCD is the current concrete benchmark and evidence source, not a permanent boundary on the thesis. xBD, xBD-S12, MultiSenGE semantic change, Harmonized Sentinel-2 L2A, and abandoned-greenhouse mapping are future or candidate directions unless their own data pipeline, labels, and evaluation are implemented.

## Problem Statement

Current research question:

```text
Can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction preserves the spatial information needed for that task?
```

The key methodological risk is spatial information. The current global pixel DS can build a subspace, but it fits PCA from unordered 13-band pixel vectors. Pixel position is only restored after scoring.

## Implemented Pipeline

Phase 1:

- loads OSCD Sentinel-2 pre/post image pairs;
- reads the actual 13-band rectified `.tif` files, not RGB previews;
- computes prior maps such as DS, DS cross-residual, pixel/CVA difference, PCA-diff, Celik PCA-kmeans, IR-MAD, and geodesic variants;
- saves maps under ignored `phase1/outputs/`.

Phase 2:

- loads raw pre/post Sentinel-2 stacks as 26 channels;
- optionally appends Phase 1 prior maps as extra channels;
- trains binary segmentation models such as U-Net and Siamese U-Net;
- evaluates stitched city-level OSCD masks under ignored `phase2/outputs/`.

## Current Evidence

Trusted local evidence:

- OSCD data loads locally.
- Raw Phase 2 input gives 26 channels.
- Raw+DS gives 27 channels.
- Raw+DS+PCA gives 28 channels.
- U-Net forward pass works.
- CUDA is available in the local `.venv`.
- The Venus KDS/KGDS demo loads Sensei's dataset from `data/venus_tpami2015/`.

Important result:

- The 2026-05-03 controlled 3-seed sweep did **not** reproduce the old raw+DS-over-raw claim.
- `E1_raw_ds` underperformed `E0_raw_unet`.
- `E3_raw_ds_pca` slightly improved IoU/F1 but not AUROC/PR-AUC.
- Siamese raw-only remains an important baseline.

Subspace correction:

- The old residual-stack DS behaved almost like raw spectral L2 on Beirut.
- It is now treated as legacy.
- Canonical/eig DS are the cleaner linear DS paths.

## Immediate Next Decision

Before more long U-Net sweeps, run a spatial subspace audit:

```text
global pixel DS vs patch-vector DS vs local-window DS
```

This directly answers Sensei's concern about breaking spatial information.

The experiment should report AUROC, PR-AUC, best F1/IoU, Otsu F1/IoU, raw-L2 correlation, valid-mask exclusion rate, runtime, and visual maps.

## Forbidden Overclaims

Do not currently claim:

- completed disaster damage segmentation;
- building damage-level prediction;
- xBD/xBD-S12 end-to-end training or evaluation;
- DS was invented in this project;
- DS priors reliably improve OSCD segmentation;
- OSCD binary change results prove disaster damage performance;
- current global pixel DS preserves spatial structure during fitting.

## Active Reading Map

- `notes/feedback.md`: advisor/senpai feedback and what it implies.
- `notes/methods.md`: DS, GDS, KDS, KGDS, OSCD, and Phase 2 method understanding.
- `notes/literature.md`: papers, datasets, and code references.
- `notes/experiments.md`: experiment evidence and next tests.
- `notes/research_paper_plan.md`: full paper-facing argument and thesis skeleton.
- `docs/RUNBOOK.md`: exact commands.
- `docs/results/OSCD_CORE_SWEEP_2026-05-03.md`: accepted sweep result summary.

## Historical Archive

The useful knowledge from the old `docs/archive/` folder has been consolidated into the active `notes/` files and project docs. The archive folder was removed after final review; tracked historical files remain available through Git/GitHub history.
