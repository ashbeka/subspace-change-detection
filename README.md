# DS Damage Segmentation

This repository is a research codebase for:

```text
Interpretable subspace-based change detection for multispectral satellite imagery.
```

The current concrete benchmark is:

```text
Sentinel-2 OSCD binary change detection with unsupervised subspace/classical prior maps.
```

This project is not finished ground truth. Code, notes, reports, and old outputs are hypotheses until checked by source material, implementation tests, experiments, advisor feedback, or external literature.

The current research question is:

```text
Can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction preserves the spatial information needed for that task?
```

## 1. If You Feel Lost, Start Here

Read in this order:

1. [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md)  
   Short current truth-status: what is implemented, what is not implemented, what not to overclaim.

2. [docs/RESEARCH_RESET_AUDIT.md](docs/RESEARCH_RESET_AUDIT.md)  
   The research reset: why the problem must be grounded, what directions are defensible, and what the minimum viable thesis path is.

3. [notes/methods.md](notes/methods.md)  
   Method understanding: how OSCD images become matrices, how DS is constructed, and why spatial information is the central concern.

4. [notes/experiments.md](notes/experiments.md)  
   Current experiment backlog, completed results, decision gates, and exact next evidence needed.

5. [notes/literature.md](notes/literature.md)  
   Papers, datasets, reference code, and what each source is useful for.

6. [docs/RUNBOOK.md](docs/RUNBOOK.md)  
   Exact setup, check, training, evaluation, and cleanup commands.

7. [docs/research_learning_map.html](docs/research_learning_map.html)  
   Interactive local research map. Public mobile version: [GitHub Pages research map](https://ashbeka.github.io/ds_damage_segmentation/).

## 2. Current Truthful Scope

Implemented now:

- OSCD Sentinel-2 pre/post loading from 13 tif bands.
- Classical prior maps: raw spectral L2/CVA, PCA-diff, Celik-style PCA-kmeans, lightweight IR-MAD, DS variants.
- Corrected canonical first-order DS path.
- Spatial DS comparison:
  - global pixel DS;
  - local-window DS;
  - patch-vector DS.
- Phase 2 supervised binary segmentation using U-Net/Siamese-style models with raw bands and optional prior channels.
- Central CLI wrapper: [project_cli.py](project_cli.py).

Not implemented as a completed claim:

- End-to-end disaster damage segmentation.
- xBD/xBD-S12 building damage-level evaluation.
- Multi-class building damage metrics.
- Full KDS/KGDS satellite pipeline.
- Verified temporal GDS/KGDS on MultiSenGE or Harmonized Sentinel-2.
- A validated new "satellite subspace method."

## 3. Current Most Important Result

First spatial DS comparison on Beirut:

```text
phase1/outputs/oscd_spatial_subspace_compare_beirut_20260613_192736/
```

Short interpretation:

- Global pixel DS performed poorly.
- Patch-vector DS was clearly better than global pixel DS.
- Patch5 beat raw L2 on AUROC and Otsu F1.
- PCA-diff still outperformed DS-family maps on Beirut.
- This is one-city evidence only; it is not a thesis claim yet.

The next experiment must check whether this holds across more OSCD cities.

## 4. What To Do Next

Immediate command for the current experiment on one city:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city beirut --rank 6 --methods global_pixel,window128,patch3,patch5 --no-save-npy
```

After that, repeat for several cities, then compare the CSV outputs. Do not start another long U-Net sweep until spatial DS has stable multi-city evidence.

Current priority order:

1. Multi-city spatial DS comparison.
2. PCA rank sensitivity: ranks 2, 4, 6, 8, 10, 12.
3. Score-definition and normalization ablation.
4. Projection/band/patch visual explanation.
5. Phase 2 raw vs raw+best-spatial-prior only if the prior survives the previous checks.

## 5. Central CLI

Use the CLI as the main project command surface:

```powershell
.\.venv\Scripts\python.exe project_cli.py
.\.venv\Scripts\python.exe project_cli.py doctor
.\.venv\Scripts\python.exe project_cli.py list all
```

Relevant current commands:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-subspace-inspect --city beirut --rank 6
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city beirut --rank 6 --methods global_pixel,window128,patch3,patch5 --no-save-npy
```

Running `project_cli.py` with no subcommand opens the interactive command center.

## 6. Repository Map

```text
phase1/
  baselines/      classical change-detection baselines
  configs/        prior-generation configs
  data/           OSCD/MultiSenGE loaders and preprocessing
  ds/             PCA and Difference Subspace scoring utilities
  eval/           evaluation, thresholds, visualization helpers
  scripts/        inspect and experiment scripts
  outputs/        ignored generated priors/results

phase2/
  configs/        OSCD segmentation and future damage templates
  data/           OSCD segmentation dataset and future damage adapter
  models/         U-Net, ResNet U-Net, fusion heads, Siamese U-Net
  train/          training loop, losses, callbacks
  eval/           stitched OSCD evaluation and comparison utilities
  viz/            prediction and prior visualization scripts
  outputs/        ignored checkpoints/evals/figures

notes/
  my_notes.md              rough personal note intake
  feedback.md              Sensei/senpai/seminar feedback
  methods.md               method and pipeline understanding
  literature.md            papers, datasets, code references
  experiments.md           experiment evidence and backlog
  reference_bookmarks.md   organized bookmark/Zotero triage
  research_paper_plan.md   thesis/paper framing

docs/
  README.md                documentation index
  PROJECT_BRIEF.md         short current truth-status
  RESEARCH_RESET_AUDIT.md  research reset and thesis path
  RUNBOOK.md               exact commands
  research_learning_map.html
  experiment_reports/
  source_records/

project_cli.py             central command wrapper
AGENTS.md                  instructions for future Codex sessions
```

## 7. Code Provenance Rule

For paper-derived or niche method files, the source trail should be visible in the file docstring:

```text
source material -> mathematical object -> satellite adaptation -> code path -> verification -> allowed claim
```

Examples:

- DS: Fukui/Maki TPAMI 2015 principal-vector Difference Subspace, adapted to Sentinel-2 sample definitions.
- CVA/raw L2: classical Change Vector Analysis / spectral difference magnitude.
- PCA-diff/Celik: PCA over difference-image vectors or local difference patches.
- IR-MAD: Nielsen Iteratively Reweighted Multivariate Alteration Detection, currently lightweight and requiring verification before strong claims.
- U-Net/Siamese U-Net: segmentation baselines adapted from U-Net and FC-Siamese change-detection literature.

Generic utilities do not need citation-level comments. Niche method implementations do.

## 8. Forbidden Overclaims

Do not claim:

- completed disaster damage segmentation;
- xBD/xBD-S12 damage evaluation;
- building damage severity classification;
- DS is invented by this project;
- DS reliably improves OSCD segmentation;
- one-city Beirut results prove the method;
- reshaping a global pixel score map means the subspace construction preserves spatial information.

Safe current phrasing:

```text
The project is testing whether spatially supported DS constructions provide useful, interpretable prior maps for multispectral satellite change detection.
```

