# Subspace Change Detection

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
   Interactive local research map. Public mobile version after the GitHub repo rename: [GitHub Pages research map](https://ashbeka.github.io/subspace-change-detection/).

## 2. Current Truthful Scope

Implemented now:

- OSCD Sentinel-2 pre/post loading from 13 tif bands.
- Classical prior maps: raw spectral L2/CVA, PCA-diff, Celik-style PCA-kmeans, lightweight IR-MAD, DS variants.
- Corrected canonical first-order DS path.
- Spatial DS comparison:
  - global pixel DS;
  - local-window DS;
  - patch-vector DS;
  - Band-Image DS and matched projector/Gram/reconstruction controls.
- Frozen xBD-S12 event-disjoint external score-map evaluation with damage,
  building-conditional, and localization views.
- Phase 2 supervised binary segmentation using U-Net/Siamese-style models with raw bands and optional prior channels.
- Central CLI wrapper: [project_cli.py](project_cli.py).

Not implemented as a completed claim:

- End-to-end disaster damage segmentation.
- Supervised xBD/xBD-S12 building damage-level classification or segmentation.
- Multi-class building damage metrics.
- Full KDS/KGDS satellite pipeline.
- Verified temporal GDS/KGDS on MultiSenGE or Harmonized Sentinel-2.
- A validated new "satellite subspace method."

## 3. Current Most Important Result

The OSCD matched-control study and xBD-S12 external transfer are complete:

```text
docs/experiment_reports/oscd_band_image_matched_spatial_controls_2026-06-22.md
docs/experiment_reports/xbd_s12_external_validation_2026-06-22.md
```

Short interpretation:

- Band-Image DS is the strongest tested DS construction on OSCD, but remains
  below spatially filtered PCA-diff as a stand-alone map.
- On all `1,577` xBD-S12 test patches, canonical DS beats matched
  cross-reconstruction in all five unseen events.
- Band-image projector distance leads full-scene damaged-pixel retrieval and
  building localization; raw L2 leads damage-vs-intact discrimination.
- On 11 training events, projector distance beats IR-MAD full-scene AP in
  10/11 events; at a 5% pixel-review budget it retrieves 38.2% of damaged
  pixels, versus IR-MAD's 30.2%.
- On five unseen test events, the corresponding secondary budget recalls are
  24.7% versus 17.8%. These are candidate-ranking results, not segmentation or
  severity estimates.
- Object polygons confirm the same coverage/specificity tradeoff: at a 5%
  scene threshold projector hits 35.8% of damaged test buildings, but also
  26.7% of intact buildings; PCA-diff is better for damage classification.
- Controlled 0-2 pixel shifts reduce projector accuracy but retain its
  absolute candidate-ranking lead; registration invariance is not claimed.
- Spatial geometry is therefore a promising candidate-localization prior, not
  yet a stand-alone damage classifier.

## 4. What To Do Next

Read the external report first, then reproduce the frozen xBD-S12 result:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split test --bootstrap 5000 --maps-per-event 3 --boundary-buffer 0 --output-dir phase1/outputs/xbd_s12_frozen_test_unbuffered_complete_20260622_111613
```

Do not start another long U-Net sweep yet. Rank/centering, naive fusion,
IR-MAD pressure, fixed budgets, and available cloud/date checks are complete.

Current priority order:

1. Seek another independent event set before a detector claim; cloud/date
   analysis remains inconclusive rather than evidence of robustness.
2. Test a fixed projector prior in a neural model only after that gate.

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
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --no-save-npy
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
- end-to-end supervised xBD/xBD-S12 damage segmentation or severity evaluation;
- building damage severity classification;
- DS is invented by this project;
- DS reliably improves OSCD segmentation;
- one-city Beirut results prove the method;
- reshaping a global pixel score map means the subspace construction preserves spatial information.

Safe current phrasing:

```text
The project is testing whether spatially supported DS constructions provide useful, interpretable prior maps for multispectral satellite change detection.
```
