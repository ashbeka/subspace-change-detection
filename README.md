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

## Quick Links

- [1. If You Feel Lost, Start Here](#1-if-you-feel-lost-start-here)
- [2. Current Truthful Scope](#2-current-truthful-scope)
- [3. Current Most Important Result](#3-current-most-important-result)
- [4. What To Do Next](#4-what-to-do-next)
- [5. Central CLI](#5-central-cli)
- [6. Repository Map](#6-repository-map)
- [7. Code Provenance Rule](#7-code-provenance-rule)
- [8. Forbidden Overclaims](#8-forbidden-overclaims)

## 1. If You Feel Lost, Start Here

Read in this order:

1. [docs/CURRENT_RESEARCH_DIRECTION.md](docs/CURRENT_RESEARCH_DIRECTION.md)
   The current control panel: what we are doing, strongest evidence, safe claims, and the next gate.

2. [docs/RESEARCH_LANES_AND_DECISION_GATES.md](docs/RESEARCH_LANES_AND_DECISION_GATES.md)
   Candidate research directions ranked as lanes with win axes and continue/pause/close decisions.

3. [docs/EXPERIMENT_RESULTS_LEDGER.md](docs/EXPERIMENT_RESULTS_LEDGER.md)
   Compact table of what we tried, what happened, and which results matter.

4. [docs/METHODS_AND_IMPLEMENTATION_REFERENCE.md](docs/METHODS_AND_IMPLEMENTATION_REFERENCE.md)
   Method cards, formulas, sample definitions, source-to-code trails, and implementation boundaries.

5. [docs/ADVISOR_FEEDBACK_AND_DECISIONS.md](docs/ADVISOR_FEEDBACK_AND_DECISIONS.md)
   Sensei/senpai/seminar feedback and the decisions/actions taken from it.

6. [docs/LITERATURE_DATASETS_AND_BASELINES.md](docs/LITERATURE_DATASETS_AND_BASELINES.md)
   What to read, cite, compare against, and reuse.

7. [docs/REPRODUCIBLE_COMMANDS.md](docs/REPRODUCIBLE_COMMANDS.md)
   Exact commands only.

8. [docs/PERSONAL_RESEARCH_NOTES.md](docs/PERSONAL_RESEARCH_NOTES.md)
   Rough human notes. Use this as intake, not polished truth.

The older `notes/` files, `docs/RESEARCH_RESET_AUDIT.md`, seminar drafts, and
many AI-generated synthesis documents are being absorbed into the active docs
above. They remain available during review, but they are no longer the first
reading path.

## 2. Current Truthful Scope

Implemented now:

- OSCD Sentinel-2 pre/post loading from 13 tif bands.
- Classical prior maps: raw spectral L2/CVA, PCA-diff, Celik-style PCA-kmeans, lightweight IR-MAD, DS variants.
- Corrected canonical first-order DS path.
- Spatial DS comparison:
  - global pixel DS;
  - local-window DS;
  - patch-vector DS;
  - Band-Image DS and matched projector/Gram/reconstruction controls;
  - multiscale Band-Image pyramids, wavelet subspaces, and successive Saab-DS.
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

The strongest current internal OSCD result is the frozen successive Saab-DS
experiment:

```text
docs/experiment_reports/oscd_successive_subspace_learning_ds_2026-06-23.md
```

- A pair-shared two-hop unsupervised Saab representation followed by canonical
  DS reaches official held-out test AP `0.3420`, AUROC `0.8861`, and Otsu F1
  `0.3312`.
- Smoothed PCA-diff reaches AP `0.3141`; PCA-diff and matched-feature L2 both
  reach about `0.3067`; matched cross-reconstruction reaches `0.3279`.
- Three feature-fitting seeds produce AP `0.3438`, `0.3420`, and `0.3405`.
- Literal spatial pyramids and wavelet Band-Image DS did not pass their
  positive gates.
- The result is pair-adaptive and OSCD-only. External transfer and a
  train-fitted transform are still required.

Earlier matched-control and transfer evidence remains relevant:

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
- Fixed HSI transfer is scene-dependent: Hermiston favors canonical DS, while
  Benton/Shenzhen do not and Farmland is polarity-confounded.
- On nine SpaceNet7 AOIs and 197 RGB building-appearance transitions, raw L2
  leads AP and canonical DS loses to matched cross-reconstruction.
- Spatial geometry is therefore an xBD-specific candidate-localization result,
  not yet a generic detector or justified neural prior.

## 4. What To Do Next

Read the successive spatial report first. The next evidence gate is to compare
the current pair-adaptive transform against a transform fitted only on training
cities, then reproduce the frozen method on a second labeled multispectral
change dataset.

Reproduce the current frozen OSCD comparison with the command in
`docs/RUNBOOK.md`, Section 24.

The earlier xBD-S12 result can be reproduced with:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split test --bootstrap 5000 --maps-per-event 3 --boundary-buffer 0 --output-dir phase1/outputs/xbd_s12_frozen_test_unbuffered_complete_20260622_111613
```

Do not start another long U-Net sweep yet. Rank/centering, naive fusion,
IR-MAD pressure, fixed budgets, registration, HSI transfer, and RGB
building-appearance transfer are complete.

Current priority order:

1. Implement and test a training-city-fitted successive transform against the
   current pair-adaptive version.
2. Seek a genuinely comparable labeled multispectral change dataset for a
   frozen external test.
3. Investigate seasonal/radiometric failures in Brasilia and Norcia.
4. Only after external confirmation, test successive Saab-DS as a neural prior.

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

docs/
  CURRENT_RESEARCH_DIRECTION.md
  RESEARCH_LANES_AND_DECISION_GATES.md
  METHODS_AND_IMPLEMENTATION_REFERENCE.md
  EXPERIMENT_RESULTS_LEDGER.md
  LITERATURE_DATASETS_AND_BASELINES.md
  ADVISOR_FEEDBACK_AND_DECISIONS.md
  REPRODUCIBLE_COMMANDS.md
  PERSONAL_RESEARCH_NOTES.md
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
