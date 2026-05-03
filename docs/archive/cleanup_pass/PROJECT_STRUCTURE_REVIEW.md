# Project Structure Review

Status: proposal and rationale  
Updated: 2026-05-03

This review separates what should be changed now from what should wait.

## Current Assessment

The repo's main problem is no longer missing code structure. The main problem is naming and narrative clutter:

- many historical docs competed with current truth
- output folders have legacy names
- configs mix canonical experiments and exploratory variants
- Phase 1 output names are not self-explanatory
- Phase 1 docs contain many tracked generated figures
- thesis/presentation material sits beside active research code

The code split into `phase1/` and `phase2/` is still reasonable and should not be destroyed casually.

## Keep This Top-Level Shape

```text
phase1/
phase2/
docs/
references/
presentation/
data/                 # ignored
```

This is understandable: Phase 1 creates priors; Phase 2 trains/evaluates segmenters.

## Improve Docs First

Active docs should be:

```text
docs/README.md
docs/PROJECT_MASTER_BRIEF.md
docs/PIPELINE_EXPLAINED.md
docs/ROADMAP.md
docs/RESULTS_OSCD_CORE_SWEEP_20260503.md
docs/REPRODUCIBILITY_CHEATSHEET.md
docs/ARTIFACT_INDEX.md
docs/CLEANUP_TRACKER.md
docs/PROJECT_STRUCTURE_REVIEW.md
```

Everything else should be archived or phase-specific.

This is mostly done in the current cleanup pass.

## Rename Outputs Later

Current legacy names:

```text
phase1/outputs/oscd_saved_priors_fast/
phase1/outputs/oscd_saved_full/
phase2/outputs/runs_gpu_150ep_20251215_233309/
```

Clearer future names:

```text
phase1/outputs/priors_oscd_core/
phase1/outputs/priors_oscd_classical_full/
phase2/outputs/sweep_oscd_core_3seed_150ep_<timestamp>/
```

Do not rename existing output folders during an active analysis. Instead, document the mapping in `ARTIFACT_INDEX.md`.

## Config Cleanup Later

Current config names are serviceable but uneven:

```text
oscd_seg_baseline.yaml
oscd_seg_E1_raw_ds.yaml
oscd_seg_priors.yaml
oscd_seg_siamese.yaml
```

Better future pattern:

```text
phase2/configs/oscd/core/E0_raw_unet.yaml
phase2/configs/oscd/core/E1_raw_ds_unet.yaml
phase2/configs/oscd/core/E1b_raw_ds_cross_unet.yaml
phase2/configs/oscd/core/E2_raw_pca_unet.yaml
phase2/configs/oscd/core/E3_raw_ds_pca_unet.yaml
phase2/configs/oscd/core/S0_raw_siamese.yaml
phase2/configs/oscd/extended/
phase2/configs/damage/
```

Do this only after adding a compatibility note or updating scripts, because the sweep script currently references existing config paths.

## Code Cleanup Later

Potential improvements:

- extract shared model-building logic used by train/eval/viz
- extract channel-count inference into one module
- add a result-analysis script for sweep summaries
- add a config validator for enabled priors vs available prior-map folders
- add tests for OSCD dataset sample shapes and prior channel loading

Do not refactor training/evaluation while producing the core result. Result stability matters more right now.

## Output Cleanup Later

Only delete output folders after:

1. `ARTIFACT_INDEX.md` is complete.
2. v5 sweep result is committed in docs.
3. interrupted progress-bar test runs are explicitly marked delete-later.
4. the user approves deletion.

Likely delete-later folders:

```text
phase2/outputs/sweep_core_150ep_E0_E1_repro_20260503_042530/
phase2/outputs/sweep_core_150ep_repro_v3_20260503_044813/
phase2/outputs/sweep_core_150ep_repro_v4_*/
```

Likely keep:

```text
phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422/
phase2/outputs/runs_gpu_150ep_20251215_233309/
phase1/outputs/oscd_saved_priors_fast/
phase1/outputs/oscd_saved_full/
```

## Recommended Next Structural Commit

After the current docs/result commit:

```text
docs: add pipeline explanation and structure review
```

Then later:

```text
scripts: add sweep result analysis helper
```

Then later, only if needed:

```text
configs: organize oscd experiment configs
```
