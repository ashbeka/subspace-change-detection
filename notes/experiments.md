# Experiment Notes

This file tracks experiment evidence, open experiment ideas, and decisions. It should stay practical.

## Current Research Question

```text
Can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction preserves the spatial information needed for that task?
```

The active benchmark is OSCD binary change detection.

## Trusted Evidence

Code and smoke evidence:

- OSCD data loads locally.
- Raw Phase 2 input gives `26` channels.
- Raw+DS gives `27` channels.
- Raw+DS+PCA gives `28` channels.
- U-Net forward pass returns `(1, 1, H, W)`.
- CUDA is available in the local `.venv`.
- Phase 1 and Phase 2 output CSV/JSON artifacts can be read.

Subspace audit evidence:

```text
legacy residual-stack DS: almost raw L2 on Beirut
canonical/eig DS: corrected, 13 x 6 basis for rank 6
```

Canonical prior evidence:

```text
canonical ds_projection AUROC: 0.6246
pixel_diff AUROC:              0.7559
pca_diff AUROC:                0.8134
```

Interpretation:

- Paper-faithful global canonical DS is weaker than simple baselines on OSCD in that run.
- This pushes the project toward spatial/local/kernel variants or toward a narrower diagnostic interpretation.

## Main Completed Sweep

Run:

```text
phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422
```

Setup:

```text
seeds = 1234, 1235, 1236
epochs = 150
device = CUDA
```

Mean test metrics:

| tag | IoU | F1 | AUROC | PR-AUC |
|---|---:|---:|---:|---:|
| E0_raw_unet | 0.2396 | 0.3588 | 0.8633 | 0.4331 |
| E1_raw_ds | 0.2213 | 0.3282 | 0.8534 | 0.3974 |
| E1b_raw_ds_cross | 0.2198 | 0.3343 | 0.8541 | 0.4002 |
| E2_raw_pca | 0.2218 | 0.3316 | 0.8559 | 0.3960 |
| E3_raw_ds_pca | 0.2460 | 0.3663 | 0.8525 | 0.4088 |
| S0_siamese | 0.2501 | 0.3645 | 0.8776 | 0.4461 |

Conclusion:

- E1 raw+DS did not improve over E0 raw-only.
- Old single-run E1 improvement is not reproduced.
- E3 raw+DS+PCA slightly improved IoU/F1 but not AUROC/PR-AUC.
- Siamese raw-only is an important baseline.
- The thesis should not claim "DS priors improve OSCD segmentation" without qualification.

## Per-City Findings

From the v5 analysis:

- `E1_raw_ds` was worst versus E0 on `dubai`, `chongqing`, and `rio`.
- `E1_raw_ds` was best on `lasvegas`.
- `E3_raw_ds_pca` was best on `milano`, `lasvegas`, and `chongqing`, but hurt `dubai`.
- `S0_siamese` was best on `brasilia` and `lasvegas`, but much worse on `norcia` and `dubai`.

Interpretation:

- Prior effects are city-dependent and metric-dependent.
- The result is not a simple "DS works" story.

## Immediate Next Experiment

Implement:

```text
phase1/scripts/audit_oscd_spatial_subspace.py
```

Compare:

- global pixel DS: one sample is one 13-band pixel.
- patch-vector DS: one sample is a `3x3x13` or `5x5x13` patch.
- local-window DS: one subspace per local image region such as `128x128`.

Initial local-window grid:

- window sizes: `64`, `128`, `256`.
- strides: `32`, `64`, `128`.
- aggregation: `mean`, `max`.
- inspect boundary artifacts from overlapping-window aggregation.

Start with:

```text
beirut
one dense urban city
one difficult/low-change city
```

Metrics:

- AUROC.
- PR-AUC.
- best F1 and IoU over thresholds.
- Otsu-threshold F1 and IoU.
- correlation with raw spectral L2.
- valid-mask exclusion rate for changed pixels.
- runtime.
- visual maps beside pre RGB, post RGB, and ground truth.

Decision:

- If global pixel DS is stable and competitive, keep it as a spectral-distribution baseline.
- If patch/window DS improves maps or metrics, pivot to spatially aware DS.
- If corrected DS variants remain weaker than raw/PCA-diff, stop claiming DS superiority.

Suggested first command shape:

```powershell
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe phase1/scripts/audit_oscd_spatial_subspace.py `
  --city beirut `
  --rank 6 `
  --methods global_pixel,patch3,patch5,window128 `
  --output_dir "phase1/outputs/oscd_spatial_subspace_audit_$tag"
```

Expected outputs:

- `metrics.csv`
- `run_metadata.json`
- `global_pixel_ds.png`
- `patch3_ds.png`
- `patch5_ds.png`
- `window128_ds.png`
- `raw_l2.png`
- `pca_diff.png`
- side-by-side comparison figure with pre RGB, post RGB, ground truth, and all tested maps.

Acceptance checks:

- canonical/eig DS maps are not near-identical to raw L2;
- rank sensitivity does not show rank 6 was arbitrary or unstable;
- valid-mask exclusions are negligible or explained;
- global pixel DS is either competitive with patch/window variants or its weakness is explicitly reported.

## Other Important Experiments To Queue

1. Valid-mask audit:
   - Count changed pixels excluded by `valid_pre AND valid_post`.
   - Report per city and overall.

2. PCA rank sensitivity:
   - Test ranks `2, 3, 4, 5, 6, 8, 10, 12`.
   - Compare variance thresholds `95%`, `99%`, `99.5%`.

3. OSCD projection visualization:
   - Compute `delta_x_ds = D D^T (x_post - x_pre)`.
   - Visualize band-wise maps, RGB composites, and norm maps.

4. OSCD KPCA/KDS prototype:
   - Start with sampled global KDS.
   - Then test local/windowed or patch-vector KDS.
   - Track memory and runtime carefully.

5. MultiSenGE GDS/KGDS prototype:
   - Build one subspace per date.
   - Use GDS/KGDS to extract multi-date difference directions.
   - Interpret through clustering, temporal grouping, or weak labels.

6. Phase 2 follow-up after spatial audit:
   - E0 raw-only.
   - raw + global canonical DS.
   - raw + best spatially aware DS prior.

7. DS scalar change-map construction:
   - Current DS prior is `||D^T (x_post - x_pre)||^2`.
   - Compare squared projection norm, unsquared norm, normalized projection energy, residual energy, and ratios such as `||D^T delta||^2 / ||delta||^2`.
   - Compare per-city vs global normalization.
   - Compare Otsu thresholding, validation-selected thresholds, and no thresholding before supervised U-Net input.
   - Check whether the scalar score map agrees with the reconstructed DS norm map from the projection-visualization task.

8. Phase 1 score-normalization audit:
   - Compare raw scores, percentile-clipped scores, and min-max-normalized scores.
   - Report whether normalization changes AUROC/PR-AUC, Otsu thresholds, best F1/IoU, and Phase 2 prior behavior.
   - Keep the wording clear: score normalization is an engineering step, not DS theory.

9. Saved-prior alignment audit:
   - For a few cities, verify prior-map shape, city name, split, and spatial alignment against OSCD pre/post tiles and masks.
   - This is needed because smoke checks only proved one patch/channel load, not full prior alignment.

## Paused Work

Pause until OSCD subspace construction is settled:

- xBD/xBD-S12 damage mapping.
- abandoned-greenhouse mapping as a main benchmark.
- foundation models or large new architectures.
- long U-Net sweeps using unverified DS priors.
- broad claims about disaster response.

## Archive-Ingested Evidence Rules

The old archive documents were reviewed in three passes: inventory, useful-claim extraction, and active-note gap check. The archive should now be treated as historical source material, not active truth. Do not delete it until the user explicitly approves.

### Thesis-usable run provenance

For any result used in a thesis or paper, preserve:

- config path and copied `config_used.yaml`;
- git commit hash and dirty/clean state;
- seed, epochs, device, and GPU/CPU;
- Phase 1 prior-map root;
- checkpoint path;
- `train_log.json`;
- `run_metadata.json`;
- evaluation JSON/CSV;
- exact split and city list;
- notes on interruption, resume, or overwrite.

One-epoch smoke runs prove liveness only. They are not performance evidence.

### Old artifact evidence, not reproduction

The old artifact:

```text
phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv
```

reported a single-seed E1 improvement over E0:

| artifact row | IoU | F1 | AUROC | PR-AUC |
|---|---:|---:|---:|---:|
| E0 raw-only | 0.2230 | 0.3428 | 0.8687 | 0.4315 |
| E1 raw+DS | 0.2725 | 0.4009 | 0.8743 | 0.4358 |
| E1 - E0 | +0.0495 | +0.0580 | +0.0056 | +0.0043 |

Metadata in the archive tied this old result to seed `1234`, CUDA Torch `2.9.1+cu126`, and git hash `735deac0caa1b9fe381c2ea98e09b93761534415`. The result is useful history, but the later 3-seed v5 sweep did not reproduce the E1 improvement. Treat the old artifact as an audit target, not proof.

### Old liveness and prior-generation evidence

Archive re-entry smoke evidence included:

- fast Phase 1 prior generation under `phase1/outputs/reentry_fast_priors_20260502_020139`;
- Phase 2 smoke output under `phase2/outputs/reentry_smoke_20260502_020139`;
- CUDA availability on an RTX 4070 SUPER for that re-entry run;
- fast Phase 1 test AUROC means: `pca_diff 0.8134`, `pixel_diff 0.7559`, `ds_projection 0.7551`, `ds_cross_residual 0.5563`;
- geodesic smoke output under `phase1/outputs/_smoke_reentry_geodesic`, with finite maps for a few train/val/test cities.

The old `PIPELINE_RERUN_LOG.txt` also showed one environment with CPU-only Torch (`2.9.1+cpu`, `cuda_available=False`). Any "GPU rerun" claim from that log is unsafe unless checked against metadata.

### Cleanup retention rules

Keep these until explicitly audited or intentionally retired:

```text
phase1/outputs/oscd_saved_priors_fast
phase1/outputs/oscd_saved_priors_fast_eig
phase1/outputs/oscd_saved_full
phase2/outputs/runs_gpu_150ep_20251215_233309
phase2/outputs/smoke_e0_e1_20260503_040723
phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422
```

Also audit before deleting:

```text
phase1/outputs/reentry_fast_priors_20260502_020139
phase1/outputs/_smoke_reentry_geodesic
phase1/outputs/oscd_geodesic_priors_rerun
phase1/outputs/oscd_saved_priors_fast_rerun
phase2/outputs/oscd_seg_E0_raw_rerun
phase2/outputs/oscd_seg_geodesic
phase2/outputs/reentry_smoke_20260502_020139
phase2/outputs/sweep_overnight_full_eig_3seeds_150ep_v2
```

Interrupted progress-bar or aborted sweep folders can be deleted later, but only after confirming they do not contain unique metrics, checkpoints, figures, or logs.

Likely delete-later after audit:

```text
phase2/outputs/sweep_core_150ep_E0_E1_repro_20260503_042530
phase2/outputs/sweep_core_150ep_repro_v3_20260503_044813
phase2/outputs/sweep_core_150ep_repro_v4_*
```

Do not rename old output folders during active analysis. Document the mapping instead, because old configs, summaries, and notes may refer to those legacy paths.

## Secondary Analysis Backlog

These items came from the old cleanup roadmap and remain useful, but they should follow the spatial subspace audit unless needed for a paper figure:

- Inspect per-city metrics and qualitative examples for `E0_raw_unet`, `E1_raw_ds`, `E3_raw_ds_pca`, and `S0_siamese`.
- Stratify where priors help or hurt by city type, land cover, seasonality, shadows/clouds, and change density.
- Add probability-map saving or rerun inference for true threshold tuning; current CSV-only summaries cannot do full threshold sweeps after the fact.
- Decide whether to run `E4_raw_pixel`, `E5_raw_celik`, and `E6_raw_irmad` in the same 3-seed controlled style.
- If manual PowerShell result comparison becomes repetitive, add a small result-analysis script rather than hand-copying tables.
- Add a config validator for enabled prior names versus available prior-map folders before long sweeps.
- Improve output/config naming only after result interpretation is stable.
- Extract shared channel-count/model-building logic later if repeated config mistakes appear.
- Treat dependency pinning as a reproducibility task before final paper runs.

### Reviewed archive sources

The useful material from these archive files has been folded into active notes or docs:

```text
docs/archive/cleanup_pass/ARTIFACT_INDEX.md
docs/archive/cleanup_pass/CLEANUP_TRACKER.md
docs/archive/cleanup_pass/PIPELINE_EXPLAINED.md
docs/archive/cleanup_pass/PROJECT_STRUCTURE_REVIEW.md
docs/archive/cleanup_pass/README.md
docs/archive/cleanup_pass/ROADMAP.md
docs/archive/consolidated_into_notes_20260606/RESEARCH_PLAN.md
docs/archive/consolidated_into_notes_20260606/SUBSPACE_METHOD_NOTES.md
docs/archive/reentry/ADVERSARIAL_REENTRY_AUDIT.md
docs/archive/reentry/IMPLEMENTATION_STATUS.md
docs/archive/reentry/NEXT_STEP_DECISION_MEMO.md
docs/archive/reentry/PROJECT_REENTRY_SYNTHESIS.md
docs/archive/reentry/PROJECT_RESET_DECISION.md
docs/archive/reentry/PROJECT_UNDERSTANDING_GUIDE.md
docs/archive/root_legacy/CODEBASE_AUDIT.md
docs/archive/root_legacy/PIPELINE_RERUN_LOG.txt
docs/archive/root_legacy/REMEMBER_TO_ACTIVATE_ENV.txt
docs/archive/root_legacy/RUN_PIPELINE.md
docs/archive/root_legacy/TEMP_DS_PRIMER.md
```

### Where archive content lives now

| Archive content type | Active location |
|---|---|
| Current project truth, scope, overclaim boundaries | `docs/PROJECT_BRIEF.md`, `notes/research_paper_plan.md` |
| Exact reproducibility commands and old pipeline commands | `docs/RUNBOOK.md` |
| Subspace implementation, DS/KDS/KGDS, spatial-risk explanation | `notes/methods.md`, `notes/feedback.md` |
| Sensei/senpai feedback and unanswered questions | `notes/feedback.md` |
| Experiment evidence, old artifact metrics, cleanup retention | `notes/experiments.md`, `docs/results/OSCD_CORE_SWEEP_2026-05-03.md` |
| Literature, baseline papers, and reference-code leads | `notes/literature.md` |
| Paper/thesis framing, contribution, decision gates | `notes/research_paper_plan.md` |
| Historical cleanup and structure policy | `docs/README.md`, `AGENTS.md` |
