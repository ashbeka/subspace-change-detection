# Experiment Notes

## Table Of Contents

- [1. Current Research Question](#1-current-research-question)
- [2. Trusted Evidence](#2-trusted-evidence)
- [3. Main Completed Sweep](#3-main-completed-sweep)
- [4. Per-City Findings](#4-per-city-findings)
- [5. Immediate Next Experiment](#5-immediate-next-experiment)
- [6. Other Important Experiments To Queue](#6-other-important-experiments-to-queue)
- [7. Paused Work](#7-paused-work)
- [8. Evidence Rules](#8-evidence-rules)
- [9. Source Ingestion Summary](#9-source-ingestion-summary)

## 1. Current Research Question

```text
Can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction preserves the spatial information needed for that task?
```

The active benchmark is OSCD binary change detection.

## 2. Trusted Evidence

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

## 3. Main Completed Sweep

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

## 4. Per-City Findings

From the v5 analysis:

- `E1_raw_ds` was worst versus E0 on `dubai`, `chongqing`, and `rio`.
- `E1_raw_ds` was best on `lasvegas`.
- `E3_raw_ds_pca` was best on `milano`, `lasvegas`, and `chongqing`, but hurt `dubai`.
- `S0_siamese` was best on `brasilia` and `lasvegas`, but much worse on `norcia` and `dubai`.

Interpretation:

- Prior effects are city-dependent and metric-dependent.
- The result is not a simple "DS works" story.

## 5. Immediate Next Experiment

The next experiment is not a blind coding task. It must be implemented as a source-linked audit so the student can trace every step:

```text
source material -> mathematical object -> satellite adaptation -> code path -> toy check -> one-city map -> thesis claim
```

Minimum source material before coding:

- TPAMI 2015 DS/GDS/KDS/KGDS paper for DS terminology and projection logic.
- Current repaired DS code in `phase1/ds/pca_utils.py` and `phase1/ds/ds_scores.py`.
- Spatial/subspace related-work anchors such as Wu-Du-Zhang HSI subspace CD, LRSD_SS, and n-mode GDS, to avoid claiming that spatial subspaces are entirely new.

Implementation note:

- Record the exact sample definition for every variant.
- Record whether the variant preserves no position, local neighborhood position, regional context, object context, or tensor modes.
- Add toy/shape checks before full-city outputs.

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

Research interpretation:

- This experiment tests a strong hypothesis, not a proven claim.
- A positive result supports "spatial support matters for DS-style satellite change priors."
- A negative result is still useful because it shows that global/patch/window DS is not the right tool under this setup.
- The goal is not to beat modern deep learning directly; the first goal is to produce interpretable geometric change evidence that can later be used alone, as a prior channel, as a diagnostic feature, or as a label-efficient aid.

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

## 6. Other Important Experiments To Queue

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

5. Spatial feature-construction screen:
   - Test whether Green Learning / PixelHop-style local transforms, wavelet components, or patch-compression features preserve spatial information better than raw global pixel samples.
   - Do not implement a full new framework at first. Start with a small feature extraction audit on one OSCD city and compare maps against raw L2, PCA-diff, and corrected DS.
   - Record whether the feature construction improves pseudo-change behavior, spatial continuity, or interpretability.
   - Include a `multiscale_subspace_pyramid` candidate inspired by Senpai's image-compression/wavelet idea:
     - whole image: `1` subspace;
     - `2x2` grid: `4` subspaces;
     - `4x4` grid: `16` subspaces;
     - optional `8x8` grid: `64` subspaces.
   - For each cell, fit pre/post subspaces and either assign a cell-level DS score or score pixels inside the cell using that cell's basis.
   - Compare pyramid aggregation against local-window DS to see whether explicit multiscale structure helps or just creates block artifacts.
   - Do not call it Green Learning in results until the exact Green Learning / PixelHop source is matched to the implementation.

6. Reference-code method family screen:
   - Review the bundled DS, MagTool, and MATLAB Subspace Toolbox code as an implementation menu, not as runtime dependencies.
   - Candidate families: KPCA/KDS, CCA/KCCA, S3CCA/TRCCA/KOTRCCA, MSM/KMSM/CMSM/KCMSM, RTW/SFA/temporal tensor methods, and Grassmann magnitude/decomposition tools.
   - For each family, define which satellite object it operates on: global 13-band pixels, `k x k x 13` patches, local windows, date subspaces, or deep-feature embeddings.
   - Produce a one-city smoke result before promoting any family to the main research path.
   - Required comparison floor: raw L2, PCA-diff, corrected canonical/eig DS, and the current best supervised baseline when applicable.

7. Harmonized Sentinel-2 sequence feasibility audit:
   - Use Sensei's recent suggestion as a concrete dataset check.
   - Count frames, date spacing, cloud/no-data rate, available bands, and spatial alignment for a small region.
   - Decide whether it supports first-order DS only, second-order DS, GDS/KGDS, or geodesic projection.
   - Do not run a full temporal method before this dataset audit.

8. MultiSenGE GDS/KGDS prototype:
   - Build one subspace per date.
   - Use GDS/KGDS to extract multi-date difference directions.
   - Interpret through clustering, temporal grouping, or weak labels.

9. Pseudo-change vs real-change audit:
   - Explicitly sample or annotate cases such as shadows, seasonal vegetation, registration differences, clouds, sensor effects, construction, and land-cover conversion.
   - Compare which methods highlight each type: raw L2, PCA-diff, canonical/eig DS, local/window DS, KDS/KPCA, and possible SSC clusters.
   - This is a candidate problem-driven thesis direction if it produces clearer evidence than generic "DS improves segmentation."

10. Phase 2 follow-up after spatial audit:
   - E0 raw-only.
   - raw + global canonical DS.
   - raw + best spatially aware DS prior.
   - Treat this as "geometry complements learning," not "geometry beats learning."
   - Only run this after Phase 1 maps are visually and metrically meaningful.

11. DS scalar change-map construction:
   - Current DS prior is `||D^T (x_post - x_pre)||^2`.
   - Compare squared projection norm, unsquared norm, normalized projection energy, residual energy, and ratios such as `||D^T delta||^2 / ||delta||^2`.
   - Compare per-city vs global normalization.
   - Compare Otsu thresholding, validation-selected thresholds, and no thresholding before supervised U-Net input.
   - Check whether the scalar score map agrees with the reconstructed DS norm map from the projection-visualization task.

12. Phase 1 score-normalization audit:
   - Compare raw scores, percentile-clipped scores, and min-max-normalized scores.
   - Report whether normalization changes AUROC/PR-AUC, Otsu thresholds, best F1/IoU, and Phase 2 prior behavior.
   - Keep the wording clear: score normalization is an engineering step, not DS theory.

13. Saved-prior alignment audit:
   - For a few cities, verify prior-map shape, city name, split, and spatial alignment against OSCD pre/post tiles and masks.
   - This is needed because smoke checks only proved one patch/channel load, not full prior alignment.

14. Object-level descriptor feasibility audit:
   - Define the object unit first: building polygon, greenhouse polygon, connected component, proposal mask, local patch, or tile.
   - Candidate datasets/use cases: xBD/xBD-S12 for buildings/damage, greenhouse mapping data for agricultural structures, or ChangeOS/object-level semantic change resources.
   - For each object, test whether a descriptor/subspace should represent pre state, post state, or pre-to-post change.
   - Required before implementation: available labels/proposals, evaluation unit, baseline object classifier/change detector, and a reason this is better than pixel-level OSCD.

15. Research reset decision gate:
   - After the spatial DS audit, decide whether the thesis is:
     - spatially aware DS for binary multispectral change;
     - interpretable classical priors for supervised CD;
     - pseudo-change diagnosis;
     - multi-date GDS/KGDS;
     - object-level/greenhouse/building pivot;
     - or an empirical benchmark showing where subspace priors help or fail.
   - Do not run another long U-Net sweep until this decision is made.

16. MultiSenGE pairing and seasonality audit:
   - Compare earliest/latest pairing against within-season pairings and date-windowed pairings.
   - Add snow/cloud/invalid-scene checks before DS/PCA-diff map generation.
   - If using multiple dates, test whether first-order DS, second-order DS, GDS, or KGDS gives a more interpretable progression signal.
   - Report whether visual changes are likely semantic land-cover change or seasonal/radiometric change.

17. IR-MAD fair-comparison audit:
   - Recheck band selection, normalization, covariance regularization, subsampling seed, and threshold calibration.
   - Do not claim IR-MAD is weak from old runs unless this audit supports it.

18. Multi-date / period-subspace DS feasibility audit:
   - Check datasets with enough aligned dates per location.
   - Compare earliest/latest, adjacent, same-season, and period-window pairings.
   - Test first-order DS before second-order DS, GDS, or KGDS.
   - Do not use unlabeled MultiSenGE visuals as performance evidence without an evaluation proxy.

19. Band-group attribution audit:
   - Compute DS basis energy by Sentinel-2 band or band group.
   - Compare VIS, red-edge, NIR, SWIR, and atmospheric bands.
   - Use this to explain whether maps are likely surface change, vegetation/soil moisture, or atmospheric artifact.

20. SSC change-type clustering pilot:
   - Only after the spatial DS audit.
   - Input candidates: raw delta, DS projection coefficients, PCA-diff features, patch/deep features.
   - Output candidates: unsupervised change-type clusters, pseudo-labels, auxiliary channels, or a strong unsupervised baseline.
   - Must define cluster count selection and validation before implementation.

21. Greenhouse application feasibility audit:
   - Treat abandoned greenhouse mapping as a possible application, not current evidence.
   - Define the task first: object mapping, abandonment classification, change detection, or temporal condition scoring.
   - Check whether labels, dates, and evaluation metrics exist before connecting it to DS/KDS/GDS.

22. Deep-feature subspace pilot:
   - Inspired by Mahyub et al. 2024 Signal Latent Subspace, not currently implemented.
   - Extract latent features from a remote-sensing CNN, U-Net encoder, or foundation model.
   - Build subspaces from patch/tile/date latent features instead of raw 13-band pixel vectors.
   - Compare latent-feature DS/GDS against raw spectral DS, local/window DS, PCA-diff, and the neural baseline.
   - Consider product-Grassmann fusion only if there are clearly defined feature factors, such as spectral, spatial, temporal, and prior-map factors.
   - This is the more explicit hybrid route: geometrical representation over learned features instead of only raw bands.

23. Multiscale subspace pyramid pilot:
   - Run only after global/window/patch DS gives a baseline.
   - Source status: Senpai idea inspired by wavelets/JPEG/Green Learning; exact formal source still needs verification.
   - Initial levels: `1x1`, `2x2`, `4x4`; add `8x8` only if runtime is manageable.
   - Initial scale weights: equal average, then compare coarse-heavy and fine-heavy weights.
   - Output: per-level maps, weighted map, runtime, block-artifact inspection, and metrics against OSCD.
   - Baselines: global canonical DS, local-window DS, raw L2/CVA, PCA-diff.

24. Temporal subspace literature pilots:
   - RTW/Deep RTW pilot: for MultiSenGE or Harmonized Sentinel-2 L2A sequences, randomly sample ordered date subsequences, build sequence-hypothesis subspaces, and compare them to same-season or event-window references.
   - SFA/SFS pilot: learn slowly varying temporal components from aligned date sequences, then test whether residuals or slow-feature subspaces separate seasonal drift from abrupt land-cover change.
   - Product-Grassmann/Hankel pilot: represent one patch as multiple subspace factors, such as spectral, spatial, and temporal/Hankel factors, then use geodesic distances for clustering or anomaly ranking.
   - G-LMSM/SLS pilot: use remote-sensing encoder features to build patch/date latent subspaces, then compare hand-built DS maps against learned Grassmann subspace matching.
   - Shape-subspace attribution pilot: adapt the human-motion DS idea by reporting which bands, patch regions, or date windows contribute most to the difference subspace.
   - Do not start these before the spatial OSCD audit; these are method-expansion tracks, not current evidence.

25. xBD-S12 metric protocol audit:
   - Only if xBD-S12 is promoted from warm extension.
   - Check whether to use xBD-S12-style `F1loc`, `F1dmg`, and `F1comp` with invalid masks/building buffers, or standard pixel IoU/F1.
   - Do not mix OSCD binary pixel metrics with damage-localization metrics without explaining the task difference.

## 7. Paused Work

Pause until OSCD subspace construction is settled:

- xBD/xBD-S12 damage mapping.
- abandoned-greenhouse mapping as a main benchmark.
- foundation models or large new architectures.
- long U-Net sweeps using unverified DS priors.
- broad claims about disaster response.

## 8. Evidence Rules

The old archive documents were reviewed in repeated passes: inventory, useful-claim extraction, active-note gap check, keyword/risk coverage, and final functional coverage across commands, artifacts, code references, risks, tasks, and results.

The `docs/archive/` folder was removed after the final review. The tracked files remain recoverable through Git/GitHub history. Active truth now lives in `notes/`, `docs/PROJECT_BRIEF.md`, `docs/RUNBOOK.md`, and accepted result reports under `docs/experiment_reports/`.

Testing policy:

- Keep narrow formula-verification tests for paper-derived methods, especially DS/KDS dimensionality, equal-subspace behavior, RKHS normalization, and projection-energy semantics.
- Do not overbuild broad application/regression tests around old code paths while the research problem and dataset choice are still being reframed.
- When adding a new baseline or method, pair it with a small toy test or reference-code comparison before using its outputs as research evidence.

## 9. Source Ingestion Summary

Detailed historical ingestion ledgers were moved out of the active experiment note to keep this file readable.

Active rule:

- Use this file for current experiment evidence, queued audits, and decision gates.
- Use `docs/source_records/final_organization_2026-06-12/prior_ingestion_ledgers.md` only when checking historical provenance.
- Old archive/research-notes/phase-doc claims are historical unless they are repeated above as current evidence or current tasks.
