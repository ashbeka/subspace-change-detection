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

Subspace inspection evidence:

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

Spatial subspace first run, Beirut, 2026-06-13:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-compare --city beirut --rank 6 --methods global_pixel,window128,patch3,patch5 --no-save-npy
```

Output:

```text
phase1/outputs/oscd_spatial_subspace_compare_beirut_20260613_192736/
```

| method | AUROC | AP | best F1 | Otsu F1 | raw-L2 corr |
|---|---:|---:|---:|---:|---:|
| raw_l2 | 0.8327 | 0.4054 | 0.4335 | 0.1740 | 1.0000 |
| pca_diff | 0.9183 | 0.5284 | 0.5105 | 0.4910 | 0.8667 |
| global_pixel | 0.7287 | 0.0754 | 0.1376 | 0.0000 | 0.1408 |
| window128s64mean | 0.7135 | 0.0781 | 0.1300 | 0.0000 | 0.2102 |
| patch3 | 0.8413 | 0.1887 | 0.2722 | 0.1767 | 0.4280 |
| patch5 | 0.8923 | 0.2729 | 0.3473 | 0.3429 | 0.4962 |

Interpretation:

- Patch-vector DS is clearly better than current global pixel DS on Beirut in this first run.
- Patch5 is stronger than raw L2 on AUROC and Otsu F1, but still weaker than PCA-diff.
- Local-window DS with `128x128` windows does not help in this run.
- This justifies multi-city spatial DS comparison; it is not yet a thesis-level claim.

Spatial subspace core5 sweep, 2026-06-14:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --output-dir phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823 --no-save-npy --continue-on-error
```

Tracked report:

```text
docs/experiment_reports/oscd_spatial_subspace_sweep_core5_2026-06-14.md
```

Ignored output:

```text
phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/
```

Design:

```text
cities: beirut,dubai,lasvegas,milano,norcia
rank4_core: global_pixel,patch3,patch5
rank6_spatial: global_pixel,window128,patch3,patch5
rank8_core: global_pixel,patch3,patch5
baselines included in each run: raw_l2,pca_diff
```

Mean results across the five cities:

| config | method | mean AUROC | mean AP | mean Otsu F1 | mean best F1 |
|---|---|---:|---:|---:|---:|
| rank8_core | pca_diff | 0.8733 | 0.3953 | 0.3403 | 0.4112 |
| all configs | raw_l2 | 0.8263 | 0.3550 | 0.2707 | 0.3849 |
| rank8_core | patch5 | 0.7783 | 0.1966 | 0.0998 | 0.2586 |
| rank8_core | patch3 | 0.7920 | 0.1897 | 0.0815 | 0.2572 |
| rank4_core | global_pixel | 0.6633 | 0.1162 | 0.0762 | 0.1761 |
| rank6_spatial | window128s64mean | 0.6434 | 0.0802 | 0.0396 | 0.1431 |

Interpretation:

- Patch-vector DS is better than global pixel DS on average.
- Local-window DS with `128x128` windows is weak as configured.
- PCA-diff and raw L2 are still much stronger than the DS-family maps overall.
- The result supports "spatial sample construction changes DS behavior," not "DS beats classical baselines."
- Do not spend time on long Phase 2 raw+spatial-prior training until the score definition and failure modes are better understood.

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

The next experiment is not another blind sweep. The core5 sweep already showed that patch-vector DS is the only DS-family candidate worth studying immediately, but it is still weaker than PCA-diff/raw L2 overall.

There are now two parallel tracks:

```text
Sensei-first track:
  generate time-sequential satellite subspaces -> first/second DS magnitude -> geodesic decomposition/projection

OSCD verification track:
  explain whether spatially aware DS fixes the spatial-information weakness on a labeled benchmark
```

The Sensei-first track has priority for advisor alignment. The OSCD track remains important because it provides labels and lets the project verify whether the subspace maps correspond to changed areas.

Immediate next task:

```text
Inspect why patch DS helps in some cities and fails in others, then test whether the score definition is the bottleneck.
```

Immediate Sensei-first task:

```text
Audit one Harmonized Sentinel-2 area/date sequence and report: date count, time step, cloud/no-data filtering, band stack shape, alignment assumptions, and candidate subspace construction.
```

After that audit, implement first/second DS and geodesic-decomposition measurements on the generated date subspaces, preferably after asking Jang/Aono about the expected input format and reference implementation.

Status 2026-06-14:

- Jang and Aono were consulted about the difficulty of adapting subspace methods to this satellite-image project.
- The remaining responsibility is now ours: define the satellite sample object, generate a sequence of subspaces, and produce a small measurable first/second DS/geodesic result.

Next artifact to create:

```text
Harmonized Sentinel-2 sequence feasibility report:
  area/date selection -> downloaded or queried frames -> valid-frame table -> per-date subspace construction -> first/second DS/geodesic readiness verdict
```

This artifact should answer Sensei's concrete questions before any new model training:

- What area is used?
- How many time frames are available?
- What is the time step or date spacing?
- Which Sentinel-2 bands are used?
- What cloud/no-data filtering is applied?
- What is one subspace in this setting?
- Are first/second DS and geodesic decomposition ready to run on the resulting sequence?

Source-linked rule for the next code change:

```text
source material -> mathematical object -> satellite adaptation -> code path -> toy check -> one-city map -> thesis claim
```

Near-term implementation:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities core5 --configs "rank8_core:8:global_pixel+patch3+patch5" --resume --no-save-npy
```

That command is mainly a reproducibility/inspection entry point. The next new script should add score-definition ablations for patch DS:

```text
score = ||D^T delta||^2
score = ||D^T delta||
score = ||D^T delta||^2 / (||delta||^2 + eps)
score = normalized patch score after per-city robust scaling
```

Acceptance checks:

- each score definition has an explicit formula and code path;
- patch scores are compared against raw L2, PCA-diff, and Celik-style patch PCA-kmeans;
- maps are inspected for false positives from water, shadows, vegetation, registration, and city-specific artifacts;
- the thesis claim remains diagnostic unless DS-family maps beat or explain baselines consistently.

Concrete near-term checklist:

1. Inspect core5 comparison grids:
   - Open every `comparison_grid.png` under `phase1/outputs/oscd_spatial_subspace_sweep_core5_20260614_004823/runs/`.
   - Record where patch5 helps and where it fails.
   - Specifically check Beirut, Dubai, Las Vegas, Milan, and Norcia for water, vegetation, shadows, registration offsets, city texture, and low-change-label effects.
   - Outcome: short failure-mode table before more code changes.

2. Patch-score definition ablation:
   - Compare squared projection norm, unsquared projection norm, normalized projection ratio, residual energy, and per-city robust scaling.
   - Keep the sample definition fixed first: patch3 and patch5 at rank 8.
   - Baselines in the same table: raw L2/CVA and PCA-diff.
   - Outcome: decide whether DS is weak because of sample construction or because the scalar score is poorly calibrated.

3. Celik pressure baseline:
   - Compare patch-vector DS directly against Celik PCA-kmeans because Celik already uses local difference-image patches.
   - This is the closest classical spatial-patch pressure baseline, not just another optional method.
   - Outcome: if Celik beats patch DS clearly, the thesis cannot claim patch DS is a strong spatial classical detector without a narrower interpretability argument.

4. IR-MAD fair-comparison audit:
   - Treat IR-MAD as a required classical multivariate baseline because it is established in remote-sensing change detection and is CCA-related.
   - Do not trust the current lightweight implementation until it is checked against Nielsen/MAD/iMAD references.
   - Outcome: either a paper-faithful IR-MAD score map in the same Phase 1 comparison table, or a clearly documented reason it is not yet fair to compare.

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

11. Prior-as-pseudo-label and flipped-pipeline experiments:
   - Motivation:
     - Current Phase 2 uses DS/PCA/CVA maps as extra input channels while training against OSCD labels.
     - A different route is to treat thresholded prior maps as pseudo-labels or auxiliary training targets.
     - This fits the label-efficient/interpretable-prior framing better than simple channel concatenation, but it is risky if the prior maps contain shadows, vegetation, water, registration artifacts, or other pseudo-change.
   - Core pseudo-label configurations:
     - `P0_supervised_only`: train U-Net or Siamese U-Net on OSCD labels only.
     - `P1_pretrain_ds_then_finetune_oscd`: pretrain on thresholded corrected/global DS pseudo-labels, then fine-tune on OSCD labels.
     - `P2_pretrain_patch_ds_then_finetune_oscd`: pretrain on thresholded best spatial DS pseudo-labels, then fine-tune on OSCD labels.
     - `P3_pretrain_pca_then_finetune_oscd`: pretrain on PCA-diff pseudo-labels, then fine-tune on OSCD labels.
     - `P4_pretrain_cva_then_finetune_oscd`: pretrain on raw L2/CVA pseudo-labels, then fine-tune on OSCD labels.
     - `P5_pretrain_irmad_then_finetune_oscd`: pretrain on audited IR-MAD pseudo-labels, then fine-tune on OSCD labels.
     - `P6_multi_prior_consensus`: use only pixels where multiple priors agree, e.g. DS + PCA-diff + CVA, as higher-confidence pseudo-labels.
     - `P7_soft_pseudo_label`: train on continuous normalized prior scores as soft targets instead of hard Otsu masks.
     - `P8_auxiliary_head`: train one head to predict OSCD labels and another head to reconstruct/predict prior maps; use the prior as an auxiliary task, not as input.
     - `P9_teacher_student`: treat prior generator or a prior-trained network as teacher; train a student network on raw pre/post inputs with distillation loss plus OSCD fine-tuning.
   - Flipped-pipeline candidate:
     - Usual pipeline: geometric method creates prior map -> neural model consumes it.
     - Flipped pipeline, likely meaning: neural model first learns/produces features or change probabilities -> subspace/DS/GDS is applied to those learned features, probability maps, or intermediate embeddings.
     - Concrete variants:
       - `F1_deep_feature_ds`: train/borrow an encoder, extract patch/tile/date embeddings, then build DS/GDS on those embeddings.
       - `F2_prediction_subspace`: train a supervised model, collect probability/logit maps across dates/cities/augmentations, then build subspaces over model responses to analyze uncertainty/change structure.
       - `F3_error_subspace`: compare model false positives/false negatives with DS/PCA/IR-MAD maps to build subspaces of failure modes.
       - `F4_prior_from_model_residual`: use the model's residual/error map to guide where geometric DS should be recomputed locally.
   - Required safeguards:
     - Do not run this until Phase 1 prior maps are audited and aligned.
     - Always compare pseudo-label quality against OSCD labels before training.
     - Report pseudo-label precision/recall/F1, changed-pixel rate, and city-wise bias before using them as targets.
     - Compare hard Otsu threshold, validation-selected threshold, percentile threshold, and soft-label variants.
     - Keep raw supervised baseline in every table.
   - Decision:
     - If pseudo-label pretraining helps under low-label settings, it supports a label-efficient prior framing.
     - If it hurts, that is evidence that current priors are useful only as diagnostics or require better spatial/temporal construction.

12. DS scalar change-map construction:
   - Current DS prior is `||D^T (x_post - x_pre)||^2`.
   - Compare squared projection norm, unsquared norm, normalized projection energy, residual energy, and ratios such as `||D^T delta||^2 / ||delta||^2`.
   - Compare per-city vs global normalization.
   - Compare Otsu thresholding, validation-selected thresholds, and no thresholding before supervised U-Net input.
   - Check whether the scalar score map agrees with the reconstructed DS norm map from the projection-visualization task.

13. Phase 1 score-normalization audit:
   - Compare raw scores, percentile-clipped scores, and min-max-normalized scores.
   - Report whether normalization changes AUROC/PR-AUC, Otsu thresholds, best F1/IoU, and Phase 2 prior behavior.
   - Keep the wording clear: score normalization is an engineering step, not DS theory.

14. Saved-prior alignment audit:
   - For a few cities, verify prior-map shape, city name, split, and spatial alignment against OSCD pre/post tiles and masks.
   - This is needed because smoke checks only proved one patch/channel load, not full prior alignment.

15. Object-level descriptor feasibility audit:
   - Define the object unit first: building polygon, greenhouse polygon, connected component, proposal mask, local patch, or tile.
   - Candidate datasets/use cases: xBD/xBD-S12 for buildings/damage, greenhouse mapping data for agricultural structures, or ChangeOS/object-level semantic change resources.
   - For each object, test whether a descriptor/subspace should represent pre state, post state, or pre-to-post change.
   - Required before implementation: available labels/proposals, evaluation unit, baseline object classifier/change detector, and a reason this is better than pixel-level OSCD.

16. Research reset decision gate:
   - After the spatial DS audit, decide whether the thesis is:
     - spatially aware DS for binary multispectral change;
     - interpretable classical priors for supervised CD;
     - pseudo-change diagnosis;
     - multi-date GDS/KGDS;
     - object-level/greenhouse/building pivot;
     - or an empirical benchmark showing where subspace priors help or fail.
   - Do not run another long U-Net sweep until this decision is made.

17. MultiSenGE pairing and seasonality audit:
   - Compare earliest/latest pairing against within-season pairings and date-windowed pairings.
   - Add snow/cloud/invalid-scene checks before DS/PCA-diff map generation.
   - If using multiple dates, test whether first-order DS, second-order DS, GDS, or KGDS gives a more interpretable progression signal.
   - Report whether visual changes are likely semantic land-cover change or seasonal/radiometric change.

18. IR-MAD fair-comparison audit:
   - Sources:
     - Nielsen 2007 regularized IR-MAD paper;
     - Nielsen/Conradsen MAD/CCA formulation;
     - Google Earth Engine MAD/iMAD tutorial and Mort Canty reference material if used.
   - Why it matters:
     - IR-MAD is a mature multivariate remote-sensing change detector.
     - It is based on CCA/MAD variates and is therefore close enough to DS to be a serious comparison pressure baseline.
     - Sensei and seminar feedback already pushed toward CCA, so ignoring IR-MAD would weaken the thesis.
   - Audit before claims:
     - Check whether `phase1/baselines/ir_mad.py` solves the correct CCA/generalized eigenproblem.
     - Verify whether iterative reweighting emphasizes likely unchanged observations correctly.
     - Recheck band selection, normalization, covariance regularization, subsampling seed, convergence behavior, chi-square weighting, and threshold calibration.
     - Test equal-image/no-change toy behavior and a simple synthetic changed-pixel case.
   - Fair comparison:
     - Compare against raw L2/CVA, PCA-diff, Celik PCA-kmeans, global canonical DS, patch3 DS, and patch5 DS.
     - Use the same OSCD cities, valid masks, score normalization policy, AUROC, PR-AUC/AP, best F1/IoU, Otsu F1/IoU, raw-L2 correlation, and visual grids.
     - Inspect whether IR-MAD reduces pseudo-change from radiometric/background effects better than DS or PCA-diff.
   - Decision:
     - If IR-MAD beats DS, use that as honest evidence that DS needs a spatial, nonlinear, temporal, or interpretability-specific justification.
     - If patch DS complements IR-MAD on specific false-positive/false-negative modes, that may support an interpretable hybrid-prior framing.
     - Do not claim IR-MAD is weak from old runs unless this audit supports it.

19. Multi-date / period-subspace DS feasibility audit:
   - Check datasets with enough aligned dates per location.
   - Compare earliest/latest, adjacent, same-season, and period-window pairings.
   - Test first-order DS before second-order DS, GDS, or KGDS.
   - Do not use unlabeled MultiSenGE visuals as performance evidence without an evaluation proxy.

20. Band-group attribution audit:
   - Compute DS basis energy by Sentinel-2 band or band group.
   - Compare VIS, red-edge, NIR, SWIR, and atmospheric bands.
   - Use this to explain whether maps are likely surface change, vegetation/soil moisture, or atmospheric artifact.

21. SSC change-type clustering pilot:
   - Only after the spatial DS audit.
   - Input candidates: raw delta, DS projection coefficients, PCA-diff features, patch/deep features.
   - Output candidates: unsupervised change-type clusters, pseudo-labels, auxiliary channels, or a strong unsupervised baseline.
   - Must define cluster count selection and validation before implementation.

22. Greenhouse application feasibility audit:
   - Treat abandoned greenhouse mapping as a possible application, not current evidence.
   - Define the task first: object mapping, abandonment classification, change detection, or temporal condition scoring.
   - Check whether labels, dates, and evaluation metrics exist before connecting it to DS/KDS/GDS.

23. Deep-feature subspace pilot:
   - Inspired by Mahyub et al. 2024 Signal Latent Subspace, not currently implemented.
   - Extract latent features from a remote-sensing CNN, U-Net encoder, or foundation model.
   - Build subspaces from patch/tile/date latent features instead of raw 13-band pixel vectors.
   - Compare latent-feature DS/GDS against raw spectral DS, local/window DS, PCA-diff, and the neural baseline.
   - Consider product-Grassmann fusion only if there are clearly defined feature factors, such as spectral, spatial, temporal, and prior-map factors.
   - This is the more explicit hybrid route: geometrical representation over learned features instead of only raw bands.

24. Multiscale subspace pyramid pilot:
   - Run only after global/window/patch DS gives a baseline.
   - Source status: Senpai idea inspired by wavelets/JPEG/Green Learning; exact formal source still needs verification.
   - Initial levels: `1x1`, `2x2`, `4x4`; add `8x8` only if runtime is manageable.
   - Initial scale weights: equal average, then compare coarse-heavy and fine-heavy weights.
   - Output: per-level maps, weighted map, runtime, block-artifact inspection, and metrics against OSCD.
   - Baselines: global canonical DS, local-window DS, raw L2/CVA, PCA-diff.

25. Temporal subspace literature pilots:
   - RTW/Deep RTW pilot: for MultiSenGE or Harmonized Sentinel-2 L2A sequences, randomly sample ordered date subsequences, build sequence-hypothesis subspaces, and compare them to same-season or event-window references.
   - SFA/SFS pilot: learn slowly varying temporal components from aligned date sequences, then test whether residuals or slow-feature subspaces separate seasonal drift from abrupt land-cover change.
   - Product-Grassmann/Hankel pilot: represent one patch as multiple subspace factors, such as spectral, spatial, and temporal/Hankel factors, then use geodesic distances for clustering or anomaly ranking.
   - G-LMSM/SLS pilot: use remote-sensing encoder features to build patch/date latent subspaces, then compare hand-built DS maps against learned Grassmann subspace matching.
   - Shape-subspace attribution pilot: adapt the human-motion DS idea by reporting which bands, patch regions, or date windows contribute most to the difference subspace.
   - Do not start these before the spatial OSCD audit; these are method-expansion tracks, not current evidence.

26. xBD-S12 metric protocol audit:
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
