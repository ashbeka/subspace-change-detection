# Project Brief

## Table Of Contents

- [1. Current Scope](#1-current-scope)
- [2. Problem Statement](#2-problem-statement)
- [3. Implemented Pipeline](#3-implemented-pipeline)
- [4. Current Evidence](#4-current-evidence)
- [5. Immediate Next Decision](#5-immediate-next-decision)
- [6. Forbidden Overclaims](#6-forbidden-overclaims)
- [7. Active Reading Map](#7-active-reading-map)
- [8. Historical Archive](#8-historical-archive)

## 1. Current Scope

The broader research direction is **interpretable subspace-based change detection for multispectral satellite imagery**.

The implemented project contains two evidence tracks:

- **Sentinel-2 OSCD binary change detection with unsupervised prior maps**;
- **first/second Difference-Subspace dynamics on registered multi-date
  multispectral sequences**.

The active pipeline is:

```text
pre/post Sentinel-2 images -> Phase 1 prior maps -> Phase 2 supervised segmentation
```

`Phase 1` and `Phase 2` are current workflow labels and folder names. They should be read as "geometric/classical prior generation" and "neural segmentation/downstream learning," not as a fixed research structure.

OSCD is the main construction benchmark, not a permanent thesis boundary.
xBD-S12 now provides an implemented external event-disjoint validation path;
MultiSenGE semantic change, Harmonized Sentinel-2 L2A, abandoned-greenhouse
mapping, and other damage datasets remain candidates unless their own pipeline
and evaluation are implemented.

## 2. Problem Statement

Current active research question:

```text
Can paper-faithful first/second Difference Subspaces and geodesic decomposition
provide distinct, spatially attributable evidence of change in registered
multispectral satellite time series, after accounting for irregular cadence,
radiometric variation, and misregistration?
```

Working problem statement as of 2026-06-17:

```text
Can spatially aware Difference Subspace construction preserve the spatial structure of multispectral Sentinel-2 images well enough to produce interpretable changed-area evidence, and where does it help or fail compared with raw spectral difference, PCA-diff, Celik/IR-MAD, and neural change-detection baselines?
```

The key methodological risk is spatial information. The current global pixel DS can build a subspace, but it fits PCA from unordered 13-band pixel vectors. Pixel position is only restored after scoring.

Current research posture:

```text
geometric/subspace change evidence first -> supervised/deep-learning integration second
```

The project is not trying to prove that DS beats deep learning outright. It is testing whether spatially aware subspace geometry can produce interpretable change priors that are useful by themselves and possibly useful as inputs, diagnostics, or label-efficient aids for neural change-detection models.

The 2026-06-19 ranked problem portfolio is in
`notes/research_paper_plan.md`, Section 16. The immediate program is temporal
first/second DS plus nuisance, localization, and external-baseline tests. The
old spatial OSCD question remains a verification track rather than the primary
novelty claim.

## 3. Implemented Pipeline

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

Temporal sequence track:

- builds one band-image subspace per registered date;
- computes adjacent first DS, triple second DS, and geodesic along/orthogonal
  components;
- reports a separately labeled irregular-cadence geodesic deviation;
- maps canonical contribution energy back to common spatial coordinates;
- runs controlled radiometric, local-change, and translation injections.

## 4. Current Evidence

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

Spatial-subspace core5 result, 2026-06-14:

- Patch-vector DS outperformed current global pixel DS on mean AP.
- `window128s64mean` was weak as configured.
- PCA-diff and raw L2 still outperformed the DS-family maps overall.
- Best DS-family mean AP was `0.1966` (`rank8_core / patch5`).
- Best PCA-diff mean AP was `0.3953`.
- This supports the claim that spatial sample construction changes DS behavior, not the claim that DS improves OSCD change detection.

Band-Image DS all-city result, 2026-06-18:

- Tested all 24 local OSCD cities with ranks 6 and 8.
- `band_image_ds` treats each Sentinel-2 band image as one flattened spatial vector: `X_pre_flat, X_post_flat in R^(N_valid_pixels x 13)`.
- `flatbands` is retained only as a legacy alias.
- It was the best DS-family method in 44/48 city/rank runs.
- Mean rank-8 `band_image_ds`: AUROC `0.8412`, AP `0.2340`, best F1 `0.2928`, Otsu F1 `0.1129`.
- Mean rank-8 `band_image_norm`: AUROC `0.8412`, AP `0.2340`, best F1 `0.2928`, Otsu F1 `0.2007`.
- Mean rank-8 `pca_diff`: AUROC `0.8392`, AP `0.2541`, best F1 `0.3076`, Otsu F1 `0.2160`.
- This supports the claim that Band-Image DS is worth studying and that score scaling affects thresholdability; it does **not** support the claim that it beats PCA-diff overall.

Classical-pressure and follow-up result, 2026-06-18:

- Repaired IR-MAD: mean AUROC `0.8471`, AP `0.2138`, Otsu F1 `0.0547`.
- Corrected Celik scalar-difference PCA-k-means adaptation: mean AUROC `0.6867`, AP `0.1621`, Otsu F1 `0.1857`.
- Rank-12 Band-Image DS is the strongest tested DS setting: mean AUROC `0.8477`, AP `0.2410`, best F1 `0.3021`; it still trails PCA-diff on mean AP.
- Equal-weight PCA-diff + Band-Image + IR-MAD rank fusion reaches AUROC `0.8708` and wins 21/24 cities against PCA-diff (`p=0.00024`), but its AP improvement is not significant and Otsu F1 falls to `0.1084`.
- A fixed-grid pixel-spectral DS pyramid failed to improve core-city AP and is stopped in that form.
- Train-city changed-area calibration gives the three-way fusion held-out test F1 `0.2670` versus PCA-diff `0.2452`, but the paired 10-city improvement is not significant (`p=0.1602`). Calibration does not establish a win.
- Full interpretation: `docs/experiment_reports/oscd_spatial_ds_baseline_pressure_2026-06-18.md`.

Temporal first/second DS result, 2026-06-19:

- Eighteen focused formula/construction tests pass.
- Full-rank centered band-image subspaces are invariant to the tested global
  per-band gain/offset but respond more strongly to a one-pixel translation
  than to the tested local change.
- On five MultiSenGE patches x 23 dates, second DS correlates strongly with
  raw/NDVI/NBR temporal curvature; static labels do not validate event accuracy.
- On the published IPOL Las Vegas sequence, whole-scene contribution maps are
  diffuse and do not cleanly reproduce the source paper's changed objects.
- Full report:
  `docs/experiment_reports/multispectral_temporal_difference_subspaces_2026-06-19.md`.

Temporal replication and context result, 2026-06-20:

- The exact IPOL C implementation was run on Las Vegas, Al Wakrah, Piraeus,
  and Beijing Airport; its output is an external agreement target, not truth.
- One-date rank-2 band-image DS and multiscale tiling did not generalize as
  changed-area detectors.
- Second/time-aware geometry has slightly higher macro temporal agreement with
  IPOL event intensity than raw interpolation residual, but lower pixel AP.
- Bidirectional temporal-context DS has poor localization and is deprioritized.
- Per-band rank-2 projection novelty nearly matches raw macro AP (`0.37` vs
  `0.38`) and improves macro AUROC (`0.98` vs `0.97`), but wins AP on only one
  of four sequences.
- Controlled five-patch MultiSenGE interventions show radiometric invariance
  and persistent-vs-transient sensitivity, but severe one-pixel translation
  sensitivity. No real-label performance claim is supported.
- Gaussian scale-space reduces global-shift response for large injected events,
  but fails to make weak small events robust. Cross-scale decay distinguishes
  the tested global shift for raw and geometric methods alike, so it is a
  generic artifact cue rather than DS-specific novelty.

## 5. Immediate Next Decision

The matched OSCD spatial-null experiment and frozen xBD-S12 external transfer
are complete.

```text
Does the OSCD spatial sample-construction result transfer under a frozen
protocol to an external multispectral change benchmark?
```

Band-Image DS is the strongest tested DS construction on OSCD (rank-12 AUROC
`0.8477`, AP `0.2410`). It significantly beats PCA-matched normalized full
spatial Gram, projector-row, and cross-reconstruction scores, but remains
below PCA-diff (`0.2541`) and spatially filtered PCA (`0.2679-0.2680`).
Smoothed PCA + Band-Image + IR-MAD reaches all-city AP `0.2780`; the DS gain
beyond a cross-reconstruction substitute is supported internally; improvement
beyond smoothed PCA alone is not confirmed.

On `1,577` xBD-S12 test patches from five unseen disasters, Band-Image DS beats
its matched cross-reconstruction control in all five events. Projector distance
leads full-scene damaged-pixel retrieval and building localization, while raw
L2 leads damage-vs-intact discrimination. The defensible interpretation is
candidate-localization geometry plus a smaller DS-specific component, not a
stand-alone damage detector.

On an identical 1,100-patch sample from 11 training events, centered rank-11
projector distance beats IR-MAD full-scene AP by `+0.00814` with interval
`[+0.00470,+0.01171]` and 10/11 wins. At a fixed 5% review budget, projector
damage recall/lift is `0.382/7.64x` on training events and `0.247/4.93x` on
unseen test events. Naive projector-plus-raw score fusion is rejected.

Object-level xBD polygons confirm a coverage/specificity tradeoff. At a 5%
scene threshold, projector damaged-building recall is `0.452` on training and
`0.358` on test events, substantially above IR-MAD, but intact-building hit
rates are also high. PCA-diff is better for damage-versus-intact object
classification. The method is a candidate generator, not a damage classifier.

Immediate next decision:

```text
registration sensitivity
-> another independent event gate
-> optional fixed projector neural-prior test
```

See
`docs/experiment_reports/oscd_band_image_matched_spatial_controls_2026-06-22.md`.

External report:
`docs/experiment_reports/xbd_s12_external_validation_2026-06-22.md`.

### Completed Temporal And HSI Evidence Path

Before more long U-Net sweeps, the project completed this Sensei-aligned
temporal sequence:

```text
whole-scene first/second/geodesic DS [done]
-> IPOL NFA replication across four sequences [done]
-> bidirectional temporal-context DS/projection novelty [done]
-> controlled persistent/transient/nuisance tests [done]
-> synthetic seasonal observation subspace regime-change test [done]
-> order-aware and local real-background interventions [done]
-> independently labeled multi-temporal evaluation slice [done: SpaceNet 7 RGB]
-> registration/smoothing robustness curve [done, controlled only]
-> MOSUM/BFAST/JUST pressure baselines
```

The independently labeled SpaceNet 7 gate used repeated monthly observations
of fixed cells, rolling rank-two RGB trajectory subspaces, and persistent-ID
building first appearances. After correcting CRS84 UDM rasterization and using
transition-local valid support, second-DS orthogonal magnitude reached macro AP
`0.1127` on four untouched confirmation AOIs. A two-radiometric rank fusion
reached `0.1910`; adding geometry reduced it to `0.1747` (`-0.0163`, 95%
interval `-0.0531` to `+0.0140`). This exact RGB trajectory detector is closed.

The active claim is now narrower. First/second/geodesic DS quantities can be
computed and interpreted as sequence descriptors, but the tested quantities
are not accurate changed-area detectors. A next route must test a materially
different role, such as multispectral feature/nuisance isolation, on fresh
held-out evidence rather than retuning the failed SpaceNet 7 construction.

The subsequent RTW phase/tempo-invariance gate is also complete. On two
held-out MultiSenGE patches, RTW reached structural-change AP `0.8078`, but the
simpler order-invariant snapshot subspace reached `0.8156`; paired RTW-minus-
snapshot AP was `-0.0078` with interval `[-0.0807,+0.0435]`. RTW was highly
sensitive to relative band-phase changes, but snapshot PCA and DTW matched that
signal, while the harder marginal-matched shape target remained weak.

The subsequent official BreizhCrops follow-up added natural crop labels, two
geographic rotations, nested RTW tuning, and 22 direct controls. Selected RTW
reached AP `0.7052` and `0.6596`, below development-selected global-shift RMS
(`0.7789`, `0.7759`) and PCA cross-reconstruction (`0.8128`, `0.8264`). Every
task-specific RTW delta was negative, including hard within-group crop pairs
and timing invariance. RTW is therefore closed as the current positive
satellite route, though not invalidated for motion recognition or untested RTW
variants. The subsequent moment-factorized local hyperspectral orientation and
wavelength-attribution gate is also complete. Its frozen orientation, DS
projection, and factor-fusion scores lost to direct controls on Hermiston,
Farmland, and Shenzhen. It remains useful formula/attribution verification, not
a positive detector result.

The 2026-06-20 seasonal-observation stress test found that rank-1 geometry can
rank an injected abrupt boundary within a sequence but is weaker than simple
NDVI/singular-value controls across sequences and remains sensitive to missing
composites, drift, and translation. The next result must therefore use verified
real temporal labels; synthetic success is not enough.

The follow-up used five real MultiSenGE temporal backgrounds with exact
controlled interventions. Unordered annual PCA was confirmed to be date-order
invariant; first-difference and block-trajectory representations restore order
sensitivity. For off-grid localized mode change, Gaussian-sigma-2 local
eigenspectrum reached AP `0.688`, NDMI `0.680`, and first DS `0.456`. This is a
competitive invariance tradeoff, not superiority: eigenspectrum rejected tested
gain/offset but still responded to missing composites and translation. The
independent-label gate has now been completed and rejected the strong detection
hypothesis. The current spatial-axis experiment above supersedes further
tuning of the same RGB trajectory detector.

Treat this as a hypothesis test, not a proven claim. Spatial-spectral subspace ideas already exist in remote sensing, so the possible thesis contribution is a careful DS/GDS-style spatial-support adaptation and evaluation for Sentinel-2 change maps, not a blanket claim that spatial satellite subspaces are new.

The next ablation should report AUROC, PR-AUC, best F1/IoU, Otsu F1/IoU, raw-L2 correlation, valid-mask exclusion rate, runtime, and visual maps.

## 6. Forbidden Overclaims

Do not currently claim:

- completed disaster damage segmentation;
- building damage-level prediction;
- xBD/xBD-S12 end-to-end training or evaluation;
- DS was invented in this project;
- DS priors reliably improve OSCD segmentation;
- OSCD binary change results prove disaster damage performance;
- current global pixel DS preserves spatial structure during fitting.

## 7. Active Reading Map

- `notes/feedback.md`: advisor/senpai feedback and what it implies.
- `notes/methods.md`: DS, GDS, KDS, KGDS, OSCD, and Phase 2 method understanding.
- `notes/literature.md`: papers, datasets, and code references.
- `notes/experiments.md`: experiment evidence and next tests.
- `notes/research_paper_plan.md`: full paper-facing argument and thesis skeleton.
- `docs/RESEARCH_RESET_AUDIT.md`: critical reset audit and ranked problem/framing plan.
- `docs/RUNBOOK.md`: exact commands.
- `docs/experiment_reports/oscd_core_sweep_3seed_150epoch_2026-05-03.md`: accepted sweep result report.

## 8. Historical Archive

The useful knowledge from the old `docs/archive/` folder has been consolidated into the active `notes/` files and project docs. The archive folder was removed after final review; tracked historical files remain available through Git/GitHub history.
