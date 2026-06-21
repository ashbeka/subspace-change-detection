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
- [10. Active Temporal Difference-Subspace Study](#10-active-temporal-difference-subspace-study)

## 1. Current Research Question

```text
Can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction preserves the spatial information needed for that task?
```

The active benchmark is OSCD binary change detection.

Final working problem statement as of 2026-06-17:

```text
Can spatially aware Difference Subspace construction preserve the spatial structure of multispectral Sentinel-2 images well enough to produce interpretable changed-area evidence, and where does it help or fail compared with raw spectral difference, PCA-diff, Celik/IR-MAD, and neural change-detection baselines?
```

This is the problem to experiment on right now. It is not a claim that DS will win. It is a measurable question tied to Sensei's spatial-information criticism.

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

Band-Image DS all-city sweep, 2026-06-18:

```powershell
$cities = 'abudhabi,aguasclaras,beihai,beirut,bercy,bordeaux,brasilia,chongqing,cupertino,dubai,hongkong,lasvegas,milano,montpellier,mumbai,nantes,norcia,paris,pisa,rennes,rio,saclay_e,saclay_w,valencia'; $tag = Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities $cities --configs "rank6_flatbands:6:global_pixel+patch3+patch5+window128s64mean+flatbands;rank8_flatbands:8:global_pixel+patch3+patch5+window128s64mean+flatbands" --output-dir "phase1/outputs/spatial_ds_allcities_flatbands_$tag" --continue-on-error
```

Tracked report:

```text
docs/experiment_reports/oscd_spatial_flatbands_allcities_2026-06-18.md
```

Ignored output:

```text
phase1/outputs/spatial_ds_allcities_flatbands_20260618_001408/
```

Historical design:

```text
cities: all 24 local OSCD cities
rank6_flatbands: global_pixel,patch3,patch5,window128s64mean,flatbands
rank8_flatbands: global_pixel,patch3,patch5,window128s64mean,flatbands
baselines included in each run: raw_l2,pca_diff
```

Naming update 2026-06-18:

- `flatbands` is now a legacy alias only.
- Active method name: `band_image_ds`.
- Reason: the method treats each Sentinel-2 band image as one flattened spatial vector; `band_image_ds` is clearer and seminar-safe.

Mean results across 24 cities:

| config | method | mean AUROC | mean AP | mean Otsu F1 | mean best F1 | mean raw-L2 corr |
|---|---|---:|---:|---:|---:|---:|
| rank8_flatbands | pca_diff | 0.8392 | 0.2541 | 0.2160 | 0.3076 | 0.8588 |
| rank6_flatbands | pca_diff | 0.8370 | 0.2535 | 0.2140 | 0.3068 | 0.8564 |
| rank8_flatbands | flatbands | 0.8412 | 0.2340 | 0.1129 | 0.2928 | 0.6596 |
| rank6_flatbands | flatbands | 0.8313 | 0.2317 | 0.1093 | 0.2913 | 0.6634 |
| all configs | raw_l2 | 0.7717 | 0.2261 | 0.1802 | 0.2873 | 1.0000 |
| rank8_flatbands | patch5 | 0.7119 | 0.1331 | 0.0663 | 0.1952 | 0.4614 |
| rank6_flatbands | patch5 | 0.7002 | 0.1293 | 0.0811 | 0.1884 | 0.4686 |
| rank8_flatbands | patch3 | 0.7021 | 0.1185 | 0.0465 | 0.1797 | 0.4682 |
| rank6_flatbands | patch3 | 0.6896 | 0.1184 | 0.0602 | 0.1818 | 0.4639 |
| rank6_flatbands | global_pixel | 0.6310 | 0.0725 | 0.0272 | 0.1251 | 0.3717 |
| rank6_flatbands | window128s64mean | 0.6300 | 0.0658 | 0.0345 | 0.1184 | 0.3600 |

Winner counts:

```text
all methods by AP across 48 city/rank runs:
  pca_diff 30, flatbands 11, raw_l2 3, patch5 2, patch3 1, window128s64mean 1

DS-family only by AP across 48 city/rank runs:
  flatbands 44, patch5 2, patch3 1, window128s64mean 1
```

Interpretation:

- Band-Image DS is now the strongest DS-family candidate by a large margin.
- It is much better than global pixel DS, patch DS, and local-window DS on mean AP.
- It does **not** beat PCA-diff on mean AP, so the thesis cannot claim this construction is a better detector overall.
- It slightly beats PCA-diff on mean AUROC (`0.8412` vs `0.8392`), which suggests useful ranking signal, but Otsu F1 is weak (`0.1129`) and score calibration/pseudo-change remain problems.
- Its raw-L2 correlation is lower than PCA-diff's (`0.6596` vs `0.8588`), so it is not just a raw-L2 duplicate.
- This is the first strong evidence that Senpai's band-image sample definition is worth studying further.

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

Post-mining decision, 2026-06-21:

```text
Next gate:
  RTW phase/tempo invariance versus seasonal-cycle shape change

Conditional follow-up:
  use verified RTW/geometry quantities as interpretable features in a shallow
  classifier only if they add information beyond scalar/harmonic controls

Parallel data-gated route:
  moment-factorized local hyperspectral change attribution
```

This replaces the earlier generic proposal to test "subspace geometry as
interpretable multispectral features for shallow learning" immediately. That
proposal is too easy to validate for the wrong reason: a classifier can exploit
mean, variance, or raw radiometric features while geometry contributes nothing.
The RTW gate first tests a specific structural claim suggested by Sensei and not
answered by the completed first/second-DS experiments.

### 5.1 RTW Phase/Tempo-Invariance Gate

Research question:

```text
Can an RTW hypo-subspace suppress changes in seasonal timing/tempo while
remaining sensitive to changes in the shape of a multispectral seasonal cycle,
beyond phase-aware harmonic, Fourier, DTW/TWDTW, and non-warped subspace controls?
```

Satellite construction:

- one sample object is a pixel, field, or fixed patch with a regularized
  multiband sequence `Z=[z_1,...,z_N]`, `z_t in R^d`;
- sample `R` temporally ordered observations repeatedly and concatenate them
  into `L` time-elastic vectors `f_l in R^(dR)`;
- fit a rank-`m` PCA hypo-subspace to `F=[f_1,...,f_L]`;
- compare reference and target hypo-subspaces through the canonical-correlation
  spectrum of `X^T Y`, using `1-mean(kappa_i^2)` as the primary dissimilarity;
- do not call the construction warp-invariant until the nuisance tests show it.

Controlled regimes over real-background seasonal curves:

| Regime | Meaning | Expected RTW behavior |
|---|---|---|
| null resampling/noise | unchanged | low score |
| circular phase shift | timing nuisance | low score |
| monotone nonlinear time warp | tempo nuisance | low score |
| missing/irregular composites | acquisition nuisance | uncertainty or low score, not false change |
| amplitude/mean shift | easy change control | may respond; not the claimed novelty |
| same-phase cycle-shape deformation | target change | high score |
| crop/irrigation regime transition | real target after controlled gate | high score on held-out sites/years |

Required baselines:

- Euclidean/CVA and SAM on aligned seasonal vectors;
- harmonic regression with explicit phase terms;
- Fourier-magnitude descriptors as a strong phase-invariant null;
- DTW, time-weighted DTW, and soft-DTW where feasible;
- non-warped PCA/MSM and M-SSA/Hankel-subspace comparison;
- simple NDVI/mean/amplitude change features.

Primary metrics:

- AP and AUROC for target-change versus timing/tempo nuisance;
- false-positive rate on phase/tempo nuisances at a validation-fixed threshold;
- target-to-nuisance score ratio and its bootstrap interval;
- response curves versus phase shift, warp severity, missing observations, and
  noise;
- held-out geography/year AP and event timing error for the real gate.

Go/no-go rule:

- **Go:** RTW exceeds phase-aware harmonic, Fourier magnitude, and TWDTW beyond
  bootstrap uncertainty on at least two real-background families, then repeats
  on independently labeled or manually verified transitions.
- **No-go:** a simpler invariant baseline matches RTW, success exists only on
  synthetic curves, or nuisance rejection destroys sensitivity to cycle-shape
  change. In that case, retain RTW as a negative row in the diagnostic study.

### 5.2 Conditional Shallow-Learning Test

Only after a geometric block passes its mechanism gate, compare nested models
under geography/year holdout and label budgets of `10%, 25%, 50%, 100%`:

```text
scalar/harmonic features only
geometry features only
scalar/harmonic + geometry
scalar/harmonic + shuffled-geometry negative control
```

Use regularized logistic regression first. Geometry features may include the
canonical-correlation spectrum, RTW dissimilarity, attention/contribution
concentration, rank/energy, and uncertainty. The claim requires a positive
held-out `Delta AP` or equal performance at a lower label budget; classifier
accuracy alone does not prove a geometric contribution.

### 5.3 Parallel Hyperspectral Attribution Gate

The strongest Codex-mining alternative remains local, moment-factorized HSI
change characterization: separate mean, total scale, normalized eigenspectrum,
and eigenspace orientation, then attribute orientation discrepancy to contiguous
wavelength intervals. Run its algebra and moment-matched tests in parallel only
if a real bitemporal HSI benchmark can be obtained. Full covariance/SPD,
MMD/energy, fused per-band effects, and unmixing are mandatory falsifiers.

### 5.4 Closed Or Deprioritized Routes

- RTW phase/tempo-invariance gate completed 2026-06-21:
  - selected on three development patches: `R=4`, `L=64`, rank `5`, raw;
  - held-out structural AP `0.8078`, versus snapshot-subspace AP `0.8156`;
  - paired AP delta `-0.0078`, 95% interval `[-0.0807,+0.0435]`;
  - marginal-matched-shape AP only `0.4533`; relative-band-phase AP `0.9511`
    was matched by snapshot PCA (`0.9550`) and DTW (`0.9459`);
  - median RTW sampling-std/score ratio `0.3012`;
  - decision: no-go for a natural-transition RTW study; retain the result in
    the diagnostic line;
  - report: `docs/experiment_reports/multisenge_rtw_invariance_gate_2026-06-21.md`.
- RTW natural-label transfer follow-up completed 2026-06-21:
  - official BreizhCrops 2017 L2A data; seven adequately represented crop
    classes, 560 sequences per region, and 2,720 pairs per run;
  - two geographic rotations: FRH01->FRH04 and FRH02->FRH03;
  - 22 controls include aligned Euclidean/RMS, global temporal shift,
    correlation, DTW/TWDTW/soft-DTW, PCA cross-reconstruction, centered and
    uncentered snapshot subspaces, deterministic shift-orbit and M-SSA
    subspaces, bandwise differences, and seasonal summaries;
  - nested RTW selection on development geography improved AP to `0.7052` and
    `0.6596`, but development-selected global-shift RMS reached `0.7789` and
    `0.7759`; paired RTW deltas were `-0.0737` and `-0.1163`, with wholly
    negative 95% intervals;
  - PCA cross-reconstruction was strongest on both holdouts (`0.8128`,
    `0.8264` AP); selected RTW had higher timing-nuisance false positives;
  - RTW also lost on natural-only, within-phenology-group hard pairs, and
    timing-invariance tasks. Close RTW as the current positive satellite route;
    do not generalize this negative result beyond the tested construction;
  - report: `docs/experiment_reports/breizhcrops_rtw_natural_label_transfer_2026-06-21.md`.
- The requested first/second DS magnitude and geodesic decomposition were
  implemented and independently tested; the rolling RGB SpaceNet 7 detector
  lost to radiometric controls and did not improve their fusion.
- Direct global/patch/Band-Image DS on OSCD is retained as diagnostic evidence,
  not the immediate optimization target.
- Generic geometry-plus-shallow-learning is deferred until geometry earns a
  distinct feature signal.

Historical OSCD experiment track:

```text
global pixel DS [done] -> patch/local DS [done] -> Band-Image DS [done] ->
fixed-grid pyramid [stopped] -> corrected Celik/IR-MAD [done]
```

Minimum fair comparisons:

- raw spectral L2 / CVA;
- PCA-diff;
- Celik PCA-kmeans if the implementation is verified;
- IR-MAD after formula audit;
- global canonical DS;
- patch3 and patch5 DS;
- local-window DS;
- Band-Image DS spatial-subspace candidate;
- multiscale subspace pyramid.

Band-Image DS spatial-subspace pilot:

- Motivation: Senpai suggested testing the opposite sample definition from current global pixel DS.
- Current global pixel DS uses `X in R^(13 x N_pixels)`, where one sample is one pixel's 13-band value vector.
- Band-Image candidate uses `X in R^(N_pixels x 13)`, where one sample is one full band image flattened into a spatial vector.
- First implementation should be one-city only and should answer shape, rank, score-map definition, runtime, and correlation with raw L2/PCA-diff.
- Required caution: with Sentinel-2 there are only 13 band samples, so rank is limited; do not call this better or more faithful until it produces interpretable maps and metrics.
- Hyperspectral extension: the idea may be more natural for hyperspectral images with hundreds of bands, where the number of band-image samples is much larger.

Status 2026-06-18:

- Implemented and swept on all 24 local OSCD cities at ranks 6 and 8.
- Strongest DS-family method in 44/48 city/rank runs.
- Still below PCA-diff on mean AP, so the next task is diagnostic score/threshold/failure-mode work, not Phase 2 training.

Minimum reporting:

- AUROC;
- PR-AUC/AP;
- best F1/IoU over thresholds;
- Otsu F1/IoU;
- raw-L2 correlation;
- runtime;
- qualitative pre/post/GT/score maps;
- city-wise failure table.

Band-Image DS score-ablation result, 2026-06-18:

- Tracked report: `docs/experiment_reports/oscd_band_image_ds_score_ablation_2026-06-18.md`.
- Core5 output: `phase1/outputs/spatial_ds_band_image_score_ablation_core5_20260618_021813/`.
- All-city output: `phase1/outputs/spatial_ds_band_image_score_ablation_allcities_20260618_022314/`.
- Attribution outputs: `phase1/outputs/spatial_ds_band_image_attribution_{city}_20260618_022904/` for `chongqing`, `nantes`, `bordeaux`, and `norcia`.

All-city mean result:

| method | mean AUROC | mean AP | mean Otsu F1 | mean best F1 | mean raw-L2 corr |
|---|---:|---:|---:|---:|---:|
| `pca_diff` | 0.8392 | 0.2541 | 0.2160 | 0.3076 | 0.8588 |
| `band_image_ds` | 0.8412 | 0.2340 | 0.1129 | 0.2928 | 0.6596 |
| `band_image_norm` | 0.8412 | 0.2340 | 0.2007 | 0.2928 | 0.7775 |
| `raw_l2` | 0.7717 | 0.2261 | 0.1802 | 0.2873 | 1.0000 |
| `band_image_ratio` | 0.7037 | 0.0586 | 0.0773 | 0.1170 | -0.0510 |
| `band_image_residual` | 0.5766 | 0.0712 | 0.0555 | 0.1203 | 0.5147 |

Interpretation:

- `band_image_norm` is the better active Band-Image DS thresholding score because it keeps the same AUROC/AP as `band_image_ds` but raises all-city Otsu F1 from `0.1129` to `0.2007`.
- The score reduction was part of the thresholding problem, but PCA-diff still wins mean AP and still has slightly better Otsu F1.
- `band_image_ratio` and `band_image_residual` should be treated as diagnostics, not primary scores.

Pressure-baseline result, 2026-06-18:

1. Repaired IR-MAD now uses paired CCA transforms and unchanged-pixel chi-square survival weighting.
2. Corrected Celik defaults to scalar CVA/L2 difference-image patches with seeded, chunked PCA/k-means fitting.
3. Rank-8 mean AP: PCA-diff `0.2541`, Band-Image DS `0.2340`, raw L2 `0.2261`, IR-MAD `0.2138`, Celik `0.1621`.
4. Band-Image DS is significantly worse than PCA-diff and significantly better than this Celik adaptation; it is not reliably different from raw L2 or IR-MAD.
5. Rank-12 Band-Image DS improves to AP `0.2410`, but still does not beat PCA-diff.
6. Three-way equal-weight rank fusion reaches AUROC `0.8708` versus PCA-diff `0.8406`, winning 21/24 cities (`p=0.00024`). Its AP gain is not significant and Otsu F1 drops to `0.1084`.
7. Fixed-grid spectral pyramid AP (`0.0762`-`0.0765`) did not improve global pixel DS (`0.0791`); stop this exact branch.
8. Inspect qualitative failure modes for:
   - Band-Image DS wins: Bordeaux, Chongqing, Milano, Paris, selected Saclay cases;
   - PCA-diff wins: Beirut, Dubai, Las Vegas, Montpellier, Mumbai, Nantes, Rio;
   - patch/local-window wins: Norcia and Saclay-e style cases.

Tracked report and outputs:

- `docs/experiment_reports/oscd_spatial_ds_baseline_pressure_2026-06-18.md`
- `phase1/outputs/spatial_ds_traditional_pressure_allcities_corrected_20260618_174228/`
- `phase1/outputs/band_image_ds_rank10_12_allcities_20260618_175418/`
- `phase1/outputs/spatial_score_rank_fusion_allcities_20260618_175956/`
- `phase1/outputs/spatial_pyramid_core5_decision_20260618_180652/`

Next decision gate:

1. Split-safe changed-area calibration: **completed 2026-06-18**.
   - Fitted one top-ranked changed-area fraction per method on 14 official training cities and froze it for 10 official test cities.
   - Three-way fusion test F1 `0.2670` versus PCA-diff `0.2452`; delta `+0.0218`, 7/10 wins, `p=0.1602`, bootstrap CI crosses zero.
   - Band-Image DS test F1 `0.2507`; its `+0.0055` delta over PCA-diff is not reliable.
   - Conclusion: rank fusion remains promising complementarity, not a proven improvement. Threshold scale alone does not solve the problem.
2. Define a pseudo-change taxonomy using representative city maps: vegetation/seasonality, water, cloud/haze, illumination, registration, and target structural change.
3. Test robust or nuisance-aware feature normalization only when it has a source and a held-out-city protocol.
4. Continue to neural/prior experiments only if the continuous geometric maps add held-out evidence beyond raw bands/PCA-diff or provide a clear diagnostic contribution.

Calibration implementation and outputs:

- `phase1/scripts/evaluate_oscd_split_calibration.py`
- `phase1/outputs/spatial_score_calibration_source_allcities_20260618_182915/`
- `phase1/outputs/oscd_split_safe_calibration_20260618_183238/`

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

4. IR-MAD fair-comparison verification: **completed 2026-06-18**.
   - The corrected implementation uses paired transforms, unchanged-pixel reweighting, and synthetic formula guards.
   - It was included in the same 24-city table as PCA-diff, raw L2, Celik, and DS variants.
   - Outcome: mean AUROC `0.8471`, AP `0.2138`, and Otsu F1 `0.0547`; retain it as a ranking baseline and calibration/failure-mode pressure test.

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
     - Related-work pressure:
       - Compare the framing against prior-guided fusion papers such as MapFormer and CGNet; do not claim novelty from "adding a prior map" alone.
       - Compare pseudo-label framing against semi-supervised CD work such as AdaSemiCD / mean-teacher methods; the DS-specific question is whether geometric pseudo-labels are useful or interpretable enough after quality filtering.
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

18. IR-MAD fair-comparison verification: **completed 2026-06-18**.
   - Sources:
     - Nielsen 2007 regularized IR-MAD paper;
     - Nielsen/Conradsen MAD/CCA formulation;
     - Google Earth Engine MAD/iMAD tutorial and Mort Canty reference material if used.
   - Why it matters:
     - IR-MAD is a mature multivariate remote-sensing change detector.
     - It is based on CCA/MAD variates and is therefore close enough to DS to be a serious comparison pressure baseline.
     - Sensei and seminar feedback already pushed toward CCA, so ignoring IR-MAD would weaken the thesis.
   - Verification performed:
     - corrected paired CCA/generalized eigenproblems and iterative unchanged-pixel weighting;
     - checked regularization, seed, convergence, chi-square statistic, equal-image behavior, and a synthetic changed block;
     - compared on all 24 cities with raw L2/CVA, PCA-diff, Celik, global DS, patch3, patch5, Band-Image DS, and local-window DS.
   - Result:
     - IR-MAD has the strongest single-method mean AUROC (`0.8471`) but lower AP (`0.2138`) than PCA-diff and Band-Image DS;
     - per-image Otsu calibration is poor (`0.0547` mean F1);
     - qualitative maps show useful ranking alongside broad seasonal/agricultural responses.
   - Decision:
     - keep repaired IR-MAD in every serious classical comparison;
     - do not claim it suppresses pseudo-change until the next nuisance/failure-mode study tests that directly;
     - use the three-way rank-fusion result only as evidence of complementary rankings.

19. Multi-date / period-subspace DS feasibility audit:
   - Check datasets with enough aligned dates per location.
   - Compare earliest/latest, adjacent, same-season, and period-window pairings.
   - Test first-order DS before second-order DS, GDS, or KGDS.
   - Do not use unlabeled MultiSenGE visuals as performance evidence without an evaluation proxy.

20. Band-group attribution audit:
   - Compute DS basis energy by Sentinel-2 band or band group.
   - Compare VIS, red-edge, NIR, SWIR, and atmospheric bands.
   - Use this to explain whether maps are likely surface change, vegetation/soil moisture, or atmospheric artifact.
   - Feature-isolation extension:
     - Treat subspace bases/projections as candidate feature isolators before downstream change detection.
     - Compare raw bands, PCA coefficients, DS projection coefficients, band-group DS coefficients, residual/background components, and optional encoder-feature subspaces as inputs to a simple downstream detector or classifier.
     - Evaluation gate: the isolated features must either improve change metrics, reduce a known pseudo-change mode, or give clearer band/region explanations. Otherwise keep them as diagnostics, not a main method.

21. SSC change-type clustering pilot:
   - Only after the spatial DS audit.
   - Input candidates: raw delta, DS projection coefficients, PCA-diff features, patch/deep features.
   - Output candidates: unsupervised change-type clusters, pseudo-labels, auxiliary channels, or a strong unsupervised baseline.
   - Must define cluster count selection and validation before implementation.

22. Hybrid geometry-learning experiment family:
   - Core framing:
     - Do not require DS/GDS to beat a neural detector at the full changed/not-changed localization task.
     - Use neural networks where they are strong: spatial localization, feature extraction, or object proposals.
     - Use subspace/geometric methods where they are defensible: interpretable evidence, region descriptors, temporal trajectories, clustering, label efficiency, and error diagnostics.
   - Contribution lanes and evidence gates:
     - Interpretability:
       - Meaning: DS/GDS should say something about the structure of the change, such as which spectral bands, local spatial patterns, dates, regions, or subspace directions define it.
       - Experiment path: `H5_neural_frontend_gds_clustering`, `H7_error_subspace_diagnostics`, `H9_object_or_region_subspace_descriptors`, `H12_band_group_product_hybrid`.
       - Evidence gate: produce region descriptors, band/date attribution, component visual grids, and cluster/error explanations. If no semantic labels exist, call the output exploratory interpretation, not semantic classification.
     - Label efficiency:
       - Meaning: geometric methods produce priors, pseudo-labels, candidate regions, auxiliary targets, or active-learning scores without dense labels.
       - Experiment path: `H2_prior_pseudo_label_pretrain`, `H3_auxiliary_prior_head`, `H8_active_learning_from_geometry`, plus reduced-label `H1_prior_channel_fusion`.
       - Evidence gate: compare 10%, 25%, 50%, and 100% label budgets against raw-only training. The claim requires equal or better performance/stability at lower label budgets.
     - Temporal/multi-date analysis:
       - Meaning: GDS/geodesic methods track evolution across three or more date subspaces, not just binary pre/post segmentation.
       - Experiment path: `H10_multidate_neural_mask_then_gds`, temporal subspace pilots in item 26, and only after a MultiSenGE/Harmonized Sentinel-2 data audit.
       - Evidence gate: show coherent temporal trajectories, date-window clusters, seasonality checks, or external/manual validation. Do not claim semantic temporal classes from unlabeled data.
     - Hybrid role:
       - Meaning: NN handles strong localization or feature extraction; subspace/GDS handles interpretation, clustering, diagnostics, temporal geometry, or label-efficient supervision.
       - Experiment path: start with `H5` or `H6`; use `H11` only if a foundation/semantic proposal source is reproducible.
       - Evidence gate: geometry must add a measurable or inspectable value beyond the NN mask alone: better low-label performance, clearer error modes, useful region grouping, or temporal explanation.
     - Negative/diagnostic study:
       - Meaning: failure can still be useful if it identifies exactly why subspace methods fail or when they complement baselines.
       - Experiment path: compare global/patch/local DS against raw L2, PCA-diff, IR-MAD, and U-Net; inspect city-specific false positives and false negatives.
       - Evidence gate: report a failure taxonomy tied to pseudo-change, spatial-information loss, rank/sample construction, registration, vegetation/water/shadow, or OSCD label policy.
   - Reliable first pilots:
     - `H1_prior_channel_fusion`: concatenate DS, patch-DS, PCA-diff, CVA, or IR-MAD score maps as extra channels to U-Net/Siamese input. Test whether priors improve IoU/F1/AUROC over raw bands and whether gains survive city-wise splits.
     - `H2_prior_pseudo_label_pretrain`: threshold or soft-normalize DS/PCA/IR-MAD maps, pretrain a segmentation model to imitate them, then fine-tune on OSCD labels. Test low-label curves: 10%, 25%, 50%, 100% labels.
     - `H3_auxiliary_prior_head`: train one shared encoder with two heads: one for OSCD labels and one for prior-map prediction or prior consistency. Test whether the auxiliary geometric task improves generalization.
     - `H4_geometry_guided_attention_or_loss`: use a geometric prior map as an attention/gating signal or loss weight so the neural model focuses on likely change regions. Compare against ordinary class-balanced loss and focal loss.
     - `H5_neural_frontend_gds_clustering`: use U-Net/Siamese/SOTA CD to localize changed pixels, then build DS/GDS/SSC descriptors from changed connected components and cluster them into change-type groups. This is the clearest "NN localizes, geometry interprets" route.
     - `H6_deep_feature_ds`: extract pre/post encoder features from a trained U-Net, Siamese CNN, or foundation model; build subspaces from patch/tile/date feature vectors; compare deep-feature DS/GDS against raw spectral DS and patch DS.
     - `H7_error_subspace_diagnostics`: collect false positives and false negatives from the neural model, build subspaces over their raw/patch/deep features, and inspect whether errors form separable geometric modes such as vegetation, water, shadow, registration, or urban texture.
     - `H8_active_learning_from_geometry`: rank unlabeled patches by disagreement between neural probability and DS/IR-MAD/geometric scores. Manually inspect or label high-disagreement samples first. Test whether this selects more informative samples than random.
     - `H9_object_or_region_subspace_descriptors`: after neural or classical change localization, build one subspace per connected component, superpixel, building, field, or greenhouse object. Use the subspace descriptor for clustering, retrieval, or later classification.
     - `H10_multidate_neural_mask_then_gds`: on MultiSenGE or Harmonized Sentinel-2, use a neural/classical detector to propose candidate regions over time, then build one subspace per date or date-window and apply GDS/geodesic quantities to summarize temporal change progression.
     - `H11_foundation_or_semantic_proposals_then_subspace`: use a foundation/semantic/open-vocabulary model to propose objects or semantic regions, then run DS/GDS/KDS on the original Sentinel-2 band data inside those regions. This separates "what object/area is this?" from "how did its spectral/spatial state change?"
     - `H12_band_group_product_hybrid`: build separate geometric priors for VIS, red-edge, NIR, SWIR, and full-band groups, then give either all maps or a learned fusion of them to the neural model. Evaluate whether band-group priors improve interpretability or performance.
     - `H13_cascade_triage`: run a cheap classical/geometric detector first to reject obvious unchanged regions, then run the neural model only on uncertain/high-change areas. Evaluate runtime, recall loss, and whether this is useful for large-area screening.
     - `H14_model_residual_geometry`: after neural prediction, compute residual/error maps against labels or high-confidence pseudo-labels, then recompute local DS/KDS only in residual-heavy regions to diagnose which spatial/spectral patterns the model misses.
   - Evaluation requirements:
     - Localization: IoU, F1, precision/recall, AUROC, AP/PR-AUC, per-city breakdown, and threshold sensitivity.
     - Label efficiency: performance versus percentage of labeled OSCD patches or cities.
     - Clustering/interpretation: if labels exist, use ARI/NMI/purity or class-wise agreement; if labels do not exist, call outputs exploratory clusters and use visual grids, temporal consistency, and manual inspection.
     - Complementarity: measure whether geometry helps where NN fails, not only whether it increases one average score.
     - Ablations: compare raw bands only, neural only, geometry only, early fusion, auxiliary loss, and post-hoc geometry.
   - Most promising order:
     - First: `H5_neural_frontend_gds_clustering`, because it gives subspace methods a clear interpretation role instead of forcing them to localize all change alone.
     - Second: `H6_deep_feature_ds`, because it matches lab-style geometric operations on learned representations.
     - Third: `H2_prior_pseudo_label_pretrain` or `H3_auxiliary_prior_head`, because these test label-efficient complementarity.
     - Fourth: `H10_multidate_neural_mask_then_gds`, if a multi-date dataset and evaluation proxy are ready.
   - Risks:
     - Hybrid methods can become unfocused if the neural part does all the work and geometry only decorates the result.
     - Do not call region clusters semantic classes unless validated with semantic/damage/object labels or credible manual interpretation.
     - Keep the contribution explicit: localization improvement, label-efficiency improvement, interpretable region clustering, temporal trajectory description, or error diagnosis.

23. Greenhouse application feasibility audit:
   - Treat abandoned greenhouse mapping as a possible application, not current evidence.
   - Define the task first: object mapping, abandonment classification, change detection, or temporal condition scoring.
   - Check whether labels, dates, and evaluation metrics exist before connecting it to DS/KDS/GDS.

24. Deep-feature subspace pilot:
   - Inspired by Mahyub et al. 2024 Signal Latent Subspace, not currently implemented.
   - Extract latent features from a remote-sensing CNN, U-Net encoder, or foundation model.
   - Build subspaces from patch/tile/date latent features instead of raw 13-band pixel vectors.
   - Compare latent-feature DS/GDS against raw spectral DS, local/window DS, PCA-diff, and the neural baseline.
   - Consider product-Grassmann fusion only if there are clearly defined feature factors, such as spectral, spatial, temporal, and prior-map factors.
   - This is the more explicit hybrid route: geometrical representation over learned features instead of only raw bands.

25. Multiscale subspace pyramid pilot:
   - Run only after global/window/patch DS gives a baseline.
   - Source status: Senpai idea inspired by wavelets/JPEG/Green Learning; exact formal source still needs verification.
   - Initial levels: `1x1`, `2x2`, `4x4`; add `8x8` only if runtime is manageable.
   - Initial scale weights: equal average, then compare coarse-heavy and fine-heavy weights.
   - Output: per-level maps, weighted map, runtime, block-artifact inspection, and metrics against OSCD.
   - Baselines: global canonical DS, local-window DS, raw L2/CVA, PCA-diff.

26. Temporal subspace literature pilots:
   - RTW/Deep RTW pilot: for MultiSenGE or Harmonized Sentinel-2 L2A sequences, randomly sample ordered date subsequences, build sequence-hypothesis subspaces, and compare them to same-season or event-window references.
   - SFA/SFS pilot: learn slowly varying temporal components from aligned date sequences, then test whether residuals or slow-feature subspaces separate seasonal drift from abrupt land-cover change.
   - Product-Grassmann/Hankel pilot: represent one patch as multiple subspace factors, such as spectral, spatial, and temporal/Hankel factors, then use geodesic distances for clustering or anomaly ranking.
   - G-LMSM/SLS pilot: use remote-sensing encoder features to build patch/date latent subspaces, then compare hand-built DS maps against learned Grassmann subspace matching.
   - Shape-subspace attribution pilot: adapt the human-motion DS idea by reporting which bands, patch regions, or date windows contribute most to the difference subspace.
   - Do not start these before the spatial OSCD audit; these are method-expansion tracks, not current evidence.

27. xBD-S12 metric protocol audit:
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

## 10. Active Temporal Difference-Subspace Study

Active question as of 2026-06-19:

```text
Can paper-faithful first/second Difference Subspaces and geodesic decomposition
provide distinct, spatially attributable evidence of change in registered
multispectral satellite time series, after accounting for irregular cadence,
radiometric variation, and misregistration?
```

This supersedes the instruction near the end of Section 6 that temporal pilots
must wait for the spatial OSCD audit. That audit has now been run and found that
simple spatial PCA/smoothing explains much of the apparent gain.

### 10.1 Completed Implementation Checks

- One date cube is represented by `X_t in R^(N_common_pixels x B_bands)`.
- Each column is one complete aligned band image flattened over the same common
  spatial mask.
- The leading left singular vectors form `S_t in Gr(r, N_common_pixels)`.
- Full-rank MultiSenGE uses `r=10`; full-rank IPOL RGBI uses `r=4`.
- Paper-faithful first magnitude is `2 sum_i(1-cos(theta_i))`.
- Paper-faithful second DS is `D(S_t, M(S_{t-1},S_{t+1}))`.
- Geodesic decomposition reports along and orthogonal components.
- The second-order paper assumes equal intervals. The project therefore also
  reports a separate time-aware deviation from the endpoint Grassmann geodesic
  at the observed acquisition fraction. It is not mislabeled as paper DS.
- Canonical spatial contributions sum to the corresponding DS magnitude.
- Eighteen focused unit tests cover equal-speed, along-geodesic, off-geodesic,
  unequal-gap, interpolation, mean-subspace, common-mask, shape, and attribution
  behavior.

### 10.2 Controlled Findings

Full-rank centered band-image subspaces on real MultiSenGE triples with injected
changes:

- local change versus global per-band gain/offset:
  - paper second/orthogonal AUROC `1.0`, AP `1.0` in the rank-10 pilot;
  - the full band-image span is invariant to the tested invertible per-band
    scaling, while centering removes constant offset.
- local change versus one-pixel translation:
  - orthogonal AUROC `0.0`; every tested translation produced a larger response
    than the local synthetic change.
- local change versus all tested negatives:
  - AUROC `0.875`, AP `0.540`.

Interpretation: the construction has a real radiometric invariance but is not
registration robust. This is a method boundary worth studying, not evidence of
general change-detection superiority.

Output:

`phase1/outputs/multisenge_temporal_injections_full_channel_20260619_193250/`

### 10.3 Real-Sequence Findings

MultiSenGE, five patches x 23 dates:

- only `15/105` triples have left/right gap ratio <= `1.5`;
- paper second and orthogonal magnitudes correlate at `0.973`;
- paper second correlates with raw second reflectance RMSE at `0.709`, NDVI
  curvature at `0.818`, and NBR curvature at `0.752`;
- the time-aware deviation correlates with paper second at `0.941` and raw
  curvature at `0.675`;
- on the 15 balanced triples, time-aware deviation and paper second correlate
  at `0.988`.

These curves currently track spectral/index curvature strongly. MultiSenGE has
static land-cover labels, so this is descriptive evidence, not supervised event
detection accuracy.

Output:

`phase1/outputs/multisenge_temporal_timeaware_core5_20260619_194845/`

IPOL Las Vegas, 20 registered RGBI dates:

- the published NFA method identifies important changes at frames 2 and 4;
- first DS ranks frame 2 highly but frame 4 only tenth among 19 pairs;
- paper second/time-aware maps at both frames respond broadly to roofs, roads,
  and scene texture rather than isolating only the published changed objects;
- the strongest paper second/time-aware response occurs at 2018-02-15, not one
  of the first-five published demonstration events.

This is important negative pressure: whole-scene temporal DS is not yet a
competitive local detector.

Output:

`phase1/outputs/ipol_vegas_temporal_timeaware_20260619_194845/`

### 10.4 Four-Sequence External Pressure Test

The exact IPOL C implementation was compiled locally and run on pseudo-gamma
versions of Las Vegas, Al Wakrah, Piraeus, and Beijing Airport. Its log-NFA
maps are external detector outputs, not ground truth.

One-date rank-2 band-image DS did not generalize. Its adjacent/along score was
near raw difference on Vegas but worse on Al Wakrah, Piraeus, and Beijing.
Multiscale tiling did not rescue that result.

For second-order rank-2 whole-scene maps, macro results over the four sequences
were:

| Method | Macro pixel AP | Macro pixel AUROC | Macro temporal Spearman with IPOL changed fraction |
|---|---:|---:|---:|
| raw time-interpolation residual | 0.1206 | 0.8282 | 0.5485 |
| paper second DS | 0.0895 | 0.8518 | 0.5934 |
| orthogonal component | 0.0951 | 0.8409 | 0.5826 |
| along component | 0.0512 | 0.7774 | 0.3065 |
| irregular-time geodesic deviation | 0.0994 | 0.8404 | 0.5898 |

Interpretation: second/time-aware geometry is slightly better aligned with
sequence-level event intensity in aggregate, but worse at pixel localization.
This motivates a descriptor/interpretation hypothesis, not a detector win.

Bidirectional temporal-context sweep (`V={3,5}`, rank `{1,2}`, per-band and
joint) produced:

- raw adjacent RMS macro AP `0.38`, AUROC `0.97`;
- best linear projection novelty (`V=3`, rank 2, per-band) macro AP `0.37`,
  AUROC `0.98`;
- best temporal-context DS macro AP about `0.10`.

The DS localization hypothesis is rejected under this construction. Projection
novelty remains a candidate because it improves AUROC in all four sequences,
but it beats raw AP only on Piraeus and still needs labeled evaluation.

Outputs:

- `phase1/outputs/temporal_context_ds_macro_4seq_20260620/`
- `phase1/outputs/temporal_context_ds_*_matrix_20260620/`

### 10.5 Persistent-Change And Nuisance Diagnostic

Five real MultiSenGE backgrounds were used with known injected masks. Four
predefined configurations varied context `V={3,5}` and rank `{1,2}`.

Rank-2, `V=3`, per-band results:

| Method | Persistent localization AP | Persistent/transient response | Gain/offset response | Translation response |
|---|---:|---:|---:|---:|
| temporal-context DS | 0.733 | 0.97x by raw sums; positive paired normalized contrast | near zero | 16.6x persistent response |
| linear projection novelty | 0.909 | 1.51x | near zero | 11.8x persistent response |
| raw adjacent RMS | 0.936 | exactly 1.00x | very large | 7.1x persistent response |

Clustered bootstrap over five patch backgrounds gives projection novelty a
persistent-vs-transient normalized contrast of `0.190` with 95% interval
`[0.170, 0.212]`. Its persistent localization AP difference from raw is
`-0.024`, interval `[-0.068, 0.018]`. The DS contrast grows at lower rank/longer
context, but localization degrades sharply. Five clusters are too few for a
publication-level uncertainty claim; this is a stable diagnostic lead.

Decision:

- retain persistence discrimination and radiometric invariance as hypotheses;
- reject current DS as the main localization score;
- treat translation robustness and labeled multi-temporal evaluation as hard
  gates before method claims.

Outputs:

- `phase1/outputs/temporal_context_injections_multisenge_*_20260620/`
- `phase1/outputs/temporal_context_injections_macro_20260620/`

### 10.6 Registration And Scale-Space Curve

Controlled horizontal shifts `{0.25,0.5,1,2}` pixels were compared with a
persistent local injection on five MultiSenGE backgrounds.

For a `32x32`, strength-0.25 event using projection novelty:

| Representation | Translation/local ratio at 1 px | Local AP |
|---|---:|---:|
| native | 26.67 | 0.982 |
| Gaussian `sigma=1` | 8.19 | 0.993 |
| Gaussian `sigma=2` | 3.17 | 0.993 |
| Gaussian `sigma=3` | 1.72 | 0.988 |
| Gaussian `sigma=4` | 1.14 | 0.984 |
| phase alignment | 0.006 | 0.877 |

The apparent robustness does not generalize to a weak `16x16` event: at one
pixel and `sigma=4`, translation/local ratio remains `32.8` and local AP falls
to `0.764`. Phase alignment corrects integer shifts well but loses event AP and
is imperfect at `0.25-0.5` pixels.

Cross-scale retention (`score_sigma / score_native`) perfectly separates the
tested global translations from persistent local injections for DS, projection
novelty, and raw difference. Projection has the largest mean log-retention
margin at `sigma=4`, but AUROC/AP are `1.0` for all three methods. Therefore
scale-space decay is a useful generic artifact diagnostic, not DS-specific
novelty.

Next robustness test must include local warps/parallax and natural labeled
events. Do not claim Gaussian smoothing or phase alignment solves registration.

Outputs:

- `phase1/outputs/temporal_context_registration_curve_*_20260620/`
- `phase1/outputs/temporal_context_scale_decay_20260620/`

### 10.7 Immediate Experiment Order

1. Test seasonal observation subspaces on a labeled regime-change protocol:
   - one aligned patch/field and one year supply a set of cloud-filtered date
     composites; each full spatial-spectral date cube is one column of `X_y`;
   - build one PCA subspace per year, then calculate first DS, second DS,
     along/orthogonal decomposition, and time-aware geodesic deviation;
   - first controlled task: irrigation start/stop, because adding/removing
     irrigation changes seasonal dynamics rather than only one acquisition;
   - use IrrMapper transitions only as weak candidate labels, manually verify a
     selected subset, and require an independently labeled repeat before a
     performance claim;
   - if Earth Engine access remains blocked, run the complete synthetic regime
     validation and preserve real-data evaluation as blocked rather than
     fabricating results.
2. Obtain an independently labeled multi-temporal evaluation slice:
   - preferred: a selectively acquired DynamicEarthNet AOI with monthly semantic labels;
   - dataset size is approximately 524-564 GB in full, so define a selective
     acquisition protocol rather than downloading blindly;
   - fallback: an independently annotated Harmonized Sentinel-2 event sequence.
3. Build a registration robustness curve:
   - translations from subpixel/interpolated shifts through at least two pixels;
   - compare pre-registration, local pooling, low-frequency/wavelet features,
     and displacement-minimized scores;
   - keep gain/offset and local persistent/transient interventions.
4. Preserve the task split in evaluation:
   - event timing/characterization: first, second, orthogonal, along, and
     time-aware geodesic descriptors;
   - changed-area localization: projection novelty, IPOL NFA, raw/index, and
     classical/deep segmentation controls.
5. Extend multiscale local temporal band-image subspaces only with a matched
   non-DS control:
   - scales: whole scene, `2x2`, `4x4`, and overlapping local tiles;
   - compute first, second, orthogonal, along, and time-aware quantities per
     tile;
   - compare spatial fusion with matched local PCA/chronochrome controls.
6. Add temporal baselines beyond the reproduced IPOL NFA:
   - MOSUM and one trend/seasonal method such as BFAST/JUST;
   - never call another detector output ground truth.
7. Acquire one Sensei-requested Harmonized Sentinel-2 sequence:
   - report region, dates, frame count, actual gaps, clouds/no-data, bands,
     spatial alignment, and event/evaluation source;
   - do not run a large sequence before this feasibility card exists.
8. Decide after the above whether the paper is:
   - a new time-aware/local temporal DS method;
   - a nuisance-robust DS method;
   - or a diagnostic benchmark explaining when temporal DS helps/fails.

### 10.8 Evidence Gates

Continue toward a method claim only if:

- known/published events are localized better than whole-scene DS;
- improvement is not reproduced by smoothing/tiling a non-DS baseline;
- registration robustness improves without destroying local-change response;
- at least one independent sequence repeats the finding;
- the result survives rank, scale, preprocessing, and cadence ablations.

Otherwise publish or present the negative result honestly: full-channel temporal
DS is radiometrically invariant but registration sensitive, and whole-scene
geometric contributions are too diffuse for changed-area localization.

Seasonal-regime study design:

| Component | Required test |
|---|---|
| Positive event | irrigation off-to-on and on-to-off transitions |
| Negative event | stable irrigated and stable non-irrigated years |
| Confounders | phenological phase shift, gradual trend, gain/offset, missing composites, clouds/noise, spatial translation/local warp |
| Geometric scores | adjacent first magnitude/geodesic, second total, along, orthogonal, time-aware deviation |
| Controls | NDVI amplitude/mean break, raw mean-spectrum L2, minimum principal angle, null/no-event score, then MOSUM/BFAST/JUST |
| Event metric | boundary AUROC/AP, event-year rank, timing error, false alarms per stable sequence |
| Characterization metric | abrupt-versus-gradual classification and effect-size separation |
| Calibration | train-free threshold analysis plus held-out-site calibration if a threshold is reported |
| Generalization | hold out geography and year; repeat on an independent label source |

Stop the irrigation route if DS scores do not exceed simple NDVI/raw/min-angle
controls on event timing, if success depends on one geography/rank, or if manual
inspection shows that IrrMapper transition noise explains the apparent result.

### 10.9 Seasonal Observation Stress-Test Result

Completed 2026-06-20:

- `800` synthetic five-year sequences (`80` per scenario), `3200` candidate
  boundaries per rank/preprocessing configuration, `200` sequence-level
  bootstrap resamples, plus noise `0.02` and `0.04` pressure runs.
- Best first DS: uncentered rank 1, AUROC `0.950`, AP `0.403`
  (`0.356-0.469`), versus raw monthly RMS AP `0.459` and NDVI-curve AP `1.000`.
- Rank-1 DS ranked the true abrupt boundary first within every event sequence,
  but did not calibrate well across sequences.
- Feature-centered rank-1 DS rejected tested gain/offset and phase nuisances,
  but false-alarm rates were `0.997` for missing composites, `0.559` for
  one-pixel translation, and `0.528` for gradual drift under a stable-cycle
  95th-percentile threshold.
- At noise `0.04`, rank-2 second-along AP was `0.698` (`0.649-0.778`) versus
  NDVI second-difference AP `0.674` (`0.622-0.730`); intervals overlap.
- Conclusion: continue only as a real-data test of relative event timing,
  geometric trajectory characterization, and orientation-plus-energy
  decomposition. Do not claim a synthetic DS performance win.

Report:
`docs/experiment_reports/seasonal_observation_subspace_stress_test_2026-06-20.md`.

### 10.8 Independently Labeled SpaceNet 7 Gate - Completed 2026-06-21

The independent-label gate was executed on SpaceNet 7 monthly RGB imagery.
One AOI was used for discovery, three for development, and four additional
AOIs were selected by growth quantile before confirmation scoring.

Frozen candidate:

```text
fixed 8x8 cells
-> six-month rolling windows
-> lag-two RGB trajectory matrix
-> rank-two subspace
-> second-DS orthogonal magnitude
```

Confirmation macro AP across four AOIs:

| Method | Mean AP |
|---|---:|
| Second-DS orthogonal magnitude | 0.1127 |
| Standardized raw pair RMS | 0.1615 |
| Standardized raw second RMS | 0.1502 |
| Two-radiometric rank fusion | 0.1910 |
| Geometry plus radiometric rank fusion | 0.1747 |

The three-score fusion was below the two-radiometric fusion by `-0.0163` macro
AP, with hierarchical 95% interval `[-0.0531, +0.0140]`. Geometry alone lost
substantially, and adding it did not improve the fair two-control fusion.

Component analysis: first DS was the strongest geometric quantity at AP
`0.1313`; second total/along were `0.1229/0.1226`; second orthogonal was
`0.1127`. First magnitude and Grassmann distance were effectively redundant
(`rho=0.9999`), while second total was dominated by along ordering
(`rho=0.9596`).

Data-integrity correction: SpaceNet UDM polygons are CRS84 and must be
reprojected to the image CRS. Valid support is intersected per transition over
only the rolling windows being compared, not over the entire sequence. One
preselected AOI had no valid six-month support and was replaced before scoring
using a mask-only deterministic rule.

Decision:

- close this exact rolling RGB trajectory-DS detector;
- do not tune its grid, window, rank, or fusion on the seven non-discovery
  AOIs;
- retain the result as evidence that verified first/second DS quantities need
  not be useful local change detectors;
- require a new mechanism and fresh held-out data for the next route.

[gap] Can subspaces isolate useful multispectral/hyperspectral characteristics
or explain nuisance/change factors even when RGB trajectory DS fails as a
detector?
[why it matters] This is a materially different role for geometry and is not
answered by the SpaceNet 7 experiment.
[next check] Design a source-grounded feature-isolation or nuisance-subspace
experiment with a matched non-subspace control and untouched evaluation data.

Report:
`docs/experiment_reports/spacenet7_temporal_subspace_validation_2026-06-21.md`.

Data gate:

- Public IrrMapper v1.2 coverage and 1986-2024 temporal extent are verified.
- AOI/frame querying is blocked until an Earth Engine-enabled Google Cloud
  project is configured.
- IrrMapper transitions remain weak labels; manual/independent verification is
  required before performance testing.

### 10.10 Real-Background Order And Localization Result

Completed 2026-06-20:

1. Corrected the temporal construction distinction:
   - unordered annual PCA is invariant to date permutation;
   - first differences and block trajectory matrices are order-aware;
   - eight formula/invariance tests pass.
2. Ran global/crop controlled interventions on five MultiSenGE patches:
   - `8` repeats, `23` dates, ranks `1,2`, four representations, four
     preprocessing modes;
   - best DS-family AP `0.448`; normalized trajectory DS AP `0.365`;
   - simple amplitude/spectrum controls were stronger.
3. Ran cell-wise localization with exact injected masks:
   - randomized off-grid supports;
   - grids `2,4,8`;
   - stable, gain/offset, phase, missing-composite, translation, global-
     amplitude, and localized-mode scenarios;
   - patch-level bootstrap, not pixel-independent bootstrap.
4. Ran a smoothing robustness curve and fair multispectral controls:
   - local eigenspectrum AP increased `0.516 -> 0.650 -> 0.688` for Gaussian
     sigma `0,1,2`;
   - translation false alarms decreased `0.847 -> 0.607 -> 0.292`;
   - missing-composite false alarms decreased `0.843 -> 0.563 -> 0.358`;
   - sigma-2 AP: eigenspectrum `0.688`, NDMI `0.680`, NBR `0.637`, NDVI
     `0.486`, first DS `0.456`;
   - paired eigenspectrum-minus-NDMI delta AP `+0.008`, 95% CI
     `-0.249` to `+0.244`; there is no superiority claim;
   - paired eigenspectrum-minus-first-DS delta AP `+0.233`, 95% CI
     `+0.084` to `+0.354` in this controlled task.

Decision:

- retain first/second DS as interpretable trajectory descriptors;
- do not treat pure DS as the winning detector;
- promote local temporal eigenspectrum as a candidate companion descriptor;
- keep order-aware trajectory matrices for event timing/order questions, not
  because they won the current localization intervention;
- stop synthetic/controlled refinement until a real-label slice is prepared.

Immediate next experiment: independently labeled temporal gate

```text
one labeled AOI/transition source
-> fixed spatial supports and temporal windows
-> local eigenspectrum + first/second DS + geodesic quantities
-> NDVI/NDMI/NBR + raw multispectral controls
-> MOSUM or BFAST/JUST pressure baseline
-> held-out region/time evaluation
```

Preferred route: selectively acquire one DynamicEarthNet AOI and its monthly
labels. Do not download the full ~524 GB archive. Alternative: configure an
Earth Engine project, extract a small IrrMapper/Sentinel-2 slice, and manually
verify transition fields before scoring.

Required decision gates:

- Event labels must be independent of the tested Sentinel-2/Planet features.
- Acquisition dates and masks must be explicit.
- Thresholds or calibration must be fit outside the held-out AOI/year.
- Eigenspectrum must beat or complement the best index/break baseline on at
  least one repeatable event class, or the contribution becomes a negative
  diagnostic.
- Missing observations and residual registration must remain explicit
  nuisance strata.

Report:
`docs/experiment_reports/seasonal_observation_subspace_stress_test_2026-06-20.md`.
