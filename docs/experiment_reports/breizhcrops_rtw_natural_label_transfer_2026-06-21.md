# BreizhCrops RTW Natural-Label Transfer Study

## 1. Why This Experiment Was Run

The first MultiSenGE RTW experiment asked whether Randomized Time Warping
(RTW) could ignore harmless seasonal timing/tempo changes while detecting
cross-band seasonal-shape changes. RTW did not exceed an ordinary snapshot
subspace, but that experiment had three important limitations:

1. only five real Sentinel-2 backgrounds were used;
2. labels came from controlled transformations rather than natural classes;
3. several direct killer controls were missing.

This second study therefore asks a stricter natural-data question:

```text
Can an RTW hypo-subspace keep same-crop Sentinel-2 trajectories similar under
natural and controlled timing variation while separating different crop
phenologies better than simpler temporal, PCA, correlation, and subspace
representations?
```

This is pairwise crop-phenology discrimination and invariance analysis. It is
not changed-area segmentation.

## 2. What The Snapshot Subspace Means

For one field, let `z_t in R^10` be the ten-band Sentinel-2 mean spectrum at
date `t`. Stack the date vectors as:

```text
Z = [z_1, ..., z_T] in R^(10 x T).
```

The **uncentered snapshot subspace** is the rank-five left singular-vector
basis of `Z`. The centered variant first subtracts the temporal mean spectrum.
Two field subspaces are compared using squared canonical correlations:

```text
d_snapshot = 1 - mean_i(cos(theta_i)^2).
```

This is an MSM-style project control: ordinary PCA subspaces plus canonical-
angle comparison. It belongs to the broad mutual-subspace lineage studied in
Fukui's laboratory, but `snapshot subspace` is a descriptive project name. It
is not Difference Subspace, not RTW, and not claimed as a new named Sensei
method. It is order-invariant: permuting dates leaves its span unchanged.

## 3. RTW Construction

For every temporal sequence:

1. sample `R` dates without replacement;
2. sort the sampled indices to retain order;
3. concatenate their ten-band vectors into `f_l in R^(10R)`;
4. repeat `L` times to form `F in R^(10R x L)`;
5. fit an uncentered PCA hypo-subspace of rank `m`;
6. compare two hypo-subspaces by the same canonical-correlation similarity.

This follows [Suryanto, Xue, and Fukui (2016)](https://doi.org/10.1016/j.imavis.2016.07.003), the later RTW description, and the bundled MATLAB `TEfeatures.m` implementation.

## 4. Natural Data And Geographic Protocol

The study uses the official [BreizhCrops](https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020) 2017 Sentinel-2 L2A benchmark.

- Ten bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
- One sample: parcel-mean multispectral time series for one labeled field.
- Cloud, edge, and saturation-flagged observations are removed.
- Minimum retained observations: 12.
- Seven crop classes have at least 40 fields in every evaluated region.
- Each run uses 560 development and 560 held-out field sequences.
- Each run scores 2,720 balanced/structured pairs.

Two independent geographic experiments were run:

| Development geography | Untouched holdout geography | Seed |
|---|---|---:|
| FRH01 | FRH04 | 2718 |
| FRH02 | FRH03 | 3141 |

No holdout label was used for method or baseline selection.

## 5. Pairwise Tasks

Each anchor field produces:

- a different field with the same crop label;
- a different crop from the same broad phenological group when available;
- a random different crop;
- a circularly phase-shifted version of itself;
- a monotonically tempo-warped version of itself.

Four tasks are reported:

1. `natural_semantic`: same-crop versus different-crop natural fields;
2. `hard_natural`: same-crop versus only within-group crop differences;
3. `timing_invariance`: timing-transformed same field versus different crop;
4. `combined`: all same/timing negatives versus all different-crop positives.

## 6. Killer Controls

The final comparison contains the requested alternatives:

| Family | Implemented control | What it tests |
|---|---|---|
| Raw distance | aligned raw/z-scored RMS | Whether direct Euclidean difference is enough. |
| Global temporal shift | best circular-shift RMS and correlation | Whether one global phase alignment explains RTW's invariance. |
| Dynamic warping | DTW, TWDTW, soft-DTW | Whether established sequence alignment is stronger. |
| Correlation | global-shift and per-band correlation | Whether temporal shape similarity is enough. |
| PCA reconstruction | symmetric rank-five cross-reconstruction RMSE | Whether each field's ordinary PCA distribution explains the other. |
| Ordinary subspace | centered/uncentered snapshot subspaces | Whether random warped samples add beyond date-spectrum PCA. |
| Shift-generated subspace | full circular shift-orbit subspace | Whether simple deterministic shift augmentation is enough. |
| Bandwise/simple statistics | mean spectrum, amplitude, spectral angle, summaries | Whether transparent seasonal statistics are enough. |
| Frequency/seasonal models | Fourier magnitude, phase-aligned harmonics | Whether explicit seasonal descriptors are enough. |
| Deterministic temporal subspace | M-SSA/Hankel | Whether a non-random ordered subspace is enough. |

The shift-orbit implementation retains its full numerical span. A formula test
showed that arbitrary rank truncation can split degenerate sine/cosine pairs
and destroy the claimed shift invariance.

## 7. RTW Selection Protocol

Two RTW evaluations were retained:

1. frozen MultiSenGE configuration: `R=4`, `L=64`, rank 5, raw reflectance;
2. nested crop-specific search on the development geography only.

The nested search screens 48 combinations:

```text
R in {2,4,8,12}
L in {32,64,128}
rank in {2,5}
preprocessing in {raw, per-sequence z-score}
```

Four finalists are re-estimated on all development pairs with three
independent RTW samplings. One configuration is frozen before holdout scoring.

The selected configurations were not geographically stable:

| Split | Selected RTW configuration |
|---|---|
| FRH01 -> FRH04 | `R=8`, `L=64`, rank 2, per-sequence z-score |
| FRH02 -> FRH03 | `R=4`, `L=128`, rank 5, per-sequence z-score |

## 8. Main Results

Combined-task held-out AP:

| Method | FRH04 AP | FRH03 AP |
|---|---:|---:|
| PCA cross-reconstruction | **0.8128** | **0.8264** |
| Global-shift RMS z-score | 0.7789 | 0.7759 |
| DTW z-score / raw | 0.7436 | 0.7911 |
| TWDTW z-score | 0.7398 | 0.7532 |
| Snapshot centered | 0.7100 | 0.7091 |
| Nested-selected RTW | 0.7052 | 0.6596 |
| Frozen RTW raw | 0.5955 | 0.6065 |

The baseline was selected on the development region, then frozen. In both
experiments it was global-shift RMS:

| Holdout | RTW minus selected baseline AP | Anchor-bootstrap 95% interval |
|---|---:|---:|
| FRH04 | -0.0737 | [-0.1084, -0.0416] |
| FRH03 | -0.1163 | [-0.1536, -0.0774] |

Nested tuning therefore improves RTW materially over the imported MultiSenGE
configuration, but does not close the gap.

## 9. Task-Specific Stress Test

Nested-selected RTW was below the development-selected simple baseline on
every task:

| Holdout | Task | AP delta | 95% interval |
|---|---|---:|---:|
| FRH04 | Natural semantic | -0.072 | [-0.105, -0.040] |
| FRH04 | Hard natural | -0.070 | [-0.125, -0.016] |
| FRH04 | Timing invariance | -0.104 | [-0.128, -0.080] |
| FRH04 | Combined | -0.074 | [-0.108, -0.042] |
| FRH03 | Natural semantic | -0.108 | [-0.142, -0.070] |
| FRH03 | Hard natural | -0.139 | [-0.179, -0.090] |
| FRH03 | Timing invariance | -0.131 | [-0.155, -0.110] |
| FRH03 | Combined | -0.116 | [-0.154, -0.077] |

At a threshold chosen on the development region for approximately 80% positive
recall, timing-nuisance false-positive rates were:

| Holdout | Selected RTW | Global-shift RMS |
|---|---:|---:|
| FRH04 | 10.9% | 1.4% |
| FRH03 | 17.0% | 0.4% |

## 10. Interpretation

RTW is not mathematically invalid and is not universally “dead.” It remains a
valid sequence representation developed for motion recognition. The project
has now established a narrower result:

```text
For parcel-level Sentinel-2 crop phenology under this RTW construction,
random ordered-subsequence subspaces do not provide incremental discrimination
or timing invariance beyond simpler global alignment and PCA controls.
```

The snapshot subspace was competitive but was not the strongest method once
the missing controls were added. PCA cross-reconstruction was consistently
best, and a direct global-shift RMS control was consistently stronger than RTW.
The data therefore do not support RTW as the project's next positive method.

An attention-weighted or learned RTW variant would be a new supervised method
question, not a harmless continuation of this experiment. It should only be
reopened if Sensei explicitly prioritizes it or if a task is identified where
random ordered subsequences supply a capability that these controls cannot.

## 11. Limitations

- Crop-class discrimination is not changed-area detection.
- Timing warps are controlled interventions, although the field trajectories
  and crop labels are real.
- Parcel means discard within-field spatial structure.
- Two independent development/holdout rotations cover four Brittany regions,
  not global agricultural conditions.
- This study does not test attention RTW, learned metric heads, or dense-image
  segmentation.

## 12. Reproduction

FRH01 to FRH04:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-rtw-breizhcrops-transfer --search-rtw --development-region frh01 --holdout-region frh04 --output-dir "phase1/outputs/breizhcrops_rtw_nested_search_frh01_frh04_$tag" --max-fields-per-class 80 --anchors-per-class 40 --rtw-replicates 3 --bootstrap 1000 --seed 2718
```

FRH02 to FRH03:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-rtw-breizhcrops-transfer --search-rtw --development-region frh02 --holdout-region frh03 --output-dir "phase1/outputs/breizhcrops_rtw_nested_search_frh02_frh03_$tag" --max-fields-per-class 80 --anchors-per-class 40 --rtw-replicates 3 --bootstrap 1000 --seed 3141
```

Primary local outputs:

- `phase1/outputs/breizhcrops_rtw_nested_search_frh01_frh04_20260621_114826`;
- `phase1/outputs/breizhcrops_rtw_nested_search_frh02_frh03_20260621_115223`.
