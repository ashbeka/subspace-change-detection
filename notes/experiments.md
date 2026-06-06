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

## Paused Work

Pause until OSCD subspace construction is settled:

- xBD/xBD-S12 damage mapping.
- abandoned-greenhouse mapping as a main benchmark.
- foundation models or large new architectures.
- long U-Net sweeps using unverified DS priors.
- broad claims about disaster response.

