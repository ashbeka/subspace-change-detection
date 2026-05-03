# OSCD Core Sweep Results

Run path:

```text
phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422
```

Status: completed  
Git hash recorded by runs: `36d5c2e98857c1a520e082b411bd6c6a710503bc`  
Seeds: `1234`, `1235`, `1236`  
Epochs: `150`  
Device: CUDA

## Question

Does `E1_raw_ds` improve over `E0_raw_unet` on OSCD test metrics across three controlled seeds?

Short answer: **No. In this sweep, raw+DS underperformed raw-only across all three seeds.**

## Mean Test Metrics

| tag | mean IoU | mean F1 | mean AUROC | mean PR-AUC |
|---|---:|---:|---:|---:|
| `E0_raw_unet` | 0.2396 +/- 0.0037 | 0.3588 +/- 0.0086 | 0.8633 +/- 0.0057 | 0.4331 +/- 0.0027 |
| `E1_raw_ds` | 0.2213 +/- 0.0087 | 0.3282 +/- 0.0105 | 0.8534 +/- 0.0082 | 0.3974 +/- 0.0217 |
| `E1b_raw_ds_cross` | 0.2198 +/- 0.0104 | 0.3343 +/- 0.0144 | 0.8541 +/- 0.0112 | 0.4002 +/- 0.0061 |
| `E2_raw_pca` | 0.2218 +/- 0.0132 | 0.3316 +/- 0.0166 | 0.8559 +/- 0.0136 | 0.3960 +/- 0.0069 |
| `E3_raw_ds_pca` | 0.2460 +/- 0.0062 | 0.3663 +/- 0.0068 | 0.8525 +/- 0.0168 | 0.4088 +/- 0.0197 |
| `S0_siamese` | 0.2501 +/- 0.0337 | 0.3645 +/- 0.0384 | 0.8776 +/- 0.0281 | 0.4461 +/- 0.0128 |

## E1 Raw+DS Minus E0 Raw

| seed | delta IoU | delta F1 | delta AUROC | delta PR-AUC |
|---:|---:|---:|---:|---:|
| 1234 | -0.0216 | -0.0346 | -0.0221 | -0.0532 |
| 1235 | -0.0055 | -0.0179 | -0.0009 | -0.0147 |
| 1236 | -0.0279 | -0.0394 | -0.0069 | -0.0392 |

Interpretation:

- DS projection alone did not improve the supervised OSCD segmenter in this controlled run.
- The old single-run artifact that showed E1 beating E0 should now be treated as non-reproduced.
- The thesis should not claim "DS prior improves OSCD segmentation" without qualification.

## E3 Raw+DS+PCA Minus E0 Raw

| seed | delta IoU | delta F1 | delta AUROC | delta PR-AUC |
|---:|---:|---:|---:|---:|
| 1234 | +0.0091 | +0.0046 | +0.0017 | -0.0079 |
| 1235 | +0.0110 | +0.0135 | -0.0261 | -0.0494 |
| 1236 | -0.0011 | +0.0043 | -0.0079 | -0.0155 |

Interpretation:

- DS+PCA slightly improved threshold metrics on average.
- It did not improve AUROC or PR-AUC on average.
- This suggests priors may affect the calibrated/thresholded mask behavior more than ranking quality.
- This is a weaker but still research-useful result: priors are not universally helpful, but combinations may shift segmentation behavior.

## Siamese Baseline

`S0_siamese` had the best mean AUROC and PR-AUC and the highest mean IoU, but its seed variance was large:

```text
IoU mean 0.2501 +/- 0.0337
F1 mean 0.3645 +/- 0.0384
AUROC mean 0.8776 +/- 0.0281
PR-AUC mean 0.4461 +/- 0.0128
```

Interpretation:

- Siamese raw-only is an important baseline, not a side detail.
- If the thesis claims prior-channel gains, it must compare against Siamese raw-only.
- The variance means we should inspect per-seed/per-city results before treating Siamese as clearly superior.

## Thesis Consequences

Unsafe claim:

```text
DS priors improve supervised OSCD segmentation.
```

Safer claim:

```text
In a controlled 3-seed OSCD sweep, DS projection alone did not improve over raw pre/post Sentinel-2 input. A combined DS+PCA prior configuration slightly improved threshold-dependent IoU/F1 but not ranking metrics, suggesting that unsupervised prior channels can alter segmentation behavior but require careful ablation and should not be overclaimed.
```

## Next Analysis

1. Inspect per-city metrics for E0, E1, E3, and S0.
2. Visualize a few cities where E1 is worse than E0.
3. Test whether threshold tuning on validation changes the IoU/F1 conclusion.
4. Decide whether the contribution is "DS-specific" or broader "prior-channel ablation."
5. Consider running E4/E5/E6 classical-prior baselines in the same 3-seed controlled style if they were not included in this core sweep.
