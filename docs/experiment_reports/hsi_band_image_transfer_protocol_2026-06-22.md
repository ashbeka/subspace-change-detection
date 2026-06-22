# Frozen HSI Band-Image Transfer Protocol

## 1. Question

Does the spatial-axis Band-Image construction that produced candidate-
localization evidence on OSCD and xBD-S12 transfer without label tuning to
labeled bitemporal hyperspectral change benchmarks?

This is a pressure test of sensor and spectral-dimensionality transfer. It is
not a third disaster benchmark and it is not a claim that hyperspectral and
Sentinel-2 change labels have identical semantics.

## 2. Frozen Construction

For every scene and date:

```text
X_t in R^(N_spatial x B_retained_bands)
```

- sample unit: one complete registered band image;
- ambient coordinates: valid spatial positions;
- rank: `11`, transferred unchanged from the frozen xBD-S12 protocol;
- fitting: centered PCA;
- normalization: joint robust band-wise z-score estimated without labels;
- band policy: remove only nonfinite/constant bands;
- score polarity: larger distance means more change; labels never reverse it.

The methods are raw L2/CVA, spectral angle, rank-11 PCA-diff, Gaussian-smoothed
PCA-diff, ten-iteration IR-MAD, canonical Band-Image DS, matched symmetric
cross-reconstruction, normalized spatial Gram distance, and spatial projector
distance.

## 3. Data And Metrics

Run all four local labeled pairs:

- Benton;
- Hermiston;
- Farmland;
- Shenzhen.

Average precision is primary. Report AUROC, best F1/IoU, Otsu F1/IoU,
runtime, score maps, and 16-pixel spatial block-bootstrap AP differences for:

- projector minus IR-MAD;
- projector minus smoothed PCA;
- canonical DS minus cross-reconstruction;
- canonical DS minus smoothed PCA.

## 4. Evidence Boundary

These datasets were inspected in earlier project experiments, but not to tune
this fixed Band-Image rank-11 transfer. The result is therefore stronger than
another OSCD ablation but weaker than a pristine unseen-dataset confirmation.

Continue to a neural-prior experiment only if geometry shows a consistent,
task-relevant contribution beyond both IR-MAD and smoothed PCA. A single-scene
win, inverse-polarity result, or DS-only win over cross-reconstruction is not
enough.

## 5. Command

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-hsi-band-image-transfer --datasets "benton,hermiston,farmland,shenzhen" --rank 11 --bootstrap 500 --output-dir "phase1/outputs/hsi_band_image_transfer_$tag"
```
