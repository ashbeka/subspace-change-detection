# Frozen SpaceNet7 Band-Image Transfer Protocol

## 1. Question

Does the candidate-localization behavior observed on xBD-S12 survive a change
of sensor and task to monthly RGB imagery with persistent building IDs?

This tests building-appearance localization, not damage severity, Sentinel-2
transfer, or the SpaceNet SCOT metric.

## 2. Frozen Construction

- spatial support: non-overlapping `128 x 128` tiles, matching xBD-S12;
- samples: the three complete RGB band images in each tile;
- subspace: centered rank-two PCA, the maximum centered RGB rank;
- target: pixels belonging to persistent building IDs appearing for the first
  time in the current monthly annotation;
- validity: intersection of only the paired months' alpha/UDM masks;
- no label-fitted threshold, rank, tile size, score polarity, or fusion.

The methods are raw L2, spectral angle, PCA-diff, smoothed PCA-diff, IR-MAD,
canonical Band-Image DS, matched cross-reconstruction, spatial Gram distance,
and projector distance.

## 3. Data And Statistics

Use all locally available AOIs under `SpaceNet7_sample`,
`SpaceNet7_validation`, and `SpaceNet7_confirmation`. Report monthly-transition
AP/AUROC, fixed 1/5/10% review-budget retrieval, AOI summaries, and a
hierarchical bootstrap that resamples AOIs and then transitions.

Primary comparisons:

- projector minus IR-MAD;
- projector minus smoothed PCA;
- canonical DS minus cross-reconstruction;
- canonical DS minus smoothed PCA.

## 4. Decision Gate

Proceed to a neural projector-prior experiment only if projector or canonical
DS adds consistent building-appearance evidence beyond both IR-MAD and
smoothed PCA across AOIs. Failure closes the claim that the xBD localization
effect transfers to RGB building construction.

## 5. Command

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spacenet7-band-image-transfer --workers 3 --bootstrap 3000 --output-dir "phase1/outputs/spacenet7_band_image_transfer_$tag"
```
