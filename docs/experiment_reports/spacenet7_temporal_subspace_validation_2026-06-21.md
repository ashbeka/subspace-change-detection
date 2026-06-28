# SpaceNet 7 Temporal Subspace Validation

## 1. Research Question

Does a rolling, spatially local trajectory-subspace construction provide
independently validated evidence of new-building areas in a real monthly image
sequence, and does Fukui-style second Difference Subspace (DS) geometry add
information beyond fair radiometric controls?

This experiment is a gate for one exact construction. It is not a validation
of every possible DS, multispectral, or satellite-time-series method.

## 2. Dataset And Labels

SpaceNet 7/MUDS supplies monthly 4 m RGB mosaics, unusable-data masks, and
building polygons with persistent identifiers. A positive event is the first
appearance of a persistent building ID in `labels_match_pix`. Those geometries
are already expressed in image-pixel coordinates and are rasterized with the
identity affine transform. UDM polygons instead declare CRS84 coordinates and
are reprojected to each image CRS before rasterization.

The discovery AOI was used to debug the protocol. Three development AOIs were
used to select one fixed candidate. Four additional AOIs were selected before
scoring, at approximate 0.1, 0.4, 0.6, and 0.9 building-growth quantiles, and
used only for confirmation:

- `L15-1691E-1211N_6764_3347_13`
- `L15-1672E-1207N_6691_3363_13`
- `L15-0586E-1127N_2345_3680_13`
- `L15-1748E-1247N_6993_3202_13`

The originally selected fourth AOI, `L15-1848E-0793N_7394_5018_13`, was
excluded before outcome scoring because long runs of months are entirely UDM-
masked and no six-month comparison has valid support. The replacement was the
nearest preceding AOI in the official sorted training inventory with adequate
UDM coverage. It was scored once.

This is cell-level construction-event ranking, not the SpaceNet SCOT tracking
metric and not Sentinel-2 validation.

## 3. Frozen Construction

Each AOI is divided into a fixed `8 x 8` grid. For one cell and month, the RGB
pixels are flattened into

```text
x_t in R^(3 * N_cell).
```

For a six-month rolling window and lag two, trajectory columns are

```text
h_j = [x_j; x_(j+1)]
H_t = [h_(t-5), ..., h_(t-1)] in R^((6 * N_cell) x 5).
```

For each comparison, validity is intersected only across dates required by the
previous, current, and (for second DS) following rolling windows. Requiring one
pixel to remain valid for the complete sequence would discard usable temporal
support and can eliminate an AOI entirely.

Rank-two PCA left singular vectors define one local trajectory subspace per
window. Consecutive subspaces provide first DS quantities. Consecutive first
DSs provide paper-guided second total, along-geodesic, and orthogonal
magnitudes. The candidate detector used the second orthogonal magnitude.

The geometry was not tuned on the four confirmation AOIs.

## 4. Fair Controls And Fixed Hybrid

Controls were computed on the same cells and dates:

- standardized raw pair RMS;
- standardized raw second-difference RMS;
- equal-weight, within-date percentile-rank fusion of those two controls.

A fixed exploratory hybrid added the second-DS orthogonal percentile rank to
the two radiometric ranks. This fusion is a project-designed diagnostic, not a
formula from the DS literature and not a trained model.

## 5. Confirmation Results

Macro results across the four untouched AOIs:

| Method | Mean AP | Median AP | AOI wins |
|---|---:|---:|---:|
| Second-DS orthogonal magnitude | 0.1127 | 0.1158 | 0 |
| Standardized raw pair RMS | 0.1615 | 0.1660 | 1 |
| Standardized raw second-difference RMS | 0.1502 | 0.1449 | 0 |
| Two-radiometric rank fusion | 0.1910 | 0.1873 | 3 |
| Geometry plus radiometric rank fusion | 0.1747 | 0.1752 | 0 |

The full geometric decomposition was also inspected descriptively:

| Geometric quantity | Macro AP | Macro AUROC |
|---|---:|---:|
| First-DS magnitude | 0.1313 | 0.5630 |
| First Grassmann geodesic | 0.1307 | 0.5627 |
| Second total magnitude | 0.1229 | 0.5503 |
| Second along magnitude | 0.1226 | 0.5451 |
| Second orthogonal magnitude | 0.1127 | 0.5055 |
| Second orthogonal fraction | 0.1100 | 0.4813 |

First-DS magnitude and first geodesic distance are nearly redundant in this
operating range (mean within-AOI Spearman `rho=0.9999`). Second total is mostly
ordered by its along component (`rho=0.9596`), while the orthogonal component
has near-zero correlation with raw second difference (`rho=-0.0314`) but also
near-random event discrimination. The decomposition exposes different
geometry; it does not make that geometry useful for this target.

Hierarchical bootstrap resampled AOIs and then monthly transitions. The fixed
three-score fusion minus the two-radiometric fusion was:

```text
delta macro AP = -0.0163
95% interval = [-0.0531, +0.0140]
P(delta > 0) = 0.189
```

The interval crosses zero and the point estimate is negative. There is no
evidence that geometry adds useful predictive information; the simpler two-
radiometric fusion wins three of four AOIs.

## 6. Interpretation And Decision

The strong hypothesis is falsified for this configuration:

```text
Rolling rank-two RGB trajectory DS does not outperform fair standardized
radiometric evidence for localizing newly appearing buildings in SpaceNet 7.
```

The second-DS implementation is still useful as a verified trajectory
descriptor, but the tested magnitude is a weak detector. Geometry alone is
substantially worse than standardized raw controls. The exact grid, window,
rank, RGB representation, and new-building target should now be closed rather
than tuned on the confirmation AOIs.

This negative result narrows the research space. It does not falsify:

- multispectral or hyperspectral feature-isolation subspaces;
- long seasonal descriptors with explicit nuisance modeling;
- semantic/object-level geometric representations;
- DS as an interpretable companion descriptor rather than a detector.

Any next route must introduce a new scientific mechanism or dataset and obtain
fresh held-out evidence. Reparameterizing this same detector on the seven
non-discovery AOIs would invalidate the confirmation gate.

## 7. Reproducibility Artifacts

- Evaluator: `phase1/scripts/evaluate_spacenet7_temporal_subspaces.py`
- Dataset loader: `phase1/data/spacenet7_dataset.py`
- Fixed-fusion analysis: `phase1/scripts/analyze_spacenet7_hybrid_validation.py`
- Confirmation analysis: `phase1/outputs/spacenet7_confirmation_validmask_hybrid_20260621_013000/`
- Main figure: `aoi_average_precision_comparison.png`
- Formula/data tests: `tests/test_spacenet7_dataset.py` and temporal-subspace tests

Primary dataset source: Van Etten et al. (2021), *The Multi-Temporal Urban
Development SpaceNet Dataset*, DOI `10.1109/CVPR46437.2021.00633`.
