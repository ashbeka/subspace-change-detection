# Seasonal Observation Subspace Stress Test

## 1. Question

Can a real temporal sample set - repeated observations of the same aligned
patch during one season - produce a useful subspace for detecting abrupt
seasonal-regime changes across years?

This is a controlled synthetic method test. It is not irrigation or satellite
change-detection performance.

## 2. Construction

For year `y`, each of twelve monthly six-band patch cubes is flattened into one
spatial-spectral observation:

```text
x_(y,m) in R^(B*N)
X_y = [x_(y,1), ..., x_(y,12)] in R^((B*N) x 12)
S_y = PCA_r(X_y)
```

The temporal dates are therefore the related sample set. Adjacent annual
subspaces use first DS and Grassmann distance; triples use paper-faithful second
DS with along/orthogonal decomposition.

Primary sources:

- [Kanai et al. 2023 time-series DS](https://arxiv.org/abs/2303.17802)
- [Fukui et al. 2024 second-order DS](https://arxiv.org/abs/2409.08563)
- [IrrMapper paper](https://doi.org/10.3390/rs12142328)
- [IrrMapper Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/UMT_Climate_IrrMapper_RF_v1_2)

## 3. Stress Protocol

The study generated five-year sequences under ten scenarios:

- stable dry and stable irrigated cycles;
- global gain/offset;
- seasonal phase shift;
- two missing/interpolated composites;
- one-pixel translation;
- gradual shape drift;
- abrupt seasonal-shape activation and removal;
- abrupt amplitude-only change.

The main run used `80` independent sequences per scenario, four candidate
boundaries per sequence, ranks `1,2,4,8`, three preprocessing modes, and `200`
sequence-level bootstrap resamples. The positive-boundary prevalence per
configuration was `7.5%` (`240/3200`). Noise-pressure runs used reflectance
noise `0.02` and `0.04` with `40` repeats and `100` bootstrap resamples.

Compared scores:

- first DS magnitude and Grassmann distance;
- minimum principal angle;
- seasonal matrix-energy and normalized singular-spectrum change;
- raw monthly-cube RMS;
- NDVI-curve RMS and NDVI-amplitude change;
- second DS total, along, and orthogonal components;
- raw and NDVI second-difference controls.

## 4. Main Results

Best configuration for each main score at noise `0.008`:

| score | AUROC | AP (95% bootstrap CI) | event-boundary top-1 | configuration |
|---|---:|---:|---:|---|
| first DS magnitude | 0.950 | 0.403 (0.356-0.469) | 1.000 | uncentered, rank 1 |
| raw monthly-cube RMS | 0.961 | 0.459 (0.394-0.532) | 1.000 | control |
| NDVI-curve RMS | 1.000 | 1.000 (1.000-1.000) | 1.000 | control |
| seasonal energy change | 1.000 | 1.000 (1.000-1.000) | 1.000 | feature-centered control |
| normalized singular-spectrum change | 1.000 | 1.000 (1.000-1.000) | 1.000 | feature-centered control |
| second DS along component | 0.905 | 0.488 (0.447-0.542) | 0.788 | feature-centered, rank 2 |
| NDVI second difference | 0.963 | 0.582 (0.557-0.618) | 0.738 | control |

At noise `0.04`, the best second-order along component reached AP `0.698`
(`0.649-0.778`) versus NDVI second-difference AP `0.674` (`0.622-0.730`). The
intervals overlap, so this is a follow-up hypothesis, not evidence of
superiority.

Rank sensitivity is severe. Rank-1 DS was the only consistently useful first
geometry. Higher ranks approached prevalence-level AP because noisy or weak
directions dominated the subspace comparison.

## 5. Failure Boundary

The geometry has one useful but limited behavior: within a sequence, rank-1 DS
placed the true abrupt boundary first in every synthetic event sequence. It did
not provide a stable absolute score across sequences.

Using a threshold fitted to stable cycles, feature-centered rank-1 DS false
alarm rates were:

| nuisance | false-alarm rate |
|---|---:|
| global gain/offset | 0.000 |
| phase shift | 0.000 |
| gradual shape drift | 0.528 |
| one-pixel translation | 0.559 |
| missing composites | 0.997 |

Therefore the current construction is:

- robust to the tested global radiometric and phase nuisances after centering;
- useful for relative within-sequence event ranking;
- not calibrated across fields/sequences;
- highly vulnerable to missing composites and registration;
- unable to justify replacing simple NDVI or singular-value controls.

The date-order permutation unit test also confirms a structural limitation:
ordinary seasonal PCA is order-invariant. An order-aware Hankel/SSA or
product-Grassmann extension is justified only after real-data evaluation.

## 6. Real-Data Gate

IrrMapper is a plausible weak-label source because its public v1.2 collection
spans 1986 through 2024 and overlaps Sentinel-2. However, IrrMapper maps are
annual random-forest predictions, not independently annotated irrigation switch
dates. The paper also documents limitations for winter seasons, orchards,
vineyards, sparse irrigation, wetlands, and changing dryland classes.

The local feasibility command verified public catalog coverage but could not
query an AOI because this machine has no Earth Engine-enabled Cloud project.
No real result was fabricated.

Next required sequence:

```text
Earth Engine project access
-> transition-prevalence and cloud-coverage report
-> small geography/year sample
-> independent visual/manual transition verification
-> first/second DS versus NDVI, raw, min-angle, MOSUM/BFAST/JUST
-> held-out geography/year evaluation
-> DynamicEarthNet or another independent-label repeat
```

## 7. Decision

Continue the real temporal-observation-set experiment, but do not frame DS as
the expected detector winner. The defensible hypothesis is narrower:

```text
Low-rank seasonal subspace geometry may provide radiometrically robust,
within-sequence timing and trajectory-characterization evidence, while
singular-value/NDVI controls carry most amplitude information.
```

The experiment should explicitly test a two-part representation:

1. subspace orientation for seasonal mode/shape change;
2. singular energy for amplitude change.

If real labels do not reproduce the timing or noisy second-along result, retain
this as a negative diagnostic and pivot to order-aware temporal embeddings or
the hyperspectral benchmark route.

## 8. Artifacts

- Main: `phase1/outputs/seasonal_regime_subspace_study_20260620_020447/`
- Noise `0.02`: `phase1/outputs/seasonal_regime_subspace_noise0p020_20260620_020905/`
- Noise `0.04`: `phase1/outputs/seasonal_regime_subspace_noise0p040_20260620_021020/`
- Feasibility: `phase1/outputs/irrigation_regime_data_feasibility_20260620_021343/`
- Method: `phase1/subspace/seasonal_observations.py`
- Runner: `phase1/scripts/evaluate_seasonal_regime_subspaces.py`
- Tests: `tests/test_seasonal_observation_subspaces.py`
