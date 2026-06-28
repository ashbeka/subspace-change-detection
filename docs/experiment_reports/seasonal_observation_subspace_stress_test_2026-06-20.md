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

## 9. Order-Aware Construction Correction

The first seasonal construction above is an unordered observation matrix. It
does **not** implement Kanai et al.'s trajectory construction. For any date
permutation matrix `P`,

```text
span(X P) = span(X).
```

At full numerical rank it is also invariant to any invertible mixing of the
date columns. Therefore it represents the annual observation span but cannot
encode temporal order.

The corrected order-aware adaptation is the multivariate block trajectory
matrix

```text
h_j = [x_j; x_(j+1); ...; x_(j+L-1)]
H_L = [h_1, ..., h_(T-L+1)] in R^((D*L) x (T-L+1)).
```

This follows the role of Kanai et al.'s scalar SSA trajectory matrix while
replacing each scalar lag by one flattened spatial-spectral observation. It is
a project adaptation, not an equation copied from that paper. A first-
difference matrix `[x_2-x_1, ..., x_T-x_(T-1)]` is retained as a simpler
order-aware control.

Eight formula/invariance tests verify equal-sequence behavior, matrix shapes,
unordered permutation/mixing invariance, order-aware permutation response, and
the covariance diagnostic.

## 10. Real-Background Controlled Study

Five MultiSenGE patches supplied real 23-date Sentinel-2 backgrounds. The
study sampled eight 32x32 crops per patch and applied known transformations to
two later pseudo-years. This produced `2560` global/crop-level rows per
preprocessing condition.

Tested events:

- global seasonal-amplitude change;
- localized red/NIR/SWIR seasonal-mode change.

Tested negatives/confounders:

- stable acquisition jitter;
- gain/offset;
- seasonal phase shift;
- missing/interpolated composites;
- one-pixel translation.

The normalized order-aware geometry did not reproduce the synthetic result.
Best DS-family AP was `0.365`; NDVI-amplitude AP was `0.875`. Across four
preprocessing modes, the best pure orientation result was AP `0.448`
(`column_l2`, unordered second-orthogonal). Unordered singular-spectrum change
reached AP `0.822`, but this aggregate result was driven mainly by the global
amplitude event.

Interpretation: DS compares basis orientation and discards singular-value
energy. An amplitude change can stay inside the same seasonal mode span and is
therefore invisible to pure orientation geometry. Feature centering plus
column normalization can remove that amplitude exactly.

## 11. Multiscale Localization And Fair Controls

The follow-up fitted one temporal subspace per grid cell and evaluated exact
injected pixels against stable/nuisance pixels. Event supports were randomized
off grid before the final run. The strongest resolution was an 8x8 grid over
32x32 crops (4x4 pixels per cell).

Gaussian low-pass support was predeclared as a registration-robustness control:

| sigma | local eigenspectrum AP | local DS AP | local NDVI AP | eigenspectrum translation false alarm | eigenspectrum missing-composite false alarm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.516 | 0.238 | 0.421 | 0.847 | 0.843 |
| 1 | 0.650 | 0.376 | 0.460 | 0.607 | 0.563 |
| 2 | 0.688 | 0.456 | 0.486 | 0.292 | 0.358 |

At sigma 2, patch-bootstrap results for localized change were:

| score | AUROC | AP (95% patch-bootstrap CI) |
|---|---:|---:|
| normalized temporal eigenspectrum change | 0.973 | 0.688 (0.578-0.792) |
| NDMI amplitude change | 0.947 | 0.680 (0.526-0.822) |
| normalized covariance-operator change | 0.950 | 0.667 (0.374-0.849) |
| NBR amplitude change | 0.937 | 0.637 (0.478-0.816) |
| NDVI amplitude change | 0.898 | 0.486 (0.416-0.590) |
| first DS magnitude | 0.923 | 0.456 (0.247-0.668) |

The eigenspectrum is therefore competitive with the best spectral-index
control, not superior to it. Its useful tradeoff is invariance to the tested
gain/offset: `0.050` false-alarm rate versus `0.970` for NDMI. It remains more
sensitive to missing composites (`0.358` versus `0.156`).

The paired patch bootstrap confirms the boundary: eigenspectrum minus NDMI was
only `+0.008` AP (95% CI `-0.249` to `+0.244`), while eigenspectrum minus first
DS was `+0.233` (95% CI `+0.084` to `+0.354`).

The earlier NDVI-only comparison overstated the result. Adding NDMI, NBR,
per-band amplitude, and multispectral curve controls removed that overclaim.

## 12. Current Finding And Decision Gate

The surviving controlled finding is:

```text
For localized seasonal spectral-mode interventions on real Sentinel-2
backgrounds, fine local support plus low-pass preprocessing makes temporal
eigenspectrum change competitive with strong spectral-index controls while
retaining better tested gain/offset invariance. Pure DS orientation is useful
but weaker. Order-aware trajectory DS restores temporal-order sensitivity but
does not improve this event-localization task.
```

This is not yet publishable performance evidence because the events are
injected and the five MultiSenGE patches do not contain event-time labels. The
next gate is an independently labeled temporal slice, preferably a selectively
acquired DynamicEarthNet AOI or manually verified IrrMapper/Sentinel-2
transition. Required comparisons are local eigenspectrum, first/second DS,
NDVI/NDMI/NBR, raw multispectral controls, and one established temporal break
method. If the result does not survive held-out real events, retain it as a
negative diagnostic about what subspace orientation discards.

Additional artifacts:

- order-aware synthetic ablation:
  `phase1/outputs/order_aware_seasonal_ablation_20260620_062052/`;
- real-background global study:
  `phase1/outputs/multisenge_order_aware_interventions_20260620_193159/`;
- off-grid multiscale study:
  `phase1/outputs/multiscale_order_aware_offgrid_20260620_195608/`;
- sigma-2 fair-control study:
  `phase1/outputs/multiscale_order_aware_fair_controls_20260620_201745/`;
- winning-configuration visualization:
  `phase1/outputs/multiscale_order_aware_winning_visual_20260620_201208/`.
