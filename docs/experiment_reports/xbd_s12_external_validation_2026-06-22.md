# xBD-S12 External Validation: Spatial Subspace Geometry

## 1. Research Question

Does the Band-Image subspace construction transfer from OSCD to an independent,
event-disjoint multispectral disaster dataset, and what information does it
actually encode: damage, building localization, or generic radiometric change?

The protocol was frozen before inspecting xBD-S12 method outputs. The primary
run evaluates all `1,577` official test-event patches from five unseen
disasters. No xBD-S12 label selected rank, preprocessing, smoothing, fusion
weights, or thresholds.

## 2. Evidence Status

- `[external experiment evidence]` Official xBD-S12 Sentinel-2 release,
  event-disjoint test split, `12 x 128 x 128` pre/post inputs.
- `[external experiment evidence]` `1,577/1,577` patches completed; zero method
  failures.
- `[implementation evidence]` Official 1st/99th-percentile Sentinel-2
  normalization and official categorical mask downsampling are reproduced.
- `[limitation]` Only five independent test events exist. The smallest possible
  two-sided Wilcoxon p-value for a consistent five-event direction is `0.0625`.
- `[limitation]` xBD labels originate at VHR resolution and are downsampled to
  4 m Sentinel-2 patches. This is coarse damaged-pixel evidence, not exact
  building-damage segmentation.

## 3. Construction And Comparisons

For each date, the twelve aligned Sentinel-2 band images are flattened into
twelve spatial samples:

```text
X_t in R^(N_spatial x 12), N_spatial = 128 x 128 valid locations.
```

Centered rank-11 PCA gives a spatial subspace for each date. The tested
geometry maps are:

1. canonical Band-Image Difference Subspace projection magnitude;
2. matched symmetric cross-reconstruction control;
3. row-wise projector distance between the two spatial subspaces;
4. normalized spatial Gram distance.

Baselines are raw L2/CVA, spectral angle, PCA-diff, smoothed/multiscale
PCA-diff, and IR-MAD. Equal-weight fusions use within-patch percentile ranks.

## 4. Primary Full-Scene Damage Retrieval

Positives are minor/major/destroyed pixels; intact buildings and background
are negatives. Average precision (AP) is primary because global prevalence is
only `0.01191`.

| Method | Mean event AUROC | Mean event AP | Global AP | Global best F1 |
|---|---:|---:|---:|---:|
| Band-image projector distance | **0.7337** | **0.03015** | **0.03675** | **0.0877** |
| IR-MAD | 0.7296 | 0.02649 | 0.03194 | 0.0725 |
| PCA + DS + IR-MAD rank fusion | 0.6831 | 0.02664 | 0.02985 | 0.0713 |
| Band-Image canonical DS | 0.6260 | 0.02124 | 0.02377 | 0.0600 |
| PCA-diff | 0.5907 | 0.01840 | 0.02123 | 0.0515 |
| Matched cross-reconstruction | 0.6153 | 0.01676 | 0.01955 | 0.0453 |
| Raw L2 / CVA | 0.4426 | 0.00831 | 0.00937 | 0.0235 |

Projector distance beats PCA-diff in all five events. Its exploratory mean
event AP delta is `+0.01175`, with event-cluster bootstrap 95% interval
`[+0.00445,+0.02032]`. It beats IR-MAD in four of five events, but that
interval crosses zero (`+0.00366`, `[-0.00186,+0.00879]`).

![Per-event full-scene AP](assets/xbd_s12_external_2026-06-22/event_ap_heatmap.png)

## 5. DS-Specific Matched-Control Result

Canonical DS is not the best individual method, but it contributes information
that the rank- and input-matched cross-reconstruction substitute does not:

| Comparison | Mean event AP delta | 95% event-bootstrap interval | Event wins | Wilcoxon p |
|---|---:|---:|---:|---:|
| DS - cross-reconstruction, unbuffered | +0.00448 | [+0.00074,+0.01099] | 5/5 | 0.0625 |
| DS-fusion - cross-fusion, unbuffered | +0.00173 | [+0.00014,+0.00413] | 5/5 | 0.0625 |
| DS - cross-reconstruction, 3-pixel stress | +0.00121 | [+0.00005,+0.00285] | 5/5 | 0.0625 |

This is transferable directional evidence for a DS-specific component. It is
not evidence that DS alone is a competitive damage segmenter.

![Paired event AP differences](assets/xbd_s12_external_2026-06-22/paired_event_ap_deltas.png)

## 6. What The Geometry Represents

Three label views separate the task:

| View | Global prevalence | Best method | Global AP | Interpretation |
|---|---:|---|---:|---|
| Full-scene damaged-pixel retrieval | 0.01191 | Projector distance | 0.03675 | Candidate ranking in the whole scene |
| Damage vs intact buildings | 0.33071 | Raw L2 | 0.39876 | Damage discrimination once building support is known |
| Building localization diagnostic | 0.03600 | Projector distance | 0.08039 | Building-related spatial structure |

Projector distance is strongest for building localization and full-scene
retrieval, but it is not strong for damage-versus-intact discrimination. Raw
L2 shows the reverse pattern. Therefore the projector's full-scene lead is
partly a localization effect. The correct interpretation is a possible
**candidate-localization prior**, not a direct damage-severity score.

![Task profile](assets/xbd_s12_external_2026-06-22/task_profile_ap_lift.png)

## 7. Boundary Stress Test

A separate sensitivity run removes both sides of all building boundaries
within three 4 m pixels. This is deliberately severe: it leaves only `3,119`
damaged pixels and `17,176` building-interior pixels across the complete test
set, versus `307,228` and `928,993` unbuffered.

The stress run therefore cannot replace the primary view. It answers whether
results vanish when edges and possible alignment artifacts are excluded.

- Canonical DS still beats cross-reconstruction in all five events.
- The PCA + DS + IR-MAD fusion has the best mean event AP (`0.00319`) for
  full-scene retrieval; IR-MAD is second (`0.00293`).
- Projector distance keeps high mean event AUROC (`0.8174`) but loses the AP
  lead, consistent with its strong boundary/localization sensitivity.
- AP/prevalence lift remains above chance for the principal geometry and
  IR-MAD methods, but absolute AP is unstable because prevalence falls to
  `0.000139`.

![Boundary sensitivity](assets/xbd_s12_external_2026-06-22/boundary_sensitivity_ap_lift.png)

## 8. Defensible Finding

The current defensible finding is:

> A spatial band-image subspace representation transfers across OSCD and
> xBD-S12 as label-free changed-structure evidence. Projector geometry is most
> useful for candidate localization, while canonical DS contributes a smaller
> but event-consistent component beyond matched cross-reconstruction. Neither
> should yet be described as a stand-alone damage classifier.

Do not claim:

- state-of-the-art damage assessment;
- pixel-accurate building damage segmentation;
- that projector distance identifies damage severity;
- statistical significance beyond the five available independent test events;
- novelty for DS itself.

## 9. Next Hypothesis And Decision Gate

The next experiment should test a two-stage, label-free composition developed
on training events only:

```text
spatial projector evidence (where changed structures may be)
    + radiometric evidence (how strongly their spectra changed)
    -> full-scene damaged-candidate ranking
```

Candidate fixed combinations are equal percentile-rank averaging and
multiplication of projector distance with raw L2 or spectral angle. They must
be evaluated by event-group cross-validation on training disasters. The five
test events have already been inspected and must not select the combination.

Before another external claim, also pressure-test centered rank
`{2,4,6,8,10,11}` versus uncentered autocorrelation construction on training
events. Continue only if the qualitative conclusion is stable across rank,
event, and boundary support.

## 10. Reproduction

Primary run:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split test --bootstrap 5000 --maps-per-event 3 --boundary-buffer 0 --output-dir phase1/outputs/xbd_s12_frozen_test_unbuffered_complete_20260622_111613
```

Boundary stress:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-evaluate --split test --bootstrap 5000 --maps-per-event 0 --boundary-buffer 3 --event-only --output-dir phase1/outputs/xbd_s12_frozen_test_boundary3_stress_20260622_114715
```

Figures:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-xbd-s12-summarize --unbuffered phase1/outputs/xbd_s12_frozen_test_unbuffered_complete_20260622_111613 --boundary phase1/outputs/xbd_s12_frozen_test_boundary3_stress_20260622_114715 --output-dir docs/experiment_reports/assets/xbd_s12_external_2026-06-22
```
