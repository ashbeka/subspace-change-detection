# OSCD Band-Image DS Matched Spatial Controls

## 1. Research Question

Does Band-Image Difference Subspace (DS) provide changed-pixel evidence that
cannot be explained by simpler spatial second-moment, projector, reconstruction,
or spatially filtered PCA controls?

This is the killer-null experiment selected by the cross-branch evidence
review. It tests whether the strongest current DS construction has a distinct
geometric contribution. It does not test semantic change, disaster damage, or
deep-learning superiority.

## 2. Construction And Matched Controls

For one date, the 13 aligned Sentinel-2 bands are flattened over the same `N`
valid pixel positions:

```text
X_t in R^(N x 13)
```

Band-Image DS fits rank-12 centered PCA bases `U_pre,U_post in R^(N x 12)`,
constructs the canonical principal-vector difference basis `D`, and scores
each spatial position from the projected 13-band difference:

```text
s_DS(i) = ||[D D^T (X_post-X_pre)]_(i,:)||_2.
```

Three matched nulls use exactly the same spatial sample axis:

1. **Normalized spatial Gram row distance.** Subtract the mean of the 13
   band-image samples at each spatial position, matching PCA on `X_t^T`, and
   normalize by the Frobenius norm, `A_t=X_t/||X_t||_F`. The score is
   `||(A_pre A_pre^T-A_post A_post^T)_(i,:)||_2`. It retains every singular
   mode, not only a rank-truncated orientation.
2. **Projector row distance.** The score is
   `||(U_pre U_pre^T-U_post U_post^T)_(i,:)||_2`. It isolates the spatial
   eigenspace change without using the observed spectral-difference vector.
3. **Cross-reconstruction novelty.** Each centered date matrix is reconstructed
   by the other date's rank-12 subspace. The score is the nonnegative excess
   cross-reconstruction energy over self-reconstruction.

All row scores are evaluated without materializing an `N x N` operator. Small
`13 x 13` identities give the exact stated row norms.

Formula tests verify:

- identical matrices produce zero for every matched control;
- the Gram and projector row formulas match explicit `N x N` calculations;
- every score is equivariant to a common permutation of spatial coordinates.

## 3. Protocol

- Dataset: all 24 OSCD cities.
- Rank: 12, the maximum useful centered Band-Image rank.
- Input: existing 13-band normalized OSCD stacks and valid masks.
- Primary metric: average precision (AP).
- Other metrics: AUROC, best F1/IoU, Otsu F1/IoU, raw-L2 correlation, runtime,
  city wins, paired bootstrap intervals, and paired Wilcoxon tests.
- No per-city tuning and no learned fusion weights.
- Official split summaries use 14 OSCD training cities and 10 test cities.

The experiment is retrospective: earlier OSCD labels influenced the choice of
rank and candidate methods. Official test-city statistics are stronger than an
all-city mean but are not a pristine external confirmation.

## 4. All-City Matched-Control Result

Output:
`phase1/outputs/band_image_matched_nulls_centering_fixed_allcities_20260622_013413/`.

An earlier run centered each band over space in the Gram and reconstruction
controls. That did not match the Band-Image PCA sample convention. The output
above is the corrected result; the earlier control numbers are superseded.

| Method | Mean AUROC | Mean AP | Mean best F1 | Mean Otsu F1 |
|---|---:|---:|---:|---:|
| Multiscale PCA-diff (`sigma=0,1,2`) | 0.8453 | **0.2680** | 0.3151 | **0.2232** |
| Smoothed PCA-diff (`sigma=1`) | 0.8416 | 0.2679 | 0.3162 | 0.2242 |
| PCA + Band-Image + IR-MAD rank fusion | **0.8708** | 0.2674 | **0.3257** | 0.1084 |
| PCA-diff | 0.8406 | 0.2541 | 0.3074 | 0.2164 |
| Band-Image DS | 0.8477 | 0.2410 | 0.3021 | 0.2036 |
| Raw L2/CVA | 0.7717 | 0.2261 | 0.2873 | 0.1802 |
| Band-Image cross-reconstruction | 0.8462 | 0.2153 | 0.2784 | 0.1801 |
| IR-MAD | 0.8471 | 0.2138 | 0.2846 | 0.0547 |
| Normalized spatial Gram row distance | 0.7069 | 0.1417 | 0.2034 | 0.1491 |
| Projector row distance | 0.7916 | 0.1024 | 0.1752 | 0.1047 |

Paired AP findings over 24 cities:

| Comparison | Mean delta | 95% city-bootstrap interval | Wins | Wilcoxon p |
|---|---:|---:|---:|---:|
| Band-Image DS minus spatial Gram | +0.0993 | [+0.0705,+0.1298] | 24/24 | <0.000001 |
| Band-Image DS minus projector row | +0.1386 | [+0.0906,+0.1921] | 23/24 | 0.000001 |
| Band-Image DS minus cross-reconstruction | +0.0257 | [+0.0128,+0.0391] | 20/24 | 0.0008 |
| Band-Image DS minus PCA-diff | -0.0130 | [-0.0290,+0.0041] | 8/24 | 0.0646 |
| Smoothed PCA minus PCA-diff | +0.0138 | [+0.0053,+0.0229] | 18/24 | 0.0035 |
| Multiscale PCA minus PCA-diff | +0.0139 | [+0.0060,+0.0220] | 19/24 | 0.0014 |
| Band-Image DS minus smoothed PCA | -0.0268 | [-0.0475,-0.0033] | 6/24 | 0.0105 |

Interpretation:

- Band-Image DS is not merely normalized full Gram change or projector-row
  orientation; it beats both decisively.
- It also beats the rank-matched cross-reconstruction control, showing that
  canonical DS projection contributes evidence not captured by cross-prediction
  alone. It nevertheless remains below ordinary PCA-diff in mean AP.
- Spatially filtered PCA is the strongest individual map and significantly
  improves the retrospective all-city AP.

## 5. Official Split Check

Mean AP by split:

| Method | Train (14) | Test (10) |
|---|---:|---:|
| Smoothed PCA | **0.2348** | 0.3141 |
| Multiscale PCA | 0.2339 | 0.3157 |
| Old PCA + Band-Image + IR-MAD fusion | 0.2265 | **0.3246** |
| PCA-diff | 0.2164 | 0.3067 |
| Band-Image DS | 0.2024 | 0.2951 |
| Cross-reconstruction | 0.1734 | 0.2739 |

On the ten official test cities, multiscale PCA minus PCA-diff is `+0.0090`
AP with interval `[-0.0037,+0.0213]`; smoothed PCA minus PCA-diff is `+0.0074`
with interval `[-0.0048,+0.0205]`. The smaller test set does not confirm either
gain independently.

## 6. Complementarity Follow-Up

Output:
`phase1/outputs/spatial_complementarity_centering_fixed_allcities_20260622_014108/`.

Equal percentile-rank fusions were tested to ask whether Band-Image DS adds to
the strongest smoothed PCA map, and whether a non-DS cross-reconstruction map
can substitute for it.

| Method | All-city AUROC | All-city AP | Train AP | Test AP |
|---|---:|---:|---:|---:|
| Smoothed PCA + Band-Image + IR-MAD | **0.8760** | **0.2780** | 0.2404 | **0.3305** |
| Smoothed PCA + cross-reconstruction + IR-MAD | 0.8742 | 0.2657 | 0.2277 | 0.3190 |
| Smoothed PCA + Band-Image | 0.8611 | 0.2716 | 0.2348 | 0.3232 |
| Smoothed PCA | 0.8416 | 0.2679 | 0.2348 | 0.3141 |
| Smoothed PCA + IR-MAD | 0.8724 | 0.2672 | 0.2366 | 0.3101 |
| PCA-diff | 0.8406 | 0.2541 | 0.2164 | 0.3067 |

Test-city comparisons:

- Smoothed PCA + Band-Image minus smoothed PCA: `+0.0091` AP, interval
  `[-0.0055,+0.0244]`, 7/10 wins, `p=0.2754`.
- Smoothed PCA + Band-Image + IR-MAD minus smoothed PCA: `+0.0165`, interval
  `[-0.0111,+0.0486]`, 7/10 wins, `p=0.2324`.
- The same three-way fusion minus PCA-diff: `+0.0238`, interval
  `[+0.0073,+0.0450]`, 8/10 wins, `p=0.0195`.
- DS fusion minus cross-reconstruction fusion: `+0.0115` AP, interval
  `[+0.0037,+0.0227]`, 9/10 wins, `p=0.0098`.

The fusion evidence is positive relative to ordinary PCA-diff and the matched
cross-reconstruction substitute. This is the strongest evidence that the DS
operation contributes distinct ranking information. It still does not prove
that DS improves over smoothed PCA alone: the test-city interval for that
comparison crosses zero. The DS three-way fusion also has the highest training
AP among the tested fusions (`0.2404`), so this comparison does not rely on
selecting it only after seeing test performance.

## 7. Qualitative Findings

- Beirut: projector distance follows broad urban texture/roads; spatial Gram
  emphasizes coastline and airport structure; Band-Image and
  cross-reconstruction share broad responses but differ in ranking; smoothed PCA ranks labeled regions more
  cleanly.
- Norcia: Band-Image DS is a real city-specific success (AP about `0.21` vs
  smoothed PCA about `0.07`) and emphasizes urban/structural changes, but also
  responds to agricultural pseudo-change.
- Across cities, smoothing wins most often; Band-Image DS is stronger in a
  small set including Norcia, Paris, and Saclay East.

Representative grids:

- `phase1/outputs/band_image_matched_nulls_centering_fixed_allcities_20260622_013413/runs/rank12_matched_spatial_nulls__beirut/comparison_grid.png`
- `phase1/outputs/band_image_matched_nulls_centering_fixed_allcities_20260622_013413/runs/rank12_matched_spatial_nulls__norcia/comparison_grid.png`

## 8. Research Decision

1. The matched-null experiment is complete. DS has a distinct contribution
   relative to matched Gram, projector, and cross-reconstruction controls, but
   does not beat the strongest individual spatial-PCA control.
2. Retain Band-Image DS as the strongest DS construction and as a city-specific
   complementary ranking component.
3. Promote spatially filtered PCA as the strongest individual empirical result,
   but mark it retrospective until external/frozen confirmation.
4. Treat the DS-vs-cross fusion result as positive internal evidence. Do not
   call it externally validated or claim superiority over smoothed PCA alone.
5. The seminar may present a positive construction result and complementary
   ranking evidence, but the publication claim requires external validation or
   a task where the geometric component has a distinct capability.

Preferred external gate: xBD-S12 (`https://github.com/prs-eth/xbd-s12`). It
provides co-registered pre/post Sentinel-2 L2A 12-band patches and xBD-derived
building-damage masks with an event split. The release is about `8.87 GB`; its
task and mask semantics differ from OSCD and require a separate, frozen
protocol rather than a direct reuse of OSCD thresholds.

## 9. Reproduction

Matched controls:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank12_matched_spatial_nulls:12:band_image_norm+band_image_spatial_gram+band_image_projector_distance+band_image_cross_reconstruction+smoothed_pca_sigma1+multiscale_pca_sigma0_1_2+ir_mad+rank_fusion_pca_band_irmad" --output-dir "phase1/outputs/band_image_matched_nulls_allcities_$tag" --continue-on-error --no-save-npy
```

Complementarity fusions:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-spatial-subspace-sweep --cities all --configs "rank12_spatial_complementarity:12:band_image_norm+band_image_cross_reconstruction+smoothed_pca_sigma1+ir_mad+rank_fusion_smoothed_pca_band+rank_fusion_smoothed_pca_irmad+rank_fusion_smoothed_pca_band_irmad+rank_fusion_smoothed_pca_cross_irmad" --output-dir "phase1/outputs/spatial_complementarity_fusions_allcities_$tag" --continue-on-error --no-save-npy
```
