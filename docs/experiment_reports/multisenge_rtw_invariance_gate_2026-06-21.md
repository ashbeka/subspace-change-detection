# MultiSenGE Randomized Time Warping Invariance Gate

## 1. Research Question

Can a Randomized Time Warping (RTW) hypo-subspace reject harmless differences
in seasonal timing or tempo while detecting changes in the cross-band shape of
a multispectral seasonal cycle, beyond simpler temporal and subspace controls?

This is a controlled mechanism experiment on real Sentinel-2 backgrounds. It
is not natural-change accuracy and does not establish a first remote-sensing use
of RTW.

## 2. Verdict

**No-go for the natural-transition RTW stage.**

RTW was strong on the pooled structural task, but it did not outperform the
simpler order-invariant snapshot subspace:

| Method | Held-out AUROC | Held-out AP |
|---|---:|---:|
| Snapshot subspace | 0.8041 | 0.8156 |
| RTW | 0.8003 | 0.8078 |
| DTW | 0.6866 | 0.7403 |
| TWDTW | 0.6276 | 0.6202 |
| Fourier magnitude | 0.4439 | 0.3640 |
| Phase-aligned harmonic | 0.5064 | 0.4036 |

The paired RTW-minus-snapshot AP difference was `-0.0078`; its patch-repeat
bootstrap 95% interval was `[-0.0807, +0.0435]`. At patch level the inference
is even less precise because only two patches were held out.

## 3. Method Construction

For a sequence `Z=[z_1,...,z_N]`, `z_t in R^d`:

1. sample `R` sequence elements without replacement;
2. sort their indices to preserve temporal order;
3. concatenate them into a Time Elastic vector `f_l in R^(dR)`;
4. repeat `L` times to form `F=[f_1,...,f_L]`;
5. fit the uncentered PCA hypo-subspace `X` from the leading eigenvectors of
   `F F^T`;
6. compare two hypo-subspaces using singular values `kappa_i` of `X^T Y`:

```text
similarity = mean_i(kappa_i^2)
change score = 1 - similarity.
```

The source construction follows Suryanto, Xue, and Fukui (2016), Hiraoka et
al. (2025), and the bundled `TEfeatures.m` reference. Satellite-specific
radiometric preprocessing was treated as an ablation rather than RTW theory.

Selected configuration, using development patches only:

```text
R = 4 ordered dates
L = 64 Time Elastic samples
rank = 5
preprocessing = raw reflectance
RTW sampling replicates = 3
```

## 4. Data And Split

- Source: five MultiSenGE Sentinel-2 patches, each with 23 dates and 10 bands.
- Sample: mean band vector from a valid `32x32` crop at each date.
- Development patches: `32TLT_3855_0`, `32TLT_5654_257`, `32TLT_4369_257`.
- Held-out patches: `32TLT_5397_1028`, `32TLT_4883_514`.
- Six crop/noise repeats per patch.
- `810` final sequence pairs.

All patches come from one tile, so this is a formula/mechanism gate rather than
geographic generalization evidence.

## 5. Controlled Transformations

Nuisances:

- independent stable jitter and sensor noise;
- global seasonal phase shifts;
- monotone nonlinear temporal warps;
- missing composites;
- global gain/offset and band-dependent gains.

Structural targets:

- `relative_band_phase`: shift only NIR/SWIR trajectories while preserving the
  marginal values of each band;
- `marginal_matched_shape`: deform a seasonal spectral mode, then rank-map each
  band back to the exact original marginal distribution.

Easy controls included mean, amplitude, ordinary phenology-shape, and season-
shortening changes. Date permutation was an order-destruction diagnostic and
was excluded from target-versus-nuisance metrics.

## 6. Baselines

- aligned raw and reference-standardized RMS;
- mean spectrum, seasonal amplitude, mean spectral angle, and NDVI difference;
- normalized Fourier magnitude;
- order-3 harmonic curves with optimal circular phase alignment;
- DTW, logistic time-weighted DTW, and soft-DTW;
- order-invariant snapshot PCA/MSM subspace;
- deterministic multichannel Hankel/M-SSA subspace.

TWDTW uses the Maus et al. logistic local cost
`dist + 1/(1+exp(-alpha*(elapsed-beta)))`, cross-checked against the public
`vwmaus/twdtw` source.

## 7. Target Decomposition

The pooled result hides two different regimes:

| Target | RTW AP | Snapshot AP | DTW AP |
|---|---:|---:|---:|
| Marginal-matched shape | 0.4533 | 0.4872 | 0.3102 |
| Relative band phase | 0.9511 | 0.9550 | 0.9459 |

The high pooled AP is therefore driven by relative band-phase changes. Those
changes alter cross-band covariance and are already detected by the simpler
snapshot subspace and DTW. RTW did not isolate a distinct advantage.

RTW was somewhat stronger than snapshot PCA when all radiometric, missing-data,
and noise nuisances were pooled (`+0.0341` AP), but the paired interval
`[-0.0328, +0.0995]` includes zero. The all-change difference was similarly
uncertain: `+0.0308`, interval `[-0.0051, +0.0670]`.

## 8. Sampling Stability

- Mean RTW sampling standard deviation: `0.0346`.
- Median sampling standard deviation: `0.0279`.
- Median sampling-standard-deviation/score ratio: `0.3012`.

This variance is not fatal, but it weakens small score differences and makes a
single RTW draw inappropriate for evaluation.

## 9. Interpretation

The experiment confirms that RTW produces an order-aware representation: date
permutation changes RTW while leaving snapshot PCA effectively unchanged. It
also suppresses several gain/missing-data nuisances.

However, order awareness was not an incremental detection advantage in the
defined satellite task. The order-invariant snapshot subspace already captured
the cross-band covariance shift that made the main structural target easy. The
harder marginal-matched shape target remained weak for both methods.

According to the preregistered rule, the project should not proceed to weakly
labeled irrigation or crop-transition claims with RTW. Doing so would search
for a favorable dataset after the controlled mechanism failed to beat its
killer null.

The useful result belongs in the diagnostic research line:

```text
RTW adds temporal-order sensitivity, but in this Sentinel-2 gate its apparent
change signal was already captured by a simpler spectral snapshot subspace.
```

## 10. Limitations

- five patches from one tile;
- controlled transformations rather than natural change labels;
- patch-mean sequences omit within-patch spatial arrangement;
- only one selected RTW family was evaluated on holdout;
- marginal matching preserves per-band distributions but is not a physical
  crop-growth simulator;
- absence of an RTW improvement does not invalidate RTW for recognition or
  other sequence tasks.

## 11. Reproduction

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-rtw-invariance-gate --output-dir phase1/outputs/multisenge_rtw_invariance_staged_full_20260621_063708
```

Output directory:

`phase1/outputs/multisenge_rtw_invariance_staged_full_20260621_063708`

Key outputs:

- `method_summary.csv`;
- `paired_method_deltas.csv`;
- `structural_target_breakdown.csv`;
- `holdout_method_ap.png`;
- `heldout_intervention_examples.png`;
- `paired_ap_deltas.png`;
- `posthoc_decision.json`.
