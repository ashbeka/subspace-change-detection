# Multispectral Temporal Difference-Subspace Report

## 1. Question

Can paper-faithful first/second Difference Subspaces and geodesic decomposition
provide distinct, spatially attributable evidence of change in registered
multispectral satellite time series?

This is an exploratory method report, not a performance claim. It directly
implements Sensei's request to generate sequential satellite subspaces and
inspect first/second magnitudes and geodesic components.

## 2. Subspace Construction

For date `t`, the common-mask band stack is rearranged as:

```text
X_t in R^(N common spatial coordinates x B band-image vectors).
```

The leading left singular vectors form:

```text
S_t in Gr(r, N).
```

Full-rank experiments use `r=B`: 10 for MultiSenGE and 4 for the IPOL RGBI
sequence. Centering removes each band's spatial mean before fitting.

This preserves pixel coordinate as an ambient dimension, unlike the old OSCD
pixel-spectrum matrix `R^(B x N)`.

## 3. Geometry

For singular values `cos(theta_i)` of `S_a^T S_b`:

```text
first magnitude = 2 sum_i (1-cos(theta_i)).
second DS = D(S_t, M(S_(t-1),S_(t+1))).
```

The second magnitude is decomposed approximately into movement along the
endpoint geodesic and movement orthogonal to it, following Fukui et al. 2024.

Because that paper assumes equal time spacing, this report also uses a
separately named diagnostic. For observed time fraction
`alpha=h_left/(h_left+h_right)`, it compares `S_t` with the endpoint Grassmann
geodesic `Gamma(alpha)`. This does not replace or rename the paper's metric.

## 4. Verification

Eighteen formula/construction tests pass:

- paper mean subspace versus projector eigenspace;
- zero second order on an equal-speed geodesic;
- along component for nonuniform motion on the same geodesic;
- orthogonal component for off-geodesic motion;
- exact known-line Grassmann interpolation;
- zero time-aware deviation for irregular constant-speed sampling;
- band-image basis shape and orthonormality;
- common valid mask;
- acquisition gap reporting;
- spatial contributions summing to DS magnitude;
- spatial-grid coverage and local selectivity;
- temporal-context endpoint replication;
- zero DS/novelty for equal contexts;
- persistent local-change localization;
- gain/offset invariance after centered L2 preprocessing;
- snapshot/Gram temporal basis equivalence to direct SVD;
- phase-correlation recovery of a known global translation.

Command:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_temporal_subspace_dynamics -v
```

## 5. Controlled Nuisance Findings

Output:

`phase1/outputs/multisenge_temporal_injections_full_channel_20260619_193250/`

Rank-10 centered full-band span:

| Positive vs negative | Second/orthogonal result |
|---|---|
| local change vs global gain/offset | AUROC 1.000, AP 1.000 |
| local change vs one-pixel translation | AUROC 0.000 |
| local change vs all negatives | AUROC 0.875, AP 0.540 |

Interpretation:

- full-rank span is invariant to tested invertible per-band scaling;
- centering removes constant per-band offset;
- translation changes the spatial ambient coordinates and dominates the local
  change signal;
- therefore registration robustness is the immediate method problem.

## 6. MultiSenGE Sequence

Output:

`phase1/outputs/multisenge_temporal_timeaware_core5_20260619_194845/`

Data used:

- five `32TLT` patches;
- 23 dates per patch;
- 10 Sentinel-2 bands;
- 12,892 to 49,387 pixels valid at every date;
- only 15 of 105 triples have left/right gap ratio <= 1.5.

All-triple correlations:

| Quantities | Correlation |
|---|---:|
| paper second vs orthogonal | 0.973 |
| paper second vs time-aware deviation | 0.941 |
| paper second vs raw second RMSE | 0.709 |
| paper second vs NDVI curvature | 0.818 |
| paper second vs NBR curvature | 0.752 |
| time-aware deviation vs raw second RMSE | 0.675 |

On 15 balanced triples, paper second and time-aware deviation correlate at
0.988. The geometry currently follows spectral/index curvature strongly.
MultiSenGE's available land-cover reference is static, so no temporal detection
accuracy is claimed.

## 7. Four-Sequence IPOL Pressure Test

External source:

Dagobert et al., *Detection and Interpretation of Change in Registered
Satellite Image Time Series*, https://doi.org/10.5201/ipol.2022.416.

The exact IPOL C implementation was compiled and run on the pseudo-gamma Las
Vegas, Al Wakrah, Piraeus, and Beijing Airport sequences. The resulting
log-NFA maps are an external method-agreement target, not ground truth.

One-date rank-2 band-image DS did not replicate its favorable Vegas behavior:

| Sequence | DS pixel AP | Raw pixel AP | DS temporal Spearman | Raw temporal Spearman |
|---|---:|---:|---:|---:|
| Las Vegas | 0.043 | 0.036 | 0.692 | 0.672 |
| Al Wakrah | 0.052 | 0.055 | 0.276 | 0.370 |
| Piraeus | 0.091 | 0.346 | -0.053 | 0.437 |
| Beijing Airport | 0.019 | 0.044 | 0.311 | 0.715 |

The one-date detector hypothesis is rejected. Multiscale tiling did not rescue
it.

Second-order rank-2 whole-scene results are different:

| Method | Macro pixel AP | Macro pixel AUROC | Macro temporal Spearman |
|---|---:|---:|---:|
| raw time-interpolation residual | 0.1206 | 0.8282 | 0.5485 |
| paper second DS | 0.0895 | 0.8518 | 0.5934 |
| orthogonal component | 0.0951 | 0.8409 | 0.5826 |
| along component | 0.0512 | 0.7774 | 0.3065 |
| irregular-time geodesic deviation | 0.0994 | 0.8404 | 0.5898 |

The geometric quantities remain worse localizers, but second/time-aware
descriptors have slightly stronger aggregate temporal agreement. This supports
a sequence-level characterization hypothesis, not a pixel-detector claim.

## 8. Bidirectional Temporal Context

At temporal boundary `t`, the new factorized construction uses:

```text
X^-_(t,c), X^+_(t,c) in R^(N common pixels x V dates)
```

for each spectral band `c`. Canonical DS compares the backward and forward
context spans. A separate linear control projects the first post-date outside
the backward span and the last pre-date outside the forward span. This control
is not DS and is not the IPOL NNLS/NFA method.

The fixed sweep used `V={3,5}`, rank `{1,2}`, and per-band/joint factorization
on all four sequences:

| Best tested method/configuration | Macro AUROC | Macro AP | Macro best F1 |
|---|---:|---:|---:|
| raw adjacent RMS | 0.97 | 0.38 | 0.43 |
| projection novelty, `V=3`, rank 2, per-band | 0.98 | 0.37 | 0.42 |
| temporal-context DS, `V=3`, rank 1, per-band | 0.91 | 0.10 | 0.17 |

Projection novelty raises AUROC on every sequence and beats raw AP on Piraeus,
but loses AP on the other three. Temporal-context DS is not competitive.

Representative maps are in:

- `phase1/outputs/temporal_context_ds_vegas_matrix_20260620/boundary_maps/`;
- `phase1/outputs/temporal_context_ds_piraeus_matrix_20260620/boundary_maps/`.

Qualitatively, DS and projection maps suppress broad radiometric response but
still emphasize coastlines, roads, and registration-sensitive edges.

## 9. Controlled Persistent/Transient Test

Five real MultiSenGE backgrounds were modified with known local masks plus
global gain, offset, and one-pixel translation. Four configurations used
`V={3,5}` and rank `{1,2}`.

For `V=3`, rank 2, per-band:

| Method | Persistent localization AP | Persistent/transient response ratio | Translation/persistent response |
|---|---:|---:|---:|
| temporal-context DS | 0.733 | 0.97x by mean sum | 16.6x |
| projection novelty | 0.909 | 1.51x | 11.8x |
| raw adjacent RMS | 0.936 | 1.00x | 7.1x |

Gain and offset response is near zero for the centered/L2 geometric methods
but very large for raw difference. Cluster bootstrap over five patch
backgrounds gives projection novelty a normalized persistent-vs-transient
contrast of `0.190`, 95% interval `[0.170,0.212]`. Its localization AP gap from
raw is `-0.024`, interval `[-0.068,0.018]`.

Low-rank DS increases persistence contrast but loses localization. A one-pixel
translation still dominates the local event. This is a useful diagnostic
property, not real-label performance.

### 9.1 Registration Scale-Space Result

For a `32x32`, strength-0.25 persistent event, Gaussian support reduces the
one-pixel translation/local ratio of projection novelty from `26.67` natively
to `3.17` at `sigma=2` and `1.14` at `sigma=4`, while local AP remains `0.984`
at `sigma=4`. Global phase correlation reduces the integer-shift ratio below
`0.01` but lowers local AP to `0.877`.

This is scale-dependent. For a weak `16x16` event, `sigma=4` still leaves a
ratio of `32.8` and AP `0.764`. Cross-scale decay separates the tested global
translations from local persistent changes perfectly, but raw difference and
DS do so too. The result is a generic diagnostic lead, not DS novelty.

## 10. Commands

IPOL sequence:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-registered-sequence-dynamics --sequence-dir tmp/ipol416_vegas --output-dir phase1/outputs/ipol_vegas_temporal_timeaware_20260619_194845 --rank 0 --preprocessing centered --top-k 20
```

MultiSenGE core five:

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-multisenge-temporal-dynamics --manifest phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --rank 10 --preprocessing centered --patch-ids "32TLT_3855_0,32TLT_4369_257,32TLT_4883_514,32TLT_5397_1028,32TLT_5654_257" --output-dir phase1/outputs/multisenge_temporal_timeaware_core5_20260619_194845
```

Temporal-context sweep:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-temporal-context-ds --sequence-dir tmp/ipol416_vegas_pseudogamma --context-sizes 3,5 --ranks 1,2 --factorizations per_band,joint --reference-lognfa-dir tmp/ipol416_reference_run_pseudogamma/lognfa_dir --output-dir "phase1/outputs/temporal_context_ds_vegas_$tag" --figure-config 5:2:per_band --figure-count 4
```

Controlled temporal-context interventions:

```powershell
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'; .\.venv\Scripts\python.exe project_cli.py phase1-temporal-context-injections --manifest phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json --target-date 20200909 --context-size 3 --rank 2 --factorization per_band --repeats 4 --max-patches 5 --output-dir "phase1/outputs/temporal_context_injections_$tag"
```

## 11. Current Claim Boundary

Supported:

- the implementation matches the tested first/second/geodesic formulas;
- full-band centered spans have the tested radiometric invariance;
- the construction is highly misregistration-sensitive;
- irregular cadence must not be silently treated as equal spacing;
- spatial contribution maps are mathematically tied to DS magnitude but are
  not automatically accurate change maps.
- temporal projection novelty distinguishes the tested persistent injection
  from its transient counterpart and rejects tested gain/offset.
- second/time-aware descriptors show a small aggregate temporal-agreement lead
  over raw residuals against IPOL outputs.

Not supported:

- temporal DS detects real change better than established methods;
- orthogonal geodesic motion equals semantic change;
- MultiSenGE curves correspond to labeled events;
- the IPOL event maps are competitive with NFA;
- the method is robust to registration or seasonality.
- IPOL agreement is labeled ground-truth accuracy;
- current temporal-context DS is a competitive local detector.

## 12. Next Decision Gate

Obtain a manageable labeled multi-temporal evaluation slice and run a
registration-robustness curve before claiming a method. DynamicEarthNet is the
best identified labeled candidate but is approximately 524 GB in full, so use
a selective AOI protocol rather than a blind download. Add MOSUM and one
seasonal/trend baseline after the label path is fixed.

The surviving paper hypothesis separates sequence characterization from
localization:

```text
first/second/geodesic descriptors -> when/how the trajectory changes
cross-context projection novelty -> where persistent change occurs
registration-robust spatial support -> required method refinement
```

If labeled experiments do not support this split, report the negative result:
temporal subspace geometry provides specific invariances and persistence cues,
but not competitive changed-area localization under residual misregistration.
