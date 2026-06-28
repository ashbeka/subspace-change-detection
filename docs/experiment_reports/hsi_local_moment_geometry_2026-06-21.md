# Local Hyperspectral Moment Geometry And DS Projection

## 1. Research Question

Can a local hyperspectral change representation separate changes in mean,
covariance scale, eigenspectrum shape, and leading-eigenspace orientation, and
can its Difference Subspace (DS) projection or band attribution add useful
changed-area evidence beyond simpler hyperspectral change detectors?

This is a narrower question than "does subspace geometry solve hyperspectral
change detection?" It tests one explicit interpretation/attribution mechanism
under strong classical and distributional controls.

## 2. Prior-Art Boundary

Subspace-based hyperspectral change detection is not new. The primary prior is
Wu, Du, and Zhang, *A Subspace-Based Change Detection Method for Hyperspectral
Images* (JSTARS 2013, DOI `10.1109/JSTARS.2013.2241396`). The benchmark also
includes classical CVA, PCA difference, IR-MAD, chronochrome, and covariance
equalization. Therefore the possible contribution was limited to the specific
factorized local representation and its basis-invariant band attribution.

## 3. Construction Card

For a local `w x w` window, one pixel is one `B`-dimensional hyperspectral
sample. Each date therefore supplies a matrix

```text
X_t in R^((w*w) x B).
```

After centering, exact dual PCA produces a rank-`r` basis
`U_t in R^(B x r)`. Principal angles are obtained from the singular values of
`U_1^T U_2`.

The reported factors are:

```text
mean                 = ||mu_2 - mu_1||_2
scale                = |log(tr(C_2) / tr(C_1))|
eigenspectrum shape  = Hellinger(normalized eigenvalue profiles)
orientation          = sqrt(sum_i(1 - cos(theta_i)^2)) / sqrt(r)
first-order DS       = mean_i 2(1 - cos(theta_i))
DS projection        = ||D^T (x_2 - x_1)||_2
projection ratio     = ||D^T delta||_2 / ||delta||_2
```

`D` is the canonical principal-vector difference basis. The per-band
orientation attribution is the row energy of `P_1 - P_2`, where
`P_t = U_t U_t^T`. It is invariant to basis signs and within-subspace rotations,
and its sum equals `||P_1-P_2||_F^2`.

Spatial layout is preserved only through local window support. Pixel ordering
inside a window is still discarded.

## 4. Data And Protocol

| Dataset | Retained bands | Spatial size | Changed / valid | Role |
|---|---:|---:|---:|---|
| Benton | 159 | 225 x 180 | 9,921 / 40,500 | development/configuration selection |
| Hermiston | 198 | 390 x 200 | 9,986 / 78,000 | held-out transfer |
| Farmland | 154 | 420 x 140 | 40,417 / 58,800 | held-out transfer and polarity stress |
| Shenzhen | 149 | 231 x 209 | 426 / 7,473 | held-out sparse-change transfer |

Four configurations were evaluated: robust joint normalization with
`(w,stride,r)=(5,3,3),(9,5,3),(13,7,5)`, and per-date standardization with
`(9,5,5)`. Benton selected one configuration per hypothesis. Hermiston,
Farmland, and Shenzhen labels were not used for that selection.

Controls included raw L2/CVA, spectral angle, PCA difference, IR-MAD,
chronochrome, covariance equalization, top-variance bands, NMF abundance-change
proxy, random-Fourier-feature MMD, normalized covariance distances, shrinkage
SPD distance, local mean, and local raw magnitude.

Metrics were AUROC, average precision (AP), best F1/IoU, Otsu F1/IoU, runtime,
qualitative maps, and 2,000-repeat spatial block-bootstrap AP differences.

## 5. Mechanism Verification

Five controlled tests pass:

- a pure mean shift changes the mean factor without creating orientation;
- isotropic scaling changes scale without creating orientation;
- matched-eigenvalue rotation is detected by orientation;
- fixed-basis eigenvalue change affects shape but not orientation;
- projector attribution is basis invariant and conserves projector distance.

The local PCA was changed from randomized SVD to deterministic exact dual PCA.
A quantized Hermiston band also exposed an unstable robust-IQR scale. Falling
back to standard deviation when robust scale is below 20% of standard scale
raised alternate-seed map Spearman agreement to at least `0.9994` and mean
band-attribution agreement to at least `0.99965`.

## 6. Held-Out Results

The frozen Benton-selected hypotheses lost to the strongest valid control on
every held-out dataset.

| Holdout | Strongest control AP | Orientation AP | DS projection AP | Factor fusion AP |
|---|---:|---:|---:|---:|
| Hermiston | covariance equalization 0.3778 | 0.1279 | 0.2217 | 0.1383 |
| Farmland | chronochrome 0.6449 | 0.5743 | 0.5036 | 0.4887 |
| Shenzhen | spectral angle 0.3056 | 0.0942 | 0.0543 | 0.0418 |

Spatial block-bootstrap AP differences versus the strongest control were
strictly negative:

| Holdout | Hypothesis | Mean AP delta | 95% interval |
|---|---|---:|---:|
| Hermiston | orientation | -0.2502 | [-0.3179, -0.1867] |
| Hermiston | DS projection | -0.1547 | [-0.2123, -0.1018] |
| Farmland | orientation | -0.0721 | [-0.1048, -0.0401] |
| Farmland | DS projection | -0.1411 | [-0.1713, -0.1125] |
| Shenzhen | orientation | -0.2053 | [-0.3144, -0.0738] |
| Shenzhen | DS projection | -0.2501 | [-0.3486, -0.1475] |

Benton itself is dominated by direct methods: PCA difference AP is `0.9695`,
local mean AP is `0.9593`, and the selected local DS projection AP is `0.9037`.

## 7. Farmland Polarity Killer Control

The best dataset-specific DS projection ratio initially looked positive:
AUROC `0.8344`, AP `0.9212`. The dataset's source code confirms the evaluated
label convention (`2=changed`, `1=unchanged`), but conventional raw distance is
strongly reversed on this scene.

Simply reversing raw L2 gives AUROC `0.9318`, AP `0.9652`. A center-sampled
local inverse-raw control gives AP `0.9602`. Even after choosing the DS-ratio
configuration on Farmland labels, its AP difference from inverse raw L2 is
`-0.0442`, 95% block-bootstrap interval `[-0.0596,-0.0310]`.

Decision: the apparent Farmland result is a score-polarity/denominator effect,
not evidence that the DS numerator adds information. Inverse controls are
diagnostics only; an unsupervised method cannot use labels to choose polarity.

## 8. Visual Findings

- Benton: raw/PCA/local-mean maps cleanly recover circular field changes; DS
  projection detects many of them but adds blur and lower ranking accuracy.
- Hermiston: covariance equalization and RFF-MMD localize field changes more
  clearly; orientation and factor fusion are diffuse.
- Farmland: inverse raw/PCA maps explain the high ratio result; DS ratio does
  not improve the ranking.
- Shenzhen: spectral angle is the strongest control; the local geometric maps
  emphasize broad spatial regions rather than the sparse labeled changes.

The source figures are in:

```text
phase1/outputs/hsi_moment_geometry_final_stable_20260621_185316/
```

## 9. Verdict

The broad positive hypothesis is rejected for this construction. Local
factorization is mathematically interpretable, but leading-eigenspace
orientation, canonical DS projection, projection ratios, and factor fusion do
not add held-out detection evidence beyond simpler controls.

The experiment contributes three durable findings:

1. factorized moment mechanisms can be verified independently before real-data
   claims;
2. local subspace attribution requires deterministic solvers and stable band
   normalization;
3. projection ratios must be tested against inverse-denominator controls,
   because a high score can be a polarity artifact.

Do not present this as a successful new HSI detector. It belongs in the larger
cross-method diagnostic and construction-verification evidence base. A future
positive attribution study would require calibrated wavelengths and known
material transitions, then must beat per-band differences, correlation/SPD,
MMD/energy, and unmixing attribution.

## 10. Reproduction

```powershell
.\.venv\Scripts\python.exe project_cli.py phase1-hsi-moment-geometry --datasets "benton,hermiston,farmland,shenzhen" --configs "joint_robust_zscore:5:3:3,joint_robust_zscore:9:5:3,joint_robust_zscore:13:7:5,per_date_zscore:9:5:5" --bootstrap 2000 --output-dir phase1/outputs/hsi_moment_geometry_final_stable_20260621_185316
```
