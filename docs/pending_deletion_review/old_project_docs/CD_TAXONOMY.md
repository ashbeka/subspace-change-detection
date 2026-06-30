# Change-Detection Taxonomy For This Research

Updated 2026-06-21. This taxonomy is mechanism-based and dataset-agnostic. It places hyperspectral methods, traditional baselines, and subspace geometry without treating “subspace” as a standalone detector family.

## 1. Define The Task Before The Method

| Axis | Values | Why it changes the method |
|---|---|---|
| Temporal support | bitemporal; repeated pre/post stacks; dense series | Determines whether a pair test, two-sample statistic, or trajectory model is identifiable. |
| Sensor | optical MS; HSI; SAR/PolSAR; heterogeneous | Determines the noise/statistical object: reflectance vector, spectral mixture, complex covariance, or cross-modal representation. |
| Output | binary map; multiple transition classes; anomaly; change mechanism; recovery trajectory | Detection accuracy cannot validate characterization or attribution. |
| Supervision | unsupervised; weak/self-supervised; supervised | Governs available invariance learning and fair baselines. |
| Spatial unit | pixel; local window; superpixel/object; region; scene | Controls registration sensitivity, sample size, and spatial information loss. |
| Change scale | abrupt/strong; subtle/material; subpixel; distributional; gradual/recovery | Determines whether means, abundances, covariances, or time-series structure are needed. |
| Evaluation | labeled pixel/object; event date; mechanism label; known wavelength support | The metric must match the output claim. |

The current research-mining target is:

```text
bitemporal satellite HSI + local windows + unsupervised/interpretable
-> binary evidence plus mechanism and wavelength attribution
```

Dense temporal HSI is a later branch, not silently assumed.

## 2. Universal Pipeline

Every method can be decomposed as:

```text
preprocess/register
-> choose sample/support
-> construct representation
-> compare dates or fit the no-change relation
-> aggregate/regularize
-> decide and/or explain
```

For a subspace proposal, complete this construction card:

| Field | Required question |
|---|---|
| Source | Which paper/equation or reference implementation? |
| Sample unit | What is one column/example: pixel, patch vector, date, lagged vector, tensor mode? |
| Input object | Exact matrix/tensor and dimensions? Centered, normalized, or uncentered? |
| Count | How many subspaces and at what spatial/temporal scale? |
| Basis | Shape, rank rule, shrinkage/eigengap handling? |
| Comparison | Canonical angle, DS, projector distance, residual, SPD metric, GDS, CCA? |
| Preserved | Spatial coordinates, wavelength order, date order, variance scale, amplitude? |
| Lost | Mean, within-window arrangement, chronology, sign, eigenvalues? |
| Decision | Threshold, clustering, calibrated probability, attribution? |
| Null | Which simpler statistic contains the same information? |

## 3. Method Families By Statistical Object

### 3.1 Point/vector differences

| Family | Object and score | Strength | Failure/null pressure |
|---|---|---|---|
| Image differencing / raw L2 | `x2-x1`, norm or per-band map | Transparent, local | Radiometry and registration. |
| CVA / ICVA / CVAPS | vector magnitude and sometimes direction | Multiband direction, simple | No learned nuisance relation. |
| SAM | spectral angle | Isotropic amplitude invariance | Rank-1 DS is equivalent information. |
| PCA-diff / PCA-Kmeans | transform difference, cluster | Unsupervised compression | Components may mix change and nuisance. |
| Spectral change vectors / HCV | retain/quantize bandwise change patterns | Multiple change types | High-dimensional noise and threshold choices. |

### 3.2 Invariant transforms and predictive residuals

| Family | Representation/operator | Invariance lever | Placement |
|---|---|---|---|
| MAD / IR-MAD | CCA canonical variate differences; iterative no-change weights | Affine/decorrelated | Mandatory classical baseline; owns much of generic invariance. |
| SFA / ISFA | slow/invariant generalized eigenvectors; transformed residual | Temporal/radiometric relation | Mandatory baseline for slow-feature claims. |
| Chronochrome | regress date 2 from date 1 | Global linear prevalent change | ACD family. |
| Covariance equalization | align date statistics then residual | First/second-order distribution relation | Direct competitor to covariance/orientation methods. |
| TLSQ/CCA/hyperbolic ACD | stacked/joint quadratic residual | Coordinate-aware prevalent relation | Mature HSI anomalous-CD cell. |
| SiROC | predict pixel from distant spatial siblings | Spatial context | Mandatory spatial-context baseline. |

### 3.3 Distribution/two-sample change

| Family | Object | What it sees | Main risk |
|---|---|---|---|
| PWTT / mean tests | repeated pre/post samples | Mean shift with variance calibration | Needs replicated dates; distributional assumptions. |
| Variance/covariance tests | local moments, Box/Wishart/GLRT variants | Dispersion/correlation change | Sample scarcity and registration. |
| Shrinkage covariance distance | regularized `C1,C2` | Stable second-order difference | May subsume eigenspace proposals. |
| SPD metrics | log-Euclidean/AIRM/Bures-like distances | Geometry of full covariance | Mixes eigenvalue and orientation mechanisms unless decomposed. |
| MMD/energy distance | kernel/general two-sample discrepancy | Nonparametric distribution shift | Harder attribution, bandwidth/sample size. |
| PolSAR omnibus/Wilks/determinant tests | complex covariance/Wishart | Sensor-specific covariance change | Adjacent precedent, not directly optical HSI. |

### 3.4 Spectral mixture and material change

| Family | Object/output | Advantage | Limitation |
|---|---|---|---|
| Linear/sparse unmixing CD | endmembers and abundance differences | Subpixel, physically named direction/intensity | Library/endmember variability, computation. |
| Coupled/multitemporal unmixing | shared endmembers, temporal abundances | Uses temporal relation and spatial support | Model misspecification. |
| Material-subspace membership | canonical angles to class/material spans | Tolerates within-material variation | Project evidence: tolerance can also hide true change; centroid SAM wins. |

### 3.5 Sparse/band-attributed spectral change

| Family | Output | Existing territory |
|---|---|---|
| Sparse PCA on difference | sparse wavelength loadings/regions | HSI CD since at least 2011. |
| Unsupervised band selection | compact discriminative band subset | Binary/multiple HSI CD. |
| ECDBS/attention | learned band importance and spatial features | Supervised end-to-end CD. |
| S3CCA-style structured sparsity | smooth contiguous array weights | Established for partial matching, not directly CD. |
| DS/projector leverage | row energy of a difference subspace/projector | Basis-invariant direction attribution; proposed ingredient. |

### 3.6 Low-rank, sparse, and tensor methods

| Family | Preserved structure | CD status |
|---|---|---|
| Low-rank + sparse decomposition | global spectral-spatial low rank vs sparse residual | Established HSI CD/anomaly family. |
| Patch tensor CD | local spatial-spectral tensor | Established HSI CD. |
| Tucker/TV HSI restoration | spatial-spectral modes and noise structure | Preprocessing, not automatically CD. |
| Product Grassmann / n-mode GDS | separate factor subspaces | Classification/analysis; CD adaptation remains crowded. |

### 3.7 Deep and self-supervised HSI CD

| Family | Examples | Fair comparison principle |
|---|---|---|
| CNN/Siamese spectral-spatial | GETNET, 3-D/2-D hybrids | Same data split and preprocessing. |
| Difference + temporal fusion | MTSFNet and related dual branches | Compare to both raw difference and concatenation. |
| Self-supervised | STMNet, pseudo-mask/contrastive methods | No-label claim must include pretraining/pseudo-label cost. |
| Band/channel selection | ECDBS, StripeCD | Separate selection benefit from downstream network. |
| Frozen-feature geometry | SLS-style future route | Must compare same encoder with cosine/L2/linear head. |

## 4. Where Subspace Geometry Sits

Subspace geometry is an operator layer that can appear inside several families.

| Geometry | Mathematical object | Legitimate CD role | Occupied/failed role |
|---|---|---|---|
| MSM/canonical angles | relation between two set spans | set membership, local orientation | Not automatically better than SAM/Mahalanobis. |
| DS | difference basis from paired canonical directions | basis-level attribution | Rank-1 detection duplicates SAM. |
| GDS/KGDS | multi-class difference/discriminant span | supervised regime separation | Unsupervised CD requires noncircular classes. |
| KDS | nonlinear subspace relation | nonlinear structure | Memory/tuning; nonlinear CD already crowded. |
| Second-order DS | middle subspace vs endpoint midpoint | trajectory description | Project trajectory claim closed by vector second difference. |
| SSA signal subspace | Hankel low-rank dynamics | normal-change modeling | Natural S2 detector transfer failed. |
| SFA/SFS | slow directions/subspace | explicit temporal invariance | SFA-CD already established. |
| RTW | subspace of time-warped tuples | phase/warp invariance | Satellite/HSI data-gated; TWDTW null. |
| Product Grassmann | per-mode tensor subspaces | mode-specific characterization | Tensor HSI CD already established. |
| Centered covariance eigenspace | top variation directions | mean/scale-quotiented orientation mechanism | Full covariance/SPD may subsume it. |

## 5. Invariance Ladder

Higher is not automatically better; each rung discards information and needs a matching change model.

| Rung | Nuisance modeled/quotiented | Representative methods | Mandatory null |
|---|---|---|---|
| 0 | none | raw difference, CVA | raw L2/per-band map |
| 1 | isotropic amplitude | SAM, normalized spectra, rank-1 DS | SAM |
| 2 | affine/correlated radiometry | MAD/IR-MAD, covariance equalization | IR-MAD/CE |
| 3 | learned slow/prevalent relation | SFA/ISFA, chronochrome, ACD | SFA/ACD |
| 4 | spatial context | SiROC, contextual prediction | SiROC/local predictors |
| 5 | temporal phase/warp | TWDTW, RTW candidate | phase-aware harmonic, TWDTW |
| 6 | mean and isotropic dispersion | centered, trace-normalized eigenspace | shrinkage covariance/SPD decomposition |

The proposed top task occupies rung 6 only for the orientation component. It must report the removed mean and dispersion components separately so invariance does not become blindness.

## 6. Challenge Map

| Challenge | Best-established families | Possible geometry contribution | Status |
|---|---|---|---|
| Subpixel registration | contextual prediction, robust ACD, alignment | local scale/uncertainty only | geometry remains sensitive |
| Spectral redundancy | PCA/MNF, band selection, sparse methods | low-rank orientation + interval attribution | open but crowded |
| Label scarcity | classical/ACD, self-supervision | unsupervised descriptive output | plausible |
| Subtle/subpixel change | unmixing, ACD, deep spectral-spatial | mean-preserving orientation evidence | provisional |
| Spectral variability | coupled unmixing, ACD | factor mean/scale/orientation | provisional |
| Interpretability | unmixing, HCV, sparse PCA/band selection | DS-basis wavelength intervals | open only if more faithful/stable |
| Dense temporal change | CCDC/BFAST, SFA, harmonic models | RTW/SSA/product Grassmann | HSI data-gated |
| Damage/recovery | PWTT, SAR stacks, CCDC/deep methods | later characterization layer | no current HSI evidence |
| Computation | band selection, sketching, low rank | small Gram/eigenspace operations | feasible, must time |

## 7. Proposed Top-Task Placement

For a local HSI window `X_t in R^(B x n)`:

```text
mean component:        mu_t
centered covariance:  C_t = Xc_t Xc_t^T/(n-1) + epsilon I
dispersion component: eigenvalues / trace / log spectrum
orientation component: top-r eigenspace U_t after centering and trace quotient
orientation score:    principal-angle / projector / DS magnitude
attribution:           diag(P_DS), structured into contiguous wavelength intervals
```

This is located at the intersection of distributional CD, subspace geometry, and structured spectral attribution. It is not a new family. Its novelty claim is the explicit mechanism/output combination.

## 8. Claims Allowed And Forbidden

**Allowed if verified:**

- first tested combination of local mean/dispersion/orientation factorization and contiguous DS-basis wavelength attribution in bitemporal satellite HSI;
- a mechanism-specific diagnostic that identifies when orientation adds evidence;
- scale invariance of the orientation component by construction and test;
- negative evidence showing covariance/unmixing makes DS unnecessary.

**Forbidden without new evidence:**

- first subspace HSI change detector;
- first covariance/distributional remote-sensing CD;
- superior to IR-MAD/SFA/unmixing/deep methods;
- detects disaster damage or recovery;
- DS uniquely detects mean-preserving change;
- spatially aware because a window was used, when within-window arrangement is discarded.

## Honest Novelty Verdict

The taxonomy leaves no open generic “subspace CD” cell. Geometry is defensible only as a mechanism inside an explicitly scoped task. The strongest provisional cell is local moment-factorized HSI change characterization with wavelength attribution; RTW phase invariance is a later, data-gated cell.

## Next Falsifiable Step

Complete the top task's construction card and prove its orientation term against the full covariance/SPD decomposition; if the terms are algebraically identical and attribution is no better, remove DS from the contribution rather than renaming the same statistic.
