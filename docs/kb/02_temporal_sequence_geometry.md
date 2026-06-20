# KB 02: Temporal And Sequence Geometry

These papers show how ordering, slowness, warping, or Hankel structure can make a sequence subspace nontrivial. Most are classification or matching papers; transfer to change detection is conditional.

## Time-Series Anomaly Detection Based On Difference Subspace Between Signal Subspaces

Kanai, Sogi, Maki, and Fukui, arXiv:2303.17802, 2023. `[P]`

**Object and math.** A scalar window becomes an SSA trajectory matrix; its leading left singular vectors form a signal subspace. A generalized DS excludes the past/present intersection. A learned normal-change subspace `D_N` is compared with the incoming DS using a direction index based on canonical relations. A magnitude index measures deviation of a log-product-of-cosines statistic from normal; the paper multiplies direction and magnitude.

**Novelty and justification.** The contribution replaces a conventional minimum canonical angle with the entire difference subspace and models normal change direction. Seven UCR anomaly series are compared with SSA angle scores, AR, GRU, and LSTM. Mean AUC is reported as 0.923, but performance is not uniformly best.

**Transfer boundary.** The real lever is the learned normal-change reference, not DS alone. The project reproduced a synthetic/artifical-splice benefit and seasonal robustness, but the method failed on natural Sentinel-2 abrupt, gradual, and phenology cases. It is evidence for “model the normal residual,” not a live top task.

**Verdict.** Faithful temporal-CD precedent with a failed natural-satellite transfer in this project.

**Next falsifiable step.** No further detector work until a natural event supplies repeated normal/change subspaces and a stronger null than harmonic deseasonalization.

## Particularities And Commonalities Of Singular Spectrum Analysis

SSA review, 2020 lineage. `[P]`

**Object and math.** SSA embeds a series into a Hankel/trajectory matrix, decomposes it by SVD, groups components, and reconstructs by diagonal averaging. Window length, rank/grouping, separability, forecasting assumptions, and Toeplitz versus basic SSA variants matter. Multivariate and 2-D extensions are established.

**Novelty and justification.** This is a methodological review, not a new detector. Its central warning is that SSA is a family whose behavior depends on embedding and grouping; “SSA subspace” is not fully specified without these choices.

**Transfer boundary.** A spectral-axis Hankel construction could make one HSI spectrum multidimensional, but it imposes a local stationarity/order model over wavelength. A temporal SSA construction needs dense multi-date HSI, which is currently scarce.

**Verdict.** Valid construction substrate, not evidence that SSA improves HSI CD.

**Next falsifiable step.** Any SSA proposal must compare raw spectra, first differences, and Hankel subspaces while sweeping window/rank and checking whether the score survives band resampling.

## Feature Sequence Representation Via Slow Feature Analysis

Kobayashi, action-classification paper. `[P]`

**Object and math.** PCA-SFA addresses the small-sample singularity of ordinary SFA, then uses the slowest projection vector as a fixed-dimensional descriptor of a frame-feature sequence.

**Novelty and justification.** The claim is efficient ordered-sequence representation under few frames. Evaluation is action classification against temporal pooling/sequence and learned-feature baselines.

**Transfer boundary.** SFA's slowness objective is meaningful for nuisance suppression, but SFA-CD already owns that transfer directly. PCA-SFA is therefore not a new HSI-CD direction.

**Verdict.** Useful small-sample implementation idea; novelty closed by SFA-CD.

**Next falsifiable step.** Use only as a computational baseline if dense HSI sequences become available.

## Slow Feature Subspace

Beleza, Shimomoto, Souza, and Fukui, *Machine Learning with Applications* 14, 100493, 2023. `[P+V]`

**Object and math.** Compute several slow SFA weight vectors, then apply SVD/PCA to their set to create a slow-feature subspace. Videos are compared through canonical angles within MSM/KMSM/CMSM/KCMSM. The inspected reversal example shows identical PCA frame spans but different SFS spans, which is the claimed order-sensitive edge.

**Novelty and justification.** It preserves the distribution of multiple slow weights instead of aggregating one descriptor. Five action datasets and raw, handcrafted, CNN, transformer, and fine-tuned baselines are used. On UCF101, frozen Swin-B SFS/KCMSM reaches 95.62% versus 99.60% for VideoMAE V2; the value proposition is low-data/no-fine-tuning, not SOTA. Added hyperparameters are acknowledged.

**Transfer boundary.** Ordered HSI time series could use SFS, but the current task is primarily bitemporal HSI. Dense satellite HSI labels do not support this as the immediate task.

**Verdict.** Strong sequence representation, data-misaligned with the immediate niche.

**Next falsifiable step.** Defer until a dense HSI series exists; first test time reversal and phase shift against harmonic/SFA baselines.

## Slow Feature Analysis For Change Detection In Multispectral Imagery

Wu, Du, and Zhang, *IEEE TGRS*, DOI 10.1109/TGRS.2013.2266673. `[P+V]`

**Object and math.** SFA solves `A W = B W Lambda`, where `A` is the covariance of bitemporal differences and `B` is signal covariance. Unchanged pixels define slow/invariant directions; transformed differences are scored by Euclidean, normalized, or iterative weighted distance. USFA, supervised SSFA, and iterative ISFA variants are given.

**Novelty and justification.** It transfers SFA's invariant representation to CD. Landsat ETM scenes compare against CVA, PCA, MAD, and IR-MAD using AUC/divergence and maps. The rendered output-band page confirms that multiple SFA components can describe different change evidence, while ISFA iteratively reweights likely unchanged pixels.

**Transfer boundary.** This is the mandatory baseline for any “slow/invariant subspace” claim. Project tests show SFA handles affine nuisance, but IR-MAD already matches that benefit.

**Verdict.** Occupies the invariant-transform CD cell; not a novelty source, but its component outputs motivate attribution.

**Next falsifiable step.** Compare any proposed factorized distribution score to ISFA/IR-MAD on the same nuisance regimes, not only raw CVA.

## Temporally Regularized CCA For Fingerspelled Word Spotting

Tanaka, Okazaki, Kato, Hino, and Fukui, FG paper. `[P]`

**Object and math.** CCA matches two feature sequences while a temporal regularizer penalizes second differences of canonical weights across one or several time lags. Kernel/orthogonal variants and multi-scale aggregation improve partial sequence spotting.

**Novelty and justification.** The target is a contiguous word embedded in continuous sign video. Baselines include ordinary CCA, aggregation variants, and kernel orthogonal subspace methods; temporally regularized variants substantially improve hit rate.

**Transfer boundary.** The method localizes a *common* temporal pattern. Change detection seeks discrepancy, so direct application has the wrong sign unless commonality is modeled first and the residual is scored. It is more relevant to temporal attribution than bitemporal HSI.

**Verdict.** Useful structured-matching principle, not a detector by substitution.

**Next falsifiable step.** Formulate and test an explicit common-component-plus-residual objective before claiming transfer.

## S3CCA: Smoothly Structured Sparse CCA For Partial Pattern Matching

Kobayashi, ICPR 2014. `[P+V]`

**Object and math.** CCA weights over array positions receive a Laplacian smoothness penalty and overlapping group-sparsity penalty. The rendered equation page confirms the objective combines the CCA quadratic form, `w^T L w`, and structured mixed norms; iterative reweighting solves the generalized eigenproblem. Output weights identify smooth contiguous matching regions.

**Novelty and justification.** The paper's novelty is simultaneous smoothness and structured sparsity for partial matching, compared with CCA and sparse/smooth alternatives.

**Transfer boundary.** Wavelength is an ordered axis, so the regularizer is a principled source for contiguous HSI band attribution. But ordinary S3CCA maximizes *commonality*. A change method must operate on DS/covariance residual evidence or invert the objective explicitly. Sparse PCA HSI-CD (2011), unsupervised band-selection HSI-CD (2017), and ECDBS (2024) already occupy generic band selection.

**Verdict.** Important ingredient for the top candidate's contiguous attribution, not a direct method transfer.

**Next falsifiable step.** Compare structured interval attribution against unstructured DS leverage, per-band standardized differences, sparse PCA loadings, and ECDBS-style band importance.

## Enhanced Grassmann Discriminant Analysis With Randomized Time Warping

Souza, Gatto, Xue, and Fukui, *Pattern Recognition* 97, 107028, 2020. `[P]`

**Object and math.** RTW samples ordered time-elastic tuples from each sequence, forms many time-elastic feature vectors, and fits a “hypo” subspace. eGDA projects class subspaces through GDS before Grassmann discriminant analysis, reducing overlap. Cambridge Gesture, KTH, and UCF Sports support the claim.

**Novelty and justification.** The method combines randomized warp-tolerant sequence representation with discriminative GDS projection. Comparisons include RTW+GDA, Hankel and other motion/action methods.

**Transfer boundary.** Phase/timing nuisance in phenology is a real structural fit, but RTW-CD must beat TWDTW and phase-aware harmonic models. Dense hyperspectral time series and labels are not currently available, so it is a high-novelty, high-data-risk secondary task.

**Verdict.** The clearest untested invariance mechanism, but not the immediate bitemporal HSI task.

**Next falsifiable step.** A controlled phase-shift versus cycle-shape-change gate against TWDTW and harmonic phase terms; no real-data program before that gate passes.

## Analysis Of Temporal Tensor Datasets On Product Grassmann Manifold

Batalo, Souza, Gatto, Sogi, and Fukui. `[P]`

**Object and math.** Each tensor mode is represented by a subspace; factor Grassmann distances combine on a product manifold. A Hankel-like temporal-mode matrix preserves ordering that plain mode unfolding loses. Outputs support visualization, clustering, and classification.

**Novelty and justification.** The contribution adds an explicitly temporal factor to product-Grassmann tensor analysis and demonstrates structure in temporal tensor datasets.

**Transfer boundary.** A spectral-spatial-temporal HSI cube fits the data type, but the paper is not CD and does not establish a change operator. Patch-tensor and deep spectral-spatial-temporal HSI-CD methods are close.

**Verdict.** Strong representation candidate, weak immediate novelty without a mode-specific falsifiable CD claim.

**Next falsifiable step.** Demonstrate that a product distance separates which *mode* changed more reliably than Tucker-factor residuals.

## Temporal-Stochastic Tensor Features For Action Recognition

Batalo, Souza, Gatto, Sogi, and Fukui, *Machine Learning with Applications* 10, 100407, 2022. `[P]`

**Object and math.** TS-PGM adds a temporal-stochastic factor by repeatedly sampling ordered frames, fitting a subspace to those samples, and combining it with ordinary tensor-mode Grassmann factors. Extensions add GDS projection.

**Novelty and justification.** Random sampling retains temporal variability without a fixed vector concatenation. Gesture/action/skeleton datasets compare product-Grassmann, tensor, and action-recognition methods.

**Transfer boundary.** Satellite acquisitions are sparse and irregular rather than long videos; stochastic tuple sampling may amplify missing-date bias. It is a later dense-series option.

**Verdict.** Not justified for current HSI data availability.

**Next falsifiable step.** Test missing-date and irregular-cadence sensitivity before any satellite claim.

## Grassmann Singular Spectrum Analysis For Bioacoustics Classification

Souza, Gatto, and Fukui. `[P]`

**Object and math.** SSA turns each sound into a signal subspace. Rather than assuming a class is the linear span of reference signals, GSSA treats examples as Grassmann points and uses subspace similarities/classification. The Anuran dataset shows an improvement over MSSA.

**Novelty and justification.** The improvement comes from representing within-class examples as multiple subspaces and comparing them geometrically, not from a new anomaly score.

**Transfer boundary.** It supports multiple local/reference subspaces instead of one global HSI background span. It does not solve change detection or specify nuisance handling.

**Verdict.** Evidence for local/multiple references, not a candidate task.

**Next falsifiable step.** If local reference libraries are used, compare nearest-reference Grassmann scoring to nearest-reference SAM/Mahalanobis.

## Honest Novelty Verdict

Temporal methods add value only when the construction truly preserves order or supplies an explicit invariance. The project has already rejected generic DS trajectory dynamics, and dense temporal HSI is a data bottleneck. S3CCA's structured sparsity survives as an attribution ingredient; RTW survives as a gated future invariance task.

## Next Falsifiable Step

For the immediate HSI task, use no temporal construction beyond bitemporal local sets. Reserve RTW/SSA/SFS for a separate phase-shift gate once a dense, independently labeled sequence exists.
