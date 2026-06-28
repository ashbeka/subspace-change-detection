# KB 03: Signal, Tensor, And Remote-Sensing Methods

## Signal Latent Subspace

Mahyub, Souza, Batalo, and Fukui, *Applied Acoustics* 225, 110181, 2024. `[P+V]`

**Object and math.** Intermediate feature vectors from multiple frozen audio/image DNNs become separate factor subspaces. Product Grassmann distance fuses the factors; a GDS projection can remove between-class commonality, producing discriminative SLS. Optional network fine-tuning is separate from the subspace head.

**Novelty and justification.** SLS replaces SSA's handcrafted oscillatory features with latent neural features while preserving small-sample/noise benefits. ESC-10, ESC-50, and an imbalanced UrbanSound variant compare MFCC/SSA, shallow classifiers, CNNs, and modern networks. The paper reports competitive rather than best overall performance and acknowledges GDS-dimension sensitivity. The visually inspected method page explicitly claims the first product-Grassmann application to signal-processing latent factors.

**Transfer boundary.** Frozen remote-sensing foundation features could become local subspaces, but deep-feature CD and subspace/attention modules already exist. A fair baseline is the same encoder's cosine/L2 feature difference; otherwise the encoder, not geometry, explains the gain. Latent features also weaken wavelength-level interpretability.

**Verdict.** Modern secondary direction, but not the best fit for an interpretable HSI task.

**Next falsifiable step.** Compare DS/canonical-angle heads to mean-pooled cosine and linear probes on identical frozen features before any end-to-end work.

## Adaptive Regularized Low-Rank Tensor Decomposition For HSI Denoising And Destriping

Li, Chu, Guan, He, and Shen. `[P+V]`

**Object and math.** LRTDAHL models observed HSI `Y` as clean signal, sparse/impulse noise, Gaussian noise, and a separately modeled stripe tensor. Tucker low-rank structure captures spatial-spectral correlations. Adaptive hyper-Laplacian spatial-spectral total variation models gradient sparsity; ADMM solves the decomposition. The inspected method page confirms explicit mode-wise HSI and stripe tensors rather than a flattened covariance.

**Novelty and justification.** The contribution is mixed-noise restoration, especially separating structured stripes while adapting the gradient-distribution prior. Denoising/destriping quality is compared with established low-rank/tensor/TV methods.

**Transfer boundary.** This paper proves tensors are natural for preserving HSI spatial-spectral modes, not that its residual is a temporal change map. Preprocessing separately by date can also create differential artifacts.

**Verdict.** Representation lesson and possible preprocessing audit; not a CD method.

**Next falsifiable step.** If used later, test paired/joint denoising against no denoising and measure whether it changes null-region false alarms.

## Spatial Context Awareness For Unsupervised Change Detection In Optical Satellite Images (SiROC)

Kondmann, Toker, Saha, Schölkopf, Leal-Taixé, and Zhu, *IEEE TGRS* 60, 2022. `[P]`

**Object and math.** Sibling regression predicts a pixel from distant spatial neighbors at one date, transfers the learned relation to another date, and scores prediction-residual change. Mutually exclusive neighborhoods form an ensemble; morphology promotes object-level changes; ensemble votes provide uncertainty.

**Novelty and justification.** The key claim is unsupervised spatial-context modeling rather than raw per-pixel differencing. Four Sentinel-2/PlanetScope datasets, including OSCD, compare conventional and deep unsupervised CD variants. The method emphasizes calibrated uncertainty and no training labels.

**Transfer boundary.** Any local HSI subspace method must face SiROC's lesson: preserving context may matter more than adding spectral geometry. SiROC is also a misregistration/context control even though it is not hyperspectral.

**Verdict.** Mandatory spatial-context baseline; it removes any novelty claim based merely on using neighborhoods.

**Next falsifiable step.** Test the proposed local distribution score on controlled one-pixel translations and compare SiROC-style contextual residuals.

## Open-Access Battle Damage Detection Via Pixel-Wise T-Test On Sentinel-1

Ballinger, *Remote Sensing of Environment* 331, 115025, 2025. `[P]`

**Object and math.** Repeated pre- and post-event Sentinel-1 amplitudes support a pixel-wise two-sample/Welch-style t statistic, followed by building-footprint aggregation. Temporal replication estimates variance instead of relying on one pre/post difference.

**Novelty and justification.** The paper frames a simple, transparent, open operational pipeline for conflict damage. Large building-footprint labels across Gaza and Ukraine, satellite baselines, coherence/single-image statistics, and learned competitors support the utility. Reported AUC is about 0.87 for the full approach; exact label counts differ across corpus/evaluation descriptions and should be cited carefully.

**Transfer boundary.** PWTT is SAR, not HSI, but it is the deep-motive null: if repeated dates are available, a simple per-pixel temporal statistic may beat elaborate geometry. It also exemplifies a defensible contribution based on accessibility, calibration, and operational validation rather than mathematical novelty.

**Verdict.** Essential comparison/framing pressure for disaster or conflict claims.

**Next falsifiable step.** Any damage-oriented extension must compare against a repeated-date mean/variance test and report calibration at the footprint/object level.

## Honest Novelty Verdict

The adjacent methods reinforce three constraints: learned features need same-encoder nulls, tensor structure is not itself CD novelty, and spatial/temporal context must beat simple transparent predictors and tests. None displaces the proposed local HSI orientation-and-attribution task, but all constrain its evaluation.

## Next Falsifiable Step

Include context, translation, paired-preprocessing, and repeated-date statistical controls in the experimental protocol before claiming a spectral-geometric advantage.
