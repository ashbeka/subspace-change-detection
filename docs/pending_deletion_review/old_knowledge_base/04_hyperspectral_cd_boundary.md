# KB 04: Hyperspectral Temporal Change-Detection Boundary

This file scopes the exact niche and the prior work most capable of invalidating a novelty claim. “Temporal” includes bitemporal CD; dense time-series HSI is treated separately because public labeled data are scarce.

## Review Boundary

### Multitemporal HSI-CD review, 2019

Liu, Marinelli, Bruzzone, and Bovolo, [*A Review of Change Detection in Multitemporal Hyperspectral Images*](https://doi.org/10.1109/MGRS.2019.2898520), *IEEE Geoscience and Remote Sensing Magazine* 7(2), 140-158. `[M]`

Repository metadata/abstract states the core niche problem precisely: HSI can reveal subtle spectral variation unavailable to multispectral sensors, but automatic methods must manage the complexity of high-dimensional change information. The paper catalogs basic concepts, categories, applications, open issues, and representative experiments. Full text was closed in this session, so no detailed challenge is attributed to it without another source.

### Practical MS/HS CD review, 2019

Kwan, [*Methods and Challenges Using Multispectral and Hyperspectral Images for Practical Change Detection Applications*](https://doi.org/10.3390/info10110353), *Information* 10(11), 353. `[P]`

The full review identifies practical bottlenecks: subpixel registration, computation with hundreds of bands, spatial-resolution limits, lack of high-resolution HSI/training data, uncertain quality of pansharpening, synthetic-band validity, temporal fusion quality, and multimodal inconsistency. It describes chronochrome and covariance equalization, noting chronochrome benefits from accurate registration while covariance equalization is more robust. This directly pressures local-covariance novelty claims.

### Latest peer-reviewed HSI-CD survey, 2025

Lv, Zhang, Sun, Lei, Benediktsson, and Liu, [*Land Cover Change Detection with Hyperspectral Remote Sensing Images: A Survey*](https://doi.org/10.1016/j.inffus.2025.103257), *Information Fusion* 123, 103257. `[M]`

Verified as a 2025 peer-reviewed survey with 196 Crossref references. It was closed access and had no public abstract in the queried scholarly indexes, so it is used as the current bibliographic boundary, not as evidence for a specific open challenge. Its reference list and the current papers below show that deep spectral-spatial fusion, unmixing, self-supervision, band selection, and tensor methods are active.

**Review conclusion.** A novelty claim must survive at least five mature families: statistical/transform methods; anomalous change detection; unmixing/subpixel CD; spatial-spectral/tensor models; and supervised/self-supervised deep networks. High spectral dimension alone is not an opening.

## Closest Subspace And Covariance Methods

### Asymmetric background-subspace HSI CD

Wu, Du, and Zhang, [*A Subspace-Based Change Detection Method for Hyperspectral Images*](https://doi.org/10.1109/JSTARS.2013.2241396), 2013. `[M]`

The date-2 pixel is a target; the date-1 corresponding pixel plus spatial neighbors or undesired-class spectra forms a background subspace. Orthogonal-subspace projection residual identifies anomaly/change. Hyperion and HJ-1A experiments report lower false alarms than their baselines and show spatial information can reduce misregistration alarms.

**Novelty pressure.** “Subspace CD for HSI” is already false as a novelty claim. The proposed task differs only if it is symmetric pre/post local-distribution *factorization*, scale-quotiented orientation, and wavelength attribution rather than target-versus-background residual.

### Covariance equalization and quadratic ACD

- Schaum and Stocker, [*Hyperspectral change detection and supervised matched filtering based on covariance equalization*](https://doi.org/10.1117/12.544026), 2004.
- Theiler, [*Quantitative comparison of quadratic covariance-based anomalous change detectors*](https://doi.org/10.1364/AO.47.000F12), 2008.
- [*Improved covariance equalization for change detection in hyperspectral images*](https://doi.org/10.1117/12.2584038), 2021.
- [*Spatio-spectral anomalous change detection in hyperspectral imagery*](https://doi.org/10.1109/GLOBALSIP.2013.6737050), 2013.

`[M]` The ACD family fits or equalizes prevalent joint/cross-date covariance and flags deviations. Theiler's comparison explicitly treats quadratic covariance-based detectors. Local/spatial variants and improved covariance equalization demonstrate that covariance is not a naive straw baseline.

**Novelty pressure.** Detecting “distributional/covariance change” is established. A Grassmann orientation score is only justified if it gives a mechanism-specific, scale-invariant, stable attribution that full covariance/SPD or conditional-prediction methods do not.

### Multiview subspace learning for anomalous HSI change

Chang, Kopp, and Ghamisi, [*Sketched Multiview Subspace Learning for Hyperspectral Anomalous Change Detection*](https://doi.org/10.1109/TGRS.2022.3220814), 2022. `[M]`

SMSL sketches large data, learns shared/specific self-representation matrices across temporal views, and uses their difference for anomalous change. Benchmark and natural hyperspectral data are compared with ACD methods.

**Novelty pressure.** Modern “subspace learning for HSI anomalous CD” is occupied. The proposed work must distinguish Grassmann orientation and wavelength explanation from learned self-representation.

## Subpixel And Unmixing Competitors

### Collaborative coupled unmixing

[Collaborative Coupled Hyperspectral Unmixing Based Subpixel Change Detection](https://doi.org/10.1109/JSTARS.2021.3104164), 2021. `[M]`

The method jointly segments and extracts multitemporal endmembers, estimates abundances, and reports endmember-specific change direction and intensity on GF-5/ZY-1 wetland imagery, with Sentinel-2 auxiliary validation.

### Fast multitemporal unmixing

[Fast Unmixing and Change Detection in Multitemporal Hyperspectral Data](https://doi.org/10.1109/TCI.2021.3112118), 2021. `[M]`

It exploits temporal abundance correlation, handles endmember variability, detects abrupt abundance changes, and supplies theoretical success conditions against MESMA.

### Joint unmixing and information coguidance

[Multitemporal hyperspectral unmixing with information coguidance](https://doi.org/10.1109/TGRS.2020.3045799), 2021 lineage. `[M]`

This family couples abundance differences with spatial information and explicitly addresses spectral variability.

**Novelty pressure.** Subpixel/material interpretation is not an unoccupied benefit. Unmixing has stronger physical semantics than an unnamed eigenspace. The proposed task must either (a) work without a trustworthy endmember library and explain wavelength intervals, or (b) lose honestly to unmixing and become a diagnostic result.

## Band Selection And Spectral Attribution Competitors

### Sparse PCA for HSI CD

[Sparse principal component analysis in hyperspectral change detection](https://doi.org/10.1117/12.897434), 2011. `[M]`

Sparse PCA on the bitemporal difference uses only 15 of 126 HyMap loadings while preserving a similar change score and identifies three important wavelength regions. This is the closest classical spectral-attribution null.

### Unsupervised band selection

[Band selection for change detection from hyperspectral images](https://doi.org/10.1117/12.2263024), 2017. `[M]`

The paper quantitatively studies unsupervised band-selection effects on binary/multiple CD, discriminability, and number of changes using Hyperion data.

### End-to-end band-selection CD

[End-to-End Hyperspectral Image Change Detection Based on Band Selection](https://doi.org/10.1109/TGRS.2024.3382638), 2024. `[M]`

ECDBS learns band importance based on correlation and uses band-specific spatial-attention blocks on three public HSI-CD datasets.

### Spectral change vectors and hierarchical spectral analysis

- [Binary spectral change vectors for unsupervised multiple HSI CD](https://doi.org/10.1109/MULTI-TEMP.2017.8035239), 2017.
- [Unsupervised hierarchical spectral analysis for HSI CD](https://doi.org/10.1109/WHISPERS.2012.6874245), 2012.

These methods preserve bandwise/multiscale spectral-change patterns and classify multiple change types.

**Novelty pressure.** “Interpretable band selection” and even wavelength-region selection are not new. The proposed opening is narrower: contiguous intervals explaining the *orientation-only component after mean and total scale are removed*. Sparse PCA, per-band standardized mean/variance, group-sparse covariance differences, and ECDBS importance remain mandatory baselines.

## Tensor, Spatial-Spectral, And Deep Competitors

- [Patch tensor-based HSI CD](https://doi.org/10.1109/IGARSS47720.2021.9554630) explicitly targets lost spatial structure using local tensor decomposition/reconstruction.
- [S2HNet spectral-spatial subspace detection](https://doi.org/10.1109/ICAEEE62219.2024.10561803) inserts PCA reduction into a hybrid 3-D/2-D CNN. Its “subspace” is dimensionality reduction, not Grassmann comparison.
- [MTSFNet](https://doi.org/10.1109/TGRS.2024.3437244) fuses difference and temporal branches.
- [STMNet](https://doi.org/10.1109/TGRS.2024.3523541) uses single-temporal masks for self-supervised HSI CD.
- [StripeCD](https://doi.org/10.1109/TIP.2024.3438100) combines untrained branches, channel selection, multiscale spatial segmentation, and a stripe sparse model.
- GETNET and later spectral-spatial-temporal networks define the supervised accuracy pressure.

**Novelty pressure.** Spatial support, PCA subspaces, tensors, band selection, and deep fusion are all occupied. Geometry can only be claimed as a transparent comparison/attribution mechanism, not as the invention of spectral-spatial HSI CD.

## Adjacent Covariance Change: PolSAR

PolSAR has a mature covariance/Wishart change-testing literature, including determinant-ratio, omnibus, Wilks' lambda, Hotelling-Lawley, and multifrequency block-covariance tests. It is a different sensor/statistical model, but it invalidates any broad claim that covariance-manifold change testing in remote sensing is new.

## Traditional Baseline Roster

The [Awesome Remote Sensing Change Detection traditional-method list](https://github.com/wenhwu/awesome-remote-sensing-change-detection#traditional-methods) includes PWTT, SiROC, CCDC, SFA, CVAPS, PCA-Kmeans, IR-MAD, MAD, ICVA, and CVA. For bitemporal HSI, the fair minimum is:

```text
raw L2/CVA, SAM, PCA-diff/PCA-Kmeans, IR-MAD, SFA/ISFA,
Wu 2013 SCD, covariance-equalization/quadratic ACD,
shrinkage covariance + SPD distances, MMD/energy,
sparse PCA/band selection, spectral unmixing,
and one competitive deep HSI-CD model.
```

## Honest Novelty Verdict

Neither subspace HSI CD, covariance/distributional CD, subpixel interpretation, nor band selection is new. The only provisionally unoccupied combination found is a *three-way local change characterization* (mean, total dispersion, scale-quotiented eigenspace orientation) with basis-invariant contiguous wavelength attribution. This is a task/output novelty, not a detector-family novelty, and the claim remains provisional until full-text database searching confirms no exact predecessor.

## Next Falsifiable Step

Run a formal database query for the exact combination `hyperspectral change detection AND (Grassmann OR eigenspace orientation OR covariance orientation) AND (band attribution OR spectral interval)`, then read every returned full text and add any exact predecessor before coding.

