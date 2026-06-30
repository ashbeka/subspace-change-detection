# References (seminar + project)

Grouped for the talk and the write-up. Lab/core methods first, then change-detection baselines, datasets, deep
backbones, related geometry work, and tools. arXiv IDs / links given where known. (Verify exact venue/year
against the PDF before a camera-ready paper — these are seminar-grade citations.)

## A. Subspace methods (the lab's lineage — cite these)
1. K. Fukui and A. Maki. **Difference Subspace and Its Generalization for Subspace-Based Methods.** IEEE TPAMI, 2015. (first-order DS, GDS, and the kernel DS / kernel trick.) https://ieeexplore.ieee.org/document/7053916
2. K. Fukui, E. M. Valois, M. de Souza, T. Kobayashi. **Second-order Difference Subspace.** 2024. arXiv:2409.08563. (1st/2nd-order DS, geodesic on the Grassmannian.) https://arxiv.org/abs/2409.08563
3. T. Kanai, N. Sogi, A. Maki, K. Fukui. **Time-series Anomaly Detection based on Difference Subspace between Signal Subspaces.** 2023. arXiv:2303.17802. (SSA signal subspaces + DS + learned non-anomalous reference.) https://arxiv.org/abs/2303.17802
4. T. Kobayashi. **S3CCA: Smoothly Structured Sparse CCA for Partial Pattern Matching.** ICPR, 2014. https://staff.aist.go.jp/takumi.kobayashi/publication/2014/ICPR2014.pdf
5. K. Fukui et al. **Spotting Fingerspelled Words from Sign Language Video by Temporally Regularized Canonical Component Analysis.** (temporally-regularized CCA.) 2016.
6. T. Kobayashi. **Slow Feature Analysis** formulation. BMVC, 2017. https://staff.aist.go.jp/takumi.kobayashi/publication/2017/BMVC2017.pdf
7. Hiraoka et al. **Randomized Time Warping (RTW).** ICIP workshop "Learning Beyond Deep Learning (LBDL)", 2025. (temporal alignment via subspaces.)
8. K. Fukui, O. Yamaguchi, et al. **Mutual Subspace Method (MSM/CMSM/KMSM)** lineage — face/image-set recognition via canonical angles between subspaces.

## B. Change-detection methods & classical baselines
9. R. C. Daudt, B. Le Saux, A. Boulch. **Fully Convolutional Siamese Networks for Change Detection (FC-EF / FC-Siam-conc / FC-Siam-diff).** ICIP, 2018. https://rcdaudt.github.io/files/2018icip-fully-convolutional.pdf
10. A. A. Nielsen. **The Regularized Iteratively Reweighted MAD Method for Change Detection (IR-MAD).** IEEE TIP, 2007. (MAD = CCA between two dates.)
11. T. Celik. **Unsupervised Change Detection in Satellite Images Using PCA and k-means Clustering.** IEEE GRSL, 2009. (PCA-diff baseline.)
12. F. Bovolo and L. Bruzzone. **A Theoretical Framework for Unsupervised Change Detection Based on Change Vector Analysis in the Polar Domain (polar-CVA).** IEEE TGRS, 2007. (change magnitude + direction; multiple change types.)
13. L. Kondmann et al. **SiROC: Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images.** IEEE TGRS, 2021/2022. (and **SemiSiROC**, 2023 — semi-supervised with an unsupervised teacher.)
14. C. Han et al. **Change Guiding Network (CGNet): Incorporating Change Prior to Guide Change Detection.** 2024. arXiv:2404.09179. https://arxiv.org/abs/2404.09179
15. L. T. Luppino et al. **Deep Image Translation with an Affinity-Based Change Prior for Unsupervised Multimodal Change Detection.** 2020. arXiv:2001.04271. https://arxiv.org/abs/2001.04271
16. B. Du et al. **Deep Slow Feature Analysis (DSFA) for Change Detection in Multispectral Imagery.** IEEE TGRS, 2019.
17. Y. Li et al. **DRMAT: multivariate breakpoint detection in remote-sensing time series.** Remote Sensing of Environment, 2024. https://github.com/YangLiOSU/DRMAT
18. J. Verbesselt et al. **BFAST**; Z. Zhu and C. Woodcock. **CCDC**; R. Kennedy et al. **LandTrendr** — time-series trend/seasonal breakpoint methods (deseasonalization baselines).
19. **A Survey of Sample-Efficient Deep Learning for Change Detection in Remote Sensing.** 2025. arXiv:2502.02835. https://arxiv.org/abs/2502.02835
20. W. G. C. Bandara, V. M. Patel. **Deep Metric Learning for Unsupervised Remote Sensing Change Detection.** WACV, 2025.
21. V. Růžička et al. **RaVÆn: unsupervised change detection of extreme events using ML on-board satellites.** Scientific Reports, 2022. arXiv:2111.02995.

## C. Datasets
22. R. C. Daudt, B. Le Saux, A. Boulch, Y. Gousseau. **Urban Change Detection for Multispectral Earth Observation Using CNNs — the OSCD (Onera Satellite Change Detection) dataset.** IGARSS, 2018. https://rcdaudt.github.io/oscd/
23. R. Gupta et al. **xBD: A Dataset for Assessing Building Damage from Satellite Imagery.** CVPR Workshops, 2019. arXiv:1911.09296. https://arxiv.org/abs/1911.09296
24. Dietrich / Hafner et al. **The Potential of Copernicus Satellites for Disaster Response: Retrieving Building Damage from Sentinel-1 and Sentinel-2 (xBD-S12).** ISPRS Congress, 2026. arXiv:2511.05461. Data: Zenodo 18960454; HF prs-eth/xbd-s12; code https://github.com/prs-eth/xbd-s12
25. R. Wenger et al. **MultiSenGE: a multimodal, multitemporal Sentinel-2 benchmark (Grand-Est, France).** 2022/2023. https://multisenge.github.io/
26. **Sentinel-2 / Copernicus programme**, European Space Agency / EU. https://www.copernicus.eu/
27. **Benton, Hermiston, Salinas, Bay Area, Santa Barbara** hyperspectral change/scene datasets (HSI-CD community benchmarks).
28. E. Dagobert et al. **Multi-date Sentinel-2 change-detection sequences (IPOL).** IPOL, 2022. (the al_wakrah / beijing / piraeus sequences.)
29. M. Rußwurm et al. **BreizhCrops** (Sentinel-2 crop time series); **SpaceNet-7** multi-temporal urban dataset.

## D. Deep-learning backbones / semantic change
30. O. Ronneberger, P. Fischer, T. Brox. **U-Net: Convolutional Networks for Biomedical Image Segmentation.** MICCAI, 2015.
31. K. He, X. Zhang, S. Ren, J. Sun. **Deep Residual Learning (ResNet).** CVPR, 2016.
32. K. Yang et al. **SECOND: a semantic change detection benchmark** (from-to land-cover transitions); **Landsat-SCD**.
33. **MC-DiSNet (DINOv3 backbone, Gaza-Change + SECOND + Landsat-SCD).** 2025. arXiv:2511.19035. (deep multi-class damage/semantic change — no geometry; positioning reference.)
34. **MS4D-Net: Multitask Semi-Supervised Semantic Segmentation for Building Damage Assessment.** (last year's future-work reference.)

## E. Where geometry genuinely beats scalars (related / positioning)
35. A. Barachant et al. **Riemannian geometry for EEG/BCI** (MDM, tangent-space classification on SPD covariance manifolds).
36. PolSAR distance survey (AIRM / Log-Euclidean / Stein geodesics on HPD coherency matrices). MDPI Remote Sensing 14(22):5873, 2022.
37. J. Theiler et al. **Chronochrome / anomalous change detection (ACD)** — covariance/whitening detectors that suppress pervasive change.
38. **SMSL: Sketched Multi-view Subspace Learning** for change detection. 2022. arXiv:2210.04271.
39. **Singular Spectrum Transformation (SST)** change-point detection; **mSSA-CUSUM** (multivariate SSA change-point). NeurIPS, 2021.
40. **Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Networks** (incl. GrNet lineage: Huang & Van Gool, "A Riemannian Network for SPD/Grassmann learning"). 2025. arXiv:2511.08628.

## F. Tools / software
41. scikit-learn (PCA, CCA, metrics); PyTorch (FC-EF U-Net); rasterio (GeoTIFF I/O); Google Earth Engine (Sentinel-2 access); awesome-remote-sensing-change-detection (method/dataset index, https://github.com/wenhwu/awesome-remote-sensing-change-detection).

---
*Tip:* for the deck use the ~12 essentials (A1–A3, B9–B12, C22–C24, D30–D31); the rest belong in the paper.
