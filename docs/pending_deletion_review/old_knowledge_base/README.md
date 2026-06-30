# Research-Mining Knowledge Base

Created 2026-06-21. This knowledge base records what the supplied papers actually establish, how they justify their claims, and what transfers to hyperspectral temporal change detection. It is evidence for the taxonomy and research synthesis, not a list of methods to implement.

## Reading And Evidence Status

- `[P]`: supplied PDF read directly; equations, experiments, and conclusion checked.
- `[P+V]`: supplied PDF read directly and a key equation/figure page visually rendered and inspected.
- `[M]`: publisher/repository metadata and abstract checked; full text was not available in this session.
- `[E]`: project experiment/report evidence, not external literature.

The 21 supplied/core PDFs were extracted to a session-local text corpus. Key pages from second-order DS, S3CCA, SLS, n-mode GDS, SFA-CD, and the HSI tensor paper were also rendered and visually checked. The 2025 *Information Fusion* survey is closed access; its bibliographic identity was verified, but claims attributed to it are limited to metadata. The 2019 practical review was read in full from its open PDF. Search coverage used Crossref, OpenAlex, Semantic Scholar, arXiv, repository OAI metadata, and DOI/publisher records because the general web-search endpoint and in-app browser were unavailable.

## Files

- [`01_core_subspace_geometry.md`](01_core_subspace_geometry.md): DS/GDS/KDS/KGDS, second-order DS, generalized MSM, n-mode GDS, and the lab geometry overview.
- [`02_temporal_sequence_geometry.md`](02_temporal_sequence_geometry.md): SSA, signal-subspace anomaly detection, SFA/PCA-SFA/SFS, TRCCA/S3CCA, RTW, product-Grassmann, temporal-stochastic tensors, and GSSA.
- [`03_signal_tensor_and_remote_sensing.md`](03_signal_tensor_and_remote_sensing.md): SLS, HSI tensor denoising, SFA-CD, SiROC, and PWTT.
- [`04_hyperspectral_cd_boundary.md`](04_hyperspectral_cd_boundary.md): exact HSI-CD niche, reviews, closest covariance/subspace/unmixing/band-attribution methods, and traditional baselines.
- [`05_project_evidence_map.md`](05_project_evidence_map.md): advisor/researcher constraints and the experimental findings that rule candidate directions in or out.

## Cross-Paper Law

Subspace geometry is not a change-detection family by itself. A method is defined by:

```text
sample/set construction -> representation -> comparison operator -> decision/attribution
```

The strongest papers gain leverage from one of four sources:

1. an invariance that a null lacks, such as SFA's slow component or RTW's time-warp tolerance;
2. a genuinely multidimensional set, not a rank-1 spectrum;
3. preserved structure, such as tensor modes or ordered Hankel embeddings;
4. an output a scalar score cannot supply, such as a basis-level direction or contiguous interval attribution.

The project evidence eliminates the first-order rank-1, generic invariance, and trajectory-curvature stories. The remaining opening is therefore deliberately narrow: characterize local spectral-distribution change after separating mean, total dispersion, and orientation, and attribute the orientation component to contiguous wavelength intervals. Even that opening is provisional until it beats shrinkage-covariance/SPD, two-sample, unmixing, and per-band attribution controls.

## Honest Novelty Verdict

No supplied paper supports a broad claim that DS is a new or generally superior satellite change detector. The only defensible positive hypothesis produced by this KB is a specific HSI change-*characterization* task combining moment quotienting, local Grassmann orientation, and wavelength attribution; novelty remains provisional because covariance and unmixing literatures are close.

## Next Falsifiable Step

Before implementation, complete the closest-prior table for the proposed orientation-and-attribution task and pre-register the covariance/SPD, MMD/energy, unmixing, sparse-PCA, and per-band nulls that would make the geometric component unnecessary.

