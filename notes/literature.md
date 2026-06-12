# Literature Notes

## Table Of Contents

- [Must-Cite Core Sources](#must-cite-core-sources)
- [Subspace And Kernel References](#subspace-and-kernel-references)
- [Reference Code](#reference-code)
- [Dataset Context](#dataset-context)
- [Bookmark Triage Rules](#bookmark-triage-rules)
- [Literature Leads](#literature-leads)
- [Reference Leads](#reference-leads)

This file is the paper and reference-code matrix. Keep entries short and useful.

## Must-Cite Core Sources

| Source | Why it matters | Project use |
|---|---|---|
| Fukui and Maki, TPAMI 2015, Difference Subspace and Generalization | Original DS/GDS and KDS/KGDS reference. | Must cite for DS, GDS, KDS, KGDS. Do not claim DS is invented here. |
| OSCD / Daudt et al. 2018 | Defines the Sentinel-2 OSCD binary change benchmark. | Main active dataset context. |
| FC-Siamese / Daudt et al. 2018 | Siamese CNN baseline for change detection. | Important Phase 2 baseline context. |
| Celik 2009 PCA-kmeans | Classical unsupervised satellite change detection. | Baseline prior, not novelty. |
| Nielsen 2007 IR-MAD | Strong classical multi/hyperspectral change detection. | Baseline prior, not novelty. |
| Close et al. 2021 Sentinel-2 LULUCF change detection | Shows Sentinel-2 change analysis can be affected by rare true change and radiometric/seasonal variability. | Supports the pseudo-change and seasonality-risk discussion. |
| Metric-CD, WACV 2025 | Modern unsupervised deep metric change detection. | Raises comparison pressure for unsupervised CD claims. |
| xBD / Gupta et al. 2019 | Building damage dataset. | Future damage extension only. |
| xBD-S12 | Sentinel-1/Sentinel-2 disaster damage dataset. | Future extension only. |
| ChangeOS | Object-based semantic change/damage framework. | Future damage comparison if xBD work resumes. |
| Remote-sensing change-detection surveys | Needed to answer Sensei's request to understand the broader method landscape. | Read before novelty claims; use to place DS among classical and deep CD methods. |
| Fukui et al. second-order DS / time-series DS | Supports multi-date subspace progression/recovery ideas. | Future MultiSenGE/GDS/second-order track, not current OSCD evidence. |
| Hiraoka et al., Attention Mechanism in Randomized Time Warping, arXiv:2508.16366 | Concrete RTW lead from Sensei; accepted to the ICIP Learning Beyond Deep Learning workshop according to archived advisor notes. | Future temporal/subspace literature, not current implementation evidence. |
| Mahyub et al. 2024, Signal Latent Subspace | Uses DNN latent features as subspaces, combines factor subspaces with product Grassmann manifold, and applies GDS for class separability. | Future deep-feature subspace bridge only; not remote-sensing evidence. |
| SSA/SFA/RTW-related Sensei leads | Alternative temporal/subspace dynamics suggested by Sensei. | Future temporal literature, not current implementation. |
| Fukui 2024, Geometry of subspace set and its application to machine learning | Lab overview of subspace method, MSM, canonical angles, DS/GDS, GDS projection, kernel variants, and Grassmann views. | Use as local conceptual guide for explaining where DS/GDS/KDS fit; cite underlying peer-reviewed papers for formal claims. |
| Green Learning / PixelHop / PixelHop++ | Senpai-suggested spatial-feature route related to successive subspace learning. | Future feature-construction route for preserving local structure before DS/KDS/SSC. |
| Wavelet transform / image compression resources | Senpai-suggested intuition for multiscale decomposition and information-preserving representations. | Future explanatory and feature-engineering lead; not current evidence. |
| Harmonized Sentinel-2 L2A | Sensei-supported candidate for time-sequential satellite experiments. | Dataset feasibility audit before first/second DS, GDS/KGDS, or geodesic claims. |

## Subspace And Kernel References

Fukui and Maki TPAMI 2015:

- Establishes DS/GDS and nonlinear KDS/KGDS.
- Sensei's Venus files are tied to this style of image-set subspace experiment.
- The current OSCD adaptation is different because samples are pixel-level 13-band vectors, not whole-image views.

S3CCA:

- Smoothly Structured Sparse CCA for partial pattern matching.
- Relevant because it preserves structure over sample indices.
- Possible future method if global pixel DS loses too much spatial structure.

Temporally Regularized CCA / KOTRCCA:

- Relevant for temporal sequences.
- More relevant to MultiSenGE than two-date OSCD.
- Also relevant to Harmonized Sentinel-2 sequence experiments if date spacing and frame count are adequate.

KCCA / Kernel CCA:

- Appears in the archived reference-code leads.
- Keep it as a future structured/kernel matching route, not as current OSCD evidence.
- Do not let KPCA/KCCA/S3CCA/TRCCA become the thesis core unless a specific experiment and comparison justify that pivot.

Green Learning / PixelHop / wavelets:

- Apple Notes and senpai feedback suggest Green Learning, PixelHop/PixelHop++, wavelet transforms, and image-compression intuition as spatial-preservation leads.
- Their possible value is not "another method to add" but a way to build local/multiscale features before subspace comparison.
- Treat them as literature to read only if the spatial DS audit shows raw global pixel subspaces are insufficient.

Signal Latent Subspace:

- Mahyub, Souza, Batalo, and Fukui, "Signal latent subspace: A new representation for environmental sound classification," Applied Acoustics 225, 110181, 2024.
- DOI: `10.1016/j.apacoust.2024.110181`.
- ScienceDirect PII: `S0003682X24003323`.
- The paper proposes subspaces from neural-network latent features, combines multiple latent-feature subspaces using product Grassmann manifold, and uses GDS projection to improve between-class discrimination.
- Use it as a method analogy for possible satellite experiments using pretrained/fine-tuned remote-sensing feature extractors, not as evidence that OSCD or greenhouse mapping is solved.

Temporal and image-set subspace ideas:

- Suryanto, Xue, and Fukui's Randomized Time Warping turns one ordered sequence into many randomly sampled time-elastic feature vectors, preserving temporal order but allowing speed variation; those vectors form a sequence-hypothesis subspace compared by canonical angles. Use this as a future MultiSenGE idea for comparing ordered date sequences, not two-date OSCD.
- Hiraoka and Fukui's Deep RTW keeps the RTW idea but replaces raw time-elastic vectors with CNN features from 2D/3D networks. Use this as a bridge toward remote-sensing encoder features plus temporal subspace matching.
- Kobayashi's PCA-SFA uses slow feature analysis to describe frame-feature sequences while handling small sample size through PCA subspaces. Use it as a candidate way to distinguish slowly varying seasonal/background components from abrupt change when a dataset has enough dates.
- Beleza et al.'s Slow Feature Subspace builds a subspace from the slowest SFA weight vectors, explicitly targeting the temporal-information loss of ordinary PCA video subspaces. Use it as a warning that unordered PCA over time can erase the temporal structure Sensei is worried about.
- Batalo et al.'s temporal tensor/Product Grassmann work represents tensor modes as subspaces and adds Hankel-like temporal embedding, then uses geodesic distances for visualization and clustering. Use it as a future route for spatial-spectral-temporal satellite tensors, especially when labels are absent.
- Souza et al.'s Grassmannian Learning Mutual Subspace Method embeds a learnable subspace-matching layer on top of CNN features and learns dictionary subspaces on the Grassmann manifold. Use it as a possible neural/subspace hybrid if hand-designed DS priors are too weak.
- Yamaguchi and Fukui's Multiple Pseudo-Whitened Mutual Subspace Method uses multiple pseudo-whitening transformations and canonical-angle similarities to improve robustness and discrimination in image-set classification. Use it as a discriminative-subspace reference, not as a direct change-map method.
- The superposed shape-subspace MVA hand-action paper stacks multiple skeleton frames into one high-dimensional subspace and selects informative frames for efficient CPU classification. Use it as an analogy for key-date or patch-stack selection in satellite sequences.
- The MVA human-motion shape-subspace paper uses DS between shape subspaces to measure whole-motion variation, landmark contribution, and coordination between body parts. Use it as an analogy for future attribution: which bands, patches, or regions contribute to a change subspace and how different regions move together.

## Reference Code

Bundled code in this repo:

- `references/reference_code/DS/utils.py`: projector-eigen style DS; close to repaired `eig` behavior.
- `references/reference_code/MagTool-main/.../magnitude.py`: subspace magnitude and difference-space style utilities.
- `references/reference_code/Subspace Toolbox/...`: MATLAB PCA, KPCA, CCA, Kernel CCA utilities.

Specific archived crosswalk:

- `references/reference_code/DS/utils.py`
  - `gen_shape_subspace`: builds a subspace from shape/motion data.
  - `gen_shape_difference_subspace`: projector-eigen style DS.
- `references/reference_code/MagTool-main/MagTool-main/magnitude.py`
  - `calcDiffSubspace`: projector/eigen difference-space style reference.
  - `calcKarcherSubspace`: mean/Karcher subspace reference.
  - `calcMagnitude`: canonical-cosine based subspace magnitude.
- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtPCA.m`
  - PCA where columns are samples.
- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtKernelPCA.m`
  - Kernel PCA reference.
- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtCCA.m`
  - CCA reference.
- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtKernelCCA.m`
  - Kernel CCA reference.
- `references/reference_code/Subspace Toolbox/cvtToolBox/SubspaceMethod/cvtBasisVector.m`
  - Builds basis vectors from data arrays.

External repos:

- `https://github.com/ComputerVisionLaboratory/SubspaceMethodsToolBox`
- `https://github.com/ComputerVisionLaboratory/SubspacesToolkit`
- `https://github.com/ma-ath/subspyces`

Archived external-file leads:

- SubspaceMethodsToolBox: `cvlPCA.m`, `cvlKPCA.m`, `cvlBasisVector.m`, `cvlCanonicalAngles.m`, `cvlKernelBasisVector.m`, `cvlKernelCanonicalAngles.m`.
- SubspacesToolkit: `computePCA.m`, `computeKernelPCA.m`, `computeBasisVectors.m`, `computeSubspacesSimilarities.m`, `computeKernelSubspacesSimilarities.m`.
- subspyces: `pca_transform.py`, `cosine_canonical_angles.py`, `vector_space.py`.

Use these as reference points, not unquestioned ground truth.

## Dataset Context

OSCD:

- Active benchmark.
- Binary change/no-change.
- Sentinel-2 pre/post images.
- Not disaster damage severity.
- Loader/reference alternatives such as TorchGeo and Hugging Face `blanchon/OSCD_MSI` are preserved as implementation leads. Before switching loaders, verify band order, preprocessing, labels, and train/test/validation split semantics; do not assume defaults match the current project split policy.

MultiSenGE:

- Useful future path for GDS/KGDS because it has multiple dates.
- Not currently the supervised Phase 2 benchmark.
- Lacks OSCD-style labels for the current main question.

xBD / xBD-S12:

- Important if the project later becomes disaster damage mapping.
- Current repo does not implement damage-specific training/evaluation.

Abandoned greenhouses:

- Useful future application.
- Can be mentioned as motivation/use case only.
- Not current evidence unless a dataset, labels, and evaluation protocol are added.

Harmonized Sentinel-2 L2A:

- Sensei recently approved trying this as a time-sequential satellite source and asked about frame count and time step.
- This should be audited before using first/second DS, GDS/KGDS, or geodesic projection on it.
- Important checks: available bands, date cadence, cloud/no-data filtering, same-region alignment, and whether there is any evaluation signal.

## Bookmark Triage Rules

When Chrome bookmarks are imported later, classify each item as one of:

- `must-cite`: needed for thesis claims.
- `implementation-reference`: useful for code or math.
- `baseline`: method to compare against.
- `background`: useful context, not urgent.
- `future`: xBD, greenhouses, MultiSenGE, semantic change, or broader application.
- `discard`: duplicate, weak, unrelated, or no longer useful.

The 2026-06-07 Chrome bookmark export was triaged into `notes/reference_bookmarks.md`. Use that file for Zotero-first reading order and the organized Chrome import file.

## Literature Leads

Source: `research-notes/refs_links/`, `research-notes/notes/sensei_notes.md`, and `research-notes/coverage_matrix.csv`.

Keep these as literature or implementation leads:

- Change detection surveys: required for broad overview and to avoid method-forcing.
- OSCD / FC-Siamese: current benchmark family and supervised baseline context.
- Metric-CD: modern unsupervised remote-sensing CD pressure; compare protocols carefully.
- MapFormer and other prior-/semantic-guided CD work: evidence that adding priors is not automatically novel.
- xBD and xBD-S12: future damage-dataset bridge; do not claim active damage evaluation until implemented.
- Transfer learning for emergency building-damage assessment: useful disaster-evaluation realism.
- ChangeOS: future object-level damage/change comparison, not active implementation.
- FC-Siam / Siamese FCN: supervised OSCD-style change-detection baseline family; use as comparison pressure for any DS-only or prior-assisted segmentation claim.
- Second-order DS, time-series DS, RTW, SSA, SFA: future temporal/subspace line if MultiSenGE or another multi-date dataset becomes central.
- Signal Latent Subspace: future route for deep-feature subspace priors or classifiers if raw spectral/local DS is insufficient.
- Celik PCA-kmeans and IR-MAD: classical baselines that must be implemented/audited carefully before using as negative evidence.

Second-pass details from `research-notes/refs_links/benchmark_watchlist.md`:

- Metric-CD is useful comparison pressure because it optimizes per image and thresholds its saved map at `0.5`; do not compare its F1/IoU directly to this repo without aligning thresholding, masking, averaging, bands, normalization, and compute.
- xBD-S12 was logged as a warm extension because it reportedly provides `10,315` aligned Sentinel-1/Sentinel-2 pre/post pairs connected to xBD-style damage labels; capture the official release repo before implementation.
- Emergency-context transfer-learning work should be used to discuss disaster generalization and emergency-realistic evaluation, not as evidence that this repo already solves damage mapping.
- MapFormer and other prior-guided / semantic-guided change-detection papers are important because they weaken any novelty claim based only on "adding prior information."
- External benchmark tables should start with a protocol-alignment note before score comparison.

Citation seeds from `research-notes/refs_links/initial_refs.bib` to preserve:

- Daudt et al. OSCD / FC-Siamese.
- MultiSenGE dataset paper.
- Remote-sensing change-detection survey material.
- Celik 2009 PCA-kmeans.
- Nielsen MAD / IR-MAD family.
- U-Net and ResNet only when the corresponding models are discussed.
- Gupta et al. xBD and Dietrich et al. xBD-S12 only for future damage framing.

## Reference Leads

Additional references and code leads from old archive notes:

| Lead | Why keep it |
|---|---|
| Fukui et al., second-order Difference Subspace | Relevant if the project moves from pairwise DS to richer subspace relationships. |
| Kanai et al. 2023, time-series anomaly detection based on DS between signal subspaces | Useful example that DS-like reasoning can support anomaly/change settings beyond classification. |
| Elhamifar and Vidal 2013, Sparse Subspace Clustering | Future clustering or pseudo-label baseline; useful only if a clear change-feature clustering experiment is defined. |
| U-Net | Required background for Phase 2 segmentation model. |
| ResNet | Required only if ResNet-backbone experiments are used. |
| MVA 2025 hand-shape SSS paper | Superposed shape subspace and key-frame selection analogy for patch-stack or key-date satellite subspaces. |
| MVA 2025 human-motion shape-subspace paper | Uses DS between shape subspaces to measure whole-motion variation, landmark contribution, and coordination; future analogy for band/patch/date contribution analysis. No reliable public URL is currently available. |
| MagTool reference code | Useful for projector/eigen and subspace-magnitude cross-checking; not ground truth. |
| DS shape/motion reference code | Useful because it uses projector-eigen style DS close to the repaired `eig` path. |
| MATLAB Subspace Toolbox | Useful for PCA, KPCA, CCA, and Kernel CCA implementation cross-checks. |

Keep these as citation or implementation-reference leads. Do not cite old archive prose itself as evidence.
