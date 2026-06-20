# Literature Notes

## Table Of Contents

- [1. Must-Cite Core Sources](#1-must-cite-core-sources)
- [2. Subspace And Kernel References](#2-subspace-and-kernel-references)
- [3. Reference Code](#3-reference-code)
- [4. Dataset Context](#4-dataset-context)
- [5. Bookmark Triage Rules](#5-bookmark-triage-rules)
- [6. Bookmark-Backed Concept Map](#6-bookmark-backed-concept-map)
- [7. Literature Leads](#7-literature-leads)
- [8. Reference Leads](#8-reference-leads)
- [9. Reset Literature Problem Map](#9-reset-literature-problem-map)
- [10. Temporal Satellite Subspace Reading Spine](#10-temporal-satellite-subspace-reading-spine)

## 1. Must-Cite Core Sources

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

## 2. Subspace And Kernel References

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

Spatial-spectral subspace and tensor-subspace boundary:

- Spatially aware subspace ideas already exist in remote sensing, especially hyperspectral image analysis.
- Wu, Du, and Zhang, 2013, "A Subspace-Based Change Detection Method for Hyperspectral Images" is a direct warning that subspace-based HSI change detection is established, not new. Reference listing: https://www.mdpi.com/2072-4292/14/11/2523
- Chen and Wang, 2017, "Spectrally-Spatially Regularized Low-Rank and Sparse Decomposition" explicitly uses spectral-spatial regularization for multitemporal HSI change feature extraction: https://www.mdpi.com/2072-4292/9/10/1044
- Gatto et al., "Tensor Analysis with n-Mode Generalized Difference Subspace," proposes n-mode GDS for tensor data because ordinary GDS does not directly preserve tensor structure: https://arxiv.org/abs/1909.01954
- Therefore the thesis should not claim the first spatial satellite subspace method. The safer contribution is an evidence-backed DS/GDS-style spatial-support adaptation for multispectral Sentinel-2 change maps.

Green Learning / PixelHop / wavelets:

- Apple Notes and senpai feedback suggest Green Learning, PixelHop/PixelHop++, wavelet transforms, and image-compression intuition as spatial-preservation leads.
- Their possible value is not "another method to add" but a way to build local/multiscale features before subspace comparison.
- Treat them as literature to read only if the spatial DS audit shows raw global pixel subspaces are insufficient.
- The remembered pyramid idea, whole image -> `2x2` -> `4x4` -> finer subspaces, should be tracked as `multiscale_subspace_pyramid` until the exact Green Learning / PixelHop / wavelet source is verified.

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

## 3. Reference Code

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

Pending reference-code reminder:

- The user has at least three additional senpai-provided reference codebases that should be added later under `references/reference_code/` or another agreed reference-code location.
- Reminder 2026-06-14: ask the user to supply these remaining reference codebases before any major new DS/GDS/KDS/KGDS, geodesic, CCA/KCCA, or spatial-subspace implementation.
- When they are supplied, inventory them before cleanup or implementation:
  - implemented method families;
  - language/dependencies;
  - expected input sample definitions;
  - source papers/equations if visible;
  - whether they can verify DS/GDS/KDS/KGDS/CCA/KCCA/spatial or temporal subspace variants.
- Do not start a major new subspace-method implementation before asking whether these reference codebases should be included in the paper-to-code comparison.

## 4. Dataset Context

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

## 5. Bookmark Triage Rules

When Chrome bookmarks are imported later, classify each item as one of:

- `must-cite`: needed for thesis claims.
- `implementation-reference`: useful for code or math.
- `baseline`: method to compare against.
- `background`: useful context, not urgent.
- `future`: xBD, greenhouses, MultiSenGE, semantic change, or broader application.
- `discard`: duplicate, weak, unrelated, or no longer useful.

The 2026-06-07 Chrome bookmark export was triaged into `notes/reference_bookmarks.md`. Use that file for Zotero-first reading order and the organized Chrome import file.

## 6. Bookmark-Backed Concept Map

Use this section as the bridge between literature concepts and the organized Chrome bookmark tree. The bookmark folders are not evidence by themselves; they are the reading queue for papers, datasets, reference implementations, and background resources.

| Concept | Why it matters | Bookmark priority folder |
|---|---|---|
| DS, GDS, KDS, KGDS, CCA/KCCA, and Sensei's subspace-method leads | Defines the mathematical family this project must understand before claiming any subspace contribution. | `Research / 01 Read First - Thesis Core / 01 Subspace DS KDS GDS And CCA` |
| OSCD, FC-Siamese, Metric-CD, Celik PCA-kmeans, IR-MAD, and CVA | Defines the current binary change-detection benchmark and the classical/deep baselines that pressure novelty claims. | `Research / 01 Read First - Thesis Core / 02 OSCD And Classical Change Detection Baselines` |
| OSCD, MultiSenGE, HR/SCD, xBD/xBD-S12, and candidate EO datasets | Clarifies which dataset supports which claim: OSCD for binary change, MultiSenGE/Harmonized Sentinel-2 for temporal ideas, xBD/xBD-S12 for future damage work. | `Research / 01 Read First - Thesis Core / 03 Datasets And Problem Setting` and `Research / 04 Datasets - Current Candidate Future` |
| Change-detection surveys and reality checks | Prevents method-forcing and helps state what is already known before proposing DS-based priors or spatial subspace variants. | `Research / 01 Read First - Thesis Core / 04 Review Papers And Reality Checks` |
| Spatial information, semantic change, patch/local DS, and prior-guided CD | Directly answers Sensei's criticism that global pixel subspaces may break spatial information. | `Research / 01 Read First - Thesis Core / 05 Spatial And Semantic Change Questions` |
| Spatial-spectral subspace and tensor/n-mode subspace methods | Important novelty boundary: related methods already exist, but they are not the same as the current Sentinel-2 DS adaptation. | `Research / 02 Methods - Subspace Geometry` and `Research / 03 Methods - Change Detection / 01 Binary And Unsupervised Change Detection` |
| Subspace implementation and reference code | Supports paper-to-code checks for PCA/KPCA/CCA/KCCA/DS/GDS and prevents hallucinated implementations. | `Research / 02 Methods - Subspace Geometry` and `Research / 08 Tools And Code - Implementation Support / 01 Reference Implementations` |
| Remote-sensing preprocessing and multitemporal/sensor issues | Necessary before interpreting change maps; clouds, registration, seasonality, sensor differences, and temporal sampling can dominate true change. | `Research / 03 Methods - Change Detection / 04 Preprocessing Multitemporal And Sensor Issues` |
| Application framing: greenhouses, urban infrastructure, humanitarian mapping, ecology | Useful for motivation and future datasets, but only becomes evidence after labels/evaluation protocols exist. | `Research / 05 Applications - Use Cases And Problem Framing` |
| ML/CV background, tutorials, and generic LLM/tooling material | Useful for personal learning but not thesis literature unless tied to the satellite/subspace/change-detection problem. | `Learning / Machine Learning and AI`, `Learning / Math`, and `Programming and Tools` |
| Lower-priority research-like papers | Preserved for future exploration without distracting from thesis-core reading. | `Research / 09 Parking Lot - Lower Priority` |

The current reading rule is: start with `Research / 01 Read First - Thesis Core`, then move to `02 Methods`, `03 Methods`, and `04 Datasets` only as needed for the next experiment or paper section. Do not start by reading generic ML tutorials when the project blocker is subspace construction, spatial preservation, or change-detection evaluation.

## 7. Literature Leads

Source: `research-notes/refs_links/`, `research-notes/notes/sensei_notes.md`, and `research-notes/coverage_matrix.csv`.

Keep these as literature or implementation leads:

- Change detection surveys: required for broad overview and to avoid method-forcing.
- OSCD / FC-Siamese: current benchmark family and supervised baseline context.
- Metric-CD: modern unsupervised remote-sensing CD pressure; compare protocols carefully.
- MapFormer and other prior-/semantic-guided CD work: evidence that adding priors is not automatically novel.
- CGNet / Change Guiding Network: named comparison pressure for "change-prior-guided" neural CD because it generates change maps from deep features and uses them as prior information for multi-scale fusion. Use it as related work if this project claims prior-guided fusion, not as evidence that DS priors are useful.
- AdaSemiCD and mean-teacher semi-supervised CD papers: named comparison pressure for pseudo-label experiments because they explicitly evaluate or filter pseudo-labels for remote-sensing CD. Use them to justify why DS/PCA/IR-MAD pseudo-labels must be quality-checked before training.
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

## 8. Reference Leads

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

## 9. Reset Literature Problem Map

This section connects the refined `Research` bookmark tree and the online reset scan to concrete reading priorities. It is not a complete bibliography. It is the minimum map needed to stop forcing subspaces onto change detection without a field-grounded reason.

| Concept | Read first | Why it matters | Thesis use | Bookmark location |
|---|---|---|---|---|
| Current field map for remote-sensing CD | Wang et al. 2024, "Advances and Challenges in Deep Learning-Based Change Detection for Remote Sensing Images" | Shows that CD is now dominated by supervised, semi-supervised, weakly supervised, unsupervised, and foundation-model DL paradigms. | Use to define the real field and avoid pretending DS is competing only with simple PCA/CVA. | `Research / 01 Read First - Thesis Core / 04 Review Papers And Reality Checks` |
| Label-efficient CD | "Toward Label-Efficient Deep Learning Change Detection for Remote Sensing Imagery" and sample-efficient CD survey | Confirms that label cost is a real field problem; divides methods into semi-supervised, weakly supervised, self-supervised, active, few-shot, and unsupervised tracks. | Supports a label-efficiency motivation, but also raises strong comparison pressure from modern methods. | `Research / 01 Read First - Thesis Core / 04 Review Papers And Reality Checks` |
| OSCD binary benchmark | Daudt OSCD and FC-Siamese papers/resources | Defines the current benchmark: 24 Sentinel-2 pairs, 13 bands, urban change labels. | Use only for binary changed-area detection, not disaster damage. | `Research / 01 Read First - Thesis Core / 02 OSCD And Classical Change Detection Baselines` |
| Classical multivariate CD | CVA, PCA-diff/Celik, MAD/IR-MAD, CCA/MAD references | Classical methods are already established and interpretable; DS must beat or explain itself against them. | Required baselines and paper-to-code verification targets. | `Research / 01 Read First - Thesis Core / 02 OSCD And Classical Change Detection Baselines` and `Research / 03 Methods - Change Detection / 01 Binary And Unsupervised Change Detection` |
| KPCA/KDS | Nielsen and Canty KPCA for change detection; Fukui/Maki TPAMI KDS/KGDS | KPCA is already used for nonlinear remote-sensing CD; KDS is lab-aligned but not automatically novel. | Future nonlinear experiment; must be framed as adaptation/verification, not invention. | `Research / 02 Methods - Subspace Geometry / 02 Kernel PCA And Kernel Subspace` |
| Spatial information loss | Sensei feedback plus spatial/semantic CD resources | The current global pixel subspace ignores position during PCA fitting; this is the strongest methodological gap. | Justifies the immediate global-pixel vs patch-vector vs local-window DS audit. | `Research / 01 Read First - Thesis Core / 05 Spatial And Semantic Change Questions` |
| Existing spatial-spectral subspace related work | Wu-Du-Zhang HSI subspace CD, LRSD_SS, spectral-spatial sparse subspace clustering, and n-mode GDS | Weakens any generic novelty claim, but strengthens the argument that spatial support is a real methodological issue. | Use to position the project as a specific DS-style Sentinel-2 adaptation/evaluation, not as the first spatial subspace method. | `Research / 02 Methods - Subspace Geometry` |
| Pseudo-change vs meaningful change | CD surveys, OSCD label policy, IR-MAD/GEE cautions | Numeric change includes shadows, clouds, seasonality, registration, sensor effects, and real land-cover changes. | A defensible problem framing if experiments show DS/local subspaces help diagnose or reduce false changes. | `Research / 03 Methods - Change Detection / 04 Preprocessing Multitemporal And Sensor Issues` |
| Open-vocabulary/foundation-model CD | TDCD, Weak Temporal Supervision, Seg2Change, ChangeVFM/Semantic-CD leads | Newer work moves toward object/class-specific and semantic change without paired labels. | Comparison pressure and possible pivot, not immediate implementation. | `Research / 03 Methods - Change Detection / 02 Semantic And Damage Change Detection` |
| Prior-guided and pseudo-label CD | MapFormer, CGNet / Change Guiding Network, AdaSemiCD, and mean-teacher semi-supervised CD papers | Prior information and pseudo-labels are already active ideas in remote-sensing CD, so the project must distinguish DS-style geometric priors from generic prior fusion or teacher-student pseudo-labeling. | Use to frame DS/PCA/IR-MAD maps as interpretable geometric prior candidates; do not claim novelty from "adding a prior map" alone. | `Research / 01 Read First - Thesis Core / 05 Spatial And Semantic Change Questions` and `Research / 03 Methods - Change Detection` |
| Multi-date DS/GDS/KGDS | Second-order DS, GDS, RTW, SFA/SSA, Harmonized Sentinel-2, MultiSenGE | GDS/KGDS need more than two subspaces; OSCD does not naturally support that beyond pre/post. | Future stronger research path if a multi-date dataset and evaluation proxy are defined. | `Research / 01 Read First - Thesis Core / 01 Subspace DS KDS GDS And CCA` and `Research / 04 Datasets - Current Candidate Future / 01 Sentinel OSCD MultiSenGE And EO Datasets` |
| Abandoned greenhouse mapping | Global-PCG-10, Sentinel-2 greenhouse index, two-temporal Sentinel-2 greenhouse mapping | Greenhouse mapping is real and Sentinel-2-relevant, but the task may be object mapping/classification rather than generic change detection. | Use as application motivation or future dataset only after labels/evaluation exist. | `Research / 05 Applications - Use Cases And Problem Framing / 01 Greenhouses Agriculture Environmental Monitoring` |
| xBD/xBD-S12 damage | xBD and xBD-S12 resources | Damage assessment is a different task with building objects and damage levels. | Future pivot/warm extension only; do not use OSCD evidence as damage proof. | `Research / 04 Datasets - Current Candidate Future / 03 Disaster Damage And Semantic Datasets` |
| Object/building-level descriptors | xBD, ChangeOS, object-level CD and greenhouse resources | Object units may preserve spatial/semantic information better than global pixel subspaces. | Candidate pivot if pixel/patch DS is too weak or if an object-labeled dataset is chosen. | `Research / 03 Methods - Change Detection / 02 Semantic And Damage Change Detection` and `Research / 05 Applications` |

### 9.1 Geometry Versus Neural-Network Framing

This is the literature-backed reason to keep subspace/geometric methods in the project without pretending they are the strongest standalone supervised change detector.

| Claim to support | Read first | What it allows us to say | What it does not allow us to say |
|---|---|---|---|
| Remote-sensing CD is still a broad practical problem, not a solved benchmark exercise. | Asokan and Anitha 2019, "Change detection techniques for remote sensing applications: a survey" ([Springer](https://link.springer.com/article/10.1007/s12145-019-00380-5)) | Classical/geometric methods can still be discussed as part of a broader method family, especially when interpretability and practical artifacts matter. | It does not prove DS is novel or better than deep learning. |
| Deep learning is the current dominant performance direction. | Wang et al. 2024, "Advances and Challenges in Deep Learning-Based Change Detection for Remote Sensing Images" ([MDPI Remote Sensing](https://www.mdpi.com/2072-4292/16/5/804)) | The thesis must compare against neural baselines or at least acknowledge modern DL comparison pressure. | It does not mean every useful contribution must be a new neural architecture. |
| Label scarcity and incomplete supervision remain active research problems. | Sample-efficient / label-efficient remote-sensing CD surveys and papers. | Subspace maps can be tested as priors, pseudo-label candidates, auxiliary targets, or low-label aids. | A prior map is not automatically a good pseudo-label; it must be evaluated against labels or useful downstream behavior. |
| Geometry can complement learning. | Lab subspace literature, DS/GDS/KDS/KGDS, Grassmann/neural-subspace hybrids, and current experiment backlog. | The defensible framing is geometry-only diagnostics first, then geometry-plus-learning if the maps show value. | Do not claim "geometric methods beat neural networks" unless controlled experiments actually show it. |

Project use:

- In the thesis introduction, state that neural networks are strong but often label-hungry and opaque.
- In the method section, present DS-style maps as interpretable geometric evidence, not as a guaranteed superior detector.
- In experiments, compare geometry-only outputs against raw L2, PCA-diff, IR-MAD, and at least one neural baseline.
- In discussion, treat negative DS results as evidence about sample construction and spatial information loss, not as project failure.

New external reset sources to keep active:

- Wang et al. 2024, DL-based CD review: CD applications include urban planning, disaster management, and national security; challenges include incomplete supervision, self-supervision, foundation models, and multimodal data.
- Parelius 2023 CD review: distinguishes relevant changes from apparent/irrelevant changes and emphasizes co-registration, spectral bands, and method families.
- Label-efficient/sample-efficient CD surveys: label scarcity is real, but the field has many modern alternatives.
- MapFormer and CGNet: prior-guided fusion is not new; this project needs a DS-specific or spatial/temporal/geometric value claim if it uses priors.
- AdaSemiCD and related mean-teacher semi-supervised CD: pseudo-labeling is not new; this project needs pseudo-label quality audits and low-label comparisons before claiming label efficiency.
- Weak Temporal Supervision / ChangeStar-style work: change labels can be avoided by exploiting single-temporal annotations or newly acquired images.
- Open-vocabulary CD / Seg2Change / TDCD: class-specific or text-driven change is now a serious alternative path for semantic/object-level research.
- Nielsen and Canty KPCA/MAD work: nonlinear PCA and CCA/MAD are already part of remote-sensing CD history, so KPCA/KDS novelty must be specific.
- Global-PCG-10 and plastic greenhouse index resources: greenhouse monitoring is legitimate but needs a separate task definition and validation plan.

Reference-resource indexes:

- Paper/resource index: `references/reference_papers/REFERENCE_RESOURCE_INDEX.md`.
- Reference-code inventory: `references/REFERENCE_CODE_INVENTORY.md`.
- Use these indexes as navigation aids only. The source papers and formula checks still decide whether an implementation is correct.

### 9.2 New Source Leads From 2026-06-17

These leads came from the updated Apple/Slack notes and new bookmark source. Treat them as reading and verification targets, not established project evidence yet.

Bookmark integration:

- Importable file: `docs/source_records/final_organization_2026-06-12/chrome_bookmarks_organized_all_2026-06-20.html`.
- Merge summary: `docs/source_records/final_organization_2026-06-12/bookmark_organization_summary_2026-06-20.json`.
- The new `Research (un-ingested)` source contributed `186` unique URLs; all were preserved in the organized output.
- Only `14` ambiguous research-like links remain parked because their titles/URLs do not identify the resource safely.

| Concept | New lead | Why it matters | Immediate action |
|---|---|---|---|
| BCD versus SCD | binary change detection, semantic change detection, and post-classification change detection resources | Clarifies that OSCD is binary changed-area detection, while many attractive ideas require semantic/object labels. | Keep OSCD experiments binary; read SCD/PCCD only if pivoting to semantic or object/state change. |
| Spatial support for subspaces | Jang's channel-wise flattening idea plus spatial-spectral HSI references | Gives a concrete alternative to unordered pixel samples. | Add a flattened-band spatial-subspace pilot after global/patch/window DS comparisons. |
| Temporal GDS/KGDS | Harmonized Sentinel-2 L2A and multi-date sequence notes | GDS/KGDS require multiple subspaces; OSCD has only two dates. | First produce a sequence feasibility report: frames, dates, bands, cloud/no-data, alignment. |
| Prior versus pseudo-label | pseudo-label / teacher-student CD papers and prior-guided CD resources | Prevents vague claims that a DS map is automatically a training label. | Evaluate prior-map quality before pseudo-label pretraining. |
| Classical comparison pressure | Celik, IR-MAD, CVA, PCA-diff, post-classification CD | Spatial DS must be compared against established classical methods, not only against itself. | IR-MAD and Celik checks are implemented and tested; keep them as required pressure baselines, while labeling Celik as a project adaptation and IR-MAD as a compact verified implementation. |
| Hybrid geometry plus neural methods | Siamese networks, U-Net/DeepLab/PSPNet-style segmentation, foundation models, deep-feature subspaces | A stronger route may use neural models for localization/features and subspaces for interpretation, clustering, or low-label priors. | Do not start large models yet; keep as follow-up if spatial DS maps show useful signal. |
| Hyperspectral/anomaly/data fusion route | MNF/PCA dimensionality reduction, hyperspectral anomaly detection, IEEE Data Fusion Contest | Richer-band imagery may be a more natural subspace application than Sentinel-2 OSCD. | Treat as pivot/future route; verify datasets and labels before implementation. |
| Application framing | abandoned greenhouse mapping, urban/infrastructure monitoring, disaster screening | These are possible motivations but not current evidence. | Use only as future application unless data and evaluation are available. |
| Research trend scans | awesome remote-sensing CD list, recent arXiv resources, SiROC, foundation/open-vocabulary CD leads | Helps prevent missing modern baselines and overclaiming novelty. | Use curated lists as discovery indices, not ground truth; add high-priority papers/repos to bookmark/literature triage, then verify with original papers/code before citing or implementing. |

## 10. Temporal Satellite Subspace Reading Spine

The active temporal study needs four literature layers. Read them in this order
instead of collecting unrelated papers.

### 10.1 Core Geometry

| Source | Exact role | Project consequence |
|---|---|---|
| Fukui et al. 2024, *Second-order Difference Subspace*, https://arxiv.org/abs/2409.08563 | Defines generalized first DS, magnitude, second DS, subspace projection, and approximate along/orthogonal decomposition. | Primary formula source. It assumes equal `Delta t`; irregular satellite dates need a separately labeled adaptation. |
| Kanai et al. 2023, *Time-series Anomaly Detection based on Difference Subspace between Signal Subspaces*, https://arxiv.org/abs/2303.17802 | Builds SSA signal subspaces from scalar sliding-window trajectory matrices and compares past/present signal subspaces. | The project block-trajectory matrix is a multivariate satellite adaptation. An unordered annual observation span is not Kanai's order-aware construction. |
| Fukui and Maki 2015, TPAMI DS/GDS/KDS/KGDS | Defines the broader DS/GDS and kernel family. | Must cite for DS lineage and avoid claiming DS itself is new. |
| Edelman, Arias, and Smith 1998, https://doi.org/10.1137/S0895479895290954 | Standard Grassmann geodesics and interpolation. | Source for evaluating the endpoint geodesic at the observed acquisition fraction. |
| Gatto et al. 2019, n-mode GDS, https://arxiv.org/abs/1909.01954 | Preserves tensor modes in generalized DS. | Future alternative if flattening spatial/spectral/time modes proves inadequate. |

### 10.2 Satellite Time-Series Change Baselines

| Source | What it measures | Comparison use |
|---|---|---|
| Dagobert et al. 2022, IPOL, https://doi.org/10.5201/ipol.2022.416 | Forward/backward novelty residuals, NFA significance, duration, multiscale patches on registered RGBI sequences. | Closest external sequence-level spatial baseline. Its C code and pseudo-gamma preprocessing were reproduced locally on four sequences; outputs are agreement targets, not labels. |
| Verbesselt et al. 2009, BFAST, https://doi.org/10.1016/j.rse.2009.08.014 | Breaks in trend and seasonal components. | Required pressure when claiming abrupt temporal change rather than ordinary seasonality. |
| Jamali et al. 2020, JUST, https://doi.org/10.3390/rs12234001 | Jump, trend, and seasonal analysis for noisy/irregular satellite time series. | Direct irregular-cadence and seasonality comparison pressure. |
| Zhu/CCDC family and dense time-series evaluations | Continuous change detection and classification. | Broader operational baseline; use when a sufficiently long sequence and event protocol exist. |
| Dagobert paper MOSUM comparison | Per-pixel moving-sum change-point baseline. | Lightweight first temporal statistical comparator. |

### 10.3 Pseudo-Change And Learned Temporal Representations

| Source | Relevance | Boundary |
|---|---|---|
| Du et al. 2019, DSFA, https://doi.org/10.1109/TGRS.2019.2930682 | Learns invariant/slow features so changed pixels differ while unchanged pixels remain stable. | Strong baseline/idea for separating slow background variation from abrupt change; not DS lineage. |
| Deep learning for satellite image time-series analysis review 2024, https://doi.org/10.1109/MGRS.2024.3393010 | Maps modern SITS representations and tasks. | Prevents comparing only against old classical methods. |
| ChangeMamba 2024, https://doi.org/10.1109/TGRS.2024.3417253 | Modern spatiotemporal state-space CD. | Performance pressure only; not a method to reimplement before the classical geometric question is answered. |
| Signal Latent Subspace 2024, https://doi.org/10.1016/j.apacoust.2024.110181 | Builds product-Grassmann subspaces from neural latent features. | Main analogy for a later raw-versus-deep temporal subspace experiment. |

### 10.4 Labeled Multi-Temporal Evaluation Candidate

| Source | What it provides | Feasibility consequence |
|---|---|---|
| Toker et al. 2022, *DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation*, https://openaccess.thecvf.com/content/CVPR2022/html/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.html | Daily Planet multispectral observations for 75 AOIs plus monthly pixel labels for seven land-cover classes; introduces semantic change segmentation evaluation. | Best identified labeled test for temporal characterization/localization, but the official archive is about 524 GB (`labels.zip` about 1.4 GB and imagery split into multi-GB latitude archives). Define one-AOI selective acquisition before downloading. |
| Official implementation, https://github.com/aysim/dynnet | Temporal baselines, splits, pretrained-model pointers, and official dataset link. | Use to understand data layout and evaluation; do not make it a dependency for the initial classical geometric test. |
| Ketchum et al. 2020, *IrrMapper*, https://doi.org/10.3390/rs12142328 | Annual 30 m irrigation classifications over 11 western US states; the current Earth Engine v1.2 catalog spans 1986 through 2024. | Enables candidate irrigation-regime transitions with strong Sentinel-2 overlap, but transitions are derived weak labels rather than manually annotated event dates. |
| Earth Engine IrrMapper v1.2 catalog, https://developers.google.com/earth-engine/datasets/catalog/UMT_Climate_IrrMapper_RF_v1_2 | Public binary annual irrigation collection and current temporal extent. | Practical source for candidate patch/year labels; requires an Earth Engine-enabled Cloud project for extraction. |
| Wenger et al. 2022, *MultiSenGE*, https://isprs-annals.copernicus.org/articles/V-3-2022/635/2022/isprs-annals-V-3-2022-635-2022.pdf | Local multi-date Sentinel-2/Sentinel-1 sequences and static LULC labels. | Useful for controlled temporal geometry and nuisance tests, but static labels do not validate event timing. |

DynamicEarthNet is not Sentinel-2 and therefore tests method generality rather
than a Sentinel-specific claim. A Harmonized Sentinel-2 event sequence with
independent annotations remains the preferred sensor-matched alternative if a
manageable label source is available.

IrrMapper's paper exposes a directly relevant gap: its annual model aggregates
seasonal reflectance statistics and explicitly does not model capture-level
spectral dynamics. That motivates testing seasonal trajectory representations.
It does **not** establish that DS is the solution. Its documented limitations
include fixed March-November season assumptions, weak response for orchards,
vineyards, sparse irrigation, and possible class-transition errors.

### 10.5 Hyperspectral And Temporal Leads From 2026-06-20

| Source | What it adds | Decision |
|---|---|---|
| Theiler and Perkins, *Proposed Framework for Anomalous Change Detection*, https://web.engr.oregonstate.edu/~wongwe/workshops/icml2006/papers/theiler.pdf | Frames rare local change against pervasive illumination, focus, atmosphere, or seasonal differences. | Strong conceptual pressure for the nuisance-subspace idea; compare against anomalous-change methods rather than only ordinary CVA. |
| Chakraborty and Ghosh 2021, https://arxiv.org/abs/2109.04990 | Unsupervised hyperspectral CD using multi-level feature-fusion autoencoders. | Modern nonlinear comparison pressure for a hyperspectral pivot. |
| Gatto et al., n-mode GDS, https://arxiv.org/abs/1909.01954 | Tensor modes are represented and compared without collapsing all structure into one matrix. | Main source for a later spatial-spectral-temporal tensor experiment; not needed for the first seasonal-subspace test. |
| CiTIUS hyperspectral CD dataset, https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset | Registered hyperspectral pairs and change-detection benchmark resources. | First practical dataset lead if the hyperspectral route is activated. |
| Nielsen, MAD/MAF source page, https://orbit.dtu.dk/en/publications/multivariate-alteration-detection-mad-and-maf-postprocessing-in-m/ | Classical invariant multivariate-change baseline lineage. | Required baseline for multispectral/hyperspectral claims. |
| Wu, Du, and Zhang 2013 subspace HSI-CD lineage | Confirms subspace-based hyperspectral change detection already exists. | Novelty cannot be "use subspaces on hyperspectral imagery"; novelty must be the exact geometric object, temporal design, or verified capability. |

The latest essential knowledge base also preserves direct DS/GDS, second-order
DS, time-series DS, S3CCA/TRCCA, signal-latent-subspace, Grassmann tracking,
SFA, SSC, tensor GDS, and toolbox links. Search-result URLs and ChatGPT links are
provenance only; original papers and repositories remain the citation sources.

### 10.6 Current Novelty Boundary

The scoped search found extensive satellite time-series change-point,
seasonality, novelty, classical subspace, hyperspectral subspace, and learned
spatiotemporal work. It did **not** find a direct publication applying Fukui's
2024 first/second DS plus geodesic decomposition to registered multispectral
satellite image sequences. This is evidence of novelty potential, not proof of
novelty. A formal systematic search and advisor confirmation are still needed.

The defensible novelty is not "subspaces for satellite images." It may be one
of:

- first validated use of this specific first/second DS geometry on SITS;
- an irregular-cadence extension with explicit separation from paper theory;
- a registration-robust/local multiscale construction;
- or an empirical result explaining what these geometric quantities detect and
  where they fail relative to NFA/MOSUM/BFAST/JUST and learned methods.

### 10.7 Research-Mining Boundary For HSI Distribution Change (2026-06-21)

The closest-prior audit changes the novelty boundary:

- Liu et al. 2019, DOI `10.1109/MGRS.2019.2898520`: exact
  multitemporal-HSI-CD review boundary.
- Lv et al. 2025, DOI `10.1016/j.inffus.2025.103257`: latest verified
  peer-reviewed HSI land-cover CD survey; closed full text in this session.
- Wu et al. 2013, DOI `10.1109/JSTARS.2013.2241396`: background-subspace HSI
  CD already exists.
- Schaum/Stocker and Theiler covariance-equalization/quadratic ACD lineage,
  including DOI `10.1364/AO.47.000F12`: covariance/distributional HSI change is
  established.
- Chang et al. 2022, DOI `10.1109/TGRS.2022.3220814`: multiview subspace
  learning for anomalous HSI CD already exists.
- Coupled/multitemporal unmixing, including DOI
  `10.1109/JSTARS.2021.3104164`, already reports endmember-specific change
  direction/intensity.
- Sparse PCA and band-selection HSI CD, including DOI `10.1117/12.897434` and
  DOI `10.1109/TGRS.2024.3382638`, already select wavelength regions/bands.

Therefore the defensible opening is not subspace, covariance, subpixel change,
or band selection alone. It is the exact combination of local
mean/scale/eigenspectrum/orientation factorization and contiguous wavelength
attribution, under direct covariance/SPD, MMD, unmixing, and sparse-band
pressure. Full-text IEEE/Scopus/Web of Science searching remains required
before any “first” claim.
