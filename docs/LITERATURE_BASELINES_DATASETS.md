# Literature, Datasets, And Baselines

## Quick Links

- [1. Purpose](#purpose)
- [2. Reading Priorities](#reading-priorities)
- [3. Concept-To-Reading Map](#concept-to-reading-map)
- [4. Recent Literature Pressure For The Problem Statement](#recent-literature-pressure-for-the-problem-statement)
- [5. Must-Cite Boundaries](#must-cite-boundaries)
- [6. Dataset Map](#dataset-map)
- [7. Baseline Pressure](#baseline-pressure)
- [8. Reference Code](#reference-code)
- [9. Knowledge-Base Concepts Absorbed](#knowledge-base-concepts-absorbed)
- [10. HSI Fair Baseline And Novelty Boundary](#hsi-fair-baseline-and-novelty-boundary)
- [11. Bookmark Policy](#bookmark-policy)
- [12. Exact Resource State](#exact-resource-state)

## Purpose

This file answers: **what should I read, cite, compare against, or reuse?**

It replaces scattered literature notes, bookmark summaries, reference-code
inventories, and dataset reminders as the active reading map.

## Reading Priorities

| Priority | Source/concept | Why read now |
|---:|---|---|
| 1 | Fukui and Maki TPAMI 2015 DS/GDS/KDS/KGDS | defines DS; prevents false novelty claims |
| 2 | PixelHop / Successive Subspace Learning | source for successive local Saab features |
| 3 | Green Learning overview | explains feed-forward/label-free feature philosophy |
| 4 | OSCD / Daudt et al. | current benchmark and baseline context |
| 5 | Nielsen IR-MAD | strong classical multivariate baseline |
| 6 | Celik PCA-kmeans, CVA, PCA-diff | classical unsupervised CD pressure |
| 7 | Remote-sensing CD surveys | field gaps, baselines, and evaluation expectations |
| 8 | DINOv2/DINOv3 and remote-sensing foundation features | modern frozen-feature pressure for any feature/prior claim |
| 9 | xBD-S12 | external Sentinel-2 disaster validation context |
| 10 | second-order/time-series DS papers | Sensei-aligned temporal subspace dynamics |
| 11 | foundation-model CD papers | future deep-feature geometry lane |

## Concept-To-Reading Map

Use this table when choosing what to read or cite for a route.

| Concept / route | Read first | Why it matters |
|---|---|---|
| DS/GDS/KDS/KGDS foundation | Fukui and Maki TPAMI 2015, IEEE `7053916` | defines the lab method; prevents false novelty claims |
| Second-order DS/geodesic | Fukui 2024 second-order DS, `https://www.arxiv.org/pdf/2409.08563` | Sensei-requested temporal/geodesic route |
| SSA + DS for time-series anomaly | `https://arxiv.org/abs/2303.17802` | shows DS on signal subspaces and useful references despite reported mistakes |
| Green Learning / PixelHop / Saab | Green Learning JVCIR 2022; PixelHop arXiv `1909.08190`; Saab/feedforward CNN papers | source for label-free local features before DS |
| IR-MAD / CCA baseline | Nielsen 2007 IR-MAD, `https://doi.org/10.1109/TIP.2006.888195` | strong classical multivariate baseline; close to correlation-space change |
| S3CCA / structured matching | `https://staff.aist.go.jp/takumi.kobayashi/publication/2014/ICPR2014.pdf` | Sensei/senpai route for smooth sparse partial matching |
| Temporally regularized CCA | IEEE `7477238` and related Fukui/Kobayashi work | sequence/view matching route |
| SFA / slow feature subspace | AIST BMVC 2017 SFA paper; Slow Feature Subspace paper | invariant/background route and nuisance control |
| Signal Latent Subspace | ScienceDirect `S0003682X24003323` | analogy for building subspaces from learned latent features |
| Spatial context CD | SiROC/spatial-context unsupervised CD resources | closest pressure for spatially aware unsupervised CD |
| HSI subspace CD | Wu/Du/Zhang 2013 HSI subspace CD and HSI-CD surveys | novelty boundary: subspace HSI CD already exists |
| Anomalous change detection | Chronochrome, covariance equalization, whitened TLSQ/ACD family | controls for nuisance-invariant residual claims |
| Open-vocabulary / foundation CD | open-vocabulary, open vocabulary, semantic, and foundation-model CD papers from bookmarks | future route for object-specific or text-conditioned change |
| Frozen dense feature priors | DINOv2/DINOv3, SAM/SAM2, RemoteCLIP, Prithvi/Clay/SatMAE/SSL4EO-style RS foundation models | mandatory modern pressure if claiming feature extraction or prior-map value; compare raw feature L2/cosine to feature-space DS/GDS |
| Semantic/object-level change | SCD surveys, SAM/CLIP/GeoAI object proposals, ChangeStar-style resources | route for "what changed into what?" beyond OSCD binary masks |
| Greenhouse monitoring | Global greenhouse maps, plastic greenhouse index, abandoned greenhouse project links | applied route only if labels/evaluation become available |
| Building-level descriptors | xBD/xBD-S12, object-detection/segmentation and damage-assessment references | object-level damage/candidate triage route |
| Sparse Subspace Clustering | Elhamifar and Vidal TPAMI 2013, IEEE `6482137` | union-of-subspaces baseline and possible change-type clustering route |
| Sentinel-2 dynamic time warping | Belgiu and Csillik 2018, `https://www.sciencedirect.com/science/article/pii/S0034425717304686` | phenology/seasonality timing control for temporal routes |
| UAV-assisted disaster management | Erdelj and Natalizio 2016, IEEE `7440563` | supports the edge/UAV deployment route and compute-resource motivation |
| Diffusion-pseudotime analogy | RamDA-seq / diffusion-map pseudotime source note, `https://www.nature.com/articles/s41467-018-02866-0` | inspiration for ordering regions from intact to damaged to recovering; not remote-sensing evidence by itself |
| Disaster-resilience planning / MCDA | disaster resilience frameworks, urban planning, HAI urban-change policy brief, MCDA/MCDM GIS papers | application layer for "what do we do with the heatmap?" |
| Multi-sensor disaster analysis | SAR/InSAR, LiDAR/DEM/topography, UAV imagery, Sentinel-2 bands | future route for clouds, terrain, structural shape, and resource-allocation use cases |
| Sparse modeling / feature selection | LASSO, sparse dictionary learning, band-selection papers | possible interpretable fusion or band-selection baseline |
| TorchGeo / geospatial ML tooling | TorchGeo docs/examples | useful engineering library for dataset loading, pretrained models, and geospatial ML baselines |
| Data Fusion Contest / multimodal benchmarks | IEEE GRSS Data Fusion Contest resources | future benchmark/resource pool; do not let it derail the current gate |
| xView2 / xBD / EuroSAT | xBD/xView2 damage, EuroSAT land-use, OSCD binary CD | historical dataset candidates and baseline context |
| MGRS / Sentinel-2 tiles | Sentinel-2 tile metadata and MGRS mapping references | needed when matching xBD-style locations back to Sentinel-2/HLS tiles |
| Snow/cloud masking | NDSI, Sentinel-2 cloud masks, QA bands | required if claiming seasonality or snow robustness |

Second-pass clarifications from the independent audit:

| Concept | Concrete source anchor | How to use it |
|---|---|---|
| xBD-S12 / Sentinel matching | official xBD-S12 repo `https://github.com/prs-eth/xbd-s12`; Zenodo `https://zenodo.org/records/18960454` | xBD-S12 is the current real example of aligning xBD-style disaster labels with Sentinel-1/2. Use it as external pressure for multispectral disaster claims, not as proof that OSCD results transfer to damage severity. |
| MGRS / HLS tiling | NASA HLS tiling notes `https://hls.gsfc.nasa.gov/products-description/tiling-system/`; NASA Earthdata HLS page | HLS uses the Sentinel-2/MGRS-style tiling system. This matters if reconstructing a reproducible event pipeline from xBD/MultiSenGE coordinates and Sentinel/HLS source imagery. |
| Change captioning / semantic descriptions | LEVIR-CC repo `https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset`; LEVIR dataset page; Dubai-CC and SECOND-CC papers/resources | These are evidence that "describe what changed" is a real route. They are mostly RGB/text datasets, so they support semantic/foundation routes but do not validate the current multispectral DS detector. |
| HSI subspace/unmixing boundary | Wu, Du, Zhang 2013 HSI subspace CD DOI `10.1109/JSTARS.2013.2241396`; Erturk/Plaza 2015 unmixing CD; SMSL/HACD and covariance-equalization/Chronochrome references | Generic "subspace for HSI CD" is already occupied. A future HSI lane must claim a narrower object: wavelength attribution, local orientation, material-subspace explanation, or a specific robustness/computation advantage. |
| IrrMapper / irrigation regime labels | Earth Engine `UMT/Climate/IrrMapper_RF/v1_2`; IrrMapper Remote Sensing 2020 DOI `10.3390/rs12142328` | Useful for annual irrigation on/off candidates and phenology/regime-change tests. Treat labels as classifier-derived annual maps, not direct manual switch-year ground truth. |
| Bookmark imports | old reference notes point to `docs/source_records/bookmarks/chrome_bookmarks_research_labeled_cleaned_2026-06-25.html`, but the current tree may not contain every historical import artifact | Do not delete bookmark source notes until the latest import file is recovered, regenerated, or explicitly abandoned. |

## Recent Literature Pressure For The Problem Statement

Use these as the first external anchors when explaining why the project matters.
Citation counts below are approximate public web/Semantic Scholar/ResearchGate
signals observed during the 2026-07-01 framing pass; verify exact counts before
putting them in a paper.

| Source | Recent/cited status | What it says for this project |
|---|---|---|
| Cheng et al., "Change Detection Methods for Remote Sensing in the Last Decade: A Comprehensive Review", Remote Sensing 2024, DOI `10.3390/rs16132355` | 2024 review; public pages show over 200 citations | CD is a broad, active field with challenges from noise, registration, illumination, complex landscapes, and spatial heterogeneity. Our problem must be framed as a specific response to representation/label/interpretability pressure, not generic CD. |
| Wang et al., "Advances and Challenges in Deep Learning-Based Change Detection for Remote Sensing Images: A Review through Various Learning Paradigms", Remote Sensing 2024, DOI `10.3390/rs16050804` | 2024 review; public pages show tens of citations | Reviews fully supervised, semi-supervised, weakly supervised, and unsupervised paradigms. This supports a sample-efficient/label-efficient framing. |
| Jiang et al., "A Survey on Deep Learning-Based Change Detection from High-Resolution Remote Sensing Images", Remote Sensing 2022, DOI `10.3390/rs14071552` | 2022 review; public pages show hundreds of citations | Establishes deep CD taxonomy, datasets, metrics, and SOTA pressure. We should not claim SOTA unless comparing to strong deep baselines. |
| Chen, Qi, and Shi, "Remote Sensing Image Change Detection With Transformers", TGRS 2021, DOI `10.1109/TGRS.2021.3095166` | 2021 transformer baseline; public pages show over 1000 citations | Strong deep models exist. Our win axis should be complementary prior evidence, interpretability, label efficiency, or lightweight computation, not beating transformers outright. |
| Zheng et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery", ICCV 2021 | 2021 weak/single-temporal supervision method; well-known code and paper | Confirms that reducing paired bitemporal annotation cost is a real problem. Our label-free priors can be positioned as a different route to reduce dependence on dense change labels. |
| Ding et al., "A Survey of Sample-Efficient Deep Learning for Change Detection in Remote Sensing", GRSM/arXiv 2025 | recent survey; citation count still young | Explicitly organizes sample-efficient CD and newer foundation/self-supervised directions. Use it for modern framing, but avoid over-relying on citation count. |
| Kakogeorgiou and Karantzalos, "Evaluating Explainable Artificial Intelligence Methods for Multi-label Deep Learning Classification Tasks in Remote Sensing", IJAEOG 2021 | 2021 XAI remote-sensing reference | Remote sensing needs explanations, but explanation evaluation is nontrivial. Our DS maps should be evaluated as interpretable priors, not just visually attractive heatmaps. |
| DINOv2/DINOv3 and VFM change-detection papers | DINOv2/DINOv3 dense features, robust scene-change papers using DINOv2, and remote-sensing foundation-model CD papers | A modern baseline can be as simple as frozen dense feature difference. If our DS prior only beats PCA/CVA but loses to DINO feature difference, the contribution must be reframed around geometry over foundation features or dropped. |
| DINOv3 efficiency pressure | Meta DINOv3 page and technical report emphasize frozen dense features plus efficient model variants for resource-constrained/on-device use | This directly challenges any DS/GDS "low compute" claim. The fair question is not "does GDS beat DINO?" but "does GDS offer a better compute/interpretability/usefulness tradeoff for the selected satellite task?" |

Problem implication:

```text
The meaningful contribution is not "DS detects changes." It is a controlled
answer to whether label-free, spatially structured subspace priors give
interpretable and complementary change evidence under the annotation and
generalization pressures recognized by recent remote-sensing CD literature.
```

## Must-Cite Boundaries

Do not claim novelty over:

- DS/GDS/KDS/KGDS themselves;
- subspace-based hyperspectral change detection in general;
- classical CVA/PCA/MAD/IR-MAD change detection;
- deep change detectors as a broad family;
- PixelHop/Green Learning feature extraction.

Possible novelty can only be specific:

```text
sample construction + satellite adaptation + controlled evidence boundary
```

## Dataset Map

| Dataset | Role | Labels | Current use | Limitation |
|---|---|---|---|---|
| OSCD | main benchmark | binary changed areas | Successive Saab-DS, spatial DS, U-Net priors | two-date, binary, not damage |
| LEVIR-CD / WHU-CD / S2Looking / SYSU-CD / DSIFN-CD | high-resolution RGB building/change benchmarks | binary building/change masks | candidate for DINO/SAM/foundation-feature pressure | not multispectral Sentinel-2; less aligned with DS-on-bands unless using deep features |
| SECOND / semantic CD datasets | semantic from-to change | semantic labels | candidate for semantic/object-level route | different task from OSCD binary CD |
| xBD-S12 | external disaster pressure | xBD-derived building damage labels with co-registered Sentinel-1/2 | candidate localization | not generic OSCD-style change; original Sentinel tiles are a separate large download |
| MultiSenGE | multi-date exploration | weak/unclear | temporal DS/RTW exploration | no clean target labels |
| Harmonized Sentinel-2 L2A | Sensei-requested sequence source | depends on chosen event | future temporal DS/GDS | needs audit |
| xView2 | disaster damage / building assessment family | building/damage labels | future context only | RGB-centric and not current active pipeline |
| EuroSAT | land-use classification | land-use class labels | historical candidate | not change detection by itself |
| Data Fusion Contest resources | multimodal remote-sensing benchmarks | varies by year | future resource pool | route must be selected before use |
| IPOL/SITS sequences | registered time-series tests | detector maps/events | temporal DS characterization | detector maps not ground truth |
| HSI datasets | spectral geometry pressure | scene-dependent labels | HSI transfer probes | not current positive route |
| SpaceNet7 | RGB building appearance | building/temporal labels | transfer stress test | raw L2 stronger |
| IrrMapper | annual irrigation status maps | classifier-derived irrigated/non-irrigated yearly labels | temporal regime-change candidate | useful for irrigation start/stop hypotheses, but switch labels need independent verification |

## Baseline Pressure

Every serious change-map experiment should compare against:

- raw spectral L2 / CVA;
- PCA-diff or smoothed PCA;
- Celik PCA-kmeans where applicable;
- IR-MAD;
- matched non-DS controls for any DS feature representation;
- supervised U-Net/Siamese baseline if claiming segmentation improvement.

Every temporal-sequence claim should compare against:

- raw residuals;
- interpolation residuals;
- seasonal/trend baselines;
- BFAST/CCDC/MOSUM/SSA-style controls when feasible.

Every deep/foundation-feature lane should compare against:

- raw embedding distance;
- cosine or Euclidean feature distance;
- PCA/cross-reconstruction controls;
- DINOv2/DINOv3 feature differences when image modality/resolution permits;
- remote-sensing foundation model features when available;
- shallow classifier or linear probe where labels exist.

Every compute-efficiency claim should report:

- preprocessing time;
- inference time;
- CPU/GPU used;
- peak GPU memory and CPU memory when feasible;
- trainable parameter count and training time;
- output quality per cost, such as AP per second, F1 per second, or review-budget
  recall at a fixed compute budget.

Every SSC or change-type clustering claim should compare against:

- k-means on the same descriptors;
- spectral clustering without sparse self-expression;
- direct raw/PCA/DS scores without clustering;
- semantic or human inspection that clusters correspond to meaningful change
  types rather than arbitrary score bins.

Every UAV/edge or lightweight-compute claim should compare against:

- runtime and memory on a fixed device or simulated budget;
- a compact CNN/U-Net or classical baseline with the same input;
- accuracy/latency/energy tradeoff, not only F1 or IoU.

## Reference Code

Reference code is teaching material, not ground truth.

Use it for:

- DS/GDS/KDS/KGDS formula checking;
- CCA/KCCA/S3CCA/TRCCA ideas;
- RTW/SFA/temporal subspace analogies;
- baseline implementation comparison;
- method provenance.

Do not copy reference code into active methods without:

```text
paper equation -> sample definition -> shape test -> toy check -> one-dataset output
```

Reference-code crosswalk preserved from old notes:

| Reference area | Files/functions to inspect | Project use |
|---|---|---|
| DS utilities | `DS/utils.py`, `MagTool` magnitude functions | verify DS magnitudes, principal angles, first/second DS behavior |
| MATLAB subspace functions | `cvtPCA.m`, `cvtKernelPCA.m`, `cvtCCA.m`, `cvtKernelCCA.m`, `cvtBasisVector.m` | paper-to-code checks for PCA/KPCA/CCA/KCCA bases |
| external toolboxes | SubspaceMethodsToolBox, SubspacesToolkit, subspyces | avoid hallucinated geometry formulas |

Pending source action:

- The user has at least three additional senpai reference codebases to add.
  Inventory them before major DS/GDS/KDS/KGDS/CCA implementation changes.

## Knowledge-Base Concepts Absorbed

From `docs/pending_deletion_review/old_knowledge_base/` and `docs/pending_deletion_review/old_research_material/*novelty*`, preserve these literature
boundaries:

| Concept cluster | What it teaches | Project implication |
|---|---|---|
| DS/GDS/KDS/KGDS/MSM/KMSM | lab lineage and canonical angle/subspace geometry | cite Fukui/Maki; do not claim DS invention |
| second-order DS/geodesic | subspace trajectory velocity/acceleration/along-orthogonal decomposition | supports temporal characterization lane |
| n-mode GDS/product Grassmann | mode-preserving geometry for tensor data | future satellite cube lane, not current proof |
| SFA/SFS/SSA/RTW | sequence invariance, slow features, trajectory subspaces, timing variation | compare before temporal novelty claims |
| S3CCA/TRCCA/KCCA | structured correlation matching | future view/sequence matching lane |
| HSI subspace CD/SMSL/covariance/unmixing | subspace HSI change is established | novelty must be exact object/evidence, not generic HSI subspace CD |
| SiROC/spatial context CD | spatial context is established in optical CD | spatial DS must beat matched spatial nulls |
| Signal Latent Subspace | deep latent features as subspaces/product Grassmann | bridge to foundation/deep-feature lane |

## HSI Fair Baseline And Novelty Boundary

Do not revive the HSI lane without this baseline pressure. Generic
"subspace-based HSI change detection" already exists.

Minimum fair baseline roster:

| Family | Baselines / closest priors |
|---|---|
| direct spectral change | raw L2/CVA, SAM |
| low-rank difference | PCA-diff, PCA-kmeans/Celik |
| correlation/invariant transforms | MAD/IR-MAD, CCA/KCCA, SFA/ISFA |
| HSI subspace CD | Wu/Du/Zhang 2013 HSI subspace CD, DOI `10.1109/JSTARS.2013.2241396`; Chen/Wang 2017 LRSD_SS |
| covariance/statistical change | Schaum/Stocker covariance equalization, Theiler quadratic ACD, shrinkage covariance/SPD distances |
| distribution tests | MMD, energy distance |
| band attribution | sparse PCA, band selection, contiguous wavelength attribution |
| mixture/material models | collaborative coupled unmixing, fast multitemporal unmixing, SMSL |
| modern HSI CD pressure | ECDBS, StripeCD, patch tensor HSI CD, at least one competitive deep HSI-CD model |
| spatial/radar context | SiROC/SemiSiROC, PWTT |

Search before HSI coding:

```text
hyperspectral change detection AND (Grassmann OR eigenspace orientation OR covariance orientation) AND (band attribution OR spectral interval)
```

Additional resource leads preserved from old research material:

- DSAMNet, DRMAT, RaVAE/RaVAE-like representation papers, CGNet;
- affinity-prior unsupervised change detection;
- DSFA and sample-efficient change-detection surveys;
- SST / mSSA-CUSUM for temporal change;
- PolSAR covariance/geometry work;
- GrNet and foundation-feature geometry references.

## Bookmark Policy

Chrome bookmarks are a reading queue, not active project truth. Important
resources should be promoted here only when they affect one of:

- a research lane;
- a baseline;
- a dataset;
- a method implementation;
- a citation boundary.

## Exact Resource State

Latest bookmark/resource state preserved from old notes. The actual organized
HTML import files are not present in the current working tree; recover them
from Git history or regenerate from a fresh Chrome export if needed.

| Item | State |
|---|---|
| latest cleaned import target | historical only; not currently in tree |
| total bookmark entries | `1869` |
| research entries | `740` |
| exact duplicate URLs | `0` |
| priority label counts | `54 / 198 / 414 / 74` |
| deduplication unit | intellectual item, not URL wrapper |

Deduplication preference:

- prefer arXiv abstract pages over direct Chrome PDF wrappers;
- preserve DOI/publisher pages when they are the citation authority;
- remove search-result and duplicate wrapper links after the real paper/tool
  link is preserved.
