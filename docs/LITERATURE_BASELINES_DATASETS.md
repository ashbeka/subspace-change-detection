# Literature, Datasets, And Baselines

## Quick Links

- [1. Purpose](#purpose)
- [2. Reading Priorities](#reading-priorities)
- [3. Concept-To-Reading Map](#concept-to-reading-map)
- [4. Must-Cite Boundaries](#must-cite-boundaries)
- [5. Dataset Map](#dataset-map)
- [6. Baseline Pressure](#baseline-pressure)
- [7. Reference Code](#reference-code)
- [8. Knowledge-Base Concepts Absorbed](#knowledge-base-concepts-absorbed)
- [9. HSI Fair Baseline And Novelty Boundary](#hsi-fair-baseline-and-novelty-boundary)
- [10. Bookmark Policy](#bookmark-policy)
- [11. Exact Resource State](#exact-resource-state)

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
| 8 | xBD-S12 | external Sentinel-2 disaster validation context |
| 9 | second-order/time-series DS papers | Sensei-aligned temporal subspace dynamics |
| 10 | foundation-model CD papers | future deep-feature geometry lane |

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
| xBD-S12 | external disaster pressure | xBD-derived building damage labels | candidate localization | not generic OSCD-style change |
| MultiSenGE | multi-date exploration | weak/unclear | temporal DS/RTW exploration | no clean target labels |
| Harmonized Sentinel-2 L2A | Sensei-requested sequence source | depends on chosen event | future temporal DS/GDS | needs audit |
| xView2 | disaster damage / building assessment family | building/damage labels | future context only | RGB-centric and not current active pipeline |
| EuroSAT | land-use classification | land-use class labels | historical candidate | not change detection by itself |
| Data Fusion Contest resources | multimodal remote-sensing benchmarks | varies by year | future resource pool | route must be selected before use |
| IPOL/SITS sequences | registered time-series tests | detector maps/events | temporal DS characterization | detector maps not ground truth |
| HSI datasets | spectral geometry pressure | scene-dependent labels | HSI transfer probes | not current positive route |
| SpaceNet7 | RGB building appearance | building/temporal labels | transfer stress test | raw L2 stronger |

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
- shallow classifier or linear probe where labels exist.

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
