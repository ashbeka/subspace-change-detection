# Literature, Datasets, And Baselines

## Quick Links

- [1. Purpose](#purpose)
- [2. Reading Priorities](#reading-priorities)
- [3. Must-Cite Boundaries](#must-cite-boundaries)
- [4. Dataset Map](#dataset-map)
- [5. Baseline Pressure](#baseline-pressure)
- [6. Reference Code](#reference-code)
- [7. Knowledge-Base Concepts Absorbed](#knowledge-base-concepts-absorbed)
- [8. HSI Fair Baseline And Novelty Boundary](#hsi-fair-baseline-and-novelty-boundary)
- [9. Bookmark Policy](#bookmark-policy)
- [10. Exact Resource State](#exact-resource-state)

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

Latest bookmark/resource state preserved from old notes:

| Item | State |
|---|---|
| latest cleaned import target | `docs/source_records/bookmarks/chrome_bookmarks_research_labeled_cleaned_2026-06-25.html` |
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
