# Reference Bookmarks

## Table Of Contents

- [Source And Preservation](#source-and-preservation)
- [Category Counts](#category-counts)
- [Zotero-First Reading Queue](#zotero-first-reading-queue)
- [Manual Additions After Bookmark Import](#manual-additions-after-bookmark-import)
- [Implementation References To Keep Visible](#implementation-references-to-keep-visible)
- [How To Use This](#how-to-use-this)
- [Import Back To Chrome](#import-back-to-chrome)
- [Categorization Notes](#categorization-notes)

This file summarizes the Chrome bookmark export for research reading and Zotero triage. The full raw and organized bookmark HTML files live in `docs/source_records/`.

## Source And Preservation

- Raw export: `docs/source_records/chrome_bookmarks_raw_2026-06-07.html`
- Organized Chrome import file: `docs/source_records/chrome_bookmarks_organized_research_2026-06-07.html`
- Raw export URLs preserved: `362`
- Raw export unique URLs: `362`
- Organized Chrome import URLs: `363` including `1` manual post-import addition.
- Duplicate URLs found in export: `0`
- Zotero context: use the high-priority paper list below to manually import the important papers first.

## Category Counts

| category | count | use |
|---|---:|---|
| 01 Must Read - Zotero First | 27 | read/import first; core thesis references |
| 02 Subspace Temporal CCA KPCA Methods | 21 | method expansion and implementation references |
| 03 Remote Sensing Change Detection And Applications | 117 | related work and application context |
| 04 Datasets Benchmarks And Data Portals | 23 | dataset/protocol references |
| 05 Implementation Reference Code | 15 | codebases to inspect or reproduce |
| 06 Disaster Damage Civil Infrastructure Future Work | 2 | future damage/application context |
| 07 General ML CV Background | 7 | background only |
| 08 Venues Search Portals And Learning Tools | 57 | search portals, venues, courses, non-paper utility |
| 09 Low Relevance Or Needs Review | 94 | kept for review; not thesis-first |

## Zotero-First Reading Queue

Start here. Some papers appear as arXiv/PDF/IEEE duplicates; import one canonical Zotero item and attach alternate URLs/PDFs if useful.

| priority | item | why |
|---|---|---|
| high | [Deep Learning-Based Change Detection in Remote Sensing: A Comprehensive Review \| IEEE Journals & Magazine \| IEEE Xplore](https://ieeexplore.ieee.org/document/11146768) | change detection survey |
| high | [Hyperspectral change detection using IR-MAD and feature reduction \| IEEE Conference Publication \| IEEE Xplore](https://ieeexplore.ieee.org/document/6048907) | IR-MAD baseline |
| high | [[2305.05813] Change Detection Methods for Remote Sensing in the Last Decade: A Comprehensive Review](https://arxiv.org/abs/2305.05813) | change detection survey |
| high | [[2511.05461] The Potential of Copernicus Satellites for Disaster Response: Retrieving Building Damage from Sentinel-1 and Sentinel-2](https://arxiv.org/abs/2511.05461) | damage dataset / warm extension |
| high | [[1911.09296] xBD: A Dataset for Assessing Building Damage from Satellite Imagery](https://arxiv.org/abs/1911.09296) | damage dataset / warm extension |
| high | [1911.09296](https://arxiv.org/pdf/1911.09296) | damage dataset / warm extension |
| high | [Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf) | damage dataset / warm extension |
| high | [[2409.08563] Second-order difference subspace](https://www.arxiv.org/abs/2409.08563) | core subspace/DS paper |
| high | [2409.08563](https://www.arxiv.org/pdf/2409.08563) | core subspace/DS paper |
| high | [2409.08563](https://arxiv.org/pdf/2409.08563) | core subspace/DS paper |
| high | [Difference Subspace and Its Generalization for Subspace-Based Methods \| IEEE Journals & Magazine \| IEEE Xplore](https://ieeexplore.ieee.org/document/7053916) | core subspace/DS paper |
| high | [Onera Satellite Change Detection Dataset - Rodrigo Caye Daudt](https://rcdaudt.github.io/oscd/) | OSCD / FC-Siamese core benchmark |
| high | [A Comprehensive Review of Remote Sensing and Artificial Intelligence Integration: Advances, Applications, and Challenges](https://www.mdpi.com/1424-8220/25/19/5965) | change detection survey |
| high | [Unsupervised Change Detection in Satellite Images Using Principal Component Analysis and --Means Clustering \| IEEE Journals & Magazine \| IEEE Xplore](https://ieeexplore.ieee.org/document/5196726) | Celik PCA-kmeans baseline |
| high | [[2303.17802] Time-series Anomaly Detection based on Difference Subspace between Signal Subspaces](https://arxiv.org/abs/2303.17802) | core subspace/DS paper |
| high | [Change Detection Methods for Remote Sensing in the Last Decade: A Comprehensive Review](https://www.mdpi.com/2072-4292/16/13/2355) | change detection survey |
| high | [Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks \| IEEE Conference Publication \| IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8518015) | OSCD / FC-Siamese core benchmark |
| high | [[1810.08468] Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks](https://arxiv.org/abs/1810.08468) | OSCD / FC-Siamese core benchmark |
| high | [Fully Convolutional Siamese Networks for Change Detection \| IEEE Conference Publication \| IEEE Xplore](https://ieeexplore.ieee.org/document/8451652) | OSCD / FC-Siamese core benchmark |
| high | [The Regularized Iteratively Reweighted MAD Method for Change Detection in Multi- and Hyperspectral Data \| IEEE Journals & Magazine \| IEEE Xplore](https://ieeexplore.ieee.org/document/4060945) | IR-MAD baseline |
| high | [[2303.17802] Time-series Anomaly Detection based on Difference Subspace between Signal Subspaces](https://web3.arxiv.org/abs/2303.17802) | core subspace/DS paper |
| high | [[2303.17859] MapFormer: Boosting Change Detection by Using Pre-change Information](https://arxiv.org/abs/2303.17859) | prior-guided CD reference |
| high | [[2303.09536] Deep Metric Learning for Unsupervised Remote Sensing Change Detection](https://ar5iv.labs.arxiv.org/html/2303.09536) | modern unsupervised CD pressure |
| high | [wgcban/Metric-CD: Official PyTorch implementation of Deep Metric Learning for Unsupervised Change Detection in Remote Sensing Images](https://github.com/wgcban/Metric-CD) | modern unsupervised CD pressure |
| high | [[2511.05461] The Potential of Copernicus Satellites for Disaster Response: Retrieving Building Damage from Sentinel-1 and Sentinel-2](https://ar5iv.labs.arxiv.org/html/2511.05461) | damage dataset / warm extension |
| high | [[2511.05461] The Potential of Copernicus Satellites for Disaster Response: Retrieving Building Damage from Sentinel-1 and Sentinel-2](https://ar5iv.labs.arxiv.org/html/2511.05461v1) | damage dataset / warm extension |
| high | [[2303.09536] Deep Metric Learning for Unsupervised Remote Sensing Change Detection](https://arxiv.org/abs/2303.09536) | modern unsupervised CD pressure |
| medium | [[2508.16366] Attention Mechanism in Randomized Time Warping](https://arxiv.org/abs/2508.16366) | Sensei/Hiraoka RTW lead; future temporal/subspace method reading |

## Manual Additions After Bookmark Import

These were added after the 2026-06-07 Chrome export. They do not change the raw export counts, but they are included in the organized Chrome import file.

| priority | item | why |
|---|---|---|
| medium | [Signal latent subspace: A new representation for environmental sound classification](https://www.sciencedirect.com/science/article/abs/pii/S0003682X24003323) | Fukui co-authored subspace + DNN latent-feature method; useful future bridge for deep-feature DS/GDS, product Grassmann fusion, and small-sample subspace thinking |

## Implementation References To Keep Visible

| item | why |
|---|---|
| [duncantmiller/ai-developer-resources: I am Duncan, the founder of OpenShiro. This public repository is used as an easy to update list of resources for AI developers including technical courses, books, and tutorials on artificial intelligence, deep learning and machine learning. PRs welcome!](https://github.com/duncantmiller/ai-developer-resources) | code/repository |
| [GitHub - iryna-kondr/scikit-llm: Seamlessly integrate LLMs into scikit-learn.](https://github.com/iryna-kondr/scikit-llm) | code/repository |
| [GitHub - satellite-image-deep-learning/techniques: Techniques for deep learning with satellite & aerial imagery](https://github.com/satellite-image-deep-learning/techniques) | code/repository |
| [SpaceNet · GitHub](https://github.com/spacenetchallenge) | code/repository |
| [GitHub - DIUx-xView/xView2_baseline: Baseline localization and classification models for the xView 2 challenge.](https://github.com/DIUx-xView/xView2_baseline) | code/repository |
| [BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) | code/repository |
| [zhu-xlab/SSL4EO-S12: SSL4EO-S12: a large-scale dataset for self-supervised learning in Earth observation](https://github.com/zhu-xlab/SSL4EO-S12?tab=readme-ov-file) | code/repository |
| [DLR-MF-DAS/SSL4EO-S12-v1.1: resources as addition to the updated version of the SSL4EO-S12 dataset, cf. https://arxiv.org/abs/2503.00168](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1?tab=readme-ov-file) | code/repository |
| [Z-Zheng/ChangeOS: ChangeOS: Building damage assessment via Deep Object-based Semantic Change Detection - (RSE 2021)](https://github.com/Z-Zheng/ChangeOS) | code/repository |
| [torchgeo/torchgeo: TorchGeo: datasets, samplers, transforms, and pre-trained models for geospatial data](https://github.com/torchgeo/torchgeo) | code/repository |
| [r-wenger/MultiSenGE-NA-Tools: Tools for MultiSenGE dataset.](https://github.com/r-wenger/MultiSenGE-NA-Tools) | code/repository |
| [r-wenger/land-use-land-cover-datasets: List of datasets and codes for remote sensing LULC applications.](https://github.com/r-wenger/land-use-land-cover-datasets?tab=readme-ov-file) | code/repository |
| [wgcban/ddpm-cd: Remote Sensing Change Detection using Denoising Diffusion Probabilistic Models](https://github.com/wgcban/ddpm-cd) | code/repository |
| [pallavijain-pj/SenCLIP](https://github.com/pallavijain-pj/SenCLIP) | code/repository |
| [seunghyeokleeme/xBD_road_damage_assessment: [Appl. Sci. 2025, 15(14), 7669] Deep Learning-Based Detection and Assessment of Road Damage Caused by Disaster with Satellite Imagery](https://github.com/seunghyeokleeme/xBD_road_damage_assessment) | code/repository |

## How To Use This

- For Zotero: manually import the `01 Must Read` papers first, then inspect categories `02`, `03`, and `06`.
- For Codex implementation work: use `05 Implementation Reference Code` and the method papers in `02`.
- For thesis related work: prioritize `01`, then surveys and benchmark papers from `03`.
- For cleanup: review `09 Low Relevance Or Needs Review` later before deleting anything from Chrome.

## Import Back To Chrome

Use Chrome Bookmark Manager -> Import bookmarks -> select `docs/source_records/chrome_bookmarks_organized_research_2026-06-07.html`.

## Categorization Notes

- Categorization is heuristic and should be treated as a starting point, not final bibliography truth.
- Search pages and portals are intentionally not marked Zotero-first even when their titles mention important papers.
- Every URL from the raw export is preserved exactly once in the organized import file.
