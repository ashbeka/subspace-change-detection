# Research Reset Audit

Date: 2026-06-12

This document is a research reset, not a motivational summary. It answers the reset prompt directly and treats the current repo as a useful but imperfect AI-assisted draft.

Evidence labels:

- `[code-evidence]`: supported by inspected code/configs/scripts.
- `[doc-claim]`: claimed by current or historical project notes.
- `[experiment-evidence]`: supported by local outputs/reports/results.
- `[advisor-feedback]`: from Sensei/senpai/seminar notes.
- `[external-source]`: supported by papers/surveys/datasets/resources.
- `[risk]`: plausible weakness or uncertainty.
- `[recommendation]`: proposed next action.

## 1. Student Concern And Reset Purpose

The core concern is valid: the project may have started from "the lab studies subspace methods" plus "satellite change detection is interesting" and then forced a bridge before identifying the actual research gap. [doc-claim]

The reset purpose is to answer:

```text
What real remote-sensing change-detection problem are we solving, and do subspace methods genuinely help?
```

The audit does not assume the answer is yes. A defensible outcome may be:

- spatially aware DS is useful;
- DS is only useful as an interpretable diagnostic;
- DS is weaker than simpler baselines;
- the project should pivot to multi-date, object-level, or semantic/open-vocabulary change detection;
- the thesis becomes an honest empirical study of where subspace priors help or fail.

## 2. Anti-Bias Rules

- Do not defend DS just because it is the lab method. [recommendation]
- Do not treat OSCD as the thesis boundary. OSCD is the current labeled benchmark. [code-evidence]
- Do not treat "prior channel improves U-Net" as true; the current controlled sweep does not support it for DS alone. [experiment-evidence]
- Do not call the current project disaster damage segmentation. xBD/xBD-S12 is not implemented as a damage pipeline. [code-evidence]
- Do not claim novelty from "using PCA/DS/KPCA/classical priors" unless the experiment shows a specific satellite-adaptation contribution. [external-source]
- Negative results are still useful if they clarify subspace construction, spatial information loss, or baseline behavior. [recommendation]

## 3. Current Project Understanding

### 3.1 Code Map

Current implementation:

- `phase1/`: geometric/classical prior generation for OSCD and exploratory MultiSenGE/Venus subspace demos. [code-evidence]
- `phase2/`: supervised OSCD binary segmentation using raw 13-band pre/post Sentinel-2 stacks plus optional prior channels. [code-evidence]
- `project_cli.py`: central command wrapper for readiness checks, Phase 1 prior generation, Phase 1 subspace audits, Venus demo, Phase 2 training/evaluation/sweeps, visualization, and cleanup previews. [code-evidence]
- `tests/`: formula checks for difference subspace and kernel difference subspace; useful for paper-to-code verification, not a complete test suite. [code-evidence]
- `docs/experiment_reports/`: curated experiment report for the 3-seed OSCD sweep. [experiment-evidence]
- `docs/source_records/`: preserved source records, bookmarks, seminar/report artifacts, and imported human-facing material. [code-evidence]
- `notes/`: active research thinking: feedback, methods, literature, experiments, bookmarks, paper plan, and rough notes. [doc-claim]

Main active entry points:

- `python project_cli.py doctor`
- `python project_cli.py list`
- `python project_cli.py phase1-subspace-inspect --city beirut`
- `python project_cli.py phase1-oscd-priors --config canonical`
- `python project_cli.py phase1-venus`
- `python project_cli.py phase2-sweep --preset core --progress-bars`
- raw script equivalents under `phase1/eval/`, `phase1/scripts/`, `phase2/train/`, `phase2/eval/`, and `phase2/scripts/`. [code-evidence]

### 3.2 Implemented Methods

Phase 1 implements or wraps:

- DS projection and cross-residual maps. [code-evidence]
- corrected canonical/projector-eig DS plus legacy residual-stack DS for reproducibility. [code-evidence]
- raw pixel difference / CVA-like difference. [code-evidence]
- PCA-diff. [code-evidence]
- Celik PCA-kmeans. [code-evidence]
- IR-MAD. [code-evidence]
- geodesic/local subspace priors. [code-evidence]
- exploratory MultiSenGE temporal/geodesic tools. [code-evidence]
- Venus KDS/KGDS learning demo. [code-evidence]

Phase 2 implements:

- U-Net.
- Siamese U-Net.
- ResNet-backbone U-Net.
- prior-fusion U-Net.
- OSCD dataset wrapper that loads raw 13-band pre/post data and optional prior maps. [code-evidence]

### 3.3 Actual Data Expectations

OSCD:

- active supervised benchmark. [code-evidence]
- 13 Sentinel-2 bands in rectified `.tif` files. [code-evidence]
- binary pixel labels for changed/unchanged. [code-evidence]
- official 14 train and 10 test image pairs are represented in the loader. [code-evidence]

MultiSenGE:

- exploratory/multi-date route only. [code-evidence]
- not current supervised benchmark. [risk]

xBD/xBD-S12:

- only a future damage template/config path exists. [code-evidence]
- no integrated damage trainer/evaluator/multi-class building-damage metrics exist. [code-evidence]

Venus `.mat` files:

- learning/reference audit for TPAMI-style KDS/KGDS, not a satellite dataset. [code-evidence]

## 4. Source Coverage Check

The final source batch is preserved in:

```text
docs/source_records/final_organization_2026-06-12/
```

| Source | Status | Active ingestion |
|---|---|---|
| `apple_notes.md` | covered | `notes/my_notes.md`, `notes/feedback.md`, `notes/methods.md`, `notes/experiments.md`, `notes/research_paper_plan.md` |
| `slack_messages_with_sensei.md` | covered | advisor feedback, DS/KDS/GDS context, reset questions |
| `slack_notes_to_myself.md` | covered | problem-driven framing, pseudo-change, presentation and experiment notes |
| `All research notes...docx` | covered | methods, literature, experiments, paper plan |
| `Geometry of subspace set...2024.pdf` | covered | subspace-method family, DS/GDS/KDS/KGDS/Grassmann context |
| `long_prompt.md` | covered | note workflow, anti-slop, research reset framing |
| `Safari links.md` | covered | merged into organized Chrome bookmark hierarchy |
| `all_bookmarks_6_12_26.html` | covered | merged with Safari links, all URLs preserved |
| `image1.jpeg`, `image2.jpeg` | covered as support only | Apple Notes attachments for spatial/Green Learning/SSC ideas |
| `prior_ingestion_ledgers.md` | historical only | provenance, not active reading path |

Apple Notes themes that are now represented:

- PixelHop / Green Learning / wavelets. [doc-claim]
- spatial information loss. [advisor-feedback]
- pseudo-change vs real change. [doc-claim]
- KPCA/KDS. [advisor-feedback]
- CCA/KCCA/S3CCA/TRCCA. [advisor-feedback]
- DS/GDS/KGDS/geodesic. [advisor-feedback]
- MultiSenGE/Harmonized Sentinel-2. [advisor-feedback]
- greenhouse mapping. [doc-claim]
- band-combination experiments. [doc-claim]
- building/object-level descriptors. [doc-claim]
- "am I forcing subspace methods?" [doc-claim]

Remaining unresolved source issue:

- `final_organizing_task_patch/` remains untracked and undeleted until the user explicitly approves deletion. [risk]

## 5. Bookmark And Literature Resource Map

Bookmark QA status:

- Chrome input bookmarks: `1060`. [code-evidence]
- Safari input links merged: `496`. [code-evidence]
- organized output entries: `1556`. [code-evidence]
- unique URLs: `1537`. [code-evidence]
- duplicate URL entries preserved: `19`. [code-evidence]
- Research entries after final false-positive cleanup: `459`. [code-evidence]
- To Review entries: `150`. [code-evidence]

Importable Chrome file:

```text
docs/source_records/final_organization_2026-06-12/chrome_bookmarks_organized_all_2026-06-12.html
```

Research reading order:

1. `Research / 01 Read First - Thesis Core`
2. `Research / 02 Methods - Subspace Geometry`
3. `Research / 03 Methods - Change Detection`
4. `Research / 04 Datasets - Current Candidate Future`
5. `Research / 05 Applications - Use Cases And Problem Framing`
6. `Research / 06 Background - Read As Needed`
7. `Research / 07 Literature Workflow - Search Venues Writing`
8. `Research / 08 Tools And Code - Implementation Support`
9. `Research / 09 Parking Lot - Lower Priority`

False positives removed from `Research` included personal searches, app-development examples, generic LSTM tutorials, generic ML-roadmap material, and voice-AI links. All URLs were preserved. [code-evidence]

The concept-to-reading bridge is maintained in `notes/literature.md`. [doc-claim]

## 6. Literature-Grounded Problem Map

### 6.1 Remote-Sensing Change Detection Is Broad And Competitive

Recent reviews place change detection across supervised, semi-supervised, weakly supervised, unsupervised, self-supervised, multimodal, and foundation-model paradigms. This means a DS project cannot claim novelty merely from producing a change map. [external-source]

Sources:

- Wang et al. 2024, "Advances and Challenges in Deep Learning-Based Change Detection for Remote Sensing Images": https://www.mdpi.com/2072-4292/16/5/804
- Parelius 2023 review of DL-based RS change detection: https://www.mdpi.com/2072-4292/15/8/2092
- 2026 Artificial Intelligence Review survey: https://link.springer.com/article/10.1007/s10462-026-11501-0

Implication:

- `[risk]` A simple "DS prior + U-Net" claim is too weak unless tied to spatial information, label efficiency, interpretability, or a clearly defined application.

### 6.2 Label Scarcity Is Real, But Not Unique To This Project

Label-efficient CD is now an active field, with semi-supervised, weakly supervised, self-supervised, active-learning, few-shot, and unsupervised methods. [external-source]

Sources:

- Toward Label-Efficient Deep Learning Change Detection: https://onlinelibrary.wiley.com/doi/10.1111/phor.70021
- Sample-efficient DL CD survey: https://arxiv.org/html/2502.02835v1

Implication:

- `[recommendation]` If the thesis uses "label-efficient" framing, compare against at least one modern label-efficient pressure source in related work, even if not reimplemented.

### 6.3 Pseudo-Change Is A Real Problem

Remote-sensing changes include relevant changes and apparent/irrelevant changes due to seasonality, illumination, clouds, registration, sensor differences, and radiometry. [external-source]

IR-MAD/MAD sources explicitly warn about georeferencing, clouds/water, and multivariate radiometric issues. [external-source]

Sources:

- Google Earth Engine iMAD tutorial: https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt1
- IRMAD service specification: https://docs.mcube.terradue.com/services/irmad/service-specs/

Implication:

- `[recommendation]` Pseudo-change diagnosis may be a stronger problem statement than "DS improves segmentation."

### 6.4 Classical Change Detection Already Includes PCA, KPCA, MAD, CCA

PCA, KPCA, CCA/MAD, and IR-MAD are already established in remote-sensing CD. Nielsen/Canty KPCA CD explicitly uses the kernel trick for nonlinear change detection and notes memory-based projection requirements. [external-source]

Sources:

- Kernel PCA for change detection: https://www2.imm.dtu.dk/pubdb/edoc/imm5667.pdf
- iMAD/MAD tutorial: https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt1

Implication:

- `[risk]` Implementing KPCA/KDS is important for lab understanding, but not automatically novel. The novelty must be in satellite-specific subspace construction, evaluation, or interpretation.

### 6.5 Semantic And Open-Vocabulary Change Detection Is Moving Fast

Newer papers target text-driven/object-specific/open-vocabulary change detection, often because users care about specific object changes and paired labels are costly. [external-source]

Sources:

- Text-Driven Change Detection, CVPRW 2026: https://openaccess.thecvf.com/content/CVPR2026W/MORSE/html/Saha_Open-Vocabulary_Change_Detection_via_Synthetic_Examples_and_Grounded_Visual_Models_CVPRW_2026_paper.html
- Weak Temporal Supervision, 2026: https://arxiv.org/html/2601.02126v1
- Seg2Change, 2026: https://arxiv.org/html/2604.11231v1

Implication:

- `[risk]` If the thesis tries to claim semantic/object-level novelty, DS must be compared against this direction or clearly stay in a narrower binary/interpretable-prior scope.

### 6.6 Greenhouse Mapping Is Real But Not Automatically Change Detection

Sentinel-2 greenhouse mapping exists, including global plastic-covered greenhouse mapping and spectral index approaches. [external-source]

Sources:

- Global-PCG-10, 10 m global Sentinel-2 greenhouse map: https://essd.copernicus.org/articles/17/5065/2025/essd-17-5065-2025.html
- Sentinel Hub Plastic Greenhouse Index: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/plastic_greenhouse/
- Two-temporal Sentinel-2 greenhouse mapping: https://www.mdpi.com/2072-4292/13/14/2820
- WUN abandoned greenhouse project: https://wun.ac.uk/wun/research/view/mapping-abandoned-greenhouses-multimodal-fusion/

Implication:

- `[recommendation]` Greenhouses are a plausible application/use case, but first decide if the task is mapping, abandonment classification, temporal monitoring, or changed-area detection.

## 7. Critique Of Current Direction

The current direction is partly defensible and partly forced.

Defensible:

- OSCD gives a controlled binary change benchmark. [external-source]
- The repo can produce classical priors and train segmentation models. [code-evidence]
- Sensei's spatial-information concern maps directly to a real methodological question: global pixel DS vs spatially aware DS. [advisor-feedback]
- Corrected canonical/eig DS exists and avoids the legacy residual-stack issue. [code-evidence]

Forced or weak:

- The old story implied DS priors improve segmentation, but the controlled E1 raw+DS sweep does not support that. [experiment-evidence]
- OSCD has only two dates per location, so GDS/KGDS are not naturally supported unless the dataset changes. [risk]
- The current global pixel subspace treats pixel spectra as unordered samples, so it can ignore roads/buildings/sea/airport position while fitting PCA. [advisor-feedback]
- xBD/damage claims are not supported by current code. [code-evidence]
- KPCA/KDS implementation alone is not novel because KPCA/MAD/CCA are established in remote-sensing CD. [external-source]

## 8. Supported vs Unsupported Claims

Supported claims:

- `[code-evidence]` The active code implements OSCD binary change segmentation with optional prior channels.
- `[code-evidence]` Phase 1 can generate DS, pixel/CVA, PCA-diff, Celik, IR-MAD, and geodesic prior maps.
- `[code-evidence]` The current OSCD DS sample is a valid pixel represented by 13 Sentinel-2 band values.
- `[code-evidence]` Old `residual` DS is preserved as `legacy_residual_stack`; canonical/eig DS are available.
- `[experiment-evidence]` The v5 3-seed sweep shows raw+DS did not improve over raw-only.
- `[experiment-evidence]` DS+PCA slightly improved threshold IoU/F1 on average but not AUROC/PR-AUC in the v5 sweep.
- `[advisor-feedback]` Sensei agrees the algorithm can make a subspace but warns that it can break spatial information.

Unsupported claims:

- `[risk]` DS priors reliably improve segmentation.
- `[risk]` The current repo solves disaster damage assessment.
- `[risk]` OSCD binary performance proves building damage or greenhouse usefulness.
- `[risk]` rank 6 is optimal.
- `[risk]` global pixel DS is the correct subspace construction for satellite imagery.
- `[risk]` KDS/KGDS are implemented in the active satellite pipeline.
- `[risk]` current priors are label-efficient in the same sense as modern label-efficient CD literature.

## 9. Ranked Candidate Research Problems

### Rank 1: Spatially Aware DS For Multispectral Binary Change Detection

Research question:

```text
Does spatially aware subspace construction improve the usefulness and interpretability of DS-based change priors for multispectral pre/post satellite imagery?
```

Motivation:

- directly answers Sensei's spatial-information warning. [advisor-feedback]
- uses current OSCD code and labels. [code-evidence]
- feasible under deadline. [recommendation]

Contribution:

- compare global pixel DS, patch-vector DS, and local-window DS against raw L2, PCA-diff, and supervised baselines.

Risks:

- may not outperform raw/PCA/U-Net. [risk]
- local/window DS may be slower and noisier. [risk]

Existing repo reuse:

- OSCD loader, canonical/eig DS, Phase 1 metrics, Phase 2 dataset/model/eval. [code-evidence]

### Rank 2: Empirical Benchmark Of Interpretable Classical Priors For OSCD

Research question:

```text
When do interpretable classical priors help or hurt supervised Sentinel-2 binary change segmentation?
```

Motivation:

- current evidence already shows priors have mixed effects. [experiment-evidence]
- useful even if DS does not win. [recommendation]

Contribution:

- rigorous ablation of raw, DS, PCA-diff, CVA, Celik, IR-MAD, geodesic/local priors, and simple fusion.

Risks:

- may be less novel methodologically. [risk]
- must compare enough baselines to be credible. [risk]

### Rank 3: Pseudo-Change Versus Real-Change Diagnosis

Research question:

```text
Can subspace/classical priors help diagnose apparent spectral change versus labeled meaningful change?
```

Motivation:

- pseudo-change is a real field problem. [external-source]
- explains why priors can hurt threshold metrics while still revealing useful signal. [experiment-evidence]

Contribution:

- visual/quantitative taxonomy of shadows, seasonality, registration, sensor effects, vegetation, construction.

Risks:

- needs case annotations or careful qualitative protocol. [risk]

### Rank 4: KPCA/KDS Nonlinear Spectral Change Representation

Research question:

```text
Does nonlinear kernel subspace comparison improve change-map quality over linear DS/PCA-diff for multispectral images?
```

Motivation:

- aligns with Sensei's TPAMI/KDS request. [advisor-feedback]
- relevant to nonlinear spectral relationships. [external-source]

Risks:

- KPCA for remote-sensing CD already exists. [external-source]
- full pixel KPCA is memory-heavy. [risk]
- needs sampling/Nyström/local windows. [recommendation]

### Rank 5: Multi-Date DS/GDS/KGDS Using Harmonized Sentinel-2 Or MultiSenGE

Research question:

```text
Can multi-date subspace trajectories reveal temporal change, recovery, or anomaly patterns better than pairwise differences?
```

Motivation:

- GDS/KGDS are naturally multi-subspace methods. [external-source]
- Sensei suggested time-sequential satellite data. [advisor-feedback]

Risks:

- labels/evaluation proxy are unclear. [risk]
- data acquisition/cloud filtering/time-step consistency must be solved first. [risk]

### Rank 6: Object-Level Greenhouse/Building Change Pivot

Research question:

```text
Can subspace/object descriptors help monitor greenhouse or building state changes with limited labels?
```

Motivation:

- greenhouse mapping is a real application. [external-source]
- object units may preserve spatial/semantic meaning better than global pixels. [recommendation]

Risks:

- new dataset/polygons/evaluation required. [risk]
- likely not feasible as the immediate two-week path. [risk]

### Rank 7: Foundation-Model/Open-Vocabulary CD Pivot

Research question:

```text
Can remote-sensing foundation-model features replace or augment handcrafted subspace priors for semantic/object-specific change?
```

Motivation:

- very current literature. [external-source]

Risks:

- may move too far from lab subspace identity. [risk]
- high implementation complexity. [risk]

### Rank 8: Building-Level Damage Mapping Pivot

Research question:

```text
Can subspace descriptors support building damage-level assessment from pre/post imagery?
```

Motivation:

- xBD/xBD-S12 are relevant. [external-source]

Risks:

- current repo does not implement this. [code-evidence]
- damage metrics and object-level labels are a separate thesis pipeline. [risk]

## 10. Conservative / Stronger / Pivot Thesis Framings

### A. Conservative Framing

Title:

```text
Evaluating Spatially Aware Subspace Priors for Sentinel-2 Binary Change Segmentation
```

Research question:

```text
Do global, patch-based, or local-window Difference Subspace priors provide useful interpretable change evidence for OSCD binary segmentation?
```

Contribution:

- careful implementation/audit of DS variants.
- comparison against raw L2/CVA, PCA-diff, and U-Net/Siamese baselines.
- honest negative or positive result.

Why it is safe:

- reuses current code and OSCD labels. [code-evidence]
- directly addresses advisor feedback. [advisor-feedback]
- can be completed with limited time. [recommendation]

### B. Stronger Research Framing

Title:

```text
Spatially Preserving Subspace Representations for Interpretable Multispectral Change Detection
```

Research question:

```text
Which subspace construction preserves spatial structure while keeping the interpretability of classical change priors?
```

Contribution:

- global pixel vs patch-vector vs local-window vs maybe feature-subspace comparison.
- pseudo-change analysis.
- possible Phase 2 prior-channel follow-up.

Why it is stronger:

- a clearer methodological question than "add DS channel."
- supports paper-style contribution if evidence is good. [recommendation]

Risk:

- requires implementation and visual/metric evidence beyond current code. [risk]

### C. Pivot Framing

Title:

```text
From Spectral Priors To Semantic Change: A Diagnostic Study Of Classical And Learned Representations
```

Research question:

```text
When do classical spectral/subspace priors fail to match semantic change labels, and what representation is needed instead?
```

Contribution:

- negative result plus pivot toward object/foundation/semantic representations.

Why it may be needed:

- if spatial DS and KDS remain weaker than simple baselines. [risk]

## 11. Recommended Thesis Framing

Recommended current framing:

```text
Spatially Aware Difference Subspace Priors for Interpretable Sentinel-2 Change Detection
```

Recommended research question:

```text
Can Difference Subspace-based priors produce useful and interpretable changed-area evidence for multispectral pre/post satellite images, and does preserving spatial context through patch or local-window subspaces improve over global pixel subspaces?
```

Why this is the best current framing:

- It does not assume DS works. [recommendation]
- It answers Sensei's critique directly. [advisor-feedback]
- It uses the current repo without claiming damage mapping. [code-evidence]
- It admits negative results. [recommendation]
- It leaves room for KDS/GDS/foundation/object pivots later. [recommendation]

## 12. Evaluation Of The Label-Efficient Prior Framing

Original proposed framing:

```text
Label-efficient and interpretable satellite image change detection using subspace representations as classical priors, compared against PCA-diff and U-Net baselines.
```

Verdict:

- Salvageable, but too broad. [risk]
- "Label-efficient" is risky unless the experiment actually reduces labels, uses weak supervision, or compares to label-efficient literature. [external-source]
- "subspace representations as classical priors" is valid but too vague. [risk]
- "compared against PCA-diff and U-Net" is necessary but not sufficient; raw L2/CVA, IR-MAD, Siamese, and at least one modern comparison pressure should be discussed. [recommendation]

Sharper version:

```text
Spatially Aware Subspace Priors for Label-Constrained Sentinel-2 Change Segmentation
```

Sharper research question:

```text
When training labels are limited or expensive, can interpretable subspace-derived prior maps improve or explain supervised Sentinel-2 binary change segmentation, and which subspace construction avoids losing spatial context?
```

Condition:

- Only use this title if experiments include a label-fraction or label-limited setting. Otherwise remove "label-constrained." [recommendation]

Backup title:

```text
An Empirical Study of Subspace-Derived Change Priors for Sentinel-2 Binary Change Detection
```

## 13. Minimum Viable Thesis Path

Smallest research question worth answering:

```text
Is global pixel DS an appropriate subspace construction for OSCD-style Sentinel-2 change maps, or do patch/local subspaces produce more spatially meaningful and competitive priors?
```

Minimum datasets:

- OSCD only. [recommendation]
- Optional small Harmonized Sentinel-2/MultiSenGE feasibility check as future work, not thesis core. [recommendation]

Minimum methods:

- raw spectral L2 / CVA.
- PCA-diff.
- corrected canonical/eig DS.
- patch-vector DS or local-window DS.
- raw U-Net.
- Siamese raw-only if Phase 2 included.
- raw + best prior only after Phase 1 maps are meaningful.

Minimum metrics:

- AUROC.
- PR-AUC.
- best F1/IoU across thresholds.
- Otsu F1/IoU.
- correlation with raw L2.
- runtime.
- valid-mask changed-pixel exclusion rate.

Required figures:

1. Pipeline figure: pre/post Sentinel-2 -> subspace prior -> segmentation/evaluation.
2. Subspace sample-definition diagram: global pixel vs patch/local.
3. OSCD qualitative maps for 3 cities: pre RGB, post RGB, GT, raw L2, PCA-diff, global DS, patch/local DS.
4. Failure-case figure: pseudo-change or city where DS hurts.
5. If Phase 2 included: prediction comparison raw U-Net vs best prior model.

Required tables:

1. Method matrix: samples, features, spatial preservation, output, complexity.
2. Phase 1 metric table by method/city.
3. Phase 2 metric table if used.
4. Ablation table: rank/window/patch size.
5. Safe claims vs forbidden claims.

Success result:

- Patch/local DS improves over global pixel DS and is competitive with raw L2/PCA-diff on maps or metrics, with interpretable visual behavior. [recommendation]

Acceptable negative result:

- Global DS and spatial variants do not outperform simple baselines, but the thesis clearly shows why: spatial information, label mismatch, pseudo-change, or subspace construction limitations. [recommendation]

## 14. Experiment Checklist

Immediate:

- implement or finish `compare_oscd_spatial_subspaces.py`. [recommendation]
- compare global pixel DS, patch-vector DS, and local-window DS. [recommendation]
- include raw L2/CVA and PCA-diff. [recommendation]
- run on Beirut plus two contrasting cities. [recommendation]
- save maps and metrics in timestamped outputs. [recommendation]

Then:

- rank sensitivity: `2, 3, 4, 5, 6, 8, 10, 12`. [recommendation]
- valid-mask changed-pixel exclusion audit. [recommendation]
- DS projection-back visualization. [recommendation]
- scalar score variant audit: squared norm, norm, normalized ratio, residual. [recommendation]
- if Phase 1 is promising, run a small Phase 2 follow-up with raw-only vs best prior. [recommendation]

Defer:

- full KDS/KGDS satellite pipeline. [recommendation]
- MultiSenGE/Harmonized Sentinel-2 GDS until dataset feasibility is known. [recommendation]
- xBD/xBD-S12 damage pipeline. [recommendation]
- foundation/open-vocabulary pivot. [recommendation]

## 15. Figures And Tables Needed

Priority figures:

1. Sample definition: TPAMI Venus vs OSCD global pixel vs patch/local satellite samples.
2. Spatial DS audit maps for 3 cities.
3. Rank/window sensitivity curves.
4. Pseudo-change examples.
5. Phase 2 prediction maps only if a small supervised follow-up is run.

Priority tables:

1. Code truth table: implemented vs planned.
2. Literature positioning table: classical, DL, label-efficient, semantic/open-vocabulary, subspace.
3. Method comparison table: raw L2, PCA-diff, global DS, patch/local DS, IR-MAD, U-Net/Siamese.
4. Thesis claim table: supported, unsupported, forbidden.

## 16. Safe Claims And Forbidden Claims

Safe claims:

- The repo currently studies OSCD Sentinel-2 binary change detection. [code-evidence]
- The current global DS treats each valid pixel as a 13-D vector and builds one subspace per date image. [code-evidence]
- Sensei's spatial-information warning is a central research concern. [advisor-feedback]
- Controlled DS-prior segmentation results are mixed/negative for DS alone. [experiment-evidence]
- The next defensible method step is spatially aware DS evaluation. [recommendation]

Forbidden claims:

- DS is a new method invented here. [external-source]
- DS already improves OSCD segmentation reliably. [experiment-evidence]
- The project has completed disaster damage segmentation. [code-evidence]
- xBD/xBD-S12 damage evaluation is implemented. [code-evidence]
- KPCA/KDS is novel merely because it is kernelized. [external-source]
- OSCD binary results prove greenhouse, semantic, or building damage performance. [risk]

## 17. Repo Cleanup Plan

No cleanup is performed in this pass.

| Component | Action later | Reason | Risk |
|---|---|---|---|
| `phase1/` | keep/rewrite selectively | active prior generation and subspace code | old names like "phase" may become misleading |
| `phase2/` | keep/rewrite selectively | active segmentation code and results | priors may still point to legacy residual maps |
| `project_cli.py` | keep | useful command surface | must remain current as scripts change |
| `notes/` | keep | active research thinking | avoid growth into another archive |
| `docs/source_records/` | keep | provenance/source records | do not treat as truth |
| `docs/experiment_reports/` | keep | curated results | should not duplicate raw outputs |
| `phase1/outputs/`, `phase2/outputs/` | keep ignored/documented | raw artifacts/evidence | clutter if not summarized |
| `references/` | keep for now | paper-to-code/reference-code value | may be large; needs inventory before deletion |
| `research-notes/` | delete/archive only after user approval | already ingested, separate repo history exists | deletion anxiety; verify Git remote first |
| `final_organizing_task_patch/` | delete only after user approval | untracked source folder already preserved | user wants final confirmation first |
| `tests/` | keep formula tests | guards against paper-inaccurate implementations | not a full product test suite |

## 18. Immediate 7-Day Action Plan

Day 1:

- finalize this audit and review with user. [recommendation]
- decide exact cities for spatial DS audit. [recommendation]

Day 2:

- implement `phase1/scripts/compare_oscd_spatial_subspaces.py`. [recommendation]
- start with global pixel DS, patch `3x3`, patch `5x5`, local window `128`. [recommendation]

Day 3:

- run Beirut plus two more cities. [recommendation]
- generate maps and `metrics.csv`. [recommendation]

Day 4:

- rank/window sensitivity. [recommendation]
- valid-mask exclusion audit. [recommendation]

Day 5:

- write a short experiment report under `docs/experiment_reports/`. [recommendation]
- update `notes/experiments.md` and `notes/research_paper_plan.md`. [recommendation]

Day 6:

- decide Phase 2 follow-up: no training, small smoke, or controlled short run. [recommendation]

Day 7:

- prepare Sensei update:
  - what problem was clarified;
  - what subspace constructions were compared;
  - whether spatial preservation helped;
  - what next dataset/method is justified. [recommendation]

## 19. Open Questions For Sensei

1. Is "changed area detection" the right task, or should the problem become object/state change such as greenhouse/building monitoring? [advisor-feedback]
2. For satellite imagery with one image per date, is a subspace built from pixel vectors a reasonable adaptation, or should samples be patches/local windows/date sequences? [advisor-feedback]
3. If spatial information is central, should the lab prefer patch-vector DS, local-window DS, CCA/S3CCA, or a temporal subspace method? [advisor-feedback]
4. Is Harmonized Sentinel-2 L2A the preferred next dataset for multi-date GDS/KGDS, and what region/time cadence should be tested first? [advisor-feedback]
5. Should KDS/KGDS be pursued for raw spectral vectors, patch features, or learned remote-sensing features? [advisor-feedback]

## Bottom Line

The project is not dead, but the old story is too weak. The strongest current path is:

```text
spatially aware subspace priors for interpretable Sentinel-2 changed-area detection
```

The next experiment must answer Sensei's concern directly:

```text
global pixel DS vs patch-vector DS vs local-window DS
```

If spatially aware DS helps, the thesis has a coherent methodological contribution. If it does not, the thesis can still be valid as a careful negative/diagnostic study or it should pivot to multi-date, object-level, or semantic/foundation-model change detection.
