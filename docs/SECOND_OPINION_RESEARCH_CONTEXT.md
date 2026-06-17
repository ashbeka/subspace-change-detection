# Second-Opinion Research Context

Purpose: give Claude, Gemini, another Codex thread, or a human reviewer enough context to critique this research direction without inheriting our old assumptions.

This is not a thesis chapter. It is a neutral briefing for an external second opinion.

## 1. Anti-Bias Rules

- Do not assume Difference Subspace (DS) is useful just because the lab studies subspace methods.
- Do not assume OSCD is the final thesis dataset. It is the current labeled benchmark.
- Do not assume `phase1/` and `phase2/` are the final research structure. They are old/current implementation folders.
- Do not call the project disaster damage assessment unless xBD/xBD-S12-style damage labels and metrics are actually implemented.
- Do not treat generated documents or old experiment summaries as ground truth.
- A negative result is acceptable if it identifies why subspace construction fails or where it helps.

## 2. Current Working Problem

The research problem to critique right now is:

```text
Can spatially aware Difference Subspace construction preserve the spatial structure of multispectral Sentinel-2 images well enough to produce interpretable changed-area evidence, and where does it help or fail compared with raw spectral difference, PCA-diff, Celik/IR-MAD, and neural change-detection baselines?
```

This problem was chosen because Sensei warned that the current subspace construction can build a subspace but may break spatial information. The immediate question is whether spatially aware sample construction fixes that in any measurable or interpretable way.

## 3. Current Project Truth

Implemented benchmark:

- OSCD Sentinel-2 binary change detection.
- Inputs are pre-change and post-change Sentinel-2 images with 13 bands each.
- Current supervised output is a binary changed/unchanged map.

Current implemented pipeline:

```text
pre/post Sentinel-2 bands -> classical/geometric prior maps -> optional supervised segmentation
```

Implemented method families:

- raw spectral difference / CVA-style maps;
- PCA-diff;
- Celik PCA-kmeans;
- IR-MAD;
- canonical/projector DS and legacy residual-stack DS;
- exploratory geodesic/local prior variants;
- U-Net and Siamese U-Net binary segmentation with optional prior channels;
- Venus KDS/KGDS learning demos.

Not implemented as evidence:

- completed xBD/xBD-S12 damage assessment;
- building damage-level classification;
- semantic change detection;
- full MultiSenGE/Harmonized Sentinel-2 evaluation;
- paper-faithful OSCD KDS/KGDS as the active benchmark method;
- proof that DS improves deep learning.

## 4. Current Evidence

Evidence that matters:

- OSCD data and 13-band `.tif` inputs load locally.
- Phase 2 raw input uses 26 channels: 13 pre + 13 post.
- Optional prior channels can be appended.
- A controlled 3-seed U-Net sweep did not reproduce the old raw+DS improvement claim; raw+DS underperformed raw U-Net, while raw+DS+PCA had a small IoU/F1 improvement but not AUROC/PR-AUC.
- The old residual-stack DS behaved almost like raw spectral L2 and is now treated as legacy.
- A spatial-subspace pilot showed patch-vector DS changes behavior and improves over global pixel DS, but PCA-diff/raw L2 still outperform DS-family maps overall.

Interpretation:

- There is evidence that subspace sample construction matters.
- There is not yet evidence that DS is a strong OSCD detector.
- The thesis can still be valid if it becomes a rigorous spatial-subspace diagnostic study rather than an overclaimed method paper.

## 5. Advisor And Lab Constraints

Sensei's key feedback:

- Clarify the purpose and problem of the research.
- DS can make a subspace, but global pixel construction may break spatial information.
- Understand how subspaces are generated from multi-channel images.
- Run and understand nonlinear DS/KDS/KGDS from the TPAMI Venus setting.
- Ask how sample definition should change: whole image views, pixels, local patches, multiple dates, or something else.

Lab alignment:

- The lab has strong interest in subspace methods, DS/GDS/KDS/KGDS, canonical angles, Grassmann geometry, and hybrid geometric/deep learning pipelines.
- A thesis is stronger if it makes the subspace construction question explicit instead of simply appending DS to an unrelated remote-sensing task.

## 6. Candidate Research Routes

### 6.1 Main Recommended Route: Spatially Aware DS

Question:

```text
Does preserving local spatial context in the subspace sample definition make DS more meaningful for Sentinel-2 changed-area detection?
```

Experiment track:

```text
global pixel DS -> patch-vector DS -> local-window DS -> multiscale spatial subspace pyramid -> fair classical baselines -> optional neural/prior follow-up
```

Why it is defensible:

- It directly answers Sensei's spatial-information criticism.
- It can be tested immediately on OSCD labels.
- It does not require claiming DS beats deep learning.
- Negative results are still informative.

Risk:

- Spatial subspace ideas already exist in remote sensing and hyperspectral literature, so novelty must be specific: a DS-style spatial-support construction/evaluation for Sentinel-2 change evidence, not "first spatial subspace method."

### 6.2 Empirical Diagnostic Study

Question:

```text
When do classical/subspace prior maps help, fail, or merely duplicate raw spectral difference in Sentinel-2 change detection?
```

This is safer if DS does not win. It can compare raw L2/CVA, PCA-diff, Celik, IR-MAD, DS variants, and U-Net/Siamese baselines city by city.

### 6.3 Hybrid Geometry Plus Neural Learning

Question:

```text
Can geometric maps or subspace descriptors complement neural change detectors as priors, auxiliary targets, region descriptors, or error diagnostics?
```

Do this only after geometry-only maps are understood. Otherwise the neural model may hide the failure of the geometric method.

### 6.4 Multi-Date GDS/KGDS

Question:

```text
Can GDS/KGDS describe temporal change trajectories across more than two Sentinel-2 dates?
```

This is lab-aligned but needs a dataset and evaluation protocol. OSCD is only pre/post and is therefore more naturally DS/KDS than GDS/KGDS.

### 6.5 Semantic/Object-Level Pivot

Question:

```text
Can subspace descriptors explain object-level or semantic change after a detector localizes changed regions?
```

Possible applications: greenhouses, buildings, xBD/xBD-S12, object-level urban change. Needs labels or credible manual validation.

## 7. Resources Available

Active notes:

- `notes/feedback.md`: advisor/senpai questions and implied tasks.
- `notes/methods.md`: method explanations and subspace construction cards.
- `notes/experiments.md`: experiment backlog and evidence gates.
- `notes/literature.md`: literature map and concept-to-reading table.
- `notes/research_paper_plan.md`: thesis framing and contribution lanes.
- `notes/reference_bookmarks.md`: organized bookmark import and Zotero triage.

Project docs:

- `docs/PROJECT_BRIEF.md`: current short truth status.
- `docs/RESEARCH_RESET_AUDIT.md`: detailed critical reset audit.
- `docs/RUNBOOK.md`: commands.

Resource indexes:

- `references/reference_papers/REFERENCE_RESOURCE_INDEX.md`
- `references/REFERENCE_CODE_INVENTORY.md`

Bookmark export:

- `docs/source_records/final_organization_2026-06-12/chrome_bookmarks_organized_all_2026-06-17.html`

## 8. Questions For A Second-Opinion LLM

1. Is the working problem a real remote-sensing problem, or is it still method-forcing?
2. Is "spatially aware DS for multispectral changed-area evidence" narrow enough for a master's thesis?
3. What exact baselines are necessary to make the result credible?
4. What literature would most threaten the novelty of spatial DS?
5. If DS loses to PCA-diff and IR-MAD, what thesis framing remains defensible?
6. Should the project stay on OSCD first, or pivot immediately to multi-date Sentinel-2, greenhouse mapping, or semantic/object change?
7. Is the contribution more likely to be method, interpretability, empirical diagnostic, label efficiency, or negative-result analysis?
8. Which experiment should be run first to most quickly falsify the current direction?

## 9. Desired Output From The Second Opinion

Ask the external reviewer or LLM to produce:

- a concise critique of the current problem statement;
- a ranked list of thesis routes;
- must-read papers not already represented in the resource map;
- minimum experiments and baselines;
- highest-risk unsupported claims;
- a recommendation to continue, narrow, or pivot.

The reviewer should be invited to disagree. The point is not to protect the current DS idea; it is to find the most defensible research path.

## 10. What The Project Should Do Next

Immediate next work:

1. Run the spatial-subspace comparison on multiple OSCD cities.
2. Compare against raw L2/CVA and PCA-diff immediately.
3. Add Celik and IR-MAD only after confirming formula and implementation behavior.
4. Inspect qualitative maps for pseudo-change, vegetation, shadows, water, registration, and city-specific failure modes.
5. Decide whether spatial DS is a main thesis path, a diagnostic negative result, or a reason to pivot.

