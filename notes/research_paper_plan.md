# Research Paper Plan

This file is the paper-facing synthesis of the project. It is not a frozen thesis plan. It should evolve when code, experiments, advisor feedback, or literature change the argument.

## 1. Working Title

Interpretable Subspace Priors for Sentinel-2 Change Detection

Alternative titles:

- Spatially Aware Difference Subspace Priors for Multispectral Change Detection
- Evaluating Subspace-Based Change Priors for Supervised Sentinel-2 Binary Segmentation
- When Do Interpretable Unsupervised Priors Help Remote-Sensing Change Segmentation?

## 2. One-Sentence Thesis

This project tests whether Difference Subspace-based unsupervised representations can provide interpretable and spatially meaningful change evidence for supervised Sentinel-2 binary change detection, and what subspace construction is appropriate for satellite imagery.

## 3. Problem Setting

Broader research direction:

```text
interpretable subspace-based change detection for multispectral satellite imagery
```

Input:

```text
pre-event Sentinel-2 image with 13 bands
post-event Sentinel-2 image with 13 bands
```

Output:

```text
binary changed / unchanged map
```

Current benchmark:

```text
OSCD: Sentinel-2 before/after image pairs with binary change labels
```

OSCD is the current implemented benchmark, not a permanent boundary. Future datasets such as Harmonized Sentinel-2 L2A, MultiSenGE, xBD-S12, or domain-specific greenhouse data should be admitted only when the research question, data pipeline, labels or evaluation proxy, and baseline comparisons are clear.

Likewise, `Phase 1` and `Phase 2` are current workflow names, not the paper's fixed conceptual structure. In the paper, describe them by function when possible:

```text
geometric/classical prior generation -> supervised neural segmentation/downstream learning
```

Current task boundary:

- This is binary change detection.
- This is not yet building damage classification.
- This is not yet semantic change detection.
- This is not yet xBD/xBD-S12 damage mapping.

## 4. Motivation

Remote-sensing change detection is useful for urban monitoring, disaster screening, land-cover change, and possible future applications such as abandoned-greenhouse mapping.

The practical difficulty is that numeric image difference is not the same as meaningful change. Shadows, seasonal vegetation, registration error, clouds, sensor effects, and land-cover texture can all create apparent differences.

Deep models can learn strong spatial filters when labels and training domains are sufficient. The reason to study subspace priors is not that DS is assumed better. The reason is to test whether an interpretable geometric representation of pre/post change can:

- expose what kind of change signal is being used;
- act as a useful prior channel for supervised segmentation;
- help diagnose where simple spectral differences fail;
- clarify whether satellite change detection needs spatially aware subspace construction.

Working interpretation:

- Raw spectral difference highlights any numerical band change: real construction, vegetation seasonality, shadow, cloud, registration shift, or sensor effect.
- Global DS asks whether the pre/post images differ along learned subspace directions, but it currently treats pixels as unordered samples.
- Patch-vector or local-window DS asks the same question while preserving local spatial context.
- A DS prior can still hurt fixed-threshold F1 if it highlights real spectral changes that OSCD does not label as target changes. Therefore report both ranking metrics such as AUROC/PR-AUC and threshold metrics such as IoU/F1.

## 5. Research Gap

Established facts:

- DS, GDS, KDS, and KGDS are established subspace methods.
- OSCD and Siamese CNN change detection are established benchmarks/methods.
- Classical unsupervised change detectors such as PCA-diff, CVA, Celik, and IR-MAD are established baselines.
- Neural change detection can be strong and must not be ignored.

Open question for this project:

```text
Does a subspace-based prior provide useful and interpretable evidence for OSCD-style Sentinel-2 change segmentation, and does that require spatially aware subspace construction?
```

The key gap is not "invent DS." The key gap is empirical and methodological: how to adapt subspace representations to multispectral satellite images without losing the spatial structure needed for change maps.

## 6. Current Method

Current global OSCD subspace construction:

```text
one valid pixel = one 13-D vector
vector entries = Sentinel-2 band values at that pixel
one date image = all valid pixel vectors
pre image  -> PCA -> one pre subspace
post image -> PCA -> one post subspace
DS compares the two subspaces
```

Current score:

```text
score_i = || D^T (x_post_i - x_pre_i) ||^2
```

Important limitation:

- PCA fitting treats pixel vectors as an unordered sample set.
- Pixel position is restored only when scalar scores are placed back into the image grid.
- Therefore the current global DS can break spatial information.

Corrected DS status:

- Old residual-stack DS is treated as legacy because it behaved almost like raw spectral L2 on Beirut.
- Canonical/eig DS are the cleaner first-order linear DS paths.
- KDS/KGDS are implemented for the Venus learning audit, not yet as the active OSCD method.

## 7. Proposed Research Path

Immediate method question:

```text
Is global pixel DS enough, or should the project use patch-vector or local-window DS?
```

Variants to compare:

1. Global pixel DS
   - one sample = one 13-band pixel
   - fastest and simplest
   - ignores spatial position during PCA fitting

2. Patch-vector DS
   - one sample = local `3x3x13` or `5x5x13` patch
   - preserves local neighborhood structure
   - higher dimensional and more expensive

3. Local-window DS
   - fit one subspace per local region, such as `128x128`
   - avoids mixing sea, roads, buildings, vegetation, and airport into one global distribution
   - runtime and aggregation must be controlled

4. Kernel/local KDS
   - future nonlinear extension
   - should use sampling, Nyström/prototypes, or local windows because full pixel KPCA is too expensive

5. MultiSenGE GDS/KGDS
   - future multi-date extension
   - useful for temporal difference spaces, but semantic interpretation needs clustering, labels, or weak supervision

## 8. Experiment Plan

Phase 1 audit before more long Phase 2 sweeps:

```text
global pixel DS vs patch-vector DS vs local-window DS
```

Start with:

- `beirut`
- one dense urban city
- one difficult or low-change city

Metrics:

- AUROC
- PR-AUC
- best F1 and IoU over thresholds
- Otsu-threshold F1 and IoU
- correlation with raw spectral L2
- valid-mask exclusion rate for changed pixels
- runtime
- qualitative maps beside pre RGB, post RGB, and ground truth

Decision gates:

- If global pixel DS is competitive and stable, keep it as a spectral-distribution baseline.
- If patch/window DS improves maps or metrics, pivot the main method to spatially aware DS.
- If corrected DS remains weaker than raw/PCA-diff, avoid DS-superiority claims and frame the result as diagnostic or negative evidence.

Phase 2 follow-up only after Phase 1 maps look meaningful:

- raw-only U-Net
- raw + global canonical DS
- raw + best spatially aware DS prior
- raw + PCA-diff / classical prior comparisons
- Siamese raw-only baseline

Reporting constraints from the archive audit:

- record git hash, config, seed, checkpoint, prior root, split, and eval files for every thesis-usable run;
- do not use one-epoch smoke runs as performance evidence;
- do not treat old single-seed artifacts as reproduced evidence;
- do not use runtime claims unless per-method timing is measured cleanly;
- state that CVA and pixel-diff are effectively duplicated L2 baselines in the current code unless corrected.

## 9. Current Evidence

Current trusted evidence:

- OSCD loader, Phase 2 dataset, U-Net forward pass, and CUDA smoke checks work.
- Phase 1 priors and Phase 2 outputs exist and can be inspected.
- Venus KDS/KGDS demo loads Sensei's dataset from `data/venus_tpami2015/`.
- Canonical/eig DS repaired the old residual-stack problem.

Important negative/qualified result:

- The 2026-05-03 3-seed OSCD sweep did not reproduce the old claim that raw+DS beats raw-only.
- `E1_raw_ds` underperformed raw-only.
- `E3_raw_ds_pca` slightly improved threshold metrics but not ranking metrics.
- This means the paper cannot honestly claim simple DS priors reliably improve OSCD segmentation.

## 10. Contribution Framing

Defensible contribution if spatial audit succeeds:

- A careful adaptation and evaluation of spatially aware Difference Subspace priors for Sentinel-2 binary change segmentation.
- Evidence about when global, patch, or window subspace construction preserves useful change information.
- A reproducible two-stage pipeline separating interpretable prior generation from supervised segmentation.

Defensible contribution if spatial audit is negative:

- A rigorous negative result showing that direct global pixel-spectral DS is insufficient for OSCD.
- Evidence that simpler priors or deep baselines outperform paper-faithful global DS.
- A clear methodological lesson: subspace adaptation to satellite imagery must account for spatial context.

Unsafe contribution claims:

- DS is new.
- DS solves change detection.
- DS improves supervised segmentation generally.
- OSCD results prove disaster damage mapping.
- The current project is xBD/xBD-S12 damage segmentation.

## 11. Paper Skeleton

1. Introduction
   - remote-sensing change detection motivation
   - interpretability and pseudo-change problem
   - why subspace priors are worth testing

2. Related Work
   - OSCD and neural/Siamese change detection
   - classical unsupervised change detection
   - DS/GDS/KDS/KGDS and subspace methods
   - prior-guided / interpretable change detection

3. Problem Definition
   - binary pre/post Sentinel-2 change detection
   - OSCD benchmark and labels
   - current exclusion of damage-level claims

4. Method
   - geometric/classical prior generation
   - corrected DS variants
   - global pixel DS
   - spatial alternatives: patch-vector and local-window DS
   - prior-channel neural segmentation

5. Experiments
   - datasets and splits
   - baselines
   - metrics
   - prior-map audit
   - segmentation ablations

6. Results
   - quantitative metrics
   - per-city behavior
   - qualitative maps
   - threshold/ranking disagreement
   - failure cases

7. Discussion
   - whether DS is useful
   - spatial information limitation
   - interpretability value
   - when priors help/hurt

8. Conclusion
   - honest contribution
   - limitations
   - future work: KDS, GDS/KGDS, MultiSenGE, xBD-S12, abandoned greenhouses

## 12. Open Decisions

High priority:

- What spatial subspace construction should become the main method?
- Is rank 6 defensible, or should rank be selected by sensitivity/variance?
- Does valid masking exclude any meaningful changed pixels?
- Should Phase 2 continue only after Phase 1 maps are strong?

Medium priority:

- Should KDS be implemented on OSCD with sampled/global, local, or prototype approximations?
- Should MultiSenGE become the GDS/KGDS experiment after OSCD?
- Which modern unsupervised/supervised baselines are mandatory for a paper?

Low priority for now:

- xBD/xBD-S12 damage pipeline.
- abandoned greenhouse benchmark.
- foundation-model extensions.

## 13. Archive-Derived Guardrails

These are old ideas or risks that should not disappear, but they should not distract the main OSCD work.

Main guardrails:

- The strongest standalone Phase 1 prior is not necessarily the strongest Phase 2 prior, and the reverse is also possible.
- OSCD validation is project-defined from training cities; document the split rather than implying it is official.
- Phase 2 prior channels are continuous maps, not thresholded labels.
- Threshold metrics and ranking metrics answer different questions; report both.
- Probability-map saving or rerun-based threshold sweeps are needed for true threshold tuning.
- `clean_house.ps1 -Aggressive` requires an explicit keep-list and user approval.

Future hooks to keep out of the main claim for now:

- xBD/xBD-S12 damage severity mapping.
- abandoned-greenhouse mapping.
- UAV/edge deployment.
- MCDA/IoT/DMaaS/operational dashboard ideas.
- graph decision layers and broad foundation-model extensions.

## 14. Research-Notes Archive Outcome

The nested `research-notes/` repo was ingested on 2026-06-07. It preserved a broader historical research program, but the current paper plan should not revive that full scope.

What survives as active:

- Interpretable subspace-based change detection for multispectral satellite images.
- OSCD as the current implemented binary change benchmark.
- The immediate need to test spatially aware subspace construction.
- Careful comparison against raw pre/post, PCA-diff, Siamese, and modern CD baselines.

What survives as warm/future:

- xBD-S12 or another Sentinel-scale damage dataset as the cleanest future damage bridge.
- MultiSenGE GDS/KGDS or second-order DS if multi-date evaluation becomes central.
- SSC as a future unsupervised change-type clustering baseline or pseudo-label source.
- Band-group attribution, PCA reconstruction, geodesic/SPD change scores, RTW/SSA/SFA/DMD, and period-subspace DS as future method hooks.
- Local/patch DS, KPCA/KDS, and structured CCA/S3CCA/KCCA are method-development routes only after the spatial-information problem is measured.
- Cold implementation details such as synchronized augmentations, edge-payload formulas, MCDA criteria, and xBD/xView2/xBD-S12 metrics are preserved for future pivots, but they do not define the current paper.

What remains cold/archive:

- Full disaster-damage/land-use/MCDA/UAV/IoT/DMaaS platform framing.
- Infrastructure-placement, evacuation-routing, LLM decision-support, graph decision layers, and broad urban-planning systems.
- Claims that the project already performs damage-level prediction.

Important correction:

- Older `research-notes/` files treated raw+DS improvement as a strong active claim. The newer controlled 3-seed v5 sweep weakened that claim. The current thesis argument must be conditional: DS-style priors are interpretable and sometimes useful, but their benefit depends on subspace construction, spatial context, dataset, city, and metric.
- Older `research-notes/` files also treated residual and eig DS as nearly equivalent in a Phase 1 report. The later corrected-subspace audit found the residual-stack route should be treated as legacy unless the result is reproduced under the repaired implementation.

## 15. File Sources

Use these files when expanding this plan:

- `notes/feedback.md`: advisor and seminar constraints.
- `notes/methods.md`: method understanding.
- `notes/literature.md`: citation and baseline matrix.
- `notes/experiments.md`: current evidence and planned audits.
- `notes/reference_bookmarks.md`: Zotero-first reading queue and organized bookmark map.
- `docs/PROJECT_BRIEF.md`: concise current truth.
- `docs/RUNBOOK.md`: exact commands.
- `docs/experiment_reports/oscd_core_sweep_3seed_150epoch_2026-05-03.md`: accepted sweep summary.
- `docs/source_records/qa_report_response_2025-11-20.pdf`: submitted QA answers; useful for spectral-vs-semantic explanation, MultiSenGE pairing caveats, seasonal-risk framing, and IR-MAD caution.
- `docs/source_records/legacy_cs_seminar_report_ds_damage_segmentation_2025.tex`: earlier seminar-report source; useful as historical writing material, but it overuses old Phase 1/2 and damage-segmentation framing compared with the current truth.
- `docs/source_records/student_feedback_channel4_2025-11-20.xlsx`: original student feedback source; extracted into `notes/feedback.md`.
