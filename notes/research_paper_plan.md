# Research Paper Plan

## Table Of Contents

- [1. Working Title](#1-working-title)
- [2. One-Sentence Thesis](#2-one-sentence-thesis)
- [3. Problem Setting](#3-problem-setting)
- [4. Motivation](#4-motivation)
- [5. Research Gap](#5-research-gap)
- [6. Current Method](#6-current-method)
- [7. Proposed Research Path](#7-proposed-research-path)
- [8. Experiment Plan](#8-experiment-plan)
- [9. Current Evidence](#9-current-evidence)
- [10. Contribution Framing](#10-contribution-framing)
- [11. Paper Skeleton](#11-paper-skeleton)
- [12. Open Decisions](#12-open-decisions)
- [13. Guardrails](#13-guardrails)
- [14. Research-Notes Ingestion Outcome](#14-research-notes-ingestion-outcome)
- [15. File Sources](#15-file-sources)
- [16. Ranked Research Problem Portfolio](#16-ranked-research-problem-portfolio)

This file is the paper-facing synthesis of the project. It is not a frozen thesis plan. It should evolve when code, experiments, advisor feedback, or literature change the argument.

## 1. Working Title

Temporal Difference-Subspace Dynamics for Multispectral Satellite Image Time Series

Alternative titles:

- Time-Aware First- and Second-Order Difference Subspaces for Multispectral Satellite Sequences
- Multiscale Geodesic Subspace Dynamics for Interpretable Satellite Time-Series Change Analysis
- Spatially Aware Difference Subspace Priors for Multispectral Change Detection
- Evaluating Subspace-Based Change Priors for Supervised Sentinel-2 Binary Segmentation
- When Do Interpretable Unsupervised Priors Help Remote-Sensing Change Segmentation?

## 2. One-Sentence Thesis

This project tests whether registered multispectral satellite sequences can be
represented as high-dimensional band-image subspaces whose first/second
Difference-Subspace and geodesic dynamics provide spatially attributable
evidence of temporal change, and how cadence, radiometry, seasonality, and
misregistration limit that evidence.

## 3. Problem Setting

Broader research direction:

```text
interpretable subspace-based change detection for multispectral satellite imagery
```

Input:

```text
registered multispectral images at dates t_1 ... t_T
with a common spatial grid and B aligned bands per date
```

Output:

```text
first-order temporal magnitudes between dates
second-order and geodesic along/orthogonal magnitudes for date triples
spatial contribution maps and optional detected-event maps
```

Current evidence datasets:

```text
MultiSenGE: 23-date Sentinel-2 exploratory sequences without temporal labels
IPOL registered SITS: 20-frame RGBI sequences with published qualitative events
OSCD: secondary labeled two-date spatial verification benchmark
```

Harmonized Sentinel-2 L2A is the next Sensei-requested independent sequence,
but it should be admitted only after documenting region, dates, actual cadence,
cloud/no-data filtering, alignment, event source, and evaluation protocol.
MultiSenGE and IPOL are first trials, not permanent thesis boundaries.

Likewise, `Phase 1` and `Phase 2` are current workflow names, not the paper's fixed conceptual structure. In the paper, describe them by function when possible:

```text
geometric/classical prior generation -> supervised neural segmentation/downstream learning
```

Current task boundary:

- This is temporal multispectral change analysis and changed-area attribution.
- It is not yet a calibrated binary event detector.
- It is not building damage classification or semantic change detection.
- It is not yet xBD/xBD-S12 damage mapping.
- OSCD binary experiments remain valid negative/diagnostic evidence, but no
  longer define the primary novelty question.

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

### Why geometric methods still matter beside neural networks

Neural networks are the stronger default when dense labels, matched training/test domains, and pure segmentation accuracy are the only goals. The thesis should not argue that a hand-built DS map will generally beat modern CNN, Siamese, Transformer, or foundation-model change detectors.

The justification for subspace/geometric methods is different:

- Interpretability: subspaces, canonical angles, projections, DS/GDS magnitudes, and geodesic quantities can be inspected as geometric evidence instead of only as learned filters.
- Label efficiency: DS, PCA-diff, CVA, IR-MAD, and KDS-style maps can act as priors, pseudo-label candidates, auxiliary targets, or region proposals when dense labels are scarce.
- Temporal structure: GDS, second-order DS, geodesic analysis, and tensor/product-Grassmann variants are more naturally tied to multi-date subspace sequences than to a single pre/post segmentation output.
- Diagnostics: if DS fails against raw L2, PCA-diff, IR-MAD, or U-Net, the failure can still reveal pseudo-change, spatial-information loss, poor sample construction, or dataset-label mismatch.
- Hybrid use: geometry can happen before learning as a prior map, during learning as an auxiliary/attention signal, or after learning as a deep-feature subspace, prediction-error subspace, or SOTA-detector-plus-GDS interpretation layer.

Literature anchor:

- Asokan and Anitha 2019 survey: remote-sensing change detection remains a broad multi-temporal spatial-change problem with many method families and persistent practical challenges.
- Wang et al. 2024 deep-learning CD review: deep learning dominates current CD research, but incomplete supervision, self-supervision/few-shot learning, foundation models, and multimodal data remain active challenges.
- Sample-efficient CD surveys: label scarcity is a real field problem, so interpretable or unsupervised priors are worth testing if they are compared honestly against modern alternatives.

Safe sentence:

```text
The goal is not to replace neural change detectors, but to test whether subspace-based geometric representations provide interpretable, label-efficient, or temporal change evidence that can complement them.
```

Contribution lanes to keep explicit:

| Lane | What it means | Experiment path | Evidence needed |
|---|---|---|---|
| Interpretability | DS/GDS should explain a change direction through spectral, spatial, date, component, or subspace differences rather than only outputting a mask. | Band-group attribution, region subspace descriptors, neural-front-end plus GDS clustering, error-subspace diagnostics. | Visual maps plus band/date/component attributions; if labels exist, cluster agreement or class-wise error analysis. |
| Feature isolation | Subspace methods may isolate dominant or discriminative multispectral characteristics before change detection, such as vegetation-like variation, built-up/soil/water-sensitive directions, band-group factors, temporal background drift, or neural latent factors. | Band-group DS, PCA/DS projection visualizations, low-rank/background versus residual maps, CCA/KCCA feature views, Signal Latent Subspace-style deep-feature DS. | Evidence that the isolated factors are stable, interpretable, and improve or explain a downstream change method beyond raw bands or simple PCA-diff. |
| Label efficiency | Subspace methods may generate priors, pseudo-labels, candidate regions, or auxiliary targets without dense human labels. | Prior pseudo-label pretraining, auxiliary prior head, active learning from geometry, prior-channel fusion under reduced-label settings. | Low-label curves showing better or more stable performance than raw-only training at the same label budget. |
| Temporal or multi-date analysis | GDS/geodesic methods can describe how subspaces evolve across many dates, which is different from ordinary binary pre/post segmentation. | MultiSenGE or Harmonized Sentinel-2 date-window GDS, second-order DS, geodesic trajectory plots, temporal clustering. | Coherent temporal trajectories, seasonality checks, manual/weak-label validation, or labels if a dataset supports them. |
| Hybrid NN + geometry | Neural models localize or produce features; subspace/GDS methods interpret, cluster, diagnose, or summarize changed regions. | `H5`, `H6`, `H7`, `H9`, `H10`, `H11` in `notes/experiments.md`. | Evidence that geometry adds interpretation, label-efficiency, error insight, or temporal structure beyond the neural mask alone. |
| Negative or diagnostic study | Even if DS does not beat PCA-diff, IR-MAD, or U-Net, the thesis can still show exactly where and why it fails or helps. | Spatial DS comparison, IR-MAD comparison, city-wise failure taxonomy, pseudo-change analysis. | Clear failure modes tied to sample construction, spatial information loss, pseudo-change, or dataset-label mismatch. |

Post-mining priority, 2026-06-21:

1. Run the RTW phase/tempo-invariance gate as the next positive-method test.
   This is Sensei-aligned and asks whether randomized ordered subsequences add a
   specific temporal invariance that harmonic/Fourier/TWDTW controls lack.
2. Keep the diagnostic paper as the evidence-backed publication spine: direct
   DS detection and first/second trajectory geometry have repeatedly failed to
   add beyond simpler controls on real data.
3. Treat geometry features plus shallow learning as conditional, not the next
   default. It becomes defensible only through nested raw-only, geometry-only,
   raw-plus-geometry, and shuffled-geometry comparisons under held-out sites and
   label budgets.
4. Keep moment-factorized local HSI orientation/wavelength attribution as the
   strongest data-gated parallel route. Do not revive generic material-subspace
   detection without covariance, MMD, unmixing, and per-band falsifiers.

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

Final working problem statement as of 2026-06-17:

```text
Can spatially aware Difference Subspace construction preserve the spatial structure of multispectral Sentinel-2 images well enough to produce interpretable changed-area evidence, and where does it help or fail compared with raw spectral difference, PCA-diff, Celik/IR-MAD, and neural change-detection baselines?
```

This is the problem to follow right now because it is:

- directly connected to Sensei's warning that the current global pixel subspace can break spatial information;
- measurable on the existing OSCD benchmark with labels;
- compatible with a negative-result thesis if DS fails;
- narrow enough to run experiments immediately;
- broad enough to leave future pivots toward temporal GDS, object-level change, semantic change, or neural/geometric hybrids.

The key gap is not "invent DS." The key gap is empirical and methodological: how to adapt subspace representations to multispectral satellite images without losing the spatial structure needed for change maps.

Research reset rule:

```text
Do not solve a problem we invented. Identify a real remote-sensing change-detection problem first, then test whether subspace methods help.
```

Current reset stance:

- The project is salvageable only if it becomes problem-driven.
- The strongest immediate problem is not "make DS work"; it is whether subspace construction can preserve spatially meaningful change evidence in multispectral images.
- A valid thesis can still exist if DS does not outperform deep learning, but only if the experiments explain when global pixel subspaces fail and whether patch/local/object/temporal subspaces address a real weakness.
- If corrected DS and spatial variants fail against raw L2, PCA-diff, and raw U-Net, the thesis should pivot toward an empirical negative/diagnostic study or a better-defined application such as multi-date monitoring or object-level greenhouse/building change.

Candidate real problems to evaluate before locking the thesis:

- pseudo-change versus meaningful change;
- spatially meaningful subspace construction for multispectral imagery;
- subspace-based feature isolation for multispectral imagery before or alongside change detection;
- label-efficient/interpretable priors for supervised change segmentation;
- multi-date temporal change/recovery analysis;
- domain-specific monitoring such as abandoned greenhouses, only if data and labels support it.

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

Research posture:

```text
Do not frame the thesis as "subspace beats deep learning."
```

The defensible posture is:

```text
geometric representation first -> neural/deep-learning integration second
```

Layer 1: geometry-only.

- DS, patch DS, local-window DS, PCA-diff, CVA, and related priors produce interpretable change maps without labels.
- This layer answers whether the subspace construction itself captures meaningful change.

Layer 2: geometry plus learning.

- A useful geometric map can be used as an extra channel, prior, diagnostic feature, pseudo-label source, attention cue, or label-efficient aid for U-Net/Siamese/future models.
- This layer answers whether the geometric representation complements supervised learning.

This matches the lab style of mixing geometrical representations and deep learning, while avoiding the unrealistic claim that a simple hand-built prior should outperform modern neural methods by itself.

Important hybrid framing:

- Strong localization can come from a neural model.
- Subspace/GDS methods can then interpret changed regions, cluster change types, describe temporal trajectories, diagnose model errors, or provide low-label priors.
- This may be stronger than asking DS alone to solve the full pixel-level localization problem.
- The contribution must be named precisely: better localization, label efficiency, interpretable clustering, temporal description, or error diagnosis. Do not let the neural model do all useful work while calling the whole result a subspace contribution.

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

5. Spatial feature construction
   - future route using Green Learning, PixelHop/PixelHop++, wavelet components, or compression-style multiscale features
   - useful only if it preserves spatial information better than global pixel samples
   - should be tested as feature construction before DS/KDS/SSC, not presented as a solved method
   - specific queued variant: `multiscale_subspace_pyramid`, where the image is represented by subspaces at whole-image, `2x2`, `4x4`, and optionally `8x8` spatial scales, then the scale responses are combined

6. Reference-code method family screen
   - use bundled DS, MagTool, and MATLAB Subspace Toolbox code as a menu of possible method families
   - candidates include KPCA/KDS, CCA/KCCA, structured/temporal CCA, mutual-subspace methods, RTW/SFA/temporal tensor methods, and Grassmann magnitude/decomposition tools
   - each candidate must first be adapted to a specific satellite sample type: pixel, patch, local window, date subspace, or deep-feature embedding
   - do not promote any candidate into the thesis argument without a one-city smoke result and a comparison against simple baselines

7. Harmonized Sentinel-2 / MultiSenGE GDS/KGDS
   - future multi-date extension
   - useful for temporal difference spaces, but semantic interpretation needs clustering, labels, or weak supervision
   - must first answer practical dataset questions: frame count, time step, cloud/no-data filtering, and evaluation proxy

8. Deep-feature latent subspaces
   - future method bridge inspired by Mahyub et al. 2024 Signal Latent Subspace
   - one sample could be a patch/tile/date embedding from a remote-sensing model rather than a raw 13-band pixel
   - useful only if it answers the spatial-information problem better than raw pixel DS

9. Hybrid NN + subspace/GDS paths
   - use neural methods for localization, representation, or proposals;
   - use subspace methods for interpretation, clustering, temporal geometry, pseudo-labels, auxiliary losses, or error diagnosis;
   - see `notes/experiments.md` section 6, item 22 for the concrete `H1`-`H14` experiment menu.

Current main candidate:

```text
Spatially aware Difference Subspace priors for interpretable Sentinel-2 changed-area detection.
```

Status:

- This is the strongest current candidate, not a proven contribution.
- Global, patch, local-window, and fixed-grid pyramid DS have now been tested. Patch/local variants improve over global pixel DS but remain well below PCA-diff; the exact fixed-grid pixel-spectral pyramid is a negative result.
- Band-Image DS is the strongest current DS construction, but even rank 12 remains below PCA-diff on mean AP. It should be retained as a component/diagnostic, not presented as a superior standalone detector.
- Label-free PCA/Band-Image/IR-MAD rank fusion provides statistically reliable AUROC complementarity but not a reliable AP or thresholded-segmentation improvement.
- Split-safe changed-area calibration makes fusion the best held-out test mask in the current comparison, but its F1 gain over PCA-diff is not significant across 10 test cities. Calibration alone is not the contribution.
- The next contribution gate is whether pseudo-change analysis or a genuinely different nuisance-aware spatial feature representation makes the complementary signal useful and interpretable.
- Existing remote-sensing literature already contains spatial-spectral subspace and low-rank/sparse ideas, so novelty must be framed as a specific DS/GDS-style adaptation and evaluation, not as the invention of spatial satellite subspaces.

Possible sharper thesis question:

```text
Can spatial support in DS-style subspace construction make geometric change priors more useful and interpretable for multispectral Sentinel-2 change detection?
```

Decision logic:

- If patch/local/flattened-band/multiscale DS improves metrics or explains false positives better than global pixel DS, keep spatially aware DS as the thesis core.
- If DS variants remain weaker than raw L2, PCA-diff, Celik, and IR-MAD, reframe as a diagnostic/negative study: global and simple spatial DS are insufficient unless paired with stronger feature construction or learning.
- If neural/foundation methods dominate but subspace descriptors explain regions, clusters, temporal progression, or errors, pivot to a hybrid geometry-plus-learning contribution.
- If OSCD's binary artificialization labels conflict too strongly with spectral/geometric change, consider semantic/object-level or multi-date datasets rather than forcing OSCD.

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
- A clear demonstration of how geometric subspace priors can complement, rather than replace, deep change-detection models.

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

Current novelty boundary:

- Not novel: DS itself, KPCA/KDS itself, PCA-diff, CVA, IR-MAD, U-Net, Siamese CD, generic spatial-spectral subspace analysis.
- Potentially novel enough for a thesis: a carefully verified spatial-support DS adaptation for Sentinel-2 change maps, compared against global pixel DS and classical/neural baselines with honest failure analysis.
- Stronger future novelty: tensor/n-mode or product-Grassmann spatial-spectral-temporal subspace construction for multi-date satellite imagery, if supported by a dataset and evaluation protocol.

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
- Should a Signal Latent Subspace-style experiment be tested with remote-sensing latent features after the spatial DS audit?
- Which modern unsupervised/supervised baselines are mandatory for a paper?

Low priority for now:

- xBD/xBD-S12 damage pipeline.
- abandoned greenhouse benchmark.
- foundation-model extensions.

## 13. Guardrails

These are old ideas or risks that should not disappear, but they should not distract the main OSCD work.

Main guardrails:

- The strongest standalone Phase 1 prior is not necessarily the strongest Phase 2 prior, and the reverse is also possible.
- OSCD validation is project-defined from training cities; document the split rather than implying it is official.
- Phase 2 prior channels are continuous maps, not thresholded labels.
- Threshold metrics and ranking metrics answer different questions; report both.
- Probability-map saving or rerun-based threshold sweeps are needed for true threshold tuning.
- `project_cli.py cleanup --aggressive --apply` requires an explicit keep-list decision and user approval.

Future hooks to keep out of the main claim for now:

- xBD/xBD-S12 damage severity mapping.
- abandoned-greenhouse mapping.
- UAV/edge deployment.
- MCDA/IoT/DMaaS/operational dashboard ideas.
- graph decision layers and broad foundation-model extensions.

## 14. Research-Notes Ingestion Outcome

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
- Local/patch DS, KPCA/KDS, Signal Latent Subspace-style deep-feature DS, and structured CCA/S3CCA/KCCA are method-development routes only after the spatial-information problem is measured.
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
- `docs/SECOND_OPINION_RESEARCH_CONTEXT.md`: neutral prompt/context for external critique of the research direction.
- `docs/RUNBOOK.md`: exact commands.
- `docs/experiment_reports/oscd_core_sweep_3seed_150epoch_2026-05-03.md`: accepted sweep summary.
- `docs/source_records/qa_report_response_2025-11-20.pdf`: submitted QA answers; useful for spectral-vs-semantic explanation, MultiSenGE pairing caveats, seasonal-risk framing, and IR-MAD caution.
- `docs/source_records/legacy_cs_seminar_report_ds_damage_segmentation_2025.tex`: earlier seminar-report source; useful as historical writing material, but it overuses old Phase 1/2 and damage-segmentation framing compared with the current truth.
- `docs/source_records/student_feedback_channel4_2025-11-20.xlsx`: original student feedback source; extracted into `notes/feedback.md`.

## 16. Ranked Research Problem Portfolio

This ranking was rebuilt on 2026-06-19 from the active notes, Apple Notes,
Sensei Slack messages, senpai feedback, the old research reset, reference-code
inventory, completed OSCD experiments, the new temporal experiments, and a
targeted literature search. It supersedes the ordering in the 2026-06-13 reset
audit because Sensei later gave a much more explicit time-sequential task.

The scores are decision aids, not probabilities. The weighting is:

- Sensei/advisor alignment: 25%.
- Evidence of a real literature or methodological problem: 20%.
- Dataset and evaluation feasibility: 20%.
- Existing implementation and preliminary evidence: 15%.
- Scope feasibility: 10%.
- Scientific value even if the hypothesis fails: 10%.

| Rank | Research problem | Decision score | Current role |
|---:|---|---:|---|
| 1 | Time-aware first/second DS and geodesic decomposition for multispectral satellite time series | 91/100 | active core |
| 2 | Radiometrically invariant but registration-robust temporal subspace change | 88/100 | required method refinement |
| 3 | Multiscale local temporal subspaces for changed-area localization | 85/100 | required spatial extension |
| 4 | What do temporal DS quantities actually detect versus established SITS change methods? | 82/100 | evaluation spine / safe paper |
| 5 | Slow pseudo-change versus abrupt structural change using SFA/SSA plus DS | 78/100 | strong follow-up |
| 6 | Spatially supported bi-temporal DS for multispectral binary change detection | 73/100 | existing diagnostic track |
| 7 | Neural localization followed by geometric temporal interpretation | 70/100 | hybrid follow-up |
| 8 | Deep latent and product-Grassmann subspace dynamics | 66/100 | higher-risk hybrid |
| 9 | Nonlinear KDS/KGDS for multi-date multispectral trajectories | 61/100 | method-understanding/future |
| 10 | First/second variation of VLM explanation subspaces for satellite sequences | 55/100 | exploratory high-risk idea |

The list does **not** authorize ten parallel implementations. Work now on ranks
1-4 as one coherent study. Rank 5 is the first planned extension. Ranks 6-10
remain available if the active study fails or a later paper needs a new route.

### 16.1 Time-Aware First/Second DS And Geodesic Decomposition

**Problem statement:** Can first-order DS magnitude, second-order DS magnitude,
and geodesic along/orthogonal decomposition identify and explain temporal
changes in registered multispectral satellite sequences when acquisition times
are irregular?

**Why this is first:** Sensei explicitly requested a time-sequential satellite
set, full channel-image subspaces in a super-high-dimensional space, changes in
first/second magnitudes, geodesic decomposition, and varied tests. The
second-order paper itself assumes equal time intervals, while real satellite
sequences often do not. That creates a concrete adaptation problem rather than
the generic claim "apply DS to satellite images."

**Hypothesis:** One date can be represented by the span of its aligned band
images, `S_t in Gr(r, N_pixels)`. First-order magnitude measures movement of
that spatial-band span; second-order magnitude measures departure from the
endpoint mean; the geodesic split distinguishes along-trajectory speed change
from off-trajectory change. For irregular dates, an additional time-aware
deviation from the endpoint geodesic at the observed acquisition fraction is
needed.

**Minimum experiment:** use full-rank channel-image subspaces on a registered
20+ frame sequence; report first, paper-faithful second, along, orthogonal,
time-aware deviation, and spatial contribution maps. Compare known event dates
and equal/unequal-gap controls.

**Baselines:** raw reflectance/index curvature, IPOL novelty/NFA, MOSUM,
BFAST/JUST/CCDC where applicable, and simple spatial PCA/chronochrome controls.

**Falsifier:** curves only track acquisition gaps, seasonal indices, or raw
pixel RMSE; contribution maps remain diffuse and miss known events.

**Curiosity:** does a structural event appear primarily as motion orthogonal to
the normal seasonal geodesic, while ordinary phenology remains along it?

**Stronger sample construction to test next:** build one annual/seasonal
subspace from repeated date composites of the same aligned patch or field:
`X_y in R^((B*N) x M_dates)`. This makes the temporal observations the genuine
sample set. Irrigation start/stop is the first candidate phenomenon because it
adds or removes a seasonal cycle. IrrMapper can propose annual transitions, but
its outputs are weak labels and require manual/independent verification.

### 16.2 Radiometrically Invariant, Registration-Robust Temporal Subspace Change

**Problem statement:** Can the useful invariance of full-channel band-image
subspaces to global per-band gain/offset be retained while suppressing their
extreme sensitivity to subpixel spatial misregistration?

**Why it is defensible:** controlled tests already show a sharp, reproducible
boundary: rank-full centered band-image DS is exactly invariant to tested
global gain/offset but a one-pixel translation produces more response than a
localized synthetic change. Registration is a recognized remote-sensing CD
failure mode, so this is not a project-invented weakness.

**Method candidates:** pre-registration, phase-correlation alignment,
translation-tolerant local pooling, multiscale tiles, low-frequency/wavelet
band-image bases, or minimizing DS over a small displacement neighborhood.

**Evaluation:** controlled gain/offset/translation/local-change injections,
real registered sequences, residual registration perturbations, runtime, and
event-localization maps.

**Success:** registration response falls substantially without destroying the
radiometric invariance or local-change response.

**Falsifier:** every robustification also removes the structural change signal.

**Curiosity:** which geometric information is truly invariant and useful, and
which information is merely a fragile consequence of pixel correspondence?

### 16.3 Multiscale Local Temporal Subspaces For Localization

**Problem statement:** Can a hierarchy of local band-image subspaces preserve
the temporal/geodesic information of DS while localizing changed regions more
cleanly than one whole-scene subspace?

**Motivation:** whole-scene temporal DS produces one interpretable global
magnitude but its IPOL Las Vegas contribution maps respond broadly to roofs,
roads, and scene texture. The IPOL baseline itself reports that local tiling
helps when change is global/frequent. Senpai's Green Learning/wavelet idea also
motivates whole image -> 2x2 -> 4x4 -> finer spatial support.

**Construction:** for each date and tile, flatten each band image inside that
tile into one spatial vector; construct one full/low-rank tile subspace; compute
first/second/time-aware quantities per tile; overlap and fuse multiple scales.

**Baselines:** whole-scene temporal DS, post-smoothed contribution maps,
multiscale PCA/chronochrome, IPOL patch novelty, and raw/index changes.

**Success:** known events are localized with fewer scene-wide edge responses,
and gains survive location/scale controls.

**Falsifier:** improvement is entirely reproduced by smoothing or generic
tiling without DS geometry.

**Curiosity:** at what spatial scale does subspace curvature correspond to an
object-level event rather than ordinary appearance fluctuation?

### 16.4 What Do Temporal DS Quantities Actually Detect?

**Problem statement:** Across registered satellite time series, what physical
or statistical changes are ranked by first DS, second DS, geodesic orthogonal
change, and time-aware geodesic deviation, and how do they differ from
established temporal change detectors?

**Contribution type:** a rigorous empirical/diagnostic study rather than a new
algorithm claim. This is the safest paper if the method does not win.

**Protocol:** compare on multiple sequences with event annotations or published
visual cases; categorize construction, demolition, vegetation/seasonality,
cloud/shadow, radiometric shifts, and registration errors. Include rank,
preprocessing, cadence, scale, and attribution ablations.

**Success:** identify repeatable regimes where each geometric component adds
information or explainable failure evidence.

**Falsifier:** all DS quantities are monotonic transformations of raw/index
difference and provide no distinct attribution.

**Curiosity:** is geodesic curvature a meaningful new signal, or only an
expensive restatement of familiar temporal residuals?

### 16.5 Slow Pseudo-Change Versus Abrupt Structural Change

**Problem statement:** Can slow-feature or singular-spectrum subspaces model
seasonal/background trajectories so that DS/geodesic residuals emphasize
abrupt land-surface change?

**Grounding:** Sensei suggested SSA/SFA-related directions; DSFA is an
established remote-sensing change method; BFAST/JUST explicitly model temporal
trend/seasonality. This route addresses the real pseudo-change problem rather
than assuming every spectral difference is target change.

**Experiment:** learn slow/background components from same-location sequences,
construct subspaces from trajectory/Hankel matrices, and compare first/second
DS residuals with DSFA, BFAST/JUST, raw indices, and the rank-1 temporal method.

**Risk:** without sufficient seasonal coverage, "slow" may simply encode the
wrong period or absorb gradual real change.

**Curiosity:** can normal seasonal motion be represented as a low-curvature
path whose departures reveal genuinely unusual events?

### 16.6 Spatially Supported Bi-Temporal DS

**Problem statement:** For two-date multispectral binary change detection,
which DS sample definition, if any, preserves enough spatial structure to add
useful evidence beyond PCA-diff, chronochrome, IR-MAD, and simple smoothing?

**Current evidence:** patch construction changes DS behavior, Band-Image DS is
stronger than old global pixel DS, but PCA-diff remains stronger on mean AP.
Spatial PCA/smoothing explains much of the apparent spatial gain. Therefore
this is now a diagnostic route, not the default novelty claim.

**Remaining defensible work:** local intrinsic DS magnitude, displacement
robustness, contribution attribution, and paired city-wise tests against
matched non-DS spatial controls.

**Falsifier:** matched PCA/chronochrome controls reproduce every gain.

**Curiosity:** is there any spatial support where canonical-angle geometry adds
information that linear projection residuals do not?

### 16.7 Neural Localization Followed By Geometric Temporal Interpretation

**Problem statement:** Can a neural/foundation detector localize candidate
regions while DS/GDS/geodesic analysis explains how each region evolves over
multiple dates?

**Why it may work:** neural networks are stronger localizers; subspace geometry
may be more defensible as a compact trajectory descriptor than as the sole
pixel detector. This matches the lab pattern of learned features plus geometry.

**Experiment:** detect objects/regions, construct one band/deep-feature
subspace per region/date, then cluster or classify trajectories by first/second
and GDS descriptors. Greenhouses, buildings, or semantic CD datasets are
possible applications only after labels and region units are available.

**Risk:** the neural model may do all useful work, leaving geometry decorative.

**Curiosity:** can geometry reveal recovery, degradation, or stable change
types that a single segmentation mask cannot express?

### 16.8 Deep Latent And Product-Grassmann Dynamics

**Problem statement:** Do subspaces built from remote-sensing encoder features
provide more meaningful temporal geometry than subspaces built directly from
raw band images?

**Grounding:** Signal Latent Subspace and Grassmannian neural methods show how
learned latent factors can become subspaces; tensor/product-Grassmann methods
preserve multiple factors. The satellite adaptation could separate spectral,
spatial, and temporal factors.

**Experiment:** freeze a remote-sensing encoder, extract patch/date features,
construct factor subspaces, compare raw versus latent first/second/GDS metrics,
and test whether product factors improve event discrimination.

**Risk:** high implementation complexity and unclear attribution; requires a
strong raw temporal baseline first.

**Curiosity:** does a learned manifold straighten nuisance variability enough
that geometric acceleration becomes semantically meaningful?

### 16.9 Nonlinear KDS/KGDS For Multi-Date Multispectral Trajectories

**Problem statement:** When linear band-image subspaces are insufficient, can
kernel subspaces capture nonlinear temporal change patterns without becoming
computationally or interpretively unusable?

**Grounding:** Sensei explicitly requested the TPAMI 2015 nonlinear KDS Venus
exercise. KPCA/KDS is not novel by itself, and KPCA change detection already
exists in remote sensing; novelty must come from a precise temporal/spatial
construction and evaluation.

**Experiment:** first reproduce Venus dimensions and equations; then use
Nyström/sampled kernel subspaces on local/date features and compare with linear
DS and ordinary KPCA change detection.

**Risk:** kernel memory, parameter sensitivity, preimage/attribution difficulty,
and weak novelty.

**Curiosity:** are relevant satellite trajectories genuinely nonlinear in a
way canonical angles in the original space cannot expose?

### 16.10 VLM Explanation-Subspace Dynamics

**Problem statement:** If a VLM produces multiple descriptions or embeddings
for each satellite date, can their first/second subspace variations reveal
semantic temporal change that raw spectral geometry misses?

**Grounding:** Sensei proposed explanation subspaces and their first/second
variation. Change captioning, ChangeCLIP, and open-vocabulary CD show that
language-aware change analysis is active, but this exact subspace framing is
unverified.

**Experiment:** generate controlled prompt ensembles per date, construct text
embedding subspaces, measure first/second/geodesic variation, and compare with
known semantic transitions and image-feature baselines.

**Risk:** prompt instability, hallucinated descriptions, weak reproducibility,
and distance from the current verified computer-vision pipeline.

**Curiosity:** can semantic explanations themselves form a smooth trajectory,
with meaningful events appearing as geometric acceleration?

### 16.11 Rejected Or Deprioritized Framings

- **Generic DS prior + U-Net:** already tested; no reliable gain and prior-guided
  fusion is not novel by itself.
- **Global pixel DS as the thesis method:** loses spatial sample identity and is
  weaker than matched classical baselines.
- **"First satellite subspace method":** false; hyperspectral, tensor,
  spatial-spectral, PCA/KPCA, MAD/CCA, and subspace CD literature exists.
- **Label-efficient DS without low-label curves:** unsupported until an actual
  10/25/50/100% label experiment is run.
- **Broad disaster platform/MCDA/UAV/IoT framing:** too many tasks and no single
  validated contribution.
- **Greenhouse or xBD damage claims from OSCD:** application mismatch; require
  their own data, labels, and metrics.
- **Generic foundation-model pivot:** crowded and weakly aligned with current
  implementation unless geometry has a clear, tested role.

### 16.12 Immediate Decision

The 2026-06-20 replication changes the active paper program. Whole-scene and
bidirectional temporal-context DS contribution maps do not currently compete
with raw difference or IPOL NFA for localization. Do not continue treating
"DS change map" as the assumed contribution.

The revised program separates two tasks:

```text
registered multi-date multispectral sequence
-> sequence-level geometric characterization:
     first/second DS + along/orthogonal + irregular-time geodesic deviation
-> pixel localization:
     projection novelty + raw/index/IPOL/classical/deep controls
-> registration-robust spatial support
-> labeled multi-temporal evaluation
```

The first labeled-data attempt is now the **seasonal observation subspace**
study, not another per-date global-pixel map. It asks whether annual subspaces
built from repeated Sentinel-2 date composites detect irrigation-regime
transitions and distinguish them from ordinary seasonal variation. A small
DynamicEarthNet slice remains the independent-label pressure test.

Controlled evidence now narrows this framing. Seasonal DS is not the expected
standalone winner: simple NDVI and singular-value controls dominate the current
synthetic boundary task. The surviving paper hypothesis is that low-rank
orientation geometry adds radiometrically robust within-sequence timing or
trajectory evidence, while singular energy carries amplitude change. This must
be demonstrated on verified real transitions and against temporal baselines.

The real-background intervention study sharpens this again. Block-trajectory
subspaces restore temporal-order sensitivity but did not improve controlled
localization. At fine local support, a smoothed temporal eigenspectrum was
competitive with NDMI/NBR and more invariant to tested gain/offset, while pure
first/second DS orientation was weaker. The immediate paper question is now
whether this orientation-versus-spectrum tradeoff repeats on independently
labeled events; it is not yet a method claim.

Current evidence:

- second/time-aware geometry has slightly stronger macro temporal agreement
  with IPOL event intensity than raw interpolation residual, but lower pixel AP;
- temporal-context DS has poor localization and is rejected in its current form;
- temporal projection novelty nearly matches raw localization, is invariant to
  tested global gain/offset, and distinguishes persistent from transient
  injected change;
- all geometric variants remain strongly translation-sensitive;
- low-frequency scale-space suppresses global translation response, but the
  gain is event-scale dependent and raw difference has the same cross-scale
  discrimination pattern;
- IPOL agreement is not ground truth, so no performance claim is authorized.
- In the off-grid MultiSenGE localization intervention, Gaussian-sigma-2 local
  eigenspectrum AP was `0.688`, NDMI `0.680`, and first DS `0.456`; the
  eigenspectrum/NDMI intervals overlap.
- Eigenspectrum gain/offset false alarms were `0.050`, but missing-composite and
  translation false alarms remained `0.358` and `0.292`.

The strongest current paper hypothesis is therefore:

```text
Can local temporal subspace orientation and eigenspectrum descriptors separate
(a) when/how a multispectral trajectory changes from (b) where a persistent
spectral-mode change occurs, while remaining distinguishable from missing-data,
radiometric, and registration artifacts?
```

This is a sharper version of ranks 1-4, not a new eleventh project. It survives
a negative outcome: if geometry does not improve localization, the empirical
result can still explain which invariances, persistence cues, and failure modes
temporal subspaces provide relative to established SITS detectors.
