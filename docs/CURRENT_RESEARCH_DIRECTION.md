# Current Research Direction

## Quick Links

- [1. Purpose](#purpose)
- [2. Current Umbrella](#current-umbrella)
- [3. Current Working Direction](#current-working-direction)
- [4. Current Best Result](#current-best-result)
- [5. Immediate Next Evidence Gate](#immediate-next-evidence-gate)
- [6. Safe Claims](#safe-claims)
- [7. Forbidden Claims](#forbidden-claims)
- [8. Active Reading Path](#active-reading-path)
- [9. Stage 1 Distillation Policy](#stage-1-distillation-policy)
- [10. Transition Map](#transition-map)
- [11. Deletion Review Queue](#deletion-review-queue)
- [12. First File-Fate Checklist](#first-file-fate-checklist)

## Purpose

This is the fastest answer to: **what is this project doing now?**

This file is the active control panel. Older notes, seminar drafts, audits, and
AI-generated reports are evidence sources, not the main reading path.

## Current Umbrella

```text
Interpretable subspace-based representations for multispectral and temporal
satellite change analysis.
```

The project is not locked to OSCD, phase1/phase2, or one subspace construction.
The active benchmark is still OSCD Sentinel-2 binary change detection because it
has labels and working code.

## Current Working Direction

The strongest current direction is:

```text
Can spatially faithful, label-free feature/subspace construction make
Difference Subspace useful as interpretable changed-area evidence in
multispectral satellite images?
```

Current best evidence:

- Successive Saab-DS is the strongest internal OSCD result.
- Band-Image/projector geometry is useful for xBD-S12 candidate localization,
  but not direct damage classification.
- Temporal first/second DS and geodesic decomposition are implemented and
  Sensei-aligned, but currently characterization rather than a strong detector.
- Global pixel DS, fixed-grid pyramid DS, wavelet DS, RTW, broad HSI transfer,
  and SpaceNet7 RGB transfer are paused as primary routes.

## Current Best Result

Full report:
`docs/experiment_reports/oscd_successive_subspace_learning_ds_2026-06-23.md`

Summary:

| Result | Value |
|---|---:|
| Successive Saab-DS OSCD test AP | 0.3420 |
| Successive Saab-DS OSCD test AUROC | 0.8861 |
| Successive Saab-DS Otsu F1 | 0.3312 |
| Smoothed PCA AP | 0.3141 |
| PCA-diff AP | 0.3067 |
| matched feature L2 AP | 0.3067 |
| matched cross-reconstruction AP | 0.3279 |

Interpretation:

- This supports a **spatial construction effect** on OSCD.
- It does not yet prove universal transfer or SOTA performance.
- The key thesis question is whether this effect survives stronger external
  validation and meaningful controls.

## Immediate Next Evidence Gate

Do not add another unrelated method family yet.

Next task:

```text
Independently reproduce the reported DS-specific U-Net/multi-prior fusion gain.
```

Why:

- It is the most consequential unverified positive claim.
- If it holds, it gives a clear win axis: DS as complementary learned-prior
  evidence.
- If it fails, the main story returns to label-free spatial Saab-DS plus
  candidate-localization and diagnostics.

Required comparison:

| Model/input | Purpose |
|---|---|
| raw bands | supervised floor |
| bands + DS | direct DS-prior test |
| bands + smoothed PCA + IR-MAD | strong no-DS prior control |
| bands + DS + smoothed PCA + IR-MAD | multi-prior fusion |
| bands + cross-reconstruction + smoothed PCA + IR-MAD | matched non-DS control |

Decision:

- **Continue** if DS improves beyond no-DS and matched-cross controls across
  seeds/cities without leakage.
- **Pause neural-prior lane** if no-DS or matched-cross catches up.

## Safe Claims

- Spatial sample construction matters for DS on Sentinel-2 OSCD.
- Global unordered pixel DS is weak under current OSCD tests.
- Successive local Saab features make DS more useful internally on OSCD.
- xBD-S12 evidence supports candidate localization, not damage severity.
- Temporal DS/geodesic quantities can describe subspace trajectories, but need
  stronger labeled sequence validation.

## Forbidden Claims

Do not claim:

- SOTA remote-sensing change detection.
- Completed disaster damage segmentation.
- xBD building damage severity classification.
- A universally valid satellite DS method.
- Registration invariance.
- Semantic change interpretation.
- That DS was invented in this project.

## Active Reading Path

Read only these first:

1. `docs/CURRENT_RESEARCH_DIRECTION.md`
2. `docs/RESEARCH_LANES_AND_DECISION_GATES.md`
3. `docs/EXPERIMENT_RESULTS_LEDGER.md`
4. `docs/METHODS_AND_IMPLEMENTATION_REFERENCE.md`
5. `docs/ADVISOR_FEEDBACK_AND_DECISIONS.md`
6. `docs/LITERATURE_DATASETS_AND_BASELINES.md`
7. `docs/REPRODUCIBLE_COMMANDS.md`
8. `docs/KNOWLEDGE_DISTILLATION_REVIEW.md` for deletion/coverage review
9. `docs/PERSONAL_RESEARCH_NOTES.md` only for rough human notes

## Stage 1 Distillation Policy

Priority for preservation:

1. Sensei/senpai feedback.
2. User-written notes.
3. Experiment results and code evidence.
4. External papers, datasets, and reference code.
5. AI-generated synthesis.

AI-generated files should be read, compressed, and then marked as
delete-candidates when their useful content is absorbed. Human/source records
should be preserved until explicitly approved for deletion.

Labels used in the distillation review:

| Label | Meaning |
|---|---|
| `[human-source]` | user, Sensei, senpai, seminar, submitted, or raw source material |
| `[curated-evidence]` | experiment report with metrics, figures, commands, or reproducible output |
| `[ai-synthesis]` | Codex/Claude/other generated interpretation or narrative |
| `[method-reference]` | formulas, method construction, or source-to-code detail |
| `[literature-resource]` | papers, datasets, benchmarks, links, or reference code |
| `[active-control]` | one of the new files that remains in the main reading path |
| `[delete-candidate]` | likely removable after review because content is absorbed |
| `[preserve]` | do not delete without explicit user approval |
| `[needs-review]` | possible unique content remains |

## Transition Map

| Old source | New active destination | Later status |
|---|---|---|
| `notes/research_paper_plan.md` | this file + `RESEARCH_LANES_AND_DECISION_GATES.md` | absorb/delete-candidate |
| `notes/experiments.md` | `EXPERIMENT_RESULTS_LEDGER.md` | absorb/delete-candidate after coverage |
| `notes/methods.md` | `METHODS_AND_IMPLEMENTATION_REFERENCE.md` | absorb/delete-candidate after coverage |
| `notes/literature.md`, `notes/reference_bookmarks.md` | `LITERATURE_DATASETS_AND_BASELINES.md` | absorb/delete-candidate after coverage |
| `notes/feedback.md` | `ADVISOR_FEEDBACK_AND_DECISIONS.md` | absorb/delete-candidate after coverage |
| `notes/my_notes.md` | `docs/PERSONAL_RESEARCH_NOTES.md` | moved |
| `docs/research/*seminar*` | this file if current; otherwise source/delete-candidate | mostly delete-candidate |
| `docs/research/claude_temporal/*` | lane/method/experiment ledgers | AI synthesis delete-candidate |
| `docs/experiment_reports/*.md` | `EXPERIMENT_RESULTS_LEDGER.md` rows | keep curated until review |
| `docs/kb/*.md` | methods/literature references | absorb/delete-candidate |
| `research-notes/` | active docs if not already absorbed | delete-candidate after audit |

## Deletion Review Queue

Do not delete these automatically in Stage 1. Use this queue for review after
coverage is checked.

Highest-priority delete candidates:

| Candidate | Why likely removable | Condition before deletion |
|---|---|---|
| `docs/research/claude_temporal/*.md` | AI-generated synthesis now represented in lanes/methods/ledger | confirm no unique result missing from `EXPERIMENT_RESULTS_LEDGER.md` |
| old seminar drafts in `docs/research/` | superseded by active direction docs and final seminar material | keep only final slides/scripts if still needed |
| `notes/*.md` except `README.md` | active knowledge moved to docs control set | compare each file against active docs |
| `research-notes/` nested repo | old distilled notes repo, repeatedly ingested | final audit that no human note is unique there |
| `docs/kb/*.md` | AI/agent knowledge base material now summarized in method/literature docs | confirm formulas and source trails are preserved |
| duplicate source-record overlap versions | preserved only for Claude absorption safety | verify Git history and active docs cover useful differences |

Not delete without explicit approval:

| Source | Reason |
|---|---|
| `docs/source_records/final_organization_2026-06-12/` | raw Apple/Slack/bookmark/source batch |
| `docs/PERSONAL_RESEARCH_NOTES.md` | user rough-note intake |
| curated experiment reports with unique figures or exact metrics | evidence provenance |
| code and datasets | require separate code/data cleanup stages |

## First File-Fate Checklist

This is the initial minimization checklist. It is deliberately conservative:
AI-generated synthesis is prioritized for absorption/deletion; human/source
records are preserved unless explicitly reviewed.

| Path/group | Fate | Reason |
|---|---|---|
| `docs/CURRENT_RESEARCH_DIRECTION.md` | keep active | current control panel |
| `docs/RESEARCH_LANES_AND_DECISION_GATES.md` | keep active | research lane queue |
| `docs/METHODS_AND_IMPLEMENTATION_REFERENCE.md` | keep active | method/source-to-code reference |
| `docs/EXPERIMENT_RESULTS_LEDGER.md` | keep active | compact result memory |
| `docs/LITERATURE_DATASETS_AND_BASELINES.md` | keep active | reading/citation/baseline map |
| `docs/ADVISOR_FEEDBACK_AND_DECISIONS.md` | keep active | Sensei/senpai decision trail |
| `docs/REPRODUCIBLE_COMMANDS.md` | keep active | commands only |
| `docs/PERSONAL_RESEARCH_NOTES.md` | keep active | rough human note intake |
| `notes/feedback.md` | absorb/delete-candidate | distilled into advisor/decision doc |
| `notes/methods.md` | absorb/delete-candidate | distilled into methods reference |
| `notes/experiments.md` | absorb/delete-candidate | distilled into experiment ledger |
| `notes/literature.md` | absorb/delete-candidate | distilled into literature/baselines doc |
| `notes/reference_bookmarks.md` | absorb/delete-candidate | distilled into literature/baselines doc |
| `notes/research_paper_plan.md` | absorb/delete-candidate | distilled into current direction and lanes |
| `docs/PROJECT_BRIEF.md` | absorb/delete-candidate | replaced by current direction if no unique status remains |
| `docs/RESEARCH_RESET_AUDIT.md` | absorb/delete-candidate | important AI reset, but its stable conclusions are now control-doc material |
| `docs/SECOND_OPINION_RESEARCH_CONTEXT.md` | keep for now | useful external-review package |
| `docs/RUNBOOK.md` | absorb/delete-candidate | should shrink into reproducible commands after command coverage check |
| `docs/CD_TAXONOMY.md` | absorb/delete-candidate | taxonomy belongs in lanes/literature if still useful |
| `docs/kb/*.md` | absorb/delete-candidate | AI knowledge base; keep only source-to-code/method facts |
| `docs/research/BOARD_CHEATSHEET.md` | keep/review | useful seminar quick reference, but not active project control |
| `docs/research/CONCEPTS_EXPLAINED.md` | absorb/delete-candidate | explanations belong in methods reference or learning map |
| `docs/research/MASTER_NARRATIVE_2026-06-22.md` | absorb/delete-candidate | superseded narrative |
| `docs/research/SEMINAR*.md`, `seminar_*.md` | keep/review final only | retain only material needed for future presentations |
| `docs/research/challenges_ranked.md` | absorb/delete-candidate | stable points now in lanes/literature |
| `docs/research/closest_methods_novelty.md` | absorb/delete-candidate | stable novelty boundaries now in literature/lanes |
| `docs/research/synthesis_specific_tasks.md` | absorb/delete-candidate | tasks now belong in lanes/ledger |
| `docs/research/claude_temporal/*.md` | absorb/delete-candidate | AI synthesis; preserve unique methods/results only |
| `docs/experiment_reports/*.md` | keep curated for now | exact metrics and provenance; later compress/delete selectively |
| `docs/source_records/**` | preserve | original records/provenance, not active truth |
| `research-notes/` | audit/delete-candidate | nested old notes repo; delete only after final uniqueness check |
