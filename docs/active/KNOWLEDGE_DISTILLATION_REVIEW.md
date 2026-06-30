# Knowledge Distillation Review

## Quick Links

- [1. Purpose](#purpose)
- [2. Review Labels](#review-labels)
- [3. Current Active Control Docs](#current-active-control-docs)
- [4. Coverage Summary](#coverage-summary)
- [5. Notes Folder Review](#notes-folder-review)
- [6. Knowledge-Base Folder Review](#knowledge-base-folder-review)
- [7. Research Folder Review](#research-folder-review)
- [8. Experiment Reports Review](#experiment-reports-review)
- [9. Other Top-Level Docs Review](#other-top-level-docs-review)
- [10. Source Records And Non-Markdown Data](#source-records-and-non-markdown-data)
- [11. Deletion Decision Rules](#deletion-decision-rules)
- [12. Next Review Pass](#next-review-pass)

## Purpose

This file is the deletion and coverage dashboard for Stage 1: Active Knowledge
Distillation.

Nothing listed here has been deleted. The goal is to show what has been
distilled into the new active docs, what is probably safe to delete later, and
what still needs human review.

## Review Labels

| Label | Meaning |
|---|---|
| `[active-control]` | keep in the main reading path |
| `[human-source]` | user, Sensei, senpai, seminar, submitted, or raw source material |
| `[curated-evidence]` | experiment report with metrics/figures/commands |
| `[ai-synthesis]` | AI-generated narrative, agenda, summary, or interpretation |
| `[method-reference]` | formulas, construction details, source-to-code explanations |
| `[literature-resource]` | papers, datasets, baselines, bookmarks, reference code |
| `[absorbed]` | useful content is represented in active docs |
| `[delete-candidate]` | likely removable after user review |
| `[preserve]` | keep unless explicitly approved for deletion |
| `[needs-review]` | possible unique content remains |

## Current Active Control Docs

| File | Responsibility |
|---|---|
| `docs/active/CURRENT_RESEARCH_DIRECTION.md` | current direction, best evidence, safe/forbidden claims, deletion queue |
| `docs/active/RESEARCH_LANES_AND_DECISION_GATES.md` | ranked research lanes, win axes, continue/pause/close gates |
| `docs/active/METHODS_AND_IMPLEMENTATION_REFERENCE.md` | method cards, implementation/source trails, flexible card types |
| `docs/active/EXPERIMENT_RESULTS_LEDGER.md` | compact experiment/result memory |
| `docs/active/LITERATURE_DATASETS_AND_BASELINES.md` | reading/citation/baseline/dataset map |
| `docs/active/ADVISOR_FEEDBACK_AND_DECISIONS.md` | Sensei/senpai/seminar asks and decisions |
| `docs/active/REPRODUCIBLE_COMMANDS.md` | commands only |
| `docs/active/PERSONAL_RESEARCH_NOTES.md` | rough human note intake |

## Coverage Summary

Reviewed in this pass:

| Group | Files | Current decision |
|---|---:|---|
| `docs/pending_deletion_review/old_notes/*.md` | 7 | transitional; absorb into active docs |
| `docs/pending_deletion_review/old_knowledge_base/**/*.md` | 8 | mostly AI/reference synthesis; absorb then delete-candidate |
| `docs/pending_deletion_review/old_research_material/**/*.md` | 24 | mixed seminar, AI synthesis, novelty, and presentation materials; review selectively |
| top-level legacy docs | 6 | absorb or keep only if still serving a distinct purpose |
| `docs/experiment_reports/**/*.md` | many | curated evidence; keep until exact metrics/figures are represented |

High-level result:

- Core current direction is now in `CURRENT_RESEARCH_DIRECTION.md`.
- Research routes are now in `RESEARCH_LANES_AND_DECISION_GATES.md`.
- Method families from KB/research-mining are now summarized in
  `METHODS_AND_IMPLEMENTATION_REFERENCE.md` and
  `LITERATURE_DATASETS_AND_BASELINES.md`.
- Experiment results and AI-reported unverified claims are now in
  `EXPERIMENT_RESULTS_LEDGER.md`.
- Sensei/seminar asks are now in `ADVISOR_FEEDBACK_AND_DECISIONS.md`.

## Notes Folder Review

| File | Labels | Distilled into | Proposed fate |
|---|---|---|---|
| `docs/pending_deletion_review/old_notes/experiments.md` | `[human-source] [curated-evidence] [absorbed]` | `EXPERIMENT_RESULTS_LEDGER.md`, `CURRENT_RESEARCH_DIRECTION.md`, `RESEARCH_LANES_AND_DECISION_GATES.md` | `[delete-candidate]` after ledger coverage check |
| `docs/pending_deletion_review/old_notes/feedback.md` | `[human-source] [absorbed]` | `ADVISOR_FEEDBACK_AND_DECISIONS.md` | `[delete-candidate]` after advisor coverage check |
| `docs/pending_deletion_review/old_notes/literature.md` | `[literature-resource] [absorbed]` | `LITERATURE_DATASETS_AND_BASELINES.md` | `[delete-candidate]` after citation/resource check |
| `docs/pending_deletion_review/old_notes/methods.md` | `[method-reference] [absorbed]` | `METHODS_AND_IMPLEMENTATION_REFERENCE.md` | `[delete-candidate]` after method-card check |
| `docs/pending_deletion_review/old_notes/reference_bookmarks.md` | `[literature-resource] [absorbed]` | `LITERATURE_DATASETS_AND_BASELINES.md` | `[delete-candidate]` after bookmark import path check |
| `docs/pending_deletion_review/old_notes/research_paper_plan.md` | `[ai-synthesis] [human-source] [absorbed]` | `CURRENT_RESEARCH_DIRECTION.md`, `RESEARCH_LANES_AND_DECISION_GATES.md` | `[delete-candidate]` after thesis-lane check |
| `docs/pending_deletion_review/old_notes/README.md` | `[active-transition]` | this review file and docs README | keep until `docs/pending_deletion_review/old_notes/` folder is empty or removed |

## Knowledge-Base Folder Review

| File | Labels | Distilled into | Proposed fate |
|---|---|---|---|
| `docs/pending_deletion_review/old_knowledge_base/01_core_subspace_geometry.md` | `[ai-synthesis] [method-reference] [absorbed]` | methods/literature docs | `[delete-candidate]` |
| `docs/pending_deletion_review/old_knowledge_base/02_temporal_sequence_geometry.md` | `[ai-synthesis] [method-reference] [absorbed]` | temporal/structured lanes and method cards | `[delete-candidate]` |
| `docs/pending_deletion_review/old_knowledge_base/03_signal_tensor_and_remote_sensing.md` | `[ai-synthesis] [literature-resource] [absorbed]` | deep-feature/tensor/spatial context rows | `[delete-candidate]` |
| `docs/pending_deletion_review/old_knowledge_base/04_hyperspectral_cd_boundary.md` | `[ai-synthesis] [literature-resource] [absorbed]` | HSI lane and novelty boundaries | `[delete-candidate]` |
| `docs/pending_deletion_review/old_knowledge_base/05_project_evidence_map.md` | `[ai-synthesis] [absorbed]` | lane table and experiment interpretation | `[delete-candidate]` |
| `docs/pending_deletion_review/old_knowledge_base/README.md` | `[ai-synthesis] [absorbed]` | this review file | `[delete-candidate]` |
| `docs/pending_deletion_review/old_knowledge_base/claude_temporal/02_temporal_sequence_warping_ssa.md` | `[ai-synthesis] [method-reference] [absorbed]` | structured temporal/CCA/SFA lane | `[delete-candidate]` |
| `docs/pending_deletion_review/old_knowledge_base/claude_temporal/03_remote_sensing_cd_and_hsi.md` | `[ai-synthesis] [literature-resource] [absorbed]` | literature/baseline map | `[delete-candidate]` |

## Research Folder Review

| File | Labels | Distilled into | Proposed fate |
|---|---|---|---|
| `docs/pending_deletion_review/old_research_material/BOARD_CHEATSHEET.md` | `[human-facing] [needs-review]` | advisor/method docs partly | keep/review for seminar use |
| `docs/pending_deletion_review/old_research_material/CONCEPTS_EXPLAINED.md` | `[ai-synthesis] [method-reference] [absorbed]` | methods reference | `[delete-candidate]` after checking explanations |
| `docs/pending_deletion_review/old_research_material/MASTER_NARRATIVE_2026-06-22.md` | `[ai-synthesis] [absorbed]` | current direction, ledger, unverified claims | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/REFERENCES.md` | `[literature-resource] [absorbed]` | literature/baselines | `[delete-candidate]` after citation check |
| `docs/pending_deletion_review/old_research_material/SEMINAR_2026_FINAL.md` | `[human-facing] [needs-review]` | current direction and advisor docs partly | keep/review if final seminar material still useful |
| `docs/pending_deletion_review/old_research_material/SEMINAR_PREP.md` | `[ai-synthesis] [human-facing] [absorbed]` | current direction/feedback docs | `[delete-candidate]` after seminar-material review |
| `docs/pending_deletion_review/old_research_material/SEMINAR_TALK_FULL.md` | `[human-facing] [needs-review]` | methods/current direction partly | keep/review; may contain useful slide scripts |
| `docs/pending_deletion_review/old_research_material/challenges_ranked.md` | `[ai-synthesis] [literature-resource] [absorbed]` | lanes/literature docs | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/closest_methods_novelty.md` | `[ai-synthesis] [literature-resource] [absorbed]` | literature/lanes docs | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/seminar_defense_QA_2026-06-22.md` | `[human-facing] [advisor-feedback] [absorbed]` | advisor/method docs | keep/review until defense Q&A is compressed further |
| `docs/pending_deletion_review/old_research_material/seminar_narrative_DSprior_2026-06-22.md` | `[ai-synthesis] [absorbed]` | current direction/ledger | `[delete-candidate]` after DS-prior rerun decision |
| `docs/pending_deletion_review/old_research_material/sensei_asks_coverage_2026-06-22.md` | `[advisor-feedback] [absorbed]` | advisor feedback table | `[delete-candidate]` after user review |
| `docs/pending_deletion_review/old_research_material/synthesis_specific_tasks.md` | `[ai-synthesis] [method-reference] [absorbed]` | lanes/method docs | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/claude_temporal/bet1_design.md` | `[ai-synthesis] [method-reference] [absorbed]` | HSI lane/unverified claims | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/claude_temporal/bet2_design.md` | `[ai-synthesis] [absorbed]` | temporal lane | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/claude_temporal/DESIGN_TEMPORAL_DS_ACCV2026.md` | `[ai-synthesis] [method-reference] [absorbed]` | temporal lane/method cards | `[delete-candidate]` after temporal method check |
| `docs/pending_deletion_review/old_research_material/claude_temporal/DRAFT_DIAGNOSTIC_PAPER_ACCV.md` | `[ai-synthesis] [absorbed]` | diagnostic lane | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/claude_temporal/METHOD.md` | `[method-reference] [absorbed]` | temporal methods | `[delete-candidate]` after source-to-code details checked |
| `docs/pending_deletion_review/old_research_material/claude_temporal/RESEARCH_AGENDA.md` | `[ai-synthesis] [absorbed]` | research lanes | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/claude_temporal/RESEARCH_APPROACHES_RANKED.md` | `[ai-synthesis] [absorbed]` | lanes | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/claude_temporal/RESEARCH_DIRECTIONS_TOP10.md` | `[ai-synthesis] [absorbed]` | lanes/literature | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/claude_temporal/seminar_ranked_problems.md` | `[ai-synthesis] [absorbed]` | current direction/lanes | `[delete-candidate]` |
| `docs/pending_deletion_review/old_research_material/claude_temporal/SUBSPACE_CONSTRUCTION_LEDGER.md` | `[method-reference] [absorbed]` | methods reference | `[delete-candidate]` after formula audit |
| `docs/pending_deletion_review/old_research_material/claude_temporal/SUPREME_AUDIT_TABLE.md` | `[ai-synthesis] [curated-evidence] [absorbed]` | experiment ledger and lanes | `[delete-candidate]` after ledger check |

## Experiment Reports Review

Experiment reports are more valuable than AI synthesis because they contain
specific commands, metrics, figures, and decisions. Keep them until the ledger
has enough exact detail or a curated figure/result package replaces them.

| File/group | Labels | Distilled into | Proposed fate |
|---|---|---|---|
| `docs/experiment_reports/oscd_successive_subspace_learning_ds_2026-06-23.md` | `[curated-evidence] [absorbed]` | current direction, methods, ledger | keep as key report |
| `docs/experiment_reports/successive_saab_trainfit_external_gate_2026-06-23.md` | `[curated-evidence] [absorbed]` | current direction, ledger | keep as key report |
| `docs/experiment_reports/xbd_s12_external_validation_2026-06-22.md` | `[curated-evidence] [absorbed]` | current direction, ledger | keep as key external report |
| `docs/experiment_reports/oscd_band_image_matched_spatial_controls_2026-06-22.md` | `[curated-evidence] [absorbed]` | ledger/methods | keep for now |
| `docs/experiment_reports/cross_branch_research_evidence_matrix_2026-06-22.md` | `[curated-evidence] [absorbed]` | ledger/lanes | `[delete-candidate]` only after exact rows checked |
| `docs/experiment_reports/oscd_spatial_*.md` and `oscd_band_image_*.md` | `[curated-evidence] [absorbed]` | ledger | keep until spatial history is fully compressed |
| `docs/experiment_reports/hsi_*.md` | `[curated-evidence] [absorbed]` | HSI lane/ledger | keep until HSI lane is closed or revived |
| `docs/experiment_reports/spacenet7_*.md` | `[curated-evidence] [absorbed]` | transfer-negative ledger rows | keep until transfer failures are fully summarized |
| `docs/experiment_reports/*rtw*.md`, `*seasonal*.md`, `*temporal*.md` | `[curated-evidence] [absorbed]` | temporal/structured lane | keep until temporal lane review |
| `docs/experiment_reports/claude_temporal/*.md` | `[curated-evidence] [ai-generated-run-report] [needs-review]` | ledger partially | review as a batch before deletion |

## Other Top-Level Docs Review

| File | Labels | Distilled into | Proposed fate |
|---|---|---|---|
| `docs/pending_deletion_review/old_project_docs/PROJECT_BRIEF.md` | `[ai-synthesis] [absorbed]` | current direction | `[delete-candidate]` if no unique truth remains |
| `docs/pending_deletion_review/old_project_docs/RESEARCH_RESET_AUDIT.md` | `[ai-synthesis] [absorbed]` | current direction, lanes, literature | `[delete-candidate]` after user confirms reset concerns are covered |
| `docs/pending_deletion_review/old_project_docs/SECOND_OPINION_RESEARCH_CONTEXT.md` | `[ai-synthesis] [external-review-context]` | partially active | keep for now |
| `docs/pending_deletion_review/old_project_docs/RUNBOOK.md` | `[command-reference] [needs-review]` | reproducible commands partially | shrink/delete-candidate after command coverage |
| `docs/pending_deletion_review/old_project_docs/CD_TAXONOMY.md` | `[ai-synthesis] [literature-resource] [absorbed]` | lanes/literature | `[delete-candidate]` |
| `docs/README.md` | `[active-control]` | active docs index | keep |
| root `README.md` | `[active-control]` | project entry point | keep |
| `AGENTS.md` | `[active-control]` | agent behavior | keep |

## Source Records And Non-Markdown Data

| Group | Current decision |
|---|---|
| `docs/source_records/**` | preserve; raw provenance and imports are not active truth |
| `references/reference_code/**` | preserve until separate reference-code cleanup; useful for source-to-code audits |
| `references/reference_papers/**` | preserve unless papers are confirmed in Zotero/bookmarks and user approves |
| `data/**` | no Stage 1 deletion; handle in data/storage cleanup |
| `phase1/outputs/**`, `phase2/outputs/**`, `tmp/**` | no Stage 1 deletion; handle only after result/figure reproducibility audit |
| `temporal/**`, `phase1/**`, `phase2/**`, `tests/**` | no Stage 1 deletion; code cleanup is a later stage |

## Deletion Decision Rules

Before deleting a file or folder:

1. Confirm its useful claims are represented in an active control doc.
2. Confirm unique numbers/figures/commands either remain in a curated report or
   are reproducible.
3. Prefer deleting AI-generated synthesis first.
4. Preserve raw human/source records until explicit approval.
5. Do not delete code, datasets, or generated outputs in Stage 1.

## Next Review Pass

Recommended next pass:

1. Compare `docs/pending_deletion_review/old_notes/experiments.md` against `EXPERIMENT_RESULTS_LEDGER.md` section
   by section.
2. Compare `docs/pending_deletion_review/old_notes/methods.md` and
   `docs/pending_deletion_review/old_research_material/claude_temporal/SUBSPACE_CONSTRUCTION_LEDGER.md` against
   `METHODS_AND_IMPLEMENTATION_REFERENCE.md`.
3. Compare `docs/pending_deletion_review/old_research_material/SEMINAR_TALK_FULL.md` and
   `docs/pending_deletion_review/old_research_material/seminar_defense_QA_2026-06-22.md` against the current seminar
   needs, then keep only one seminar-facing file if needed.
4. After user review, delete the first batch of `[ai-synthesis]
   [delete-candidate]` files.
