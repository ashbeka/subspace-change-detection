# Docs Index

## Table Of Contents

- [1. Active Docs](#1-active-docs)
- [2. Source Records](#2-source-records)
- [3. Active Notes](#3-active-notes)
- [4. Historical Archive](#4-historical-archive)
- [5. Research Notes Rule](#5-research-notes-rule)

## 1. Active Docs

Read these first:

1. `CURRENT_RESEARCH_DIRECTION.md`
   - Current control panel: active direction, strongest evidence, safe claims, forbidden claims, and next gate.

2. `RESEARCH_LANES_AND_DECISION_GATES.md`
   - Candidate research lanes ranked by feasibility, win axis, evidence, and continue/pause/close gates.

3. `EXPERIMENT_RESULTS_LEDGER.md`
   - Compact evidence table: what was tried, what happened, and what decision each result supports.

4. `METHODS_AND_IMPLEMENTATION_REFERENCE.md`
   - Method cards, formulas, sample definitions, source-to-code trails, and implementation boundaries.

5. `ADVISOR_FEEDBACK_AND_DECISIONS.md`
   - Sensei/senpai/seminar feedback and what decisions/actions came from it.

6. `LITERATURE_DATASETS_AND_BASELINES.md`
   - Papers, datasets, baselines, bookmarks, reference code, and comparison pressure.

7. `REPRODUCIBLE_COMMANDS.md`
   - Exact commands only.

8. `KNOWLEDGE_DISTILLATION_REVIEW.md`
   - Coverage dashboard for old notes/research/kb files and deletion candidates.

9. `PERSONAL_RESEARCH_NOTES.md`
   - Rough human note intake. Preserve intent; translate stable implications into active control docs.

Detailed reports remain under `experiment_reports/`, but they are no longer the
first reading path. Use `EXPERIMENT_RESULTS_LEDGER.md` first, then open a
report only when you need exact implementation details, figures, or metrics.

9. `research_learning_map.html`
   - Local interactive concept/experiment map. Use it to see what to read and which code path supports each current or future experiment.

## 2. Source Records

`source_records/` preserves original human-facing project records:

- `student_feedback_channel4_2025-11-20.xlsx`
  - Student feedback received after the first seminar.
- `qa_report_response_2025-11-20.pdf`
  - Submitted QA response report after the first seminar.
- `legacy_cs_seminar_report_ds_damage_segmentation_2025.tex`
  - Earlier generated/submitted CS seminar report source.
- `chrome_bookmarks_raw_2026-06-07.html`
  - Raw Chrome bookmark export preserved before organization.
- `chrome_bookmarks_organized_research_2026-06-07.html`
  - Importable organized Chrome bookmark file preserving every URL once.
- `presentations/legacy_2025_oscd_ds_seminar/`
  - Historical seminar slide PNGs and talk notes. These are provenance, not current evidence; some result claims were superseded by later controlled sweeps.
- `final_organization_2026-06-12/`
  - Raw source batch for the final notes/bookmark organization pass, including Apple Notes, Slack exports, Word notes, Fukui subspace PDF, image attachments, Chrome export, Safari links, moved historical ingestion ledgers, the organized Chrome import file, and the final coverage audit.

These files are preserved as provenance. They are not the active project truth.
Useful knowledge from them should be reflected in the active control docs above.

## 3. Active Notes

The old `../notes/` folder is being absorbed during Active Knowledge
Distillation. Do not add new active project knowledge there unless the migration
is explicitly paused.

The rough personal note intake has moved to:

- `PERSONAL_RESEARCH_NOTES.md`

## 4. Historical Archive

`docs/archive/` was removed after a final ingestion audit. Its useful knowledge was consolidated into project docs. The deleted tracked files remain recoverable from Git/GitHub history, but they are no longer part of the active reading path.

The old `phase1/docs/` and `phase2/docs/` folders were also removed after consolidation. Their useful commands, method details, historical result caveats, and experiment plans are being absorbed into the active control docs. Their generated figures were not kept as active docs; regenerate or recover from Git history only if a specific figure is needed.

## 5. Research Notes Rule

The active knowledge location is now this `docs/` control-doc set.

`research-notes/` is a nested external notes repo/archive. It is useful background, but it should not be treated as the current project source of truth. Important current material should be extracted into the active docs above. Its current useful content was ingested previously; delete the folder only after the user confirms the new active docs cover it sufficiently.
