# Docs Index

## Table Of Contents

- [1. Active Docs](#1-active-docs)
- [2. Source Records](#2-source-records)
- [3. Active Notes](#3-active-notes)
- [4. Historical Archive](#4-historical-archive)
- [5. Research Notes Rule](#5-research-notes-rule)

## 1. Active Docs

Read these first:

1. `PROJECT_BRIEF.md`
   - Short project truth: scope, pipeline, evidence, next decision, and forbidden overclaims.

2. `research/synthesis_specific_tasks.md`
   - 2026-06-21 research-mining convergence: ranked specific tasks and the full, falsifiable plan for local moment-factorized, wavelength-attributed satellite HSI change characterization.

3. `research/closest_methods_novelty.md`
   - Closest HSI subspace, covariance/ACD, unmixing, band-selection, tensor, and deep methods; safe novelty boundaries and killer baselines.

4. `CD_TAXONOMY.md`
   - Mechanism-based CD taxonomy, invariance ladder, construction-card requirements, and placement of subspace geometry.

5. `kb/README.md`
   - Entry point to the per-paper research-mining knowledge base and project evidence map.

6. `research/challenges_ranked.md`
   - Ranked HSI temporal-change challenges with problem fit, novelty pressure, risk, and decision gates.

7. `RESEARCH_RESET_AUDIT.md`
   - Critical reset document: current code truth, literature-grounded problem map, ranked thesis framings, minimum viable thesis path, and 7-day action plan.

8. `RUNBOOK.md`
   - Exact commands for setup, liveness checks, prior generation, training, evaluation, sweeps, visualization, and troubleshooting.

9. `SECOND_OPINION_RESEARCH_CONTEXT.md`
   - Neutral context package for Claude/Gemini/another reviewer. Use it when asking an external model or person to challenge the project direction.

10. `experiment_reports/oscd_spatial_ds_baseline_pressure_2026-06-18.md`
   - Current spatial-DS decision report: corrected Celik/IR-MAD pressure baselines, Band-Image rank sensitivity, label-free fusion, fixed-grid pyramid stop decision, and next evidence gate.

11. `experiment_reports/seasonal_observation_subspace_stress_test_2026-06-20.md`
   - Synthetic and real-background controlled tests of unordered, trajectory, first/second DS, local eigenspectrum, nuisance robustness, and the independent-label evidence gate.

12. `experiment_reports/oscd_core_sweep_3seed_150epoch_2026-05-03.md`
   - Curated report for the historical OSCD 3-seed neural sweep.

13. `research_learning_map.html`
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

These files are preserved as provenance. They are not the active project truth. Useful knowledge from them should be reflected in `../notes/` or the active docs.

## 3. Active Notes

The active thinking files are:

- `../notes/feedback.md`
- `../notes/my_notes.md`
- `../notes/methods.md`
- `../notes/literature.md`
- `../notes/experiments.md`
- `../notes/reference_bookmarks.md`
- `../notes/research_paper_plan.md`

## 4. Historical Archive

`docs/archive/` was removed after a final ingestion audit. Its useful knowledge was consolidated into the active `../notes/` files and project docs. The deleted tracked files remain recoverable from Git/GitHub history, but they are no longer part of the active reading path.

The old `phase1/docs/` and `phase2/docs/` folders were also removed after consolidation. Their useful commands, method details, historical result caveats, and experiment plans now live in `../notes/methods.md`, `../notes/experiments.md`, and `RUNBOOK.md`. Their generated figures were not kept as active docs; regenerate or recover from Git history only if a specific figure is needed.

## 5. Research Notes Rule

`../notes/` is the current active notes location.

`research-notes/` is a nested external notes repo/archive. It is useful background, but it should not be treated as the current project source of truth. Important current material should be extracted into `../notes/` or the active docs above. Its current useful content was ingested into active notes on 2026-06-07; delete the folder only after the user confirms the ingestion is sufficient.
