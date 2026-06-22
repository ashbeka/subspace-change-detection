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

2. `RESEARCH_RESET_AUDIT.md`
   - Critical reset document: current code truth, literature-grounded problem map, ranked thesis framings, minimum viable thesis path, and 7-day action plan.

3. `RUNBOOK.md`
   - Exact commands for setup, liveness checks, prior generation, training, evaluation, sweeps, visualization, and troubleshooting.

4. `SECOND_OPINION_RESEARCH_CONTEXT.md`
   - Neutral context package for Claude/Gemini/another reviewer. Use it when asking an external model or person to challenge the project direction.

5. `experiment_reports/oscd_spatial_ds_baseline_pressure_2026-06-18.md`
   - Current spatial-DS decision report: corrected Celik/IR-MAD pressure baselines, Band-Image rank sensitivity, label-free fusion, fixed-grid pyramid stop decision, and next evidence gate.

6. `experiment_reports/seasonal_observation_subspace_stress_test_2026-06-20.md`
   - Synthetic and real-background controlled tests of unordered, trajectory, first/second DS, local eigenspectrum, nuisance robustness, and the independent-label evidence gate.

7. `experiment_reports/oscd_core_sweep_3seed_150epoch_2026-05-03.md`
   - Curated report for the historical OSCD 3-seed neural sweep.

8. `experiment_reports/spacenet7_temporal_subspace_validation_2026-06-21.md`
   - Independent-label SpaceNet 7 gate for rolling first/second trajectory DS; records the negative result against standardized radiometric controls.

9. `experiment_reports/multisenge_rtw_invariance_gate_2026-06-21.md`
   - Controlled MultiSenGE test of Randomized Time Warping against snapshot-subspace, Fourier/harmonic, DTW/TWDTW, M-SSA, and radiometric controls; records the no-go decision for a natural-transition RTW study.

10. `experiment_reports/breizhcrops_rtw_natural_label_transfer_2026-06-21.md`
   - Natural-label and timing-invariance transfer study on all four official BreizhCrops regions with nested RTW selection and 22 killer controls.

11. `experiment_reports/hsi_local_moment_geometry_2026-06-21.md`
   - Four-dataset held-out test of local hyperspectral mean/scale/eigenspectrum/orientation factorization, DS projection, and basis-invariant band attribution.

12. `experiment_reports/cross_branch_research_evidence_matrix_2026-06-22.md`
   - Consolidated Codex/Claude/Antigravity method-results matrix, Sensei-task status, ranked research problems, safe claims, and seminar recommendation.

13. `experiment_reports/oscd_band_image_matched_spatial_controls_2026-06-22.md`
   - Decisive matched-null experiment for Band-Image DS versus spatial Gram, projector, cross-reconstruction, spatial PCA, IR-MAD, and label-free fusions.

14. `experiment_reports/xbd_s12_external_validation_2026-06-22.md`
   - Frozen event-disjoint xBD-S12 transfer, damage/localization task decomposition, matched-control statistics, and boundary stress test.

15. `experiment_reports/hsi_band_image_transfer_2026-06-22.md`
   - Fixed rank-11 transfer to four labeled HSI pairs, including the positive Hermiston regime and failed broad transfer.

16. `experiment_reports/spacenet7_band_image_transfer_2026-06-22.md`
   - Nine-AOI, 197-transition RGB building-appearance gate that rejects generic projector/DS transfer.

15. `research_learning_map.html`
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
