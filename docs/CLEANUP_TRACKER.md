# Cleanup Tracker

Created: 2026-05-03
Status: active cleanup tracker; destructive actions not approved

This tracker keeps cleanup deliberate. It separates low-risk documentation cleanup from repository restructuring and artifact deletion.

## Cleanup Rules

- Do not delete datasets, outputs, checkpoints, or notes without an explicit keep/delete decision.
- Do not move active code until reproducibility commands are stable.
- Prefer adding indexes and replacing stale entry points before archiving files.
- Preserve historical docs until their useful content is merged or explicitly marked obsolete.
- Every cleanup commit should be small enough to review.

## Cleanup Started In This Pass

| path | action | reason | risk |
|---|---|---|---|
| `README.md` | rewritten as a concise current entry point | Old README contained stale/broad scope language and mojibake; new README points to current truth docs. | Low: historical details remain in other docs. |
| `docs/REPRODUCIBILITY_CHEATSHEET.md` | added | Creates one operational command sheet for setup, liveness checks, Phase 1, Phase 2, sweeps, visualization, and cleanup. | Low. |
| `docs/CLEANUP_TRACKER.md` | added | Gives cleanup a controlled, reviewable process. | Low. |
| `docs/ROADMAP.md` | added | Creates one active project-management document. | Low. |
| `docs/ARTIFACT_INDEX.md` | added | Documents generated outputs before any output cleanup. | Low. |
| `docs/PIPELINE_EXPLAINED.md` | added | Explains the full implemented pipeline in a student-readable way. | Low. |
| `docs/RESULTS_OSCD_CORE_SWEEP_20260503.md` | added | Captures the completed 3-seed OSCD core sweep. | Low. |
| `docs/PROJECT_STRUCTURE_REVIEW.md` | added | Reviews current structure and proposes future cleanup. | Low. |
| `docs/archive/reentry/` | added | Holds superseded reset/re-entry/status docs. | Low: archived, not deleted. |
| `docs/archive/root_legacy/` | added | Holds superseded root-level logs, primers, and old pipeline docs. | Low: archived, not deleted. |

No files were deleted. No outputs were cleaned. No code was moved.

## Active Files To Keep

| path | why |
|---|---|
| `phase1/` | Active prior-generation implementation. |
| `phase2/` | Active supervised OSCD segmentation implementation. |
| `docs/PROJECT_MASTER_BRIEF.md` | Current project truth-status document. |
| `docs/ROADMAP.md` | Current experiment/decision roadmap. |
| `docs/REPRODUCIBILITY_CHEATSHEET.md` | Current operational reproducibility guide. |
| `docs/ARTIFACT_INDEX.md` | Current generated-artifact inventory and cleanup gate. |
| `phase1/outputs/oscd_saved_priors_fast/` | Required prior maps for current Phase 2 configs. Ignored by git. |
| `phase1/outputs/oscd_saved_full/` | Contains classical baseline priors and summaries. Ignored by git. |
| `phase2/outputs/runs_gpu_150ep_20251215_233309/` | Important old artifact to audit/reproduce before deleting. Ignored by git. |
| `phase2/outputs/smoke_e0_e1_20260503_040723/` | Fresh liveness smoke run. Ignored by git. |

## Candidate Docs To Merge Or Archive Later

These should not be deleted yet. First decide whether each has unique content not already in `PROJECT_MASTER_BRIEF.md` or `REPRODUCIBILITY_CHEATSHEET.md`.

| path | proposed action | reason | approval needed |
|---|---|---|---|
| `docs/archive/reentry/ADVERSARIAL_REENTRY_AUDIT.md` | archived | Valuable skeptical audit, now mostly superseded for orientation. | Done |
| `docs/archive/reentry/IMPLEMENTATION_STATUS.md` | archived | Useful old status, but can conflict with current code. | Done |
| `docs/archive/reentry/NEXT_STEP_DECISION_MEMO.md` | archived | Next-step content is now represented in `ROADMAP.md`. | Done |
| `docs/archive/reentry/PROJECT_REENTRY_SYNTHESIS.md` | archived | Re-entry context, not canonical current truth. | Done |
| `docs/archive/reentry/PROJECT_RESET_DECISION.md` | archived | Historical reset marker. | Done |
| `docs/archive/reentry/PROJECT_UNDERSTANDING_GUIDE.md` | archived | Superseded by master brief and roadmap. | Done |
| `phase1/docs/phase1_run_guide.md` | merge useful commands into cheat sheet, then keep phase-local | Phase-local details may remain useful. | Yes |
| `phase2/docs/phase2_run_guide.md` | merge useful commands into cheat sheet, then keep phase-local | Phase-local details may remain useful. | Yes |
| `docs/archive/root_legacy/TEMP_DS_PRIMER.md` | archived | Temporary explanatory doc. | Done |
| `docs/archive/root_legacy/CODEBASE_AUDIT.md` | archived | Some claims are stale relative to current code. | Done |
| `docs/archive/root_legacy/PIPELINE_RERUN_LOG.txt` | archived | Historical log, not canonical instructions. | Done |
| `docs/archive/root_legacy/RUN_PIPELINE.md` | archived | Superseded by reproducibility cheat sheet. | Done |
| `docs/archive/root_legacy/REMEMBER_TO_ACTIVATE_ENV.txt` | archived | Superseded by README and cheat sheet setup commands. | Done |

## Output Cleanup Gate

Before deleting any ignored output folders, create an artifact index with:

- path
- generated date if known
- command/config if known
- whether it has been reproduced
- whether it is referenced in docs/thesis
- keep/archive/delete recommendation

Do not run aggressive cleanup until this table exists.

Safe preview only:

```powershell
powershell -ExecutionPolicy Bypass -File clean_house.ps1 -WhatIf
```

Aggressive preview only:

```powershell
powershell -ExecutionPolicy Bypass -File clean_house.ps1 -Aggressive -WhatIf
```

## Proposed Next Cleanup Commit

Recommended next cleanup after the active sweep finishes:

1. Inspect per-city metrics and qualitative examples from the v5 sweep.
2. Decide whether to run threshold tuning or E4/E5/E6 classical baselines.
3. Do not delete outputs until `ARTIFACT_INDEX.md` is updated and deletion is explicitly approved.

The docs archive restructure is applied in the worktree and should be committed with the v5 result audit.
