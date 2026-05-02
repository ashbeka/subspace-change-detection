# Cleanup Tracker

Created: 2026-05-03
Status: cleanup started, destructive actions not yet approved

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

No files were deleted. No outputs were cleaned. No code was moved.

## Active Files To Keep

| path | why |
|---|---|
| `phase1/` | Active prior-generation implementation. |
| `phase2/` | Active supervised OSCD segmentation implementation. |
| `docs/PROJECT_MASTER_BRIEF.md` | Current project truth-status document. |
| `docs/REPRODUCIBILITY_CHEATSHEET.md` | Current operational reproducibility guide. |
| `RUN_PIPELINE.md` | Still useful legacy PowerShell pipeline; should be merged or shortened later. |
| `phase1/outputs/oscd_saved_priors_fast/` | Required prior maps for current Phase 2 configs. Ignored by git. |
| `phase1/outputs/oscd_saved_full/` | Contains classical baseline priors and summaries. Ignored by git. |
| `phase2/outputs/runs_gpu_150ep_20251215_233309/` | Important old artifact to audit/reproduce before deleting. Ignored by git. |
| `phase2/outputs/smoke_e0_e1_20260503_040723/` | Fresh liveness smoke run. Ignored by git. |

## Candidate Docs To Merge Or Archive Later

These should not be deleted yet. First decide whether each has unique content not already in `PROJECT_MASTER_BRIEF.md` or `REPRODUCIBILITY_CHEATSHEET.md`.

| path | proposed action | reason | approval needed |
|---|---|---|---|
| `docs/ADVERSARIAL_REENTRY_AUDIT.md` | archive later | Valuable skeptical audit, now mostly superseded for orientation. | Yes |
| `docs/IMPLEMENTATION_STATUS.md` | merge/archive later | Useful old status, but can conflict with current code. | Yes |
| `docs/NEXT_STEP_DECISION_MEMO.md` | merge/archive later | Next-step content should become roadmap entries. | Yes |
| `docs/PROJECT_REENTRY_SYNTHESIS.md` | archive later | Re-entry context, not canonical current truth. | Yes |
| `docs/PROJECT_RESET_DECISION.md` | archive later | Historical reset marker. | Yes |
| `docs/PROJECT_UNDERSTANDING_GUIDE.md` | merge/archive later | Likely overlaps with master brief. | Yes |
| `phase1/docs/phase1_run_guide.md` | merge useful commands into cheat sheet, then keep phase-local | Phase-local details may remain useful. | Yes |
| `phase2/docs/phase2_run_guide.md` | merge useful commands into cheat sheet, then keep phase-local | Phase-local details may remain useful. | Yes |
| `TEMP_DS_PRIMER.md` | merge/archive later | Temporary explanatory doc. | Yes |
| `CODEBASE_AUDIT.md` | archive later | Some claims are stale relative to current code. | Yes |
| `PIPELINE_RERUN_LOG.txt` | archive later | Historical log, not canonical instructions. | Yes |

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

Recommended next cleanup after review:

1. Create `docs/archive/`.
2. Move superseded root and re-entry docs into `docs/archive/`.
3. Leave `PROJECT_MASTER_BRIEF.md`, `REPRODUCIBILITY_CHEATSHEET.md`, and a future `ROADMAP.md` as the active docs.
4. Update any links broken by archival moves.
5. Do not touch code or outputs in that commit.

This is a restructure and should be approved before execution.
