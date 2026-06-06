# Docs Index

`docs/` is for project-facing documentation: concise project status, exact commands, accepted results, and historical archive.

Use `../notes/` for active research thinking: feedback, method understanding, literature triage, and experiment ideas.

## Active Docs

Read these first:

1. `PROJECT_BRIEF.md`
   - Short project truth: scope, pipeline, evidence, next decision, and forbidden overclaims.

2. `RUNBOOK.md`
   - Exact commands for setup, liveness checks, prior generation, training, evaluation, sweeps, visualization, and troubleshooting.

3. `results/OSCD_CORE_SWEEP_2026-05-03.md`
   - Accepted summary of the current major OSCD 3-seed sweep.

## Active Notes

The active thinking files are:

- `../notes/feedback.md`
- `../notes/methods.md`
- `../notes/literature.md`
- `../notes/experiments.md`
- `../notes/research_paper_plan.md`

## Archive Rule

`archive/` contains historical re-entry, cleanup, and old planning documents. Its useful knowledge has been consolidated into the active `../notes/` files. Keep it searchable until deletion is explicitly approved, but do not treat it as active truth. If an archive file conflicts with active docs or notes, trust the active files unless code or fresh experiments prove otherwise.

## Research Notes Rule

`../notes/` is the current active notes location.

`research-notes/` is a nested external notes repo/archive. It is useful background, but it should not be treated as the current project source of truth. Important current material should be extracted into `../notes/` or the active docs above.
