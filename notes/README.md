# Research Notes Index

## Table Of Contents

- [Files](#files)
- [Where To Add A New Note](#where-to-add-a-new-note)
- [Rules](#rules)

These notes are the active project brain. They are for understanding, planning, and deciding what to implement next.

`docs/` is still for project-facing documentation such as commands, accepted result summaries, and polished project status. `notes/` is where research thinking should live.

## Files

- `feedback.md`: Sensei, senpai, seminar, and self-critique feedback translated into concrete research tasks.
- `my_notes.md`: Personal rough notes and idea intake before Codex translates them into the functional notes.
- `methods.md`: How the current pipeline and methods work, including DS, GDS, KDS, KGDS, OSCD, and Phase 2 segmentation.
- `literature.md`: Papers, codebases, datasets, and why each matters.
- `experiments.md`: Experiment evidence, open experiment ideas, decision gates, and next actions.
- `reference_bookmarks.md`: Chrome bookmark triage, Zotero-first reading queue, and organized import-file pointer.
- `research_paper_plan.md`: Paper-facing synthesis: problem, gap, method, experiments, evidence, contribution framing, and paper skeleton.

## Where To Add A New Note

- Advisor or seminar comment: add it to `feedback.md`.
- Rough personal thought, unclear idea, or half-formed research doubt: add it to `my_notes.md` first.
- Method explanation or code understanding: add it to `methods.md`.
- Paper, bookmark, reference code, or dataset: add it to `literature.md`.
- Large bookmark batches or Zotero triage: add the summary to `reference_bookmarks.md`, then promote must-cite items into `literature.md`.
- Result, experiment idea, failed run, or next test: add it to `experiments.md`.
- Paper/thesis argument, contribution framing, or section outline: add it to `research_paper_plan.md`.
- Exact command that should be reused: keep it in `docs/RUNBOOK.md`, then reference it from `experiments.md` if needed.

## Rules

- Keep notes short enough to reread.
- Keep enough detail to preserve research sophistication for later paper writing.
- Prefer claims tied to code, experiment output, advisor feedback, or paper references.
- Do not preserve old wording just because it was written confidently.
- Do not make a new note file unless the existing four categories cannot reasonably hold it.
- Treat `research-notes/`, Apple Notes, Chrome bookmarks, and archived docs as input sources to ingest, not active truth.
- Treat `my_notes.md` as the active human note intake. Codex should periodically translate it into the other functional notes without erasing the original rough thought unless the user asks.
- Treat every project story as a draft. The code, notes, paper plan, and folder structure may change when evidence improves.
- Treat `phase1/` and `phase2/` as current workflow labels, not a fixed research ontology. The present split is prior generation versus neural/downstream learning, but future work can reorganize around clearer methods or datasets.
- Codex should act as a skeptical co-researcher and co-developer: propose structure changes when they reduce confusion, but avoid creating new files unless they have a clear responsibility.
