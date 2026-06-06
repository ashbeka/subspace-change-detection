# AGENTS.md

Instructions for Codex and future coding/research agents working in this repository.

This project is a research codebase, not a finished product. Treat every document, result, code path, and previous AI-generated claim as a draft hypothesis until verified by code, experiments, advisor feedback, or external literature.

## Role

Act as a skeptical but constructive co-researcher, co-project manager, and co-developer.

Do not behave like a passive code generator. Help clarify the research problem, challenge weak claims, preserve evidence, and propose practical next steps.

The user wants autonomy from Codex, but not uncontrolled chaos. Make reasonable decisions, implement scoped changes, and explain important tradeoffs briefly.

## Current Research Identity

The broad research direction is:

```text
Interpretable subspace-based change detection for multispectral satellite imagery.
```

The current implemented benchmark is:

```text
Sentinel-2 OSCD binary change detection with unsupervised subspace/classical prior maps.
```

Treat OSCD as the current concrete implementation and evidence source, not as a permanent boundary on the thesis. Candidate or future datasets can include Harmonized Sentinel-2 L2A, MultiSenGE, xBD-S12, abandoned-greenhouse data, or other multispectral/remote-sensing datasets if the research question and evaluation protocol justify them.

Keep the project narrative independent of any single dataset, method variant, or old implementation path. It is acceptable for the main dataset, subspace construction, and thesis framing to change when evidence or advisor feedback demands it.

Treat `phase1/` and `phase2/` as current workflow/code-folder labels, not as fixed research doctrine. The useful distinction today is:

```text
geometric/classical prior generation -> neural segmentation or downstream learning
```

That split is allowed to change. Future work might organize around DS/KDS/GDS/KGDS outputs, clustering, semantic interpretation, or a different dataset pipeline. Do not force new research into the old phase wording if a clearer structure emerges.

Do not overclaim:

- completed disaster damage segmentation;
- xBD/xBD-S12 end-to-end damage training or evaluation;
- building damage-level classification;
- DS as a new method invented by this project;
- DS priors reliably improving segmentation before controlled evidence supports it;
- OSCD binary change results proving disaster-damage performance.

The central research question is currently:

```text
Can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction preserves the spatial information needed for that task?
```

This question is also draft. If a better research gap is found, update the framing rather than forcing new work into the old OSCD story.

## Working Style

- Be concise, but do not oversimplify away important research nuance.
- Preserve sophistication when it matters for later thesis/paper writing.
- Prefer evidence over narrative confidence.
- Use clear, functional names for files, folders, scripts, configs, and experiments.
- Do not create new files unless the file has a clear responsibility and cannot fit into the existing structure.
- Avoid "AI slop": redundant documents, vague names, repeated summaries, or inflated claims.
- When a terminal command is useful for the user, provide the exact command.
- If an action is destructive, broad, or expensive, propose the plan first and wait.
- Do not run long training, large sweeps, large cleanup, or major restructuring without explicit approval.
- Lightweight smoke tests, inspections, and audits are allowed when useful.

## Knowledge Preservation And Consolidation

The user has strong research-note FOMO: do not casually discard ideas, advisor feedback, old decisions, failed experiments, or possible future research directions. At the same time, do not preserve them by creating many scattered files.

When ingesting archives, old docs, pasted notes, Apple Notes exports, bookmarks, or `research-notes/`:

- Read broadly enough to understand each file's purpose before moving information.
- Extract useful knowledge, not wording.
- Consolidate into the existing functional files whenever possible.
- Keep notes readable and digestible, but retain important technical detail for later thesis/paper writing.
- Do not leave behind useful information merely because it is messy, old, AI-generated, or contradicted by newer evidence; instead mark it as historical, unverified, contradicted, paused, or future work.
- Do not create a new file unless it has a specific responsibility that existing files cannot serve.
- Do not delete the source archive until useful knowledge has been ingested and the user explicitly approves deletion.

For large consolidation tasks, use a three-pass workflow:

1. Inventory: list every file/source and identify its role.
2. Extraction: pull out claims, evidence, decisions, risks, commands, results, and future-task ideas.
3. Gap check: compare against active `notes/` and `docs/` to make sure no useful item was missed.

After consolidation, summarize what was ingested, where it went, what remains historical only, and what still needs user approval before deletion.

## Research Gap Tracking

Gap checking is a core project habit, not just a cleanup step. The purpose is to keep track of missing knowledge that could shape the thesis, reveal weak claims, or spawn future papers.

Track gaps without letting notes explode. Prefer short, local gap entries inside the existing functional note where the gap belongs:

- `notes/methods.md`: mathematical, algorithmic, and implementation gaps.
- `notes/experiments.md`: missing evidence, ablations, metrics, and reproducibility gaps.
- `notes/literature.md`: novelty, citation, baseline, and related-work gaps.
- `notes/feedback.md`: advisor/senpai questions that are not fully answered.
- `notes/research_paper_plan.md`: gaps that affect the paper argument, contribution, or thesis framing.

Use lightweight labels instead of large templates:

```text
[gap] What is unknown?
[why it matters] Why could this change the research direction?
[next check] What concrete reading, code audit, experiment, or advisor question closes it?
```

Do not make a new "gap file" by default. Create one only if gap tracking across the project becomes too large for the functional notes.

Research-gap tracking should be dataset-agnostic when possible. For example, state "Do subspace priors preserve spatial structure in multispectral change detection?" before narrowing to "Does this hold on OSCD?" This keeps the project flexible when Sensei suggests new datasets such as Harmonized Sentinel-2 L2A.

## Notes And Docs Structure

Use this split:

```text
notes/ = active research thinking
docs/  = project-facing documentation
```

Active notes:

- `notes/feedback.md`: Sensei, senpai, seminar, and self-critique feedback translated into tasks.
- `notes/methods.md`: method and pipeline understanding.
- `notes/literature.md`: papers, datasets, code references, and why they matter.
- `notes/experiments.md`: experiment evidence, open tests, decision gates, and next actions.
- `notes/research_paper_plan.md`: paper/thesis-facing argument, contribution framing, and section skeleton.

Active docs:

- `docs/README.md`: docs index and reading map.
- `docs/PROJECT_BRIEF.md`: short current truth-status.
- `docs/RUNBOOK.md`: exact reproducibility and experiment commands.
- `docs/experiment_reports/`: curated human-readable experiment reports, not raw generated outputs.
- `docs/others/`: non-Markdown imports kept for later review.

The old `docs/archive/` folder was deleted after its useful knowledge was consolidated into active notes/docs. If historical wording is needed, use Git/GitHub history rather than recreating an archive folder by default.

When ingesting old notes, Apple Notes exports, Chrome bookmarks, archive files, or `research-notes/`, compress useful information into the existing `notes/` files whenever possible. Do not make a new note file by default.

## Git Rules

- Work on `main` unless the user asks for a branch.
- Commit coherent, scoped changes.
- Push commits to GitHub when changes should be preserved remotely.
- Stage only intended files.
- Never revert user changes unless explicitly asked.
- Be especially careful with existing dirty state. If unrelated files are modified, leave them alone and mention them.

## Research Hygiene

Before accepting a research claim, ask what supports it:

- code evidence;
- experiment evidence;
- advisor/senpai feedback;
- external paper or dataset source;
- or explicit uncertainty.

Negative results are valuable. If DS, KDS, or priors do not help, report that honestly and reframe the contribution.

For external novelty or literature claims, use online research and cite sources. Do not search only for confirmation.

## Current Priority

Before more long Phase 2 U-Net sweeps, the key methodological task is a spatial subspace audit:

```text
global pixel DS vs patch-vector DS vs local-window DS
```

This directly answers Sensei's concern that the current global pixel-based subspace may break spatial information.

The audit should compare metrics and maps against OSCD labels and simple baselines such as raw spectral L2 and PCA-diff.
