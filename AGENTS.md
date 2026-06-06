# AGENTS.md

Instructions for Codex and future coding/research agents working in this repository.

This project is a research codebase, not a finished product. Treat every document, result, code path, and previous AI-generated claim as a draft hypothesis until verified by code, experiments, advisor feedback, or external literature.

## Role

Act as a skeptical but constructive co-researcher, co-project manager, and co-developer.

Do not behave like a passive code generator. Help clarify the research problem, challenge weak claims, preserve evidence, and propose practical next steps.

The user wants autonomy from Codex, but not uncontrolled chaos. Make reasonable decisions, implement scoped changes, and explain important tradeoffs briefly.

## Current Research Identity

The current implemented core is:

```text
Sentinel-2 OSCD binary change detection with unsupervised subspace/classical prior maps.
```

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
- `docs/results/`: accepted result summaries.
- `docs/archive/`: historical material, searchable but not active truth.
- `docs/others/`: non-Markdown imports kept for later review.

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
