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
- Prefer one-line copy-paste commands for user-facing execution. Multi-line PowerShell commands are acceptable for readability in docs, but always provide a one-line equivalent when the user is expected to run it.
- Prefer `project_cli.py` for common project checks, run listings, formula tests, wrapped sweeps, and cleanup previews. Use raw script commands when debugging or when the CLI does not expose the needed option.
- When suggesting a concrete next step, experiment, audit, reading task, code task, or decision gate, persist it in the appropriate active note during the same turn whenever it is more than a one-off answer. Do not leave important future work only in chat.
- If an action is destructive, broad, or expensive, propose the plan first and wait.
- Do not run long training, large sweeps, large cleanup, or major restructuring without explicit approval.
- Lightweight smoke tests, inspections, and audits are allowed when useful.

## Knowledge Preservation And Consolidation

The user has strong research-note FOMO: do not casually discard ideas, advisor feedback, old decisions, failed experiments, or possible future research directions. At the same time, do not preserve them by creating many scattered files.

`notes/my_notes.md` is the user's rough personal note intake. When the user has a raw thought, unclear idea, half-formed research doubt, or copied personal note, it should go there first. Codex should then translate the useful content into the functional notes without pretending the rough note is already verified truth.

When processing `notes/my_notes.md`:

- preserve the user's intent and wording style enough that the note remains recognizable;
- extract actionable implications into `feedback.md`, `methods.md`, `literature.md`, `experiments.md`, or `research_paper_plan.md`;
- avoid inventing a neat narrative that the user did not actually decide;
- do not erase rough notes merely because they have been translated unless the user explicitly asks for cleanup.

Future note intake may move from local Markdown to Notion or another notes app via MCP. If that happens, treat the external note source as the new rough-note intake layer, but keep the same workflow: preserve the raw note, extract useful research content, and distill it into the functional repo notes without creating a second scattered knowledge system.

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

## Suggestion Persistence

Chat is not durable enough for research planning. If Codex recommends something that the project may need later, it must either:

- update the relevant active note immediately; or
- explicitly say why it is only a transient answer and does not need persistence.

Default locations:

- `notes/experiments.md`: experiment/audit/backlog tasks, metrics, commands, decision gates, and "do not run X until Y" constraints.
- `notes/methods.md`: method relationships, formulas, implementation caveats, source-to-code interpretation, and subspace construction details.
- `notes/literature.md`: papers, baselines, datasets, reference code, citations, and novelty/comparison pressure.
- `notes/feedback.md`: advisor/senpai/seminar questions and unanswered concerns.
- `notes/research_paper_plan.md`: thesis framing, safe claims, forbidden claims, and argument structure.
- `docs/RUNBOOK.md`: commands the user is expected to rerun.
- `docs/PROJECT_BRIEF.md`: only major status changes that affect the current project truth.

When reviewing recent chat for missed tasks, convert only actionable items into notes. Avoid copying conversational wording or creating new files. If the task already exists, strengthen the existing entry instead of duplicating it.

## Reference Code Discovery

Reference code added by the user, senpais, advisors, old projects, or external repos should be treated as research material before it is treated as clutter.

When new reference code appears:

- Inventory it by folder, language, entry points, dependencies, and implemented method families.
- Check whether active project code imports it. If not, label it as reference-only rather than useless.
- Identify what concepts it implements, such as PCA, KPCA, DS, GDS, KDS, KGDS, CCA/KCCA, S3CCA/TRCCA, MSM/KMSM, RTW, SFA, Grassmann geometry, attribution, or metric/decomposition utilities.
- Map each useful concept to a possible project role: paper-to-code verification, baseline implementation, future experiment, advisor explanation, visualization, or thesis related work.
- Record where the useful knowledge belongs in the active structure:
  - `notes/methods.md` for algorithm details and adaptation ideas;
  - `notes/experiments.md` for candidate experiments and evidence gates;
  - `notes/literature.md` for paper/code provenance and citation context;
  - `notes/research_paper_plan.md` only if it affects the paper-facing research path.
- Do not assume reference code is ground truth. Compare it with papers, toy tests, dimensions, and active code behavior.
- Do not delete or archive reference code just because it is unused by runtime. First explain what it can teach us, what is irrelevant, what is duplicated, and what would be required to adapt it.

The goal is project self-scanning: when files or folders are added, Codex should ask "what can this teach or enable?" before deciding "what can be removed?"

## Notes And Docs Structure

Use this split:

```text
notes/ = active research thinking
docs/  = project-facing documentation
```

Active notes:

- `notes/feedback.md`: Sensei, senpai, seminar, and self-critique feedback translated into tasks.
- `notes/my_notes.md`: personal rough-note intake before ideas are translated into functional notes.
- `notes/methods.md`: method and pipeline understanding.
- `notes/literature.md`: papers, datasets, code references, and why they matter.
- `notes/experiments.md`: experiment evidence, open tests, decision gates, and next actions.
- `notes/reference_bookmarks.md`: bookmark triage, Zotero-first queue, and organized Chrome import pointer.
- `notes/research_paper_plan.md`: paper/thesis-facing argument, contribution framing, and section skeleton.

Active docs:

- `docs/README.md`: docs index and reading map.
- `docs/PROJECT_BRIEF.md`: short current truth-status.
- `docs/RUNBOOK.md`: exact reproducibility and experiment commands.
- `docs/experiment_reports/`: curated human-readable experiment reports, not raw generated outputs.
- `docs/source_records/`: original human-facing records received/submitted during the project, raw imports, and importable organized source records. Preserve these, but do not treat them as current truth without checking active notes.

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

Do not solve a problem that the project invented only to justify a preferred method. Start from real remote-sensing change-detection problems, advisor feedback, benchmark limitations, and literature gaps; then decide whether DS, GDS, KDS, KGDS, CCA/KCCA, SSC, Green Learning, neural methods, or another tool actually helps.

## Paper-To-Code Verification

The project is still being reframed, so do not build a large brittle test suite around every old code path. However, keep and prioritize small formula-verification tests for methods translated from papers.

Use tests and audits to answer:

- Does the implementation match the paper's mathematical object, dimensions, and edge cases?
- Does a reference implementation or toy example produce the same behavior?
- Does the code fail gracefully for equal/shared subspaces, invalid ranks, masks, and degenerate inputs?
- Does the produced score map mean what the method claims it means?

This is especially important for DS, GDS, KDS, KGDS, PCA-diff, Celik PCA-kmeans, IR-MAD, CVA, CCA/KCCA/S3CCA, and any future spatial or temporal subspace variant. Formula-verification tests are not proof that the research problem is solved; they are guards against hallucinated or paper-inaccurate implementations.

For every new method implementation, preserve a source-to-code trail:

```text
source material -> math object -> satellite adaptation -> code path -> test -> one-city output -> thesis claim
```

For every niche method file, include a concise top-level source/provenance docstring. It should say which paper, reference code, or established method family guided the implementation; what was adapted for this project; and what verification status the implementation has. Do not over-comment generic utilities, but do document DS/GDS/KDS/KGDS, PCA-diff, Celik, IR-MAD, CCA/KCCA/S3CCA, CVA, spatial DS variants, U-Net/Siamese baselines, and future paper-derived methods.

Before coding a paper-derived or reference-code-derived method, identify the paper section/equation or reference file, the exact satellite sample definition, input/output dimensions, projection or scoring equation, code files/functions to edit, and the toy or smoke check that will prove the implementation is at least shape- and formula-consistent.

When reporting an implemented experiment to the user, explain what source material guided it, what was adapted for Sentinel-2 or another dataset, and what remains an implementation choice rather than paper theory.

For subspace experiments, always explain subspace construction first. Use the construction-card fields from `notes/methods.md`: variant name, source/reference, sample unit, input matrix/tensor, subspace count, basis shape, fitting method, comparison/score, spatial information preserved/lost, code path, and verification. This is seminar-critical; do not assume "we build a subspace" is enough explanation.

## Current Priority

Before more long Phase 2 U-Net sweeps, the key methodological task is a spatial subspace comparison guided by this working problem:

```text
Can spatially aware Difference Subspace construction preserve the spatial structure of multispectral Sentinel-2 images well enough to produce interpretable changed-area evidence, and where does it help or fail compared with raw spectral difference, PCA-diff, Celik/IR-MAD, and neural change-detection baselines?
```

The original immediate experiment track was:

```text
global pixel DS -> patch-vector DS -> local-window DS -> multiscale spatial subspace pyramid -> fair classical baselines -> optional neural/prior follow-up
```

This directly answers Sensei's concern that the current global pixel-based subspace may break spatial information.

The comparison should report metrics and maps against OSCD labels and simple baselines such as raw spectral L2 and PCA-diff.

As of 2026-06-22, the OSCD matched-control study and frozen xBD-S12 external
transfer are complete. Spatial band-image projector distance behaves mainly as
candidate/building-localization evidence. Canonical DS beats matched
cross-reconstruction on five unseen test events but not consistently on 11
training events, so the DS-specific effect is event-dependent. Raw L2 is
stronger for damage-vs-intact discrimination.

The current experiment track is now:

```text
another independent event gate
-> optional neural-prior test
```

Rank/centering, naive geometry/radiometry fusion, IR-MAD pressure, fixed review
budgets, object retrieval, registration shifts, and available cloud/date
checks are complete. Do not tune new
combinations on the five already inspected xBD-S12 test events. Develop any
new mechanism on training disasters and seek another independent gate before
making a confirmatory detector claim.
