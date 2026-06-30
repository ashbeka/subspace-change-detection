# Docs Start Here

## Quick Links

- [1. First Reading Path](#1-first-reading-path)
- [2. What Each Active Doc Owns](#2-what-each-active-doc-owns)
- [3. Supporting Material](#3-supporting-material)
- [4. Distillation Rule](#4-distillation-rule)
- [5. Deletion Review Rule](#5-deletion-review-rule)

## 1. First Reading Path

Read these files first, in this order:

1. `CURRENT_RESEARCH_DIRECTION.md`
   - Current truth, strongest evidence, safe claims, forbidden claims, and immediate next gate.

2. `RESEARCH_ROUTES.md`
   - All research routes, including current, candidate, weak, parked, and rejected routes.

3. `EXPERIMENTS_RESULTS.md`
   - Compact ledger of completed experiments, pending experiments, negative results, commands, and decision gates.

4. `METHODS_REFERENCE.md`
   - Subspace construction, formulas, source-to-code trails, method cards, and implementation caveats.

5. `LITERATURE_BASELINES_DATASETS.md`
   - Papers, datasets, baselines, bookmark-derived resources, and reference-code context.

6. `HUMAN_NOTES_AND_FEEDBACK.md`
   - Sensei/senpai/seminar feedback plus distilled human-facing decisions.

7. `PERSONAL_RESEARCH_NOTES.md`
   - Rough user note intake. Preserve intent here, then translate stable implications into the other active docs.

8. `COMMANDS.md`
   - Exact reproducible commands.

9. `DISTILLATION_REVIEW.md`
   - Coverage dashboard showing what old files were reviewed, what was absorbed, and what remains for deletion review.

## 2. What Each Active Doc Owns

The active docs are the only normal reading path. They should contain no unnecessary provenance chatter and no repeated copies of the same idea.

| Active doc | Responsibility |
|---|---|
| `CURRENT_RESEARCH_DIRECTION.md` | current project framing, safest claims, strongest evidence, current next action |
| `RESEARCH_ROUTES.md` | evolving research problem routes and decision gates |
| `METHODS_REFERENCE.md` | formulas, dimensions, construction cards, implementation references |
| `EXPERIMENTS_RESULTS.md` | results, metrics, commands, pending tests, go/no-go outcomes |
| `LITERATURE_BASELINES_DATASETS.md` | papers, datasets, baselines, bookmarks, reference code |
| `HUMAN_NOTES_AND_FEEDBACK.md` | advisor/senpai/seminar feedback and human priorities |
| `PERSONAL_RESEARCH_NOTES.md` | rough personal notes before distillation |
| `COMMANDS.md` | command-only runbook |
| `DISTILLATION_REVIEW.md` | coverage and deletion-readiness dashboard |

## 3. Supporting Material

Supporting material is preserved, but it is not the first reading path.

| Folder | Purpose |
|---|---|
| `source_records/` | raw provenance: Apple Notes, Slack exports, reports, submitted files, bookmark exports, absorbed Claude overlap copies |
| `experiment_reports/` | detailed reports, figures, metrics, and assets for specific experiments |
| `pending_deletion_review/` | old notes, old AI drafts, old project docs, old research material, and old learning-map files waiting for review |
| `figures/` | reusable figures and generated visual aids |

## 4. Distillation Rule

Every non-active file should be read and routed by usefulness:

```text
human feedback -> HUMAN_NOTES_AND_FEEDBACK.md
research problem or possible contribution -> RESEARCH_ROUTES.md
method/formula/pipeline/sample definition -> METHODS_REFERENCE.md
experiment/result/command/metric/decision gate -> EXPERIMENTS_RESULTS.md or COMMANDS.md
paper/dataset/baseline/reference code -> LITERATURE_BASELINES_DATASETS.md
current thesis claim/risk -> CURRENT_RESEARCH_DIRECTION.md
rough personal wording -> PERSONAL_RESEARCH_NOTES.md
coverage/fate -> DISTILLATION_REVIEW.md
```

If an important idea does not fit those destinations, the routing system should evolve. Do not discard useful information just because it was not anticipated.

## 5. Deletion Review Rule

No source record or old file should be deleted merely because it is old, AI-generated, repetitive, or messy.

Deletion becomes reasonable only when:

1. the file is listed in `DISTILLATION_REVIEW.md`;
2. any unique knowledge has been distilled into the active docs;
3. duplicate content has no unique nuance left;
4. the user approves deletion.

Exact experiment reports and source records may remain preserved even after their conclusions are distilled, because they carry provenance, figures, commands, and auditability.
