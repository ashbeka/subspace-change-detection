# Distillation Review

## Quick Links

- [1. Purpose](#1-purpose)
- [2. What Counts As Covered](#2-what-counts-as-covered)
- [3. Current File Inventory](#3-current-file-inventory)
- [4. Active Reading Path](#4-active-reading-path)
- [5. What Was Newly Absorbed](#5-what-was-newly-absorbed)
- [6. Source Groups](#6-source-groups)
- [7. Deletion Readiness](#7-deletion-readiness)
- [8. How You Can Verify Without Reading 200 Files](#8-how-you-can-verify-without-reading-200-files)
- [9. Remaining Checks](#9-remaining-checks)

## 1. Purpose

This file is the compact dashboard for Stage 1: active knowledge distillation.

The goal is:

```text
many scattered docs -> one active reading path -> old files ready for review
```

No old file is deleted in this stage. The active docs become the place to read
and work from; old docs become evidence to review before deletion.

## 2. What Counts As Covered

A source file is covered only if at least one of these happened:

- its unique idea, result, method detail, command, feedback, or citation was
  moved into an active doc;
- it contained only duplicate information already represented in active docs;
- it is raw provenance that should be preserved, not treated as active truth;
- it is an experiment report whose key result is in `EXPERIMENTS_RESULTS.md`
  while the full report remains as detailed evidence.

If a useful idea does not fit the current categories, the active route/method
structure must evolve. It should not be discarded just because it is awkward.

## 3. Current File Inventory

Current audit ledger:

```text
docs/source_records/distillation_file_ledger_2026-07-01.csv
```

That CSV is the file-by-file proof surface. It contains one row per non-active
file with path, extension, size, hash, group, role, suggested destination,
status, and first headings.

Current count after moving active docs to the root `docs/` folder:

| Group | Files |
|---|---:|
| `docs/experiment_reports/` | 116 |
| `docs/source_records/` | 59 |
| `docs/pending_deletion_review/` | 49 |
| `docs/figures/` | 10 |
| Total non-active files in ledger | 234 |

The active root docs themselves are not counted as files to distill.

## 4. Active Reading Path

Use these files first:

1. `docs/README.md`
2. `docs/CURRENT_RESEARCH_DIRECTION.md`
3. `docs/RESEARCH_ROUTES.md`
4. `docs/EXPERIMENTS_RESULTS.md`
5. `docs/METHODS_REFERENCE.md`
6. `docs/LITERATURE_BASELINES_DATASETS.md`
7. `docs/HUMAN_NOTES_AND_FEEDBACK.md`
8. `docs/PERSONAL_RESEARCH_NOTES.md`
9. `docs/COMMANDS.md`
10. `docs/DISTILLATION_REVIEW.md`

Supporting folders are not the first reading path.

## 5. What Was Newly Absorbed

This pass added missing nuance from high-priority human and source records:

| Source theme | Active destination |
|---|---|
| Sensei asked for first/second DS, geodesic, temporal satellite subspaces, survey reading, more bands than RGB, and deeper original technique | `HUMAN_NOTES_AND_FEEDBACK.md`, `RESEARCH_ROUTES.md`, `METHODS_REFERENCE.md` |
| Senpai asked what the task is and what happens after a heatmap | `HUMAN_NOTES_AND_FEEDBACK.md`, `RESEARCH_ROUTES.md` |
| Personal motivation around Gaza/reconstruction and disaster-response usefulness | `HUMAN_NOTES_AND_FEEDBACK.md`, `RESEARCH_ROUTES.md` |
| SSC as union-of-subspaces baseline, pseudo-label route, and temporal recovery clustering route | `RESEARCH_ROUTES.md`, `METHODS_REFERENCE.md`, `LITERATURE_BASELINES_DATASETS.md` |
| UAV/edge deployment as a possible win axis | `RESEARCH_ROUTES.md`, `HUMAN_NOTES_AND_FEEDBACK.md`, `LITERATURE_BASELINES_DATASETS.md` |
| Disaster-resilience planning, MCDA/MCDM, infrastructure-routing/application layer | `RESEARCH_ROUTES.md`, `METHODS_REFERENCE.md`, `LITERATURE_BASELINES_DATASETS.md` |
| Diffusion-map pseudotime analogy for damage/recovery progression | `RESEARCH_ROUTES.md`, `METHODS_REFERENCE.md`, `LITERATURE_BASELINES_DATASETS.md` |
| Multi-sensor ideas: SAR/InSAR, LiDAR/DEM/topography, UAV imagery | `RESEARCH_ROUTES.md`, `LITERATURE_BASELINES_DATASETS.md` |
| Construction-change data as a possible alternative to disaster labels | `HUMAN_NOTES_AND_FEEDBACK.md` |
| QA report distinction between spectral-change priors and semantic OSCD masks | `HUMAN_NOTES_AND_FEEDBACK.md`, `METHODS_REFERENCE.md` |
| MultiSenGE earliest/latest pairing risk and seasonality protocol | `METHODS_REFERENCE.md`, `HUMAN_NOTES_AND_FEEDBACK.md` |
| Slow Feature Subspace paper details: SFA weight-vector subspace, canonical-angle comparison, temporal-order preservation | `METHODS_REFERENCE.md`, `LITERATURE_BASELINES_DATASETS.md` |

The most important correction from this pass is that open-ended routes are now
kept in `RESEARCH_ROUTES.md`; they are not limited to a fixed list of ten.

## 6. Source Groups

### Human And Source Records

Preserve these for now. They are raw provenance, not clutter:

- Apple/Slack/source-record Markdown files;
- `All research notes - this is the updated one not the one on the laptop.docx`;
- `qa_report_response_2025-11-20.pdf`;
- `student_feedback_channel4_2025-11-20.xlsx`;
- submitted seminar/report source files;
- original bookmark/source imports if present.

Their useful content should be represented in active docs, but the raw source
records should not be deleted without explicit approval.

### Experiment Reports

Keep for now. The active ledger stores the compact result memory, but detailed
reports still contain commands, figures, settings, and exact context.

Future cleanup can keep only the strongest reports after the result/figure
reproducibility audit.

### Pending Deletion Review

This folder contains mostly old notes, AI-generated synthesis, and previous
documentation variants. These are the first deletion candidates after user
review because their useful knowledge should now live in the active docs.

### Figures

Figures are not automatically redundant. They should be reviewed with the
experiment or seminar material they support.

## 7. Deletion Readiness

Ready for user review first:

- old AI synthesis files under `docs/pending_deletion_review/`;
- old route/ranking/agenda files after comparing against `RESEARCH_ROUTES.md`;
- old method explanation files after comparing against `METHODS_REFERENCE.md`;
- old literature/bookmark files after comparing against
  `LITERATURE_BASELINES_DATASETS.md`.

Not ready for deletion in this stage:

- raw source records;
- curated experiment reports with unique figures/commands;
- data, code, outputs, or reference code;
- any file whose ledger row still looks unclear after spot-checking.

## 8. How You Can Verify Without Reading 200 Files

Use this procedure:

1. Open `docs/source_records/distillation_file_ledger_2026-07-01.csv`.
2. Filter by `status` or `destination`.
3. Spot-check a few files from each group:
   - one human/source record;
   - one old notes file;
   - one old AI synthesis file;
   - one experiment report;
   - one figure record.
4. Search a term you care about in active docs, for example:

```powershell
rg -n "open-vocabulary|SSC|pseudotime|UAV|MCDA|IR-MAD|second-order|geodesic" docs --glob "*.md"
```

5. If the concept is absent or too thin, the file is not ready for deletion.

This is the practical guarantee: every source file has a ledger row, and every
important concept should have a searchable active-doc location.

## 9. Remaining Checks

Before any deletion:

- run the ledger count again and confirm every non-active file is listed;
- review the CSV rows marked unclear or preserve;
- spot-check old AI synthesis docs for any unique experiment number;
- confirm exact bookmark import files are either regenerated or recovered from
  Git history if needed;
- choose whether seminar scripts should be kept as presentation material or
  compressed into future slide notes.
