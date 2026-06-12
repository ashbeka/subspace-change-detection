# Prior Ingestion Ledgers

This file preserves detailed historical ingestion ledgers moved out of `notes/experiments.md` during the final organization pass. It is provenance, not the active experiment plan.

## Research-Notes Ingestion Ledger

Source repo: nested `research-notes/` at commit `44f2671`, clean against `origin/main` at ingestion time.

Status:

- Ingested into active notes on 2026-06-07.
- The nested repo was not edited.
- Deletion of `research-notes/` remains a later user-approved cleanup step.
- Old claims that raw+DS was the best prior are historical because the newer 3-seed v5 sweep did not reproduce that simple claim.
- Final comprehensive audit on 2026-06-07 checked all `30` tracked files against the ingestion ledger and found no missing source files. Two over-compressed details were restored explicitly: the Hiraoka/RTW Sensei lead and the OSCD_MSI/TorchGeo loader caveat.

### Ingestion ledger

| source file | active destination | retained knowledge |
|---|---|---|
| `research-notes/.gitignore` | none | empty file; no research content |
| `research-notes/README.md` | `notes/README.md`, `AGENTS.md` | notes workflow and source-of-truth caution |
| `research-notes/paths_menu.md` | `notes/research_paper_plan.md`, `notes/experiments.md` | active/warm/cold scope model, now treated as flexible guardrail |
| `research-notes/decisions_log.md` | `notes/research_paper_plan.md`, `notes/experiments.md` | ADR history, benchmark discipline, scope-lock cautions |
| `research-notes/gaps_towatch.md` | `notes/methods.md`, `notes/experiments.md`, `notes/literature.md` | method, benchmark, temporal, and future-extension gaps |
| `research-notes/coverage_matrix.csv` | `notes/methods.md`, `notes/experiments.md`, `notes/literature.md` | idea inventory compressed into method hooks and backlog |
| `research-notes/glossary.md` | `notes/methods.md`, `notes/literature.md` | acronym/method definitions already covered or folded in |
| `research-notes/Drafts_digest.md` | `notes/research_paper_plan.md`, `notes/methods.md`, `notes/experiments.md` | broad proposal material bucketed into active/future tracks |
| `research-notes/spec_snippets.md` | `notes/methods.md`, `notes/experiments.md` | delta, SSC, MCDA, and payload formulas as future hooks |
| `research-notes/master/current_scope.md` | `notes/research_paper_plan.md`, `notes/experiments.md` | old active scope and warning that OSCD is not damage mapping |
| `research-notes/master/master_outline.md` | `notes/research_paper_plan.md` | thesis skeleton, now corrected by newer evidence |
| `research-notes/master/master_proposal.md` | `notes/research_paper_plan.md`, `notes/methods.md` | broad historical proposal, formulas, future system ideas |
| `research-notes/master/master_skeleton.md` | `notes/research_paper_plan.md` | paper outline intent |
| `research-notes/master/session_audit_2026-03-24.md` | `notes/research_paper_plan.md`, `notes/experiments.md`, `notes/feedback.md` | March 2026 reset reasoning and guardrails |
| `research-notes/master/appendix_ds_math.tex` | `notes/methods.md` | DS notation, projection score, first/second temporal deltas |
| `research-notes/master/slides_equations.tex` | `notes/methods.md` | reusable equations: deltas, SSC, Dice/CE, MCDA |
| `research-notes/notes/sensei_notes.md` | `notes/feedback.md`, `notes/literature.md`, `notes/methods.md` | Sensei guidance on DS, surveys, second-order DS, TPAMI, RTW/SSA/SFA |
| `research-notes/notes/senpais_notes.md` | `notes/feedback.md`, `notes/methods.md` | task clarity, post-map purpose, SSC justification, multi-image subspaces |
| `research-notes/notes/my_notes.md` | `notes/feedback.md`, `notes/methods.md`, `notes/experiments.md` | safe vs risky paths, SSC hypotheses, MultiSenGE/xBD-S12 thoughts |
| `research-notes/notes/old_notes.md` | `notes/feedback.md`, `notes/research_paper_plan.md` | broad old ideas moved to cold/future archive |
| `research-notes/phases/phase1_report.md` | `notes/methods.md`, `notes/experiments.md`, `docs/RUNBOOK.md` | Phase 1 implementation details and old result caveats |
| `research-notes/spec/spec_phase1_ds_oscd.md` | `notes/methods.md`, `notes/experiments.md` | Phase 1 spec, thresholding, baselines, and reproducibility requirements |
| `research-notes/refs_links/benchmark_watchlist.md` | `notes/literature.md`, `notes/experiments.md` | OSCD, Metric-CD, xBD, xBD-S12, MapFormer, benchmark discipline |
| `research-notes/refs_links/Links.md` | `notes/reference_bookmarks.md`, `notes/literature.md` | bookmark-style reference leads |
| `research-notes/refs_links/initial_refs.bib` | `notes/literature.md` | citation seeds for OSCD, MultiSenGE, Celik, IR-MAD, U-Net, ResNet, xBD |
| `research-notes/refs_links/fixed.html` | `notes/reference_bookmarks.md` | older bookmark/reference export style |
| `research-notes/refs_links/full_eh_referenecs.xml` | `notes/reference_bookmarks.md`, `notes/literature.md` | broad reference leads, not active truth |
| `research-notes/scripts/render_deltas.py` | `notes/methods.md`, `notes/experiments.md` | synthetic figure idea only; not migrated as code |
| `research-notes/scripts/render_mcda_demo.py` | `notes/methods.md` | MCDA demo idea kept as future decision-layer hook |
| `research-notes/scripts/render_payload_chart.py` | `notes/methods.md` | payload formula kept as future deployment hook |

### Second-pass ingestion audit

User-triggered second pass on 2026-06-07:

- Re-read all `30` tracked files from the nested `research-notes/` repo at commit `44f2671`, including the empty `.gitignore`.
- Compared high-density archive files against `notes/methods.md`, `notes/experiments.md`, `notes/literature.md`, `notes/feedback.md`, and `notes/research_paper_plan.md`.
- Strengthened the active notes where the first pass was too compressed.
- Did not edit the nested `research-notes/` repo.

Details recovered or made more explicit:

- Phase 1 baseline and threshold details: PCA-diff rank/energy choices, Celik window/PCA/k-means defaults, per-tile Otsu, and train-calibrated global threshold grid.
- MultiSenGE pairing details: `earliest_latest`, `adjacent`, and `first_mid_last`, with seasonality/cloud caution.
- Quick-figure hooks: Delta1/Delta2 synthetic grids, MCDA toy heatmaps, and payload-vs-embedding curves.
- Cold-extension specifications: xView2 scale numbers to verify, xBD-S12 damage metrics, quadratic-weighted kappa, synchronized augmentations, and edge/server payload formulas.
- Benchmark-watchlist specifics: Metric-CD protocol caveat, xBD-S12 paired-data claim, MapFormer prior-guided caution, emergency transfer-learning relevance, and comparison-protocol checklist.
- Historical conflict note: old residual/eig equivalence claims must be overridden by the newer corrected-DS audit unless reproduced under the current code.
- Final detail check: exact Hiraoka/RTW context and OSCD_MSI/TorchGeo loader alternatives were made visible in active notes so the old repo is not needed for those leads.

Second-pass conclusion:

- The active notes now retain the old repo's technical hooks without reviving the old broad thesis narrative.
- Remaining deletion of `research-notes/` is still a separate cleanup decision, not performed here.

### Thesis-usable run provenance

For any result used in a thesis or paper, preserve:

- config path and copied `config_used.yaml`;
- git commit hash and dirty/clean state;
- seed, epochs, device, and GPU/CPU;
- Phase 1 prior-map root;
- checkpoint path;
- `train_log.json`;
- `run_metadata.json`;
- evaluation JSON/CSV;
- exact split and city list;
- notes on interruption, resume, or overwrite.

One-epoch smoke runs prove liveness only. They are not performance evidence.

### Old artifact evidence, not reproduction

The old artifact:

```text
phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv
```

reported a single-seed E1 improvement over E0:

| artifact row | IoU | F1 | AUROC | PR-AUC |
|---|---:|---:|---:|---:|
| E0 raw-only | 0.2230 | 0.3428 | 0.8687 | 0.4315 |
| E1 raw+DS | 0.2725 | 0.4009 | 0.8743 | 0.4358 |
| E1 - E0 | +0.0495 | +0.0580 | +0.0056 | +0.0043 |

Metadata in the archive tied this old result to seed `1234`, CUDA Torch `2.9.1+cu126`, and git hash `735deac0caa1b9fe381c2ea98e09b93761534415`. The result is useful history, but the later 3-seed v5 sweep did not reproduce the E1 improvement. Treat the old artifact as an audit target, not proof.

### Old liveness and prior-generation evidence

Archive re-entry smoke evidence included:

- fast Phase 1 prior generation under `phase1/outputs/reentry_fast_priors_20260502_020139`;
- Phase 2 smoke output under `phase2/outputs/reentry_smoke_20260502_020139`;
- CUDA availability on an RTX 4070 SUPER for that re-entry run;
- fast Phase 1 test AUROC means: `pca_diff 0.8134`, `pixel_diff 0.7559`, `ds_projection 0.7551`, `ds_cross_residual 0.5563`;
- geodesic smoke output under `phase1/outputs/_smoke_reentry_geodesic`, with finite maps for a few train/val/test cities.

The old `PIPELINE_RERUN_LOG.txt` also showed one environment with CPU-only Torch (`2.9.1+cpu`, `cuda_available=False`). Any "GPU rerun" claim from that log is unsafe unless checked against metadata.

### Cleanup retention rules

Keep these until explicitly audited or intentionally retired:

```text
phase1/outputs/oscd_saved_priors_fast
phase1/outputs/oscd_saved_priors_fast_eig
phase1/outputs/oscd_saved_full
phase2/outputs/runs_gpu_150ep_20251215_233309
phase2/outputs/smoke_e0_e1_20260503_040723
phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422
```

Also audit before deleting:

```text
phase1/outputs/reentry_fast_priors_20260502_020139
phase1/outputs/_smoke_reentry_geodesic
phase1/outputs/oscd_geodesic_priors_rerun
phase1/outputs/oscd_saved_priors_fast_rerun
phase2/outputs/oscd_seg_E0_raw_rerun
phase2/outputs/oscd_seg_geodesic
phase2/outputs/reentry_smoke_20260502_020139
phase2/outputs/sweep_overnight_full_eig_3seeds_150ep_v2
```

Interrupted progress-bar or aborted sweep folders can be deleted later, but only after confirming they do not contain unique metrics, checkpoints, figures, or logs.

Likely delete-later after audit:

```text
phase2/outputs/sweep_core_150ep_E0_E1_repro_20260503_042530
phase2/outputs/sweep_core_150ep_repro_v3_20260503_044813
phase2/outputs/sweep_core_150ep_repro_v4_*
```

Do not rename old output folders during active analysis. Document the mapping instead, because old configs, summaries, and notes may refer to those legacy paths.

## Secondary Analysis Backlog

These items came from the old cleanup roadmap and remain useful, but they should follow the spatial subspace audit unless needed for a paper figure:

- Inspect per-city metrics and qualitative examples for `E0_raw_unet`, `E1_raw_ds`, `E3_raw_ds_pca`, and `S0_siamese`.
- Stratify where priors help or hurt by city type, land cover, seasonality, shadows/clouds, and change density.
- Add probability-map saving or rerun inference for true threshold tuning; current CSV-only summaries cannot do full threshold sweeps after the fact.
- Decide whether to run `E4_raw_pixel`, `E5_raw_celik`, and `E6_raw_irmad` in the same 3-seed controlled style.
- If manual PowerShell result comparison becomes repetitive, add a small result-analysis script rather than hand-copying tables.
- Add a config validator for enabled prior names versus available prior-map folders before long sweeps.
- Improve output/config naming only after result interpretation is stable.
- Extract shared channel-count/model-building logic later if repeated config mistakes appear.
- Treat dependency pinning as a reproducibility task before final paper runs.

### Deleted archive sources reviewed before removal

The useful material from these former archive files was folded into active notes or docs before deletion:

```text
docs/archive/cleanup_pass/ARTIFACT_INDEX.md
docs/archive/cleanup_pass/CLEANUP_TRACKER.md
docs/archive/cleanup_pass/PIPELINE_EXPLAINED.md
docs/archive/cleanup_pass/PROJECT_STRUCTURE_REVIEW.md
docs/archive/cleanup_pass/README.md
docs/archive/cleanup_pass/ROADMAP.md
docs/archive/consolidated_into_notes_20260606/RESEARCH_PLAN.md
docs/archive/consolidated_into_notes_20260606/SUBSPACE_METHOD_NOTES.md
docs/archive/reentry/ADVERSARIAL_REENTRY_AUDIT.md
docs/archive/reentry/IMPLEMENTATION_STATUS.md
docs/archive/reentry/NEXT_STEP_DECISION_MEMO.md
docs/archive/reentry/PROJECT_REENTRY_SYNTHESIS.md
docs/archive/reentry/PROJECT_RESET_DECISION.md
docs/archive/reentry/PROJECT_UNDERSTANDING_GUIDE.md
docs/archive/root_legacy/CODEBASE_AUDIT.md
docs/archive/root_legacy/PIPELINE_RERUN_LOG.txt
docs/archive/root_legacy/REMEMBER_TO_ACTIVATE_ENV.txt
docs/archive/root_legacy/RUN_PIPELINE.md
docs/archive/root_legacy/TEMP_DS_PRIMER.md
```

### Where archive content lives now

| Archive content type | Active location |
|---|---|
| Current project truth, scope, overclaim boundaries | `docs/PROJECT_BRIEF.md`, `notes/research_paper_plan.md` |
| Exact reproducibility commands and old pipeline commands | `docs/RUNBOOK.md` |
| Subspace implementation, DS/KDS/KGDS, spatial-risk explanation | `notes/methods.md`, `notes/feedback.md` |
| Sensei/senpai feedback and unanswered questions | `notes/feedback.md` |
| Experiment evidence, old artifact metrics, cleanup retention | `notes/experiments.md`, `docs/experiment_reports/oscd_core_sweep_3seed_150epoch_2026-05-03.md` |
| Literature, baseline papers, and reference-code leads | `notes/literature.md` |
| Paper/thesis framing, contribution, decision gates | `notes/research_paper_plan.md` |
| Historical cleanup and structure policy | `docs/README.md`, `AGENTS.md` |

## Phase Docs Consolidation Ledger

The old `phase1/docs/` and `phase2/docs/` folders were reviewed and removed on 2026-06-07 to keep the project reading path centralized.

| source | active destination | retained knowledge |
|---|---|---|
| `phase1/docs/spec_phase1_ds_oscd.md` | `notes/methods.md`, `notes/experiments.md`, `docs/RUNBOOK.md`, `project_cli.py` | old Phase 1 scope, OSCD/MultiSenGE distinction, 13-band preprocessing, valid-mask/z-score rules, DS/PCA-diff/Celik/IR-MAD baseline definitions, thresholding protocol, metrics, and reproducibility requirements |
| `phase1/docs/phase1_report.md` | `notes/methods.md`, `notes/experiments.md` | Phase 1 implementation map, score-map output paths, baseline caveats, old test AUROC/F1/IoU summaries, and the conflict that old residual/eig DS claims predate the corrected DS audit |
| `phase1/docs/phase1_run_guide.md` | `docs/RUNBOOK.md`, `project_cli.py` | OSCD prior-generation commands, saved-map layout, MultiSenGE visualization commands, and how Phase 2 loads Phase 1 priors |
| `phase1/docs/phase1_scripts_cli.md` | `project_cli.py`, `docs/RUNBOOK.md` | old script cheat sheet; replaced by central CLI wrappers and raw command fallbacks |
| `phase1/docs/spec_summary_notes.txt` | `notes/methods.md`, `notes/experiments.md` | short summary of Phase 1 assumptions; retained as caveats rather than current truth |
| `phase2/docs/spec_phase2_oscd_seg.md` | `notes/methods.md`, `notes/experiments.md`, `docs/RUNBOOK.md`, `project_cli.py` | OSCD segmentation data flow, raw/prior channel definitions, model families, losses, metrics, config matrix, and future damage-adapter boundary |
| `phase2/docs/phase2_report.md` | `notes/experiments.md`, `notes/research_paper_plan.md`, `docs/experiment_reports/oscd_core_sweep_3seed_150epoch_2026-05-03.md` | old single-seed Phase 2 table, per-city example logic, visualization strategy, and warning that old raw+DS improvement is superseded by the newer 3-seed sweep |
| `phase2/docs/phase2_run_guide.md` | `docs/RUNBOOK.md`, `project_cli.py` | training/evaluation/visualization commands and future knobs such as longer training, ImageNet pretraining, and pseudo-label pretraining |
| `phase2/docs/phase2_scripts_cli.md` | `project_cli.py`, `docs/RUNBOOK.md` | old command cheat sheet; replaced by `project_cli.py` wrappers and runbook commands |
| `phase1/docs/figs/`, `phase2/docs/figs/` | none active | generated figures only; removed from active docs to reduce clutter. Recover from Git history or regenerate from outputs/scripts if needed |

Important retained caveat:

- Old phase docs say residual-stack DS outperformed eig/canonical DS and that single-seed raw+DS improved OSCD segmentation. Those statements are historical. Current active evidence is the corrected DS audit plus the 3-seed v5 sweep; do not cite the old phase docs as current truth.
