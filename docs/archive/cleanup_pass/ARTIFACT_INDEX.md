# Artifact Index

Status: active artifact tracking document  
Updated: 2026-05-03

This index prevents old outputs from becoming accidental ground truth. Generated outputs stay ignored by git, but important artifact paths should be documented before any cleanup.

## Rules

- Do not delete an output folder until it is listed here.
- Mark interrupted runs clearly.
- Treat old results as evidence to audit, not proof.
- Keep Phase 1 prior-map folders needed by Phase 2 configs.
- Keep at least one fresh successful controlled sweep before deleting older comparable runs.

## Active / Important Artifacts

| path | status | produced by | keep/archive/delete later | reason |
|---|---|---|---|---|
| `phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422/` | completed | `run_phase2_sweep.ps1 -Preset core -Epochs 150 -Seeds 1234,1235,1236 -ProgressBars` | keep | Current controlled OSCD core reproduction. Main evidence source. |
| `phase2/outputs/smoke_e0_e1_20260503_040723/` | completed smoke | manual E0/E1 1-epoch train+eval | keep until full sweep is audited | Liveness evidence only; not performance evidence. |
| `phase2/outputs/runs_gpu_150ep_20251215_233309/` | completed old artifact | older 150-epoch GPU sweep | keep until reproduced | Reports E1 raw+DS over E0 raw, but must be audited against fresh sweep. |
| `phase1/outputs/oscd_saved_priors_fast/` | required prior maps | Phase 1 fast OSCD prior generation | keep | Used by current Phase 2 core configs. |
| `phase1/outputs/oscd_saved_full/` | classical baseline maps/results | Phase 1 full OSCD baselines | keep | Needed for Celik/IR-MAD comparison and old artifact audit. |
| `phase1/outputs/oscd_saved_priors_fast_eig/` | eig prior maps | Phase 1 eig variant | keep for now | Needed only if eig experiments are revisited. |

## Interrupted / Non-Final Sweep Artifacts

| path | status | produced by | keep/archive/delete later | reason |
|---|---|---|---|---|
| `phase2/outputs/sweep_core_150ep_E0_E1_repro_20260503_042530/` | interrupted | first core sweep attempt | delete later after v5 succeeds | Stopped to improve progress visibility. |
| `phase2/outputs/sweep_core_150ep_repro_v3_20260503_044813/` | interrupted | progress-bar test sweep | delete later after v5 succeeds | Stopped during progress-display debugging. |
| `phase2/outputs/sweep_core_150ep_repro_v4_20260503_045516/` | interrupted/test | progress-display test sweep | delete later after v5 succeeds | Not final evidence. |
| `phase2/outputs/sweep_core_150ep_repro_v4_20260503_045721/` | interrupted/test | progress-display test sweep | delete later after v5 succeeds | Not final evidence. |

## Older Local Artifacts To Audit

| path | status | keep/archive/delete later | reason |
|---|---|---|---|
| `phase2/outputs/sweep_overnight_full_eig_3seeds_150ep_v2/` | old local artifact | audit before deleting | May contain broader/full/eig results. Need summary before deciding. |
| `phase2/outputs/oscd_seg_E0_raw_rerun/` | old local run | audit later | Could be redundant after v5. |
| `phase2/outputs/oscd_seg_geodesic/` | old local run | audit later | Related to geodesic prior experiments. |
| `phase2/outputs/reentry_smoke_20260502_020139/` | old smoke | delete later after notes are merged | Re-entry liveness evidence likely superseded. |
| `phase1/outputs/reentry_fast_priors_20260502_020139/` | old re-entry output | delete later after notes are merged | Likely superseded by canonical prior folders. |
| `phase1/outputs/_smoke_reentry_geodesic/` | smoke | delete later | Smoke-only. |
| `phase1/outputs/oscd_geodesic_priors_rerun/` | geodesic rerun | audit later | May be useful if geodesic prior is revisited. |
| `phase1/outputs/oscd_saved_priors_fast_rerun/` | rerun priors | audit later | Could be redundant with canonical fast priors. |

## What Counts As Final Evidence

A Phase 2 result is thesis-usable only if it has:

- `config_used.yaml`
- `best.ckpt`
- `train_log.json`
- `run_metadata.json`
- `eval/oscd_seg_eval_results.json`
- `eval/oscd_seg_eval_summary.csv`
- seed recorded
- git hash recorded
- same split and evaluation protocol as comparison runs

The current final evidence source is the completed v5 3-seed core sweep. Result audit:

```text
docs/RESULTS_OSCD_CORE_SWEEP_20260503.md
```

## Cleanup Gate

Do not run aggressive cleanup until:

1. v5 sweep result remains committed in docs.
2. Per-city and qualitative checks are done or explicitly deferred.
3. Interrupted progress-bar test runs are confirmed unnecessary.
4. The user explicitly approves output deletion.
