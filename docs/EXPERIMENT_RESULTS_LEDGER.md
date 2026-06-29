# Experiment Results Ledger

## Quick Links

- [1. Purpose](#purpose)
- [2. Evidence Status](#evidence-status)
- [3. Main Ledger](#main-ledger)
- [4. Current Evidence Interpretation](#current-evidence-interpretation)
- [5. Unverified / AI-Reported Claims](#unverified--ai-reported-claims)
- [6. Delete-Candidate Report Groups After Absorption](#delete-candidate-report-groups-after-absorption)

## Purpose

This is the compact evidence table. Old reports can be read only when detail is
needed. If an experiment is not in this ledger, it is not part of the active
project memory.

## Evidence Status

| Status | Meaning |
|---|---|
| verified | rerunnable or backed by report/code/output |
| needs rerun | interesting but not independently reproduced |
| negative | tested and not a current positive route |
| diagnostic | useful for explanation/failure analysis |
| candidate | promising but not thesis-ready |

## Main Ledger

| Date | Lane | Question | Data | Best result | Baseline/control | Status | Decision/report |
|---|---|---|---|---|---|---|---|
| 2026-06-13 | spatial DS | does patch/window preserve spatial information better than global pixel DS? | OSCD Beirut | patch5 AUROC 0.8923, AP 0.2729 | PCA-diff AP 0.5284 | diagnostic | `oscd_spatial_subspace_sweep_core5_2026-06-14.md` |
| 2026-06-18 | Band-Image DS | does band-image sample definition help? | OSCD all cities | Band-Image rank8 AUROC 0.8412, AP 0.2340 | PCA-diff AP 0.2541 | candidate | `oscd_spatial_flatbands_allcities_2026-06-18.md` |
| 2026-06-18 | classical pressure | does DS survive strong classical baselines? | OSCD | fusion AP 0.2780 | smoothed PCA AP 0.2679 | diagnostic/candidate | `oscd_spatial_ds_baseline_pressure_2026-06-18.md` |
| 2026-06-22 | matched spatial controls | is Band-Image DS distinct from matched spatial nulls? | OSCD | DS beats matched Gram/projector/cross controls | smoothed PCA still strong | candidate | `oscd_band_image_matched_spatial_controls_2026-06-22.md` |
| 2026-06-22 | xBD-S12 transfer | does band-image geometry transfer to disaster data? | xBD-S12 | projector retrieves 24.7% damaged pixels at 5% review budget | IR-MAD 17.8% | candidate | `xbd_s12_external_validation_2026-06-22.md` |
| 2026-06-23 | Successive Saab-DS | do local label-free features make DS useful? | OSCD official split | AP 0.3420, AUROC 0.8861 | PCA-diff AP 0.3067, matched L2 AP 0.3067 | verified internal | `oscd_successive_subspace_learning_ds_2026-06-23.md` |
| 2026-06-23 | train-fitted/external Saab gate | does Saab-DS survive frozen fitting and xBD transfer? | OSCD, xBD-S12 | OSCD train-fitted AP 0.3381 | pair-adaptive AP 0.3420 | verified internal; external negative | `successive_saab_trainfit_external_gate_2026-06-23.md` |
| 2026-06-19/20 | temporal first/second DS | can first/second/geodesic quantities be computed on sequences? | MultiSenGE/IPOL | curves and decompositions generated | raw residual/event maps stronger for localization | diagnostic | `multispectral_temporal_difference_subspaces_2026-06-19.md` |
| 2026-06-21 | RTW | does time-warped subspace help temporal crop/sequence comparison? | MultiSenGE/BreizhCrops | selected RTW AP 0.7052/0.6596 | PCA cross-recon 0.8128/0.8264 | negative | `breizhcrops_rtw_natural_label_transfer_2026-06-21.md` |
| 2026-06-21 | HSI local moment geometry | does moment-factorized local HSI orientation help? | HSI datasets | mechanism tests useful | detector transfer negative | negative/diagnostic | `hsi_local_moment_geometry_2026-06-21.md` |
| 2026-06-22 | HSI Band-Image transfer | does Band-Image geometry generalize to HSI? | Hermiston/Benton/etc. | scene-dependent | not consistent | paused | `hsi_band_image_transfer_2026-06-22.md` |
| 2026-06-22 | SpaceNet7 RGB transfer | does RGB building appearance geometry transfer? | SpaceNet7 | raw L2 leads AP | DS loses to controls | negative | `spacenet7_band_image_transfer_2026-06-22.md` |
| 2026-06-22 | cross-branch synthesis | which routes worked? | all reports | Successive Saab-DS and xBD projector are strongest | many negatives | verified synthesis | `cross_branch_research_evidence_matrix_2026-06-22.md` |
| 2026-06-28 | learned DS-prior fusion | does DS help U-Net beyond no-DS controls? | OSCD | reported positive | not independently rerun | needs rerun | absorbed Claude result; verify next |

## Current Evidence Interpretation

1. The strongest **internal** detector route is Successive Saab-DS on OSCD.
2. The strongest **external** geometry result is xBD-S12 candidate localization
   with Band-Image/projector geometry.
3. Temporal DS/geodesic is Sensei-aligned but currently a descriptor, not a
   winning detector.
4. The learned-prior result may become important, but it is not trusted until
   reproduced in the active main code path.

## Unverified / AI-Reported Claims

These items were generated during Claude/Codex exploration and are useful, but
not thesis claims until reproduced or audited.

| Claim | Source group | Current label | Required action |
|---|---|---|---|
| DS-specific multi-prior U-Net improves over no-DS and matched-cross controls | `docs/research/*DSprior*`, Claude temporal synthesis | `needs rerun` | reproduce exact split/config/seeds in main |
| teacher-student / pseudo-label distillation could produce label-scarce payoff | `MASTER_NARRATIVE_2026-06-22.md` | `hypothesis` | design controlled low-label experiment |
| MultiSenGE pretraining then OSCD fine-tuning may improve label scarcity | `MASTER_NARRATIVE_2026-06-22.md` | `hypothesis` | require multi-seed and leakage-safe protocol |
| temporal DS can become an ACCV-style paper route | `DESIGN_TEMPORAL_DS_ACCV2026.md` | `candidate` | needs labeled sequence and baseline ladder |
| HSI N<<B distributional change is a possible niche | `bet1_design.md`, `challenges_ranked.md` | `candidate/paused` | needs real labeled bitemporal HSI benchmark |

## Delete-Candidate Report Groups After Absorption

Do not delete yet. These are likely candidates after this ledger is checked:

| Group | Reason |
|---|---|
| old seminar drafts in `docs/research/` | current framing now lives in active control docs |
| Claude AI synthesis docs under `docs/research/claude_temporal/` | useful points are lane/method/ledger rows |
| overlapping old `notes/*.md` | being replaced by active docs |
| duplicated report narratives | summarized here; keep only reports with unique figures/details |
| `research-notes/` nested repo | old note structure already distilled repeatedly |
