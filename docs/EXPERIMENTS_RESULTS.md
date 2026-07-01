# Experiment Results Ledger

## Quick Links

- [1. Purpose](#purpose)
- [2. Evidence Status](#evidence-status)
- [3. Main Ledger](#main-ledger)
- [4. Detailed Metrics Preserved From Old Reports](#detailed-metrics-preserved-from-old-reports)
- [5. Current Evidence Interpretation](#current-evidence-interpretation)
- [6. Unverified / AI-Reported Claims](#unverified--ai-reported-claims)
- [7. Candidate Next Gates](#candidate-next-gates)
- [8. Delete-Candidate Report Groups After Absorption](#delete-candidate-report-groups-after-absorption)

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
| 2026-06-18 | Band-Image score ablation | do score variants improve threshold behavior? | OSCD all cities | Band-Image norm AP `0.2340`, Otsu F1 `0.2007` | Band-Image DS Otsu F1 `0.1129` | diagnostic | `oscd_band_image_ds_score_ablation_2026-06-18.md` |
| 2026-06-18 | classical pressure | does DS survive strong classical baselines? | OSCD | fusion AP 0.2780 | smoothed PCA AP 0.2679 | diagnostic/candidate | `oscd_spatial_ds_baseline_pressure_2026-06-18.md` |
| 2026-06-22 | matched spatial controls | is Band-Image DS distinct from matched spatial nulls? | OSCD | DS beats matched Gram/projector/cross controls | smoothed PCA still strong | candidate | `oscd_band_image_matched_spatial_controls_2026-06-22.md` |
| 2026-06-22 | xBD-S12 transfer | does band-image geometry transfer to disaster data? | xBD-S12 | projector retrieves 24.7% damaged pixels at 5% review budget | IR-MAD 17.8% | candidate | `xbd_s12_external_validation_2026-06-22.md` |
| 2026-06-23 | Successive Saab-DS | do local label-free features make DS useful? | OSCD official split | AP 0.3420, AUROC 0.8861 | PCA-diff AP 0.3067, matched L2 AP 0.3067 | verified internal | `oscd_successive_subspace_learning_ds_2026-06-23.md` |
| 2026-06-23 | train-fitted/external Saab gate | does Saab-DS survive frozen fitting and xBD transfer? | OSCD, xBD-S12 | OSCD train-fitted AP 0.3381 | pair-adaptive AP 0.3420 | verified internal; external negative | `successive_saab_trainfit_external_gate_2026-06-23.md` |
| 2026-06-19/20 | temporal first/second DS | can first/second/geodesic quantities be computed on sequences? | MultiSenGE/IPOL | curves and decompositions generated | raw residual/event maps stronger for localization | diagnostic | `multispectral_temporal_difference_subspaces_2026-06-19.md` |
| 2026-06-21 | RTW | does time-warped subspace help temporal crop/sequence comparison? | MultiSenGE/BreizhCrops | selected RTW AP 0.7052/0.6596 | PCA cross-recon 0.8128/0.8264 | negative | `breizhcrops_rtw_natural_label_transfer_2026-06-21.md` |
| 2026-06-21 | RTW invariance gate | does RTW beat snapshot subspace under controlled timing variation? | MultiSenGE | snapshot AP `0.8156`, RTW AP `0.8078` | RTW delta `-0.0078` | negative | `multisenge_rtw_invariance_gate_2026-06-21.md` |
| 2026-06-20 | seasonal observation stress | does local eigenspectrum help seasonal event observation? | SITS/seasonal sequence | local eigenspectrum AP `0.688` | NDMI AP `0.680`, first DS AP `0.456` | diagnostic | `seasonal_observation_subspace_stress_test_2026-06-20.md` |
| 2026-06-21 | HSI local moment geometry | does moment-factorized local HSI orientation help? | HSI datasets | mechanism tests useful | detector transfer negative | negative/diagnostic | `hsi_local_moment_geometry_2026-06-21.md` |
| 2026-06-22 | HSI Band-Image transfer | does Band-Image geometry generalize to HSI? | Hermiston/Benton/etc. | scene-dependent | not consistent | paused | `hsi_band_image_transfer_2026-06-22.md` |
| 2026-06-22 | SpaceNet7 RGB transfer | does RGB building appearance geometry transfer? | SpaceNet7 | raw L2 leads AP | DS loses to controls | negative | `spacenet7_band_image_transfer_2026-06-22.md` |
| 2026-06-21 | SpaceNet7 temporal geometry | does rolling RGB trajectory DS help? | SpaceNet7 | second-DS orthogonal AP `0.1127` | radiometric fusion AP `0.1910` | negative | `spacenet7_temporal_subspace_validation_2026-06-21.md` |
| 2026-06-22 | cross-branch synthesis | which routes worked? | all reports | Successive Saab-DS and xBD projector are strongest | many negatives | verified synthesis | `cross_branch_research_evidence_matrix_2026-06-22.md` |
| 2026-06-28 | learned DS-prior fusion | does DS help U-Net beyond no-DS controls? | OSCD | reported positive | not independently rerun | needs rerun | absorbed Claude result; verify next |

## Detailed Metrics Preserved From Old Reports

These are compact extractions from old notes/reports. They prevent old files
from staying active only because they contain one useful number.

### OSCD Supervised / Prior Sweeps

| Item | Preserved result | Interpretation |
|---|---|---|
| early canonical/global DS | canonical DS AUROC `0.6246`; pixel spectral difference AUROC `0.7559`; PCA-diff AUROC `0.8134` | global pixel DS was weak |
| v5 3-seed supervised sweep | raw U-Net `E0` PR-AUC `0.4331`; raw+DS `E1` PR-AUC `0.3974`; raw+DS+PCA `E3` IoU `0.2460`; Siamese PR-AUC `0.4461` | older DS-prior path was not enough |
| Claude-side DS-prior table | bands `0.4825`; DS-fusion `0.5128`; no-DS fusion `0.4804`; cross-fusion `0.4870` | interesting but `needs rerun` |
| fusion-scarcity claim | DS-specific at label budgets `n=2` and `n=14`; neutral/tie at `n=4/7` | do not claim until reproduced |
| MultiSenGE pseudo-label pretraining | scratch beat DS-fusion/SPCA pretraining at every budget; DS-fusion was only "less bad" | negative for pretraining claim |
| normalization/band subset | z-score much stronger than min-max; RGB+NIR+SWIR roughly matched all-13; all-13+DS-fusion best in that Claude run | useful ablation lead, not final claim |

### xBD-S12 / Matched Controls

| Item | Preserved result | Interpretation |
|---|---|---|
| Band-Image matched controls | DS AP `0.2410`; Gram `0.1417`; projector `0.1024`; cross `0.2153`; DS-cross delta `+0.0257` | DS effect exists in matched-control OSCD setting |
| DS fusion vs cross fusion | test delta `+0.0115`, `p=0.0098` | unverified positive, rerun before claim |
| xBD projector AP | projector global AP `0.03675`, mean event AP `0.03015`; IR-MAD mean event AP `0.02649`; DS-cross delta `+0.00448` | small external candidate-localization edge |
| xBD 5% review budget | test damaged-pixel recall projector `0.247` vs IR-MAD `0.178` | candidate triage, not damage classification |
| xBD object recall | test projector `0.358` vs IR-MAD `0.209`; train projector `0.452` | high recall but specificity caveat remains |
| registration stress | AP `0.03612 -> 0.03379 -> 0.03198` under increasing shifts | not registration-invariant |

### Negative / Diagnostic Closures

| Item | Preserved result | Interpretation |
|---|---|---|
| component-wise pseudo-change filtering | Dubai DS AP `0.3922 -> 0.4344`; Beirut `0.3597 -> 0.3375`; Milano `0.1452 -> 0.0925` | diagnostic only; not robust unsupervised method |
| Benton change-type clustering | polar-CVA greatly beat DS directions | DS did not solve semantic/change-type clustering |
| real HSI Hermiston diagnostic | CVA `0.981` vs DS `0.534` in Claude report | HSI route needs stronger mechanism |
| synthetic RTW diagnostic | DTW `0.979` vs RTW `0.549` in Claude report | RTW did not beat simpler sequence controls |
| attribution diagnostic | DS `0.177-0.257` vs per-band `0.600-1.000` in Claude report | bandwise/simple attribution can dominate |
| small-sample N<<B diagnostic | SET_SUBSPACE `0.652-0.673` vs patch_mean_SAM `0.994-0.997` in Claude report | small-sample subspace idea not yet enough |
| SpaceNet7 RGB transfer | DS lost to controls; temporal geometry AP `0.1127` vs radiometric fusion `0.1910` | RGB building appearance transfer closed for now |
| Successive Saab transfer | train-fitted OSCD AP/AUROC/Otsu F1 `0.3381/0.8912/0.3315`; seed APs `0.3375,0.3381,0.3389`; xBD best successive AP `0.01900` below projector `0.03015` and IR-MAD `0.02649` | internal OSCD positive; external successive transfer negative |

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
| DS-specific multi-prior U-Net improves over no-DS and matched-cross controls | `docs/pending_deletion_review/old_research_material/*DSprior*`, Claude temporal synthesis | `needs rerun` | reproduce exact split/config/seeds in main |
| teacher-student / pseudo-label distillation could produce label-scarce payoff | `MASTER_NARRATIVE_2026-06-22.md` | `hypothesis` | design controlled low-label experiment |
| MultiSenGE pretraining then OSCD fine-tuning may improve label scarcity | `MASTER_NARRATIVE_2026-06-22.md` | `hypothesis` | require multi-seed and leakage-safe protocol |
| temporal DS can become an ACCV-style paper route | `DESIGN_TEMPORAL_DS_ACCV2026.md` | `candidate` | needs labeled sequence and baseline ladder |
| HSI N<<B distributional change is a possible niche | `bet1_design.md`, `challenges_ranked.md` | `candidate/paused` | needs real labeled bitemporal HSI benchmark |

## Candidate Next Gates

These are not results. They are the next small tests that decide whether the
route deserves implementation effort.

| Gate | Question | Dataset/task | Minimum controls | Pass condition |
|---|---|---|---|---|
| Satellite latent subspace proof | does a tile/object/region represented as a subspace of patch-level satellite features add value? | clean satellite patch classification first, then change/anomaly triage | mean-pooled features, raw/foundation feature vector, linear probe, non-subspace distance | subspace geometry improves low-label accuracy, retrieval, or calibrated triage at equal feature source |
| Foundation-feature DS pressure | does DS/GDS add geometry beyond frozen DINO/Prithvi/SAM/RemoteCLIP feature differences? | one benchmark where frozen features can be extracted reliably | raw embedding L2/cosine, PCA/cross-reconstruction, shallow classifier | DS/GDS improves or explains errors beyond feature distance without huge compute cost |
| OSCD/Saab evidence confirmation | is the Successive Saab-DS gain reproducible and not just feature extraction? | OSCD official split plus one external multispectral test if available | Saab L2, Saab PCA, matched cross-reconstruction, IR-MAD/CVA | DS-specific score beats matched controls across held-out cities/seeds |

## Delete-Candidate Report Groups After Absorption

Do not delete yet. These are likely candidates after this ledger is checked:

| Group | Reason |
|---|---|
| old seminar drafts in `docs/pending_deletion_review/old_research_material/` | current framing now lives in active control docs |
| Claude AI synthesis docs under `docs/pending_deletion_review/old_research_material/claude_temporal/` | useful points are lane/method/ledger rows |
| overlapping old `docs/pending_deletion_review/old_notes/*.md` | being replaced by active docs |
| duplicated report narratives | summarized here; keep only reports with unique figures/details |
| `research-notes/` nested repo | old note structure already distilled repeatedly |
