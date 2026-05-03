# Project Reset Decision

Date: 2026-05-02  
Basis: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, `docs/NEXT_STEP_DECISION_MEMO.md`, repo files, and the user-run minimal reproduction result:

```text
dataset_len 125
x (26, 128, 128) y (1, 128, 128) valid (1, 128, 128)
channels {'n_raw': 26, 'n_priors': 0, 'total': 26}
logits (1, 1, 128, 128)
```

Latest Phase 1 artifact smoke result:

```text
Command exited 0.
[train] beirut: saved geodesic_dist in 2.57s
[val] abudhabi: saved geodesic_dist in 1.25s
[test] brasilia: saved geodesic_dist in 0.40s
```

Created artifacts:

- `phase1/outputs/_smoke_reentry_geodesic/run_metadata_geodesic.json`
- `phase1/outputs/_smoke_reentry_geodesic/oscd_change_maps/train/geodesic_dist/beirut_score.npy`
- `phase1/outputs/_smoke_reentry_geodesic/oscd_change_maps/train/geodesic_dist/beirut_mask.png`
- `phase1/outputs/_smoke_reentry_geodesic/oscd_change_maps/val/geodesic_dist/abudhabi_score.npy`
- `phase1/outputs/_smoke_reentry_geodesic/oscd_change_maps/val/geodesic_dist/abudhabi_mask.png`
- `phase1/outputs/_smoke_reentry_geodesic/oscd_change_maps/test/geodesic_dist/brasilia_score.npy`
- `phase1/outputs/_smoke_reentry_geodesic/oscd_change_maps/test/geodesic_dist/brasilia_mask.png`

Observed score arrays:

- `brasilia_score.npy`: `(433, 469)`, `float32`, min `0.10983837395906448`, max `1.0`, mean `0.2992086410522461`, finite `True`
- `beirut_score.npy`: `(1180, 1070)`, `float32`, min `0.13095642626285553`, max `1.0`, mean `0.5397144556045532`, finite `True`
- `abudhabi_score.npy`: `(799, 785)`, `float32`, min `0.15237773954868317`, max `1.0`, mean `0.697766900062561`, finite `True`

Latest controlled artifact audit result:

```text
Artifact audited:
phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv

Rows: 8
Columns: tag, mean_iou, mean_f1, mean_auroc, mean_pr_auc

E0_raw:
mean_iou    0.222987924850298
mean_f1     0.3428411527512548
mean_auroc  0.8687277832244259
mean_pr_auc 0.43149607130710976

E1_raw_ds:
mean_iou    0.27253517867726224
mean_f1     0.40085612219536626
mean_auroc  0.8743190336229665
mean_pr_auc 0.4358442122982414

E1_raw_ds minus E0_raw:
mean_iou    +0.04954725382696423
mean_f1     +0.05801496944411144
mean_auroc  +0.0055912503985405815
mean_pr_auc +0.0043481409911316216

Best rows in this artifact:
mean_iou    E1_raw_ds 0.27253517867726224
mean_f1     E1_raw_ds 0.40085612219536626
mean_auroc  E1_raw_ds 0.8743190336229665
mean_pr_auc E5_raw_celik 0.44757725362046774
```

Cross-check:

- `phase2/outputs/runs_gpu_150ep_20251215_233309/E0_raw_unet/eval/oscd_seg_eval_summary.csv` contains the same E0 test metrics as the aggregate artifact.
- `phase2/outputs/runs_gpu_150ep_20251215_233309/E1_raw_ds_unet/eval/oscd_seg_eval_summary.csv` contains the same E1 test metrics as the aggregate artifact.
- `E0_raw_unet/run_metadata.json` reports config `phase2/configs/oscd_seg_baseline.yaml`, seed `1234`, device `cuda`, torch `2.9.1+cu126`, git hash `735deac0caa1b9fe381c2ea98e09b93761534415`.
- `E1_raw_ds_unet/run_metadata.json` reports config `phase2/configs/oscd_seg_E1_raw_ds.yaml`, seed `1234`, device `cuda`, torch `2.9.1+cu126`, git hash `735deac0caa1b9fe381c2ea98e09b93761534415`.

## 1. What We Currently Trust

[trusted] The Phase 2 raw OSCD data/model path is alive at minimum-contract level: OSCD val data loads, raw pre/post Sentinel-2 stacks produce 26 channels, the binary mask shape is coherent, and a U-Net forward pass returns `(1, 1, 128, 128)` logits.

[trusted] The Phase 1 OSCD geodesic-prior artifact path is alive at smoke-test level: the CLI ran with exit code 0, read OSCD, and wrote finite `geodesic_dist` score maps and mask PNGs for one train, one val, and one test city.

[trusted] The existing Phase 2 150-epoch summary artifact internally supports the specific claim that `E1_raw_ds` beats `E0_raw` on test mean IoU and mean F1 in that saved artifact. The aggregate CSV and individual eval summary CSVs agree.

[trusted] The implemented project is currently best described as Sentinel-2 OSCD binary change segmentation with optional unsupervised prior channels, not end-to-end disaster damage segmentation. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 1", "Phase A - 7", and "Bottom Line".

[trusted] Phase 1 and Phase 2 code entry points exist for prior generation and OSCD segmentation. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 2"; repo files `phase1/eval/run_oscd_eval.py`, `phase1/eval/run_oscd_geodesic_priors.py`, `phase2/train/train_oscd_seg.py`, `phase2/eval/evaluate_oscd_seg.py`.

[trusted] The safest current thesis lane is change segmentation with interpretable unsupervised multispectral priors. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C" and "Bottom Line"; `docs/NEXT_STEP_DECISION_MEMO.md`, sections 1 and 3.

## 2. What We Do Not Trust

[unverified] We do not yet trust any reported numerical result, including Phase 1 AUROC values or Phase 2 IoU/F1/AUROC/PR-AUC tables. The audit did not reproduce them. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C".

[trusted] Narrow update: we now trust that one existing Phase 2 artifact internally contains and consistently repeats the E1-vs-E0 result values listed above.

[unverified] We still do not trust those values as freshly reproduced experiment results, because no checkpoint evaluation or training rerun was performed in this artifact audit.

[unverified] We do not yet trust that saved checkpoints and old output directories are reproducible or compatible with the current code/environment.

[unverified] We do not trust damage-segmentation claims beyond "future/warm extension." The integrated xBD/xBD-S12 damage pipeline is missing. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C", damage row.

[unverified] We do not trust old chat narratives, old audits, or thesis prose as current truth unless converted into testable claims and checked against code/results.

## 3. What Failed Or Looks Risky

[trusted] The minimal Phase 2 forward-pass smoke did not fail.

[trusted] Phase 1 geodesic-prior generation did not fail in the one-city-per-split smoke test.

[risk] This does not verify full Phase 1 evaluation, DS projection, PCA-diff, pixel/CVA, Celik, IR-MAD, threshold calibration, metrics, or runtime claims.

[risk] The run emitted `NotGeoreferencedWarning` warnings from rasterio. This did not stop score-map generation, but it means geospatial transform metadata should not be assumed from this smoke test.

[risk] Phase 1 runtime comparisons are unsafe because timing is not isolated per method. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 6" and "Phase C"; `phase1/eval/run_oscd_eval.py`.

[risk] CVA and pixel-difference appear effectively duplicated as L2 magnitude scores; treating them as independent baselines would overstate the method set. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 4" and "Phase C".

[risk] Priors-only claims are risky: no clear priors-only config was observed, and `PriorsFusionUNet` is fragile for zero-channel branches. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C".

[risk] CUDA assumptions are risky. The previous rerun log reported CPU-only PyTorch. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C"; `PIPELINE_RERUN_LOG.txt`.

## 4. Current Best Research Question

[trusted] Current safest research question:

> Do unsupervised multispectral Sentinel-2 change priors, especially DS and related classical change scores, improve supervised OSCD binary change segmentation over raw pre/post Sentinel-2 alone, and which priors help or hurt?

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 7" and "Bottom Line".

## 5. Smallest Next Serious Experiment

[trusted] The Phase 1 OSCD geodesic-prior artifact smoke has now been run successfully. It was the smallest serious experiment because the Phase 2 raw forward path had already passed, while Phase 1 prior artifact generation was still unverified after reset.

Command:

```powershell
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m phase1.eval.run_oscd_geodesic_priors `
  --config phase1/configs/oscd_geodesic_priors.yaml `
  --oscd_root data/OSCD `
  --output_dir phase1/outputs/_smoke_reentry_geodesic `
  --max_cities 1
```

Expected output files:

- `phase1/outputs/_smoke_reentry_geodesic/run_metadata_geodesic.json`
- `phase1/outputs/_smoke_reentry_geodesic/oscd_change_maps/<split>/geodesic_dist/<city>_score.npy`
- Any quick-look mask/image files emitted by the script.

Observed success:

[trusted] Success means Phase 1 CLI, OSCD loading, geodesic-prior computation, and artifact writing are alive for a small controlled run.

Remaining failure boundary:

[risk] This result does not prove full experiment reproducibility, reported metrics, checkpoint compatibility, CUDA support, or damage mapping.

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase D - Step 1".

## 6. Thesis Claims Allowed Today

[trusted] "The current implementation contains a two-phase Sentinel-2 change-segmentation pipeline: Phase 1 prior generation and Phase 2 OSCD binary segmentation."

[trusted] "A minimal Phase 2 raw OSCD forward pass succeeds on local data."

[trusted] "The current code supports 13-band pre/post Sentinel-2 input as 26 raw channels for binary OSCD segmentation."

[trusted] "The project is not yet an end-to-end disaster damage segmentation system."

[trusted] "xBD/xBD-S12 damage mapping is a future or warm extension, not a verified result."

[unverified] "Previous result tables exist as claims, but they are not yet reproduced in this reset."

## 7. Thesis Claims Forbidden Until Verified

[paused] Do not claim DS improves OSCD segmentation until old checkpoints/results are audited or a fresh controlled evaluation is run.

[trusted] Narrow exception: it is now allowed to say, "In the saved 150-epoch artifact `oscd_priors_ablation_summary_extended.csv`, raw+DS has higher test mean IoU and F1 than raw-only."

[paused] Do not claim any specific IoU, F1, AUROC, or PR-AUC value as current truth.

[paused] Do not phrase the artifact-audited numbers as reproduced results yet. Say "saved artifact reports" or "artifact audit found", not "we reproduced".

[paused] Do not claim damage segmentation, xBD, xBD-S12, ordinal damage classification, or disaster-damage evaluation has been implemented end-to-end.

[paused] Do not claim runtime superiority or efficiency comparisons from Phase 1 outputs.

[paused] Do not claim CVA is an independent baseline from pixel difference without implementation correction or explicit caveat.

[paused] Do not claim priors-only experiments exist unless an actual config and successful run are found or created later.

[paused] Do not claim external benchmark competitiveness or state of the art.

## 8. What Should Be Paused Or Ignored For Now

[paused] Long training, sweeps, cleanup scripts, CUDA reruns, and xBD/xBD-S12 work should remain paused. Source: `docs/NEXT_STEP_DECISION_MEMO.md`, section 9.

[paused] Broad historical directions should not steer the next work item: SSC-heavy pipelines, UAV/edge deployment, MCDA, IoT, DMaaS, graph decision layers, and operational dashboards. Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase B", C31-C34 and "Phase C".

[paused] Paper reading should be limited until the local artifact chain is verified. Reading more literature cannot prove the current repo is alive.

## 9. Old Chat Distillation

[paused] Do not do broad old-chat distillation before the Phase 1 artifact smoke.

[next-action] After the Phase 1 artifact smoke, old-chat distillation can be useful only as claims extraction, not story preservation. Extract exactly:

- concrete experiment claims: command, config, dataset path, output path, seed, device, date, metric table
- implementation decisions: what was changed, why, and which file/config it affected
- known bugs or caveats: failed runs, incompatible checkpoints, broken paths, environment notes
- thesis-scope decisions: active, warm, paused, forbidden
- result provenance: which artifact supports which number

[paused] Do not extract motivational narrative, broad thesis ambition, literature summaries, or confident conclusions unless they become testable claims.

Source: `docs/NEXT_STEP_DECISION_MEMO.md`, section 10; `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Evidence Policy", "Phase C", and "Bottom Line".

## 10. What Codex Should Focus On Next

[trusted] Codex has interpreted the Phase 1 OSCD geodesic-prior artifact smoke and recorded it as alive at smoke-test level.

[trusted] Codex has also performed one controlled, non-destructive artifact audit of the saved Phase 2 150-epoch summary. The audit verified internal artifact consistency for the reported E1 raw+DS versus E0 raw-only result.

[next-action] Codex should next help choose whether to do a fresh checkpoint evaluation of E0/E1 on CPU, or to pause and distill old chats into a claims/provenance table now that the minimal live checks and one artifact audit have passed.
