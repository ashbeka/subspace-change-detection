# Project Re-entry Synthesis

## 1. Current best understanding of the project

[repo-evidence] This project is currently about Sentinel-2 binary change segmentation, not completed disaster damage segmentation.

In practice, it has two working parts:

- Phase 1 generates unsupervised change maps on OSCD using Difference Subspace, PCA-diff, pixel/CVA-style baselines, Celik, IR-MAD, and geodesic side methods.
- Phase 2 trains supervised OSCD segmentation models using raw pre/post Sentinel-2 bands, optionally with Phase 1 change maps as extra input channels.

[recommendation] The safest current identity is: interpretable subspace-derived change priors for supervised Sentinel-2 change segmentation. Damage mapping and xBD/xBD-S12 should remain future work until a real damage pipeline exists.

## 2. What the clean repo audit says

- [repo-evidence] Repo structure:
  - `phase1/`: DS/classical change detection, OSCD/MultiSenGE loaders, baselines, priors, evaluation, visualization.
  - `phase2/`: OSCD segmentation datasets, models, training, evaluation, visualization, configs, sweep scripts.
  - `data/`: local ignored OSCD, MultiSenGE, and xBD assets.
  - `phase1/outputs/` and `phase2/outputs/`: local ignored artifacts, useful but not automatically trusted.
- [repo-evidence] Implemented methods:
  - DS projection and cross-residual, residual and eigen DS variants, sliding-window DS.
  - Pixel diff, CVA, PCA-diff, Celik local PCA+k-means, IR-MAD.
  - Grassmann/geodesic helpers, geodesic priors, second-order DS utilities.
  - U-Net, ResNet-U-Net, PriorsFusionUNet, Siamese U-Net.
- [repo-evidence] Runnable scripts:
  - Phase 1: `phase1.eval.run_oscd_eval`, `phase1.eval.run_oscd_geodesic_priors`, `phase1.eval.run_multisenge_viz`, `phase1.eval.run_multisenge_temporal_geodesic`.
  - Phase 2: `phase2.train.train_oscd_seg`, `phase2.eval.evaluate_oscd_seg`, `phase2.eval.compare_priors_effect`, visualization scripts.
- [repo-evidence] Local setup:
  - `.venv` works.
  - Python 3.10.11.
  - CUDA-enabled `torch 2.9.1+cu126` is available.
  - Read-only smoke checks loaded raw OSCD patches and raw+DS+PCA prior patches.
- [repo-evidence] Expected datasets:
  - OSCD at `data/OSCD`.
  - MultiSenGE at `data/MultiSenGE`.
  - xBD archives at `data/xbd`, but no `train.csv`, `val.csv`, or `test.csv` for the generic adapter.
- [risk] Current risks:
  - Old outputs may be stale or mixed.
  - Old performance claims need an artifact audit or fresh rerun.
  - Damage-segmentation claims are unsafe.
  - Phase 2 does not currently switch to `DamageDatasetAdapter`.
  - Thresholding protocol is inconsistent between Phase 1 and Phase 2.
- [recommendation] Smallest next experiment from the clean audit:
  - Fresh Phase 1 fast priors.
  - One-epoch Phase 2 raw baseline and raw+DS run.
  - Evaluate both and confirm the pipeline is alive.

## 3. What the old chats add

- [chat-context] Useful reasoning:
  - The strongest research angle is not "DS is the best unsupervised detector"; it is "a standalone detector is not necessarily the best downstream segmentation prior."
  - Old evidence suggested `pca_diff` had the best Phase 1 AUROC, while `ds_projection` was more useful as a Phase 2 prior in one 150-epoch run.
  - "Raw" means pre/post Sentinel-2 bands stacked as 26 CNN input channels.
  - Priors are continuous score maps used as extra input channels, not labels and not Otsu masks.
  - Current DS fits PCA bases separately to pre and post valid pixel spectra, with matrices shaped roughly `(13, N_valid_pixels)`.
  - Current configured DS rank was remembered as `rank_r = 6`.
  - The main first-order DS score is projection energy, approximately `||D^T delta_i||^2`.
- [chat-context] Useful old bugs/fixes to remember:
  - Phase 2 evaluation was previously criticized for random val/test transforms and patch-overlap evaluation; old sessions say fixes moved evaluation toward deterministic stitched full-tile metrics. This still needs current-code verification before relying on old results.
  - PowerShell seed parsing caused sweep failures in an older script state.
  - Earlier training used CPU-only PyTorch despite a GPU being present; the current clean audit shows CUDA is now available.
  - MultiSenGE filesystem scale caused problems; manifest-based sampling was added to avoid scanning huge directories repeatedly.
- [chat-context] Useful commands:
  - Phase 1 fast priors:
    `python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_priors_fast.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_saved_priors_fast --save_change_maps`
  - Phase 2 train:
    `python -m phase2.train.train_oscd_seg --config <config> --oscd_root data/OSCD --phase1_change_maps_root <change_maps> --output_dir <run> --device cuda`
  - Phase 2 eval:
    `python -m phase2.eval.evaluate_oscd_seg --config <config> --oscd_root data/OSCD --phase1_change_maps_root <change_maps> --checkpoint <run>/best.ckpt --output_dir <run>/eval --device cuda`
- [chat-context] Useful research framing:
  - Keep OSCD as the defensible thesis core.
  - Keep xBD-S12 as a warm extension, not a current result.
  - Keep MultiSenGE only if it serves temporal/geodesic/second-order DS reasoning, not as a vague side demo.
  - Treat SCD, SSC, UAV, MCDA, DMaaS, and land-use ideas as cold archive unless explicitly reopened.
- [chat-context] Previous confusion worth preventing:
  - DS is not reconstructing a 13-band image in the current main pipeline.
  - Spatial layout is not used when fitting the global DS basis; it is restored only when scores are mapped back to image coordinates.
  - Phase 2 raw U-Net does not train on Otsu-thresholded Phase 1 masks.
  - OSCD binary change segmentation is not the same thing as disaster damage assessment.

## 4. What old chat content should NOT be trusted

- [risk] Do not trust old metrics as current facts without verifying the exact CSVs, configs, seeds, checkpoint paths, git hash, and threshold protocol.
- [risk] Do not trust "DS improves segmentation" as a final result. It is a plausible old single-seed claim, not a reproduced conclusion from this re-entry.
- [risk] Do not trust damage mapping, xBD, or xBD-S12 as implemented. The current repo does not have an integrated damage trainer/evaluator.
- [risk] Do not trust claims that geodesic priors, second-order DS, KPCA, KCCA, S3CCA, or TRCCA should become the thesis core. They are ideas or side branches until matched experiments prove usefulness.
- [risk] Do not trust old CPU/GPU status. The old sessions diagnosed CPU-only Torch; the clean audit found CUDA-enabled Torch now.
- [risk] Do not trust old web context packs, copied online research summaries, or PDF-scraped snippets as source-of-truth.
- [unverified] Old claims about fixed evaluation methodology still need a current-code audit if they become thesis-critical.
- [unverified] Old Phase 1 runtime numbers may be misleading because timing may have been assigned per method from combined tile processing.
- [unverified] Old cleanup actions may have removed or mixed outputs; output folders are evidence, not proof.

## 5. Reconciled project direction

- [recommendation] Focus now:
  - Re-establish the OSCD Phase 1 to Phase 2 pipeline with fresh, small, reproducible runs.
  - Frame the thesis around interpretable unsupervised priors for supervised Sentinel-2 change segmentation.
  - Verify whether DS projection, PCA-diff, or other priors help under a clean matched protocol.
- [recommendation] Pause:
  - xBD/xBD-S12 damage segmentation.
  - Full multi-seed 150-epoch sweeps.
  - Geodesic/second-order DS as headline methods.
  - README/thesis performance claims based on old outputs.
- [recommendation] Ignore for now:
  - SSC/UAV/MCDA/DMaaS/land-use scope.
  - SCD as a near-term scope jump.
  - KPCA/KCCA unless a small sampled prototype is explicitly planned later.
- [recommendation] Verify next:
  - Fresh priors can be generated.
  - Phase 2 can train/evaluate on CUDA from fresh outputs.
  - Raw vs raw+DS pipeline comparison produces expected files.
  - Result summaries are generated from the exact commands you intend to cite.

## 6. Current research question

[recommendation] Best current research question:

Can unsupervised Sentinel-2 change maps from Difference Subspace and related classical methods serve as useful, interpretable input priors for supervised OSCD binary change segmentation compared with raw pre/post Sentinel-2 bands alone?

The current project should answer this on OSCD first. Disaster damage mapping should be described only as a future transfer direction.

## 7. Smallest next concrete experiment

### Experiment: Fresh OSCD E0 vs E1 one-epoch pipeline smoke

- [recommendation] Goal:
  - Confirm that the current repo can regenerate Phase 1 fast priors and use them in Phase 2 for one raw-only model and one raw+DS model.
- [recommendation] Why it matters:
  - It converts the restart from document audit to executable evidence.
  - It checks the core thesis path without spending days on training.
  - It avoids trusting stale output folders.
- [repo-evidence] Expected inputs:
  - `data/OSCD`
  - `phase1/configs/oscd_priors_fast.yaml`
  - `phase2/configs/oscd_seg_baseline.yaml`
  - `phase2/configs/oscd_seg_E1_raw_ds.yaml`
  - CUDA-enabled `.venv`
- [recommendation] Commands, PowerShell from repo root:

```powershell
.\.venv\Scripts\Activate.ps1
$py=".\.venv\Scripts\python.exe"
$ts=Get-Date -Format "yyyyMMdd_HHmmss"
$oscd="data/OSCD"
$p1="phase1/outputs/reentry_fast_priors_$ts"
$maps=Join-Path $p1 "oscd_change_maps"
$out="phase2/outputs/reentry_smoke_$ts"
& $py -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_priors_fast.yaml --oscd_root $oscd --output_dir $p1 --save_change_maps
& $py -m phase2.train.train_oscd_seg --device cuda --max_epochs 1 --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $maps --output_dir (Join-Path $out "E0_raw") --overwrite_output_dir
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $maps --checkpoint (Join-Path $out "E0_raw/best.ckpt") --output_dir (Join-Path $out "E0_raw/eval")
& $py -m phase2.train.train_oscd_seg --device cuda --max_epochs 1 --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $maps --output_dir (Join-Path $out "E1_raw_ds") --overwrite_output_dir
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $maps --checkpoint (Join-Path $out "E1_raw_ds/best.ckpt") --output_dir (Join-Path $out "E1_raw_ds/eval")
```

- [repo-evidence] Expected outputs:
  - `$p1/oscd_eval_results.json`
  - `$p1/oscd_eval_summary.csv`
  - `$p1/oscd_change_maps/{train,val,test}/{ds_projection,ds_cross_residual,pixel_diff,pca_diff}/...`
  - `$out/E0_raw/best.ckpt`
  - `$out/E0_raw/eval/oscd_seg_eval_summary.csv`
  - `$out/E1_raw_ds/best.ckpt`
  - `$out/E1_raw_ds/eval/oscd_seg_eval_summary.csv`
- [recommendation] Success criteria:
  - All commands exit successfully.
  - CUDA is used.
  - Both evaluation summary CSVs exist.
  - Metrics are finite where expected.
  - No missing-prior errors occur.
- [risk] Failure criteria:
  - Priors are missing or misnamed.
  - CUDA fails.
  - Checkpoint is not written.
  - Evaluation crashes or produces empty summaries.
  - Input channel mismatch occurs.

### Fresh run result: 20260502_020139

- [repo-evidence] Status:
  - All five commands completed successfully.
  - CUDA was used: `NVIDIA GeForce RTX 4070 SUPER`.
  - Phase 1 fresh fast-prior generation completed.
  - E0 raw one-epoch train/eval completed.
  - E1 raw+DS one-epoch train/eval completed.
- [repo-evidence] Output roots:
  - Phase 1: `phase1/outputs/reentry_fast_priors_20260502_020139`
  - Phase 2: `phase2/outputs/reentry_smoke_20260502_020139`
- [repo-evidence] Key generated artifacts:
  - `phase1/outputs/reentry_fast_priors_20260502_020139/oscd_eval_results.json`
  - `phase1/outputs/reentry_fast_priors_20260502_020139/oscd_eval_summary.csv`
  - `phase1/outputs/reentry_fast_priors_20260502_020139/oscd_change_maps/`
  - `phase2/outputs/reentry_smoke_20260502_020139/E0_raw/best.ckpt`
  - `phase2/outputs/reentry_smoke_20260502_020139/E0_raw/eval/oscd_seg_eval_summary.csv`
  - `phase2/outputs/reentry_smoke_20260502_020139/E1_raw_ds/best.ckpt`
  - `phase2/outputs/reentry_smoke_20260502_020139/E1_raw_ds/eval/oscd_seg_eval_summary.csv`
- [repo-evidence] Artifact counts:
  - Phase 1 output files: `195`
  - Phase 2 output files: `14`
- [repo-evidence] Fresh Phase 1 fast-prior test AUROC means:
  - `pca_diff`: `0.8134309456698187`
  - `pixel_diff`: `0.7558751604773204`
  - `ds_projection`: `0.7550638137273665`
  - `ds_cross_residual`: `0.5562552202069637`
- [repo-evidence] One-epoch Phase 2 E0 raw metrics:
  - val IoU/F1/AUROC/PR-AUC: `0.028380117802617132`, `0.05496211386600161`, `0.5778291054144029`, `0.09009701863729708`
  - test IoU/F1/AUROC/PR-AUC: `0.07400114939210967`, `0.1345568013784828`, `0.6680342904561287`, `0.1195362708356104`
- [repo-evidence] One-epoch Phase 2 E1 raw+DS metrics:
  - val IoU/F1/AUROC/PR-AUC: `0.04706912268098712`, `0.08813548289770812`, `0.6065145330797433`, `0.08031480275334074`
  - test IoU/F1/AUROC/PR-AUC: `0.0712090027264045`, `0.12771985916030953`, `0.707039968967581`, `0.11832519139847245`
- [repo-evidence] E1 minus E0 deltas:
  - val IoU: `+0.018689004878369987`
  - val F1: `+0.03317336903170651`
  - val AUROC: `+0.028685427665340435`
  - val PR-AUC: `-0.009782215883956336`
  - test IoU: `-0.002792146665705167`
  - test F1: `-0.00683694221817327`
  - test AUROC: `+0.03900567851145231`
  - test PR-AUC: `-0.0012110794371379546`
- [repo-evidence] Run metadata:
  - E0 config: `phase2/configs/oscd_seg_baseline.yaml`
  - E1 config: `phase2/configs/oscd_seg_E1_raw_ds.yaml`
  - seed: `1234`
  - device: `cuda`
  - torch: `2.9.1+cu126`
  - git hash: `9c43e8bdb8062d32daf60fabcc7904973fb167d0`
- [trusted] Interpretation:
  - The fresh OSCD Phase 1 -> Phase 2 pipeline is alive end to end.
  - Fresh Phase 1 fast priors reproduce the broad old pattern for fast methods: PCA-diff has the highest test AUROC, DS projection and pixel diff are close, and DS cross-residual is much weaker.
  - The one-epoch E0/E1 run is a pipeline smoke test, not a performance experiment.
- [risk] Do not use the one-epoch E0/E1 metrics as thesis performance evidence.
- [risk] This run does not prove that DS improves final segmentation. It only proves the raw+DS path trains and evaluates from fresh priors.

## 8. What I should read/understand next

- [recommendation] Concepts first:
  - How DS constructs a difference subspace from two PCA bases.
  - What the DS projection score means per pixel.
  - Why Phase 1 priors are continuous input features, not labels.
  - How patch predictions are stitched into full-tile OSCD metrics.
- [recommendation] Repo files/docs first:
  - `docs/IMPLEMENTATION_STATUS.md`
  - `phase1/ds/pca_utils.py`
  - `phase1/ds/ds_scores.py`
  - `phase1/eval/run_oscd_eval.py`
  - `phase2/data/oscd_seg_dataset.py`
  - `phase2/train/train_oscd_seg.py`
  - `phase2/eval/evaluate_oscd_seg.py`
  - `phase2/configs/oscd_seg_baseline.yaml`
  - `phase2/configs/oscd_seg_E1_raw_ds.yaml`
- [recommendation] Postpone:
  - xBD/xBD-S12 implementation.
  - SCD benchmarks.
  - KPCA/KCCA/S3CCA/TRCCA.
  - Full geodesic/second-order DS thesis framing.
  - SOTA comparisons.

## 9. Recommended AGENTS.md content later

Do not create `AGENTS.md` yet. When it is created, it should include:

- [recommendation] Treat repo/code and `docs/IMPLEMENTATION_STATUS.md` as more reliable than old chats.
- [recommendation] Do not overclaim damage segmentation; current core is OSCD binary change segmentation.
- [recommendation] Do not trust ignored output folders without artifact verification.
- [recommendation] Avoid code changes unless explicitly requested.
- [recommendation] Keep all run commands from repo root.
- [recommendation] Use timestamped output directories for experiments.
- [recommendation] Never run `clean_house.ps1 -Aggressive` without explicit approval and a list of what will be kept.
- [recommendation] Before citing metrics, record config, command, seed, epoch count, checkpoint, git hash, and output path.
- [recommendation] Prefer small smoke runs before long sweeps.
- [recommendation] Keep xBD/xBD-S12, SCD, KPCA/KCCA, and broad disaster-planning ideas out of active scope unless explicitly reopened.

## Immediate next step

The Section 7 fresh OSCD E0 vs E1 one-epoch pipeline smoke experiment has been run successfully. The next decision should be whether to run a controlled longer E0/E1 comparison, audit old checkpoints further, or pause for claims/provenance distillation.
