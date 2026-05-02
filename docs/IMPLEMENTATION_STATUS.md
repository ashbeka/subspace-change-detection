# Implementation Status

Audit date: 2026-05-02. Scope: repository files and local filesystem state only. No old chat logs or external context were used.

## 1. Project Intent

- [observed] The project is named `DS_damage_segmentation` and the main docs frame it as Difference-Subspace priors for change and eventual damage segmentation (`README.md`, `ds_damage_segmentation.tex`).
- [observed] The implemented core is currently Sentinel-2 binary change work on OSCD: Phase 1 generates unsupervised change maps, and Phase 2 trains/evaluates OSCD segmentation models (`phase1/eval/run_oscd_eval.py`, `phase2/train/train_oscd_seg.py`, `phase2/eval/evaluate_oscd_seg.py`).
- [inferred] The damage-segmentation aim is future-facing. xBD archives exist locally, and a generic adapter exists, but the trainer/evaluator are wired to OSCD, not to xBD damage labels (`data/xbd`, `phase2/data/damage_dataset_adapter.py`, `phase2/configs/damage_dataset_template.yaml`).

## 2. Repo Structure

- [observed] Root docs and run guides: `README.md`, `RUN_PIPELINE.md`, `CODEBASE_AUDIT.md`, `PIPELINE_RERUN_LOG.txt`, `ds_damage_segmentation.tex`.
- [observed] `phase1/`: DS/classical change detection. Important subdirs are `phase1/data`, `phase1/ds`, `phase1/subspace`, `phase1/baselines`, `phase1/priors`, `phase1/eval`, `phase1/configs`, `phase1/scripts`, and `phase1/docs`.
- [observed] `phase2/`: supervised OSCD segmentation with optional prior channels. Important subdirs are `phase2/data`, `phase2/models`, `phase2/train`, `phase2/eval`, `phase2/viz`, `phase2/configs`, `phase2/scripts`, and `phase2/docs`.
- [observed] `data/`: local, git-ignored data. It contains unpacked OSCD, unpacked MultiSenGE, and xBD archives (`.gitignore`, `data/OSCD`, `data/MultiSenGE`, `data/xbd`).
- [observed] `phase1/outputs/` and `phase2/outputs/`: local, git-ignored experiment outputs/checkpoints/metrics. Treat as evidence of past runs, not trusted results (`.gitignore`).
- [observed] `references/reference_papers/`: PDFs related to DS, subspace methods, U-Net, change detection, xBD, and remote-sensing damage. `references/reference_code/` is git-ignored local reference code.
- [observed] `docs/` exists but is currently untracked by Git and contains reset/re-entry notes (`git status --short`, `docs/ADVERSARIAL_REENTRY_AUDIT.md`, `docs/NEXT_STEP_DECISION_MEMO.md`, `docs/PROJECT_RESET_DECISION.md`).
- [observed] `prompt_stuff/` and `research-notes/` exist locally but were not used as audit sources because they look like generated/parallel notes rather than current implementation source.

## 3. Runnable Code Right Now

- [observed] CLI help imports successfully for `phase1.eval.run_oscd_eval`, `phase1.eval.run_oscd_geodesic_priors`, `phase1.eval.run_multisenge_viz`, `phase1.eval.run_multisenge_temporal_geodesic`, `phase2.train.train_oscd_seg`, `phase2.eval.evaluate_oscd_seg`, and `phase2.eval.compare_priors_effect`.
- [observed] Local `.venv` is usable: Python 3.10.11, `torch 2.9.1+cu126`, CUDA available, `rasterio 1.4.3`, `numpy 2.2.6`, `sklearn 1.7.2`.
- [observed] A read-only Phase 2 smoke check loaded `phase2/configs/oscd_seg_baseline.yaml`, built `OSCDSegmentationDataset`, produced a sample shaped `(26, 256, 256)`, and a U-Net forward pass returned `(1, 1, 256, 256)`.
- [observed] A read-only priors smoke check loaded `phase2/configs/oscd_seg_priors.yaml`; the first patch had `(28, 256, 256)`, consistent with 26 raw channels plus DS and PCA priors.
- [risk] Smoke checks do not prove full training reproducibility, checkpoint compatibility, metric correctness, or that old result claims are still valid.

## 4. Setup And Dependencies

- [observed] Setup expects a root virtualenv `.venv` and installs both `phase1/requirements.txt` and `phase2/requirements.txt` (`README.md`, `RUN_PIPELINE.md`, `REMEMBER_TO_ACTIVATE_ENV.txt`).
- [observed] Phase 1 requirements: `numpy`, `scipy`, `scikit-learn`, `rasterio`, `matplotlib`, `pyyaml`.
- [observed] Phase 2 adds `torch` and `torchvision` and defaults training/evaluation to CUDA (`phase2/requirements.txt`, `phase2/train/train_oscd_seg.py`).
- [observed] `RUN_PIPELINE.md` includes a CUDA PyTorch reinstall command and a GPU sanity check.
- [risk] Dependency versions are not pinned. Reproducing old results may depend on the current local `.venv`, not only the requirements files.

## 5. Implemented / Partial / Planned Methods

- [observed] Implemented Phase 1 methods: global and sliding-window DS projection/cross-residual, residual-stacked DS, eigen DS variant, PCA helpers, pixel L2 difference, CVA, PCA-diff, Celik local PCA+k-means, and a lightweight IR-MAD (`phase1/ds/ds_scores.py`, `phase1/ds/pca_utils.py`, `phase1/baselines/*.py`).
- [observed] Implemented geometric/temporal methods: Grassmann distance, subspace magnitude, geodesic projection, local geodesic priors, and second-order DS utilities (`phase1/subspace/geodesic.py`, `phase1/subspace/second_order_ds.py`, `phase1/priors/geodesic_priors.py`).
- [observed] Implemented Phase 2 models: plain U-Net, ResNet-34 U-Net wrapper, PriorsFusionUNet, and Siamese U-Net (`phase2/models/*.py`).
- [observed] Implemented training/eval pieces: BCE+Dice loss, Adam/AdamW schedulers, checkpointing, patch stitching, tile-level validation/evaluation, AUROC and PR-AUC (`phase2/train/*.py`, `phase2/eval/*.py`).
- [partial] `DamageDatasetAdapter` is a generic CSV-based dataset adapter, not an xBD/xBD-S12 implementation. It expects `train.csv`, `val.csv`, and `test.csv` or configured CSVs with pre/post/mask columns (`phase2/data/damage_dataset_adapter.py`, `phase2/configs/damage_dataset_template.yaml`).
- [planned] True multi-class disaster damage segmentation, xBD/xBD-S12 extraction/indexing, damage-specific training scripts, damage metrics, and Phase 3 are only planned or templated (`README.md`, `phase2/docs/spec_phase2_oscd_seg.md`, `ds_damage_segmentation.tex`).

## 6. Referenced / Expected Datasets

- [observed] OSCD is expected at `data/OSCD` with `onera_satellite_change_detection dataset__images`, train labels, and test labels. Local OSCD has 24 city folders, 1,272 TIFFs, 72 PNGs, 24 GeoJSONs, and split text files.
- [observed] Phase 1 official train/test city lists are hard-coded in `phase1/data/oscd_dataset.py`; Phase 2 configs use 11 train, 3 val, and 10 test cities (`phase2/configs/oscd_seg_baseline.yaml`).
- [observed] MultiSenGE is expected under `data/MultiSenGE` with `s2/`, `s1/`, `labels/`, and `ground_reference/`. Local MultiSenGE has many S2 TIFFs and 8,157 JSON labels.
- [observed] xBD local data is mostly archives/parts and PDFs under `data/xbd`; `data/xbd/train.csv`, `val.csv`, and `test.csv` are absent.
- [risk] `data/`, `phase1/outputs/`, and `phase2/outputs/` are ignored by Git, so another machine may not have the data or generated priors.

## 7. Important Scripts / Configs / Notebooks

- [observed] Main Phase 1 prior generation: `python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_priors_fast.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_saved_priors_fast --save_change_maps`.
- [observed] Full/slower Phase 1 baselines: `phase1/configs/oscd_default.yaml`; geodesic priors: `phase1/eval/run_oscd_geodesic_priors.py` with `phase1/configs/oscd_geodesic_priors.yaml`.
- [observed] MultiSenGE scripts: `phase1/scripts/build_multisenge_manifest.py`, `phase1/eval/run_multisenge_viz.py`, and `phase1/eval/run_multisenge_temporal_geodesic.py`.
- [observed] Core Phase 2 configs: `oscd_seg_baseline.yaml`, `oscd_seg_E1_raw_ds.yaml`, `oscd_seg_E2_raw_pca.yaml`, `oscd_seg_priors.yaml`, `oscd_seg_siamese.yaml`, `oscd_seg_baseline_resnet.yaml`, `oscd_seg_priors_resnet.yaml`, and `oscd_seg_priors_fusion.yaml`.
- [observed] Main Phase 2 scripts: `phase2/train/train_oscd_seg.py`, `phase2/eval/evaluate_oscd_seg.py`, `phase2/eval/compare_priors_effect.py`, `phase2/viz/viz_seg_predictions.py`, `phase2/viz/viz_oscd_combined.py`.
- [observed] Automation scripts: `RUN_PIPELINE.md`, `phase2/scripts/run_phase2_sweep.ps1`, `phase2/scripts/run_phase2_core_experiments.ps1`.
- [observed] No notebooks were found in the repository file listing.

## 8. Incomplete / Broken / Unclear / Risky Areas

- [risk] The repo title and thesis draft say damage mapping, but the runnable implementation is OSCD binary change segmentation. Damage claims should be treated as future work until an xBD/xBD-S12 pipeline exists.
- [risk] Local generated outputs may be stale or mixed. Example: `phase1/outputs/oscd_saved_priors_fast/oscd_eval_summary.csv` reports only DS/pixel/PCA methods, while the same output folder also contains Celik and IR-MAD change-map directories.
- [risk] Old metrics and checkpoints in `phase1/outputs/` and `phase2/outputs/` are not ground truth. They need fresh reruns or a strict artifact audit before being cited.
- [risk] `phase2/train/train_oscd_seg.py` and `phase2/eval/evaluate_oscd_seg.py` always construct `OSCDSegmentationDataset`; they do not switch on `dataset.name` and do not use `DamageDatasetAdapter`.
- [unclear] Thresholding choices are inconsistent across stages: Phase 1 calibrates global thresholds on train, while Phase 2 defaults segmentation evaluation to threshold `0.5` unless configs override it.
- [unclear] CVA is implemented as the same L2 magnitude as `pixel_diff` with a separate name (`phase1/baselines/cva.py`, `phase1/baselines/pixel_diff.py`).
- [risk] `clean_house.ps1` can delete output directories, especially with `-Aggressive`. It uses `SupportsShouldProcess`, but it still needs care because outputs are the only local record of some old runs.
- [risk] Many docs/results discuss 150-epoch outcomes, but only a read-only smoke check was performed in this audit. Do not claim performance improvements without rerunning or verifying exact artifacts.

## 9. Assumptions Present But Unverified

- [unclear] OSCD split choices are treated as official or accepted: Phase 1 hard-codes 14 train and 10 test cities, while Phase 2 pulls three train cities into validation.
- [unclear] OSCD band statistics in `phase1/data/oscd_band_stats.json` are assumed correct and reusable for MultiSenGE subsets.
- [unclear] The current normalization and valid-mask strategy assumes zero is NODATA and uses `min_valid_bands`; cloud/SCL handling is not apparent.
- [unclear] Saved Phase 1 priors are assumed aligned with OSCD tiles and Phase 2 patches; the smoke check only verified shape/channel loading for one patch.
- [unclear] The DS formulation and eigen variant are implemented, but their mathematical equivalence/appropriateness for Sentinel-2 change segmentation is a research hypothesis.
- [unclear] xBD/xBD-S12 is assumed to be a natural extension path, but no indexed local damage dataset or damage-specific trainer currently verifies that.

## 10. Implied Research Question

- [inferred] The repo currently implies this concrete question: Do unsupervised multispectral Sentinel-2 change priors, especially Difference-Subspace and PCA-diff maps, improve or explain supervised OSCD binary change segmentation compared with raw pre/post Sentinel-2 bands alone?
- [inferred] A safer thesis framing today is change segmentation with interpretable subspace priors, with disaster damage segmentation as future transfer work.

## 11. Smallest Next Experiment

- [recommendation] Re-run one controlled minimal OSCD experiment from scratch: generate Phase 1 fast priors into a new timestamped/smoke output, then train `phase2/configs/oscd_seg_baseline.yaml` and `phase2/configs/oscd_seg_E1_raw_ds.yaml` for `--max_epochs 1` on CUDA, evaluate both, and compare whether the pipeline completes and produces coherent metrics.
- [recommendation] Keep this first experiment about pipeline integrity, not performance. Success criteria: fresh priors exist, one raw model and one raw+DS model train/evaluate without missing files, result JSON/CSV files exist, and one visualization can be generated for a test city.
- [recommendation] After that, run a small but meaningful controlled comparison: E0 raw vs E1 raw+DS vs E2 raw+PCA for a fixed seed and modest epoch budget, with the same threshold protocol.

## 12. Docs To Create Or Update Next

- [recommendation] Update `README.md` to explicitly say the current runnable project is OSCD binary change segmentation, while damage segmentation is future work.
- [recommendation] Create a short `docs/RUNBOOK.md` with the exact minimal commands for: environment check, Phase 1 fast priors, Phase 2 smoke train/eval, and visualization.
- [recommendation] Create `docs/RESULTS_AUDIT.md` before citing any old metrics. It should map each result claim to the exact config, command, git hash, output path, seed, epoch count, and whether it was freshly reproduced.
- [recommendation] Create `docs/DATA_LAYOUT.md` documenting expected OSCD, MultiSenGE, and xBD layouts, including which local paths are ignored by Git.
- [recommendation] If damage work resumes, create `docs/DAMAGE_PIPELINE_PLAN.md` before coding: define target dataset form, labels/classes, CSV index schema, preprocessing, train/eval scripts, metrics, and the first xBD smoke test.
