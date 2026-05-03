# Adversarial Re-Entry Audit

Date: 2026-05-02  
Root repository: `DS_damage_segmentation`  
Nested notes repository: `research-notes` (the prompt called this `resaerch-notes`; the actual folder name is `research-notes`)  
Audit posture: adversarial, code-first, claims are not treated as truth unless supported by runnable code/config/data evidence.

## Evidence Policy

- [observed] Directly observed from source, configs, scripts, directory structure, or lightweight logs.
- [claim] Stated in README, reports, thesis draft, run notes, or research notes.
- [inferred] Reasonable conclusion from observed files, but not directly asserted by one file.
- [unverified] Not reproduced during this audit.
- [contradiction] Narrative claim conflicts with code/config/evidence or with another narrative claim.
- [risk] A weakness that can invalidate reproducibility, interpretation, or thesis framing.
- [recommendation] Concrete action before relying on the claim.

No code was modified during the audit. Only this requested document was created.

## Files Surveyed

[observed] Phase A inspected source/config/script/data-structure material only: `phase1/`, `phase2/`, `configs`, `scripts`, `requirements*.txt`, data directory shape, and lightweight run logs where needed. Phase A intentionally did not read `README.md`, `CODEBASE_AUDIT.md`, `ds_damage_segmentation.tex`, phase reports, or notes.

[observed] Phase B then inspected narrative material: `README.md`, `CODEBASE_AUDIT.md`, `ds_damage_segmentation.tex`, `RUN_PIPELINE.md`, `PIPELINE_RERUN_LOG.txt`, `phase1/docs/`, `phase2/docs/`, and narrative/textual files under `research-notes/`, including `README.md`, `paths_menu.md`, `decisions_log.md`, `drafts_digest.md`, `gaps_towatch.md`, `glossary.md`, `master/*.md`, `master/*.tex`, `notes/*.md`, `phases/phase1_report.md`, `spec/*.md`, `spec_snippets.md`, `coverage_matrix.csv`, and `refs_links/benchmark_watchlist.md`. Bibliography/bookmark dumps such as `Links.md`, `.bib`, `.xml`, and image assets were inventoried but not treated as evidence for project truth beyond showing reference-gathering activity.

## Phase A - Code-First Blind Audit

### 1. What The Repo Objectively Appears To Implement

[observed] The repository implements a two-phase Sentinel-2 change-detection/change-segmentation research pipeline.

[observed] `phase1` implements unsupervised or classical change-score generation over OSCD and MultiSenGE:

- Difference Subspace (DS) scores.
- PCA-difference scores.
- Pixel-difference/CVA-style scores.
- Celik-style local PCA plus k-means.
- IR-MAD.
- Geodesic/prior maps and second-order temporal DS utilities.

[observed] `phase2` implements supervised binary OSCD change segmentation:

- Raw pre/post Sentinel-2 stacks as input.
- Optional Phase 1 prior channels appended to raw inputs.
- U-Net, ResNet-backbone U-Net, prior-fusion U-Net, and Siamese U-Net variants.
- BCE plus Dice loss.
- Patch training and stitched city-level evaluation.

[observed] The codebase name says damage segmentation, and there is a generic `DamageDatasetAdapter`, but the integrated runnable path is OSCD binary change segmentation, not disaster damage segmentation.

[inferred] The codebase is best described as "unsupervised multispectral change priors for supervised Sentinel-2 change segmentation" rather than a completed "damage segmentation" pipeline.

### 2. Main Runnable Entry Points

[observed] Phase 1 OSCD evaluation:

```powershell
python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_default.yaml --oscd_root data/OSCD --output_dir phase1/outputs/<run> --save_change_maps
```

[observed] Phase 1 OSCD geodesic priors:

```powershell
python -m phase1.eval.run_oscd_geodesic_priors --config phase1/configs/oscd_geodesic_priors.yaml --oscd_root data/OSCD --output_dir phase1/outputs/<run>
```

[observed] Phase 1 MultiSenGE manifest and temporal/geodesic runs:

```powershell
python -m phase1.scripts.build_multisenge_manifest --multisenge_root data/MultiSenGE --output_path phase1/outputs/<manifest>.json
python -m phase1.eval.run_multisenge_temporal_geodesic --config phase1/configs/multisenge_temporal_geodesic.yaml --multisenge_root data/MultiSenGE --manifest phase1/outputs/<manifest>.json --output_dir phase1/outputs/<run>
```

[observed] Phase 2 training:

```powershell
python -m phase2.train.train_oscd_seg --config phase2/configs/oscd_seg_baseline.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/<run> --device cpu --max_epochs 1
```

[observed] Phase 2 evaluation:

```powershell
python -m phase2.eval.evaluate_oscd_seg --config phase2/configs/oscd_seg_baseline.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --checkpoint phase2/outputs/<run>/best.ckpt --output_dir phase2/outputs/<eval> --device cpu
```

[observed] Phase 2 visualization scripts exist: `phase2.viz.viz_seg_predictions` and `phase2.viz.viz_oscd_combined`.

[risk] PowerShell sweep scripts exist (`phase2/scripts/run_phase2_core_experiments.ps1`, `phase2/scripts/run_phase2_sweep.ps1`), but they are not minimal liveness checks. Some modes can overwrite or retain/delete outputs. `clean_house.ps1` is explicitly a cleanup script and should not be used during re-entry.

### 3. What Data The Code Expects

[observed] OSCD is expected under `data/OSCD` with the Onera directory naming convention:

- `onera_satellite_change_detection dataset__images/<city>/imgs_1_rect/B01.tif...B12.tif,B8A.tif`
- `onera_satellite_change_detection dataset__images/<city>/imgs_2_rect/B01.tif...B12.tif,B8A.tif`
- labels under `onera_satellite_change_detection dataset__train_labels/<city>/cm` or `...test_labels/<city>/cm`

[observed] The OSCD loader prefers `cm.png` when present, then falls back to `*-cm.tif`.

[observed] Local OSCD samples exist. For example, `brasilia/imgs_1_rect` contains 13 band TIFFs, and its label folder contains `brasilia-cm.tif` and `cm.png`.

[observed] MultiSenGE is expected under `data/MultiSenGE` with `labels/*.json`, `s2/*.tif`, `ground_reference/*.tif`, and optional S1 files.

[risk] The MultiSenGE manifest code constructs S1 relative paths like `s1/<name>.tif`, while the observed local S1 files appear under `data/MultiSenGE/s1/s1/<name>.tif`. S1-enabled MultiSenGE runs may silently fail or miss S1 unless this path assumption is checked.

[observed] `data/xbd` contains large xBD archives/tars/parts and PDFs, but no observed adapter CSVs matching the generic damage adapter's expected `pre`, `post`, and `mask` path columns.

[observed] `DamageDatasetAdapter` expects CSV indexes with pre/post/mask file paths. It is not wired into the OSCD trainer/evaluator.

### 4. Models And Methods Actually Implemented

[observed] Phase 1 implemented methods:

- DS residual-stacked and eigen variants in `phase1/ds/pca_utils.py` and `phase1/ds/ds_scores.py`.
- Geodesic prior utilities in `phase1/priors/geodesic_priors.py` and `phase1/subspace/geodesic.py`.
- Second-order DS/time-series utilities in `phase1/subspace/second_order_ds.py`.
- Pixel L2 difference.
- CVA-style score.
- PCA-diff.
- Celik PCA+k-means.
- IR-MAD.

[risk] CVA and pixel-difference are effectively the same L2 magnitude score in the current code. Treating them as independent baselines would overstate method diversity.

[observed] Phase 2 implemented models:

- `UNet2D`
- `UNet2DResNetBackbone`
- `PriorsFusionUNet`
- `SiameseUNet2D`

[observed] Phase 2 implemented training/eval components:

- OSCD patch dataset with raw pre/post stacking and optional priors.
- Train-only augmentation logic.
- BCE plus Dice loss.
- IoU/F1/precision/recall/accuracy, AUROC, and PR-AUC metrics.
- Stitched full-city evaluation from overlapping patch predictions.

[risk] `PriorsFusionUNet` assumes both raw and prior branches have positive channel counts. It is not robust to a priors-only or raw-only fusion configuration with zero channels.

### 5. Experiments Implied By Configs And Scripts

[observed] Phase 1 configs imply:

- Full OSCD evaluation across DS, pixel/CVA/PCA/Celik/IR-MAD.
- Fast OSCD prior generation with residual DS, pixel, and PCA variants.
- Eigen-variant DS runs.
- OSCD geodesic-prior generation.
- MultiSenGE manifesting, visualization, and temporal/geodesic experiments.

[observed] Phase 2 configs imply:

- E0 raw U-Net baseline.
- E1 raw plus DS projection.
- E1b raw plus DS cross residual.
- E2 raw plus PCA-diff.
- E3 raw plus DS plus PCA-diff.
- E4 raw plus pixel-difference.
- E5 raw plus Celik.
- E6 raw plus IR-MAD.
- Siamese U-Net baseline.
- ResNet raw and ResNet raw plus DS/PCA.
- Prior-fusion U-Net.
- Geodesic-prior experiment.

[observed] Most Phase 2 training configs default to 50 epochs; documentation and scripts also describe 150-epoch runs. Those are not appropriate first re-entry commands.

### 6. Incomplete, Fragile, Duplicated, Or Suspicious Parts

[risk] The repository title and several docs frame "damage segmentation", but the integrated code path is OSCD binary change segmentation.

[risk] A generic damage adapter exists, but there is no integrated `train_damage_seg.py`, no xBD-S12 loader wired into training/evaluation, and no multi-class damage loss/metric pipeline.

[risk] `phase2.train.train_oscd_seg` requires `--phase1_change_maps_root` even for raw-only configs where no prior channels are enabled. This is harmless but confusing for re-entry and scripts.

[risk] Phase 2 configs omit explicit `band_order`; the loader falls back to a 13-band default. This is functional but leaves an important experimental assumption implicit.

[risk] `phase1.eval.run_oscd_eval.aggregate_metrics()` is a stub returning an empty dict, though the main run path appears not to depend on it.

[risk] Phase 1 runtime attribution is not per-method isolated. In the full-tile path and tiled path, elapsed time is assigned in ways that can give the same or shared runtime to multiple methods. Runtime claims from those outputs should be considered invalid until instrumented per method.

[risk] Existing outputs are mostly git-ignored local artifacts. Result reproducibility depends on local files that may not exist on another machine.

[risk] `PIPELINE_RERUN_LOG.txt` later shows CPU-only PyTorch and an incomplete Phase 2 rerun log. This undermines any claim that the rerun completed the full pipeline.

[observed] A lightweight AST parse of the Phase 1/Phase 2 Python files succeeded, so the Python files are syntactically loadable at parse level.

### 7. Research Question Implied By Code Alone

[inferred] Ignoring all narrative docs, the code implies this research question:

> Do unsupervised multispectral change-score maps and subspace priors (DS, PCA-diff, pixel/CVA, Celik, IR-MAD, geodesic) improve supervised Sentinel-2 OSCD binary change segmentation over raw pre/post Sentinel-2 alone, and which prior channels help or hurt under a consistent split and metric protocol?

[contradiction] The code does not imply a completed disaster-damage-mapping thesis. It implies a change-segmentation thesis with possible future transfer to damage datasets.

## Phase B - Claims Extraction

[claim] The table below extracts material narrative claims only. It does not accept them as true. Repeated claims across files are deduplicated where they express the same assertion.

| ID | Claim Type | Claim | Source File(s) |
| --- | --- | --- | --- |
| C01 | [implementation claim] | The project is "Difference-Subspace Priors for Change & Damage Segmentation". | `README.md` |
| C02 | [implementation claim] | Phase 1 implements DS and classical change detection on OSCD and MultiSenGE. | `README.md`, `phase1/docs/phase1_report.md`, `research-notes/phases/phase1_report.md` |
| C03 | [implementation claim] | Phase 2 implements OSCD segmentation with raw S2 and DS/PCA prior channels. | `README.md`, `phase2/docs/phase2_report.md`, `research-notes/master/current_scope.md` |
| C04 | [implementation claim] | Phase 3 / damage-labeled xBD or xBD-S12 work is future or not yet implemented. | `README.md`, `phase2/docs/phase2_report.md`, `research-notes/master/current_scope.md`, `research-notes/paths_menu.md` |
| C05 | [method claim] | DS projection and DS cross-residual are the primary subspace change scores. | `README.md`, `phase1/docs/spec_phase1_ds_oscd.md`, `research-notes/spec/spec_phase1_ds_oscd.md` |
| C06 | [method claim] | Baselines include pixel difference, CVA, PCA-diff, Celik PCA+k-means, and IR-MAD. | `README.md`, `phase1/docs/phase1_report.md`, `research-notes/phases/phase1_report.md` |
| C07 | [method claim] | Phase 1 uses OSCD train split only for global threshold calibration, with val/test held out. | `phase1/docs/phase1_report.md`, `research-notes/spec/spec_phase1_ds_oscd.md` |
| C08 | [method claim] | MultiSenGE is an unlabeled development/visualization testbed, not a primary labeled damage benchmark. | `phase1/docs/phase1_report.md`, `research-notes/drafts_digest.md`, `research-notes/master/master_outline.md` |
| C09 | [method claim] | Phase 2 compares raw-only, raw+single-prior, raw+multiple-prior, ResNet, fusion, and Siamese variants. | `README.md`, `phase2/docs/phase2_report.md`, `RUN_PIPELINE.md` |
| C10 | [method claim] | Priors-only sanity checks are part of the Phase 2 comparison. | `README.md`, `phase2/docs/spec_phase2_oscd_seg.md` |
| C11 | [result claim] | Phase 1 test AUROC: `pca_diff` is strongest at about 0.813. | `README.md`, `phase1/docs/phase1_report.md`, `ds_damage_segmentation.tex`, `research-notes/master/current_scope.md` |
| C12 | [result claim] | Phase 1 `ds_projection` is competitive but below `pca_diff`, around AUROC 0.755. | `README.md`, `phase1/docs/phase1_report.md`, `ds_damage_segmentation.tex` |
| C13 | [result claim] | Phase 1 `ds_cross_residual` is weak, around AUROC 0.53-0.56 depending on report. | `README.md`, `phase1/docs/phase1_report.md`, `research-notes/phases/phase1_report.md` |
| C14 | [result claim] | Phase 1 pixel-diff/CVA are around AUROC 0.756; Celik around 0.649; IR-MAD around 0.704. | `README.md`, `phase1/docs/phase1_report.md`, `ds_damage_segmentation.tex` |
| C15 | [result claim] | Phase 2 150-epoch single-seed E0 raw U-Net result is about IoU 0.223, F1 0.343, AUROC 0.869, PR-AUC 0.431. | `README.md`, `phase2/docs/phase2_report.md`, `ds_damage_segmentation.tex` |
| C16 | [result claim] | Phase 2 E1 raw+DS projection is the best current mean IoU/F1 U-Net result, about IoU 0.273 and F1 0.401. | `README.md`, `phase2/docs/phase2_report.md`, `research-notes/master/current_scope.md` |
| C17 | [result claim] | Phase 2 Celik prior is strong and has best PR-AUC in the listed U-Net runs, about 0.448. | `README.md`, `phase2/docs/phase2_report.md`, `ds_damage_segmentation.tex` |
| C18 | [result claim] | Phase 2 fusion raw+DS+PCA has highest AUROC around 0.888 but not best IoU/F1. | `README.md`, `phase2/docs/phase2_report.md`, `ds_damage_segmentation.tex` |
| C19 | [result claim] | DS projection improves IoU on 8/10 held-out OSCD test cities. | `research-notes/master/current_scope.md`, `research-notes/master/session_audit_2026-03-24.md` |
| C20 | [result claim] | Per-city Chongqing figures show different priors/models winning different metrics. | `phase2/docs/phase2_report.md` |
| C21 | [implementation claim] | Phase 1 saved outputs include JSON/CSV summaries and change maps under `oscd_change_maps/<split>/<method>/<city>_score.npy`. | `README.md`, `phase1/docs/phase1_run_guide.md` |
| C22 | [implementation claim] | Phase 2 saved outputs include checkpoints, metrics JSON/CSV, predictions, and visualizations. | `README.md`, `phase2/docs/phase2_run_guide.md`, `RUN_PIPELINE.md` |
| C23 | [implementation claim] | The full rerun pipeline should run with CUDA. | `RUN_PIPELINE.md` |
| C24 | [implementation claim] | A rerun log shows Phase 1 fast priors and geodesic priors completed successfully. | `PIPELINE_RERUN_LOG.txt` |
| C25 | [implementation claim] | The rerun log begins Phase 2 E0 training. | `PIPELINE_RERUN_LOG.txt` |
| C26 | [risk claim] | Main Phase 2 evidence is single-seed and threshold 0.5; more seeds and threshold tuning are needed. | `README.md`, `phase2/docs/phase2_report.md`, `research-notes/master/current_scope.md` |
| C27 | [risk claim] | OSCD is not a disaster damage benchmark. | `research-notes/master/current_scope.md`, `research-notes/master/master_outline.md`, `research-notes/paths_menu.md` |
| C28 | [risk claim] | Current code does not implement a real damage-segmentation pipeline. | `research-notes/master/current_scope.md`, `research-notes/master/master_skeleton.md` |
| C29 | [thesis claim] | Active thesis should be interpretable unsupervised multispectral priors for Sentinel-2 change segmentation. | `research-notes/master/current_scope.md`, `research-notes/master/master_outline.md`, `research-notes/paths_menu.md` |
| C30 | [thesis claim] | The strongest standalone unsupervised detector is not necessarily the strongest downstream segmentation prior. | `research-notes/master/master_outline.md`, `research-notes/master/master_skeleton.md` |
| C31 | [thesis claim] | Older broad disaster-resilience/SSC/UAV/MCDA/DMaaS ideas are cold archive, not active thesis scope. | `research-notes/README.md`, `research-notes/paths_menu.md`, `research-notes/decisions_log.md` |
| C32 | [future-work claim] | xBD-S12 is the only warm extension path. | `research-notes/paths_menu.md`, `research-notes/master/master_outline.md`, `research-notes/decisions_log.md` |
| C33 | [future-work claim] | Future work includes multi-seed validation, threshold tuning, ImageNet pretraining, deeper fusion, MultiSenGE pseudo-label pretraining, and xBD/xBD-S12 damage transfer. | `README.md`, `phase2/docs/phase2_report.md`, `ds_damage_segmentation.tex` |
| C34 | [future-work claim] | SSC, geodesic-weighted SSC, UAV/edge deployment, MCDA, IoT, DMaaS, graph decision layers, uncertainty, and operational dashboards are future or archived ideas. | `research-notes/drafts_digest.md`, `research-notes/gaps_towatch.md`, `research-notes/paths_menu.md` |
| C35 | [method claim] | xBD-S12 is a medium-resolution bridge from VHR damage labels to Sentinel-1/2 damage mapping. | `research-notes/drafts_digest.md`, `research-notes/refs_links/benchmark_watchlist.md` |
| C36 | [risk claim] | External benchmark comparisons are unsafe unless thresholding, masking, averaging, and compute protocols are aligned. | `research-notes/refs_links/benchmark_watchlist.md`, `research-notes/decisions_log.md` |
| C37 | [implementation claim] | `CODEBASE_AUDIT.md` says Phase 1 DS core is consistent with DS papers/reference and methodologically solid. | `CODEBASE_AUDIT.md` |
| C38 | [risk claim] | `CODEBASE_AUDIT.md` says Phase 2 had critical bugs: val/test augmentation, non-stitched eval, JSON-only eval outputs. | `CODEBASE_AUDIT.md` |
| C39 | [implementation claim] | `CODEBASE_AUDIT.md` says Siamese baseline is not fully implemented. | `CODEBASE_AUDIT.md` |
| C40 | [thesis claim] | `ds_damage_segmentation.tex` frames the thesis as satellite damage mapping, while also saying true damage datasets are future transfer work. | `ds_damage_segmentation.tex` |

## Phase C - Contradiction Matrix

| Claim | Source File | Supporting Code/Config Evidence | Missing Evidence | Contradiction If Any | Confidence | Action Needed |
| --- | --- | --- | --- | --- | --- | --- |
| [claim] Project is damage segmentation / satellite damage mapping. | `README.md`, `ds_damage_segmentation.tex` | [observed] Generic `DamageDatasetAdapter`; xBD archives exist locally. | Integrated xBD/xBD-S12 loader, training script, damage labels index CSVs, multi-class damage loss/metrics, damage eval. | [contradiction] Runnable core is OSCD binary change segmentation, not damage segmentation. | high | [recommendation] Reframe thesis as change segmentation unless a real damage pipeline is implemented and evaluated. |
| [claim] Active thesis is Sentinel-2 change priors for OSCD segmentation. | `research-notes/master/current_scope.md`, `paths_menu.md` | [observed] Phase 1 prior generation and Phase 2 OSCD segmentation code/configs match this. | Independent reproduction of reported results. | No major contradiction; this is the best-aligned narrative. | high | [recommendation] Make this the default thesis story for now. |
| [claim] Phase 1 implements DS and classical baselines on OSCD. | `README.md`, `phase1/docs/phase1_report.md` | [observed] `phase1/ds`, `phase1/baselines`, `phase1/eval/run_oscd_eval.py`, configs. | Mathematical validation against papers/reference outputs. | No direct contradiction. | high | [recommendation] Treat implementation existence as true; treat correctness claims as needing unit/reference tests. |
| [claim] Phase 1 runtime per method/tile is reported. | `phase1/docs/spec_phase1_ds_oscd.md`, result summaries | [observed] Runtime fields are emitted in Phase 1 evaluation. | Per-method isolated timing. | [contradiction] `run_oscd_eval.py` attributes shared elapsed time to methods in ways that are not true per-method timings. | high | [recommendation] Do not use runtime comparisons in thesis until instrumentation is fixed and rerun. |
| [claim] CVA is a separate baseline from pixel difference. | `README.md`, Phase 1 reports/specs | [observed] Both baseline names are present. | Distinct CVA implementation beyond L2 magnitude. | [contradiction] Current code makes CVA effectively identical to pixel L2 difference. | high | [recommendation] Collapse them in interpretation or implement a meaningfully distinct CVA variant. |
| [claim] Phase 1 `pca_diff` AUROC about 0.813 and strongest standalone. | `README.md`, reports, thesis draft | [observed] Code can compute PCA-diff; local output paths are referenced. | Fresh rerun or checksum-backed result verification during this audit. | No code contradiction, but result is [unverified]. | medium | [recommendation] Recompute or at least audit saved CSV/JSON plus git hash before citing. |
| [claim] DS projection AUROC about 0.755, competitive but not strongest. | `README.md`, reports, thesis draft | [observed] DS projection implemented and output naming exists. | Fresh rerun or artifact integrity check. | No code contradiction, but result is [unverified]. | medium | [recommendation] Reproduce Phase 1 fast/full run before final thesis tables. |
| [claim] DS cross-residual is weaker. | Phase 1 reports, current notes | [observed] DS cross residual implemented and configs include it. | Fresh artifact check. | Minor inconsistency in exact reported AUROC range across docs. | medium | [recommendation] Use exact value only from a current rerun summary. |
| [claim] Phase 2 implements raw/prior OSCD segmentation. | `README.md`, Phase 2 docs | [observed] `OSCDSegmentationDataset`, `train_oscd_seg.py`, `evaluate_oscd_seg.py`, configs. | None for implementation existence. | No contradiction. | high | [recommendation] Use this as the implemented supervised component. |
| [claim] Priors-only sanity checks exist. | `README.md`, `phase2/docs/spec_phase2_oscd_seg.md` | [observed] Feature flags can disable raw S2 conceptually. | Actual priors-only config and robust model branch for zero raw channels. | [contradiction] No obvious priors-only config observed; `PriorsFusionUNet` is fragile for zero-channel branches. | high | [recommendation] Remove priors-only claims or add explicit tested configs. |
| [claim] Siamese baseline is not fully implemented. | `CODEBASE_AUDIT.md` | [observed] `phase2/models/siamese_unet.py` and a Siamese config/script path exist. | Evidence that Siamese run completed and was evaluated. | [contradiction] The audit is outdated at least at implementation-existence level. | high | [recommendation] Mark `CODEBASE_AUDIT.md` historical; do not use as current truth. |
| [claim] Phase 2 eval has val/test augmentation, per-patch eval, and JSON-only output bugs. | `CODEBASE_AUDIT.md` | [observed] Current dataset sets `is_train = split == "train"`; eval stitches city predictions; eval writes CSV/JSON. | Regression test proving behavior. | [contradiction] Current code appears to have fixed these alleged bugs. | high | [recommendation] Keep the warning only as historical context; verify with a smoke eval. |
| [claim] Phase 2 E1 raw+DS is best IoU/F1 in 150-epoch run. | README, Phase 2 report, thesis draft | [observed] Config and model path support such a run; local outputs are referenced. | Fresh rerun; multi-seed statistics; checkpoint/result hash. | No code contradiction, but result is [unverified] and single-seed. | medium | [recommendation] Treat as preliminary until at least artifact audit plus one smoke eval; ideally multi-seed rerun. |
| [claim] Fusion raw+DS+PCA has highest AUROC. | README, Phase 2 report, thesis draft | [observed] `PriorsFusionUNet` implemented. | Fresh eval and threshold analysis. | No code contradiction, but "fusion" is a lightweight 1x1 reweight plus U-Net, not a sophisticated fusion method. | medium | [recommendation] Describe architecture modestly and reproduce evaluation before citing. |
| [claim] DS improves IoU on 8/10 held-out cities. | `research-notes/master/current_scope.md` | [observed] Per-city evaluation machinery likely exists. | Direct per-city result table verification. | [unverified] No code/config evidence alone proves the 8/10 count. | low | [recommendation] Write a small result-audit script over existing JSON/CSV before using this claim. |
| [claim] Rerun pipeline completed Phase 1 and started Phase 2. | `PIPELINE_RERUN_LOG.txt` | [observed] Log shows Phase 1 fast priors exit 0 and geodesic exit 0. | Phase 2 exit status and resulting metrics. | [contradiction] The log ends after starting Phase 2 E0 command; it does not prove Phase 2 completion. | high | [recommendation] Do not cite the rerun as a full pipeline rerun. |
| [claim] Full pipeline should run on CUDA. | `RUN_PIPELINE.md` | [observed] Scripts default to CUDA; trainer checks CUDA availability. | CUDA-enabled PyTorch in current environment. | [contradiction] `PIPELINE_RERUN_LOG.txt` reports Torch CPU build and `cuda_available=False`. | high | [recommendation] Use `--device cpu` for smoke tests; reinstall CUDA PyTorch only when doing real reruns. |
| [claim] xBD/xBD-S12 is a warm damage extension. | README, notes, thesis draft | [observed] xBD archives exist; generic adapter exists. | Extracted/indexed xBD-S12 dataset, loader, train/eval, damage metrics. | No contradiction if stated as future. Contradiction if stated as current result. | high | [recommendation] Keep as future/warm only until code and data index exist. |
| [claim] MultiSenGE can support temporal/geodesic DS analysis. | Phase 1 docs, notes | [observed] MultiSenGE loaders and temporal/geodesic scripts exist; S2 path shape observed. | Validation of S1 path and manifest coverage. | [risk] Optional S1 path may be wrong for current local layout. | medium | [recommendation] First run manifest smoke on S2 only; audit S1 path before claiming multimodal use. |
| [claim] OSCD split is train/val/test and val is held out. | Phase 2 docs/configs | [observed] Configs hardcode 11 train, 3 val, 10 test city lists. | Confirmation this matches intended official split or a documented custom split. | Potential wording issue: OSCD official split is usually train/test; val is project-defined from train. | medium | [recommendation] Document exact city lists and call val a project validation split unless externally verified. |
| [claim] Phase 1 DS core is methodologically solid and correct. | `CODEBASE_AUDIT.md` | [observed] DS formulas/utilities are implemented and syntactically parse. | Unit tests against analytical cases or reference implementation. | Not contradicted, but overconfident for this audit. | medium | [recommendation] Add minimal numerical tests before relying on "correct" in thesis language. |
| [claim] Older broad disaster-resilience platform ideas remain relevant. | `drafts_digest.md`, `gaps_towatch.md`, old notes | [observed] No integrated code for SSC-heavy core, UAV/edge, IoT, MCDA service, DMaaS, graph decision layer. | Implementations, datasets, experiments. | [contradiction] They conflict with current implemented scope if presented as current work. | high | [recommendation] Move all broad-system language to future work or appendix only. |

## Phase D - Minimal Reproduction Plan

Goal: test whether the project is alive without running long training, deleting outputs, or relying on narrative claims.

### Step 0 - Environment And Forward-Pass Smoke

[recommendation] Run this first. It creates no output files. It checks imports, OSCD loader, one sample, channel inference, model construction, and one CPU forward pass.

```powershell
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

@'
from pathlib import Path
import yaml
import torch

from phase2.data.oscd_seg_dataset import OSCDSegmentationDataset
from phase2.train.train_oscd_seg import infer_channel_counts, build_model

cfg = yaml.safe_load(Path("phase2/configs/oscd_seg_baseline.yaml").read_text(encoding="utf-8"))

# Keep this as a liveness check, not a reported experiment.
cfg["dataset"]["patch_size"] = 128
cfg["dataset"]["patch_overlap"] = 0
cfg["model"]["base_channels"] = 8
cfg["model"]["depth"] = 2

ds = OSCDSegmentationDataset(
    oscd_root=Path("data/OSCD"),
    split="val",
    cfg=cfg,
    phase1_change_maps_root=Path("phase1/outputs/oscd_saved_priors_fast/oscd_change_maps"),
)
sample = ds[0]
counts = infer_channel_counts(cfg)
model = build_model(cfg, counts["total"]).eval()
with torch.no_grad():
    logits = model(sample["x"].unsqueeze(0))

print("dataset_len", len(ds))
print("x", tuple(sample["x"].shape), "y", tuple(sample["y"].shape), "valid", tuple(sample["valid"].shape))
print("channels", counts)
print("logits", tuple(logits.shape))
'@ | & $py -
```

Required data:

- `data/OSCD` with the expected city folders and labels.
- `phase1/data/oscd_band_stats.json`.

Expected output files:

- None.

What success means:

- Imports work.
- OSCD paths and labels are readable.
- Dataset returns tensors.
- Model accepts the inferred channel count.
- Forward pass produces a logits tensor.

What failure means:

- Import failure: environment/requirements issue.
- File-not-found failure: data path/layout issue.
- Shape/channel failure: config/model/dataset contract issue.
- Raster read failure: local data corruption or incompatible raster dependency.

### Step 1 - Small Artifact-Producing Phase 1 Smoke

[recommendation] If Step 0 passes, run a tiny Phase 1 geodesic-prior smoke to prove CLI output writing works. This is not a full Phase 1 reproduction.

```powershell
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m phase1.eval.run_oscd_geodesic_priors `
  --config phase1/configs/oscd_geodesic_priors.yaml `
  --oscd_root data/OSCD `
  --output_dir phase1/outputs/_smoke_reentry_geodesic `
  --max_cities 1
```

Required data:

- `data/OSCD`.
- Phase 1 config `phase1/configs/oscd_geodesic_priors.yaml`.

Expected output files:

- `phase1/outputs/_smoke_reentry_geodesic/run_metadata_geodesic.json`
- `phase1/outputs/_smoke_reentry_geodesic/oscd_change_maps/<split>/geodesic_dist/<city>_score.npy`
- Quick-look mask/image files if the script emits them for the selected city.

What success means:

- Phase 1 CLI is callable.
- OSCD loader works beyond one patch.
- Output writing works.
- At least one score map is generated.

What failure means:

- Config/schema failure: Phase 1 config drift.
- Data failure: OSCD layout or label problem.
- Numerical failure: geodesic/DS dependency issue.
- Output failure: path/permission issue.

### Step 2 - Optional Existing-Checkpoint Eval Smoke

[recommendation] Only run this after checking that the checkpoint path exists. It is still not training, but it evaluates full OSCD val/test and can take longer than Step 0.

```powershell
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m phase2.eval.evaluate_oscd_seg `
  --config phase2/configs/oscd_seg_baseline.yaml `
  --oscd_root data/OSCD `
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps `
  --checkpoint phase2/outputs/runs_gpu_150ep_20251215_233309/E0_raw_unet/best.ckpt `
  --output_dir phase2/outputs/_smoke_reentry_eval_E0 `
  --device cpu
```

Required data:

- `data/OSCD`.
- Existing checkpoint at the path above, or replace with a known existing `best.ckpt`.
- For raw-only config, prior maps are not used, but the CLI still requires `--phase1_change_maps_root`.

Expected output files:

- Evaluation JSON/CSV under `phase2/outputs/_smoke_reentry_eval_E0`.
- Any metadata file emitted by the evaluator.

What success means:

- Checkpoint can be loaded.
- Evaluation pipeline runs stitched city-level inference.
- Metrics files are written.

What failure means:

- Missing checkpoint: artifact availability issue, not necessarily code failure.
- State-dict mismatch: config/checkpoint mismatch.
- Data or shape failure: dataset/model contract issue.
- CPU runtime too slow: use a smaller local eval hook later, but do not jump to training.

### What Not To Run Yet

[recommendation] Do not run these during first re-entry:

- 50-epoch or 150-epoch Phase 2 training.
- `phase2/scripts/run_phase2_core_experiments.ps1`.
- `phase2/scripts/run_phase2_sweep.ps1`.
- Any sweep with cleanup/retention behavior.
- `clean_house.ps1`, especially `-Aggressive`.
- CUDA commands until `torch.cuda.is_available()` is true in the intended Python environment.
- xBD/xBD-S12 damage training, because the integrated pipeline and index files are not established.
- Any command that deletes, overwrites, or normalizes old outputs before artifact provenance is audited.

## Bottom Line

[observed] The codebase is alive enough to support a conservative re-entry path around OSCD change segmentation and Phase 1 priors.

[contradiction] The strongest contradiction is thesis identity: "damage segmentation" is not currently implemented end-to-end. The implemented work is Sentinel-2 binary change segmentation with unsupervised prior channels.

[risk] Reported results may be real local artifacts, but they were not reproduced in this audit and are mostly single-seed. Runtime claims are especially unsafe because Phase 1 timing is not per-method isolated.

[recommendation] The immediate safe thesis framing is: interpretable unsupervised multispectral change priors for supervised Sentinel-2 change segmentation, with xBD-S12/damage mapping as future or warm extension only.
