# Phase 2 CLI Cheat Sheet (PowerShell)

Assume:
- Repo root: `E:\research_projects\DS_damage_segmentation`
- Root venv activated: `.\.venv\Scripts\Activate.ps1`
- OSCD root: `phase1\data\raw\OSCD`
- Phase‑1 change maps (priors): `phase1\outputs\oscd_saved\oscd_change_maps`

Change to Phase‑2 directory first:

```powershell
cd E:\research_projects\DS_damage_segmentation\phase2
```

## 1. Training – U‑Net Experiments (E0–E3)

- **E0 – U‑Net, raw pre/post only (baseline)**  
  `python -m train.train_oscd_seg --config configs/oscd_seg_baseline.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir outputs/oscd_seg_E0_raw`

- **E1 – U‑Net, raw + DS projection prior**  
  `python -m train.train_oscd_seg --config configs/oscd_seg_E1_raw_ds.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir outputs/oscd_seg_E1_raw_ds`

- **E2 – U‑Net, raw + PCA‑diff prior**  
  `python -m train.train_oscd_seg --config configs/oscd_seg_E2_raw_pca.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir outputs/oscd_seg_E2_raw_pca`

- **E3 – U‑Net, raw + DS + PCA‑diff priors**  
  `python -m train.train_oscd_seg --config configs/oscd_seg_priors.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir outputs/oscd_seg_E3_raw_ds_pca`

Common variant for quick smoke tests (any config): add `--max_epochs 1` to run just one epoch.

## 2. Training – ResNet‑U‑Net and PriorsFusionUNet

- **ResNet‑U‑Net, raw only**  
  `python -m train.train_oscd_seg --config configs/oscd_seg_baseline_resnet.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir outputs/oscd_seg_E0_raw_resnet`

- **ResNet‑U‑Net, raw + DS + PCA‑diff priors**  
  `python -m train.train_oscd_seg --config configs/oscd_seg_priors_resnet.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir outputs/oscd_seg_E3_raw_ds_pca_resnet`

- **PriorsFusionUNet (raw + DS + PCA‑diff, separate priors branch)**  
  `python -m train.train_oscd_seg --config configs/oscd_seg_priors_fusion.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir outputs/oscd_seg_E3_raw_ds_pca_fusion`

## 3. Evaluation

Script: `phase2/eval/evaluate_oscd_seg.py`

Example – evaluate U‑Net raw‑only (`E0`) and save metrics:

- **E0 – U‑Net raw only**  
  `python -m eval.evaluate_oscd_seg --config configs/oscd_seg_baseline.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E0_raw/best.ckpt --output_dir outputs/oscd_seg_E0_raw/eval`

Same pattern for other experiments (swap config/checkpoint/output_dir):

- **E1 – U‑Net raw + DS**  
  `python -m eval.evaluate_oscd_seg --config configs/oscd_seg_E1_raw_ds.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E1_raw_ds/best.ckpt --output_dir outputs/oscd_seg_E1_raw_ds/eval`

- **E2 – U‑Net raw + PCA‑diff**  
  `python -m eval.evaluate_oscd_seg --config configs/oscd_seg_E2_raw_pca.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E2_raw_pca/best.ckpt --output_dir outputs/oscd_seg_E2_raw_pca/eval`

- **E3 – U‑Net raw + DS + PCA‑diff**  
  `python -m eval.evaluate_oscd_seg --config configs/oscd_seg_priors.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E3_raw_ds_pca/best.ckpt --output_dir outputs/oscd_seg_E3_raw_ds_pca/eval`

- **ResNet‑U‑Net raw only**  
  `python -m eval.evaluate_oscd_seg --config configs/oscd_seg_baseline_resnet.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E0_raw_resnet/best.ckpt --output_dir outputs/oscd_seg_E0_raw_resnet/eval`

- **ResNet‑U‑Net raw + DS + PCA‑diff**  
  `python -m eval.evaluate_oscd_seg --config configs/oscd_seg_priors_resnet.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E3_raw_ds_pca_resnet/best.ckpt --output_dir outputs/oscd_seg_E3_raw_ds_pca_resnet/eval`

- **PriorsFusionUNet**  
  `python -m eval.evaluate_oscd_seg --config configs/oscd_seg_priors_fusion.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E3_raw_ds_pca_fusion/best.ckpt --output_dir outputs/oscd_seg_E3_raw_ds_pca_fusion/eval`

Evaluation outputs:
- `oscd_seg_eval_results.json` – per‑city metrics.
- `oscd_seg_eval_summary.csv` – split‑level mean IoU/F1/AUROC.

## 4. Visualization

### 4.1 Per‑city segmentation summaries

Script: `phase2/viz/viz_seg_predictions.py`

- **ResNet raw‑only summaries (test cities)**  
  `python -m viz.viz_seg_predictions --config configs/oscd_seg_baseline_resnet.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E0_raw_resnet/best.ckpt --output_dir outputs/oscd_seg_E0_raw_resnet/figs_seg --cities test`

Swap config/checkpoint/output_dir for other experiments if you want their figures.

### 4.2 Combined DS/PCA + segmentation figures

Script: `phase2/viz/viz_oscd_combined.py`

- **ResNet raw‑only combined DS/PCA/segmentation (test)**  
  `python -m viz.viz_oscd_combined --config configs/oscd_seg_baseline_resnet.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E0_raw_resnet/best.ckpt --output_dir outputs/oscd_seg_E0_raw_resnet/figs_combined --cities test`

- **PriorsFusionUNet combined figures (test)**  
  `python -m viz.viz_oscd_combined --config configs/oscd_seg_priors_fusion.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint outputs/oscd_seg_E3_raw_ds_pca_fusion/best.ckpt --output_dir outputs/oscd_seg_E3_raw_ds_pca_fusion/figs_combined --cities test`

## 5. Notes

- Training runs (especially ResNet and Fusion) can be GPU/CPU intensive; use `--max_epochs` for quick smoke tests.
- All commands assume Phase‑1 priors have been generated and saved via `phase1/eval/run_oscd_eval.py` with `--save_change_maps`.
