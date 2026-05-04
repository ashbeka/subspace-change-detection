# Phase 2 CLI Cheat Sheet (PowerShell)

Assume:
- Repo root: `E:\research_projects\DS_damage_segmentation`
- Root venv activated: `.\.venv\Scripts\Activate.ps1`
- OSCD root: `data\OSCD`
- Phase 1 change maps (priors):
  - Fast priors (DS/PCA/pixel; used by E0–E4): `phase1\outputs\oscd_saved_priors_fast\oscd_change_maps`
  - Full priors (adds Celik/IR‑MAD; required by E5–E6): `phase1\outputs\oscd_saved_full\oscd_change_maps`

All commands below are run from the repo root:

```powershell
cd E:\research_projects\DS_damage_segmentation
```

GPU: Phase 2 CLIs default to `--device cuda` and will error if CUDA isn’t available (to avoid accidentally training on CPU). For debugging only, add `--device cpu`.

Notes:
- To match the latest runs, pass `--epochs 150` (this also updates cosine scheduler `T_max` to match).
- For quick smoke tests, add `--max_epochs 1`.
- To resume after a crash, add `--resume` and keep the same `--output_dir`.

## 1. Training — U‑Net Experiments (E0–E6)

- **E0 — U‑Net, raw pre/post only (baseline)**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/core/E0_raw_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E0_raw`

- **E1 — U‑Net, raw + DS projection prior**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/core/E1_raw_ds_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E1_raw_ds`

- **E1b — U‑Net, raw + DS cross‑residual prior**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/core/E1b_raw_ds_cross_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E1b_raw_ds_cross`

- **E2 — U‑Net, raw + PCA‑diff prior**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/core/E2_raw_pca_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E2_raw_pca`

- **E3 — U‑Net, raw + DS + PCA‑diff priors**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/core/E3_raw_ds_pca_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca`

- **E4 — U‑Net, raw + pixel‑diff prior**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/extended/E4_raw_pixel_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E4_raw_pixel`

- **E5 — U‑Net, raw + Celik prior (requires full Phase 1 run)**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/extended/E5_raw_celik_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_full/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E5_raw_celik`

- **E6 — U‑Net, raw + IR‑MAD prior (requires full Phase 1 run)**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/extended/E6_raw_irmad_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_full/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E6_raw_irmad`

Resume after a crash (continues from `last.ckpt` if present, else `best.ckpt`):

- `python -m phase2.train.train_oscd_seg ... --output_dir <same_dir_as_before> --resume`

If you want to start a fresh run in the same output directory, add `--overwrite_output_dir` (this will overwrite `train_log.json` / checkpoints).

## 2. Training — ResNet‑U‑Net and PriorsFusionUNet

- **ResNet‑U‑Net, raw only**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/extended/E0_raw_resnet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E0_raw_resnet`

- **ResNet‑U‑Net, raw + DS + PCA‑diff priors**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/extended/E3_raw_ds_pca_resnet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca_resnet`

- **PriorsFusionUNet (raw + DS + PCA‑diff, separate priors branch)**  
  `python -m phase2.train.train_oscd_seg --config phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion`

## 3. Evaluation

Script: `phase2/eval/evaluate_oscd_seg.py`

Example — evaluate U‑Net raw‑only (`E0`) and save metrics:

- `python -m phase2.eval.evaluate_oscd_seg --config phase2/configs/oscd/core/E0_raw_unet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --checkpoint phase2/outputs/oscd_seg_E0_raw/best.ckpt --output_dir phase2/outputs/oscd_seg_E0_raw/eval`

Same pattern for other experiments (swap config, change‑maps root, checkpoint, output_dir).

Evaluation outputs:
- `oscd_seg_eval_results.json` — per‑city metrics
- `oscd_seg_eval_summary.csv` — mean IoU/F1/AUROC/PR‑AUC

## 4. Visualization

### 4.1 Per‑city segmentation summaries

Script: `phase2/viz/viz_seg_predictions.py`

- **ResNet raw‑only summaries (test cities)**  
  `python -m phase2.viz.viz_seg_predictions --config phase2/configs/oscd/extended/E0_raw_resnet.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --checkpoint phase2/outputs/oscd_seg_E0_raw_resnet/best.ckpt --output_dir phase2/outputs/oscd_seg_E0_raw_resnet/figs_seg --cities test`

### 4.2 Combined priors + segmentation figures

Script: `phase2/viz/viz_oscd_combined.py`

- **PriorsFusionUNet combined figures (test)**  
  `python -m phase2.viz.viz_oscd_combined --config phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml --oscd_root data/OSCD --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps --checkpoint phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion/best.ckpt --output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion/figs_combined --cities test`

