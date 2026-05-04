# Phase 2 Run Guide – OSCD Segmentation with DS/PCA Priors

This file is a **practical guide** for Phase 2:

- Which configs correspond to which experiments.
- How to train, evaluate, and visualize.
- How to run ResNet and PriorsFusionUNet variants.
- How to (later) try longer training and ImageNet pretraining.

All commands assume you start in the **repo root**.

GPU: Phase 2 CLIs default to `--device cuda` and will error if CUDA isn’t available (to avoid accidentally training on CPU). Install a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ (or pass `--device cpu` for debugging).

OSCD root and Phase‑1 change maps are always:

- `--oscd_root data/OSCD`
- `--phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps`

The priors folder above is produced by Phase 1 (recommended):
- `python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_priors_fast.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_saved_priors_fast --save_change_maps`

---

## 1. Configs and Experiments

Key configs in `phase2/configs/`:

- `E0_raw_unet.yaml`
  - U‑Net, raw S2 only (`E0`).
- `E1_raw_ds_unet.yaml`
  - U‑Net, raw + DS prior (`E1`).
- `E1b_raw_ds_cross_unet.yaml`
  - U‑Net, raw + DS cross‑residual prior (extra DS baseline).
- `E4_raw_pixel_unet.yaml`
  - U‑Net, raw + pixel‑diff prior (classical baseline prior).
- `E5_raw_celik_unet.yaml`
  - U‑Net, raw + Celik prior (classical baseline; requires Phase‑1 full baseline run).
- `E6_raw_irmad_unet.yaml`
  - U‑Net, raw + IR‑MAD prior (classical baseline; requires Phase‑1 full baseline run).
- `E2_raw_pca_unet.yaml`
  - U‑Net, raw + PCA‑diff prior (`E2`).
- `E3_raw_ds_pca_unet.yaml`
  - U‑Net, raw + DS + PCA‑diff (`E3`).
- `S0_raw_siamese.yaml`
  - Siamese baseline (raw only).
- `E0_raw_resnet.yaml`
  - ResNet‑U‑Net, raw only (stronger baseline).
- `E3_raw_ds_pca_resnet.yaml`
  - ResNet‑U‑Net, raw + DS + PCA‑diff.
- `E3_raw_ds_pca_fusion.yaml`
  - PriorsFusionUNet, raw + DS + PCA‑diff (two‑branch fusion).

All configs share:

- `dataset.*` – OSCD splits, patch size, augmentations.
- `features.*` – which priors are enabled.
- `model.*` – model type and channels.
- `training.*` – epochs, batch size, optimizer, scheduler, loss.
- `phase1.change_maps_root` – where priors from Phase 1 live.

---

## 2. Training – Core U‑Net Experiments

### 2.1 E0 – U‑Net raw only

```bash
python -m phase2.train.train_oscd_seg \
  --config phase2/configs/oscd/core/E0_raw_unet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --output_dir phase2/outputs/oscd_seg_E0_raw
```

### 2.2 E1 – U‑Net raw + DS

```bash
python -m phase2.train.train_oscd_seg \
  --config phase2/configs/oscd/core/E1_raw_ds_unet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --output_dir phase2/outputs/oscd_seg_E1_raw_ds
```

### 2.3 E2 – U‑Net raw + PCA‑diff

```bash
python -m phase2.train.train_oscd_seg \
  --config phase2/configs/oscd/core/E2_raw_pca_unet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --output_dir phase2/outputs/oscd_seg_E2_raw_pca
```

### 2.4 E3 – U‑Net raw + DS + PCA‑diff

```bash
python -m phase2.train.train_oscd_seg \
  --config phase2/configs/oscd/core/E3_raw_ds_pca_unet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca
```

You can use `--max_epochs N` to do quick test runs without editing the
config.

Resume after a crash (continues from `last.ckpt` if present, else `best.ckpt`):

```bash
python -m phase2.train.train_oscd_seg \
  --config <same_config_as_before> \
  --oscd_root data/OSCD \
  --phase1_change_maps_root <same_change_maps_root_as_before> \
  --output_dir <same_output_dir_as_before> \
  --resume
```

If you want to start a fresh run in the same output directory, add
`--overwrite_output_dir` (this will overwrite `train_log.json` /
checkpoints).

---

## 3. Training – ResNet and PriorsFusionUNet

### 3.1 ResNet‑U‑Net raw only

Config: `phase2/configs/oscd/extended/E0_raw_resnet.yaml`

```bash
python -m phase2.train.train_oscd_seg \
  --config phase2/configs/oscd/extended/E0_raw_resnet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --output_dir phase2/outputs/oscd_seg_E0_raw_resnet
```

### 3.2 ResNet‑U‑Net raw + DS + PCA‑diff

Config: `phase2/configs/oscd/extended/E3_raw_ds_pca_resnet.yaml`

```bash
python -m phase2.train.train_oscd_seg \
  --config phase2/configs/oscd/extended/E3_raw_ds_pca_resnet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca_resnet
```

### 3.3 PriorsFusionUNet (raw + DS + PCA‑diff)

Config: `phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml`

```bash
python -m phase2.train.train_oscd_seg \
  --config phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion
```

---

## 4. Evaluation

Script: `phase2/eval/evaluate_oscd_seg.py`

Example: evaluate U‑Net raw‑only model (`E0`) and save metrics:

```bash
python -m phase2.eval.evaluate_oscd_seg \
  --config phase2/configs/oscd/core/E0_raw_unet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --checkpoint phase2/outputs/oscd_seg_E0_raw/best.ckpt \
  --output_dir phase2/outputs/oscd_seg_E0_raw/eval
```

Same pattern for other runs; just swap config, checkpoint, and output
dir:

- ResNet raw‑only:
  - `--config phase2/configs/oscd/extended/E0_raw_resnet.yaml`
  - `--checkpoint phase2/outputs/oscd_seg_E0_raw_resnet/best.ckpt`
  - `--output_dir phase2/outputs/oscd_seg_E0_raw_resnet/eval`
- PriorsFusionUNet:
  - `--config phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml`
  - `--checkpoint phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion/best.ckpt`
  - `--output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion/eval`

Outputs include:

- `oscd_seg_eval_results.json` – per‑city metrics.
- `oscd_seg_eval_summary.csv` – split‑level metrics table.

Use `phase2/eval/compare_priors_effect.py` to aggregate multiple
experiments into a single priors‑ablation CSV.

---

## 5. Visualization

### 5.1 Per‑city segmentation summaries

Script: `phase2/viz/viz_seg_predictions.py`

Example (ResNet raw‑only, test cities):

```bash
python -m phase2.viz.viz_seg_predictions \
  --config phase2/configs/oscd/extended/E0_raw_resnet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --checkpoint phase2/outputs/oscd_seg_E0_raw_resnet/best.ckpt \
  --output_dir phase2/outputs/oscd_seg_E0_raw_resnet/figs_seg \
  --cities test
```

This produces per‑city figures like
`phase2/outputs/oscd_seg_E0_raw_resnet/figs_seg/chongqing_seg_summary.png` showing:

- Pre RGB, Post RGB, GT overlay.
- Segmentation probability map and binary mask.

Same command works for other configs (swap config/checkpoint/output).

### 5.2 Combined DS/PCA + segmentation figures

Script: `phase2/viz/viz_oscd_combined.py`

ResNet raw‑only combined figures (test split):

```bash
python -m phase2.viz.viz_oscd_combined \
  --config phase2/configs/oscd/extended/E0_raw_resnet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --checkpoint phase2/outputs/oscd_seg_E0_raw_resnet/best.ckpt \
  --output_dir phase2/outputs/oscd_seg_E0_raw_resnet/figs_combined \
  --cities test
```

PriorsFusionUNet combined figures:

```bash
python -m phase2.viz.viz_oscd_combined \
  --config phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --checkpoint phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion/best.ckpt \
  --output_dir phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion/figs_combined \
  --cities test
```

Each per‑city PNG (e.g.
`phase2/outputs/oscd_seg_E0_raw_resnet/figs_combined/brasilia_combined_summary.png`) shows:

- Row 1: Pre RGB, Post RGB, GT overlay.
- Row 2: DS projection, PCA‑diff, segmentation probability.
- Row 3: DS mask (Otsu), segmentation mask (0.5 threshold).

These are the main interpretability tools for comparing DS/PCA priors to
segmentation behavior.

---

## 6. Future Tweaks (longer training, ImageNet)

High‑level ideas are described in detail in
`phase2/docs/phase2_report.md` (Section 7). Short summary:

- **Longer training + more seeds**:
  - Increase `training.epochs` in configs (e.g. 100–150).
  - Re‑run key experiments (E0/E3, ResNet baselines) with new
    `--output_dir` names (e.g. `..._long`, `..._seed1`).
- **ImageNet pretraining (ResNet)**:
  - In `E0_raw_resnet.yaml` /
    `E3_raw_ds_pca_resnet.yaml`, set:
    ```yaml
    model:
      type: unet2d_resnet
      num_classes: 1
      pretrained: true
    ```
  - Re‑run training with a possibly lower learning rate and a new
    `--output_dir` (e.g.
    `outputs/oscd_seg_E0_raw_resnet_imagenet`).
- **MultiSenGE pretraining / advanced priors usage**:
  - Left for a Phase‑2b/3 extension; see the report for the conceptual
    plan (pseudo‑label pretraining, loss weighting, mid‑layer fusion).

This run guide, together with `phase2_report.md`, should give you enough
structure to revisit and extend Phase 2 experiments at any later time
without re‑deriving all the commands.+
