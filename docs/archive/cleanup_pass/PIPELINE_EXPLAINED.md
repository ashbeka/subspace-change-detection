# Pipeline Explained

This document explains the project from raw data to final metrics.

## One-Sentence Pipeline

Phase 1 turns pre/post Sentinel-2 image pairs into unsupervised change maps; Phase 2 trains neural segmentation models that use either raw pre/post Sentinel-2 bands alone or raw bands plus those Phase 1 change maps as extra input channels.

## Data

The main dataset is OSCD:

```text
data/OSCD/
  onera_satellite_change_detection dataset__images/
  onera_satellite_change_detection dataset__train_labels/
  onera_satellite_change_detection dataset__test_labels/
```

Each city has:

- pre-event Sentinel-2 image bands
- post-event Sentinel-2 image bands
- binary change mask

The current Phase 2 task is binary semantic segmentation:

```text
pixel = changed or unchanged
```

It is not building damage classification.

## Phase 1: Prior Generation

Phase 1 computes unsupervised change scores from pre/post imagery.

Main command family:

```powershell
python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_priors_fast.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_saved_priors_fast --save_change_maps
```

Important output:

```text
phase1/outputs/oscd_saved_priors_fast/oscd_change_maps/<split>/<method>/<city>_score.npy
```

Example:

```text
phase1/outputs/oscd_saved_priors_fast/oscd_change_maps/train/ds_projection/beirut_score.npy
```

That file is a prior map: a numeric image where high values mean the unsupervised method thinks there may be change.

## Why `oscd_saved_priors_fast`?

`oscd_saved_priors_fast` is not "fast training." It is a saved Phase 1 prior-map folder.

The name means:

- `oscd`: dataset
- `saved_priors`: change maps were saved for later Phase 2 use
- `fast`: generated with a faster Phase 1 config

It is used in Phase 2 because the segmentation dataset can load those saved `.npy` maps as extra channels.

Why not always use `oscd_saved_full`?

- `oscd_saved_priors_fast` contains the core priors needed by the current configs, especially DS and PCA.
- `oscd_saved_full` includes slower/full classical baselines such as Celik and IR-MAD.
- Full is useful for broader comparisons, but it is not required for every core raw/DS/PCA experiment.

Better future naming:

```text
phase1/outputs/priors_oscd_core/
phase1/outputs/priors_oscd_classical_full/
```

The current names are legacy but understandable once mapped.

## Phase 2: Dataset Construction

The key file is:

```text
phase2/data/oscd_seg_dataset.py
```

For each patch, it creates:

```text
x = input tensor
y = binary change mask
valid = valid-pixel mask
```

Raw-only E0 input:

```text
13 pre bands + 13 post bands = 26 channels
```

Raw+DS E1 input:

```text
26 raw channels + 1 ds_projection prior = 27 channels
```

Raw+DS+PCA E3 input:

```text
26 raw channels + 1 ds_projection prior + 1 pca_diff prior = 28 channels
```

The prior is not a label. It is just another input feature.

## Phase 2: Model Training

The key file is:

```text
phase2/train/train_oscd_seg.py
```

It does this:

1. Read a YAML config.
2. Build the OSCD patch dataset.
3. Infer input channel count.
4. Build the model.
5. Train with BCE + Dice loss.
6. Validate by stitching patch predictions back to full city maps.
7. Save `best.ckpt`, `last.ckpt`, `train_log.json`, and `run_metadata.json`.

Most core configs use:

```text
model: UNet2D
base_channels: 64
depth: 4
batch_size: 8
epochs: 150 in the current sweep
```

The Siamese baseline uses:

```text
model: SiameseUNet2D
```

## Phase 2: Evaluation

The key file is:

```text
phase2/eval/evaluate_oscd_seg.py
```

It does this:

1. Load a checkpoint.
2. Run patch inference for each val/test city.
3. Stitch patch probabilities back into full-size city maps.
4. Threshold probabilities at `0.5`.
5. Compute IoU, F1, AUROC, and PR-AUC.
6. Write JSON and CSV result files.

Important outputs:

```text
eval/oscd_seg_eval_results.json
eval/oscd_seg_eval_summary.csv
```

## Current Core Sweep

Current completed run:

```text
phase2/outputs/sweep_core_150ep_repro_v5_20260503_052422
```

It trained:

```text
6 configs x 3 seeds = 18 models
```

The core comparison was:

```text
E0_raw_unet vs E1_raw_ds
```

Result:

```text
E1_raw_ds did not beat E0_raw_unet in this sweep.
```

## What To Read In Code

Read in this order:

1. `phase1/eval/run_oscd_eval.py`
2. `phase1/ds/ds_scores.py`
3. `phase2/data/oscd_seg_dataset.py`
4. `phase2/models/unet2d.py`
5. `phase2/train/train_oscd_seg.py`
6. `phase2/eval/evaluate_oscd_seg.py`

That path explains the full implemented pipeline.
