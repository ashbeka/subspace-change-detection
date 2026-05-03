# GPU Run Pipeline (Phase 1 + Phase 2) - PowerShell

Run everything in **one PowerShell window**.

**What to copy/paste:** only the lines inside the ```powershell``` blocks.  
Each line is a complete command (no line continuations).

---

## 0) Start session (copy/paste)

```powershell
cd E:\research_projects\DS_damage_segmentation
.\.venv\Scripts\Activate.ps1
$py=".\.venv\Scripts\python.exe"
```

---

## 0.1) Install deps (first time only)

```powershell
& $py -m pip install -r phase1/requirements.txt
& $py -m pip install -r phase2/requirements.txt
& $py -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## 0.2) GPU sanity check

This must print `cuda True` and your GPU name:

```powershell
& $py -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## 1) Phase 1 (run once): generate OSCD priors (change maps)

Set paths (copy/paste once):

```powershell
$oscd="data/OSCD"
$phase1Out="phase1/outputs/oscd_saved_priors_fast"
$changeMaps="phase1/outputs/oscd_saved_priors_fast/oscd_change_maps"
```

Fast priors (recommended; used by Phase 2):

```powershell
& $py -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_priors_fast.yaml --oscd_root $oscd --output_dir $phase1Out --save_change_maps
```

Optional (slower): full classical baselines (adds `cva`, `celik`, `ir_mad`)

```powershell
$phase1OutFull="phase1/outputs/oscd_saved_full"
$changeMapsFull="phase1/outputs/oscd_saved_full/oscd_change_maps"
& $py -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_default.yaml --no_window --oscd_root $oscd --output_dir $phase1OutFull --save_change_maps
```

Optional (much slower): geodesic priors

```powershell
& $py -m phase1.eval.run_oscd_geodesic_priors --config phase1/configs/oscd_geodesic_priors.yaml --oscd_root $oscd --output_dir phase1/outputs/oscd_geodesic_priors
```

---

## 2) Phase 2 (GPU): choose epochs + output folder

Pick the epoch budget and create a fresh output folder for this run session:

```powershell
$epochs=150
$outRoot="phase2/outputs/runs_gpu_${epochs}ep_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
```

Optional smoke test (recommended before long runs):

```powershell
$run=Join-Path $outRoot "SMOKE_E0_raw_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --max_epochs 1 --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run --overwrite_output_dir
```

---

## 3) Phase 2 (GPU): run experiments one-by-one

For each experiment: run **Train**, then **Eval**.

### E0 (U-Net, raw only)

```powershell
$run=Join-Path $outRoot "E0_raw_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_baseline.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### E1 (U-Net, raw + DS projection prior)

```powershell
$run=Join-Path $outRoot "E1_raw_ds_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_E1_raw_ds.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### E1b (U-Net, raw + DS cross-residual prior) (extra DS baseline; optional)

```powershell
$run=Join-Path $outRoot "E1b_raw_ds_cross_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_E1b_raw_ds_cross.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_E1b_raw_ds_cross.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### E2 (U-Net, raw + PCA-diff prior)

```powershell
$run=Join-Path $outRoot "E2_raw_pca_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_E2_raw_pca.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_E2_raw_pca.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### E3 (U-Net, raw + DS + PCA-diff priors)

```powershell
$run=Join-Path $outRoot "E3_raw_ds_pca_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_priors.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_priors.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### E4 (U-Net, raw + pixel-diff prior) (baseline prior; optional)

```powershell
$run=Join-Path $outRoot "E4_raw_pixel_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_E4_raw_pixel.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_E4_raw_pixel.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### E5 (U-Net, raw + Celik prior) (classical baseline; requires full Phase 1 run)

```powershell
$run=Join-Path $outRoot "E5_raw_celik_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_E5_raw_celik.yaml --oscd_root $oscd --phase1_change_maps_root $changeMapsFull --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_E5_raw_celik.yaml --oscd_root $oscd --phase1_change_maps_root $changeMapsFull --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### E6 (U-Net, raw + IR-MAD prior) (classical baseline; requires full Phase 1 run)

```powershell
$run=Join-Path $outRoot "E6_raw_irmad_unet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_E6_raw_irmad.yaml --oscd_root $oscd --phase1_change_maps_root $changeMapsFull --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_E6_raw_irmad.yaml --oscd_root $oscd --phase1_change_maps_root $changeMapsFull --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### S0 (Siamese baseline, raw only) (deep baseline; optional)

```powershell
$run=Join-Path $outRoot "S0_siamese_raw_only"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_siamese.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_siamese.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

---

## 4) Recommended best-results models (run after E0-E3)

### ResNet-U-Net (raw only)

```powershell
$run=Join-Path $outRoot "E0_raw_resnet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_baseline_resnet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_baseline_resnet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### ResNet-U-Net (raw + DS + PCA-diff priors)

```powershell
$run=Join-Path $outRoot "E3_raw_ds_pca_resnet"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_priors_resnet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_priors_resnet.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

### PriorsFusionUNet (raw + DS + PCA-diff, fusion model)

```powershell
$run=Join-Path $outRoot "E3_raw_ds_pca_fusion"
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config phase2/configs/oscd_seg_priors_fusion.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir $run
& $py -m phase2.eval.evaluate_oscd_seg --device cuda --config phase2/configs/oscd_seg_priors_fusion.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "eval")
```

---

## 5) Resume / restart (after a crash)

Resume (continues from `last.ckpt` if present, else `best.ckpt`):

```powershell
& $py -m phase2.train.train_oscd_seg --device cuda --config <same_config_as_before> --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir <same_output_dir_as_before> --resume
```

If `last.ckpt` is corrupted (common after a hard reset), delete it and resume from `best.ckpt`:

```powershell
Remove-Item -Force <same_output_dir_as_before>\last.ckpt
& $py -m phase2.train.train_oscd_seg --device cuda --config <same_config_as_before> --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir <same_output_dir_as_before> --resume_ckpt <same_output_dir_as_before>\best.ckpt
```

Restart a run in an existing output directory:

```powershell
& $py -m phase2.train.train_oscd_seg --device cuda --epochs $epochs --config <config> --oscd_root $oscd --phase1_change_maps_root $changeMaps --output_dir <same_output_dir_as_before> --overwrite_output_dir
```

---

## 6) Compare experiments (single CSV)

```powershell
& $py -m phase2.eval.compare_priors_effect --summaries (Join-Path $outRoot "E0_raw_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E1_raw_ds_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E2_raw_pca_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E3_raw_ds_pca_unet\\eval\\oscd_seg_eval_results.json") --tags E0_raw E1_raw_ds E2_raw_pca E3_raw_ds_pca --output (Join-Path $outRoot "oscd_priors_ablation_summary.csv")
```

Extended comparison (includes extra baselines, if you ran them):

```powershell
& $py -m phase2.eval.compare_priors_effect --summaries (Join-Path $outRoot "E0_raw_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E1_raw_ds_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E1b_raw_ds_cross_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E4_raw_pixel_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E2_raw_pca_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E3_raw_ds_pca_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E5_raw_celik_unet\\eval\\oscd_seg_eval_results.json") (Join-Path $outRoot "E6_raw_irmad_unet\\eval\\oscd_seg_eval_results.json") --tags E0_raw E1_raw_ds E1b_raw_ds_cross E4_raw_pixel E2_raw_pca E3_raw_ds_pca E5_raw_celik E6_raw_irmad --output (Join-Path $outRoot "oscd_priors_ablation_summary_extended.csv")
```

If you also ran the Siamese baseline, add it to the command above:

```powershell
# Add: (Join-Path $outRoot "S0_siamese_raw_only\\eval\\oscd_seg_eval_results.json") and tag S0_siamese
```

---

## 7) Visualize predictions (optional)

Segmentation summary PNGs (test cities):

```powershell
$run=Join-Path $outRoot "E3_raw_ds_pca_unet"
& $py -m phase2.viz.viz_seg_predictions --device cuda --config phase2/configs/oscd_seg_priors.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "figs_seg") --cities test
```

Combined DS/PCA prior + segmentation figure:

```powershell
$run=Join-Path $outRoot "E3_raw_ds_pca_unet"
& $py -m phase2.viz.viz_oscd_combined --device cuda --config phase2/configs/oscd_seg_priors.yaml --oscd_root $oscd --phase1_change_maps_root $changeMaps --checkpoint (Join-Path $run "best.ckpt") --output_dir (Join-Path $run "figs_combined") --cities test
```

---

## 8) Optional: multi-seed sweep (hands-off)

```powershell
powershell -ExecutionPolicy Bypass -File phase2/scripts/run_phase2_sweep.ps1 -Preset core -Epochs 150 -Seeds "1234,1235,1236" -OutputTag "core_150ep"
```
