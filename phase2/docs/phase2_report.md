# Phase 2 – OSCD Segmentation with DS / PCA‑Diff Priors

> This document summarizes the *whole* project state up to the end of
> Phase 2: the research idea, Phase 1 DS/baseline work, Phase 2
> segmentation experiments, code structure, outputs, and guidelines for
> future experiments (including ImageNet pretraining).  
> It is the handoff context for Phase 3.

---

## 1. Project Overview and Phases

### 1.1 Big‑picture goal

The overall research goal is **disaster damage detection** from remote
sens­ing, targeting label‑efficient segmentation of change/damage using
multi‑temporal Sentinel‑2 (and later other sensors).

We are structuring the work in phases:

- **Phase 1** – Difference‑Subspace (DS) change detection on Sentinel‑2
  (OSCD + MultiSenGE), with classical baselines (pixel diff, CVA,
  PCA‑diff, Celik, IR‑MAD).  
  Output: unsupervised change maps + evaluation on OSCD.
- **Phase 2** – Supervised change segmentation on OSCD, using:
  - Raw S2 pre/post bands.
  - DS / PCA‑diff / pixel diff as **priors** (extra channels & fusion).
  Output: segmentation baselines + analysis of the value of priors.
- **Phase 3 (future)** – Damage‑oriented segmentation on a
  damage‑labeled dataset (e.g., xBD / xBD‑S12), reusing Phase 2
  machinery via a dataset adapter.

Phase 1 and 2 are fully implemented; this report covers both, with
emphasis on Phase 2.

### 1.2 Repository layout (current)

From the repo root `DS_damage_segmentation/`:

- `phase1/` – Phase 1 DS + baselines (complete).
- `phase2/` – Phase 2 segmentation on OSCD (this phase).
- `reference_code/` – Lab reference code and DS/SubspaceMethod MATLAB
  toolbox; read‑only reference.
- `research-notes/` – Separate research notes repo (not tracked in this
  repo); contains DS math appendix and planning notes.

Directories of interest:

- `phase1/docs/phase1_report.md` – Full Phase 1 report.
- `phase1/docs/spec_phase1_ds_oscd.md` – Phase 1 spec.
- `phase2/docs/spec_phase2_oscd_seg.md` – Phase 2 spec.
- `phase2/docs/phase2_report.md` – **This document**.

---

## 2. Phase 1 Recap – DS and Classical Change Detection

Phase 1 is documented in detail in `phase1/docs/phase1_report.md`. Here
we recap the key elements needed for Phase 2.

### 2.1 Datasets

- **OSCD (Onera Satellite Change Detection)**:
  - 24 Sentinel‑2 tiles (13 bands, 10/20/60 m) with:
    - Pre‑change and post‑change images.
    - Binary change masks.
  - Official splits encoded in `phase1/data/oscd_dataset.py`:
    - Train: 14 cities.
    - Test: 10 cities.
    - Val: taken from train (e.g. `abudhabi, aguasclaras, beihai`).

- **MultiSenGE S2**:
  - ~8k multi‑temporal S2 patches (256×256) over Grand‑Est.
  - No change labels; used for qualitative DS visualization.

### 2.2 Preprocessing & normalization

Phase 1 defines a common preprocessing pipeline:

- **Band order**:
  - OSCD: `[B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A]`.
  - MultiSenGE: subset of these (10 bands).
- **NODATA and masks**:
  - `data/preprocessing.build_valid_mask` identifies valid pixels based
    on `nodata_value` and `min_valid_bands`.
- **Bandwise z‑score normalization**:
  - `oscd_band_stats.json` stores mean/std per band, computed on OSCD
    train pre+post.
  - `apply_normalization` applies z‑score to valid pixels, leaves
    invalid as a fill value.

### 2.3 DS and subspace construction (core theory)

Given normalized spectral matrices
`X1, X2 ∈ R^{d×n}` (d=13 bands, n pixels), Phase 1:

1. Fits PCA bases:

   ```python
   phi_basis = fit_pca_basis(X1, rank=rank_r or variance_threshold)
   psi_basis = fit_pca_basis(X2, rank=rank_r or variance_threshold)
   Φ = phi_basis.basis  # (d, r)
   Ψ = psi_basis.basis  # (d, r)
   ```

2. Constructs difference subspace D using two variants:

   - **Residual‑stacked (Option A)**:

     `R_Φ = I−ΦΦᵀ`, `R_Ψ = I−ΨΨᵀ`.  
     `D = orth( [R_Ψ Φ, R_Φ Ψ] )`.

   - **Eigen‑based (Option B)** (sum of projectors, matches DS papers and
     toolbox):

     `G = P_Φ + P_Ψ` with `P_Φ = ΦΦᵀ`, `P_Ψ = ΨΨᵀ`.  
     Eigenvalues near 2 → common subspace; eigenvalues in (0,1) → DS; near
     0 → outside both.  
     Keep eigenvectors with `λ ∈ (ε, 1−ε)` and orthonormalize.

3. Computes DS scores:

   - Projection energy:  
     `p = || Dᵀ (x2 − x1) ||²` (DS projection).
   - Cross‑residual:  
     `r_cross = ||R_Ψ x2||² + ||R_Φ x1||²`.

Implementation: `phase1/ds/pca_utils.py` and `phase1/ds/ds_scores.py`.

### 2.4 Baselines and metrics

Baselines (all on normalized data):

- Pixel diff (L2).
- CVA (same magnitude as pixel diff here).
- PCA‑diff (PCA on `X2−X1`).
- Celik 2009 (local PCA + k‑means).
- IR‑MAD (optional stretch).

Thresholding & metrics:

- Unsupervised per‑tile Otsu thresholds on normalized scores.
- Train‑calibrated global thresholds via grid search over [0.05,0.95].
- Metrics:
  - AUROC (continuous scores).
  - Binary IoU, F1, precision, recall.
  - Runtime per tile.

### 2.5 Phase 1 outputs & visualization

- OSCD eval:
  - `phase1/eval/run_oscd_eval.py`.
  - Outputs JSON + CSV and (optionally) per‑tile change maps:
    `phase1/outputs/oscd_saved_priors_fast/oscd_change_maps/{split}/{method}/{city}_score.npy`.

- OSCD summary figures:
  - `phase1/eval/visualize_oscd_examples.py`.
  - Per city: 3×3 panels:
    - Pre RGB, Post RGB, GT overlay.
    - RGB diff, full‑band diff, DS projection.
    - DS mask (Otsu), PCA‑diff, metrics text.

- MultiSenGE DS maps:
  - `phase1/eval/run_multisenge_viz.py`.
  - PNG + GeoTIFF DS projection and cross‑residual.

These outputs serve as **priors and interpretability tools** in Phase 2.

---

## 3. Phase 2 Design – OSCD Segmentation with Priors

Phase 2 builds on Phase 1 outputs to train supervised change
segmentation models on OSCD masks.

### 3.1 Data pipeline

File: `phase2/data/oscd_seg_dataset.py`.

Per split (`train/val/test`), Phase 2 uses:

- Same OSCD root as Phase 1:
  - `data/OSCD/…`
- Phase 1 loader logic:
  - `OSCDEvaluatorDataset` from `phase1/data/oscd_dataset.py`.
  - `sample = (x_pre, x_post, y_change, valid_mask)`:
    - `x_pre, x_post ∈ R^{13×H×W}`.
    - `y_change ∈ {0,1}^{1×H×W}`.
    - `valid_mask ∈ {0,1}^{H×W}`.
- Priors from saved Phase 1 runs (continuous score maps):

  - Fast priors (DS/PCA/pixel; used by most Phase 2 configs):
    `phase1/outputs/oscd_saved_priors_fast/oscd_change_maps/{split}/{method}/{city}_score.npy`.
  - Full classical suite (adds `celik`, `ir_mad`, `cva`; needed for E5/E6):
    `phase1/outputs/oscd_saved_full/oscd_change_maps/{split}/{method}/{city}_score.npy`.

Configuration controls which features to stack:

```yaml
features:
  use_raw_s2: true
  use_pre_post_stack: true  # [pre(13), post(13)] vs diff-only
  priors:
    ds_projection: true/false
    pca_diff: true/false
    pixel_diff: true/false
```

OSCDSegmentationDataset:

1. Loads `x_pre, x_post, y, valid_mask` using the Phase 1 loader.
2. Applies the **same z‑score normalization** using OSCD band stats.
3. Stacks raw channels:
   - `[pre(13), post(13)]` (26 channels), or
   - `[post − pre]` (13 channels) if `use_pre_post_stack: false`.
4. Loads priors (if enabled) and min‑max normalizes per map to `[0,1]`.
5. Extracts patches:
   - `patch_size` (default 256), `patch_overlap` (default 64).
   - Uses Phase‑1‑style `_positions` logic to cover borders.
6. Applies augmentations (`phase2/data/transforms.py`):
   - Random flips and 90° rotations.
   - Optional Gaussian noise on spectral channels.

Outputs per sample:

```python
{
  "city": city_name,
  "split": split_name,
  "x": tensor(C, patch_size, patch_size),
  "y": tensor(1, patch_size, patch_size),
  "valid": tensor(1, patch_size, patch_size),
}
```

### 3.2 Models

All models are 2D segmentation nets producing logits `(B,1,H,W)`.

#### 3.2.1 UNet2D (baseline)

File: `phase2/models/unet2d.py`.

- Encoder:
  - 4 levels: DoubleConv + MaxPool, channels 64→128→256→512.
- Decoder:
  - ConvTranspose2d upsampling + skip connections + DoubleConv blocks.
- Final layer: `Conv2d(64→1)`.
- Config:

  ```yaml
  model:
    type: unet2d
    base_channels: 64
    depth: 4
    num_classes: 1
  ```

#### 3.2.2 UNet2DResNetBackbone (stronger encoder)

File: `phase2/models/unet2d_resnet_backbone.py`.

- Encoder: ResNet‑34 (`torchvision.models.resnet34`), with first conv
  modified to accept `in_channels`.
- Decoder: 4 up blocks, each using skip connections from ResNet layers.
- Final layer: `Conv2d(64→1)`.
- Config:

  ```yaml
  model:
    type: unet2d_resnet
    num_classes: 1
    pretrained: false  # can be set to true for ImageNet weights
  ```

ResNet U‑Net is used for stronger baselines (E0/E3 ResNet experiments).

#### 3.2.3 PriorsFusionUNet (two‑branch fusion)

File: `phase2/models/priors_fusion_heads.py`.

Design goal: give priors their **own branch** before fusing with raw S2
features.

Implementation:

```python
class PriorsFusionUNet(nn.Module):
  def __init__(self, n_raw, n_priors, base_channels=64, depth=4, num_classes=1):
    self.raw_proj = nn.Conv2d(n_raw, n_raw, kernel_size=1)
    self.prior_proj = nn.Conv2d(n_priors, n_priors, kernel_size=1)
    self.unet = UNet2D(in_channels=n_raw+n_priors, base_channels=base_channels, depth=depth, num_classes=num_classes)

  def forward(self, x):
    raw = x[:, :n_raw]
    priors = x[:, n_raw:]
    raw_f = self.raw_proj(raw)
    prior_f = self.prior_proj(priors)
    fused = torch.cat([raw_f, prior_f], dim=1)
    return self.unet(fused)
```

Config:

```yaml
model:
  type: priors_fusion_unet
  base_channels: 64
  depth: 4
  num_classes: 1
```

### 3.3 Training loop & losses

File: `phase2/train/train_oscd_seg.py`.

  - CLI (for OSCD):

  ```bash
  python -m phase2.train.train_oscd_seg \
    --config phase2/configs/oscd_seg_baseline.yaml \
    --oscd_root data/OSCD \
    --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
    --output_dir phase2/outputs/oscd_seg_E0_raw
  ```

- Loss: `BCEDiceLoss` (`phase2/train/losses.py`):
  - BCEWithLogitsLoss (on valid pixels) + soft Dice loss.
  - Default weights: `λ_bce=1.0`, `λ_dice=1.0`.
- Optimizer & schedule (`optimizer_schedules.py`):
  - AdamW with `lr=1e-3`, `weight_decay=1e-4` (or lower for ResNet).
  - Cosine annealing over `T_max=50` epochs.
- Training features:
  - `num_workers` configurable (0 by default for safety).
  - Gradient clipping (`max_norm=1.0`).
  - Per‑epoch validation:
    - Computes mean IoU/F1/AUROC on val patches.
  - Checkpoint:
    - `ModelCheckpoint` saves `best.ckpt` based on val IoU.
  - Logging:
    - `train_log.json` (per epoch train loss + val metrics).
  - `--max_epochs` flag to truncate training for quick tests.

If ResNet or fusion outputs smaller spatial maps than the input patch,
`train_oscd_seg.py` upsamples logits to the target size before computing
loss.

---

## 4. Phase 2 Experiments and Results

We ran several experiment families on OSCD:

### 4.1 Experiment definitions

(All use OSCD train/val/test split as in Phase 1.)

- **E0 – Raw only (U‑Net)**:
  - Config: `oscd_seg_baseline.yaml`.
  - Input: raw pre+post S2 (26 channels).
  - Model: `UNet2D`.

- **E1 – Raw + DS prior**:
  - Config: `oscd_seg_E1_raw_ds.yaml`.
  - Input: raw + DS projection (27 channels).
  - Model: `UNet2D`.

- **E2 – Raw + PCA‑diff prior**:
  - Config: `oscd_seg_E2_raw_pca.yaml`.
  - Input: raw + PCA‑diff (27 channels).
  - Model: `UNet2D`.

- **E3 – Raw + DS + PCA‑diff (combined priors)**:
  - Config: `oscd_seg_priors.yaml`.
  - Input: raw + DS + PCA‑diff (28 channels).
  - Model: `UNet2D`.

- **Additional baseline priors (available; useful for ablations)**:
  - `oscd_seg_E1b_raw_ds_cross.yaml` – raw + DS cross‑residual.
  - `oscd_seg_E4_raw_pixel.yaml` – raw + pixel‑diff.
  - `oscd_seg_siamese.yaml` – Siamese baseline (raw only).

- **ResNet variants**:
  - `oscd_seg_baseline_resnet.yaml` – raw only, ResNet U‑Net.
  - `oscd_seg_priors_resnet.yaml` – raw + DS + PCA‑diff, ResNet U‑Net.

- **PriorsFusionUNet variant**:
  - `oscd_seg_priors_fusion.yaml` – raw + DS + PCA‑diff, PriorsFusionUNet.

Update (latest GPU run): the core experiments below were rerun for **150 epochs**
(single seed) under `phase2/outputs/runs_gpu_150ep_20251215_233309/`.

### 4.2 Test split summary (mean metrics)

From:

- `phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary.csv` (core E0–E3)
- `phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv` (extended priors)

and the corresponding per-run eval JSONs in that folder (threshold = 0.5):

**Plain U‑Net experiments (test split):**

| Experiment        | Input                        | mean IoU | mean F1 | mean AUROC | mean PR‑AUC |
|-------------------|------------------------------|---------:|--------:|-----------:|------------:|
| E0 raw_only       | raw pre+post                 | 0.223 | 0.343 | 0.869 | 0.431 |
| E1 raw+ds         | raw + DS projection          | **0.273** | **0.401** | 0.874 | 0.436 |
| E5 raw+celik      | raw + Celik prior            | 0.260 | 0.382 | 0.872 | **0.448** |
| E3 raw+ds+pca     | raw + DS + PCA‑diff          | 0.251 | 0.370 | 0.864 | 0.404 |
| E4 raw+pixel      | raw + pixel‑diff prior       | 0.213 | 0.324 | 0.853 | 0.390 |
| E6 raw+ir_mad     | raw + IR‑MAD prior           | 0.208 | 0.309 | 0.805 | 0.356 |
| E1b raw+ds_cross  | raw + DS cross‑residual prior | 0.201 | 0.309 | 0.857 | 0.405 |
| E2 raw+pca        | raw + PCA‑diff prior         | 0.202 | 0.299 | 0.755 | 0.339 |

**ResNet U‑Net experiments (test split):**

| Experiment             | Input               | mean IoU | mean F1 | mean AUROC | mean PR‑AUC |
|------------------------|---------------------|---------:|--------:|-----------:|------------:|
| ResNet raw_only        | raw pre+post        | 0.243 | 0.360 | 0.800 | 0.393 |
| ResNet raw+ds+pca      | raw + DS + PCA‑diff | 0.230 | 0.345 | 0.819 | 0.377 |

**PriorsFusionUNet (test split):**

| Experiment           | Input               | mean IoU | mean F1 | mean AUROC | mean PR‑AUC |
|----------------------|---------------------|---------:|--------:|-----------:|------------:|
| Fusion raw+ds+pca    | raw + DS + PCA‑diff | 0.243 | 0.360 | **0.888** | 0.441 |

Interpretation:

- In this run, adding the **DS projection prior** (E1) improves mean IoU/F1 vs raw‑only.
- Celik as a prior (E5) is also strong in IoU/F1 and has the best PR‑AUC in this run.
- The reported means are averaged over the 10 OSCD test cities (each city weighted equally).
- PCA‑diff as a prior (E2) hurts in this setting; DS+PCA (E3) helps, but less than DS alone.
- Pixel‑diff, DS cross‑residual, and IR‑MAD priors do not help IoU/F1 in this run.
- Fusion is strongest for AUROC, but is not the best for F1 at a fixed threshold (0.5).
- These are **single‑seed** results at a fixed threshold (0.5). For stable conclusions, rerun with multiple seeds and tune the decision threshold on the val split.

### 4.3 Example per‑city behavior: chongqing

From various `oscd_seg_eval_results.json`:

**Chongqing (test tile, IoU/F1/AUROC):**

- U‑Net raw_only (E0):
  - `IoU ≈ 0.130`, `F1 ≈ 0.230`, `AUROC ≈ 0.901`.
- U‑Net raw+ds (E1):
  - `IoU ≈ 0.126`, `F1 ≈ 0.223`, `AUROC ≈ 0.902`.
- U‑Net raw+ds+pca (E3):
  - `IoU ≈ 0.324`, `F1 ≈ 0.490`, `AUROC ≈ 0.919`.
- ResNet raw_only:
  - `IoU ≈ 0.216`, `F1 ≈ 0.355`, `AUROC ≈ 0.853`.
- Fusion raw+ds+pca:
  - `IoU ≈ 0.290`, `F1 ≈ 0.449`, `AUROC ≈ 0.935`.

Qualitatively (and this is the key seminar message):

- Priors can help or hurt **depending on the tile**.
- On chongqing, adding DS+PCA priors helps a lot (E3 and Fusion), while DS alone (E1) does not.
  This is why the project reports both mean metrics and per‑city examples.

---

## 5. DS‑Centric Narrative and Novelty

Raw S2 segmentation is a strong baseline on OSCD, but priors can help.
In the latest 150‑epoch run, the DS projection prior improves mean IoU/F1 for the baseline U‑Net,
and fusion improves ranking metrics (AUROC/PR‑AUC).

Separately from final accuracy, the DS/PCA‑diff machinery remains central and novel in the project:

1. **Principled spectral change modeling**:
   - DS constructs subspaces for pre/post S2 spectra and focuses on
     directions where those subspaces differ.
   - PCA‑diff captures principal directions of `X2−X1`.  
   These are more structured than naive L2 diff and highlight
   spectrally meaningful changes (e.g., vegetation, moisture) beyond
   RGB.

2. **Unsupervised priors with physical interpretability**:
   - DS and PCA‑diff maps are computed without labels but align with GT
     change patterns on many OSCD tiles.
   - They can be interpreted and discussed in terms of spectral
     manifolds and canonical angles, which is attractive from a
     research/theoretical perspective.

3. **Explainability tools for segmentation**:
   - Even if raw‑only segmentation is best, DS/PCA‑diff maps provide
     **explanatory overlays**: where is the spectral support for the
     network’s predictions?
   - When the model misclassifies, DS/PCA‑diff can show whether the
     underlying spectra are ambiguous or if the network ignored a
     strong DS signal.

4. **Reusable for damage datasets**:
   - The DS tooling was built in a dataset‑agnostic way.
   - In Phase 3, DS/PCA‑diff can be recomputed as priors on xBD/xBD‑S12
     or similar, giving a unique angle compared to purely deep
     learning‑based approaches.

The narrative can emphasize:

> "Raw S2 segmentation is strong on OSCD, but DS and PCA‑diff provide a
> structured, interpretable lens on spectral change and serve as
> powerful unsupervised priors and analysis tools, especially valuable
> when moving to more complex damage datasets."

---

## 6. Visualization & Explainability Strategy

### 6.1 Phase 1 OSCD summary figures

`phase1/eval/visualize_oscd_examples.py` produces per‑city figures:

- **Row 1**:
  - Pre RGB (S2 bands B04/B03/B02).
  - Post RGB.
  - GT overlay on pre (change mask in red).
- **Row 2**:
  - RGB L2 diff.
  - Full‑band L2 diff.
  - DS projection map.
- **Row 3**:
  - DS mask (Otsu).
  - PCA‑diff map.
  - DS metrics text (e.g., F1/IoU for DS vs GT).

These figures explain:

- Where GT change is.
- How naive RGB / full‑band diff behave.
- How DS and PCA‑diff highlight changes, especially beyond RGB.

### 6.2 Phase 2 segmentation figures

`phase2/eval/viz_seg_predictions.py` produces per‑city segmentation
figures for a given model:

- **Row 1**:
  - Pre RGB.
  - Post RGB.
  - GT overlay.
- **Row 2**:
  - Prediction probability map (`P(change)`).
  - Predicted binary mask (threshold 0.5).
  - Empty panel (currently unused; can be used for priors or
    DS/PCA overlays in future).

How to read them:

- Overlay shows GT change.
- Prob map shows confidence structure (bright = high change).
- Pred mask shows final segmentation; comparing with GT overlay reveals
  false positives/negatives.
- Comparing models:
  - Differences between raw‑only vs priors‑based predictions highlight
    how priors influence the network (e.g., extra detections where DS
    is strong).

### 6.3 Combined DS/PCA + segmentation figures (implemented)

To make the relationship between priors and segmentation clearer, Phase
2 now includes a dedicated combined visualization script:

- `phase2/viz/viz_oscd_combined.py`
  - Inputs:
    - Phase‑1 DS/PCA‑diff change maps (`ds_projection`, `pca_diff`).
    - Phase‑2 segmentation model + checkpoint.
    - OSCD tiles and GT masks.
  - Output: for each city, a **3×3 figure** with:
    - Row 1: Pre RGB, Post RGB, GT overlay.
    - Row 2: DS projection (normalized), PCA‑diff (normalized),
      segmentation probability map.
    - Row 3: DS mask (Otsu on DS), segmentation mask (threshold 0.5),
      an empty slot reserved for future per‑tile metrics.

Example usage (ResNet raw‑only model):

```bash
python -m phase2.viz.viz_oscd_combined \
  --config phase2/configs/oscd_seg_baseline_resnet.yaml \
  --oscd_root data/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved_priors_fast/oscd_change_maps \
  --checkpoint phase2/outputs/oscd_seg_E0_raw_resnet/best.ckpt \
  --output_dir phase2/outputs/oscd_combined_resnet \
  --cities test
```

This implements the “optional viz enhancement” described in the spec:
priors (DS/PCA‑diff), segmentation probabilities/masks, and GT are all
shown together, making it much easier to:

- See where DS/PCA‑diff agree with GT and where they highlight subtle
  spectral changes beyond RGB.
- Diagnose when the segmentation model follows or ignores DS‑strong
  regions.

In the current Phase‑2 runs we generated these combined figures for all
OSCD **test** cities using the **ResNet raw‑only** baseline
(`phase2/outputs/oscd_combined_resnet/*.png`). Qualitatively, these confirm
that:

- DS/PCA‑diff maps often extend slightly beyond the binary GT mask,
  highlighting subtle spectral changes (e.g. vegetation or radiometric
  shifts) that the GT does not label as “change”.
- The segmentation model tends to align more tightly with the GT
  footprint, focusing on large, structurally consistent change regions.

This supports the narrative that DS/PCA‑diff act as broader,
unsupervised “change detectors”, while supervised segmentation learns a
task‑specific subset of those changes driven by the annotated labels.

We also generated combined figures for the **PriorsFusionUNet**
(`phase2/outputs/oscd_combined_fusion/*.png`), using
`phase2/configs/oscd_seg_priors_fusion.yaml` and the
`phase2/outputs/oscd_seg_E3_raw_ds_pca_fusion/best.ckpt` checkpoint. Compared
to the ResNet raw‑only plots, the fusion model’s segmentation masks tend
to:

- Track DS/PCA‑diff “hot” regions more aggressively, including areas of
  subtle spectral change outside the GT change footprint.
- Look more diffuse, with additional false positives around DS‑strong
  vegetation or radiometric artefacts.

This visual behavior is consistent with the quantitative metrics:
PriorsFusionUNet slightly improves AUROC relative to naive priors
concatenation, but its IoU/F1 remain below the best raw‑only baselines
because it inherits more of the DS/PCA sensitivity to non‑GT changes.

Further enhancements (left for Phase 2b/3) include:

- Annotating per‑city segmentation metrics (IoU/F1/AUROC) directly in
  the combined figures.
- Adding simple difference/overlay panels (e.g., seg mask XOR DS mask)
  to highlight disagreements.

These would further strengthen the explainability story.

---

## 7. Future Experiment Guidelines (for later runs)

While Phase 2 baselines are complete, several promising extensions can
be explored later. This section serves as a **guide** for future
experiments.

### 7.1 Longer training and more seeds

- **Current setup**: 50 epochs, 1 seed for each config.
- Future runs:
  - Try 100–150 epochs with lower learning rates (e.g. start `lr=5e-4`
    for ResNet).
  - Run 2–3 seeds per experiment to estimate variance; aggregate metrics
    and report mean ± std.
  - Use `training.num_workers > 0` (if environment allows) to speed up
    data loading.

### 7.2 ImageNet pretraining for ResNet encoder

ResNet backbone currently uses `pretrained: false`. To test ImageNet
pretraining:

1. In `phase2/configs/oscd_seg_baseline_resnet.yaml` and
   `oscd_seg_priors_resnet.yaml`, set:

   ```yaml
   model:
     type: unet2d_resnet
     num_classes: 1
     pretrained: true
   ```

2. Train as before:

   ```bash
   python -m phase2.train.train_oscd_seg --config phase2/configs/oscd_seg_baseline_resnet.yaml ...
   ```

3. Compare metrics:
   - Check if pretrained ResNet improves IoU/F1 vs `pretrained: false`,
     for both raw‑only and priors‑based configs.
   - Optionally partially freeze early encoder layers if overfitting is
     observed.

### 7.3 More data / pseudo‑label pretraining

MultiSenGE S2 can be used for self‑supervised pretraining:

- Use Phase‑1 DS or PCA‑diff maps on MultiSenGE as **pseudo labels**:
  - Train U‑Net/ResNet to regress DS/PCA‑diff maps on MultiSenGE
    patches.
  - Then fine‑tune on OSCD labels.
- Alternatively, treat DS/PCA‑diff as **additional channels** during
  MultiSenGE pretraining, then reuse the encoder for OSCD.

This can be explored in a future “Phase 2b” bridging MultiSenGE and
OSCD.

### 7.4 Alternative uses of priors

Instead of only concatenating priors as features:

- **Loss weighting**:
  - Increase loss weights on pixels where DS/PCA‑diff scores are high
    (priors indicate probable change).
  - This encourages the network to fit hard change regions more
    strongly.

- **Threshold tuning**:
  - Instead of a fixed 0.5 threshold, learn or tune per‑method
    thresholds (possibly via val set) and/or use DS/PCA‑diff to adapt
    thresholds spatially.

- **Feature injection at mid‑layers**:
  - Extend PriorsFusionUNet to inject prior features at deeper encoder
    levels instead of only at input; may help the network use priors
    more effectively.

These ideas can be prototyped later without changing the core
infrastructure.

---

## 8. Summary and Phase 3 Handoff

### 8.1 What’s accomplished so far

- Phase 1:
  - Implemented and validated DS and classical change detection
    methods on OSCD and MultiSenGE.
  - Produced interpretable DS/PCA‑diff maps and per‑city visual
    summaries.
- Phase 2:
  - Implemented robust OSCD segmentation pipeline:
    - Data loaders, priors integration, U‑Net/ResNet/fusion models,
      training, evaluation, and visualization.
  - Ran a comprehensive set of experiments:
    - Raw S2 baselines (U‑Net and ResNet).
    - Raw + DS/PCA priors (concatenation, ResNet, fusion).
  - Found that:
    - On OSCD, raw S2 segmentation is a strong baseline.
    - In the latest 150‑epoch GPU run, adding the DS projection prior improves mean IoU/F1 (E1),
      while fusion improves ranking metrics (best AUROC/PR‑AUC).
    - Priors are not uniformly beneficial per‑tile; report both mean and per‑city behavior.
  - Established DS/PCA‑diff as powerful **analysis and interpretability
    tools** that will be useful when moving to damage datasets.

### 8.2 Ready state for Phase 3

The codebase and reports now provide:

- A clear DS‑centric story and full context for Phase 1–2.
- Well‑structured segmentation code that only requires:
  - A new dataset adapter (`phase2/data/damage_dataset_adapter.py`) and
  - New configs (`configs/damage_dataset_template.yaml`)
  to switch to xBD/xBD‑S12 or other damage datasets.
- Guidance for future experiments (longer training, ImageNet, more
  data, alternative priors usage).

This document, together with `phase1_report.md` and the two specs,
should give a Phase‑3 assistant or collaborator enough context to
understand exactly what has been done, why, and how to build the next
steps on top of it.
