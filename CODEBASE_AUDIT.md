# DS_damage_segmentation — Codebase & Research Audit

Date: 2025-12-12

This document is a correctness-oriented audit of **Phase 1 (Difference-Subspace change detection)** and **Phase 2 (OSCD segmentation with DS/PCA-diff priors)**, cross-checked against the project’s documentation and the included reference papers/code under `references/`.

The intent is to answer:

1. Is the math/implementation consistent with the cited DS/subspace literature?
2. Are the engineering choices sound for remote sensing data?
3. Where are the correctness risks (bugs, evaluation leakage, methodological issues)?
4. What are the most valuable next fixes/improvements?

---

## Executive Summary (High Signal)

**Phase 1**
- The core DS implementation in `phase1/ds/pca_utils.py` and `phase1/ds/ds_scores.py` is consistent with the DS papers and the lab reference code.
- Both DS variants are implemented correctly:
  - Residual-stacked DS: `D = orth([R_ΨΦ, R_ΦΨ])`
  - Eigen-based DS: DS from eigenvectors of `P_Φ + P_Ψ` with eigenvalues in `(ε, 1−ε)`
- Phase 1 is methodologically solid; remaining items are mostly efficiency, robustness, and reproducibility improvements.

**Phase 2**
- The Phase 2 code correctly wires: Phase 1 OSCD loader → normalization → channel stacking (raw + priors) → U-Net/ResNet-U-Net → BCE+Dice loss.
- Validation/test are deterministic (random augmentations/noise are train-only).
- Evaluation is tile-level: overlapping patches are stitched into full-tile probability maps and metrics are computed once per city/tile.
- PR-AUC is supported; a Siamese baseline and `ds_cross_residual` prior option are available for ablations.

**Reference code**
- `references/reference_code/MagTool-main/.../magnitude.py` numpy-mode bug (`value.get()`) has been fixed in this repo.

---

## Implementation Status (Fixes Applied)

Key fixes implemented in-code (and used for the rerun results referenced below):

- Phase 1 valid masks use both pre/post (`phase1/data/oscd_dataset.py`) and DS window validity uses both (`phase1/ds/ds_scores.py`).
- Phase 1 IR-MAD subsampling is now seeded (`phase1/baselines/ir_mad.py`, wired in `phase1/eval/run_oscd_eval.py`).
- Phase 2 train-only augmentations/noise; val/test deterministic (`phase2/data/oscd_seg_dataset.py`, `phase2/data/transforms.py`).
- Phase 2 eval is stitched tile-level, including PR-AUC (`phase2/eval/evaluate_oscd_seg.py`, `phase2/eval/metrics_segmentation.py`).
- Phase 2 training checkpoints use stitched tile-level validation (`phase2/train/train_oscd_seg.py`).
- Siamese baseline added (`phase2/models/siamese_unet.py`); DS cross-residual prior supported (`phase2/data/oscd_seg_dataset.py`).

**Rerun outputs (quick CPU run, 5 epochs per model):**
- Phase 1 priors: `phase1/outputs/oscd_saved_priors_fast/oscd_change_maps`
- Phase 2 experiment outputs: `phase2/outputs/oscd_seg_*_v2`
- Ablation summary: `phase2/outputs/oscd_priors_ablation_summary_v2_with_eig.csv`

---

## Reference Materials Used

**Papers (local PDFs in `references/reference_papers/`)**
- Fukui & Maki (2015) — *Difference Subspace and Its Generalization for Subspace-Based Methods*
- Fukui et al. (2024) — *Second-order difference subspace*
- Kanai et al. (2023) — *Time-series Anomaly Detection based on Difference Subspace between Signal Subspaces*
- Ronneberger et al. — *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- He et al. — *Deep Residual Learning for Image Recognition*
- Daudt et al. — *Fully Convolutional Siamese Networks for Change Detection* (reference baseline concept; not fully implemented here)

**Reference code (under `references/reference_code/`)**
- `references/reference_code/DS/` (senpai DS/magnitude code; shape/motion domain)
- `references/reference_code/MagTool-main/` (DS + 2nd-order DS tooling)
- `references/reference_code/Subspace Toolbox/` (MATLAB-style subspace toolbox; canonical angles, SVD bases)

---

## Phase 1 — DS Change Detection (Correctness Audit)

### 1) What Phase 1 implements (pipeline)

Core pipeline (per OSCD tile):
1. Load `X_pre, X_post` as `(C,H,W)` and `Y_change` as `(1,H,W)` via `phase1/data/oscd_dataset.py`.
2. Build a `valid_mask` via `phase1/data/preprocessing.build_valid_mask`.
3. Apply bandwise z-score normalization using `phase1/data/oscd_band_stats.json`.
4. Compute DS scores and baselines.
5. Threshold scores (Otsu per tile and/or train-calibrated global threshold).
6. Evaluate AUROC/IoU/F1 and optionally save change maps for Phase 2 priors.

### 2) DS math and code mapping

**PCA/subspace basis**
- `phase1/ds/pca_utils.fit_pca_basis`
  - Input: `X ∈ ℝ^{d×n}` (columns are pixel vectors).
  - Uses sklearn PCA (centered) and returns `Φ ∈ ℝ^{d×r}` with orthonormal columns.
  - This matches the subspace-basis construction used in the toolbox (`cvtBasisVectorSVD`) and DS literature.

**Residual projectors**
- `phase1/ds/pca_utils.residual_projector(Φ)` computes `R_Φ = I − ΦΦᵀ`.

**Difference subspace variants**
- Residual-stacked (“Option A”, Phase 1 default):
  - `phase1/ds/pca_utils.difference_subspace(phi, psi)`
  - Implements `D = orth([R_ΨΦ, R_ΦΨ])`.
  - This is a valid geometric DS construction (difference pieces orthonormalized).
- Eigen-based (“Option B”, paper/toolbox style):
  - `phase1/ds/pca_utils.difference_subspace_eig(phi, psi)`
  - Computes `G = P_Φ + P_Ψ` and keeps eigenvectors with eigenvalues in `(ε, 1−ε)`.
  - This matches DS paper definitions (2015) and the refined definition in the 2024 second-order DS paper where eigenvalue ≈1 directions are excluded from both D and the principal component subspace (Karcher mean).

**Scores**
- `phase1/ds/ds_scores._compute_ds_matrix_scores`
  - Projection energy: `p = ||Dᵀ(x2 − x1)||²`.
  - Cross residual: `r_cross = ||R_Ψ x2||² + ||R_Φ x1||²`.

**Local DS**
- `phase1/ds/ds_scores.sliding_window_ds` recomputes DS on windows and aggregates (mean/max).

### 3) Baselines and evaluation code notes

Baselines (all reasonable/standard):
- Pixel diff / CVA: `phase1/baselines/pixel_diff.py`, `phase1/baselines/cva.py` (both L2 magnitude; naming differs).
- PCA-diff: `phase1/baselines/pca_diff.py` (PCA on `X2−X1`, keep PCs by variance threshold).
- Celik local PCA+kmeans: `phase1/baselines/celik_pca_kmeans.py` (implemented as a heavy but plausible baseline).
- IR-MAD: `phase1/baselines/ir_mad.py` (iterative reweighting; acceptable as a stretch baseline; note randomness).

Thresholding/metrics:
- Otsu: `phase1/eval/thresholding.py:otsu_threshold` (valid implementation).
- Train-calibrated global threshold grid search: `phase1/eval/thresholding.py:grid_search_threshold`.
- Binary metrics and AUROC: `phase1/eval/metrics.py`.

### 4) Phase 1 correctness risks / improvement notes (prioritized)

**High priority (robustness / research correctness)**
- **Fixed:** OSCD valid mask is now built from both pre and post images (`valid = valid_pre & valid_post`).

**Medium priority**
- **Fixed:** IR-MAD subsampling is now seeded via `np.random.default_rng(random_state)`.
- **Efficiency**: repeated PCA fits per tile/window are correct but slow; consider caching/cupy for large-scale runs (only if needed).

**Low priority**
- DS eigen selection thresholds: current defaults (`eps=1e-6`, `1-eps`) are reasonable; could be exposed/configurable.
- DS score normalization: percentile clip is practical but not part of DS theory; document as an engineering choice (already mostly done in Phase 1 report).

---

## Phase 2 — Segmentation with DS/PCA-diff Priors (Correctness Audit)

### 1) What Phase 2 implements (design intent)

Goal: supervised OSCD change segmentation, optionally using Phase 1 unsupervised maps as extra channels (“priors”).

Key elements:
- Dataset: `phase2/data/oscd_seg_dataset.py`
- Models: `phase2/models/unet2d.py`, `phase2/models/unet2d_resnet_backbone.py`, `phase2/models/priors_fusion_heads.py`
- Training: `phase2/train/train_oscd_seg.py` (BCE+Dice)
- Evaluation: `phase2/eval/evaluate_oscd_seg.py` (+ `phase2/eval/metrics_segmentation.py`)
- Visualization (full-tile stitching): `phase2/viz/viz_seg_predictions.py`, `phase2/viz/viz_oscd_combined.py`

### 2) Data pipeline: correctness and pitfalls

**Normalization**
- Phase 2 reuses Phase 1 z-score stats (`phase1/data/oscd_band_stats.json`) via `apply_normalization`.
- This is correct and desirable for consistency across phases.

**Feature construction**
- Raw channels:
  - default is `[pre(13), post(13)] → 26 channels`.
  - optional diff-only `[post−pre] → 13 channels`.
- Priors:
  - `_load_priors(...)` loads `phase1/outputs/.../{split}/{method}/{city}_score.npy`.
  - Priors are min-max normalized per tile to `[0,1]` and appended as extra channels.
  - This is a reasonable engineering choice; it is not “DS theory” but is consistent with using priors as network inputs.

**Patch extraction**
- Patch list is computed for each city with overlap (`patch_size`, `patch_overlap`).
- Training is patch-wise; this is standard for large remote-sensing tiles.

**Fixed: val/test augmentations**
- Random augmentations/noise are applied only when `split == "train"`; validation/test are deterministic.

**Fixed: augmentation controls**
- `flip` and `rotate90` are controlled separately.
- `RandomGaussianNoise` applies only to raw channels and only on valid pixels (not to priors / nodata).

### 3) Models: correctness and reference alignment

**UNet2D (`phase2/models/unet2d.py`)**
- Standard encoder/decoder + skip connections, BatchNorm, ReLU, transposed-conv upsampling.
- Output is logits; sigmoid is applied in loss/eval. This is correct for binary segmentation.
- Matches the U-Net design family (minor implementation differences are normal).

**ResNet encoder U-Net (`phase2/models/unet2d_resnet_backbone.py`)**
- Uses torchvision ResNet34 blocks (aligned with the ResNet paper’s architecture).
- Important note if using ImageNet pretraining with non-3-channel input:
  - conv1 weights are inflated from the pretrained RGB filters when `pretrained: true` and `in_channels != 3` (better than random init).
- Output is upsampled to GT size in training/eval; correct but should be noted (native output is H/2).

**PriorsFusionUNet (`phase2/models/priors_fusion_heads.py`)**
- Current design is a minimal early fusion: separate 1×1 conv on raw/prior groups then concatenate.
- It is correct code-wise, but is closer to “light feature reweighting” than a strong two-branch fusion architecture.

### 4) Training loop: correctness and issues

**Loss**
- `phase2/train/losses.py:BCEDiceLoss` computes BCE (valid pixels only) + soft Dice (valid pixels only).
- This is standard and correct.

**Validation inside training**
- `train/train_oscd_seg.py:evaluate` averages metrics over **patches**, not per tile/city.
  - This can overweight larger tiles or overlap patterns.
  - It’s acceptable for quick feedback, but the checkpoint monitor (“best val IoU”) may not match final per-city eval.

### 5) Evaluation methodology: key gap

**Current evaluation (`phase2/eval/evaluate_oscd_seg.py`)**
- Computes metrics per patch and averages them per city, then averages cities.
- Two issues:
  1. If val/test augmentations are enabled, evaluation is not meaningful (see critical bug above).
  2. Overlapping patches cause pixels to be evaluated multiple times; this differs from standard “one prediction per pixel per tile”.

**Visualization scripts show the “correct” inference style**
- `phase2/viz/viz_seg_predictions.py` and `phase2/viz/viz_oscd_combined.py` stitch patches into full-tile probability maps by averaging overlaps.
- Recommended: reuse this stitching approach in evaluation, then compute metrics once per city on the stitched map.

**Spec mismatch**
- The Phase 2 spec expects an `oscd_seg_eval_summary.csv`, but `phase2/eval/evaluate_oscd_seg.py` currently writes only JSON.
  - `compare_priors_effect.py` can read JSON, so this is not blocking, but it’s a mismatch.

### 6) Phase 2 improvement/repair checklist (prioritized)

**Critical (must fix for valid results)**
1. Disable random augmentations for `val` and `test` splits (`phase2/data/oscd_seg_dataset.py`).
2. Ensure evaluation is deterministic:
   - set random seeds (torch/numpy),
   - disable stochastic transforms,
   - consider turning off dropout (already in eval mode).

**High priority (methodological correctness)**
3. Evaluate on stitched full-tile predictions (like the viz scripts), not per overlapping patch.
4. Separate augmentation toggles:
   - respect `rotate90` separately from flips,
   - apply Gaussian noise only to raw spectral channels (not priors), and only on training.

**Medium priority (better baselines / research quality)**
5. If using pretrained ResNet with non-RGB input, improve conv1 adaptation:
   - replicate/average pretrained weights across channels, or
   - learn a 1×1 “spectral→RGB” adapter before feeding the pretrained backbone.
6. Add optional class-imbalance handling:
   - `pos_weight` in BCEWithLogitsLoss or focal loss,
   - sampling strategies (more change-heavy patches).
7. Make train-time “val IoU” match official eval style (per city on stitched maps).

**Low priority (engineering)**
8. Cache per-city loaded tiles and priors in `OSCDSegmentationDataset` to avoid disk reload per patch.
9. Add an evaluation CSV summary writer for easy ablation aggregation.

---

## Reference Code Audit Notes (not part of runnable pipeline)

### MagTool-main correctness + bug

`references/reference_code/MagTool-main/MagTool-main/magnitude.py`
- The DS-related subspace decomposition logic matches DS papers:
  - sum space / overlap / Karcher mean / difference subspace via eigenvalues of `P1+P2`.
  - magnitude via singular values of `basis1ᵀbasis2`.
- **Fixed:** `adjustEig(...)` now uses `float(value)` in numpy mode (and `value.get()` in cupy mode).

### DS reference code (shape/motion domain)

`references/reference_code/DS/utils.py`
- Eigen-based DS and magnitude computations are consistent with DS definitions.
- Second-order/geodesic decomposition helpers align conceptually with the second-order DS paper.

---

## Suggested Next Actions (practical)

If the immediate goal is “trustworthy Phase 2 results”:
1. Fix Phase 2 val/test augmentation.
2. Update evaluation to stitch full tiles (reuse viz logic) and compute per-city metrics once per tile.
3. Re-run the Phase 2 experiment suite (`phase2/scripts/run_phase2_core_experiments.ps1`) and regenerate ablation summaries.

If the goal is “stronger science”:
4. Add a Siamese baseline (from the referenced fully-convolutional Siamese change detection paper) as a Phase 2 comparator.
5. Explore where DS priors help/hurt by stratifying results by city, land cover, seasonality, cloud/shadow artifacts, etc.
