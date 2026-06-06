# Subspace Method Notes

This file is the current working note for the subspace correctness issue raised by Sensei and senpais. It is intentionally skeptical: it separates what the code actually does from what the papers and old project notes might imply.

## 1. Short Answer

The active OSCD pipeline represents each valid Sentinel-2 pixel as one 13-dimensional spectral vector. For one city/tile, the code forms:

```text
X_pre  in R^(13 x N)
X_post in R^(13 x N)
```

where `N` is the number of valid pixels. PCA is fit separately to `X_pre` and `X_post`, producing two rank-`r` subspaces in 13-dimensional spectral space.

The important correction is this: the old default `subspace_variant: residual` is now treated as `legacy_residual_stack`. It is preserved for reproducibility, but it should not be used as the main paper-faithful DS implementation. With `rank_r=6` in 13-D OSCD data, the legacy residual-stack basis can become `(13,12)` and behave almost like raw spectral L2 difference.

The corrected first-order DS path is now `subspace_variant: canonical`, which builds DS from principal vectors/canonical angles. `subspace_variant: eig` is also repaired so that equal/shared subspaces return an empty DS basis instead of falling back to the legacy residual-stack basis.

## 2. Current Code Mapping

- `phase1/data/preprocessing.py`
  - `vectorize_cube` maps `(C,H,W)` to `(C,N)`.
  - For OSCD, `C=13`, so columns are pixel spectra.
  - Pixel positions are not used to fit the global subspace, but row/column indices are preserved to put score maps back on the image grid.

- `phase1/ds/pca_utils.py`
  - `fit_pca_basis`: PCA basis, shape `(d,r)`.
  - `legacy_residual_stack_difference_subspace`: old residual-stack DS.
  - `difference_subspace_eig`: projector-eigen DS, repaired to return empty basis when no DS directions exist.
  - `difference_subspace_canonical`: paper-faithful first-order DS from principal vectors.
  - `build_difference_subspace`: dispatches `legacy_residual_stack`, `eig`, or `canonical`.

- `phase1/ds/ds_scores.py`
  - Computes per-pixel projection energy:

```text
score(pixel) = || D^T (x_post - x_pre) ||^2
```

  - Also computes cross-residual:

```text
||R_post x_post||^2 + ||R_pre x_pre||^2
```

## 3. What Was Wrong Or Unsafe

The old default DS construction was:

```text
D = orth([R_post Phi, R_pre Psi])
```

where `Phi` and `Psi` are pre/post PCA bases. For equal-rank subspaces, this can produce up to `2r` directions. In OSCD with `rank_r=6`, this means a `(13,12)` basis. Since the ambient space has only 13 spectral dimensions, this can cover almost the whole spectral space.

Observed audit result on `beirut` before repair:

```text
legacy residual-stack D shape: (13,12)
corr(legacy DS score, raw spectral L2): approximately 0.99999
eig/canonical D shape: (13,6)
corr(eig/canonical DS score, raw spectral L2): approximately 0.19
```

That means old `oscd_saved_priors_fast` should be treated as a legacy residual-stack prior, not clean evidence that paper-faithful DS improved segmentation.

## 4. What Changed

Implemented code-level changes:

- Preserved the old residual-stack method as `legacy_residual_stack`.
- Kept `residual` as a backwards-compatible alias for `legacy_residual_stack`.
- Added canonical DS using principal vectors:

```text
Phi^T Psi = U Sigma V^T
D = (Phi U - Psi V) {2(I - Sigma)}^(-1/2)
```

- Fixed projector-eigen DS so it no longer falls back to residual-stack when no eig directions exist.
- Added `phase1/configs/oscd_priors_canonical.yaml`.
- Added `phase1/scripts/audit_oscd_subspace.py`.
- Added `phase1/subspace/kernel_difference_subspace.py`.
- Reworked `phase1/scripts/venus_kds_demo.py` from a shared-KPCA prototype into a paper-formula KDS/KGDS projection audit for the Venus data.
- Added unit tests under `tests/`.

## 5. Venus Dataset Status

Sensei's Venus files are dataset artifacts under `data/venus_tpami2015/`:

```text
venus_tpami2015_no_accessories.mat        -> venus_nothing  shape (480,640,1,300)
venus_tpami2015_earrings.mat              -> venus_er2      shape (480,640,1,300)
venus_tpami2015_earrings_necklace.mat     -> venus_er_neck  shape (480,640,1,300)
```

These are not like OSCD. The natural TPAMI-style representation is:

```text
one whole downsampled grayscale view = one vector
300 views = 300 samples
```

The demo script downsamples each view to `63 x 48`, so each sample vector is length `3024`, and each sculpture set becomes:

```text
X in R^(3024 x 300)
```

This is the representation Sensei is likely pointing toward when he asks about the TPAMI 2015 nonlinear difference subspace experiment.

## 6. KDS/KGDS Status

The active OSCD pipeline does not implement nonlinear KDS/KGDS from TPAMI 2015.

The Venus audit path now implements the paper-formula KDS/KGDS projection machinery outside OSCD:

1. Load the Venus whole-image vectors.
2. Build linear PCA subspaces and a linear canonical DS visualization in image space.
3. Fit nonlinear kernel subspace bases as `e_i = sum_l a_li phi(x_l)`.
4. Build the KDS/KGDS matrix `E^T E` from kernel inner products between nonlinear basis vectors.
5. Select the smallest positive eigen-directions and normalize them as RKHS basis vectors.
6. Project whole-image views with TPAMI Eq. 16/17 using only kernel evaluations.

This is paper-faithful for the kernel coefficient/projection equations. It is still not a full reproduction of the TPAMI visualization figures, because preimage reconstruction/search is not implemented.

## 7. Answer To Sensei

Use this as the concise current answer:

> Currently I generate OSCD subspaces by treating each valid Sentinel-2 pixel as a 13-D spectral vector and fitting PCA separately to pre/post matrices. I found that our old residual-stack DS implementation was not faithful enough: with rank 6 it produced a 12-D projection and behaved almost like raw spectral difference. I separated it as a legacy variant and added canonical/projector-eigen DS for the paper-faithful linear version. For the Venus data, I now implemented the TPAMI-style kernel coefficient/projection equations for KDS and KGDS using whole image views as samples; the remaining missing part is preimage reconstruction for visualization.

## 8. Commands

Run unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Audit one OSCD city:

```powershell
.\.venv\Scripts\python.exe phase1/scripts/audit_oscd_subspace.py --city beirut --rank 6
```

Run the Venus KDS/KGDS audit:

```powershell
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe phase1/scripts/venus_kds_demo.py --venus_root data/venus_tpami2015 --output_dir "phase1/outputs/venus_kds_audit_$tag"
```

Generate corrected canonical DS OSCD priors:

```powershell
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe phase1/eval/run_oscd_eval.py `
  --config phase1/configs/oscd_priors_canonical.yaml `
  --oscd_root data/OSCD `
  --output_dir "phase1/outputs/oscd_priors_canonical_$tag" `
  --save_change_maps
```

## 9. Sources

- Fukui and Maki, "Difference Subspace and Its Generalization for Subspace-Based Methods", TPAMI 2015: https://pubmed.ncbi.nlm.nih.gov/26440259/
- Author PDF: https://www.cs.tsukuba.ac.jp/~kfukui/papers/TPAMI2408358-revised.pdf
- S3CCA, "Smoothly Structured Sparse CCA For Partial Pattern Matching": https://staff.aist.go.jp/takumi.kobayashi/publication/2014/ICPR2014.pdf
- Temporally regularized CCA/KOTRCCA overview: https://cir.nii.ac.jp/crid/1360865817576852480
- SubspaceMethodsToolBox: https://github.com/ComputerVisionLaboratory/SubspaceMethodsToolBox
- SubspacesToolkit: https://github.com/ComputerVisionLaboratory/SubspacesToolkit
- subspyces: https://github.com/ma-ath/subspyces

## 10. Research Consequences

Do not claim old Phase 2 priors prove that paper-faithful DS improves OSCD segmentation. The old E1/E3 experiments mostly used `oscd_saved_priors_fast`, which came from the legacy residual-stack path.

The next defensible experiment is:

1. Generate canonical DS priors with `phase1/configs/oscd_priors_canonical.yaml`.
2. Create Phase 2 configs that explicitly point to the canonical prior output.
3. Run a small E0 raw vs canonical-DS prior comparison.
4. Compare against pixel L2 and PCA-diff, because canonical DS may be weaker on OSCD but more faithful to the subspace paper.

This is a better research story than pretending the old residual-stack behavior was correct DS.

## 11. What Was Missing From The First Pass

The first implementation pass fixed the code path and produced smoke evidence, but it did not fully answer the broader research request. The missing pieces were:

- A clear DS / GDS / KDS / KGDS explanation.
- A crosswalk between this repo, Sensei's TPAMI 2015 paper, the bundled reference code, and the three GitHub repos.
- A better explanation of why Sensei mentioned KPCA and CCA.
- A stronger statement of what we can and cannot tell Sensei now.
- A reading pathway for understanding this part of the thesis.

Those are filled in below.

## 12. DS, GDS, KDS, KGDS In Plain Terms

### DS: Difference Subspace

DS is for two subspaces.

If object/class/time condition A gives a subspace `Phi`, and object/class/time condition B gives a subspace `Psi`, DS is the subspace containing the directions that separate them. In the TPAMI paper, the definition is tied to canonical angles/principal vectors between the two subspaces.

For this project's OSCD adaptation:

```text
pre image pixels  -> PCA -> Phi in R^(13 x r)
post image pixels -> PCA -> Psi in R^(13 x r)
DS(Phi, Psi)      -> D in R^(13 x k)
pixel score       -> ||D^T (x_post - x_pre)||^2
```

The adaptation is reasonable as an experiment, but it is not the original TPAMI image-set setting. In TPAMI, one whole image view is usually one vector; in OSCD, one pixel spectrum is one vector.

### GDS: Generalized Difference Subspace

GDS is for more than two class subspaces. Instead of extracting difference directions between exactly two subspaces, it builds a constraint/difference space from several class subspaces so subspace methods such as SM/MSM can compare projected subspaces more discriminatively.

Important: the current OSCD binary pre/post implementation is not really GDS. Calling the current OSCD method "GDS" would be unsafe unless we implement a multi-subspace formulation and show how the multiple subspaces are defined.

### KDS: Kernel Difference Subspace

KDS is the nonlinear version of DS. The paper maps samples into a high-dimensional feature space through a kernel, then constructs subspaces there using KPCA. This is why Sensei asked about the nonlinear difference subspace and the kernel trick.

The active OSCD pipeline does not implement KDS. The Venus audit now implements KDS projection in the paper's coefficient form, but only for the Venus whole-image dataset.

### KGDS: Kernel Generalized Difference Subspace

KGDS is the nonlinear version of GDS. It uses kernel subspaces and a generalized difference/constraint space for multiple classes. This is the likely end target if Sensei wants the TPAMI2015 nonlinear method reproduced faithfully.

The Venus audit now implements KGDS projection for the three Venus sculpture sets. It still does not implement preimage reconstruction.

## 13. Exact Reference-Code Crosswalk

### This repo: active implementation

- `phase1/data/preprocessing.py`
  - `vectorize_cube`: OSCD `(C,H,W)` to `(C,N)`.
  - This answers Sensei's "how do you generate a subspace from multi-channel images?" question.

- `phase1/ds/pca_utils.py`
  - `fit_pca_basis`: PCA basis generation.
  - `legacy_residual_stack_difference_subspace`: old unsafe default, preserved only for reproducibility.
  - `difference_subspace_eig`: projector-eigen DS.
  - `difference_subspace_canonical`: canonical/principal-vector DS.

- `phase1/ds/ds_scores.py`
  - Converts DS basis into per-pixel score maps.

- `phase1/scripts/audit_oscd_subspace.py`
  - The script to prove the above behavior from code and data.

- `phase1/scripts/venus_kds_demo.py`
  - Venus linear DS plus paper-formula KDS/KGDS projection audit.

- `phase1/subspace/kernel_difference_subspace.py`
  - RKHS KDS/KGDS coefficient bookkeeping:
    - RBF kernel matrix.
    - kernel subspace basis coefficients.
    - `E^T E` between nonlinear subspace bases.
    - smallest-positive-eigenvalue KDS/KGDS basis.
    - Eq. 16/17 projection coordinates.

### Bundled reference code in this repo

- `references/reference_code/DS/utils.py`
  - `gen_shape_subspace`: builds a subspace from 3D motion/shape data.
  - `gen_shape_difference_subspace`: projector-eigen style DS.
  - This is close to our repaired `eig` path, not to the old residual-stack default.

- `references/reference_code/MagTool-main/MagTool-main/magnitude.py`
  - `calcDiffSubspace`: uses projector sums and selects the difference eigen-space.
  - `calcKarcherSubspace`: related to mean/Karcher subspace.
  - `calcMagnitude`: subspace magnitude from canonical cosines.
  - This supports the conclusion that the old `(13,12)` residual-stack behavior should not be treated as the clean DS baseline.

- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtPCA.m`
  - MATLAB PCA on `X` where columns are samples.

- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtKernelPCA.m`
  - Kernel PCA support.

- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtCCA.m`
  - CCA support.

- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtKernelCCA.m`
  - Kernel CCA support.

- `references/reference_code/Subspace Toolbox/cvtToolBox/SubspaceMethod/cvtBasisVector.m`
  - Builds subspace basis vectors from data arrays.

### External GitHub repos

- `https://github.com/ComputerVisionLaboratory/SubspaceMethodsToolBox`
  - Key files:
    - `src/utils/cvlPCA.m`
    - `src/utils/cvlKPCA.m`
    - `src/functions/cvlBasisVector.m`
    - `src/functions/cvlCanonicalAngles.m`
    - `src/functions/cvlKernelBasisVector.m`
    - `src/functions/cvlKernelCanonicalAngles.m`
  - This is the closest modern MATLAB reference for PCA/KPCA/subspace basics.

- `https://github.com/ComputerVisionLaboratory/SubspacesToolkit`
  - Key files:
    - `src/functions/computePCA.m`
    - `src/functions/computeKernelPCA.m`
    - `src/functions/computeBasisVectors.m`
    - `src/functions/computeSubspacesSimilarities.m`
    - `src/functions/computeKernelSubspacesSimilarities.m`
  - This appears to be a cleaner newer educational toolkit.

- `https://github.com/ma-ath/subspyces`
  - Key files:
    - `subspyces/transform/pca_transform.py`
    - `subspyces/metrics/cosine_canonical_angles.py`
    - `subspyces/core/vector_space.py`
  - Useful for Python subspace concepts, but it is not a DS/KDS implementation.

## 14. Why Sensei Mentioned KPCA

The TPAMI paper explicitly extends DS/GDS to KDS/KGDS using nonlinear kernel mapping and KPCA. It does this because image sets from multi-view objects can have nonlinear structure: views rotating around a 3D object do not necessarily lie well in one simple linear subspace.

The Venus files are exactly this kind of multi-view object data:

```text
venus_tpami2015_no_accessories:         300 views
venus_tpami2015_earrings:               300 views
venus_tpami2015_earrings_necklace:      300 views
each raw view:          480 x 640 grayscale
demo downsampled view:  63 x 48 = 3024-dimensional vector
matrix per object:      R^(3024 x 300)
```

That is why KPCA matters more for the Venus reproduction than for the current OSCD pixel-spectral DS implementation.

For OSCD, a kernel method is possible, but it is not straightforward:

- Full-image pixel KPCA would require kernels over hundreds of thousands to millions of pixel samples, which is computationally expensive.
- Sampling or local windows would be needed.
- A kernel-space score map needs a clearly defined out-of-sample projection for each pixel.
- We should not claim KDS/KGDS on OSCD until this is mathematically and computationally specified.

This is now a major research task, not a side idea. A defensible next version should test at least one computationally feasible OSCD KDS design, for example:

- sampled global KDS: fit KPCA/KDS on a controlled sample of valid pre/post pixel vectors, then project all pixels out-of-sample;
- local/windowed KDS: fit small KDS models per patch or per local region;
- prototype/Nystrom KDS: approximate the kernel matrix with representative pixel prototypes;
- patch-vector KDS: use small spatial patches as samples instead of individual pixels.

The key question is whether nonlinear KDS captures useful pre/post differences that linear canonical DS and simple pixel/PCA differences miss.

## 15. Why Sensei Mentioned CCA

CCA is the mathematical family behind canonical correlations/angles. Subspace similarity is often measured through canonical angles, and CCA-like methods compute directions that maximize correlation between two sets.

The S3CCA paper is not DS, but it is relevant because it treats two matrices:

```text
X in R^(d x m)
Y in R^(d x n)
```

where columns are local feature vectors and the column index can have spatial/temporal meaning. It learns map weights over the array dimension, with smoothness and structured sparsity. This directly connects to Sensei/senpai questions about whether we are losing pixel position information.

Current OSCD global DS loses spatial position during PCA fitting, then restores only the score map positions after scoring. S3CCA-style thinking would preserve/regularize spatial structure in the matching itself. That is a possible future method, but it is not currently implemented.

The temporally regularized CCA / KOTRCCA paper is relevant for temporal sequences, not directly for two-date OSCD. It matters more if the project returns to MultiSenGE multi-date temporal subspace modeling.

## 16. MultiSenGE GDS/KGDS Research Path

MultiSenGE may be a more natural dataset for GDS/KGDS than OSCD because it can provide more than two temporal observations of the same area. The rough mapping is:

```text
OSCD binary change:
pre date  -> subspace A
post date -> subspace B
DS/KDS compares two subspaces

MultiSenGE temporal change:
date 1 -> subspace A
date 2 -> subspace B
date 3 -> subspace C
...
GDS/KGDS compares multiple subspaces
```

This could become a profitable research path:

1. Build one subspace per date/patch/tile from Sentinel-2 band vectors.
2. Use GDS or KGDS to extract directions that capture differences across multiple dates.
3. Project each date, patch, or pixel/patch sample into that difference space.
4. Interpret the projected outputs through clustering, thresholding, supervised classification, or weakly supervised labeling.

Important limitation: GDS/KGDS does not automatically produce semantic labels such as construction, vegetation change, flood, or disaster damage. It produces a difference/projection space. A separate interpretation step is required:

- unsupervised clustering of projected features;
- supervised classification if labels exist;
- weak supervision using known event dates or masks;
- temporal pattern grouping;
- change-type discovery followed by manual/semantic interpretation.

Research question:

```text
Can GDS/KGDS over multiple Sentinel-2 dates extract more interpretable temporal difference directions than pairwise DS/KDS, and can those directions support non-binary or semantic change interpretation?
```

This should be documented as a high-value future experiment. It is not implemented yet in the active pipeline.

## 17. Position Information: What Is Lost And What Is Preserved

Current global OSCD DS:

- Preserves spectral vector values.
- Preserves pixel coordinates only for mapping scores back to `(H,W)`.
- Does not use pixel coordinates in PCA/subspace fitting.
- Does not enforce local spatial smoothness.
- Does not model neighborhoods unless `sliding_window_ds` or geodesic priors are used.
- Excludes pixels marked invalid by the pre/post valid mask.

So the answer is:

> We do not lose each pixel's spectral vector, but the global subspace construction ignores spatial position. Position is only used after scoring to reconstruct the image-shaped score map.

This is an important limitation and should be stated honestly.

There is also a separate validity-mask risk:

```text
valid_mask = valid_pre AND valid_post
```

This is meant to avoid nodata, missing-band, and rectification-border artifacts. However, it should be verified against the ground-truth change mask. If the valid-mask rule excludes many labeled changed pixels, then the PCA/DS pipeline may be silently dropping real change evidence.

Future check:

```text
For each OSCD city, count:
- total labeled changed pixels
- changed pixels excluded by valid_mask
- percentage of changed pixels excluded
```

This should be near zero or explicitly explained.

## 17A. Critical Next Task: Spatially Aware Subspace Validation

The largest open methodological risk is not whether the code can compute a DS score. It can. The risk is whether the current OSCD adaptation defines a meaningful subspace for a satellite image.

Current global OSCD DS treats the valid pixels in one date image as an unordered sample set:

```text
sample i          = one valid pixel
sample vector x_i = 13 Sentinel-2 band values at that pixel
X_pre             = [x_1, x_2, ..., x_N] in R^(13 x N)
```

This means PCA sees the distribution of 13-band pixel values, but it does not know where each pixel came from in the image. A road pixel, sea pixel, vegetation pixel, and building pixel are all just columns in the same matrix. Position is restored only after scoring, when scalar scores are placed back onto the `(H,W)` image grid.

This is not automatically wrong. It may be a valid spectral-distribution adaptation. But it must be tested against spatially aware alternatives before the thesis can claim that the subspace construction is appropriate.

### Research Question

```text
Is a global pixel-spectral subspace enough for OSCD change detection, or do local/patch-based subspaces preserve important spatial structure that improves change maps and downstream segmentation?
```

### Variants To Compare

#### A. Global Pixel DS: Current Baseline

Definition:

```text
one sample = one valid pixel
vector length = 13 bands
one subspace = all valid pixels from one date image
```

This is the current method. It should remain as the reference baseline because it is simple, fast, and already implemented.

Required checks:

- Use corrected `canonical` or repaired `eig` DS, not legacy residual-stack DS.
- Run on at least `beirut` first, then the full OSCD test split.
- Compare against raw spectral L2, PCA-diff, CVA, Celik, and IR-MAD where available.
- Record DS basis shape, PCA rank, explained variance, raw-L2 correlation, AUROC, PR-AUC, best F1, best IoU, Otsu F1/IoU, and runtime.

Decision value:

- If global pixel DS is weak but cleanly different from raw L2, it is a useful baseline but not enough as the main contribution.
- If it performs competitively, it can be framed as a spectral-distribution DS adaptation, while still acknowledging that it ignores spatial position during subspace fitting.

#### B. Patch-Vector DS: Preserve Neighborhood Texture

Definition:

```text
one sample = one local patch centered at a valid pixel
3x3 patch vector = 13 x 3 x 3 = 117 values
5x5 patch vector = 13 x 5 x 5 = 325 values
```

Instead of giving PCA only the 13 band values at one pixel, this gives PCA the local neighborhood around that pixel. This preserves some spatial texture and context.

Implementation tasks:

- Add patch extraction for pre/post Sentinel-2 cubes.
- Start with small patch sizes: `3x3` and `5x5`.
- Use sampling if full-image patch extraction is too large.
- Fit PCA subspaces for pre and post patch vectors.
- Compute DS score for each center pixel using the pre/post patch difference vector.
- Save scalar DS maps and optional projected-patch norm maps.

Required checks:

- Compare patch sizes `1x1`, `3x3`, and `5x5`.
- Track memory and runtime because patch vectors quickly become high-dimensional.
- Verify patch borders and valid masks are handled explicitly.
- Compare maps visually against global pixel DS and ground truth.

Decision value:

- If patch-vector DS improves change-map metrics or produces cleaner spatial maps, it answers Sensei/senpai concerns directly: the project should move from pure pixel-spectral DS to spatial patch DS.
- If it does not help, we can report that local patch context did not improve this OSCD setup and keep global pixel DS as a simpler baseline.

#### C. Local-Window DS: One Subspace Per Image Region

Definition:

```text
one local window = e.g. 128x128 or 256x256 pixels
one sample       = one 13-D pixel inside that window
one subspace     = one PCA subspace per date per window
```

This keeps the vector definition at 13 bands, but it prevents a single global subspace from mixing unrelated parts of the city. Roads, coast, dense urban areas, and vegetation can have local subspaces.

Implementation tasks:

- Audit the existing `sliding_window_ds` path and confirm whether it uses canonical/eig DS or still behaves like legacy residual-stack DS.
- If needed, update local-window DS to use corrected `canonical` DS.
- Test windows:

```text
window sizes: 64, 128, 256
strides:      32, 64, 128
aggregation: mean, max
```

Required checks:

- Compare local-window DS against global pixel DS with the same rank.
- Report runtime, because local PCA/DS can become expensive.
- Save per-city maps and city-level metrics.
- Inspect boundary artifacts caused by window aggregation.

Decision value:

- If local-window DS improves maps, the thesis can argue that the subspace idea is useful but must be localized for remote-sensing scenes.
- If it is too slow, it can still become a prototype or ablation rather than the production prior generator.

#### D. Coordinate-Augmented DS: Optional Sanity Check

Definition:

```text
sample vector = [13 band values, alpha*x_coord, alpha*y_coord]
```

This is not paper-faithful DS and should be treated as a sanity check, not a main method. It tests whether adding position directly to the sample vector changes the learned subspace.

Risks:

- Coordinates and reflectance values have different units.
- Results may depend heavily on coordinate scaling `alpha`.
- This can artificially encode location instead of change structure.

Decision value:

- Use only as an exploratory diagnostic.
- Do not build the main thesis around this unless Sensei explicitly encourages it.

#### E. CCA/S3CCA-Inspired Spatial Matching: Later Research Track

S3CCA and temporally regularized CCA matter because they preserve structure over sample indices rather than treating samples as fully exchangeable. This is probably too large for the immediate repair task, but it should remain a serious research track if Sensei says the pixel-sample DS definition is insufficient.

Possible future question:

```text
Can a structured CCA/subspace method preserve spatial arrangement better than PCA subspaces built from unordered pixel samples?
```

### Minimum Script To Build

Create a focused script, not a broad pipeline rewrite:

```text
phase1/scripts/audit_oscd_spatial_subspace.py
```

Proposed command:

```powershell
.\.venv\Scripts\python.exe phase1/scripts/audit_oscd_spatial_subspace.py `
  --city beirut `
  --rank 6 `
  --methods global_pixel,patch3,patch5,window128 `
  --output_dir phase1/outputs/oscd_spatial_subspace_audit_$tag
```

The script should save:

- `metrics.csv`
- `run_metadata.json`
- `global_pixel_ds.png`
- `patch3_ds.png`
- `patch5_ds.png`
- `window128_ds.png`
- `raw_l2.png`
- `pca_diff.png`
- side-by-side comparison figure with pre RGB, post RGB, ground truth, and all tested maps

### Metrics To Report

For each city/method:

- AUROC
- PR-AUC / average precision
- best F1 over thresholds
- best IoU over thresholds
- Otsu-threshold F1/IoU
- correlation with raw spectral L2
- percentage of ground-truth change pixels excluded by valid mask
- runtime
- peak memory if easy to measure

For Phase 2 follow-up:

- raw-only U-Net
- raw + global canonical DS
- raw + patch DS
- raw + window DS

Only run Phase 2 after Phase 1 maps look meaningful. Do not spend long training time on a DS prior that is already weak as an unsupervised change map.

### Acceptance Criteria

The current global OSCD adaptation becomes defensible if:

- canonical/eig DS maps are not near-identical to raw L2;
- the method has stable behavior across several cities;
- rank sensitivity does not show that rank 6 was arbitrary or unstable;
- valid-mask exclusions are negligible or explained;
- global pixel DS is competitive with patch/window variants, or its weakness is explicitly reported.

The project should pivot toward spatially aware DS if:

- patch-vector or local-window DS gives clearer maps and better AUROC/F1/IoU;
- global pixel DS fails mainly in mixed land-cover scenes;
- Sensei confirms that unordered pixel-sample subspaces are too weak for the intended subspace-method contribution.

The project should pause DS-as-prior claims if:

- corrected canonical DS remains consistently worse than simple raw spectral/PCA baselines;
- patch/window variants do not improve it;
- Phase 2 only benefits from priors that are effectively raw difference maps.

### Short Explanation For The Thesis

If the spatial audit is successful:

> The initial global spectral subspace treats valid pixel spectra as unordered samples. To test whether spatial context matters, we compare it with patch-vector and local-window subspace constructions. This separates the effect of spectral-distribution modeling from the effect of local spatial structure.

If the spatial audit is negative:

> The experiments show that a direct global pixel-spectral adaptation of DS is not sufficient for OSCD. This negative result motivates either localized subspace construction, kernelized/local DS, or a narrower thesis claim focused on interpretable prior diagnostics rather than DS superiority.

## 18. Projection Back To Image Space

There are two different "projection back" ideas:

### OSCD score-map reconstruction

The active OSCD code does not reconstruct a spectral image from the DS projection. It computes a scalar per pixel:

```text
score_i = ||D^T (x_post_i - x_pre_i)||^2
```

Then `devectorize_cube` puts those scalar scores back into the original `(H,W)` grid using saved row/column indices.

So the output is not a reconstructed image in the original 13-band space. It is a scalar map.

### TPAMI DS visualization

In the TPAMI Venus/object setting, projecting an image vector onto DS/KDS can visualize difference components. For linear DS, the projected vector can be reshaped back to image size. For KDS, proper visualization requires handling nonlinear feature-space projection/preimage or related visualization machinery. Our Venus script now computes paper-formula KDS/KGDS projection coordinates and energies, but it still does not reconstruct the preimage needed for TPAMI-style emphasized-image visualization.

Future OSCD visualization task:

```text
For each pixel:
delta_x      = x_post - x_pre
delta_x_ds   = D D^T delta_x
residual     = delta_x - delta_x_ds
```

This would reconstruct the part of each 13-band change vector that lies in the DS basis. Visualizing `delta_x_ds` as band-wise maps, RGB composites, or norm maps could help answer:

- What spectral-band combinations does DS emphasize?
- Is DS extracting meaningful changes or just noise/artifacts?
- How does DS projection differ from raw spectral difference and PCA-diff?

This is not needed for the scalar DS prior, but it is valuable for interpretation and for explaining "projection back" to Sensei/senpais.

## 19. What The Current Experiments Say

### OSCD audit

Command:

```powershell
.\.venv\Scripts\python.exe phase1/scripts/audit_oscd_subspace.py --city beirut --rank 6
```

Observed:

```text
X_pre/X_post shape: (13, 1262600)
Phi/Psi shape:      (13, 6)
legacy D shape:     (13, 12), corr with raw L2 = 0.999990
eig D shape:        (13, 6),  corr with raw L2 = 0.190370
canonical D shape:  (13, 6),  corr with raw L2 = 0.190364
```

Interpretation: the old default was not a defensible clean DS implementation. The repaired canonical/eig paths are geometrically much closer to the paper.

### Venus KDS/KGDS audit

Command:

```powershell
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe phase1/scripts/venus_kds_demo.py --venus_root data/venus_tpami2015 --output_dir "phase1/outputs/venus_kds_audit_$tag"
```

Existing verified output:

```text
phase1/outputs/venus_kds_faithful_20260508_001732/
  run_summary.json
  venus_montage.png
  venus_linear_pca_basis.png
  venus_linear_ds_basis.png
  venus_kernel_kds_pair_energy.png
  venus_kernel_kds_pair_coordinates.png
  venus_kernel_kgds_three_class_energy.png
  venus_kernel_kgds_three_class_coordinates.png
  venus_kernel_spectra.png
```

Observed:

```text
Venus raw files:              (480, 640, 1, 300)
Downsampled matrix per class: (3024, 300)
KDS setup:                    2 classes, 100-D kernel subspace each, 100-D KDS
KGDS setup:                   3 classes, 150-D kernel subspace each, 300-D KGDS
KDS basis norm error:         5.402e-11
KGDS basis norm error:        4.595e-11
```

Interpretation: this is now a paper-formula implementation of the kernel coefficient and projection parts of KDS/KGDS. It should still not be presented as a complete TPAMI figure reproduction because the preimage search step is missing.

### Canonical OSCD prior generation

Existing verified output:

```text
phase1/outputs/oscd_priors_canonical_20260507_044621/
  oscd_change_maps/
  oscd_eval_results.json
  oscd_eval_summary.csv
  run_metadata.json
```

Test split summary:

```text
canonical ds_projection AUROC: 0.6246
pixel_diff AUROC:              0.7559
pca_diff AUROC:                0.8134
```

Interpretation: paper-faithful canonical DS is weaker than simple baselines on OSCD in this run. This is not a failure; it is important research evidence. It means the paper-faithful DS idea may not transfer strongly to global pixel-spectral OSCD without local/spatial/kernel extensions.

## 20. Reading Path For You

Read in this order:

1. `docs/SUBSPACE_METHOD_NOTES.md`
   - This file. Read it first.

2. `phase1/scripts/audit_oscd_subspace.py`
   - Read only the data flow: load city, normalize, vectorize, PCA, compare DS variants.

3. `phase1/data/preprocessing.py`
   - Focus on `vectorize_cube` and `devectorize_cube`.

4. `phase1/ds/pca_utils.py`
   - Focus on `fit_pca_basis`, `difference_subspace_canonical`, `difference_subspace_eig`, and `legacy_residual_stack_difference_subspace`.

5. `phase1/ds/ds_scores.py`
   - Focus on `_compute_ds_matrix_scores`.

6. TPAMI 2015 DS/GDS paper:
   - Read abstract and introduction first.
   - Then read canonical angles and DS sections.
   - Then read KDS/KGDS sections only after linear DS is clear.

7. `phase1/scripts/venus_kds_demo.py`
    - Read this after the TPAMI intro. It maps Sensei's Venus files into code.

8. `phase1/subspace/kernel_difference_subspace.py`
   - Read this beside TPAMI Sections 6.1-6.3. It is the code version of the kernel coefficient/projection equations.

9. S3CCA paper:
    - Read only after you understand current DS, because it is a separate CCA-based idea.

## 21. Updated Sensei Answer

Short answer:

> I audited the implementation. In OSCD, I currently generate one PCA subspace per time image by treating each valid pixel as a 13-D Sentinel-2 spectral vector, so `X_pre` and `X_post` are `13 x N`. This is not per-channel PCA and not whole-image-vector PCA. I found a problem: the old residual-stack DS variant produced a 12-D basis for rank-6 subspaces in 13-D and behaved almost exactly like raw spectral difference. I separated it as a legacy method and added canonical/projector-eigen DS, which gives a 6-D DS and matches the TPAMI linear DS formulation much better. I also implemented the TPAMI-style KDS/KGDS coefficient and projection equations for the Venus data: each whole image view is one 3024-D sample, KDS uses two 100-D kernel subspaces, and KGDS uses three 150-D kernel subspaces. The remaining missing part is preimage reconstruction, so I can show projection-energy/coordinate diagnostics now, but not yet the exact TPAMI emphasized-image visualizations.

More technical answer:

> The current OSCD adaptation is a linear spectral-subspace method. It uses pixel spectra as samples, fits PCA subspaces in `R^13`, then scores each pixel by projecting `x_post - x_pre` onto the DS basis. Pixel positions are preserved only for reconstructing the score map, not for subspace fitting. This is different from the TPAMI Venus setting, where each whole image view is a high-dimensional vector and 300 views form the image-set subspace. For Venus, I now compute nonlinear basis vectors as kernel combinations, form `E^T E`, take the smallest positive eigen-directions for KDS/KGDS, and project inputs using kernels as in Eq. 16/17. Next I need either preimage visualization for Venus or a careful decision about whether a local/kernel OSCD version is mathematically justified.

## 22. Immediate Next Research Tasks

1. Review the Venus KDS/KGDS outputs with Sensei:
   - Show the data shape, KDS/KGDS ranks, and projection-energy diagnostics.
   - Say clearly that preimage reconstruction is not implemented yet.

2. Decide whether to implement preimage search:
   - Needed if we want images like TPAMI Fig. 12/13.
   - Not needed if the immediate goal is just mathematical understanding and OSCD direction-setting.

3. Run the spatially aware OSCD subspace audit from Section 17A before more long Phase 2 sweeps:
   - Compare global pixel DS, patch-vector DS, and local-window DS.
   - Start with `beirut`, then add at least one dense urban city and one difficult/low-change city.
   - Report AUROC, PR-AUC, best F1, best IoU, Otsu F1/IoU, raw-L2 correlation, valid-mask exclusions, runtime, and qualitative maps.
   - This is now the main answer to the concern that global PCA ignores pixel position during subspace fitting.

4. Use the spatial audit to decide whether OSCD should use:
    - global spectral DS as a simple baseline,
    - patch-vector DS as the main spatially aware adaptation,
    - local/windowed spectral DS as the main spatially aware adaptation,
    - spatially structured CCA-like matching as a larger future method,
    - or kernel/local DS.

5. Only after the Phase 1 spatial audit, run a small Phase 2 segmentation comparison:
   - E0 raw-only.
   - raw + global canonical DS.
   - raw + best spatially aware DS variant, if it produces meaningful Phase 1 maps.
   - This checks whether a mathematically cleaner and spatially justified DS prior helps supervised OSCD segmentation.

6. Implement and test an OSCD KPCA/KDS prototype:
   - Start with sampled global KDS because it is simplest.
   - Fit kernel subspaces from sampled valid pre/post 13-band pixel vectors.
   - Use out-of-sample kernel projection to score all pixels.
   - Compare against canonical DS, raw spectral difference, and PCA-diff.
   - Track memory/runtime carefully because full pixel KPCA is infeasible.

7. Implement and test a MultiSenGE GDS/KGDS prototype:
   - Start with one tile/patch and several dates.
   - Build one linear or kernel subspace per date.
   - Use GDS/KGDS to extract multi-date difference directions.
   - Test interpretation routes: clustering, temporal grouping, supervised labels if available, or weak labels from known events.
   - Be explicit that GDS/KGDS gives a difference space, not semantic change classes by itself.

8. Audit OSCD valid-mask impact:
   - Measure how many ground-truth changed pixels are excluded by `valid_mask`.
   - Report this per city and overall.
   - If non-trivial, inspect whether exclusions are nodata/registration artifacts or real changed areas.

9. Run PCA-rank sensitivity experiments:
   - Current fixed rank `6` is a practical hyperparameter, not a proven optimum.
   - Compare ranks such as `2, 3, 4, 5, 6, 8, 10, 12`.
   - Also compare variance-threshold selection, e.g. keep enough PCs for `95%`, `99%`, and `99.5%` variance.
   - Report how rank affects DS map quality, baseline-prior metrics, and Phase 2 segmentation performance.
   - Avoid treating rank `6` as theoretically special unless experiments justify it.

10. Add DS projection-reconstruction visualizations:
   - Compute `D D^T (x_post - x_pre)` per pixel.
   - Save band-wise maps, RGB-like composites, and norm maps.
   - Compare visually with raw `x_post - x_pre`, canonical DS score, and PCA-diff.
   - Use this to explain what "projection onto DS" means in image space.

11. Experiment with DS scalar change-map construction:
   - Current DS map is one scalar per pixel: `||D^T (x_post - x_pre)||^2`.
   - Compare squared projection norm, unsquared norm, normalized projection energy, residual energy, and ratios such as `||D^T delta||^2 / ||delta||^2`.
   - Test per-city vs global normalization before thresholding or Phase 2 use.
   - Compare Otsu thresholding, validation-selected thresholds, and no thresholding before supervised U-Net input.
   - Check whether the scalar score map agrees with the reconstructed DS norm map from task 9.

The current evidence suggests global canonical spectral DS alone is probably weak for OSCD, but it is the correct baseline to understand before trying more complex variants.
