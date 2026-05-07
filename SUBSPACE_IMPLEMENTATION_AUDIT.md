# Subspace Implementation Audit

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

Sensei's files at repo root are valid MATLAB arrays:

```text
venus_nothing.mat  -> venus_nothing  shape (480,640,1,300)
venus_er2.mat      -> venus_er2      shape (480,640,1,300)
venus_er_ne.mat    -> venus_er_neck  shape (480,640,1,300)
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
.\.venv\Scripts\python.exe phase1/scripts/venus_kds_demo.py --output_dir "phase1/outputs/venus_kds_audit_$tag"
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
venus_nothing:          300 views
venus_er2:              300 views
venus_er_neck:          300 views
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

## 16. Position Information: What Is Lost And What Is Preserved

Current global OSCD DS:

- Preserves spectral vector values.
- Preserves pixel coordinates only for mapping scores back to `(H,W)`.
- Does not use pixel coordinates in PCA/subspace fitting.
- Does not enforce local spatial smoothness.
- Does not model neighborhoods unless `sliding_window_ds` or geodesic priors are used.

So the answer is:

> We do not lose each pixel's spectral vector, but the global subspace construction ignores spatial position. Position is only used after scoring to reconstruct the image-shaped score map.

This is an important limitation and should be stated honestly.

## 17. Projection Back To Image Space

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

## 18. What The Current Experiments Say

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
.\.venv\Scripts\python.exe phase1/scripts/venus_kds_demo.py --output_dir "phase1/outputs/venus_kds_audit_$tag"
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

## 19. Reading Path For You

Read in this order:

1. `SUBSPACE_IMPLEMENTATION_AUDIT.md`
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

## 20. Updated Sensei Answer

Short answer:

> I audited the implementation. In OSCD, I currently generate one PCA subspace per time image by treating each valid pixel as a 13-D Sentinel-2 spectral vector, so `X_pre` and `X_post` are `13 x N`. This is not per-channel PCA and not whole-image-vector PCA. I found a problem: the old residual-stack DS variant produced a 12-D basis for rank-6 subspaces in 13-D and behaved almost exactly like raw spectral difference. I separated it as a legacy method and added canonical/projector-eigen DS, which gives a 6-D DS and matches the TPAMI linear DS formulation much better. I also implemented the TPAMI-style KDS/KGDS coefficient and projection equations for the Venus data: each whole image view is one 3024-D sample, KDS uses two 100-D kernel subspaces, and KGDS uses three 150-D kernel subspaces. The remaining missing part is preimage reconstruction, so I can show projection-energy/coordinate diagnostics now, but not yet the exact TPAMI emphasized-image visualizations.

More technical answer:

> The current OSCD adaptation is a linear spectral-subspace method. It uses pixel spectra as samples, fits PCA subspaces in `R^13`, then scores each pixel by projecting `x_post - x_pre` onto the DS basis. Pixel positions are preserved only for reconstructing the score map, not for subspace fitting. This is different from the TPAMI Venus setting, where each whole image view is a high-dimensional vector and 300 views form the image-set subspace. For Venus, I now compute nonlinear basis vectors as kernel combinations, form `E^T E`, take the smallest positive eigen-directions for KDS/KGDS, and project inputs using kernels as in Eq. 16/17. Next I need either preimage visualization for Venus or a careful decision about whether a local/kernel OSCD version is mathematically justified.

## 21. Immediate Next Research Tasks

1. Review the Venus KDS/KGDS outputs with Sensei:
   - Show the data shape, KDS/KGDS ranks, and projection-energy diagnostics.
   - Say clearly that preimage reconstruction is not implemented yet.

2. Decide whether to implement preimage search:
   - Needed if we want images like TPAMI Fig. 12/13.
   - Not needed if the immediate goal is just mathematical understanding and OSCD direction-setting.

3. Run a small Phase 2 segmentation comparison using canonical priors:
   - E0 raw-only vs raw+canonical DS.
   - This checks whether mathematically faithful DS helps supervised OSCD segmentation.

4. Decide whether OSCD should use:
    - global spectral DS,
    - local/windowed spectral DS,
    - spatially structured CCA-like matching,
    - or kernel/local DS.

5. Implement and test an OSCD KPCA/KDS prototype:
   - Start with sampled global KDS because it is simplest.
   - Fit kernel subspaces from sampled valid pre/post 13-band pixel vectors.
   - Use out-of-sample kernel projection to score all pixels.
   - Compare against canonical DS, raw spectral difference, and PCA-diff.
   - Track memory/runtime carefully because full pixel KPCA is infeasible.

The current evidence suggests global canonical spectral DS alone is probably weak for OSCD, but it is the correct baseline to understand before trying more complex variants.
