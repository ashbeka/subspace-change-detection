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
- Added `phase1/scripts/venus_kds_demo.py`.
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

`phase1/scripts/venus_kds_demo.py` adds a diagnostic prototype:

1. Load the Venus whole-image vectors.
2. Build linear PCA subspaces.
3. Build a linear canonical DS basis and visualize it in image space.
4. Build a shared RBF-KPCA coordinate embedding.
5. Fit subspaces in that KPCA coordinate space.
6. Compute canonical DS in that nonlinear coordinate space.

This is useful for understanding and showing progress, but it is not yet a complete paper-faithful KDS/KGDS implementation with exact kernel-space operators and preimage reconstruction.

## 7. Answer To Sensei

Use this as the concise current answer:

> Currently I generate OSCD subspaces by treating each valid Sentinel-2 pixel as a 13-D spectral vector and fitting PCA separately to pre/post matrices. I found that our active old residual-stack DS implementation is not faithful enough: with rank 6 it produces a 12-D projection and behaves almost like raw spectral difference. I have now separated that as a legacy variant and added canonical/projector-eigen DS for the paper-faithful linear version. Nonlinear KDS/KGDS from TPAMI2015 is not yet integrated into OSCD; I am first reproducing it on the Venus multi-view data you gave me, where each whole image view is one sample vector.

## 8. Commands

Run unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Audit one OSCD city:

```powershell
.\.venv\Scripts\python.exe phase1/scripts/audit_oscd_subspace.py --city beirut --rank 6
```

Run the Venus prototype:

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
