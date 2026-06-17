# Reference Code Inventory

Purpose: map the reference code that can help verify subspace methods, baselines, and future experiments. Reference code is not ground truth. It is evidence to compare against papers, dimensions, toy examples, and active project behavior.

Current rule: do not force-add large ignored reference-code folders to Git. Keep them local unless a later decision uses Git LFS, a separate archive repo, or a small extracted verification fixture.

## 1. Summary

| Folder | Tracked files | Total files | Size | Status |
|---|---:|---:|---:|---|
| `references/reference_code/DS` | 9 | 11 | 0.04 MB | Small tracked reference. |
| `references/reference_code/MagTool-main` | 4 | 7 | 0.06 MB | Small tracked reference. |
| `references/reference_code/Subspace Toolbox` | 774 | 950 | 470.92 MB | Large tracked legacy toolbox. Review before any cleanup. |
| `references/reference_code/cv_motion3d_public-main` | 0 | 33 | 67.81 MB | Local ignored reference. |
| `references/reference_code/real-time-subspaceMethod-main` | 0 | 40 | 10.54 MB | Local ignored reference. |
| `references/reference_code/shape-subspace-motion-analysis-main` | 0 | 2554 | 162.00 MB | Local ignored reference. |
| `references/reference_code/subspace-toolbox-new-version` | 0 | 931 | 911.84 MB | Local ignored reference; probably newer/duplicated toolbox. |
| `references/reference_code/SubspaceMethodsToolBox-main` | 0 | 30 | 0.76 MB | Local ignored reference. |
| `references/reference_code/subspyces-main` | 0 | 42 | 0.68 MB | Local ignored reference. |

## 2. Method-Relevance Map

| Folder | Language | Main method family | Useful files / entry points | Project use | Verification needed |
|---|---|---|---|---|---|
| `DS` | Python | Shape/motion DS magnitude and contribution visualization. | `utils.py`, `sm.py`, `visual_contribution_first.py`, `visual_contribution_geodesic.py`, `visual_contribution_second_ds.py` | Useful analogy for DS contribution analysis. It shows a temporal/shape setting where each frame or motion state can form a subspace and DS magnitude is visualized. | Check formulas against DS/GDS papers before adapting. It uses C3D motion data, not satellite imagery. |
| `MagTool-main` | Python | Difference subspace, Karcher/mean subspace, second-order DS magnitude, PCA/QR construction. | `MagTool-main/README.md`, `construct.py`, `magnitude.py` if present in nested folder. | Strong cross-check candidate for DS magnitude, similarity, Karcher subspace, first/second difference decomposition, and column-vector conventions. | Verify exact definitions, eigenvalue thresholds, and whether "magnitude" matches our score-map use. |
| `Subspace Toolbox` | MATLAB/C/toolbox | Broad older CVLab-style subspace toolbox: PCA/KPCA, MSM/KMSM/CMSM, SVM/random forest utilities. | `cvlToolBox`, `cvtToolBox`, `cvlPCA`, `cvlKPCA`, canonical-angle utilities. | Historical reference for lab-style subspace methods and MATLAB conventions. | Large and partly old; do not rely on it blindly. Identify the exact function before adapting. |
| `subspace-toolbox-new-version` | MATLAB/C/toolbox | Newer or duplicated version of the broad toolbox. | `cvlToolBox`, `cvtToolBox`, `toolbox.zip`. | Potential newer reference for PCA/KPCA/canonical-angle utilities. | Compare with tracked `Subspace Toolbox`; keep local until we know whether it supersedes or duplicates the old one. |
| `SubspaceMethodsToolBox-main` | MATLAB | Mutual Subspace Method, CMSM, KMSM, canonical angles, KPCA. | `docs/README.md`, `examples/msm.m`, `examples/kmsm.m`, `src/functions/cvlCanonicalAngles.m`, `src/functions/cvlKernelCanonicalAngles.m`, `src/utils/cvlPCA.m`, `src/utils/cvlKPCA.m` | Cleanest compact MATLAB reference for MSM/KMSM and kernel subspace construction. Useful for Sensei/Santos-style subspace verification. | It is a classification/matching toolbox, not a change-map method. Adaptation to Sentinel-2 must define sample unit and score map separately. |
| `real-time-subspaceMethod-main` | MATLAB | MSM, KMSM, KOMSM/KCMSM-style real-time subspace methods. | `readme.txt`, `cvlBasisVector.m`, `cvlKernelBasisVector.m`, `cvlCanonicalAngles.m`, `cvlKernelCanonicalAngles.m`, `cmsm_realtime.m`, `kmsm.m`, `msm.m` | Useful for sample/basis conventions: `X` as `d x N x K`, subspace dimension or cumulative variance ratio, Gaussian kernel parameter. | Need paper matching. Good for dimensions and APIs, not automatically for DS. |
| `subspyces-main` | Python/PyTorch | General Python subspace library: VectorSpace, generators, PCA transforms, metrics. | `subspyces/core/vector_space.py`, `generators/`, `transform/pca_transform.py`, `metrics/` | Useful design reference if we later make a cleaner Python subspace API. It can also guide tests for vector-space objects and transforms. | It is generic, not DS/GDS-specific. Do not rewrite project around it before the experiment needs are clear. |
| `cv_motion3d_public-main` | Python | Motion DS visualization: first difference, second difference, geodesic/orthogonal components. | `readme.md`, `visual_contribution_first.py`, `visual_contribution_geodesic.py`, `visual_contribution_second.py` | Useful visual and conceptual analogy for contribution decomposition, especially if we later show which bands/patches contribute to change. | Japanese README labels need source-paper confirmation. Not satellite data. |
| `shape-subspace-motion-analysis-main` | Python/Jupyter | Shape subspace motion analysis, Grassmann projection, KNN, skeleton normalization, sequence analysis. | `grassmann_projected_visualization.py`, `enhanced_knn_classifier.py`, `sequence_projected_3d_hand_action_gm_visu.py`, `utils.py`, notebooks/logs. | Possible analogy for spatial/shape subspace representation and classification over subspaces. Useful if we study region/object subspaces or contribution visualizations. | Large and likely exploratory. Do not copy algorithms without tracing paper/source and input dimensions. |

## 3. Immediate Uses

1. Cross-check canonical angles and PCA/KPCA conventions before revising any DS/KDS code.
2. Compare DS magnitude and equal-subspace behavior against `MagTool-main`.
3. Use `SubspaceMethodsToolBox-main` and `real-time-subspaceMethod-main` to explain sample matrix conventions to Sensei: vectors are columns, data is `d x N`, subspace dimension can be fixed or variance-based.
4. Use motion/shape references only as analogies for contribution visualization, not as remote-sensing evidence.
5. If a spatial DS method is implemented, add a source-to-code note linking:
   `spatial sample definition -> basis construction -> DS comparison -> score map -> metric output`.

## 4. What Not To Do

- Do not claim any reference code proves our satellite adaptation is correct.
- Do not mix image-set subspace methods and pixel-distribution subspaces without explicitly stating the sample unit.
- Do not import entire large ignored folders into Git without approval.
- Do not use `Subspace Toolbox` and `subspace-toolbox-new-version` interchangeably until duplication/version differences are checked.

## 5. Next Checks

| Check | Why |
|---|---|
| Compare our canonical/projector DS on toy subspaces against `MagTool-main`. | Ensures paper-derived dimensions and equal-subspace edge cases behave correctly. |
| Run one PCA/KPCA toy example in MATLAB/Python reference code and our code. | Prevents silent convention errors before KDS/KGDS work. |
| Identify exact paper behind `cv_motion3d_public-main` and shape-motion code. | Needed before citing or adapting contribution decomposition. |
| Decide whether `Subspace Toolbox` and `subspace-toolbox-new-version` are duplicates. | Large local/tracked footprint needs cleanup planning later. |

