# Literature Notes

This file is the paper and reference-code matrix. Keep entries short and useful.

## Must-Cite Core Sources

| Source | Why it matters | Project use |
|---|---|---|
| Fukui and Maki, TPAMI 2015, Difference Subspace and Generalization | Original DS/GDS and KDS/KGDS reference. | Must cite for DS, GDS, KDS, KGDS. Do not claim DS is invented here. |
| OSCD / Daudt et al. 2018 | Defines the Sentinel-2 OSCD binary change benchmark. | Main active dataset context. |
| FC-Siamese / Daudt et al. 2018 | Siamese CNN baseline for change detection. | Important Phase 2 baseline context. |
| Celik 2009 PCA-kmeans | Classical unsupervised satellite change detection. | Baseline prior, not novelty. |
| Nielsen 2007 IR-MAD | Strong classical multi/hyperspectral change detection. | Baseline prior, not novelty. |
| Close et al. 2021 Sentinel-2 LULUCF change detection | Shows Sentinel-2 change analysis can be affected by rare true change and radiometric/seasonal variability. | Supports the pseudo-change and seasonality-risk discussion. |
| Metric-CD, WACV 2025 | Modern unsupervised deep metric change detection. | Raises comparison pressure for unsupervised CD claims. |
| xBD / Gupta et al. 2019 | Building damage dataset. | Future damage extension only. |
| xBD-S12 | Sentinel-1/Sentinel-2 disaster damage dataset. | Future extension only. |
| ChangeOS | Object-based semantic change/damage framework. | Future damage comparison if xBD work resumes. |

## Subspace And Kernel References

Fukui and Maki TPAMI 2015:

- Establishes DS/GDS and nonlinear KDS/KGDS.
- Sensei's Venus files are tied to this style of image-set subspace experiment.
- The current OSCD adaptation is different because samples are pixel-level 13-band vectors, not whole-image views.

S3CCA:

- Smoothly Structured Sparse CCA for partial pattern matching.
- Relevant because it preserves structure over sample indices.
- Possible future method if global pixel DS loses too much spatial structure.

Temporally Regularized CCA / KOTRCCA:

- Relevant for temporal sequences.
- More relevant to MultiSenGE than two-date OSCD.

KCCA / Kernel CCA:

- Appears in the archived reference-code leads.
- Keep it as a future structured/kernel matching route, not as current OSCD evidence.
- Do not let KPCA/KCCA/S3CCA/TRCCA become the thesis core unless a specific experiment and comparison justify that pivot.

## Reference Code

Bundled code in this repo:

- `references/reference_code/DS/utils.py`: projector-eigen style DS; close to repaired `eig` behavior.
- `references/reference_code/MagTool-main/.../magnitude.py`: subspace magnitude and difference-space style utilities.
- `references/reference_code/Subspace Toolbox/...`: MATLAB PCA, KPCA, CCA, Kernel CCA utilities.

Specific archived crosswalk:

- `references/reference_code/DS/utils.py`
  - `gen_shape_subspace`: builds a subspace from shape/motion data.
  - `gen_shape_difference_subspace`: projector-eigen style DS.
- `references/reference_code/MagTool-main/MagTool-main/magnitude.py`
  - `calcDiffSubspace`: projector/eigen difference-space style reference.
  - `calcKarcherSubspace`: mean/Karcher subspace reference.
  - `calcMagnitude`: canonical-cosine based subspace magnitude.
- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtPCA.m`
  - PCA where columns are samples.
- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtKernelPCA.m`
  - Kernel PCA reference.
- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtCCA.m`
  - CCA reference.
- `references/reference_code/Subspace Toolbox/cvtToolBox/analysis/cvtKernelCCA.m`
  - Kernel CCA reference.
- `references/reference_code/Subspace Toolbox/cvtToolBox/SubspaceMethod/cvtBasisVector.m`
  - Builds basis vectors from data arrays.

External repos:

- `https://github.com/ComputerVisionLaboratory/SubspaceMethodsToolBox`
- `https://github.com/ComputerVisionLaboratory/SubspacesToolkit`
- `https://github.com/ma-ath/subspyces`

Archived external-file leads:

- SubspaceMethodsToolBox: `cvlPCA.m`, `cvlKPCA.m`, `cvlBasisVector.m`, `cvlCanonicalAngles.m`, `cvlKernelBasisVector.m`, `cvlKernelCanonicalAngles.m`.
- SubspacesToolkit: `computePCA.m`, `computeKernelPCA.m`, `computeBasisVectors.m`, `computeSubspacesSimilarities.m`, `computeKernelSubspacesSimilarities.m`.
- subspyces: `pca_transform.py`, `cosine_canonical_angles.py`, `vector_space.py`.

Use these as reference points, not unquestioned ground truth.

## Dataset Context

OSCD:

- Active benchmark.
- Binary change/no-change.
- Sentinel-2 pre/post images.
- Not disaster damage severity.

MultiSenGE:

- Useful future path for GDS/KGDS because it has multiple dates.
- Not currently the supervised Phase 2 benchmark.
- Lacks OSCD-style labels for the current main question.

xBD / xBD-S12:

- Important if the project later becomes disaster damage mapping.
- Current repo does not implement damage-specific training/evaluation.

Abandoned greenhouses:

- Useful future application.
- Can be mentioned as motivation/use case only.
- Not current evidence unless a dataset, labels, and evaluation protocol are added.

## Bookmark Triage Rules

When Chrome bookmarks are imported later, classify each item as one of:

- `must-cite`: needed for thesis claims.
- `implementation-reference`: useful for code or math.
- `baseline`: method to compare against.
- `background`: useful context, not urgent.
- `future`: xBD, greenhouses, MultiSenGE, semantic change, or broader application.
- `discard`: duplicate, weak, unrelated, or no longer useful.

## Archive-Ingested Reference Leads

Additional references and code leads from old archive notes:

| Lead | Why keep it |
|---|---|
| Fukui et al., second-order Difference Subspace | Relevant if the project moves from pairwise DS to richer subspace relationships. |
| Kanai et al. 2023, time-series anomaly detection based on DS between signal subspaces | Useful example that DS-like reasoning can support anomaly/change settings beyond classification. |
| U-Net | Required background for Phase 2 segmentation model. |
| ResNet | Required only if ResNet-backbone experiments are used. |
| MagTool reference code | Useful for projector/eigen and subspace-magnitude cross-checking; not ground truth. |
| DS shape/motion reference code | Useful because it uses projector-eigen style DS close to the repaired `eig` path. |
| MATLAB Subspace Toolbox | Useful for PCA, KPCA, CCA, and Kernel CCA implementation cross-checks. |

Keep these as citation or implementation-reference leads. Do not cite old archive prose itself as evidence.
