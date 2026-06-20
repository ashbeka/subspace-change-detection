# KB 05: Project Evidence And Researcher Constraints

This map prevents literature mining from reviving hypotheses the project has already falsified.

## Advisor And Researcher Steers

Sources: `notes/my_notes.md`, `notes/feedback.md`, `notes/methods.md`, `notes/experiments.md`, `notes/literature.md`, `notes/research_paper_plan.md`, Sensei Slack records, source-record coverage audits, and all reports under `E:/research_projects/sccd-claude/docs/experiment_reports/`.

| Constraint or steer | Research consequence | KB link |
|---|---|---|
| Sensei's bar is a first/unique trial, not necessarily best performance. | Prefer a new, precise characterization task over an unsupported SOTA claim. | KB 01/02 |
| The project should start from a real CD problem, not invent a problem to protect DS. | Tie candidates to HSI spectral variability, subpixel change, band redundancy, registration, and interpretability. | KB 04 |
| Current global pixel subspaces can destroy spatial information. | Use local support and state explicitly that an unordered local distribution still discards within-window arrangement. | KB 03/04 |
| S3CCA/TRCCA, SSA/SFA, RTW, tensors, geodesics, and latent subspaces were advisor/senpai suggestions. | Treat each as a mechanism candidate; do not force all into one method. | KB 02/03 |
| Hyperspectral imagery may be a more natural home than low-band Sentinel-2. | Compare against HSI-specific SCD, ACD, unmixing, band selection, and deep models. | KB 04 |
| Explain the sample and matrix before saying “we build a subspace.” | Every candidate needs a construction card and preserved/lost-information statement. | KB 01 |
| Avoid disaster/xBD overclaims. | The immediate task is HSI change characterization; damage/recovery remains motivation and future validation. | KB 03 |

## Findings That Close Directions

### Rank-1 first-order DS is SAM

`[E]` On low-dimensional per-pixel spectra, first-order DS magnitude is a monotone spectral-angle score. No proposal may claim that object as a new detector.

### Plain subspace detection loses on real data

`[E]` Corrected OSCD all-city results and Salinas material-subspace tests show PCA-diff, SAM/CVA, and IR-MAD equal or outperform global/patch/band-image DS. Expanding the material subspace increases tolerance to changed pixels and can reduce discrimination.

### Generic invariance is occupied

`[E]` IR-MAD and SFA remain strong under affine, nonlinear, and spatial nuisance. A proposed invariant detector must beat them, not raw CVA alone.

### Trajectory dynamics is largely closed

`[E]` The mean-spectrum vector second difference recovers direction curvature; DS curvature was weaker on realistic data; second-order DS did not fire in the intended test; subspace dynamics was not nuisance-invariant. Artificial-splice SSA-DS success did not transfer to natural abrupt, gradual, or phenological events.

### Mean-preserving distributional change is the residual

`[E]` In a controlled centered patch, an orientation rotation with constant mean was detected by subspace/eigenspace scores while mean-based tools were blind. But covariance statistics also detected it, and the tested uncentered subspace was not scale-invariant. Therefore the residual is not “DS detects what covariance cannot.” It is the narrower question of whether a centered, scale-quotiented orientation component yields stable and interpretable information beyond full covariance/two-sample tests.

### Local eigenspectrum can be competitive but is not validated

`[E]` On controlled MultiSenGE interventions, a smoothed local eigenspectrum reached AP 0.688 versus NDMI 0.680 and first-DS 0.456; confidence intervals overlapped, translation and missing composites caused strong false alarms, and labels were injected rather than natural. This supports factorization experiments, not a performance claim.

## Candidate Survival Matrix

| Candidate | Structural lever | Killer null | Status |
|---|---|---|---|
| More first-order DS detection | none beyond angle | SAM/CVA | closed |
| Second-order/geodesic satellite trajectory | curvature | vector second difference, BFAST/CCDC | closed for top task |
| Learned normal SSA-DS | normal-direction invariance | harmonic deseasonalization, SFA/IR-MAD | natural-data failure |
| Material-subspace membership | intra-material tolerance | centroid SAM/Mahalanobis/unmixing | negative |
| Generic spatial/tensor subspace | preserved structure | patch tensor, SiROC, raw local PCA/covariance | crowded |
| RTW phenology | phase/warp invariance | TWDTW, phase-aware harmonic | open but data-gated |
| S3CCA change attribution | contiguous wavelengths | sparse PCA, per-band score, ECDBS | open only after reformulation |
| Local moment-factorized HSI change | separates mean/scale/orientation and attributes bands | covariance/SPD, MMD, unmixing, per-band attribution | strongest provisional task |
| Geometry on frozen features | label-efficient learned representation | same-encoder cosine/L2 | future, encoder-confounded |

## Decision Rule

The top task is accepted only if all of the following hold:

1. The orientation score is invariant to mean shift and isotropic scale by construction and in a toy check.
2. It detects orientation-only changes with matched mean and eigenspectrum.
3. It adds mechanism classification or wavelength attribution beyond shrinkage covariance/SPD and MMD/energy.
4. The attribution is more stable/physically faithful than standardized per-band differences, sparse PCA, and unmixing.
5. It does not collapse under eigengap degeneracy, one-pixel translation, or realistic sample scarcity.

## Honest Novelty Verdict

The notes and experiments eliminate the broader subspace-detector narrative. They support one narrow research question about *factorized local spectral-distribution change and its explanation*. This is a hypothesis under severe null pressure, not an established method contribution.

## Next Falsifiable Step

Write a paper-to-code construction card and a four-regime moment-matched toy protocol (mean-only, scale-only, orientation-only, null) before any real-data acquisition or model training.
