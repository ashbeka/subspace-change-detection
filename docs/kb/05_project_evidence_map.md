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
| Band-Image/spatial-axis DS | spatial support inside flattened band vectors | smoothed PCA, spatial Gram/projector, cross-reconstruction | strongest DS result; distinct from matched geometric nulls, still below spatial PCA |
| Generic spatial/tensor subspace | preserved structure | patch tensor, SiROC, raw local PCA/covariance | crowded |
| RTW phenology | phase/warp invariance | snapshot subspace, global shift, PCA cross-reconstruction, TWDTW | tested negative on MultiSenGE and BreizhCrops |
| S3CCA change attribution | contiguous wavelengths | sparse PCA, per-band score, ECDBS | open only after reformulation |
| Local moment-factorized HSI change | separates mean/scale/orientation and attributes bands | covariance/SPD, MMD, unmixing, per-band attribution | mechanisms verified; held-out detector rejected |
| Geometry on frozen features | label-efficient learned representation | same-encoder cosine/L2 | future, encoder-confounded |

## Decision Rule

The spatial-axis result advances beyond OSCD only if all of the following hold:

1. A compatible external multispectral dataset is selected before inspecting method scores.
2. Preprocessing, rank, smoothing scales, controls, fusion, and primary metric are frozen.
3. Band-Image DS is compared with spatial PCA, matched Gram/projector, cross-reconstruction, and IR-MAD.
4. It adds repeatable AP or calibrated F1 beyond the strongest matched control.
5. The claim remains candidate ranking if only AUROC improves.

## Honest Novelty Verdict

The notes and experiments eliminate the broad claim that a generic subspace
detector is sufficient. Band-Image DS is the strongest DS construction, but
the matched-null experiment finds no performance contribution unique to DS:
cross-reconstruction is significantly weaker under matched centering, while
spatial PCA is stronger.
The remaining positive question is external transfer of the construction and
its city-specific complementarity. The local HSI factorization is a negative
detector result and a useful verification case.

## Next Falsifiable Step

Select one compatible external labeled multispectral benchmark, freeze the
OSCD-derived method definitions, and test transfer before another neural-prior
experiment.
