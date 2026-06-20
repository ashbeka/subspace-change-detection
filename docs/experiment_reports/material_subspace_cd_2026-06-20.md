# Material-signature subspace CD (MSM/GDS) — NEGATIVE, fairly tested (2026-06-20)

Script: `temporal/experiments/material_subspace_cd.py` · `temporal/outputs/material_subspace_cd/`
The user-proposed idea (e): a MATERIAL is a SUBSPACE (span of its spectral signatures); detect material change
via subspace membership (MSM), with the common-across-materials subspace removed (GDS), on hyperspectral
(Salinas 204-band, where the subspace is genuinely multi-dim → DS≠SAM). The lab's MSM/GDS essence adapted to CD.

## Result — AUC(detect material change) vs illumination/per-band nuisance
| alpha | CVA | **SAM_centroid** | MSM | GDS_centroid | GDS_MSM |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 0.839 | **0.961** | 0.788 | 0.783 | 0.501 |
| 0.2 | 0.580 | **0.834** | 0.527 | 0.659 | 0.519 |
| 0.4 | 0.526 | 0.544 | 0.556 | 0.560 | 0.527 |

Fairness check (K-sweep @ α=0.15): SAM_centroid 0.918 vs MSM K=2 **0.808**, K=3 0.740, K=8 0.586 —
MSM is worse than SAM at EVERY K, and monotonically worse as the subspace widens.

## Verdict — NEGATIVE (the subspace HURTS)
**SAM-to-mean (a single discriminative, amplitude-invariant direction) beats every subspace method (MSM/GDS)
under nuisance, and the more subspace dimensions, the worse.** Principled reasons (not a tuning artifact):
1. SAM's angle is already illumination-invariant → it handles the dominant nuisance without any subspace.
2. The material SUBSPACE trades discrimination for TOLERANCE — it tolerates changed pixels too, which is
   exactly wrong for *detecting* change (you want intolerance to real change). Wider subspace ⇒ more tolerance ⇒
   worse detection (monotone K trend).
3. GDS removes the common subspace, but correlated class means live MOSTLY in that common subspace, so GDS
   removes discriminative information → GDS_centroid < SAM_centroid, GDS_MSM ≈ chance.

## Significance
This closes the user-proposed material-subspace direction (e) and, importantly, the LAB'S OWN MSM/GDS framing
adapted to HSI CD — another confirmation of the diagnostic: subspace geometry adds nothing (here it HURTS) vs a
simple SAM/mean for satellite change detection. The "DS≡SAM unless multi-dim" lesson recurs: here the subspace
IS multi-dim, but multi-dimensionality (tolerance) is a liability for detection, not an asset.
