# H-A POSITIVE: invariant-residual (SFA-CD) beats SAM/CVA/DS under nuisance

Date 2026-06-19. Code: temporal/experiments/hsi_nuisance_invariance.py. Data: Salinas 204-band (synthetic
bitemporal CD: real class-swap changes + controlled global-affine radiometric nuisance, strength alpha).

## Result [experiment-evidence]
AUC(change) vs nuisance strength:
| alpha | SFA-CD | SAM | CVA | DS | rawL1 |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.05| 1.00 | 0.94 | 0.96 | 0.74 | 0.96 |
| 0.1 | 1.00 | 0.86 | 0.85 | 0.43 | 0.84 |
| 0.2 | 1.00 | 0.79 | 0.62 | 0.46 | 0.60 |
| 0.4 | 0.98 | 0.64 | 0.50 | 0.51 | 0.46 |

H-A SUPPORTED: the invariant-residual method (SFA-CD) is near-flat under nuisance while SAM/CVA/DS/rawL1
collapse (CVA -> chance). +0.34 over the best scalar at alpha=0.4. Plain shape-DS ALSO collapses -> the win
is from modeling-the-invariant, NOT from subspace-shape-comparison. First clear positive for the direction.

## Honest caveat (the novelty gap — do not overclaim)
The nuisance is GLOBAL AFFINE (per-band gain/offset + illumination), which SFA-CD/IR-MAD are DESIGNED to
remove -- so SFA winning here is partly by-construction (the easy case), and SFA-CD itself is established
(Wu/Du/Zhang). This validates the MECHANISM but is not yet OUR novelty. Where the novelty lives:
1. HARDER nuisance: nonlinear (BRDF/atmospheric scattering) and spatially-varying -- linear SFA/IR-MAD fail;
   a KERNEL (KSFA/KDS) or local/geometric invariant-residual could win. This is the real gap.
2. GEOMETRIC invariant model: GDS-common-subspace projection as the invariant model -- does it match SFA-CD
   AND add subspace ATTRIBUTION (which spectral directions are nuisance vs change) that the chi-square lacks?
3. Proper bar: add IR-MAD (the established affine-invariant CD baseline) -- SFA/GDS must match/beat it.
4. Real bitemporal HSI-CD benchmark (real nuisance) once downloaded.

## Sharpened H-A statement
"A geometric/nonlinear invariant-residual change detector that handles harder (nonlinear, spatially-varying)
radiometric nuisance than linear SFA/IR-MAD, with interpretable band/structure attribution." Defensible +
novel; the global-affine result is the floor, not the contribution.
