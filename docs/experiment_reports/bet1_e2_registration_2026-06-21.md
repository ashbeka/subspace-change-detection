# BET 1 / E2 — registration pillar + Bet 1 conclusion: CLOSED (2026-06-21)

Script: `temporal/experiments/bet1_e2_registration.py` · `temporal/outputs/bet1_e2_registration/`
Real spatial Salinas image, 7x7 patches, actual pixel-shift misregistration + illumination; injected sub-pixel
distributional change. AUC(change vs no-change), both carrying the same misregistration.

## Result
| condition | per_pix_CVA | per_pix_SAM | patch_mean_CVA | patch_mean_SAM | shrink_corr | SET_SUBSPACE |
|---|---|---|---|---|---|---|
| clean (sh0,a0) | 0.718 | 0.703 | 1.000 | **1.000** | 0.957 | 0.545 |
| sh0,a0.1 | 0.652 | 0.718 | 0.775 | **1.000** | 0.953 | 0.534 |
| sh2,a0.0 | 0.659 | 0.672 | 0.918 | **0.932** | 0.899 | 0.518 |
| sh2,a0.1 (misreg+illum) | 0.636 | 0.679 | 0.733 | **0.930** | 0.893 | 0.521 |

## Findings
- **Registration robustness is REAL but SHARED.** Per-pixel CVA degrades under misregistration (0.72→0.64) and
  illumination; patch methods stay high. But the winner is **patch_mean_SAM (0.93–1.0)** — registration-robust
  (overlapping-window mean) AND illumination-robust (angle) AND it detects the change. shrink_corr is second
  (0.89–0.96). The registration advantage does NOT favor the subspace; it favors all patch methods.
- **SET_SUBSPACE is the WORST method (0.52, near chance)** in every condition — far below patch_mean_SAM and
  shrink_corr. It does not beat the simple patch nulls.

## BET 1 CONCLUSION — comprehensively CLOSED (both structural pillars)
- **Pillar 1 (E1, the N<<B distributional crux):** subspace flat ~0.55–0.62 at every N; sample/shrink
  correlation reaches 0.99; no small-sample denoising free lunch (top-k truncation discards the subtle change).
- **Pillar 2 (E2, registration robustness):** subspace worst (0.52); patch_mean_SAM / shrink_corr win and are
  already registration+illumination robust.
The failure is STRUCTURAL: (a) the subspace's fixed-rank truncation is a liability for subtle low-rank change
(loses it below the material's high-rank covariance), and (b) registration robustness is a property of ANY
patch/set comparison, not of the subspace specifically. E3 (more change types) and E4 (more real data) cannot
rescue a method that is worst on BOTH pillars (it already lost across change types in E1).
- Secondary (non-subspace) finding: **patch_mean_SAM is a strong, simple, registration+illumination-robust CD
  baseline** for subtle distributional change — but that's a known simple method, not a subspace contribution.

## Decision
Per the pre-registered gate and the user's plan: Bet 1 has no positive lead → PIVOT TO BET 2 (multi-dimensional
damage/recovery trajectory characterization). The diagnostic gains another decisive, real-data row.
