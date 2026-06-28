# Whiteboard cheat-sheet (one page — glance before/at the board)

## The 5 equations (write these if asked "show me the math")
```
1.  Samples (spatially-faithful):   X_t = [vec(b_1) | ... | vec(b_B)]  ∈ ℝ^(N×B)      (N = H·W pixels)
        each band-image = ONE sample;  the date = ONE subspace
2.  Per-date subspaces (rank r=12): Φ = U_pre^(1:r),  Ψ = U_post^(1:r)  ∈ ℝ^(N×r)
3.  Canonical angles:               Φᵀ Ψ = U Σ Vᵀ,     Σ = diag(cos θ_i)
4.  Difference Subspace:            D = (ΦU − ΨV) · [2(I − Σ)]^(−1/2),   magnitude = Σ_i 2(1 − cos θ_i)
5.  Change score (pixel i):         s_i = Σ_b ‖ P_D δ_b ‖²,   δ_b = X_post,b − X_pre,b,   P_D = D Dᵀ
    (xBD headline) Projector dist:   d_i = ‖ P_Φ − P_Ψ ‖_row i,   P_Φ = Φ Φᵀ   (how much the subspace rotated)
```

## The numbers to remember
- **OSCD ladder (AP):** raw CVA 0.23 → PCA 0.25 → smoothed-PCA 0.27 → IR-MAD 0.21 → DS 0.24 → fusion 0.28 → U-Net 0.48 → U-Net+DS-fusion **0.51**
- **DS-specific in fusion (OSCD):** M4(+DS) 0.513 > M5(no DS) 0.480 > M6(look-alike) 0.487; score-level p=**0.0098**
- **xBD (AUROC):** raw 0.44 → PCA 0.59 → IR-MAD 0.73 → projector **0.73**; **top-5% → 25% of damage (≈5× lift)**; beats PCA 5/5 events
- **CCA family (Benton AUC):** CVA 0.98 > IR-MAD 0.96 > sparse-CCA(S3CCA) 0.91 > plain CCA 0.71 > DS 0.51
- **Otsu (OSCD):** Otsu F1 ≈ 0.15 vs no-threshold U-Net F1 ≈ 0.51 → train raw, no Otsu
- **Change-type (Benton NMI):** polar-CVA 0.83 > raw 0.79 > DS 0.47 (geometry loses)
- **Normalization:** z-score 0.48 vs 0–1 min-max 0.22

## 3 things to draw on the board (rehearse once each)
1. **Two constructions** — a cube "B bands × H×W"; arrow A "slice into pixel-vectors (loses position)"; arrow B "slice into band-vectors (keeps position) ← ours".
2. **Difference subspace** — two tilted planes (pre, post); mark the shared line (cos θ≈1) and the sticking-out direction (cos θ small); brace the difference directions = D; drop δ's shadow onto D = the score.
3. **The diagnostic table** — 3 columns: change type | what wins | why. Rows: brightness→CVA; low-dim→SAM(=DS); spread→IR-MAD; type→polar-CVA. Conclusion: geometry helps in fusion + disaster localization, not raw detection.

## One-line answers to the 5 likeliest questions
- *Simple baseline first?* — "Yes, raw CVA is the floor (0.23/0.44), worst of all; we climb to IR-MAD and geometry."
- *Subspace dimension / losing pixels?* — "12 PCs; band-image construction keeps spatial position (per-pixel loses it)."
- *DS vs GDS?* — "First-order canonical DS (pairwise); GDS is multi-class; 2nd-order/geodesic is multi-date."
- *CCA / kernel / Otsu?* — "All tested: IR-MAD is CCA; sparse-CCA = S3CCA flavor; kernel-DS closed (< CVA); trained raw with no Otsu (best)."
- *Why doesn't geometry win?* — "Optical reflectance isn't a native manifold object (unlike PolSAR/EEG covariances); a covariance/direction statistic captures the same thing — that's the diagnostic, and it's the contribution."
