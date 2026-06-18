# Failsafe Draft — Diagnostic Paper (robust to either fork)

Status: **failsafe / side draft.** Kept per user request as the low-risk ACCV target that is true and
publishable **regardless** of whether the subspace method "wins." Promote to primary only if the
multi-dimensional DS advantage (multivariate SSA battery) fails adversarial verification.

Working title:

```text
When Does Subspace Geometry Help Satellite Change Detection?
A Diagnostic Study of Difference Subspaces from Spectral Pixels to Temporal Dynamics
```

## 1. Why this framing is safe

It does not depend on DS beating anything. Its contribution is a *characterization* — the precise
conditions under which the lab's Difference Subspace machinery is informative vs redundant for
satellite change detection. Every result we already have supports it, positive or negative. It directly
answers the student's founding doubt (my_notes.md §3.1: *"am I using subspaces because they solve a
documented problem, or because the lab method exists and I am searching for a place to put it?"*) and
sits comfortably under Sensei's "first and unique trial, need not be top performance" bar.

## 2. The one-line finding (the paper's spine)

> The Difference Subspace adds value over scalar/angular baselines **only when the state or dynamics is
> genuinely multi-dimensional and the change is structural**. For spectrally low-rank regions or
> level/signature shifts (the common bi-temporal case) DS provably collapses to spectral angle and is
> "forced." For multi-mode temporal dynamics (multivariate SSA) the *full* DS aggregates evidence across
> canonical angles, beating conventional single-angle SSA and remaining robust under noise, while its
> basis attributes the change to specific bands/modes — capabilities scalar detectors lack.

## 3. Evidence already in hand (branch claude/temporal-ds)

1. **Negative half (rigorous):** at the working operating point, first-order spectral DS ≡ spectral angle
   (Spearman 0.9988 with SAM; forcing rank≥2 → chance). Fair one-line baselines (SAM, per-date-normalized
   differencing) tie or beat it. `[experiment-evidence]` (L1 + adversarial verdict)
2. **Boundary (mechanistic):** DS is exactly invariant to multiplicative gain — useful, but the *same*
   invariance makes it blind to subspace-preserving (brightness-only) change (AUC 0.465). Gain-invariance
   and detection completeness are in tension.
3. **Positive half (regime-specific):** multivariate SSA on multi-mode change — full DS AUC 1.0 vs
   min-angle 0.50 (chance); noise-robust (1.0→0.75) where min-angle stays at chance and the best scalar
   degrades; per-band attribution 100% (chance 16.7%). `[experiment-evidence]` (multiband battery; under
   adversarial verification 2026-06-18).
4. **Honest nuance:** DS does not always win (freq_split: min-angle wins). The contribution is the *map*
   of when each method is appropriate, not a leaderboard claim.

## 4. Paper structure

1. Introduction — the method-forcing risk in applying subspace geometry to change detection; the question.
2. Background — DS/GDS/second-order DS (Fukui 2024), SSA min-angle anomaly detection (Kanai 2023), SAM,
   classical CD baselines; the rank-1 collapse of DS to spectral angle (lemma).
3. Constructions — spectral-pixel, band-image, per-date spatial, and temporal/M-SSA subspaces; for each,
   the sample unit, ambient space, and what change it can/cannot see.
4. The collapse result — DS = spectral angle for low-rank states (proof + synthetic confirmation + fair
   baselines). This is the "forced" boundary.
5. The multi-dimensional regime — multivariate SSA; full DS vs min-angle vs strong scalars under noise;
   attribution; second-order/geodesic recovery decomposition.
6. (If feasible in time) one real Sentinel-2 case — a documented event time series; do the synthetic
   regimes appear? Otherwise frame real-data as the immediate future work with the protocol fixed.
7. Discussion — a decision guide: which construction for which change type; when to NOT use DS.
8. Conclusion — honest contribution: a characterization + a noise-robust, interpretable temporal-DS
   detector for multi-mode structural change; limitations (registration sensitivity, regime-specificity).

## 5. Figures (mostly already generated)

- Collapse: DS-vs-SAM scatter (ρ=0.9988); fixed-rank ablation (k≥2 → chance).
- Multi-dim win: multiband AUC bars + the noise-sweep crossover (full DS vs min-angle vs best scalar).
- Attribution: difference-subspace per-band energy localizing the changed band.
- Recovery: d_pre rise/fall + off-geodesic onset.
- Decision-guide table: change type × best method.

## 6. What still must be done before submission (shared with the positive path)

- Pass the multiband adversarial verification (stronger PSD/spectral scalars; construction robustness;
  min-angle-variant strawman check).
- ≥8 seeds and fair baselines on every figure (L1 verdict mandate).
- One real Sentinel-2 dynamics example if time permits (GEE), else fix the protocol as future work.
- Register the construction/hyperparameters before any real-data run (no post-hoc tuning).
