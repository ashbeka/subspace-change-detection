# EXPERIMENT QUEUE — one item per autonomous (or manual) run

The auto-resume runner (`scripts/auto_resume.ps1`) runs the TOP unchecked item, commits, then checks it off
with a one-line result. Keep each item scoped to a SINGLE run. Priority reflects CONTINUITY §4/§6: H-A
*detection* is closed → lead with H-B (trajectory), then H-C, then the real-data tests when data lands.
Always report the trivial/standard null (SAM/CVA/IR-MAD/scalar 2nd-difference) beside the method.

## Up next
- [ ] **SSDS-validate (the LIVE LEAD).** ssds_longseries gave the first real positive (a_hat+D_N localizes a
  real change, seasonality-robust, null-splice clean). Now harden it: (a) IR-MAD + SFA-CD adapted to the
  sliding-window setting as STRONG baselines (a_hat must match/beat — IR-MAD owns detection elsewhere); (b) a
  NATURAL gradual change (deforestation / irrigation onset via GEE) instead of a splice; (c) multi-case (several
  splice points + a 2nd location pair) for consistency. Script builds on `temporal/experiments/ssds_longseries.py`.
- [ ] **SSDS-S2 (faithful Kanai signal-subspace DS on real S2 — earlier under-powered test).** Use the validated
  `temporal/signal_subspace_ds.py` (L0 passed AUC ~0.98). Learn the non-anomalous reference D_N from a NORMAL
  seasonal period of a cached/GEE S2 series; test whether departure-from-D_N detects the real event (fire onset
  / irrigation switch) WITHOUT firing on normal seasonal swings. Multivariate option: per-band or M-SSA signal
  subspace. Baselines: crude raw temporal-DS velocity (the version that FAILED), NDVI/index difference, SSA
  θ₁ + mean-angle. This is where "model-the-normal, flag-the-residual" earns its keep or dies. Construction
  ledger row REQUIRED before running.
- [ ] **2nd-order DS fair test.** Give it a genuine ACCELERATION signal (a change in the RATE of subspace
  motion, e.g. degradation that suddenly speeds up) — the case it was never given. Null: scalar 2nd-difference.
- [ ] **Distributional-change real test (was #1).** Real Salinas mean-preserving structural change vs proper
  covariance/Riemannian baselines (cov-Frobenius, log-Euclidean). Still queued; lower priority than SSDS-S2.
- [ ] **H-C-1 (geometry on richer features, stand-in).** Build per-pixel/patch subspaces from MNF/PCA-reduced
  features (foundation-model stand-in) vs raw bands on Salinas; DS/canonical-angle CD vs raw-band DS and vs
  SAM/CVA. Does a richer feature space make the subspace meaningful? Falsifier: no gain over raw-band DS.
- [ ] **H-A-real (blocked on data).** When a labeled bitemporal HSI-CD `.mat` is in `data_hsi/`: run
  `hsi_dimensionality` + SFA-CD / GDS-residual / DS vs SAM/CVA/IR-MAD, AUC. The decisive real test. SKIP if no
  new `.mat` present.

## Done (newest first)
- [x] H-B-1 trajectory curvature — NEGATIVE: redundant with mean-vector ||Δ²m||; 2nd-order DS never fired — 2026-06-20
- [x] H-B-1b structure+nuisance — C1 nuisance NEG; C2 mean-preserving distributional change = modest non-unique edge — 2026-06-20
- [x] H-A nuisance (SFA-CD held ~1.0 under affine while scalars collapsed; but established method, easy case) — 2026-06-19
- [x] H-A 1a hard-nuisance gating (IR-MAD holds → detection novelty closed; my IR-MAD has an additive-change bug) — 2026-06-19

## H-B status: LARGELY CLOSED (do not drill more synthetic H-B). Residual = distributional-change thread → needs Task B lit-check.
