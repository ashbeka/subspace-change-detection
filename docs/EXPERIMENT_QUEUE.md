# EXPERIMENT QUEUE — one item per autonomous (or manual) run

The auto-resume runner (`scripts/auto_resume.ps1`) runs the TOP unchecked item, commits, then checks it off
with a one-line result. Keep each item scoped to a SINGLE run. Priority reflects CONTINUITY §4/§6: H-A
*detection* is closed → lead with H-B (trajectory), then H-C, then the real-data tests when data lands.
Always report the trivial/standard null (SAM/CVA/IR-MAD/scalar 2nd-difference) beside the method.

## Up next
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
