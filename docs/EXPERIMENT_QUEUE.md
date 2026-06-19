# EXPERIMENT QUEUE — one item per autonomous (or manual) run

The auto-resume runner (`scripts/auto_resume.ps1`) runs the TOP unchecked item, commits, then checks it off
with a one-line result. Keep each item scoped to a SINGLE run. Priority reflects CONTINUITY §4/§6: H-A
*detection* is closed → lead with H-B (trajectory), then H-C, then the real-data tests when data lands.
Always report the trivial/standard null (SAM/CVA/IR-MAD/scalar 2nd-difference) beside the method.

## Up next
- [ ] **H-B-1 (synthetic trajectory: abrupt vs gradual).** Build a multi-date (≥5) HSI pixel series with (a)
  abrupt step change and (b) gradual ramp change of matched total magnitude. Compute 1st-order DS velocity and
  2nd-order DS acceleration (temporal/subspace.py: `second_order_*`) + geodesic along/orthogonal split.
  FALSIFIER: 2nd-order DS must separate abrupt from gradual where a scalar 2nd-difference of |x_t−x_{t-1}|
  CANNOT. Report ROC/separation vs the scalar null. Write `docs/experiment_reports/HB1_trajectory_*.md`.
- [ ] **H-B-2 (degradation vs recovery via distance-to-baseline).** Use `temporal/dynamics.py`
  `distance_to_baseline`: synthesize a degrade-then-recover series; test whether 2nd-order DS / geodesic
  curvature distinguishes recovery from continued degradation where velocity sign alone is ambiguous. Null:
  scalar d_pre slope. (The Gaza-recovery motive.)
- [ ] **H-B-3 (real S2 trajectory sanity).** On a cached S2 series (Tenerife recovery / irrigation), does the
  geodesic along/orthogonal split suppress the seasonal cycle that killed plain temporal DS? Null: raw NDVI
  trajectory. Honest verdict — this is the known hard case.
- [ ] **H-C-1 (geometry on richer features, stand-in).** Build per-pixel/patch subspaces from MNF/PCA-reduced
  features (foundation-model stand-in) vs raw bands on Salinas; DS/canonical-angle CD vs raw-band DS and vs
  SAM/CVA. Does a richer feature space make the subspace meaningful? Falsifier: no gain over raw-band DS.
- [ ] **H-A-real (blocked on data).** When a labeled bitemporal HSI-CD `.mat` is in `data_hsi/`: run
  `hsi_dimensionality` + SFA-CD / GDS-residual / DS vs SAM/CVA/IR-MAD, AUC. The decisive real test. SKIP if no
  new `.mat` present.

## Done (newest first)
- [x] H-A nuisance (SFA-CD held ~1.0 under affine while scalars collapsed; but established method, easy case) — 2026-06-19
- [x] H-A 1a hard-nuisance gating (IR-MAD holds → detection novelty closed; my IR-MAD has an additive-change bug) — 2026-06-19
