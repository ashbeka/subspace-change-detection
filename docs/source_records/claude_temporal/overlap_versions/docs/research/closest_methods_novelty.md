# Closest existing methods & the novelty boundary — per candidate direction

Created 2026-06-21 (research-mining, `docs/CRUX_PROMPT.md` step 6). For each candidate, the **closest prior
work**, **how they justify/frame novelty**, the **baselines they compare against** (so we can mirror them), and
the **honest novelty boundary** (what is already done vs what is genuinely ours). Citations are from the KB +
web search; `[S]` claims should be verified against the primary PDF before being printed in a paper.

---

## Direction A — Change *characterization* via 1st/2nd-order DS + geodesic split (Rank 1)
- **Closest prior.**
  - *Second-order DS* (Fukui 2024, KB 01): defines the geodesic velocity/acceleration split — but only on 3-D
    shape + biometric SSA signals; **no satellite/CD application.**
  - *Kanai SSA-DS* (2023): DS between signal subspaces for time-series anomaly — **1-D signals, not satellite.**
  - *BFAST / CCDC* (Verbesselt; Zhu & Woodcock): break-vs-trend decomposition of optical time series — the
    *established* abrupt-vs-gradual tool, but **per-band harmonic regression, not subspace/manifold geometry.**
  - *Dynamic subspace tracking on the Grassmannian* (Blocker/Balzano `[S]`): subspace trajectory estimation —
    tracking, not change-*type* characterization for imagery.
- **How they justify novelty.** Fukui 2024: "first higher-order extension of DS; velocity/acceleration of a
  subspace series." BFAST/CCDC: "decompose trend+season+break for monitoring." → **Our novelty hook: first
  application of the 2nd-order DS geodesic decomposition to satellite image time series, as a change-*type*
  descriptor (abrupt/structural vs gradual/smooth) — and the first test of whether it adds over the scalar
  2nd-difference and over BFAST/CCDC break flags.**
- **Baselines to mirror.** BFAST/CCDC break-vs-trend; scalar $\|\Delta^2 m\|$ (magnitude+direction); NBR-slope
  recovery; the self-consistency check $\mathrm{Mag}(\mathcal D^2)\approx|\frac{d}{dt}\mathrm{Mag}(\mathcal D^1)|$.
- **Novelty boundary.** *Not novel:* DS, 2nd-order DS, subspace trajectories, abrupt-vs-gradual CD (BFAST/CCDC).
  *Genuinely ours:* the geodesic along/orthogonal split **as a change-type label on satellite series**, and the
  **falsifiable comparison** to the scalar/harmonic nulls. **Verdict: novel as a first-and-unique
  *characterization* trial (Sensei's bar), NOT as a detection-accuracy claim.**

## Direction B — Material/region-subspace HSI CD (MSM/GDS membership change) (Rank 2 positive)
- **Closest prior.**
  - *A Subspace-Based CD Method for HSI* (Wu/Du, IEEE 6459550 `[S]`): background-subspace **target-detection**
    residual — subspace CD for HSI **already exists**, but not as canonical-angle membership.
  - *Abundance-Indicated / endmember subspace* for HSI classification (`[S]`): "a material's signatures span a
    low-dim subspace" — **the material-as-subspace idea already exists in classification.**
  - *MSM / GDS* (Fukui, KB 01): set-as-subspace membership by canonical angles — image sets, not HSI CD.
  - *Sketched Multi-view Subspace Learning (SMSL)* HACD (`[S]`): subspace learning for anomalous change.
- **How they justify novelty.** Endmember-subspace papers: "handle intra-class spectral variability with limited
  samples." MSM: "set⇄set is more robust than image⇄image." → **Our hook: reframe CD as 'did this location leave
  its *material subspace*' (canonical-angle membership change), which is invariant to intra-material/illumination
  variation by construction, and — because HSI material subspaces are genuinely multi-dimensional — carries
  information SAM (rank-1) cannot.**
- **Baselines to mirror.** SAM-to-class-centroid, CVA, IR-MAD, the *covariance/Mahalanobis* null (the key one:
  a material is also a Gaussian, so a Mahalanobis-to-class null must be beaten by the *subspace*, not just the
  mean), spectral-unmixing CD.
- **Novelty boundary.** *Not novel:* subspace CD for HSI, material=subspace, MSM. *Possibly ours:* the
  **canonical-angle/DS membership-change** operator + the **multi-dimensionality threshold** + **attribution**.
  **Verdict: thin novelty margin and a likely-falsified detector** (the project's pattern: the row's scalar
  matched the subspace every time). Pursue only with the multi-dim threshold + attribution as the real
  contribution, and pre-register the Mahalanobis null.

## Direction C — RTW / warp-invariant subspace for phenological-phase-invariant CD (Rank 3, highest novelty)
- **Closest prior.**
  - *RTW + eGDA* (Hayashi/Souza/Fukui 2019; Hiraoka 2025, KB 02): warp-invariant sequence subspace — **gestures/
    actions, never RS/CD.**
  - *DTW / TWDTW for SITS* (Petitjean; Maus dtwSat `[S]`): time-warping for SITS **classification**; *"Multiannual
    CD in SITS based on DTW"* (`[S]`) — DTW-based **CD exists**, but aligns-then-compares (1-NN distance), not a
    warp-*invariant subspace*.
  - *BFAST/CCDC* with harmonic phase terms — the fixed-phase null.
- **How they justify novelty.** RTW: "model a sequence's temporal info as a low-dim subspace; warp-robust;
  ≡ self-attention." TWDTW: "robust to inter-year phenological distortion." → **Our hook: first warp-*invariant
  subspace* representation for satellite time series, detecting a change in the *shape* of the seasonal cycle
  while being invariant to its *phase/timing* — the nuisance that defeats fixed-phase deseasonalization.**
- **Baselines to mirror.** Harmonic deseasonalization **with a phase term**, TWDTW-distance CD, phenology-metric
  (SOS/EOS/amplitude) differencing, SFA-CD.
- **Novelty boundary.** *Not novel:* time-warping in RS, DTW-CD, RTW itself. *Genuinely ours:* RTW's
  warp-*invariance* as a **CD** mechanism + the phase-shift-vs-shape-change distinction. **Verdict: the most
  genuinely novel and Sensei-endorsed lever, but data-blocked + the phenology detector was burned once.** Needs a
  controlled phase-shift test to prove the invariance is not just harmonic deseasonalization in disguise.

## Direction D — Geometry on deep/foundation features (SLS-style, H-C) (Rank 4)
- **Closest prior.** *SLS / SLS-PGM* (Mahyub 2024 `[S]`): subspaces from deep latent features — sound, not RS.
  *Deep-feature change detection* (Siamese/transformer CD): dominant but black-box. *CGLCS-Net* (`[S]`):
  "subspace-based self-attention" module **inside** a CD net — subspace-as-component already appears.
- **How they justify novelty.** SLS: "rich features make the subspace meaningful + small-sample robust." → **Our
  hook: subspaces from a frozen RS foundation model's features, compared by DS/canonical angles — illumination-
  robust, label-efficient, interpretable geometry on top of strong features (first RS application).**
- **Baselines.** The foundation model's own cosine/feature-diff CD; Siamese CD; linear-probe.
- **Novelty boundary.** *Not novel:* deep-feature CD, subspace modules in nets. *Possibly ours:* DS/GDS on frozen
  foundation-features as a *label-free, interpretable* CD head. **Verdict: modern and defensible but engineering-
  heavy and the contribution risks being "the foundation model did the work."** Defer behind A/B.

## Direction E — Invariance-residual subspace CD (SFA/GDS-common) (CLOSED for detection)
- **Closest prior.** SFA-CD/ISFA, IR-MAD, the **whole ACD family** (chronochrome, covariance equalization,
  whitened-TLSQ≡CCA, SFA-ACD, SMSL — KB 03). **All coordinate/affine-invariant, model-background-flag-residual.**
- **Novelty boundary.** **Closed.** The invariance-detection cell is occupied top-to-bottom with strong methods;
  the project confirmed IR-MAD owns affine/nonlinear/spatial nuisance (AUC 0.98–1.00). *Survives only as
  attribution* (which directions are nuisance vs change — the chi-square lacks this). **Verdict: do not pursue as
  a detector; fold the attribution remnant into A.**

---
### Cross-cutting novelty lesson
Every candidate's *detection* novelty is thin or closed, because the closest prior already occupies the
detection cell. The **durable, defensible novelty across A–D is the same shape: subspace geometry as a
*characterization/attribution/invariance-representation* layer that produces something the scalar null
structurally cannot** (a change-*type*, a named direction, a warp-invariant representation) — not a higher AUC.
This is also exactly Sensei's bar ("first and unique trial, need not be top performance").
### Honest novelty verdict
A (characterization) and C (RTW warp) have the cleanest "first and unique" stories; B and D have crowded
neighbours; E is closed. 
### Next falsifiable step
For the chosen task, write the exact null that, if it matches the subspace, kills the novelty — and run it FIRST
(the project's discipline rule, burned 4×). Detailed per-task in `synthesis_specific_tasks.md`.
