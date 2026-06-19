# Top 10 Research Directions — ranked, defensible, publishable

Created 2026-06-19. Purpose: collapse a year of scattered ideas into 10 focused problem statements, ranked
most-convincing-and-defensible → least. Ranking weights, in the order the student asked: **(1) Sensei's
instructions, (2) senpai advice, (3) the student's notes, (4) my research/evidence** — but always gated by
*defensibility* (will it survive scrutiny, is there real labeled data, is the novelty boundary clear) and
*curiosity* (is it genuinely interesting). Where Sensei's favorite and "what the evidence says will work"
disagree, both are stated honestly.

## The two meta-findings that constrain every direction
- **A. Dimensionality.** Verified this project: first-order DS ≡ spectral angle for low-dim spectra (13-band
  S2). At **128–242 bands** (AVIRIS/Hyperion HCD benchmarks, real labels) the spectral subspace is genuinely
  multi-dimensional → DS/CCA/SFA can finally beat a 1-D angle. **This is the single biggest reason to expect
  a different outcome than the S2 failures.** It is a *hypothesis to pre-register*, not a guarantee (synthetic→
  real burned us 3×).
- **B. Invariant-subspace nuisance removal.** All 8 papers reduce to: build the *common/invariant/slow*
  subspace, project it out, change = the residual. SFA (slow), GDS (common), 2nd-order DS (along-geodesic),
  S3CCA (structured-sparse window), TRCCA (orthogonalized background), Kanai (reference normal-DS). This is the
  cure for "seasonality/oscillation swamps DS."
- **Data reality.** Labeled hyperspectral CD = **bitemporal** (Bay Area/Santa Barbara AVIRIS 224-band;
  Hermiston/River/Wetland Hyperion 128–198-band; pixel labels, some multiclass). Labeled hyperspectral **time
  series do not exist publicly** → temporal/SFA-seasonal directions are validated on synthetic + S2; the
  bitemporal-testable directions are the most *defensible*. None are on GEE → local download + numpy/rasterio.

---

## #1 — First-order Difference Subspace of the spectral *signal* for hyperspectral CD (the dimensionality-threshold result)
- **Problem.** Does treating a hyperspectral pixel's 300+-band spectrum as a *signal* and building a genuinely
  multi-dimensional spectral subspace let the full Difference Subspace (multi-angle) beat the spectral-angle/
  CVA/PCA-diff baselines it ties or loses to at 13 bands — i.e. is there a spectral-dimensionality *threshold*
  where subspace geometry starts to help?
- **Method.** Per pixel/region, build the spectral subspace by SSA delay-embedding along wavelength (Hankel over
  bands) or band-group covariance; DS magnitude `2·Σ(1−cosθ)` between dates; vs spectral angle (= min-angle),
  CVA, PCA-diff, IR-MAD. Report DS-vs-angle gap **as a function of band count** (subsample 13→50→224).
- **Data.** Bay Area / Santa Barbara (224-band AVIRIS, binary labels) + Hermiston (multiclass). Real, instant download.
- **Why novel + defensible.** Validates *my* hard-won finding ("DS=angle unless multi-dim") as a clean,
  falsifiable, mechanistic claim on real labels. Not "apply DS to HSI" (done) — the contribution is the
  *dimensionality threshold* + which construction. Testable this week.
- **Serves.** Sensei (DS, his core), my research (the finding), curiosity (the signal reframe).
- **Risk.** Subspace HSI-CD exists → novelty must be the threshold + attribution, not the method alone. First
  check: does the spectral subspace have intrinsic rank >1 (else it collapses again).
- **First experiment.** L0 on AVIRIS: DS vs spectral-angle AUC vs band-count curve.

## #2 — SFA × Difference Subspace: change as the residual outside a *learned invariant* subspace
- **Problem.** Replace the symmetric difference `x−y` (which the seasonal/illumination cycle inflates) with a
  *learned invariant subspace*: does measuring change as the energy that *leaks out* of the SFA-slow / GDS-common
  subspace beat magnitude-based DS and the strong scalar baselines, on real hyperspectral CD?
- **Method.** SFA-CD (Wu/Du/Zhang): generalized eigenproblem `A W = B W Λ` (A = bi-date difference cov, B = signal
  cov), chi-square `Σ D_j²/σ_j²` weighting the slow/invariant components; ISFA iterative reweighting (IR-MAD
  analogue). DS interpretation: DS spans the *change* directions, SFA-slow spans the *invariant* directions —
  combine ("DS relative to the SFA-invariant background"). Optional KSFA (kernel).
- **Data.** Bitemporal AVIRIS/Hyperion (SFA-CD is bitemporal — directly testable on real labels). Seasonal-time-
  series extension on synthetic + S2.
- **Why novel + defensible.** The DS↔SFA *duality* (change-subspace vs invariant-subspace) is a genuine
  conceptual contribution; bitemporal → real labels. Directly cures the documented seasonality failure (fix B).
- **Serves.** Sensei (pointed to SFA explicitly), senpai (Beleza/Kobayashi SFA), my research (the seasonality fix).
- **Risk.** SFA-CD itself exists → novelty is the DS-fusion + hyperspectral dimensionality + attribution. Gradual
  change can land in the slow subspace (helps abrupt > gradual).

## #3 — Second-order Difference Subspace + geodesic decomposition for change-*type* (smooth/seasonal vs abrupt/structural)
- **Problem.** Can the second-order DS (Grassmann acceleration) — split into along-geodesic (smooth = seasonal)
  vs orthogonal-to-geodesic (abrupt = structural) — detect structural change while *suppressing* seasonal motion,
  and double as a change-type classifier?
- **Method.** Fukui 2024: `D2(S1,S2,S3)=D(S2, Karcher-mean(S1,S3))`; orthogonal-to-geodesic component as the
  structural-change score; validate `Mag(D2) ≈ |d/dt Mag(D1)|`. Substrate: spectral-SSA subspace trajectory
  (single hyperspectral image) or multi-date S2/MultiSenGE.
- **Data.** Spectral-SSA on AVIRIS/Hyperion; multi-date on S2/MultiSenGE (synthetic for the seasonal claim).
- **Why novel + defensible.** **Sensei's flagship 2024 paper, first application to satellite — maximal "first and
  unique" novelty + theoretical depth he explicitly rewards.** The along/orthogonal split turns the earlier
  failure mode (oscillation ≈ change) into the discriminative signal.
- **Serves.** Sensei (highest — his own paper), curiosity (velocity/acceleration of a scene on a manifold).
- **Risk.** Highest-novelty but the geodesic-suppresses-seasonality claim is *unproven* (paper flags the proof as
  future work; a year-long cycle is only *locally* geodesic). Needs ≥3 dates. Pre-register a falsifier.

## #4 — Structured-sparse / temporally-regularized CCA for *attributable* change (which bands & dates) — Sensei-mandated
- **Problem.** Can S3CCA/TRCCA learn a smooth+contiguous *partial window* over wavelength (which absorption bands)
  and over dates (when), giving an interpretable, attributable, label-free change map?
- **Method.** S3CCA (Kobayashi): CCA between two feature arrays with a Laplacian-smoothness + overlapping
  structured-sparsity penalty on the *signal axis* (wavelength or date); min canonical angle = match score, the
  sparse weights = the contiguous band/date interval driving the change. TRCCA adds temporal smoothness +
  orthogonalization of the shared background.
- **Data.** Hermiston (5-class) for band-attribution-vs-physics; bitemporal AVIRIS.
- **Why novel + defensible.** **Sensei explicitly told the student to research S3CCA and TRCCA for satellite** —
  direct mandate. Contribution is *interpretability/attribution* (which bands/dates), not raw detection.
- **Serves.** Sensei (explicit mandate, high weight), my notes (band-contribution / "which bands detect which change").
- **Risk.** The min-angle score alone stays rank-1 (doesn't beat detection); the value is attribution — frame it
  as the interpretability contribution, combine with #1's multi-angle DS for detection.

## #5 — Kernel Difference Subspace (KDS/KGDS) for nonlinear hyperspectral CD
- **Problem.** Does a nonlinear (RKHS) difference subspace capture nonlinear spectral-mixing / material-manifold
  change that linear DS and KPCA-CD miss, on high-band hyperspectral?
- **Method.** Fukui-Maki TPAMI 2015 KDS/KGDS: kernel PCA subspaces + kernel canonical angles + difference subspace
  in feature space; the Venus demo code in `phase1/subspace/kernel_difference_subspace.py` ports over.
- **Data.** Bitemporal AVIRIS/Hyperion; Venus as the method-verification fixture.
- **Why novel + defensible.** **Sensei explicitly wants KDS run** (his TPAMI flagship). Hyperspectral gives the
  kernel genuine room.
- **Serves.** Sensei (explicit), senpai (Santos toolkit).
- **Risk.** KPCA-CD is established → novelty must be the *difference-subspace-in-RKHS* + dimensionality, not the
  kernel alone. Memory-heavy at scene scale (needs sampling/Nyström).

## #6 — Semantic / band-group change: which spectral subbands drive which change-*type*
- **Problem.** From the DS/SFA basis, recover a *physically meaningful* grouping of spectral subbands per change
  type (vegetation red-edge vs SWIR moisture vs mineral absorption), turning a change map into an explanation.
- **Method.** Per-band DS-basis energy (verified 100% on synthetic) + band-group subspaces; validate the recovered
  bands against known physics for each labeled transition.
- **Data.** Hermiston (5 crop-transition classes), Wetland (multiclass) — real multiclass labels.
- **Why novel + defensible.** Attribution is the *robust* part of the method (it worked where detection didn't);
  multiclass labels make it testable; interpretable CD is a hot, publishable angle.
- **Serves.** My notes (the explicit "build a dataset to clarify change type / which bands" idea), curiosity
  (semantic meaning attached to band separation), the Gaza motive (which process is degrading).
- **Risk.** Needs the detection to carry *some* signal first (pair with #1).

## #7 — Anomalous (nuisance-invariant) change detection via difference subspace
- **Problem.** Separate *pervasive* change (illumination/seasonal/atmospheric — nuisance) from *anomalous*
  change (a new object/material) using a subspace formulation of anomalous-change-detection (ACD).
- **Method.** Subspace ACD: predict date-2 from date-1 within the common subspace (chronochrome/DS analogue);
  residual outside it = anomalous change. Connects to Kanai's reference-normal-DS (#3 paper) and SFA-invariant.
- **Data.** Viareggio 2013 (airborne, 127-band, designed for ACD) — gated access; or self-built.
- **Why novel + defensible.** ACD is an established, well-defined task with a benchmark; a subspace/DS ACD is
  novel and lab-aligned (Sensei's anomaly-DS lineage).
- **Serves.** Sensei (anomaly-DS), the pseudo-change-vs-real-change theme in the notes.
- **Risk.** Viareggio is airborne + gated; ACD methods are competitive.

## #8 — Product Grassmann Manifold fusion of spectral + spatial + temporal subspace factors
- **Problem.** Fuse independent subspace factors (spectral signal, spatial patch, temporal sequence) as one point
  on a product manifold for change detection that preserves spatial info (Sensei's original criticism) without
  flattening pixels into a fake image-set.
- **Method.** SLS-PGM (Mahyub 2024): per-factor subspace, product-manifold similarity `√Σ s_j²`, GDS per factor.
- **Data.** Hyperspectral (spectral+spatial factors); +S2/MultiSenGE for the temporal factor.
- **Why novel + defensible.** Senpai's (Mahyub, labmate) method, first satellite application; directly answers the
  spatial-information criticism that started the whole reset.
- **Serves.** Senpai (Mahyub/SLS), Sensei (spatial-info concern, PGM/Grassmann).
- **Risk.** Spectral/spatial factors are correlated → the "independent factors / graceful degradation" selling
  point weakens; fusion benefit unproven; complex.

## #9 — Abandoned-greenhouse / agricultural-state monitoring via subspace state-change descriptors (the applied / reconstruction-adjacent product)
- **Problem.** Detect state change of a known object (greenhouse active→abandoned; field cultivated→fallow) as a
  change in its multispectral/temporal subspace descriptor, where manual mapping is the current bottleneck.
- **Method.** Object/parcel subspace state descriptor (SFA/DS over its time series); state-change = subspace
  rotation; semantic-segment-then-CD.
- **Data.** Ramdani-lab greenhouse data (needs sourcing); Global-PCG-10 greenhouse map; IrrMapper for fallow.
- **Why novel + defensible.** Real-world product the student cares about (Ramdani collaboration, Gaza-reconstruction
  motive); "even VLMs struggle, done manually" = a real gap.
- **Serves.** My notes (greenhouse use case), the deep motivation (useful product for reconstruction), curiosity.
- **Risk.** Data/labels not in hand → application-driven, less method-novelty; defer until a method (#1–#3) works.

## #10 — Diagnostic: *when* does subspace geometry help satellite change detection? (the honest, guaranteed-publishable failsafe)
- **Problem.** Characterize the regimes where subspace methods beat / tie / lose to simple baselines: low-dim vs
  high-dim spectra, amplitude vs structural change, abrupt vs gradual, seasonal-confounded vs not.
- **Method.** The full evidence: synthetic multi-mode positive + attribution + the 3 real S2 negatives + the
  hyperspectral dimensionality result, as a decision map.
- **Data.** Everything already produced + the hyperspectral benchmarks.
- **Why novel + defensible.** Most honest, complete, *certain* outcome; meets Sensei's "first and unique trial"
  bar even as a negative/characterization; protects against all the other risks.
- **Serves.** My research (the rigorous record), the founding question ("are we forcing subspaces?").
- **Risk.** Lower curiosity; a negative/diagnostic is a harder top-venue sell — but it's the safety net under #1–#9.

---

## The unifying thread (and the recommended start)
All ten are facets of one program: **"Hyperspectral imagery as a high-dimensional spectral *signal*, represented
by subspaces, with change measured as the residual outside a learned invariant subspace."** The dimensionality
fix (#1) and the invariant-subspace cure (#2/#3) are the load-bearing ideas; #4–#8 are constructions of the
invariant subspace or the comparison; #6 is the interpretability payoff; #9 the application; #10 the failsafe.

**Recommended start: #1 then #2**, because both are *testable on real labeled hyperspectral this week*, both
validate the two meta-findings, and they de-risk everything above them. Run #1's dimensionality-threshold
experiment on Bay Area/Hermiston first; if DS genuinely beats spectral-angle at 224 bands (where it tied at 13),
the whole program is alive and #3 (Sensei's flagship 2nd-order DS) becomes the high-novelty headline. If it does
*not*, #10 is the honest paper and we know the wall is fundamental, not fixable by dimensionality.
