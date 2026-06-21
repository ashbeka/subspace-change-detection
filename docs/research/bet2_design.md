# BET 2 — Multi-dimensional damage/recovery TRAJECTORY characterization: design + pre-registration

Created 2026-06-21, after Bet 1 closed (both pillars). Decomposition + math + pre-registered nulls/gate/falsifier
BEFORE coding, with an HONEST prior.

## 1. Problem statement
Given a multi-band time series over a disaster-affected area, characterize the RECOVERY TRAJECTORY — its
stage, rate, and spectral HETEROGENEITY (different materials/processes recovering on different spectral axes at
different rates) — not just a binary "recovered yet" flag. Geometry angle: the recovery is a trajectory of the
local spectral subspace on the Grassmann manifold (pre-disaster → damaged → recovering); 2nd-order DS / geodesic
/ distance-to-baseline characterize it. Deep motive: post-disaster reconstruction (damage vs recovery).

## 2. The honest decomposition — what must hold, and the two nulls
Recovery characterization needs to output something a SINGLE INDEX cannot. Two nested nulls:
- **Null A — scalar recovery index** (NBR/NDVI and its slope/recovery-rate): the STANDARD post-fire recovery
  monitor. A multi-dimensional descriptor must capture recovery structure this misses (e.g. two recoveries with
  the SAME NBR curve but DIFFERENT materials).
- **Null B — full multi-band per-band trajectory** (per-band recovery curves / multi-band distance-to-baseline):
  the "just use all the bands" null. **This is the killer:** the SUBSPACE/geometry must add over per-band
  multi-band, or its only contribution is "use multi-band not a single index" (true but not novel, not geometry).

So the gate is two-step: (1) does multi-dim beat the scalar index? (almost surely yes — more info), and (2) does
the SUBSPACE/geometry beat the full multi-band per-band descriptor? (the real falsifier).

## 3. HONEST PRIOR (do not oversell)
LOW for geometry-as-hero (~20-25%). Reasons: (a) my earlier temporal-trajectory geometry FAILED — H-B (2nd-order
DS redundant with mean-vector Δ²m), signal-subspace DS (failed on real S2); (b) across every Bet, the subspace
was redundant with per-band/correlation; recovery is a temporal trajectory, the same regime that failed.
The PROBLEM (multi-dim recovery characterization) is valuable and could be a strong APPLICATION even if the
descriptor is simple — but the GEOMETRY is unlikely to be the hero. Falsifier (gate 2) will decide quickly.

## 4. Pre-registered experimental program
- **G1 — recovery-TYPE gate (semi-synthetic on real spectra, validatable).** Two recovery types with MATCHED
  scalar index (NBR+NDVI) but different spectral paths/endpoints (R2 = R1 perturbed in non-index bands).
  Trajectory D → R over T dates. Classify type A vs B with: scalar-index features (≈0.5 by construction),
  full multi-band descriptor, SUBSPACE/geometric trajectory descriptor. **Falsifier:** if SUBSPACE does not beat
  the multi-band per-band descriptor, Bet 2's geometry angle dies (contribution reduces to "use multi-band").
- **G2 — path-vs-endpoint.** Matched ENDPOINT, different PATH (different per-band recovery rates / intermediate
  material). Does the trajectory GEOMETRY (2nd-order DS / geodesic) capture path differences the endpoint misses,
  beyond a per-band recovery-rate vector?
- **G3 — real data (Tenerife fire recovery, in hand; + a longer GEE recovery series).** Descriptive: does the
  multi-dim recovery trajectory reveal heterogeneity/structure the NBR curve misses? Honest, qualitative + any
  available quantitative anchor.
- **G4 — adversarial verification (workflow) if G1/G2 show a real geometry edge.**

## 5. Decision rule
G1 is the gate. If the subspace beats the multi-band per-band null on recovery-type, push G2-G4. If not (the
honest prior), document Bet 2's geometry angle as closed (multi-dim recovery is valuable but not subspace-
specific) and proceed to the H-C lead per the user's plan.
