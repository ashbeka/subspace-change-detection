# Open challenges in hyperspectral × temporal × change detection — RANKED by fit to our tools + data

Created 2026-06-21 (research-mining, `docs/CRUX_PROMPT.md` step 5). Sources: the multitemporal-HSI-CD reviews
(KB 03), the ACD literature, the project's own hard findings (`docs/CONTINUITY.md`). Ranking criterion:
**fit = (does subspace geometry have a *structural* advantage here?) × (is there real data to test it?) ×
(is the cell genuinely open?) × (Sensei "novel first trial" fit) — minus (is the cell already owned?).**

Each challenge: the gap · why geometry might fit · the trivial/standard null that already addresses it ·
the honest verdict.

---

## Rank 1 — C6: Change *characterization* beyond binary (abrupt-structural vs gradual-smooth vs seasonal; trajectory; recovery)
- **Gap.** Almost all CD outputs a binary/score map. *Which kind* of change (abrupt regime break vs gradual
  drift vs seasonal oscillation vs recovery) is rarely produced, yet it is what a disaster/reconstruction analyst
  actually needs (damage onset vs slow recovery).
- **Why geometry fits (structural).** The 2nd-order DS **geodesic along/orthogonal decomposition** (KB 01) is a
  *2-D* descriptor: along-geodesic = smooth speed change, orthogonal = off-path structural break. A subspace
  trajectory on the Grassmann manifold *is* the natural object for "velocity/acceleration of a scene." This is a
  decomposition a scalar cannot output.
- **Trivial null.** Scalar 2nd-difference of the mean spectrum $\|\Delta^2 m\|$ (has a magnitude *and* a
  direction); BFAST/CCDC break-vs-trend flags; the sign of an NBR slope (recovery).
- **Verdict.** **Best fit.** Sensei's flagship (2nd-order DS), first satellite application = maximal novelty,
  testable on the **real S2 series already in hand** (abrupt fire vs gradual reservoir/phenology), and it
  **sidesteps the falsified "better detector" claim** — it asks "characterize," not "detect better." Risk:
  redundancy with $\|\Delta^2 m\|$ direction (untested for the *faithful* acceleration signal — the open question).

## Rank 2 — C1: Exploit the high spectral dimensionality (the DS ≠ SAM threshold)
- **Gap.** Reviews repeatedly name "how to use 100–300 bands without band-redundancy noise" as the core HSI-CD
  problem. Band selection throws information away; a subspace *uses* the redundancy as structure.
- **Why geometry fits (structural).** First-order DS ≡ SAM **only** when the spectrum is low-rank; at 200+ bands
  the material/region subspace is genuinely multi-dimensional (verified: Salinas 204-band, DS⊥SAM corr −0.03,
  SSA-subspace rank>1). This is the *one* condition the low-dim S2 failures lacked.
- **Trivial null.** SAM, CVA, IR-MAD on the full band set (CVA 0.96 ≥ SAM 0.92 ≫ DS 0.70 on Salinas *class
  proxy* — already negative once).
- **Verdict.** **Strong fit, testable on REAL labels** (Bay Area/Hermiston bitemporal HSI), but the one direct
  probe so far was **negative** (DS lost to SAM/CVA even at 204 bands on the class proxy). Salvage = test on the
  *true* bitemporal benchmark, not the class proxy, and report the DS−SAM gap vs band count as the falsifiable
  claim. If still negative, C1-detection is fundamentally closed (a clean, publishable diagnostic result).

## Rank 3 — C3′: Phenological-*phase*-invariant detection of cycle-*shape* change (the warp rung)
- **Gap.** The recurring project killer: seasonality/phenology dominates and the standard null (harmonic
  deseasonalization, fixed phase/period) handles the *common* case — but it **false-alarms on a phase shift**
  (the same cycle arriving earlier/later year-to-year or along a geographic gradient). Detecting a change in the
  *shape* of the seasonal cycle while ignoring *when* it happens is genuinely under-addressed.
- **Why geometry fits (structural).** **RTW** (KB 02) builds a **warp-invariant subspace** of a sequence; a
  change in cycle-shape moves the subspace, a pure phase shift does not. This is an invariance the scalar null
  structurally lacks. Sensei explicitly endorsed RTW.
- **Trivial null.** Harmonic deseasonalization **with a phase term**; DTW/TWDTW align-then-compare CD; phenology-
  metric (SOS/EOS/amplitude) differencing.
- **Verdict.** **Highest genuine novelty** (RTW never used in RS/CD) and attacks the exact failure mode — but
  **data-blocked** (no labeled HSI/phenology-change time series) and the phenology *detector* angle was already
  burned once (SSDS phenology 0/3). Promote only if a phase-shift-vs-shape-change *controlled* test on S2 +
  synthetic separates RTW from a phase-aware harmonic model. High risk, high reward.

## Rank 4 — C5: Attribution — which bands / materials / dates drive the change
- **Gap.** CD maps say *where*, rarely *which spectral mechanism* (red-edge vs SWIR-moisture vs mineral) or
  *which date interval*. Interpretable CD is a hot, publishable angle and the part of subspace methods that
  **worked where detection didn't** (per-band DS-basis energy 100% on synthetic).
- **Why geometry fits.** The DS **basis** names the change directions; **S3CCA/TRCCA** (Sensei-mandated) give a
  smooth, contiguous band/date interval. Scalars give a magnitude, not a named direction.
- **Trivial null.** Per-band difference magnitude ranking; PCA-loading inspection; SHAP on a classifier.
- **Verdict.** **Robust, defensible, but a *secondary* contribution** (weak as a standalone headline). Best as a
  *component* of Rank 1/2 (the thing geometry adds on top of detection).

## Rank 5 — C4: Subtle / sub-pixel / mean-preserving distributional change
- **Gap.** Mixed-pixel / partial-damage change shifts the local spectral *distribution* while the per-pixel mean
  holds — invisible to mean-based CD.
- **Why geometry fits.** A patch subspace captures distributional structure the mean discards.
- **Trivial null.** Local covariance-Frobenius / RX; second-moment statistics.
- **Verdict.** **Real but non-unique** (ordinary covariance also catches it; not scale-invariant — `CONTINUITY`
  §2/Top-10 #C). Keep as a fallback / part of the diagnostic, not a headline.

## Rank 6 — C2: Labeled-HSI-time-series scarcity (a meta-challenge, not a method gap)
- **Gap.** No public labeled hyperspectral *time series* → temporal/SFA/2nd-order claims are validated on
  synthetic + S2 multispectral only; only *bitemporal* HSI has real labels.
- **Why it matters.** It **constrains every ranking above**: C6/C3′ (temporal) are only synthetic+S2-testable;
  C1/C5 (bitemporal) are the only real-label-testable HSI cells. This is why the *most defensible* positive shot
  is bitemporal (C1/C5) and the *highest-novelty* shot (C6/C3′ temporal) is the riskier one.
- **Verdict.** Not a research target itself, but the **binding constraint** on which target is fundable. Could
  itself become a contribution (a small curated labeled HSI/S2 change-characterization set) — but data work,
  low method-novelty.

---
### Honest novelty verdict
The reviews' challenges geometry can *uniquely* serve are **C6 (characterization), C5 (attribution), and the
C3′ warp rung**; the detection challenges (C1-as-leaderboard, C3-pseudo-change) are **owned** by IR-MAD/SFA/ACD.
The ranking therefore inverts the project's original instinct: lead with **characterization (C6) + the
dimensionality diagnostic (C1)**, not with a detection win.
### Next falsifiable step
Pick the Rank-1 task and pre-register its falsifier (does the geodesic split beat $\|\Delta^2 m\|$ direction at
change-*type* classification on the real S2 abrupt-vs-gradual cases?) — detailed in `synthesis_specific_tasks.md`.
