# Synthesis — ranked specific candidate tasks + full plan for the top one

Created 2026-06-21 (research-mining, `docs/CRUX_PROMPT.md` step 8). Inputs: KB (`docs/kb/`), the notes
(`E:/research_projects/subspace-change-detection/notes/` + `source_records/`), the niche reviews + closest-method
analysis (`closest_methods_novelty.md`), the ranked challenges (`challenges_ranked.md`), and the project's hard
findings (`docs/CONTINUITY.md`). Written to be blunt and independent per the standing feedback.

## 0. The two facts that decide everything
1. **The detection lever is closed.** Across 4+ real-data trials, subspace detectors lost to a scalar in the
   same row (SAM/CVA/IR-MAD/harmonic deseasonalization), and the invariance-detection cell is occupied
   top-to-bottom by IR-MAD/SFA/the ACD family (with citations, KB 03). **Do not propose "a subspace detects
   change better."** That is the path that has failed every time.
2. **Two structural openings remain**, both *characterization/representation*, not detection: **(i)** what a
   scalar/covariance structurally *cannot output* — a change-**type**, a named change **direction**, a
   warp-**invariant** representation; **(ii)** the one regime where DS ≠ SAM by construction — **genuine
   multi-dimensionality** (100+ bands, or a temporal/SSA embedding). Every viable task lives in this intersection.

**Deadline reality:** ACCV2026 = **2026-07-05 (~2 weeks)**. New data acquisition or heavy engineering is
infeasible in that window. This splits the recommendation into *what to submit now* vs *the novel task to pursue*.

---

## 1. The ranked tasks

| # | Task | Lever | Open? | Sensei fit | Testable NOW (data in hand)? | Risk | Rank |
|---|---|---|---|---|---|---|---|
| **T1** | **Geometric change-TYPE characterization** via 1st/2nd-order DS + geodesic along/orthogonal split on S2 time series | C6 characterization | ⬜ open (no satellite app) | **flagship** (his own 2nd-order DS) | **yes** (real S2 fire/reservoir/phenology already fetched) | redundancy with scalar Δ²m | **1** |
| **T2** | **Phenological-phase-invariant CD** via RTW warp-invariant subspaces (cycle-shape change, phase-blind) | C3′ warp invariance | ⬜ open (RTW never in RS) | high (Sensei endorsed RTW) | partial (S2 yes; labeled phenology-change no) | data + may collapse to phase-aware harmonic | **2** |
| **T3** | **Material-subspace bitemporal HSI CD** + the DS≠SAM **dimensionality threshold** + DS-basis attribution | C1 dimensionality | ◑ thin (subspace-CD/HSI exists) | core (DS/MSM/GDS) | only if a bitemporal HSI `.mat` is downloaded | thin novelty; likely-redundant detector | **3** |
| **T4** | **The diagnostic** — when/why does subspace geometry help/equal scalar CD (the full pre-registered ladder) | meta | ✅ supported | meets bar (novel first characterization) | **yes** (evidence in hand) | a negative result is a harder top-venue sell | **failsafe** |
| **T5** | **DS/GDS on frozen RS foundation features** (SLS-style, H-C) | deep-feature geometry | ⬜ open | roadmap endpoint (subspace⇄DNN) | no (needs extractor + engineering) | "the foundation model did the work" | defer |

**Blunt recommendation.**
- **Submit now (ACCV2026):** **T4 (the diagnostic) as the spine + T1 as its positive section.** T4 is done and
  guaranteed-publishable; T1 is the one *novel positive* runnable on data already in hand, and even its *failure*
  mode (redundancy) feeds T4. Do **not** bet the 2-week paper on a brand-new detection win.
- **The genuinely novel task to pursue (deep-research energy, post-/beyond-ACCV):** **T2 (RTW warp-invariant
  CD).** It is the only un-burned cell with a *structural* invariance the standard null lacks, and Sensei
  explicitly asked to incorporate RTW. Gate it on solving the phase-shift controlled test (below).
- **The most data-defensible positive shot (if a bitemporal HSI benchmark lands):** **T3**, but only with the
  threshold + attribution as the contribution, not a leaderboard claim.

The full plan below is for **T1** (the top *runnable, novel, Sensei-aligned* task). T2's go/no-go test is in §3.

---

## 2. FULL PLAN — T1: Geometric change-type characterization on satellite image time series

### 2.1 Precise problem statement
Given a per-region multi-band Sentinel-2 (or, later, HSI) **time series**, do not just flag *whether/where* a
change occurred — **characterize its TYPE**: abrupt/structural break vs gradual/smooth drift vs seasonal
oscillation vs recovery, and **attribute** it to spectral modes. Concretely: represent the region's dynamics as
a **subspace trajectory** $S(0),S(1),\dots$ on the Grassmann manifold (via multivariate SSA), and use the
**first/second-order Difference Subspace and its geodesic along/orthogonal decomposition** as the change-type
descriptor. Test, on real documented events, whether this geometric decomposition provides a change-type signal
that the scalar nulls (2nd-difference of the mean spectrum; BFAST/CCDC break-vs-trend) **cannot**, and where it
is merely redundant with them.

### 2.2 Why it is novel (vs the closest methods — `closest_methods_novelty.md` Direction A)
- 2nd-order DS / geodesic split (Fukui 2024): defined only on 3-D shape + biometric signals → **first satellite
  application** (Sensei's exact "first and unique trial").
- BFAST/CCDC: established abrupt-vs-gradual decomposition but **per-band harmonic regression**, not subspace/
  manifold geometry that captures *joint cross-band* structure. The novelty is the **manifold-geometric**
  change-type decomposition + the honest map of when it adds over per-band harmonics.
- The contribution is **characterization + attribution**, NOT a detection-AUC claim → it sidesteps the closed
  detection lever entirely.

### 2.3 Method (math — all implemented/verified, `docs/METHOD.md`, `SUBSPACE_CONSTRUCTION_LEDGER`)
1. **Construction.** Region $R$, $B$-band index series $x_b(t)$, window $W$. Per-band Hankel $H_b\in
   \mathbb{R}^{L\times(W-L+1)}$; stack bands $H=[H_1;\dots;H_B]\in\mathbb{R}^{BL\times\cdot}$; **signal subspace**
   $S(t)=$ top-$k$ left singular vectors of $H$ (energy-based rank). Sliding $W$ → trajectory $S(t)$. (M-SSA makes
   $S(t)$ genuinely multi-dimensional — the property the whole thing needs; DS≠min-angle here, verified.)
2. **Velocity (1st-order DS).** $d_1(t)=\mathrm{Mag}(\mathcal D(S(t),S(t{+}\tau)))=\sum_i 2(1-\cos\theta_i)$.
3. **Acceleration (2nd-order DS).** $d_2(t)=\mathrm{Mag}(\mathcal D(S(t{+}\tau),\mathcal M(S(t),S(t{+}2\tau))))$,
   $\mathcal M$ = Karcher mean (eigvecs of $P_t+P_{t+2\tau}$, eigenvalue $>1$). $d_2=0$ ⇔ constant-velocity.
4. **Geodesic decomposition (the change-TYPE descriptor).** $\mathrm{Mag}(\mathcal D^2)\simeq
   \underbrace{\mathrm{Mag}(\mathcal D(S(t{+}\tau),\mathcal W(S(t),S(t{+}2\tau))))}_{\text{orthogonal = off-path
   STRUCTURAL break}} + \underbrace{\mathrm{Mag}(\mathcal D(\omega(S(t{+}\tau)),\mathcal M))}_{\text{along =
   smooth SPEED change}}$. **Change-type label:** orthogonal-dominant ⇒ abrupt/structural; along-dominant ⇒
   gradual/smooth (exactly the walking-vs-jumping result of Fukui 2024).
5. **Recovery axis.** $d_{\mathrm{pre}}(t)=\mathrm{Mag}(\mathcal D(S(t),S_{\mathrm{pre}}))$ — rising = degrading,
   falling = recovering (guard B8/B12↔NBR circularity).
6. **Attribution.** Per-band DS-basis energy $\sum_{\text{rows }b}\|\text{DS basis}\|^2$ → which bands/modes.

### 2.4 Fair baselines (the nulls — run FIRST, per the discipline rule)
- **Scalar mean-spectrum dynamics:** $\|\Delta m(t)\|$ (velocity null) and $\|\Delta^2 m(t)\|$ **with its
  direction** (acceleration null). *This is the killer null* — the project already found trajectory curvature
  redundant with $\|\Delta^2 m\|$ for the crude constructions; T1 must show the **geodesic split** adds a
  change-type discrimination the scalar 2nd-difference direction does not.
- **BFAST / CCDC** break-vs-trend flags (the established abrupt-vs-gradual standard).
- **Harmonic deseasonalization** residual (K=3) — the seasonality null that beat the SSDS detector.
- **Min canonical angle SSA** (the conventional single-angle predecessor DS must beat — Kanai's bar).
- **Self-consistency** $d_2\approx|\frac{d}{dt}d_1|$ (correctness, not validity).

### 2.5 Experiment plan (all on data already fetched — no new acquisition)
- **L0 (sanity).** identical-date $\Rightarrow d_1=d_2=0$; self-consistency corr. ✅ already passing.
- **L1 (synthetic, multi-seed ≥8).** Inject (a) abrupt step, (b) gradual ramp, (c) pure phase-shifted season,
  (d) seasonal-shape change, into a stable multi-mode series. **Falsifiable claim:** the geodesic
  orthogonal/along ratio separates (a) from (b) with AUC/accuracy above the $\|\Delta^2 m\|$-direction null.
- **L2 (real S2, change-TYPE).** Use the series already in hand: **abrupt** = Tenerife/Creek fire onset;
  **gradual** = GERD reservoir fill, Saudi irrigation phenology onset; **seasonal-only** = Iowa Corn Belt
  (null). Score: does orthogonal-dominant correctly label the fire and along-dominant the reservoir/phenology,
  where the scalar 2nd-difference direction does not?
- **L3 (recovery).** Tenerife post-fire $d_{\mathrm{pre}}$ vs NBR slope (circularity-guarded).
- **L4 (attribution).** Which bands light up per change type vs known physics (fire→SWIR/NBR; phenology→red-edge).

### 2.6 Metrics
Change-**type** classification accuracy / AUC (abrupt vs gradual) — the headline; event-onset localization
(±5-day hit) as a secondary; recovery-slope correlation; attribution hit-rate vs physics; **always beside the
scalar/harmonic null** and with ≥8 seeds + reported variance (L1 mandate). Report *runtime* (student-feedback ask).

### 2.7 Falsifier (pre-registered — what kills T1)
If the **geodesic along/orthogonal split does NOT classify abrupt-vs-gradual change-type better than the
direction of the scalar 2nd-difference $\|\Delta^2 m\|$** (and not better than BFAST/CCDC break-vs-trend) on
both synthetic L1 and the real L2 cases, then the geodesic decomposition adds nothing over the mean-vector
2nd-difference, and T1 collapses into a (still-publishable) **negative row of the diagnostic (T4)**: "the
subspace geodesic split is redundant with the scalar 2nd-difference for satellite change-type." That outcome is
acceptable and pre-committed — it is why T1 is low-risk despite the redundancy prior.

### 2.8 Honest expectation
The prior (from the crude H-B tests) is that **magnitude** $d_2$ is redundant with $\|\Delta^2 m\|$. The *narrow*
live hope is that the **2-D along/orthogonal split** (a decomposition, not a scalar) carries change-*type*
information the scalar magnitude discards, *because* M-SSA encodes joint cross-band+temporal structure a per-band
harmonic misses. If true → a genuine, Sensei-flagship, first-and-unique positive. If false → a clean diagnostic
result. Either way the 2-week ACCV submission (T4 + this section) is safe.

---

## 3. T2 go/no-go (the higher-ceiling novel bet — gate before investing)
**Controlled phase-shift test (S2 + synthetic, no new labels needed):** build (a) a series with a pure
**phase shift** of the seasonal cycle (nuisance) and (b) a series with a **cycle-shape change, same phase**
(true change). Compute: RTW warp-invariant subspace distance vs harmonic-deseasonalization-**with-phase-term**
residual vs TWDTW distance. **Go if** RTW stays flat on (a) and fires on (b) while the phase-aware harmonic
either false-alarms on (a) or misses (b) — i.e. RTW's warp-invariance is *not* reducible to a phase term.
**No-go if** a phase-aware harmonic matches RTW on both — then the warp lever collapses to known deseasonalization
and T2 dies. This single test decides whether T2 is worth the data-acquisition effort.

---
## Honest novelty verdict (whole synthesis)
There is **no open lever for "subspace detects satellite change better"** — that is closed, with citations. The
**genuinely novel, defensible tasks are characterization/representation**: T1 (change-type via geodesic split,
runnable now, Sensei-flagship, low-risk) and T2 (RTW warp-invariant cycle-shape CD, highest-novelty,
data-gated). The diagnostic (T4) is the guaranteed-publishable spine that holds regardless. This *inverts* the
project's founding instinct (find a better subspace detector) into its defensible form (use subspace geometry to
**characterize** change where scalars structurally can't).

## Next falsifiable step (single, concrete)
Run T1's **L1 synthetic change-type test FIRST**: does the geodesic along/orthogonal ratio classify abrupt-vs-
gradual above the $\|\Delta^2 m\|$-direction null (≥8 seeds)? That one number decides whether T1 is a positive
section of the ACCV paper or a negative row of the diagnostic — and it needs no new data.
