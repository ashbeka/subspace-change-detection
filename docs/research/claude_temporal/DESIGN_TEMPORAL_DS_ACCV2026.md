# Design Document — Temporal Difference Subspaces for Satellite Image Time Series

Status: **active design, sprint mode**
Created: 2026-06-17
Target: ACCV 2026 (submission **2026-07-05**, ~18 days out), conference Dec 16–18 2026, Osaka.
Owner: Abdelrahman (Fukui Lab). Co-author/agent: Claude Code.

This document is self-contained. A fresh Claude Code session should be able to start
implementing from this file plus the repo. It supersedes the bi-temporal "spatially aware DS
on OSCD" direction as the active thesis/paper core. The bi-temporal OSCD path is retained only
as the `T = 2` special case and as a labeled-benchmark sanity check.

Read the companion critique first if you have not:
- `docs/SECOND_OPINION_RESEARCH_CONTEXT.md` (the neutral briefing)
- `docs/RESEARCH_RESET_AUDIT.md` (Codex's audit; good evidence, different recommendation)

---

## 0. One-paragraph summary

We move the Difference Subspace (DS) machinery from the spatial axis of a single image pair —
where the "set of samples" is fake and the method loses 2:1 to PCA-diff — to the **temporal axis
of a dense Sentinel-2 time series**, where a set of samples genuinely exists. For each ground
region we build a **sequence of subspaces over sliding date-windows**. The **first-order DS** between
consecutive window subspaces measures the **velocity** of land-surface change (change onset); the
**second-order DS** measures the **acceleration** (degradation vs recovery); **geodesic decomposition**
on the Grassmann manifold separates smooth seasonal drift from abrupt events. This is the first
application of Fukui-lab first/second-order DS + geodesic subspace dynamics to satellite image
time series. We do not claim to beat neural change detectors; we claim a novel, interpretable,
theoretically grounded **temporal** change-and-recovery descriptor, validated on documented events,
post-fire recovery proxies, and controlled synthetic injections.

---

## 1. Why this direction (decision record)

- **The bi-temporal DS is a category error.** DS/MSM/KMSM/KGDS are *image-set* methods: a subspace
  needs a *set* of related observations. A single pre/post pair has one image per date, so the project
  manufactured a set from the pixels of one image. The resulting `D ∈ ℝ^(13×6)` is a single matrix
  applied identically to every pixel — a fixed band re-weighting that cannot encode spatial position.
  Measured result: canonical DS correlates 0.19 with raw L2 and scores mean AP 0.197 vs PCA-diff 0.395.
  `[experiment-evidence]` `docs/experiment_reports/oscd_spatial_subspace_sweep_core5_2026-06-14.md`.
- **A temporal window is a real set.** SSA builds a signal subspace from a trajectory window; a window
  of dates is a legitimate image-set. This is the setting of Sensei's own papers.
- **Advisor alignment.** Fukui's two flagship papers are temporal:
  - Second-order difference subspace, arXiv `2409.08563`: first/second-order DS = *velocity/acceleration*
    of subspace dynamics along a geodesic on the Grassmann manifold (validated on temporal shape + biometric time series).
  - Time-series anomaly detection via DS between signal subspaces, arXiv `2303.17802`: past-window vs
    present-window signal subspaces (SSA), DS magnitude as anomaly score.
  - Slack steering, repeatedly: "temporal analysis," "geodesic for smooth changes," "sequential satellite
    images," "Is the time step five days?", and explicitly: *"the first/second DSs and the geodesic
    decomposition work effectively for the analysis of sequential satellite images... at least, such a
    trial is the first and unique one in this research topic, although it may not achieve top performance."*
- **Student alignment.** `slack_notes_to_myself.md:355`: *"Lack of Temporal Subspace Methods for Multi-Date
  Satellite Image Analysis... track damage progression... predict recovery trajectories."* Serves the
  stated long-term goal (post-disaster reconstruction: is an area still degrading or recovering?).
- **Sensei's bar is novelty, not SOTA.** This removes the impossible requirement to beat U-Net on OSCD.

---

## 2. Problem statement and contributions

### 2.1 Problem statement

```text
Given a dense multispectral Sentinel-2 time series over a region, can a sequence of date-window
subspaces with first-order DS (velocity) and second-order DS (acceleration), plus geodesic
decomposition, (a) detect the onset of abrupt land-surface change and (b) distinguish ongoing
degradation from recovery, more interpretably than pairwise spectral differencing and the
conventional SSA minimum-angle method?
```

### 2.2 Claimed contributions (paper-facing)

1. **A temporal subspace framework for satellite image time series**: per-region date-window
   subspaces forming a subspace trajectory on the Grassmann manifold.
2. **First- and second-order DS as velocity/acceleration of land change**: first-order DS magnitude
   localizes change onset in time; second-order DS sign/orientation separates degradation from recovery.
3. **Geodesic decomposition for seasonal-vs-abrupt separation**: smooth (geodesic) component ≈ seasonal
   drift; residual ≈ abrupt event — a built-in pseudo-change filter.
4. **Validation protocol** combining documented real events, a recovery proxy (NBR), and controlled
   synthetic injection, with the bi-temporal (OSCD, `T = 2`) case shown as a degenerate special instance.

### 2.3 Explicitly NOT claimed (forbidden overclaims)

- Beating neural / Siamese / foundation-model change detectors on accuracy.
- Disaster damage *classification* or building damage severity.
- That DS is invented here.
- Semantic change detection.
- Any claim from unlabeled data without an evaluation proxy.

---

## 3. Method specification

> **NOTE (2026-06-18): §11 supersedes §3.2 and §4 wherever they conflict.** After reading the lab
> reference code (MagTool, Soto-san's cv_motion3d) and an adversarial review of this plan, the subspace
> construction and the verification protocol were corrected. §11 is the authoritative version.

### 3.1 Data object

For an area of interest (AOI), a cloud-masked Sentinel-2 L2A time series:
```text
cube  X ∈ ℝ^(D_dates × B × H × W)      B = bands used (e.g., 10–13), D_dates = number of valid dates
```
Partition the AOI spatially into **regions** R (choose ONE for the first result; support the rest later):
- `region = pixel-neighborhood` (e.g., the k×k patch centred on each pixel) → dense spatial maps.
- `region = tile/superpixel` (e.g., 32×32 blocks, or SLIC superpixels) → coarse but fast; recommended for Gate 0.
- `region = whole AOI` → a single global temporal curve; cheapest sanity check.

### 3.2 Temporal subspace construction (PRIMARY — image-set-over-dates, Sensei-faithful)

This mirrors TPAMI-2015 Venus (sample = flattened image view, set = 300 views) with **views → dates**,
and honours Sensei's note to "use 13-dimensional subspaces in a super-high-dimensional vector space."

```text
For region R and a temporal window W = {t, t+1, ..., t+T-1} of T consecutive valid dates:
    sample vector at date τ:  v_τ = flatten( X[τ, :, R] )   ∈ ℝ^(B·|R|)     (the "super-high-dim" space)
    sample set for window W:  V_W = [ v_t, v_{t+1}, ..., v_{t+T-1} ]        ∈ ℝ^((B·|R|) × T)
    subspace:                 S_W = top-r left singular vectors of centered V_W   (r ≤ T-1; r≈ small)
Slide W across time (stride s) → subspace trajectory  S_1, S_2, ..., S_M  on the Grassmann manifold.
```

Recommended starting hyperparameters (sweep later): `T ∈ {4,5,6}`, `stride s = 1`, `r` by variance ≥ 0.95 capped at `T-1`, region = 32×32 tile.

### 3.3 First-order DS (velocity)

Between two consecutive window subspaces `S_a = span(Φ)`, `S_b = span(Ψ)` (orthonormal bases `Φ, Ψ`):
```text
Use the corrected canonical/eig DS already in the repo (NOT legacy_residual_stack).
G = Φᵀ Ψ ;  SVD: G = U Σ Vᵀ ;  canonical angles θ_i = arccos(σ_i).
First-order difference subspace basis D1 spans the directions of disagreement between S_a and S_b.
Velocity score:   d1(t) = ‖D1(t)‖  (sum of sin²θ_i, i.e. the DS "magnitude"; define exactly and test
                  against equal-subspace edge case → must give 0).
```
Per-region time series `d1(t)` = change velocity. A spatial map at time `t` = `d1(t)` over all regions.

### 3.4 Second-order DS (acceleration)

Follow arXiv `2409.08563`. Conceptually: take the Karcher (Grassmann) mean of neighbouring subspaces and
combine first-order differences to obtain the second-order DS, analogous to a central second difference
approximating acceleration along the geodesic.
```text
d2(t) = magnitude of second-order DS at time t   (acceleration of subspace motion)
sign / orientation of D2 relative to the post-event change direction  →  degradation vs recovery:
    accelerating away from pre-event subspace   → ongoing degradation
    decelerating / returning toward pre-event   → recovery
```
**Implementation note:** Aono-kun and Jang-kun hold the lab's reference implementations of second-order
DS and geodesic decomposition (Sensei pointed here repeatedly). Get their code, verify dimensions on a
toy case, then adapt — do not reinvent. Record the source-to-code trail per `README.md §7`.

### 3.5 Geodesic decomposition (seasonal vs abrupt)

Project the subspace trajectory onto the geodesic connecting window endpoints (or a fitted smooth geodesic);
the **on-geodesic** component ≈ smooth seasonal/illumination drift, the **off-geodesic residual** ≈ abrupt
event. Use the residual magnitude as a pseudo-change-suppressed event score. Sensei: *"we have recently
proved mathematically that the geodesic projection works as expected, so we can use it with ease."*

### 3.6 Alternative construction (SSA trajectory, most literal to the anomaly paper)

Per pixel/region, reduce to a scalar or low-dim index series (e.g., NBR, NDVI, or PC1 of bands); build a
Hankel/trajectory matrix with embedding length L; SVD → signal subspace; past-window vs present-window DS.
Keep this as a secondary experiment / robustness comparison; it gives the finest spatial localization and
is the closest match to `2303.17802`.

### 3.7 Outputs

- Per-region temporal curves: `d1(t)`, `d2(t)`, geodesic-residual(t).
- Per-timestep spatial maps of `d1` (and `d2`) over the AOI.
- An event-onset timestamp per region (change-point in `d1`).
- A degradation/recovery label per region post-event (sign of `d2` / geodesic trend).

---

## 4. Datasets and evaluation

All data via **Google Earth Engine** (`COPERNICUS/S2_SR_HARMONIZED`, ~5-day cadence) to avoid building a
download/preprocessing pipeline under deadline. Cloud-mask with the Scene Classification Layer (SCL) /
QA60 or s2cloudless; require a minimum count of valid dates per window; record the masking choice.

### 4.1 Track A — documented-event onset detection (PRIMARY, falsifiable)

Pick **8–12 AOIs with a known event date** and a clear optical signature, e.g.:
- Wildfires with official ignition/containment dates (fire perimeter databases give date + polygon).
- Volcanic lava flows (e.g., La Palma 2021, ignition 2021-09-19).
- Floods with documented dates; large construction/land-clearing with datable starts.
Metric: does `d1(t)` peak within ±δ dates of the true event? Report **detection rate**, **temporal
localization error (days)**, and **change-point AUC** vs baselines. Use the event polygon as the spatial
region; use nearby unaffected polygons as negatives.

### 4.2 Track B — recovery characterization (the second-order story)

Post-fire AOIs: validate `d2(t)` / geodesic trend against the **NBR recovery curve** (Normalized Burn
Ratio recovery is a standard, citable post-fire recovery proxy — no manual labels needed). Report
correlation between the second-order/geodesic recovery signal and NBR slope; show second-order DS sign
flips from "degrading" to "recovering" as vegetation returns.

### 4.3 Track C — synthetic injection (controlled, fully labeled)

Take stable AOIs (desert/stable urban) with low real change; inject controlled step and ramp anomalies of
varying magnitude into the time series; measure **detection AUC vs injected magnitude** and **temporal
localization error**. This gives a clean, reviewer-proof controlled curve.

### 4.4 Track D — bi-temporal special case (continuity with prior work)

Show that with `T = 2` the framework reduces to bi-temporal first-order DS; run it through the existing
OSCD pipeline and report it beside PCA-diff/raw L2 — honestly, including that it is not the strong case.
This connects the temporal method to the only labeled binary benchmark and reuses existing code.

### 4.5 Baselines (mandatory)

- Raw spectral L2 / CVA magnitude over time (per-date difference series).
- PCA-diff over time.
- **SSA minimum-angle** between past/present subspaces — the conventional method Sensei's DS paper
  improves on; this is the key ablation (DS magnitude vs min-angle).
- NBR-Δ / NDVI-Δ temporal baselines (for fire/vegetation).
- IR-MAD (bi-temporal, Track D only) — audit the implementation before citing (`notes/methods.md §IR-MAD`).
- (Optional, only if time) one off-the-shelf neural CD on Track D; likely skip for the sprint.

### 4.6 Metrics

Detection: AUC / AP of change-point vs known date; temporal localization error (days); per-AOI hit rate.
Recovery: correlation with NBR slope; sign-agreement rate.
Synthetic: detection AUC vs magnitude; false-positive rate on stable windows.
Always report the equal-subspace edge case (must be 0) and runtime.

---

## 5. Reuse map (what carries over from the repo)

- `phase1/ds/pca_utils.py` — `difference_subspace_canonical`, `difference_subspace_eig`, `fit_pca_basis`
  (use these; **avoid** `legacy_residual_stack`).
- `phase1/data/preprocessing.py` — `vectorize_cube`, band normalization, valid masks.
- `phase1/data/` OSCD loader — for Track D (`T = 2`).
- `phase1/scripts/venus_kds_demo.py` + `phase1/subspace/kernel_difference_subspace.py` — reference for
  image-set subspace construction and for an optional KDS extension.
- `tests/` — extend the existing DS/KDS formula checks with second-order DS + geodesic toy tests.
- `project_cli.py` — add `temporal-*` subcommands alongside the phase1/phase2 ones.
- Lab code from **Aono-kun / Jang-kun** — second-order DS + geodesic decomposition reference implementations.

New code (keep small, well-scoped):
```text
temporal/
  gee_fetch.py        # GEE export of cloud-masked S2 time series for an AOI + date range
  series.py           # build the date cube, valid-date masking, region partitioning
  subspace_seq.py     # date-window subspace trajectory construction
  ds_dynamics.py      # first-order d1(t), second-order d2(t), geodesic residual
  events.py           # change-point extraction, onset timestamp, degradation/recovery label
  eval_tracks.py      # Track A/B/C/D metrics
  configs/            # AOI lists, event dates, window/rank/region sweeps
```

---

## 6. Sprint plan (18 days to 2026-07-05)

**Gate 0 (Days 1–3, by Jun 20): make-or-break.** Pull 3 event AOIs + 2 stable AOIs via GEE. Build the
date cube, tile regions, construct the date-window subspace trajectory, compute `d1(t)`. **Question: does
`d1(t)` spike within ±δ of the known event date?** If YES → proceed. If NO after honest debugging
(window length, region size, masking) → STOP and invoke the fallback (§8).

**Days 4–6:** Scale Track A to 8–12 AOIs. Implement second-order DS `d2(t)` + geodesic residual (with
Aono/Jang code). Add SSA-min-angle, CVA-over-time, NBR-Δ baselines. Synthetic injection (Track C).

**Days 7–9:** Track B recovery validation vs NBR. Track D OSCD `T = 2`. Lock metrics; generate figures.

**Days 10–13:** Ablations: window `T`, rank `r`, region size/type, geodesic on/off, DS-vs-min-angle.
Freeze results tables.

**Days 14–17:** Write the paper (ACCV format). Internal review with Aono/Jang/Sensei. Polish figures.

**Day 18 (Jul 5):** Submit.

Daily rule: every result that enters the paper gets a logged config, seed, AOI list, event-date source,
and runtime. No smoke run becomes evidence.

---

## 7. Figures and tables for the paper

Figures:
1. Concept: subspace trajectory on the Grassmann manifold; first/second-order DS as velocity/acceleration.
2. Construction: date-window → subspace; image-set-over-dates vs Venus views (one diagram).
3. Track A: `d1(t)` curve with the known event date marked, for 3–4 AOIs, vs SSA-min-angle baseline.
4. Track B: `d2(t)` / geodesic recovery signal overlaid on the NBR recovery curve.
5. Spatial `d1` map at the event timestep vs ground-truth event polygon.
6. Track C: detection AUC vs injected magnitude.

Tables:
1. Method/construction comparison (sample unit, ambient dim, subspace dim, what each order measures).
2. Track A detection metrics by AOI, DS vs baselines.
3. Recovery correlation (Track B).
4. Ablations (T, r, region, geodesic on/off).
5. Safe-claims vs forbidden-claims.

---

## 8. Risk register and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Cloud gaps corrupt the time series | High | SCL/s2cloudless masking; min-valid-date thresholds; subspace methods tolerate some missingness; report valid-date counts. |
| `d1` does not localize the event (Gate 0 fails) | Medium | Try SSA-trajectory construction (§3.6), larger regions, NBR-reduced series; if still null → fallback below. |
| Second-order DS / geodesic code not ready in time | Medium | Ship Track A (first-order) as the minimum paper; fold second-order into the thesis if it slips past Jul 5. |
| 18 days is too short for full submission | Medium-High | **Minimum publishable result = Track A + Track C + OSCD T=2.** If even that slips: target an ACCV workshop, or submit to the next venue and keep this as the thesis core (it is strong either way). |
| Reviewer: "this is just SSA/anomaly detection" | Medium | The novelty is *second-order* DS (acceleration / recovery) + geodesic seasonal separation on multispectral satellite series — none of which SSA-min-angle does. Foreground that. |
| Reviewer: "no SOTA comparison" | Medium | Frame as interpretable temporal descriptor, not a CD accuracy contest; include min-angle + classical temporal baselines; cite Sensei's "first and unique trial" positioning. |

**Minimum publishable result (defend this line):** Track A (first-order DS event detection on ≥8 documented
AOIs, beating SSA-min-angle and CVA on temporal localization) + Track C (synthetic AUC curve) + Track D
(OSCD T=2 sanity). Everything else is upside.

---

## 9. Open decisions to confirm with Sensei / senpais (ask early, Days 1–2)

1. Get Aono-kun / Jang-kun second-order DS + geodesic reference code and the exact DS-magnitude definition.
2. Confirm the AOI/event list and the recovery proxy (NBR) are acceptable as evaluation.
3. Confirm `COPERNICUS/S2_SR_HARMONIZED` is the intended sequential dataset (Sensei previously endorsed it).
4. Confirm the ACCV framing: interpretable temporal subspace dynamics, "first and unique trial," not SOTA.

---

## 10. Provenance (source → claim trail)

```text
Fukui & ... , "Second-order difference subspace", arXiv 2409.08563
    → first/second-order DS = velocity/acceleration of subspace dynamics on Grassmann geodesic
    → satellite adaptation: window-of-dates subspaces, per-region d1(t)/d2(t)
... , "Time-series anomaly detection based on DS between signal subspaces", arXiv 2303.17802
    → SSA past/present signal-subspace DS magnitude as anomaly score
    → satellite adaptation: SSA-trajectory construction (§3.6), min-angle baseline
Fukui & Maki, TPAMI 2015 (KDS/KGDS, Venus)
    → image-set subspace + nonlinear DS; "views → dates" analogy; optional KDS extension
```
Every implemented method file must carry the `source → math object → satellite adaptation → code path →
verification → allowed claim` docstring trail (repo convention, `README.md §7`).

---

## 11. RESOLVED construction + verification protocol (authoritative)

Added 2026-06-18 after reading MagTool (Jang) + cv_motion3d (Soto-san) and an adversarial review.

### 11.0 What we are actually measuring (and why it dissolves "what is change?")

The bi-temporal OSCD framing forced a *semantic* definition of change and then validated against labels
that only mark **urban artificialization** — *"labelling as Change only urban growth … ignoring natural
changes"* (OSCD authors, apple_notes.md:271,348). That mismatch is the source of the long-standing
"what is change in our context?" problem (slack_notes_to_myself.md:85). The temporal framing **changes the
question** from "is this pixel semantically changed?" to:

```text
Does the scene's subspace TRAJECTORY respond, at the right TIME, to a physically documented event,
and stay quiet when nothing happens — and can we tell abrupt disturbance from gradual drift/recovery?
```

We never need to resolve "what is change semantically," because we validate against ground truth that is
unambiguous on its own terms: an event **date** (external registry), an **injected** magnitude (synthetic),
and a **recovery curve** (NBR). That is the whole point of the pivot.

### 11.1 Subspace construction — RESOLVED (supersedes §3.2/§3.6)

The reference code (Soto-san, `cv_motion3d`) builds the subspace from **one time instant**, not a window:
`generate_shape_subspace` = top-3 left singular vectors of the centered (markers × coords) matrix of a
**single frame** (utils.py:23-27). Temporality enters only by comparing S(t) and S(t+τ) at a fixed lag.
My earlier §3.2 ("window-of-dates") was a *different* construction; this conflation was the review's main
catch. Resolution:

- **PRIMARY = Construction A (per-date spatial-spectral subspace; motion-faithful).** For each date t, take
  the (pixels × bands) matrix of the AOI (optionally a short ±k-date neighborhood for cloud robustness),
  center it, SVD, keep the top-r left singular vectors → `S(t)` = an r-dim subspace **in pixel-space**
  (its basis vectors are eigen-images, so spatial structure is preserved — this directly answers Sensei's
  "DS breaks spatial information" criticism). Marker→pixel, frame→date, exactly per the reference code.
  Compare `S(t)` vs `S(t+τ)` with the corrected canonical DS. Outputs BOTH a per-date scalar curve AND a
  per-pixel contribution map (the change localizer) — the dual output of `display_motion_score_contribution`.
- **ABLATION = Construction B (per-region SSA-trajectory; Kanai/anomaly-faithful).** Per region/pixel,
  Hankel-embed a scalar index series (NBR/NDVI/PC1), SVD past-window vs present-window → signal subspaces;
  DS magnitude = anomaly score. This is the construction with the cleanest min-angle baseline (Kanai 2023)
  and the finest temporal localization. Run it as the comparison, not the headline.

State which construction produced every number. They are different methods with different invariances.

### 11.2 The three curves and the recovery fix

- `d1(t) = ‖DS(S(t), S(t+τ))‖` — VELOCITY (onset). `calcMagnitude = 2·Σ(1−cosθ_i)` (magnitude.py:223).
- `d2(t)` — ACCELERATION via `calc2ndMagDecomposed`, deviation of S(t+τ) from the Karcher midpoint of
  S(t),S(t+2τ) (magnitude.py:284-325). Split into `mag_along` (on-geodesic) and `mag_orth` (off-geodesic =
  abrupt regime break).
- **Recovery — CORRECTED.** MagTool returns **nonnegative magnitudes**; there is **no sign**, so the old
  "second-order DS sign flips degrading→recovering" claim (§3 Track B) is **not supported by the code**.
  Replace it with a **distance-to-baseline trajectory**: fix a pre-event reference subspace `S_pre`; track
  `d_pre(t) = ‖DS(S(t), S_pre)‖`. **Rising = degrading (moving away from pre-event state); falling =
  recovering (returning).** Second-order DS then characterizes the *curvature* of this recovery (accelerating
  vs stalling). This needs only MagTool magnitudes. (A signed directional operator is a possible novel
  extension, but is NOT in the reference code and must be defined+tested before any signed claim.)

### 11.3 Verification ladder — CORRECTED (supersedes §4). Each rung: measure / ground truth / metric / falsifier.

- **L0 Sanity (label-free, do first).** Identical-date subspaces must give magnitude ≈0 (mathematical
  invariant, magnitude.py:223). Stable-desert near-identical dates must give `d1 ≪ d1(event)`. Self-
  consistency: `d2(t)` should track the finite-difference derivative of `d1(t)` — but treat this as a
  **correctness check only, NOT task validity** (the review's catch; do not import Fukui's r=0.948 as a
  target — re-derive on our data). FALSIFIES IF identical-date magnitude ≠ 0 or desert spikes like events.
- **L1 Synthetic injection (fully labeled, ZERO circularity — the rigor anchor).** Inject step/ramp of known
  time t₀ and magnitude into a stable AOI. Metric: detection AUC vs magnitude (must be monotone), localization
  error `|argmax d1 − t₀|`, FPR on un-injected windows. FALSIFIES IF the spike isn't at t₀ / doesn't scale /
  stable windows spike too.
- **L2 Known-event localization (real data, the headline).** AOI with a documented event date+polygon from
  an EXTERNAL registry (MTBS/NIFC/EFFIS fire perimeters; La Palma 2021-09-19). Metric: does `d1(t)` peak
  within **pre-registered δ = ±1 revisit (~5 days)** of the true date; report hit rate, localization error,
  change-point AUC. **Gate-0 pre-registration (review-mandated):** freeze ONE {construction, T, r, region,
  mask, τ, bands} config and the event/AOI list **before** computing any DS on event data; do all tuning on
  L0/L1 stable/synthetic data only; define failure (e.g. <4/6 event AOIs within δ = STOP→fallback). The
  event date must NEVER be derived from DS.
- **L3 Recovery vs INDEPENDENT proxy (guard circularity).** `d_pre(t)` recovery trajectory vs the NBR
  recovery slope. **Circularity warning (review):** NBR=(B8−B12)/(B8+B12); if B8/B12 dominate the subspace,
  agreement is by construction. Mitigations: (a) prefer a non-reflectance ground truth where possible —
  MTBS **dNBR burn-severity class polygons** are human-adjudicated products, not your raw series; (b) if
  only NBR is available, the DS claim survives **only if DS tracks recovery timing BETTER than raw-L2 does**
  (report DS-vs-NBR and rawL2-vs-NBR side by side). FALSIFIES IF `d_pre` is uncorrelated with recovery, or no
  better than raw L2.
- **L4 Comparative + mandatory nulls.** Beside DS on every rung, report: **(NEW, review-mandated) the trivial
  detector** — mean |reflectance(t)−reflectance(t−τ)|, single-band B12 step, NBR step — as the PRIMARY null
  (the spatial precedent DS-vs-rawL2 corr 0.19, AP 0.197 vs 0.395 proves the trivial null can win); the
  **windowed-variance/moving-average null** (shares temporal smoothing, lacks Grassmann geometry); and
  **SSA min-angle** (Kanai 2023: DS 0.923 AUC vs SSA-θ1 0.829 — reproducing DS>min-angle on satellite data is
  the central comparative claim). FALSIFIES IF DS ≈ trivial null, or strictly worse than min-angle.

### 11.4 Extra controls the review demanded (non-optional)

- **Temporal placebo AOIs**: snow on/off, cloud-shadow residual, illumination-angle swing, the 2022-01-25
  harmonization step. `d1` must NOT spike there. If it spikes on everything that moves reflectance, it is a
  generic change-magnitude meter, not an event detector.
- **Irregular-sampling control**: cloud gaps make τ span variable calendar time; larger gaps mechanically
  inflate magnitude. Regress `d1` on inter-acquisition day-gap; show event spikes survive gap normalization.
- **Multiple-comparisons null for L2**: permute event dates / score `d1` peaks against random dates → show hit
  rate beats chance.
- **Hard pipeline assertion**: use `COPERNICUS/S2_SR_HARMONIZED` only (avoids the +1000-DN 2022-01-25
  artifact). Cloud-mask with SCL/s2cloudless (NEW work — current OSCD pipeline does none) and require a
  minimum valid-date count per window.
- **Label-light validators (from Kanai 2023, need no extra labels)**: (a) sweep T / overlap / r and show AUC
  is stable; (b) Grassmann+MDS separability plot — embed per-window subspaces/DS as Grassmann points, show
  degrading vs recovering windows form separable clusters for DS but not for min-angle.

### 11.5 Success definition (leaderboard-independent)

A real, publishable finding (under Sensei's "novel and unique, need not be SOTA" bar) requires ALL of:
L0 passes (math correct); L1 passes (responds to controlled change, scales, quiet when stable); L2/Gate-0
passes on ≥4/6 frozen event AOIs at pre-registered δ; DS **beats the trivial null and matches/beats SSA
min-angle** (L4); and at least one non-circular recovery result (L3 via dNBR classes or DS>rawL2 on NBR
timing). NOT required: beating a neural detector or winning OSCD. If Gate-0 fails after frozen-config
testing, the honest output is a **negative result** + the documented fallback — not a claim from visuals.

### 11.6 Implementation gotchas (for the building agent)

- `cv_motion3d` README mislabels the second/geodesic files; the **code is authoritative** (`*_second.py` =
  acceleration via S2-vs-Karcher-mean; `*_geodesic.py` = along/orth split).
- The second-order/geodesic code is **unverified on multi-channel satellite data** (slack:258). Run a toy
  dimensional check before any `d2` result enters the paper.
- MagTool/lab conventions: vectors are **columns**, basis is autocorrelation-PCA (no mean-centering in the
  MATLAB toolbox; the motion code DOES center per frame — pick and document one), subspace dim by fixed r or
  variance ratio. Identical subspaces → magnitude 0 is the universal sanity invariant.
