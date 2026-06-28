# Research Agenda — Ranked Problem Statements & Experiment Catalog

Created: 2026-06-17. Owner: Abdelrahman (Fukui Lab) + Claude.

Purpose: a living catalog of candidate research problems, ranked by importance, each with ranked
sub-experiments. This is the **menu of future/alternative work** so good ideas are not lost while we
sprint on the active direction. The **active** direction (P1) has its own full spec in
`docs/DESIGN_TEMPORAL_DS_ACCV2026.md`. Everything below P1 is parked-but-tracked.

Legend: **[ACTIVE]** pursuing now · **[COMPANION]** strengthens the active paper · **[FALLBACK]** if active
fails · **[FUTURE]** thesis-extension / next paper · **[PARK]** good idea, not now.

Cross-cutting menus (reused across problems) are at the bottom: §A method-construction menu, §B
evaluation-strategy menu.

---

## P1 — Temporal first/second-order DS + geodesic for change & recovery  **[ACTIVE]**

Question: *Given a dense Sentinel-2 time series, can per-region date-window subspaces with first-order DS
(velocity), second-order DS (acceleration), and geodesic decomposition detect change onset and distinguish
degradation from recovery, more interpretably than pairwise differencing and SSA minimum-angle?*

Why #1: the only framing where the DS "set of samples" is real (a window of dates); maximally
advisor-aligned (his two flagship papers are exactly this); novel ("first and unique trial" per Sensei);
serves the reconstruction/recovery motive; falsifiable via known events + recovery proxies + synthetic
injection; and OSCD survives as the `T=2` special case so nothing built is wasted.

Novelty boundary: NOT "first temporal subspace method" (SSA/anomaly-DS exist). The novelty is **second-order
DS (acceleration/recovery) + geodesic seasonal-vs-abrupt separation on multispectral satellite time series.**

Ranked sub-experiments:
1. **E1.1 Gate-0 feasibility probe** — one event AOI: does first-order DS magnitude spike at the known date?
2. **E1.2 First-order onset detection** — date-window subspace trajectory; `d1(t)` change-point; AUC vs known events across 8–12 AOIs.
3. **E1.3 Second-order degradation/recovery** — `d2(t)` sign vs an independent recovery proxy (NBR).
4. **E1.4 Geodesic decomposition** — on-geodesic (seasonal) vs off-geodesic residual (abrupt) as a pseudo-change filter.
5. **E1.5 Synthetic injection control** — step/ramp anomalies → detection AUC vs magnitude; FPR on stable windows.
6. **E1.6 OSCD T=2 special case** — connect to the labeled binary benchmark; reuse existing pipeline.
7. **E1.7 SSA-trajectory variant** — per-pixel Hankel-embedding subspace (most literal to the anomaly paper); finest localization.
8. **E1.8 Ablations** — window length T, rank r, region unit (pixel-nbhd / tile / superpixel), DS-magnitude vs min-angle.
9. **E1.9 KGDS / multi-window** — nonlinear or >2-window generalized DS once linear works.

---

## P2 — Temporal pseudo-change suppression via background subspace  **[COMPANION]**

Question: *Can a slow/background temporal subspace (SFA-flavored) separate seasonal/illumination drift from
genuine abrupt change, so DS responds to real events and not to vegetation phenology or sun-angle?*

Why: pseudo-change is the #1 practical failure of all spectral CD; this directly hardens P1 and gives the
paper a "why it doesn't false-alarm on seasons" figure. Strong companion, not a separate thesis.

Ranked sub-experiments:
1. **E2.1** Background/slow subspace per region from a same-season or full-year window; residual = anomaly.
2. **E2.2** PCA/subspace reconstruction-error of the post window against a pre/normal-state subspace as a change/anomaly score (the "stuff that doesn't fit the normal subspace" idea from `slack_notes_to_myself.md`).
3. **E2.3** Geodesic on-vs-off component (shared with E1.4) quantified as a seasonality-suppression metric.
4. **E2.4** Stress test on a no-event but high-seasonality AOI: a good method stays quiet (negative control).

---

## P3 — Hybrid: neural localizer + subspace temporal interpreter  **[FUTURE]**

Question: *Let a neural/classical detector localize candidate changed regions; can DS/GDS/geodesic
descriptors then interpret each region's temporal trajectory (onset, degradation vs recovery, change-type
clustering) better than the mask alone?* (= `H5`/`H10` in `notes/experiments.md`.)

Why later: only meaningful once P1 shows the geometry carries signal; otherwise the net does all the work
and the subspace part is decoration. Strong thesis chapter / second paper.

Ranked sub-experiments:
1. **E3.1** NN localizes changed connected components → build per-region temporal subspace → cluster change-types (ARI/NMI if labels).
2. **E3.2** Deep-feature temporal DS — subspaces from encoder/foundation-model embeddings per date vs raw-spectral DS.
3. **E3.3** Error-subspace diagnostics — where the detector errs, characterized by subspace geometry.

---

## P4 — Spatially aware DS on OSCD (diagnostic)  **[FALLBACK]**

Question: *Do global / patch-vector / local-window DS preserve enough spatial structure to be useful
interpretable priors for OSCD binary change segmentation?*

Why parked: already losing 2:1 to PCA-diff; not novel (spatial-spectral subspace CD exists). Keep only as
(a) a fallback if temporal data acquisition fails before the deadline, or (b) one honest negative/diagnostic
chapter ("the global pixel subspace is the wrong object — here's the evidence"). Codex's Rank 1; demoted here.

Ranked sub-experiments: global vs patch vs window DS; score-definition ablation; rank sweep; Celik-patch
baseline; failure-mode taxonomy. (Largely already run — see `docs/experiment_reports/`.)

---

## P5 — Object-level / greenhouse / building change  **[FUTURE]**

Question: *Can per-object (building / greenhouse / superpixel) temporal subspace descriptors monitor
state change — active→abandoned greenhouse, intact→damaged building — with limited labels?*

Why later: real application (ties to the Ramdani-lab abandoned-greenhouse use case and to disaster damage),
and object units carry the semantic/spatial identity that pixel DS destroys. Needs polygons/proposals + a
labeled or weakly-labeled evaluation. Natural post-thesis extension toward the reconstruction goal.

Ranked sub-experiments:
1. **E5.1** Object proposals (SAM/superpixels/building footprints) → per-object multispectral time series → temporal DS state descriptor.
2. **E5.2** Post-classification change at object level vs direct subspace-state comparison.
3. **E5.3** Abandoned-greenhouse pilot with the Tsukuba project's manual labels as ground truth.

---

## P6 — KDS/KGDS nonlinear spectral change  **[PARK / component of P1]**

Question: *Does kernelized (nonlinear) DS improve temporal change representation over linear DS for
multispectral series?*

Why parked: KPCA-CD already exists (not novel alone); full-pixel KPCA is memory-heavy. Best used as a
**component inside P1** (E1.9) — kernelize the temporal date-window subspaces with sampling/Nyström — not as
a standalone thesis. Reference: TPAMI 2015 + Venus demo already in repo.

---

## P7 — Tensor / n-mode GDS for spectral–spatial–temporal satellite cubes  **[FUTURE]**

Question: *Can n-mode GDS keep spectral, spatial, and temporal axes separate instead of flattening, giving a
more faithful satellite subspace representation?* (Gatto et al., n-mode GDS, arXiv 1909.01954.)

Why later: theoretically elegant and very lab-aligned; potentially the strongest *method-development* paper
after P1 establishes the temporal signal is real. High implementation cost.

---

## §A — Method-construction menu (sample-unit choices, reused across P1–P7)

| Construction | "Set" / samples | Subspace ambient space | Preserves | Best for |
|---|---|---|---|---|
| date-window image-set (PRIMARY) | the T dates in a window | flattened region (B·\|R\|) | temporal modes of a region | P1 onset/recovery |
| SSA trajectory (per pixel) | Hankel rows of a 1-D index series | embedding length L | fine temporal dynamics | P1.7 localization |
| patch-vector | k×k×B patches | patch-feature space | local layout | P4 |
| local-window | pixels in a window | B | regional distribution | P4 |
| object/superpixel | pixels in an object, over dates | object×time | semantic unit | P5 |
| deep-feature | encoder embeddings per date | latent space | learned structure | P3.2 |
| tensor / n-mode | the cube itself | per-mode | all axes explicitly | P7 |

## §B — Evaluation-strategy menu (how to get verifiable signal without dense labels)

(Detailed verification ladder lives with P1's design doc; this is the index.)
- **Controlled synthetic** — inject known change → recoverable, fully labeled, reviewer-proof.
- **Known-event localization** — documented disaster dates (fire-perimeter DBs) → temporal-localization metric.
- **Independent recovery proxy** — NBR/NDVI recovery curve validates the second-order/recovery claim *only if*
  the proxy is not the same channel the DS fires on (watch circularity).
- **Negative controls** — stable / high-seasonality AOIs the method must stay quiet on.
- **Classical-temporal baselines** — SSA min-angle, CVA-over-time, PCA-diff-over-time.
- **Labeled special case** — OSCD (T=2) as a sanity anchor to the existing binary benchmark.
- **Lab-internal sanity** — equal-subspace edge case = 0 magnitude; consistency with MagTool/toolbox conventions.

---

_Update rule: when an experiment is run, move its result to `notes/experiments.md` / an experiment report and
mark it here. When a parked problem becomes active, give it its own design doc like P1's._
