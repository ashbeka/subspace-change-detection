# CONTINUITY — master resume state (read this FIRST in any new session)

Last updated 2026-06-19. Purpose: a fresh session (new chat / new subscription / context reset) can continue
the research with zero loss of the intellectual thread. Companion: `STATUS.md` (terse), `docs/METHOD.md`,
`docs/RESEARCH_DIRECTIONS_TOP10.md`, `docs/CD_TAXONOMY.md`, the experiment reports, and the memory files.

## 0. Where work lives
- Branch **`claude/temporal-ds`** in worktree **`E:/research_projects/sccd-claude`** (Codex owns the main dir
  `E:/research_projects/subspace-change-detection` on its own branch — do NOT switch the main dir's branch).
- Run: `cd E:/research_projects/sccd-claude && E:/research_projects/subspace-change-detection/.venv/Scripts/python.exe -m temporal.experiments.<name>`
- GEE live: `ee.Initialize(project='subspace-change-detection')`. Branch pushed to origin.

## 1. The journey (one paragraph)
Project began **method-first** (Fukui lab has Difference Subspace; searched for a satellite home → forcing).
Bi-temporal DS on 13-band Sentinel-2 failed (DS ≡ spectral angle). Pivoted to **temporal DS on S2 time
series** → 3 pre-registered REAL-data failures (fire onset, fire recovery, irrigation switch) — DS lost to
one-line scalars because the seasonal cycle dominates. Re-grounded around **"hyperspectral satellite = a
high-dimensional spectral SIGNAL"** (8-paper deep-mine). First real-HSI test (Salinas 204-band class proxy):
**L0 viability passed** (spectral subspace rank>1, DS⊥SAM) but the **detection proxy was NEGATIVE** (DS lost
to SAM/CVA; the dimensionality-threshold prediction was falsified). Then a grounded analysis of SAM/CVA
reframed the whole direction (§3).

## 2. Verified findings (hard evidence — do not re-litigate)
- **DS ≡ spectral angle for low-dim spectra** (13-band S2: corr 0.99). At 204 bands DS ⊥ SAM (corr −0.03) and
  spectral-SSA subspace rank is genuinely >1 — so the construction is NOT degenerate at high band count.
- **Subspace methods repeatedly lose to simple scalars on REAL data** (4× now: 3 S2 + Salinas proxy). The
  synthetic→real gap has burned us every time — pre-register + always report the trivial/scalar null.
- **On Salinas (clean, no nuisance): CVA 0.96 ≥ SAM 0.92 ≫ DS 0.70**, DS−SAM widens *negative* with bands.

## 3. The KEY reframings (the intellectual gold — newest, most important)
- **Geometry is NOT a top-level CD family — it is a *mechanism*.** Direct comparison factorizes as
  **REPRESENTATION × COMPARISON-OPERATOR × DECISION**. SAM = (raw spectrum)×(angle); DS = (subspace)×(canonical
  angle). Subspace geometry is one (representation, operator) choice that appears ACROSS families (direct,
  temporal, learned-feature, hybrid). **Do not silo it; insert it where it has a structural edge.** Geometry-
  vs-CNN is a false dichotomy; geometry-*inside* (on deep features, in temporal trajectories) is the move.
- **SAM/CVA grounding (why they win, where there's room):** CVA = magnitude (amplitude-sensitive, needs
  radiometric normalization); SAM = direction (amplitude-invariant). Together ≈ the full difference vector.
  **They are beaten in the literature ONLY by stronger INVARIANCE** (IR-MAD/MAD = affine-invariant; SFA =
  slow/invariant) — i.e. methods that *model the no-change distribution and flag the residual*, not by "more
  angles." → **Plain shape-DS competes on SAM's home turf and loses (Salinas confirmed). The room is in
  invariance/nuisance-modeling.**
- **The unifying strategy across IR-MAD/SFA/GDS/anomalous-change:** *model the invariant/common component;
  the residual is change.* This is the cure for the seasonality/nuisance failure and the defensible niche.
- **Research posture:** be **problem-first, method-aware** (the original sin was method-first/forcing).

## 4. The THREE live hypotheses (kept, to be iteratively improved + experimented; from Top-10 #2/#3/#8)
**H-A — Invariance/nuisance-robust unsupervised CD via the invariant-subspace residual.** Claim: model the
no-change/common/slow subspace (GDS-common / SFA-slow / IR-MAD), measure change as the residual outside it;
beats SAM/CVA *specifically under date-to-date illumination/atmospheric/sensor nuisance* (where SAM/CVA fail).
*Defensible:* this is the only regime the grounding says subspaces can win; testable on bitemporal HSI-CD (has
nuisance) and on controlled nuisance-injection. *Falsifier:* if it does not beat SAM/CVA/IR-MAD under nuisance.
*This is the most defensible — START HERE.*

**H-B — Change-trajectory characterization via 2nd-order Difference Subspace.** Claim: don't just detect IF
change — characterize HOW (velocity = 1st-order DS, acceleration = 2nd-order DS, abrupt vs gradual via the
geodesic along/orthogonal split). *Defensible:* Sensei's flagship 2024 paper, first satellite application,
less-crowded niche (few methods characterize change dynamics), max novelty. *Risk:* geodesic-suppresses-
seasonality is unproven; needs ≥3 dates or spectral-SSA; no labeled HSI time series (synthetic + S2 only).

**H-C — Subspace geometry on deep/foundation features (SLS-style).** Claim: build subspaces FROM a remote-
sensing foundation-model/CNN's features (not raw bands) and compare by DS/canonical angles; rich features make
the subspace meaningful + illumination-robust + small-sample. *Defensible:* modern, competitive (SLS-PGM
lineage), answers "geometry-inside-deep-learning." *Risk:* needs a feature extractor; more engineering.

## 5. Data state
- HAVE: Salinas 204-band AVIRIS (`data_hsi/salinas.mat`, gitignored). Cached S2 irrigation/Tenerife series.
- BLOCKED (USER ACTION): labeled bitemporal HSI-CD benchmarks (Bay Area/Santa Barbara AVIRIS 224-band;
  Hermiston 5-class / River / Wetland Hyperion) — CiTIUS gitlab clone/archive corrupt, SicongLiuRS github 404,
  GEE Hyperion no coverage. **Download a `.mat` via browser → drop in `data_hsi/` →**
  `python -m temporal.experiments.hsi_dimensionality data_hsi/<file>.mat` (runner is built + auto-detects format).

## 6. Next experiments (precise, runnable — in priority order)
1. **H-A controlled nuisance test (no new data):** take Salinas; synthesize "date 2" = Salinas + structured
   radiometric nuisance (per-band gain/offset, atmospheric haze, illumination scaling) + injected real changes
   on a subset; test whether SFA-CD/GDS-residual beats SAM/CVA *as nuisance grows* (SAM should hold, CVA should
   degrade, the invariant-subspace method should hold or win). This directly tests H-A's core claim with data
   in hand. Pre-register; report SAM/CVA/IR-MAD nulls.
2. **H-A on real bitemporal HSI-CD** (once a benchmark `.mat` is downloaded): SFA-CD / GDS-residual / DS vs
   SAM/CVA/IR-MAD, AUC. The decisive real test.
3. **H-B:** spectral-SSA subspace trajectory + 1st/2nd-order DS on a multi-date series (synthetic dynamics +
   S2); the geodesic along/orthogonal split as abrupt-vs-gradual.
4. **H-C:** extract features from a small RS encoder, build subspaces, DS — only after H-A/H-B.

## 7. Discipline (non-negotiable — burned 4×)
Pre-register construction + falsifier BEFORE looking at labels; ALWAYS report the trivial/SAM/CVA null beside
the method; never claim a synthetic win transfers to real without the real test; the honest diagnostic (Top-10
#10: "when does subspace geometry help CD, and when is it just spectral angle") is the guaranteed-publishable
failsafe under every hypothesis.

## 8. Resume command
Say: **"continue on claude/temporal-ds (worktree E:/research_projects/sccd-claude) — read docs/CONTINUITY.md;
start with H-A experiment #1."** Memory files auto-load; git log + reports + this doc reconstruct everything.
