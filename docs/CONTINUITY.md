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
**H-A — Invariance/nuisance-robust unsupervised CD via the invariant-subspace residual.** [DETECTION ANGLE
CLOSED, 2026-06-19] Floor: SFA-CD beat scalars under affine nuisance. BUT 1a gating showed **IR-MAD already
handles affine/nonlinear/spatial nuisance near-perfectly for distinct change** (AUC 0.98-1.00) — no gap for a
kernel/local method; and synthetic nuisance+change cannot reliably adjudicate this (results swing with design;
my IR-MAD even has an additive-change normalization bug). So H-A's *detection* novelty is dominated by IR-MAD.
Survives only as: (i) interpretable **attribution** (which spectral directions = nuisance vs change) IR-MAD's
chi-square lacks — modest; (ii) the **real bitemporal HSI-CD benchmark** as the only trustworthy test (blocked).
Do NOT generate more synthetic H-A experiments (drilling). Reports: hsi_HA_nuisance / hsi_HA_hard_nuisance _2026-06-19.

**H-B — Change-trajectory characterization via 2nd-order Difference Subspace.** Claim: don't just detect IF
change — characterize HOW (velocity = 1st-order DS, acceleration = 2nd-order DS, abrupt vs gradual via the
geodesic along/orthogonal split). *Defensible:* Sensei's flagship 2024 paper, first satellite application,
less-crowded niche (few methods characterize change dynamics), max novelty. *Risk:* geodesic-suppresses-
seasonality is unproven; needs ≥3 dates or spectral-SSA; no labeled HSI time series (synthetic + S2 only).
[STATUS 2026-06-20 — RE-OPENED after reading the papers] My crude H-B-1/1b synthetics showed trajectory
curvature redundant with the mean-vector 2nd-difference ||Δ²m|| AND a modest distributional-change edge
(rotate/stable AUC 1.0 vs cov-Frob 0.70). BUT see docs/SUBSPACE_CONSTRUCTION_LEDGER.md §3: (1) my 2nd-order DS
impl is FAITHFUL (verified vs arXiv:2409.08563) — it read ~0 only because my constructions had NO acceleration
signal (uniform rotation = constant velocity = zero acceleration); it is UNTESTED for its real purpose. (2) My
"temporal DS failed on S2" never implemented the lab's actual method — Kanai'23 (arXiv:2303.17802) uses SSA
signal subspaces + a LEARNED non-anomalous reference DS D_N (model-the-normal, flag-the-residual = the
invariance lever) + direction/magnitude indices; I never built D_N. So temporal-DS/H-B is NOT fairly closed.
NEW LEAD: implement the faithful Kanai signal-subspace DS (with D_N) + test where raw temporal DS failed (S2
seasonality) and on HSI. Reports: HB1_trajectory / HB1b_*_2026-06-20.
[UPDATE 2026-06-20] Faithful Kanai signal-subspace DS BUILT + L0-validated (AUC 0.98, beats min-angle;
ssds_L0_2026-06-20). On real S2 (Tenerife fire, ssds_S2_tenerife_2026-06-20): it FAILS to localize the abrupt
fire (margin<1); D_N no help; crude band-subspace velocity localizes (fragile); NBR domain index works. BUT
under-powered — 84 dates force tiny windows, <1yr pre-fire ⇒ D_N can't learn seasonality; and fire is an abrupt
SPECTRAL change (band-subspace's job), not the temporal-oscillatory change the method targets. Fair test needs a
GEE multi-year regular series with a GRADUAL change. Construction lesson: subspace must represent the right thing
(spectral identity vs temporal structure) for the change type. DECISION POINT — see report's 'Next options'.
[GENUINE LEAD 2026-06-20 — ssds_longseries] Fetched 7-yr S2 (Iowa Corn Belt + Amazon, GEE). FAIR test of the
faithful signal-subspace DS with learned D_N: (1) seasonality-ROBUST — a_hat corr w/ seasonal transitions 0.055
vs scalar NDVI-diff 0.58 (scalar false-alarms every green-up; subspace doesn't); (2) DETECTION — a_hat+D_N is
the ONLY method that localizes a real change (splice CornBelt→Amazon, margin 3.10) vs no-D_N mean-angle (1.04,
fails) and crude TDS (2.05, fails) → D_N earns its keep in DETECTION; (3) NULL-SPLICE control PASSED (a_hat
doesn't localize a no-change join, 1.96 vs 3.10). FIRST real positive in the project. NOT yet proven: splice is
artificial (1.6x real-vs-artifact sep), n=1, and IR-MAD/SFA bars NOT yet compared (IR-MAD owns detection
elsewhere). NEXT: IR-MAD/SFA sliding-window baselines + a NATURAL gradual change (deforestation/irrigation) +
multi-case. This is the live lead — pursue it. Report: ssds_longseries_2026-06-20.
[LEAD SURVIVES make-or-break 2026-06-20 — ssds_validate] a_hat BEATS the STANDARD harmonic deseasonalization
(BFAST/CCDC; K=3, given its fair change-point/mean-shift form) AND windowed SFA-CD: a_hat is the ONLY method
that localizes the splice (margin 3.10 vs harmonic_cp 1.19, SFA 0.99), seasonality-robust (0.055 vs harmonic
0.149), null-splice clean. Likely edge = M-SSA models JOINT cross-band+temporal structure that per-band harmonic
misses (theory-consistent: multi-dim ⇒ DS≠SAM). Strongest positive so far. Caveats: n=1 splice (large/artificial
change), windowed-SFA showing likely my poor adaptation, per-pixel IR-MAD untested. NEXT to convert lead→result:
NATURAL gradual change (deforestation/irrigation) + multi-case + per-pixel IR-MAD. Report: ssds_validate_2026-06-20.
[LEAD DOWNGRADED 2026-06-20 — ssds_natural] The splice win did NOT replicate on REAL continuous natural changes.
a_hat localized 0/2: on the Creek Fire (abrupt) it fails (margin 0.50) while the CRUDE band-subspace velocity
nails it (margin 9.96, 20d off) — abrupt spectral change is the band-subspace's job; a_hat is insensitive to
level/spectral shifts (a fire is just a trend inside its 1.5-yr window). On GERD reservoir (gradual multi-year
fill) there is no single change-point to localize. Neither tests a_hat's actual niche (a change in the seasonal
OSCILLATION STRUCTURE = phenology), which I could not get clean real data for. So a_hat's niche is NARROW +
UNPROVEN; the splice was niche-specific + join-artifact. Recurring pattern holds. For the Gaza DAMAGE motive
(abrupt) the band-subspace is the right tool, not a_hat. Report: ssds_natural_2026-06-20. RE-GROUND/decision next.
[LEAD CLOSED 2026-06-20 — ssds_phenology, the user-chosen FINAL FAIR SHOT] Tested a_hat on its ACTUAL niche
(seasonal-OSCILLATION-STRUCTURE change = real Saudi irrigation onsets, flat desert→emerging cycle, 5yr flat
pre). a_hat FAILS on 0/3 (AUC 0.55-0.66), beaten by harmonic deseasonalization AND the trivial NDVI mean-shift
(0.78-0.83) on EVERY site. So a_hat fails on abrupt (fire), gradual signal-loss (reservoir), AND its own niche
(phenology) — the only "win" was the artificial splice. ⇒ the subspace-as-better-DETECTOR thesis is
COMPREHENSIVELY FALSIFIED on real data (with H-A IR-MAD + H-B-crude mean-vector). THE CONTRIBUTION IS THE
DIAGNOSTIC: "when/why does subspace geometry help/fail for satellite CD, and when is it just spectral-angle /
mean-vector / harmonic-deseasonalization." Next: consolidate the diagnostic paper. Report: ssds_phenology_2026-06-20.

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
1. **[DONE 2026-06-19 — H-A POSITIVE]** Controlled nuisance test on Salinas: SFA-CD held AUC ~1.0 under
   global-affine radiometric nuisance while SAM(0.64)/CVA(0.50=chance)/DS(0.51) collapsed (+0.34 over best
   scalar). Plain shape-DS also collapsed → the win is from MODELING-THE-INVARIANT, not shape-comparison.
   Report: `docs/experiment_reports/hsi_HA_nuisance_2026-06-19.md`. **Caveat:** global-affine is the easy case
   (SFA/IR-MAD are built for it); SFA-CD itself is established → not yet OUR novelty. NOVELTY GAP / NEXT:
   1a. **Harder nuisance** — nonlinear (BRDF/atmospheric) + spatially-varying; linear SFA fails there →
       test KERNEL (KSFA/KDS) and local/patch invariant-residual (where geometry/nonlinearity wins). THE gap.
   1b. **GDS-common-subspace** as the invariant model vs SFA-CD — does the subspace version match AND add
       band/structure ATTRIBUTION (nuisance directions vs change directions) the chi-square lacks?
   1c. Add **IR-MAD** as the established affine-invariant bar (SFA/GDS must match/beat it).
   1d. Real bitemporal HSI-CD benchmark (real nuisance) once a `.mat` is downloaded.
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

## 9. NOTE/KNOWLEDGE ACCESS RULE (important)
The user keeps adding raw notes (apple_notes, slack_notes_to_myself, slack_messages_with_sensei, etc.) to the
**MAIN repo** and has Codex ingest them — these live at the ABSOLUTE path
`E:/research_projects/subspace-change-detection/docs/source_records/final_organization_2026-06-12/` (and the
ingested `notes/` there). This worktree's own copies are STALE. RULE: always read the user's latest notes/
knowledge from the **main-repo absolute path** (same disk, always readable from any session root). Write our
experiments/method-docs to THIS worktree (claude/temporal-ds). notes = read from main repo; code+docs = write here.
