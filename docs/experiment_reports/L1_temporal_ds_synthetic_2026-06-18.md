# L1 — Temporal Difference Subspace on Synthetic Multispectral Time Series

Date: 2026-06-18. Branch: `claude/temporal-ds`. Status: **preliminary; under adversarial verification.**
Code: `temporal/`, `tests/test_temporal_ds.py`, `temporal/experiments/synth_injection.py`.
Outputs (gitignored): `temporal/outputs/L1_synth/` (metrics.json + fig1..fig5).

## 1. Purpose

L1 of the verification ladder (`docs/DESIGN_TEMPORAL_DS_ACCV2026.md` §11.3): a **fully-labeled,
zero-circularity** synthetic test of whether the temporal Difference Subspace responds to a known
injected change and is robust to radiometric confounds — before spending any real-data effort. The
ground truth (event time, region, magnitude) is known by construction.

## 2. Setup

- **Scene** (`temporal/synth.py`): 64×64, 10 bands, 60 dates, 4 materials in a blocky layout.
  An abrupt material→"burn" conversion is injected in a region at `t0`. Realistic confounds added to
  every date: global multiplicative **seasonal gain**, per-band additive **haze**, gradual **spectral
  drift**, and **sensor noise**. The confounds are the point — a good detector must spike on the
  structural event, not the radiometric swings.
- **Constructions compared** (per 8×8 tile, subspace trajectory over dates):
  - `spatial`: one subspace per date from the tile's (bands × pixels) matrix.
  - `temporal`: sliding-window subspace from the (bands·pixels × T_w) date-stack.
  - × centering on/off; energy-based rank (keep 99% variance).
- **Scores**: DS velocity `d1(t)`, DS distance-to-baseline `d_pre(t)`, second-order off-geodesic `orth`.
- **Mandatory null baselines** (§11.4): trivial raw-reflectance difference; SSA minimum-angle.
- **Metrics**: changed-**state** detection AUC (all tiles × dates), onset localization error,
  localization IoU, event/confound contrast.

## 3. L0 correctness (prerequisite)

`tests/test_temporal_ds.py`: 8/8 pass. `[experiment-evidence]`
Identical→0; rotation-within-span→0; orthogonal→2k; known angle→2(1−cosθ); diff-subspace of identical
empty; Karcher of identical = the subspace; constant-velocity geodesic→~0 acceleration; accelerating
geodesic→acceleration peaks at the speed change and correlates with |d/dt d1|.

## 4. Two methodological findings (the load-bearing ones)

1. **Centering destroys signal in homogeneous tiles.** A spatially uniform tile (one material) has
   near-zero variance after centering, so the subspace becomes **noise-dominated** and DS is blind to
   the event (state AUC 0.496 ≈ chance). This is the *same* failure that sank bi-temporal global-pixel
   DS. Fix: use the lab **autocorrelation-PCA convention (uncentered)**, which preserves the dominant
   material direction. `[experiment-evidence]`
2. **Rank must match the true rank, or DS is noise-dominated.** Forcing a fixed rank (e.g. 4) on a
   genuinely rank-1 window pads the subspace with arbitrary noise directions; DS then measures noise
   churn, not change (e.g. DS = 5.69 under a *pure gain* change that should give 0). Fix:
   **energy-based rank** (keep components to 99% variance, lab-faithful `cvlPCA`). After this fix, DS = 0
   under pure gain. `[experiment-evidence]`

## 5. Results (best construction: `temporal` window, uncentered, energy-rank, T_w=6–8)

**Mechanism — gain invariance** `[experiment-evidence]`
- DS magnitude under a pure multiplicative gain change = **0.0**; raw mean-abs distance = 0.136.
- The subspace is exactly scale-invariant; raw differencing is not. This is *why* DS resists confounds.

**Changed-state detection (AUC over all tiles × dates)** `[experiment-evidence]`
| construction | DS AUC | trivial AUC | onset loc-err (DS) | IoU (DS) |
|---|---|---|---|---|
| temporal, uncentered | **0.998** | 0.712 | 3 | **1.0** |
| spatial, uncentered | 0.998 | 0.75 | 1 | 0.33 |
| temporal, centered | 0.828 | 0.712 | 14 | 0.0 |
| spatial, centered | 0.496 | 0.75 | 14 | 0.0 |

**Confound-robustness sweep (the headline, fig5)** `[experiment-evidence]`
As seasonal-gain amplitude grows 0 → 0.45:
- DS distance-to-baseline AUC: 1.0 → 1.0 → 1.0 → 0.93 → 0.87 → **0.86** (robust).
- trivial raw distance AUC: 1.0 → 0.94 → 0.63 → 0.49 → 0.49 → **0.49** (collapses to chance).
- They cross near amplitude ≈ 0.1. With no confound, trivial is perfect (1.0) and DS is also strong;
  DS's value appears precisely when radiometric confounds are present — the real-world regime.

**Magnitude sweep (H4)** `[experiment-evidence]`
DS state-AUC monotone 0.81 → 0.998 with injected magnitude; trivial 0.44 → 0.75. DS event/confound
contrast scales 3.9 → 135.

**Figures**: fig1 event-onset curves; fig2 magnitude sweep; fig3 localization map vs region; fig4
recovery scenario (d_pre rises then falls); fig5 confound-robustness crossover.

## 6. What this does and does NOT show

Shows (on synthetic): the temporal DS, with the right construction (uncentered window, energy-rank), is
a confound-robust changed-state descriptor that beats the trivial raw-difference null when radiometric
confounds are present, and exactly invariant to multiplicative gain. The mechanism is understood, not
incidental.

Does NOT show: anything about real Sentinel-2 data; that DS beats *confound-robust* baselines (spectral
angle / NDVI-distance) — that is exactly what the adversarial verification (below) is testing; that the
synthetic confound model matches reality.

## 7. Robustness — adversarial verification VERDICT (the L1 win was largely an artifact)

A 3-probe adversarial workflow attacked the §5 result and **falsified the dramatic claim.** `[experiment-evidence]`

- **First-order DS ≡ spectral angle (SAM) at the operating point.** At `center=False, energy=0.99` the
  selected rank collapses to ~1; rank-1 DS magnitude = `2(1−cosθ)` = monotone in spectral angle.
  `Spearman(DS, SAM) = 0.9988`. **Force rank k≥2 and DS drops to chance (~0.52) even with no confound** —
  the multidimensional subspace algebra contributed *nothing* on this synthetic.
- **Fair one-line baselines match or beat DS.** SAM ties DS exactly (0.862 vs 0.858); per-date global
  normalization then differencing (`norm_l1`) beats DS in every regime and seed (flat 1.0). "DS=0 under
  pure gain" is shared by cosine/SAM/normalize — not unique to subspaces.
- **The "collapse to 0.49" was a single-seed (seed=0) cherry-pick**; over ≥8 seeds the trivial baseline
  averages ~0.67, never chance.
- **Fragilities:** sub-pixel registration jitter hits DS ~5–10× harder than raw differencing and wrecks
  onset localization (1→10 dates); DS is **blind to subspace-preserving (brightness-only) change**
  (AUC 0.465). The confound generator was global-only, structurally favoring any gain-invariant method.

Verdict: **GO-WITH-FIXES** with a narrower claim. Mandatory fixes adopted: report ≥8 seeds; use SAM /
cosine / norm_l1 as the fair baselines (raw L1 only as a labeled "no-correction" reference); add the
fixed-rank ablation and DS-vs-SAM correlation; add jitter + a subspace-preserving event type.

## 8. The gate experiment — does the SUBSPACE machinery beat scalars anywhere? (partial yes)

`temporal/synth_dynamics.py` + `experiments/dynamics_gate.py`: detect a **dynamics change** (SSA
trajectory subspaces, past vs present) where mean/variance are preserved. Detection AUC, ≥6 seeds: `[experiment-evidence]`

| mode | SSA-DS | SSA-min-angle | mean-shift | var-shift | acf1-shift |
|---|---|---|---|---|---|
| amp_collapse | 1.00 | 1.00 | 0.18 | 1.00 | 1.00 |
| **freq_shift** (mean&var preserved) | **1.00** | 1.00 | 0.15 | 0.26 | 1.00 |
| noise_replace (mean&var preserved) | 1.00 | 1.00 | 0.57 | 0.85 | 1.00 |
| trend_onset | 0.68 | 0.82 | 1.00 | 0.60 | 0.92 |

- **Genuine niche found:** on `freq_shift` the temporal subspace beats the best scalar (mean/var) by
  **+0.74 AUC** — a structural dynamics change that level/variance/SAM detectors are blind to. This is
  the satellite-relevant case (loss/shift of phenological cycle structure) and the right home for DS.
- **Open weakness:** `SSA-DS == SSA-min-angle` (Δ=0.0) on every mode, and `acf1` (a one-liner) also hits
  1.0 on the structural modes. The *full* Difference Subspace does not yet beat the conventional
  single-angle SSA or a simple temporal scalar on these clean changes.

## 9. Honest standing conclusion (answers the project's founding question)

The experiments converge on a precise, rigorous answer to *"are we forcing subspace methods?"*:

> **The Difference Subspace adds value over scalar/angular baselines only when the state or dynamics is
> genuinely MULTI-DIMENSIONAL and the change is STRUCTURAL (in that multi-dim structure), not a level or
> single-direction shift. When a region is spectrally low-rank (≈one signature) or the change is a
> signature/level shift — the common satellite case — DS collapses to spectral angle and is "forced."**

This is itself a publishable, defensible finding (and exactly the kind of honest result Sensei's "first
and unique trial, need not be top performance" bar allows). It also defines the ONLY paths to a
non-forced positive contribution.

## 10. The remaining decisive test (GO / honest-negative fork)

Find a regime where the **full DS beats min-angle AND acf1 AND mean/var** — i.e. where multiple canonical
angles genuinely matter:
1. **Compound/subtle dynamics changes** (multi-harmonic phenology; a harmonic appears/disappears; phase
   decoupling) — Kanai's regime where DS>min-angle (they got 0.923 vs 0.846 on real signals).
2. **Multi-material region mixing change** (genuine spectral rank ≥2; the *mix* changes, not one swap).
3. **Second-order / geodesic recovery** on a multi-dimensional state, where degradation-vs-recovery
   direction has no scalar equivalent — the genuinely novel, advisor-aligned (Fukui 2024) piece.

If one of these shows seed-averaged separation of DS over all simple baselines → proceed to real
Sentinel-2 (L2). If none does → the honest thesis is the negative/diagnostic result in §9.
