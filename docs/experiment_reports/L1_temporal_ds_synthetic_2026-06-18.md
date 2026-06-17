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

## 7. Robustness (adversarial verification — IN PROGRESS)

A 3-probe adversarial pass is running to attack this result: (A) fairer gain-invariant baselines
(spectral-angle, NDVI-distance, per-date normalization) — if a one-line invariant baseline matches DS,
novelty is threatened; (B) harder data (textured multi-material tiles, sub-pixel registration jitter,
partial-region events, non-stationary confounds); (C) construction/hyperparameter robustness and whether
the win is an artifact of the unequal-dim magnitude generalization. **This section will be updated with
the verdict and the honest breaking conditions before the result is used in any claim.**

## 8. Next steps

1. Fold adversarial-verification fixes into the experiment and the design doc.
2. If the finding survives vs fair invariant baselines → proceed to L2 (real Sentinel-2 known-event
   localization via GEE) with the frozen construction.
3. If a simple invariant baseline matches DS → the contribution shifts to second-order/geodesic
   (acceleration, degradation-vs-recovery) where there is no trivial scalar equivalent.
