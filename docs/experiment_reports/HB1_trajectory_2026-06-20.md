# H-B-1 — Subspace change-trajectory geometry vs scalar dynamics (2026-06-20)

**Status: NEGATIVE for the naive claim. Mechanism real but dominated by a trivial vector baseline.**
Script: `temporal/experiments/hb1_trajectory.py` · metrics: `temporal/outputs/hb1_trajectory/metrics.json`

## Pre-registered claim
On a multi-date spectral-patch trajectory, the difference-subspace-of-difference-subspaces curvature
`kappa(t)=magnitude(DS(S_{t-1},S_t), DS(S_t,S_{t+1}))` (and 2nd-order DS magnitude) recovers the change
TYPE — separating a DIRECTION-change (`turn`) from a SPEED-change (`accel`) and from constant motion
(`straight`) — where SCALAR dynamics cannot. Matched-speed construction so straight/turn have identical
per-step speed. Falsifier: a scalar null separates the regimes as well as the subspace → geometry adds nothing.

## Result (40 trajectories/regime; full table in metrics.json)
| comparison | kappa | d2 | sc_acc | **sc_d2** | total_change |
|---|---|---|---|---|---|
| L0 clean: turn/straight | 0.960 | 0.534 | 0.616 | **1.000** | 1.000 |
| L0 clean: turn/accel | 0.963 | 0.504 | 1.000 | 0.993 | 1.000 |
| L1 salinas: turn/straight | 0.569 | 0.525 | 0.547 | **1.000** | 1.000 |
| L1 salinas: turn/accel | 0.503 | 0.518 | 1.000 | 0.846 | 1.000 |

- **Mechanism exists (L0):** kappa spikes at the turn (mean curve 2.66 vs ~1.0 baseline). The geometry
  genuinely senses a change of change-direction.
- **But it is beaten by a trivial vector null:** `sc_d2 = ||Δ²(mean spectrum)|| = 1.000` separates turn from
  straight perfectly; kappa = 0.960. On realistic Salinas, kappa collapses to 0.569 while sc_d2 stays 1.000.
- **2nd-order DS magnitude `d2` did NOT fire** for the turn (0.50–0.53) — the k=2 uncentered-PCA 2nd-order
  magnitude is noise-dominated here; it does not carry the direction signal in this setup.
- `total_change`=1.000 confounded (turning shortens net displacement) — so turn/straight also differ in total
  displacement; another reason a plain magnitude separates them. Matched only per-step speed, not net path.

## Root cause (the real lesson)
"Scalars can't see direction-change" is FALSE. The patch **mean is a vector**; its 2nd-difference
`||Δ²m||` is exactly the change in the velocity *vector* → it captures curvature/direction-change directly.
Only the magnitude-only scalar `sc_acc` (change in *speed*) is blind to a turn. So the standard vector
baseline (patch-mean trajectory + finite differences) already recovers the change type. The subspace
trajectory adds nothing over it for direction-change in a clean mean. **Same structural pattern as
`finding-ds-equals-spectral-angle` and IR-MAD-owns-detection: the simple vector baseline captures the
target; geometry is redundant.**

## Where the subspace CAN still win (next falsifier → H-B-1b)
The subspace is only non-redundant where the mean-vector null MUST fail:
1. **Amplitude/illumination nuisance** — corrupts the mean-vector trajectory (sc_d2 spikes everywhere) but
   not the amplitude-invariant direction subspace. Test: turn-vs-straight under per-date illumination; does
   kappa beat sc_d2 as nuisance grows?
2. **Mean-preserving structural change** — composition/spread changes while the patch MEAN stays put
   (invisible to mean/CVA/SAM = essentially the whole CD field). Test: `stable` (fixed spread) vs
   `structure` (spread direction rotates, mean fixed); does subspace velocity/kappa detect it while sc_d2 ≈ 0.5?

## Verdict
Naive H-B-1 (subspace trajectory beats scalar dynamics for change-type) **NOT supported**. The geometric
mechanism is real but redundant with the patch-mean vector trajectory. H-B survives ONLY if the subspace
wins on invariance (case 1) or mean-preserving structure (case 2) — the cases the mean-vector cannot touch.
Proceeding to H-B-1b to test exactly those. Do NOT take this construction to real data; it is dominated.
