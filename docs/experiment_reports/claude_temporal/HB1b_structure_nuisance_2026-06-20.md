# H-B-1b — The only places a subspace trajectory can beat the mean-vector (2026-06-20)

Scripts: `temporal/experiments/hb1b_structure.py`, `temporal/experiments/hb1b_verify.py`
Metrics: `temporal/outputs/hb1b_structure/`, `temporal/outputs/hb1b_verify/`
Follow-up to the H-B-1 negative (subspace trajectory curvature redundant with `||Δ²(mean spectrum)||`).

## C1 — direction-change under amplitude nuisance: NEGATIVE
turn-vs-straight AUC as global-illumination + per-band-gain nuisance grows:
| alpha | kappa | sc_d2 (raw mean) | scn_d2 (normalized mean) |
|---|---|---|---|
| 0.0 | 0.999 | 1.000 | 1.000 |
| 0.1 | 0.548 | 0.561 | 0.624 |
| 0.3 | 0.599 | 0.554 | 0.530 |
The subspace curvature collapses to ~chance under nuisance, **same as the mean nulls** — per-band gain is a
diagonal linear map that rotates subspaces too, so the subspace is NOT nuisance-invariant. No edge.

## C2 — mean-preserving structural change: REAL detection edge, but the novelty claim FAILS verification
Regimes (patch_i = A + s(t)·g_i·dvec(t); **mean = A always**): `stable` (fixed), `rotate` (spread DIRECTION
rotates, variance const), `scale` (variance grows, direction fixed).
| comparison | sub_vel | eig_angle | cov_frob | var_vel | mean nulls |
|---|---|---|---|---|---|
| rotate/stable (detect structural) | **1.000** | **1.000** | 0.698 | 0.605 | 0.58 |
| scale/stable (should IGNORE variance) | 0.920 | 0.924 | 1.000 | 1.000 | 0.89 |
| rotate/scale (isolate structural vs variance) | 1.000 | 1.000 | **1.000** | 1.000 | 0.91 |

- **Holds:** the subspace is the most SENSITIVE detector of mean-preserving distributional change
  (rotate/stable 1.0 vs cov-Frobenius 0.70, per-band-variance 0.61, mean-based ~0.58 = blind). The whole
  mean-based CD toolkit (CVA/SAM/IR-MAD/image-differencing) cannot see this; the subspace can. Likely edge
  over raw covariance-Frobenius = robustness to per-date sampling noise (Grassmann small-sample strength).
- **Fails:** the subspace is NOT scale-invariant (scale/stable 0.92) — it fires for mere variance growth.
  And covariance-Frobenius ALSO separates rotate from scale (1.0). So "the subspace UNIQUELY isolates
  structural-from-variance change" is FALSE; a covariance statistic does it too.

## Verdict (blunt)
**H-B (change-trajectory characterization via 2nd/1st-order Difference Subspace) is NOT supported as a
distinctive contribution.** Across H-B-1 + 1b: trajectory curvature is redundant with the mean-vector;
2nd-order DS (`sub_d2`) never fired; the subspace is not nuisance-invariant. The ONLY residual is a MODEST,
non-unique edge: subspaces are a sensitive/sample-robust detector of **mean-preserving distributional
change** — but that pivots away from "dynamics/2nd-order DS" toward distributional/covariance-based CD, a
likely-existing area. This needs the LITERATURE (Task B) to adjudicate novelty before any real-data spend.
Same recurring pattern as H-A: a simple/standard baseline matches the subspace; the subspace's real lever is
invariance or capturing structure the mean discards — not geometry-of-the-trajectory per se.

## Next
Do NOT drill more synthetic H-B. Shift weight to Task B research mining to answer: (a) is mean-preserving /
distributional / covariance-change detection an established niche, and where is its open gap? (b) where do
subspaces have a *documented* edge over mean/covariance baselines in HSI CD? Keep H-C (deep features)
as the remaining untested experimental hypothesis.
