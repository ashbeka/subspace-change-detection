# BET 2 / G1 — recovery-trajectory geometry gate: FALSIFIED (2026-06-21)

Script: `temporal/experiments/bet2_g1_trajectory.py` · `temporal/outputs/bet2_g1_trajectory/`
Classify recovery PATH-type (fast / slow / detour-through-another-material) from the time series. Killer null:
the full multi-band per-band trajectory. Pre-reg: docs/research/bet2_design.md.

## Result — 3-class path accuracy (chance 0.33)
| illumination | scalar_index curve | **multi-band trajectory** | trajectory GEOMETRY |
|---|---|---|---|
| 0.0 | 0.667 | **1.000** | 0.400 |
| 0.1 | 0.692 | 0.717 | 0.342 |

## Verdict — geometry CLOSED
- The trajectory GEOMETRY (subspace velocity + 2nd-order DS + geodesic along/orth + distance-to-baseline) is
  **near-chance (0.34-0.40)** — the WORST descriptor, beaten even by the scalar index curve. The per-date
  subspace barely encodes the recovery path; the geometric features do not discriminate fast/slow/detour.
- The **multi-band per-band trajectory wins decisively (1.0 clean)**. Per the pre-registered gate (geometry must
  beat the multi-band null), Bet 2's geometry angle is FALSIFIED — consistent with the honest prior and with the
  earlier temporal-trajectory failures (H-B 2nd-order DS, signal-subspace DS).
- G2/G3/G4 cannot rescue a near-chance descriptor; the detour path-type (the path-vs-endpoint case) was already
  included in G1 and geometry failed it.

## Honest secondary finding (valuable, but NOT geometry)
**Multi-dimensional recovery characterization beats single-index monitoring:** the multi-band trajectory
classifies recovery path-type at 1.0 where the standard scalar NBR/NDVI curve gets only 0.67. This is a real,
useful point for post-disaster recovery monitoring — but it is "use all the bands," not a subspace/geometry
contribution.

## Decision (per the user's plan)
Bet 1 (both pillars) and Bet 2 (geometry) are now closed. Subspace geometry is redundant/worse than a simpler
multi-band/scalar statistic in EVERY tested cell, including the two best post-hoc bets. Remaining: the H-C lead
(geometry on deep/foundation features) — engineering-heavy, and any win is attributable to the features, not the
geometry. The DIAGNOSTIC remains the strong, finished contribution.
