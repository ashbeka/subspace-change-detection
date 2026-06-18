# Real-data Gate-0 — Tenerife 2023 wildfire (NEGATIVE on onset, with insight)

Date: 2026-06-18. Branch `claude/temporal-ds`. Code: `temporal/experiments/real_gate0.py`,
`temporal/gee_fetch.py`. Construction FROZEN from `temporal/configs/L2_preregistered.json` (no tuning).

## Setup
- Event: 2023 Tenerife wildfire, ignition 2023-08-15 (Arafo / Corona Forestal pine forest, >14,000 ha).
- Burn AOI (28.370 N, 16.452 W, 400 m) + control AOI (Vilaflor pine forest, 28.150 N, 16.630 W).
- Live GEE fetch, cloud-masked Harmonized S2, 2023-01-01 → 2024-06-30. 84 valid dates (burn), cadence ~6.4 d.
- Frozen M-SSA: W=24 dates, L=12, rank=8, energy=0.99. Metric: d1(t) peak within ±5 d of event.

## Result `[experiment-evidence]`
| signal | onset localization | note |
|---|---|---|
| **NBR (domain index)** | sharp drop −2..+13 d (0.455 → −0.139) | localizes the fire to within days |
| **M-SSA d1 (full DS), frozen** | **+63 d, peak/median 1.2** | FAIL ±5 d; weak, smeared |
| trivial per-date reflectance step | +133 d | single-step detector is noisy/seasonal |
| **DS per-band attribution** | top band **B11 (SWIR)** | physically correct for a burn ✓ |
| control AOI d1 | peak/median 1.1 | burn barely exceeds control |

## Honest interpretation
**Gate-0 did not pass.** Two causes, both consistent with prior findings:
1. **Window-in-dates smears onset.** W=24 dates ≈ 154 real days at S2 cadence, so the past/present boundary
   blends ~5 months — any abrupt onset is smeared by months (the +63 d peak). Frozen from dense-regular
   synthetic; real S2 is sparse/irregular.
2. **A fire onset is a sharp level/signature shift — the regime where DS ≡ spectral angle** (L1 verdict).
   DS has no structural edge here; the domain index NBR wins decisively. This is *predicted*, not surprising.

What DID work: **attribution** (DS basis correctly flags SWIR/B11 for the burn) — the subspace captures the
burn signature even though d1 is not an onset detector.

## Conclusion (follows the pre-registered rule)
The pre-registration says: Gate-0 fail after honest, event-blind debugging → **invoke the diagnostic-paper
fallback** rather than keep drilling. We follow it. The honest, complete story is now:

> Temporal Difference Subspaces robustly beat the conventional single-angle SSA on **multi-mode dynamics**
> change and provide correct **band/mode attribution** (synthetic, verified; attribution confirmed on a real
> burn), but they are **not** a better detector than the domain index for **abrupt-onset** events — DS
> reduces to spectral-angle behavior for sharp shifts. This maps the method's true scope.

This becomes `docs/DRAFT_DIAGNOSTIC_PAPER_ACCV.md` (now anchored by real data, not just synthetic).

## Not pursued (to avoid goalpost-moving)
Post-fire **recovery dynamics** (gradual, the second-order/geodesic regime) is the one place DS could still
add real-data value over NBR — but testing it now would be moving the goalpost off the pre-registered onset
test, and it risks NBR circularity. Flagged as the single most promising *honest* next experiment, to be
pre-registered separately if pursued.
