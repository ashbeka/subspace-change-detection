# SSDS long-series FAIR test — faithful signal-subspace DS on multi-year S2 (2026-06-20)

Script: `temporal/experiments/ssds_longseries.py` · `temporal/outputs/ssds_longseries/`
Data: GEE-fetched 7-yr S2 (2018–2024), Iowa Corn Belt (457 dates, strong annual cycle) + Amazon (contrast).
Resampled to a regular 10-day grid (254 pts). Windows w=36 (~1 yr), M=18, span=53, τ=6. D_N + z-stats from the
first ~2.9 yr only (no leakage). **The fair test the 84-date Tenerife series could not support.**

## TEST 1 — seasonality false-alarm (held-out normal years; LOWER = more robust)
| method | \|corr with \|dNDVI/dt\|\| | CV |
|---|---|---|
| a_hat NDVI (D_N) | 0.016 | 1.69 |
| a_hat M-SSA (D_N) | 0.055 | 1.43 |
| mean-angle NDVI (no D_N) | 0.141 | 0.51 |
| mean-angle M-SSA (no D_N) | 0.041 | 0.11 |
| crude band-subspace velocity | 0.039 | 2.25 |
| **NDVI \|diff\| (scalar)** | **0.583** | 1.13 |

- **All SUBSPACE methods are decorrelated from the seasonal cycle** (corr ≤ 0.06 for a_hat); the **scalar
  NDVI-diff is strongly seasonally driven (0.58)** — it false-alarms every green-up/senescence. So the
  subspace/temporal-structure representation IS seasonality-robust where the scalar index is not.
- Caveat: a_hat has high CV (1.4–1.7) — decorrelated from season but still spiky (non-seasonal, likely
  cloud-interpolation noise); mean-angle is flatter (CV 0.1–0.5). a_hat trades flatness for sensitivity.

## TEST 2 — controlled detection (splice Corn Belt → Amazon at 4.3 yr; D_N from Corn Belt normal)
| method | localized | margin (change/background) |
|---|---|---|
| **a_hat M-SSA (D_N)** | **Yes** | **3.10** |
| mean-angle M-SSA (no D_N) | No | 1.04 |
| crude band-subspace velocity | No | 2.05 |

- **a_hat (with D_N) is the ONLY method that cleanly localizes the real change.** The no-D_N mean-angle FAILS
  (margin 1.04) because the signal subspace also drifts during normal seasons → high background; the crude TDS
  also fails to localize. D_N subtracts the normal seasonal DS, so only the genuine change stands out.
- **This is D_N earning its keep — in DETECTION, not in seasonal correlation.** First real-data signal that the
  faithful method has a genuine advantage over both the crude version and the scalar null.

## NULL-SPLICE CONTROL (critical falsifier — splice Corn Belt to a phase-aligned earlier Corn-Belt stretch)
| | localized | margin |
|---|---|---|
| real splice (CornBelt→Amazon) | Yes | 3.10 |
| **null splice (no real change)** | **No** | 1.96 |
`a_hat` does NOT localize the no-change join → it detects the genuine change, not the concatenation artifact.
Honest caveat: the null-join still produces a modest response (1.96 vs 3.10), so real-vs-artifact separation is
~1.6× — detectable but not large. A NATURAL gradual change (no splice) is the cleaner proof.

## Verdict — GENUINE LEAD (first real positive), not yet proven
The faithful signal-subspace DS with a learned D_N reference, on multi-year real S2: (i) is seasonality-robust
like other subspace methods (unlike the scalar index), and (ii) uniquely detects a real change against the
seasonal background, beating the no-D_N mean-angle and the crude temporal-DS. This is the first encouraging
real-data result in the project. **But hold the champagne:**
- n=1 splice; the splice (Corn Belt→Amazon) is an ARTIFICIAL, abrupt, large change — not a natural gradual one.
- a_hat is spiky in normal periods (high CV) → non-seasonal false-alarm risk.
- The established invariance bars (IR-MAD, SFA-CD) are NOT yet compared — and IR-MAD owns detection elsewhere.

## Decisive validation
1. [DONE 2026-06-20] **Null-splice control PASSED** — a_hat does not localize a no-change join (1.96 vs 3.10).
2. **Multi-case:** several splice points + a second location pair → is localization/margin consistent or luck?
3. **Strong baselines:** IR-MAD / SFA-CD adapted to the sliding-window setting (a_hat must match/beat them) —
   CRITICAL, since IR-MAD owns detection elsewhere. Current nulls (crude TDS, mean-angle, NDVI-diff) are weak.
4. **Natural gradual change:** a real land-cover transition (deforestation / irrigation onset), not a splice —
   the cleaner proof (no concatenation artifact; the modest 1.96 null-response is a splice-specific artifact).
