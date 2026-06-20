# SSDS-validate — the lead vs the STANDARD harmonic baseline (2026-06-20)

Script: `temporal/experiments/ssds_validate.py` · `temporal/outputs/ssds_validate/`
Tests the ssds_longseries lead (a_hat = M-SSA signal-subspace DS with learned D_N) against the proper strong
baseline for time-series SATELLITE change detection: **harmonic deseasonalization** (BFAST/CCDC family — fit the
seasonal cycle on the normal period, flag the residual; K=3 harmonics + trend, per band) plus a windowed SFA-CD.
Data: Iowa Corn Belt 7-yr S2, regular 10-day grid; Amazon for the splice.

FAIRNESS FIX: a deseasonalized residual is a STEP (stays elevated post-change), not a change-point spike, so for
a fair change-point comparison the harmonic residual is given its breakpoint form `harmonic_cp` = sliding
mean-shift of the residual (BFAST detects the breakpoint, not just elevation). a_hat is already a change-point score.

## Results
| | T1 seasonality \|corr dNDVI\| (lower=robuster) | T2 splice: localized / margin | T3 null-splice: localized / margin |
|---|---|---|---|
| **a_hat (M-SSA, D_N)** | **0.055** | **Yes / 3.10** | No / 1.96 |
| harmonic residual (K=3) | 0.149 | No / 1.34 | No / 1.42 |
| harmonic_cp (fair breakpoint) | — | No / 1.19 | No / 1.15 |
| windowed SFA-CD | 0.012 | No / 0.99 | No / 1.00 |

## Findings
- **a_hat BEATS the standard harmonic deseasonalization** (and windowed SFA) on localizing the change: it is the
  ONLY method whose global peak sits at the change (margin 3.10), with a clean real-vs-null separation
  (3.10 vs 1.96 = 1.6×). Harmonic and SFA fail to localize (margins ~1.0–1.3, peaks elsewhere).
- a_hat is also seasonality-robust (corr 0.055) — better than harmonic (0.149), comparable to SFA (0.012).
- Likely mechanism for the edge: a_hat (M-SSA) models the JOINT cross-band + temporal structure, whereas the
  standard harmonic fits each band independently → it misses a change carried by the cross-band structure.
  This is the genuine, theory-consistent advantage of the subspace representation (multi-dimensional ⇒ DS ≠ SAM).

## Caveats (still a LEAD, not a finished result)
- n=1 splice; the change (Corn Belt → Amazon) is LARGE and ARTIFICIAL — a subtle/natural change is untested.
- The windowed SFA-CD's poor showing (0.99) is likely my adaptation (SFA on temporally-autocorrelated window
  dates), NOT a fair defeat of SFA — do not over-read "beats SFA".
- True spatial IR-MAD needs per-pixel data (different structure); not tested here.

## Verdict
The faithful signal-subspace DS with a learned D_N reference, on multi-year real S2, **beats the standard
harmonic-deseasonalization change-point baseline** while being seasonality-robust and null-splice-clean. The
lead survives its make-or-break test. This is the strongest positive in the project so far and is consistent
with subspace theory (joint multi-band structure). NEXT (to convert lead → result): a NATURAL gradual change
(deforestation / irrigation onset), multi-case statistics (many change locations/types), and per-pixel IR-MAD.
