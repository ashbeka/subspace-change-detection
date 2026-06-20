# T1 L1 — geometric change-TYPE characterization: REDUNDANT (folds into diagnostic) (2026-06-20)

Script: `temporal/experiments/t1_changetype.py` · `temporal/outputs/t1_changetype/`
The research-mining session's TOP task (T1, `docs/research/synthesis_specific_tasks.md`). Pre-registered L1
falsifier: does the 2nd-order DS geodesic ALONG/ORTHOGONAL split classify abrupt-vs-gradual change-TYPE better
than the scalar 2nd-difference of the mean spectrum (||Δ²m|| + its along/orth direction)? Construction = the
PROPER M-SSA signal-subspace trajectory (joint cross-band+temporal), not HB1's weak patch subspaces. 40 seeds.

## Result — classify abrupt vs gradual (AUC)
| feature | AUC |
|---|---|
| subspace orth/along ratio | 0.981 |
| subspace peak orth | 0.977 |
| scalar Δ²m orth/along ratio | 0.971 |
| **scalar Δ²m peak orth** | **1.000** |
| **scalar ‖Δ²m‖ magnitude** | **1.000** |

## Verdict — T1 REDUNDANT (pre-committed outcome)
The scalar 2nd-difference of the mean spectrum separates abrupt from gradual PERFECTLY (1.000); the subspace
geodesic split (0.98) adds nothing. Per T1's own pre-registered falsifier (synthesis §2.7), T1 collapses into a
**negative row of the diagnostic (T4)**: "the subspace geodesic along/orthogonal split is redundant with the
scalar 2nd-difference for satellite change-type." This confirms the HB1 prior AND the mining's redundancy prior,
now with the proper M-SSA construction — so it is not a construction artifact.

Reason: abrupt-vs-gradual IS an acceleration-magnitude distinction (abrupt = high ‖Δ²m‖, gradual = low), which a
scalar captures directly; the geodesic *decomposition* carries no extra change-type signal here.

## Significance + what remains
- Detection (every variant) AND characterization-via-T1 both fold into the diagnostic. The **DIAGNOSTIC (T4) is
  the contribution** — independently reached by experiments AND the parallel research-mining.
- T1's richer sub-angles (phase-shift vs seasonal-shape change; recovery direction-reversal; band attribution)
  were not separately tested, but the CORE killer-null falsifier failed → T1 is not a strong positive.
- The one remaining genuinely-novel POSITIVE bet is **T2 — RTW warp-invariant (phenological-phase-invariant) CD**
  (RTW never used in remote sensing; Sensei-endorsed). It has a clean go/no-go: does a warp-invariant subspace
  distance stay flat on a pure phase-shift (nuisance) and fire on a same-phase shape-change (real), where a
  phase-aware harmonic cannot? That test (synthesis §3) is the next experiment.
