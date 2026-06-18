# Pre-registered recovery test (Tenerife 2023) — NEGATIVE

Date 2026-06-18, branch claude/temporal-ds. Config: `temporal/configs/L2b_recovery_preregistered.json`
(frozen before outcomes; DS bands EXCLUDE NBR's B8/B12 as a circularity guard; visible-only strict variant).
Code: `temporal/experiments/real_recovery.py`. 251 (burn) / 227 (control) cloud-free S2 dates, 2021-2025.

## Result vs the 4 pre-registered claims `[experiment-evidence]`
| claim | result | verdict |
|---|---|---|
| C1 sustained disruption | burn d_pre sustained **0 d** vs control **303 d** | FAIL (backwards) |
| C4 control quiet | control sustained 303 d (wanted ~0) | FAIL |
| C3 beats min-angle | burn post−pre rise DS +0.052 vs min-angle +0.025 | both at noise level |
| C2 non-circular value | DS slope −1.4e-3 (main) / −2.1e-3 (strict); NBR +6.2e-4; raw +5.3e-4 | directional only, control slope larger |

burn d_pre post-fire median 6.015 vs pre-fire 6.013 → **negligible**; the fire leaves no clean signature in
the (B8/B12-excluded) multi-band temporal subspace, and seasonal/noise variability dominates d_pre.

## Conclusion
Both real-data tests failed: onset (NBR wins decisively) and recovery (d_pre does not separate burn from
control). The verified synthetic multi-mode-dynamics advantage does **not** transfer to this real wildfire
setting, because real change here is either a sharp signature shift (onset → DS ≡ spectral angle) or is
swamped by phenological variability (recovery). Per the pre-registered rule (now triggered twice), we
**commit to the diagnostic paper**: a rigorous map of when subspace geometry helps satellite change
detection (verified controlled multi-mode regime + correct attribution) and when it does not (real
wildfire onset and recovery). This honestly answers the project's founding question.

Caveat: one event / two AOIs / one construction. The negative pattern is consistent with synthetic theory,
but breadth (more events, other change types) would strengthen the diagnostic claim.
