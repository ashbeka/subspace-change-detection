# L3 irrigation switch (DS's verified niche, real labels) — NEGATIVE

Date 2026-06-18, branch claude/temporal-ds. Config: `temporal/configs/L3_irrigation_preregistered.json`
(frozen). Code: `temporal/experiments/real_irrigation.py`. 24 IrrMapper-labeled sites (6 switched-on, 6
switched-off, 6 constant-irrigated, 6 constant-dryland), Columbia Basin WA, dense S2 2016-2023.

## Result vs pre-registered claims `[experiment-evidence]`
| metric | AUC(switched vs constant) | verdict |
|---|---|---|
| **DS d1 (M-SSA)** | **0.417 (below chance)** | C1 FAIL |
| min-angle SSA | 0.382 | — |
| **NDVI seasonal-amplitude step** | **0.729** | beats DS decisively |
| trivial reflectance-diff | 0.500 | — |

C1 (DS AUC>0.7): FAIL. C3 (DS >= min-angle & NDVI): FAIL. C2 localize: median 1y, within +-1y 58%.

## Why DS failed even here
The DS change magnitude (`d1_max`) is dominated by the **seasonal cycle**, which rotates the subspace
strongly in EVERY site (irrigated or not, switched or not) -> constant sites score as high as switched
ones (d1_max 11-15 across the board). DS conflates ordinary seasonal oscillation with the structural
cycle gain/loss. A targeted one-line scalar (year-over-year NDVI seasonal-amplitude change) isolates the
actual change and wins (0.73 vs 0.42).

## Conclusion
Three pre-registered real-data tests now NEGATIVE: fire onset (NBR wins), fire recovery (no separation),
irrigation switch (NDVI-amplitude wins, DS below chance). DS was given its best-matched real regime with
real labels and still lost to a one-line scalar. The verified synthetic advantage does NOT transfer to
real satellite change detection, because real series carry a dominant seasonal cycle that the DS magnitude
cannot separate from structural change. **The honest, complete result is the DIAGNOSTIC paper.**

One principled (not yet tried) refinement that the failure mode points to: a PHASE-ALIGNED / same-season
year-over-year DS (compare summer-yearA subspace vs summer-yearB subspace) to remove the seasonal rotation
and isolate the cycle change. This is a legitimate construction improvement, not goalpost-moving, but it is
the LAST honest attempt before committing to the diagnostic write-up.
