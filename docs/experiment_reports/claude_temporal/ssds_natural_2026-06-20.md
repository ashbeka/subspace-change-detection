# SSDS-natural — the lead does NOT replicate on real natural changes (2026-06-20)

Script: `temporal/experiments/ssds_natural.py` · `temporal/outputs/ssds_natural/`
Tests a_hat (M-SSA signal-subspace DS + learned D_N) on REAL CONTINUOUS natural changes (no splice artifact):
GERD reservoir filling (~2020, gradual land→water) and Creek Fire CA (Sept 2020, abrupt + recovery). D_N from
the pre-change normal years. Baselines: harmonic change-point, crude band-subspace velocity, NDVI mean-shift.

## Results (localization within ±120 days of the documented change; margin = peak/background)
| location | method | peak | days off | localized | margin |
|---|---|---|---|---|---|
| GERD reservoir (gradual) | a_hat (D_N) | 2022-08 | 760 | No | 1.23 |
| | harmonic_cp | 2024-06 | 1450 | No | 0.97 |
| | crude band-subspace | 2019-10 | 260 | No | 2.92 |
| Creek Fire (abrupt) | **a_hat (D_N)** | 2021-04 | 210 | **No** | **0.50** |
| | harmonic_cp | 2021-01 | 140 | No | 1.41 |
| | **crude band-subspace** | 2020-09 | 20 | **Yes** | **9.96** |
| | NDVI mean-shift | 2020-09 | 20 | Yes | 2.57 |

**a_hat localized 0/2 real natural changes.**

## Findings (blunt)
- **The splice win (ssds_validate: a_hat beats harmonic) did NOT generalize to real continuous changes.** The
  splice was an artificial concatenation that happened to be a phenology-structure change (a_hat's niche) AND
  carried a join artifact.
- **a_hat is insensitive to abrupt level/spectral changes.** On the fire it fails (margin 0.50 = below
  background) because within a 1.5-yr signal-subspace window a fire is just a trend, not an oscillation-structure
  change. The **crude band-subspace velocity nails the fire (margin 9.96, 20d)** — abrupt spectral change is the
  band/spectral subspace's job, exactly as the Tenerife case showed.
- **The reservoir (gradual multi-year fill) has no single change-point** — an ill-posed localization test; no
  method localizes it.
- **Neither case tests a_hat's actual niche:** a change in the seasonal OSCILLATION STRUCTURE (phenology — e.g.
  crop-rotation / rainfed→irrigated / forest→cropland). I could not find clean real data for that blindly
  (the Rondônia deforestation point changed in 2019, leaving too little pre-period for D_N).

## Verdict — LEAD DOWNGRADED to a narrow, unproven niche
The faithful signal-subspace DS works in synthetic (L0) and on the splice (its niche + artifact), but does NOT
beat simple/appropriate methods on the real changes available. Its niche — gradual phenological/oscillation
structure change — is narrow, and the common RS-CD targets (fire, damage, water = abrupt or signal-loss) are
owned by band-subspace/scalar methods. The project's recurring pattern holds. NOTE for the Gaza-damage motive:
DAMAGE is abrupt → the band-subspace (crude_TDS) is the right tool there and works cleanly (margin 9.96), not a_hat.

## Options
(a) Fair-test a_hat's true niche — find REAL phenology/oscillation-structure change data (crop rotation,
    rainfed→irrigated) with ≥2 yr pre-period. (b) Pursue the band-subspace-for-abrupt-damage thread (crude_TDS
    is clean on fire — but is it > SAM/CVA? unknown). (c) Consolidate the diagnostic (now very strongly supported).
