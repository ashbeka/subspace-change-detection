# SSDS-S2 — faithful signal-subspace DS on a real S2 fire (Tenerife Arafo 2023) (2026-06-20)

Script: `temporal/experiments/ssds_s2_tenerife.py` · `temporal/outputs/ssds_s2_tenerife/`
Data: 84 cloud-masked S2 dates (2023-01..2024-06), 8 bands + NDVI + NBR, single Arafo burn region (n=1 event).

## Result (regular-grid resampled; fire = 2023-08-27, idx 36)
| method | peak | localized | margin (fire/background) |
|---|---|---|---|
| a_hat NDVI (D_N) | 2024-06-28 | No | 0.27 |
| a_hat M-SSA (D_N) | 2024-06-21 | No | 0.42 |
| SSA mean-angle (no D_N) | 2023-12-29 | No | 0.86 |
| **raw band-subspace velocity (crude)** | 2023-08-21 | **Yes** | 4.20 |
| CVA all-band | 2023-12-29 | No | 1.01 |
| NDVI |Δ| | 2023-02-13 | No | 1.82 |
| NBR |Δ| (domain index) | 2023-01-31 | No (resampling smeared the drop) | 2.71 |
(Irregular-sampling run: faithful methods also failed; raw-TDS peaked PRE-fire (wrong); NBR|Δ| localized on raw data.)

## Findings
- **The faithful Kanai signal-subspace DS FAILS to localize the abrupt fire** (margin < 1 — fire response below
  background), on both irregular and regular sampling. The **D_N reference does not help** (a_hat ≈ mean-angle).
- The **crude band-subspace velocity localizes** the fire after resampling (margin 4.2) but is FRAGILE (it
  peaked pre-fire on irregular data). A fire is an abrupt SPECTRAL-IDENTITY change — caught by the band/spectral
  subspace, NOT by the SSA TEMPORAL-structure subspace (whose home is oscillatory 1D signals like ECG).

## Critical caveat — this series cannot FAIRLY test the faithful method (under-powered, not refuted)
- The paper uses windows w≈220 over strongly periodic signals. 84 total dates force tiny windows (w=12) → the
  SSA signal subspace cannot stabilize. And <1 year of pre-fire data ⇒ D_N CANNOT learn a seasonal cycle — the
  whole point of D_N. So the seasonality-robustness claim was never actually testable here.
- Change-TYPE mismatch: the faithful temporal method targets changes in OSCILLATORY/temporal structure; a fire
  is an abrupt level/spectral step. Its fair home is GRADUAL/phenological change over MULTI-YEAR history.

## Verdict
For abrupt multispectral satellite change (fire), the faithful signal-subspace DS is the WRONG tool and fails;
a spectral/band representation or a domain index (NBR) is appropriate. The faithful method's fair test
(multi-year regular series + gradual/oscillatory change + learnable seasonal D_N) is PENDING DATA (needs a GEE
long-history fetch). Construction lesson (ledger): WHAT the subspace represents — spectral identity vs temporal
structure — must match the change type; this is the crux of "how we build the subspaces."

## Next options (decision point)
(a) GEE long-history fetch (multi-year, gradual change) → the faithful method's fair test.
(b) Pivot to the diagnostic contribution (now strongly supported across H-A, H-B-crude, H-B-faithful).
(c) Explore the band/spectral-subspace direction (GDS on band sets) for abrupt change vs spectral baselines.
