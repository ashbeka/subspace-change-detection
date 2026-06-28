# SSDS-phenology — the final fair shot FAILED (2026-06-20)

Script: `temporal/experiments/ssds_phenology.py` · `temporal/outputs/ssds_phenology/`
The user-chosen "final fair shot": test a_hat on its ACTUAL niche — a change in the seasonal OSCILLATION
STRUCTURE (not a level/spectral shift). Real Saudi irrigation-onset sites: flat desert (amp ~0.02, 2018-2022)
→ a NEW seasonal cycle emerges (amp 0.1-0.26, 2023-2024). D_N from the 5-yr flat pre-period (no leakage).
Metric: AUC separating post-onset vs flat pre-onset windows (detection of the emergence).

## Results — a_hat detects the onset on 0/3 sites, beaten by TRIVIAL baselines on all 3
| site | a_hat (D_N) | harmonic_resid | NDVI_meanshift | crude_TDS |
|---|---|---|---|---|
| saudi2 | 0.553 | 0.719 | **0.826** | 0.620 |
| saudi3 | 0.663 | 0.728 | **0.777** | 0.587 |
| saudi_jawf | 0.629 | 0.822 | 0.799 | 0.588 |

- **a_hat fails even on its home turf** (AUC 0.55–0.66, post/pre ratio ~1.0 — it barely fires when the cycle
  emerges). Beaten by harmonic deseasonalization AND by the trivial NDVI mean-shift on EVERY site.
- Reason: the flat desert pre-period → D_N is a NOISE reference → the DS departure when the cycle emerges is
  weak/unstructured. Meanwhile the change is an amplitude/level effect that NDVI mean-shift catches directly.

## CONCLUSION — the subspace-as-better-detector thesis is comprehensively falsified on real data
Tally of the faithful signal-subspace DS (a_hat) on REAL change:
- abrupt (fire): FAILS; the crude band-subspace / scalar win (margin 9.96).
- gradual signal-loss (reservoir): no method localizes; a_hat weak.
- phenology onset (a_hat's own niche): FAILS; harmonic + trivial NDVI mean-shift win (0/3).
- the only "win" was an ARTIFICIAL splice (niche-specific + concatenation artifact).
Combined with H-A (IR-MAD owns detection) and H-B-crude (redundant with the mean-vector), the evidence is now
overwhelming and coherent: **subspace geometry does not beat simple/appropriate baselines for satellite change
detection across every fair real-data test.** Further method-tuning to rescue it = the method-forcing we were
warned against.

## The contribution is the DIAGNOSTIC
"When and why does subspace geometry help (and fail) for satellite change detection — and when is it just
spectral angle / a mean-vector / harmonic-deseasonalization statistic?" This is now rigorously supported by a
full ladder of pre-registered experiments with trivial/standard nulls. It is the defensible, publishable result
(Sensei's bar: a first, honest, well-characterized finding). Next: consolidate the diagnostic paper from the
experiment_reports + CD_TAXONOMY + the construction ledger.
