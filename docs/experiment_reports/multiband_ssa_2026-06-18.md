# Multivariate SSA — the multi-dimensional regime where full DS beats min-angle

Date: 2026-06-18. Branch: `claude/temporal-ds`. Status: **strong preliminary; under adversarial verification.**
Code: `temporal/synth_multiband.py`, `temporal/experiments/multiband_gate.py`.
Outputs (gitignored): `temporal/outputs/multiband_gate/`.

## 1. Why this experiment

The L1 adversarial verdict proved first-order spectral DS ≡ spectral angle for low-rank states, and
demanded a regime where the *full* (multi-angle) Difference Subspace beats the conventional min-angle SSA
**and** simple scalars before any real-data work. This experiment supplies it, using a genuinely
multi-dimensional construction: **multivariate SSA** (stack per-band Hankel trajectory matrices → a joint
signal subspace) on multi-band index series with **multi-mode structural change** (a harmonic disappears;
mean and per-band variance preserved, so level/variance detectors are blind).

## 2. Results (multi-seed, std ≤ 0.05)

**Detection AUC by change mode** `[experiment-evidence]`
| mode | full DS | min-angle | multi-lag ACF | variance | mean |
|---|---|---|---|---|---|
| harmonic_dropout | **1.00** | 0.50 (chance) | 1.00 | 0.06 | 0.84 |
| freq_split | 0.85 | **1.00** | 0.99 | 0.41 | 0.33 |
| recovery | 0.97 | 0.59 | 1.00 | 0.82 | 0.57 |

**Noise sweep (harmonic_dropout) — the decisive figure** `[experiment-evidence]`
| noise | full DS | min-angle | best scalar | DS − min-angle |
|---|---|---|---|---|
| 0.10 | 1.00 | 0.73 | 1.00 | +0.27 |
| 0.25 | 1.00 | 0.51 | 1.00 | +0.49 |
| 0.50 | 1.00 | 0.46 | 0.98 | +0.54 |
| 0.80 | 0.99 | 0.45 | 0.70 | +0.54 |
| 1.20 | 0.75 | 0.45 | 0.64 | +0.30 |

**Attribution** `[experiment-evidence]`: the difference-subspace per-band energy identifies the changed
band at **100%** (chance 16.7%, n=150) on `freq_split`. No scalar detector provides this.

## 3. Interpretation

- **The full DS decisively beats the conventional single min-angle** on distributed multi-mode change
  (1.0 vs chance), reproducing Kanai et al. 2023's argument (aggregate evidence across canonical angles)
  on satellite-like multivariate dynamics. This is the within-family contribution that justifies the
  subspace construction over the conventional SSA scalar.
- **Noise robustness:** at noise ≥ 0.8 the full DS beats *both* min-angle and the best scalar — the
  multi-angle aggregation is a genuine SNR advantage.
- **Interpretability:** DS-basis attribution localizes the changed band perfectly — a capability scalars
  structurally lack (Soto-san's per-joint motion contribution, transplanted to bands/modes).
- **Honest nuance / limits:** DS is **not** universally best. On a single-dominant-mode change
  (`freq_split`) min-angle wins (1.0 vs 0.85). The contribution is regime-specific: DS wins on
  *distributed* multi-mode change, robust to noise.

## 4. Adversarial verification — VERDICT (this one survives; the claim narrows but holds)

Three probes attacked it (`temporal/experiments/probe_spectral_scalars.py`, `probe_mbprobe_angles.py`,
`probe_mb_construction.py`). Unlike the L1 win, this one survives — with an honest, narrower claim. `[experiment-evidence]`

**(A) Stronger frequency-domain scalars.** Added PSD-L2/L1, spectral entropy, full autocorrelation, and an
*oracle* 2nd-harmonic-power detector. Result on the harmonic_dropout noise sweep (DS vs best strong scalar):
| noise | DS | best strong scalar | margin |
|---|---|---|---|
| 0.25 | 1.00 | 1.00 (spec_ent / acf_full) | **+0.00 (TIE)** |
| 0.80 | 0.99 | 0.67 (spec_ent) | **+0.32 (DS)** |
| 1.20 | 0.74 | 0.58 (spec_ent) | **+0.15 (DS)** |
→ On *clean* data strong scalars tie DS. DS's genuine edge is **noise robustness**, not raw power.

**(B) Construction robustness.** DS-AUC = 0.94–1.00 across W∈{16,24,32}, L∈{8,12,16}, rank∈{4,8,12};
DS−min-angle margin a stable **+0.45 to +0.57** everywhere. **Not knife-edge** — the opposite of the L1
result (which collapsed to chance at rank≥2).

**(C) Is it "DS" or just "aggregate the angles"?** Across all modes/noise, `ds_mag ≡ mean_all ≡ sum_1mcos`
identically (the DS magnitude *is* the aggregate of all canonical angles). So the detection power is
"aggregate ALL canonical angles" — a one-line generalization of single-angle SSA, which *is* the DS
magnitude. And the win is **not** a harmonic_dropout cherry-pick: DS beats min-angle on 4 of 5 distributed
modes (amp_redistribute is the exception — there DS and min-angle both fail, only ACF catches it).

## 5. Honest, defensible claim (verified)

> The multivariate temporal Difference Subspace — aggregating evidence across **all** canonical angles
> between past/present signal subspaces — detects **distributed multi-mode structural change** in
> satellite-like time series **robustly under noise**, substantially beating the conventional single-angle
> SSA at every noise and construction setting, and beating the best frequency-domain scalar once noise is
> non-trivial (≥0.8). Its basis **uniquely attributes** the change to specific bands/modes (100% vs 16.7%
> chance). Caveats, stated up front: on clean data strong spectral scalars tie it; the detection magnitude
> equals an aggregate of canonical angles (not a new scalar); and it is regime-specific (fails on
> amplitude-redistribution change). Second-order/geodesic recovery decomposition adds an interpretability
> axis no scalar provides.

**Gate decision: GO** to real Sentinel-2 data with this narrowed claim (noise-robust multi-mode detection
+ attribution + recovery), since a genuine multi-dimensional regime where the subspace machinery robustly
beats min-angle AND scalars now exists and is construction-stable. Next: a real S2 dynamics case via GEE
(needs Earth Engine credentials), pre-registering W/L/rank from §4(B).
