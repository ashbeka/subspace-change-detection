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

## 4. Open (adversarial verification in progress)

A 3-probe pass is attacking this: (A) stronger frequency-domain scalars (PSD/spectral entropy/full-ACF) —
does DS keep its noise edge vs the *best* scalar? (B) construction robustness (rank/energy/W/L) — is the
win knife-edge as the L1 one was? (C) is the win a special case, and does a trivial min-angle variant
("mean of all angles" / "sum of all angles" ≈ DS) capture the same advantage — i.e. is the real lesson
"aggregate all angles" rather than "use DS"? Verdict + honest final claim to follow before real-data work.
