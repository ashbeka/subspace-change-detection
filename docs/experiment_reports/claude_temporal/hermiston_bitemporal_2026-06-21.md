# BITEMP-HSI on REAL Hermiston (242-band bitemporal, 5-class change-type GT) (2026-06-21)

Script: `temporal/experiments/hermiston_bitemporal.py` · `temporal/outputs/hermiston_bitemporal/`
**The first REAL bitemporal hyperspectral test in the project** (all prior were synthetic / Salinas class-proxy).
Closes the long-data-blocked C1/C5 questions on a genuine benchmark.

## C1 — DETECTION: CLOSED on real data (the capstone diagnostic row)
| method | AUC (balanced subset) |
|---|---|
| **CVA** (L2 diff) | **0.981** |
| SAM (spectral angle) | 0.848 |
| IR-MAD | 0.562 |
| **DS (wavelength-Hankel spectral-SSA)** | **0.534 (near-chance)** |

DS−SAM gap vs band count: −0.29 (20) · −0.24 (40) · −0.03 (80) · −0.28 (160) · **−0.31 (242)** — negative
everywhere and does NOT grow with bands. **The dimensionality hypothesis ("DS beats SAM at 200+ bands") is
FALSIFIED on real data.** The Hermiston change is amplitude-dominated (crop↔soil reflectance shifts) → CVA wins;
DS is amplitude-invariant so it throws away the very signal that matters → near-chance. This confirms both
research-minings' prior (DS weaker at higher ranks) on a REAL benchmark, not a proxy. Decisive closure.

## C5 — ATTRIBUTION: different ranking, but not validated (likely partly noise)
- DS-basis band leverage vs per-band differencing: **corr = 0.051** (nearly orthogonal rankings — DS names
  DIFFERENT bands). DS top-5 [124,126,165,176,223] vs per-band-diff top-5 [79,80,97,98,217].
- Per change-type DS top-3: class1 [126,176,178], class2 [120,126,165], class4 **[239,240,241]**, class5
  [121,126,165]. Classes 1/2/5 share bands 126/165 (consistent); **class4 picks the noisy spectrum-edge bands
  239–241 — a red flag that the DS ranking is partly NOISE-driven, not physical change.**
- Verdict: "different ≠ better." Per Codex's gate (Challenge 2), attribution is a positive only if it recovers
  affected wavelengths MORE accurately/stably than per-band diff / sparse-PCA — UNVALIDATED here, and the edge-band
  artifact warns it may be noise. Not a positive yet.

## Overall verdict
**The decisive real-data evidence: subspace/DS detection is closed — it does not beat SAM/CVA/IR-MAD even at 242
real bands, and is near-chance on amplitude-dominated change.** This is the strongest single row of the
DIAGNOSTIC (T4): geometry adds nothing for detection on a real hyperspectral benchmark. The attribution thread
(C5) gives a different but unvalidated/possibly-noisy ranking — keep as a gated future probe, not a claim.
Positive hunt continues with T2 (RTW warp-invariance). Construction caveat: tiny classes (class 3, 79 px) and
edge bands are numerically fragile — guarded.
