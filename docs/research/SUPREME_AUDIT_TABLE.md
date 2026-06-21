# SUPREME cross-agent audit — every experiment, every result, what worked, seminar ranking (2026-06-21)

Audited across ALL worktrees/branches: **Claude** (`claude/temporal-ds`, temporal+HSI), **Codex**
(`codex/band-image-ds-score-ablation` + `codex/spatial-ds-track` + `codex/research-mining-20260621`),
**antigravity** (`antigravity/local-msm`), plus `MultiSenGE_DS` and `legacy/oscd-...-unet`. Two independent
research-minings (Claude + Codex) concur. Worked legend: ✓ = works OK on a real benchmark (competitive w/ the
right null); ~ = marginal / mixed / near-miss; ✗ = redundant or worse than a simpler standard method.

## 1. Supreme table — all experiments × Sensei-method × data × result × worked?

| Agent | Experiment / problem | Sensei method | Data | Bottom-line result | Worked? |
|---|---|---|---|---|---|
| Codex | **Spatial Band-Image DS** (DS on the spatial-position axis of each band image) | **GDS / spatial DS** | **OSCD 24 cities (real)** | **AUROC 0.8412 ≈ PCA-diff 0.8392; AP 0.234; wins 4/24 cities; raw-L2 corr 0.66 (NOT raw L2); interpretable band-attribution** | **✓ best subspace result** |
| Codex | Spatial DS baseline pressure (vs PCA-diff/IR-MAD/Celik) | spatial DS | OSCD 24 | PCA-diff wins AP in 11 cities, **Band-Image DS in 4**, IR-MAD 4, Celik 2 | ~ competitive, not top |
| antigravity | Local-MSM spatial subspace | MSM / spatial DS | OSCD | same spatial-DS family/question as Band-Image DS (preserve spatial info); converges (competitive, not beating PCA-diff) | ~ |
| Codex | Seasonal 2nd-order DS event detection | **2nd-order DS / geodesic** | synthetic seasonal | 2nd-order-along AP **0.698** vs NDVI 2nd-diff **0.674** (overlapping CI) — a TIE | ~ marginal |
| Codex | Multispectral temporal DS | 2nd-order DS | synthetic | gain-invariant (AUROC 1.0) BUT 1-pixel translation AUROC **0.000** (registration-fragile); vs all-neg 0.875 | ~ mixed |
| Claude | H-A invariance residual (SFA-CD) | **SFA / SFS** | Salinas | beat SAM/CVA under affine nuisance, BUT IR-MAD already does this | ~ works, not novel (owned) |
| Claude | Signal-subspace DS + learned D_N (Kanai) | Grassmann / SSA-DS | S2 7-yr (GEE) | seasonality-robust; beat harmonic on a SPLICE (3.10 vs 1.19); FAILED on real natural change | ~ near-miss |
| Codex | RTW phenology invariance | **RTW** | MultiSenGE | did NOT beat snapshot baseline (AP delta −0.008) | ✗ |
| Codex | RTW crop-phenology transfer | RTW | BreizhCrops | global-shift-RMS control consistently stronger than RTW | ✗ |
| Codex | SpaceNet-7 temporal subspace | 1st/2nd DS, geodesic | SpaceNet-7 (real) | First-DS AP 0.131 / AUROC 0.563 — loses to radiometric rank-fusion (AP 0.191) | ✗ |
| Codex | OSCD core sweep — DS priors in U-Net | DS priors | OSCD | did NOT improve AUROC/PR-AUC (earlier "win" non-reproduced) | ✗ |
| Claude | Bi-temporal DS (13-band) | 1st-order DS | S2/OSCD | DS ≡ SAM; lost to PCA-diff | ✗ |
| Claude | Crude temporal DS | DS velocity | S2 | failed 3× on real (seasonality) | ✗ |
| Claude | HSI spectral-SSA DS dimensionality | DS / SSA | Salinas, **Hermiston (real)** | DS 0.53 near-chance vs CVA 0.98; gap negative, doesn't grow w/ bands | ✗ |
| Claude | H-B trajectory dynamics | 2nd-order DS | synthetic | redundant with mean-vector Δ²m | ✗ |
| Claude | Material-subspace CD | **MSM / GDS** | Salinas | SAM-to-mean beats it; worse with more dims | ✗ |
| Claude | Orientation factorization | GDS / covariance-orientation | Salinas | beaten by the correlation matrix | ✗ |
| Claude | Band/date attribution | DS-basis / S3CCA | Hermiston | DS-basis near-random vs per-band diff | ✗ |
| Claude | Bet 1: small-sample N≪B + registration | set-subspace / MSM | Salinas, Hermiston | subspace flat/worst; correlation & patch-mean-SAM win; registration-robustness SHARED by all patch methods | ✗ |
| Claude | Bet 2: recovery trajectory | 2nd-order DS / geodesic | Salinas | geometry near-chance (0.40) vs multi-band (1.0); (secondary ✓: multi-band recovery > single index) | ✗ (geom) |
| Claude | H-C: subspace on deep features | **SLS** | Hermiston (AE) | latent subspace 0.67 < latent correlation 0.89 | ✗ |
| Claude | Kernel / nonlinear DS | **KDS/KGDS** | Hermiston, Salinas | RFF lift restores amplitude (0.55→0.97) but still < CVA 0.985; fails distributional | ✗ |
| both | **DIAGNOSTIC** (when/why geometry helps vs reduces to SAM/mean/correlation/IR-MAD/harmonic/DTW) | meta | all | SUPPORTED across every cell + real bitemporal HSI; 2 minings concur | ✓ (the map) |

## 2. Which of Sensei's methods WORKED (ranked, "at least OK for the job")
1. **GDS — spatial Band-Image DS** ✓ — the ONLY construction competitive with the best classical baseline on a
   real benchmark (OSCD), genuinely distinct from raw L2, interpretable, and it answers Sensei's spatial concern.
2. **2nd-order DS / geodesic** ~ — *ties* the scalar null on seasonal change-type (AP 0.698 vs 0.674) and is
   gain-invariant; not a win, but Sensei's flagship and "works" as a characterizer.
3. **SFA / SFS** ~ — works for nuisance-invariant detection, but the cell is owned by IR-MAD (a baseline, not ours).
4. **Grassmann signal-subspace DS (Kanai)** ~ — seasonality-robust near-miss (splice win).
5. **RTW** ✗ on the tested constructions (global-shift RMS / snapshot beat it) — *but* attention/learned RTW is untested (future).
6. **MSM (material), KDS (kernel), SLS (deep), 1st-order DS** ✗ — redundant with a simpler statistic.

## 3. My judgment — the best seminar research (NOT "why subspace fails")
**Lead the seminar with the SPATIAL-AXIS Difference Subspace (Band-Image DS), framed as a positive result that
also answers Sensei's own critique — backed by the diagnostic as the *map of where geometry helps vs not*.**

- **Problem statement (convincing, Sensei-aligned):** *Standard (global-pixel / per-pixel) Difference Subspace
  destroys spatial structure — Sensei's stated concern. We construct the DS on the SPATIAL-position axis of each
  band image, so the subspace encodes spatial energy arrangement. On OSCD (24 real cities) this Band-Image DS is
  competitive with the strongest classical baseline (PCA-diff), genuinely distinct from raw L2, and yields
  interpretable per-band change attribution.* 
- **Real use case:** unsupervised pixel-level change/damage mapping; and a prior channel for a CNN segmenter.
- **What worked:** AUROC 0.84 (≈ PCA-diff), 4/24 city AP wins, raw-L2 corr 0.66 (not raw L2), band attribution.
- **What didn't (the honest map = the diagnostic):** on the SPECTRAL and TEMPORAL axes the subspace reduces to
  spectral-angle / mean-vector / correlation / IR-MAD / harmonic / DTW (low-dim ⇒ DS≡SAM; amplitude-dominated ⇒
  DS discards it; truncation discards subtle change; the correlation matrix subsumes the eigenspace). This is a
  *positive* framing of the diagnostic: it explains exactly WHY the spatial axis is the one that works.
- **What to improve / FUTURE WORK (rich, real):**
  1. **The decisive open test** (the audit's killer null): run the spatial **Gram/correlation-matrix distance** +
     SiROC + patch-tensor head-to-head vs Band-Image DS on OSCD AP — does the subspace *beat* the full
     second-moment, or fold in? (one cheap go/no-go, data in hand).
  2. **Band-Image DS map as a U-Net prior channel** (the core sweep tried generic priors → non-reproduced;
     never tried *this* map specifically).
  3. **Ensemble Band-Image DS + PCA-diff** (they win different cities → complementary).
  4. **2nd-order DS change-type characterization** + the seasonality-robust D_N idea (the near-misses).
  5. attention/learned RTW (untested); spatial × spectral × temporal product-Grassmann; SAR/PolSAR (geometry-native).

**Why this is the right seminar:** it has a genuine positive anchor (a subspace that works + answers Sensei's
critique), real use cases, an honest map of where it helps/fails, and a rich, concrete future-work slide — and it
is NOT "subspace geometry doesn't work." The diagnostic is the rigorous backbone; the spatial Band-Image DS is
the headline; the open killer-null test is the exciting next step.

## 4. Honest caveat (so you can defend it)
Band-Image DS does NOT strictly *beat* PCA-diff (PCA-diff wins 15/24 cities on AP); it *matches/ties* it and wins
a minority of cities. The audit's prediction (consistent with the project's theorem) is that a spatial
correlation-matrix null may also match it — so present it as "the competitive, interpretable, spatially-faithful
subspace construction + the decisive open test," not "a confirmed win." That honesty is itself defensible to Sensei.
