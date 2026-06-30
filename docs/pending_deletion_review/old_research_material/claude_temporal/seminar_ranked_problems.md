# Research problem statements we tackled — results + seminar ranking (2026-06-21)

Ranked for a SEMINAR to Sensei + senpais (positive results NOT required; weight = genuine interest × richness of
future work × defensibility × Sensei-fit). Every result is from a committed, pre-registered experiment report.

## Top tier — the seminar headline + the genuinely-interesting directions with rich future work

| # | Problem statement | What happened (result) | Future-work richness | Seminar value |
|---|---|---|---|---|
| **1** | **DIAGNOSTIC: when/why does subspace geometry help vs reduce to spectral-angle / mean-vector / correlation / IR-MAD / harmonic / DTW for satellite CD?** (the Representation × Operator × Decision map) | **SUPPORTED.** Geometry redundant/worse in *every* tested cell, on real bitemporal HSI, against the right nulls; 2 minings concur. | ★★★★ benchmark + protocol; extend to more datasets/sensors; formalize the DS≡SAM theory + invariance ladder | **HEADLINE / backbone.** Most defensible; Sensei's "novel first characterization" bar exactly. |
| **2** | **Seasonality-robust temporal CD via signal-subspace DS with a LEARNED non-anomalous reference D_N** (Kanai 2023 → first satellite application) | **NEAR-MISS.** Seasonality-robustness *worked* (corr 0.055 vs scalar 0.58); beat harmonic deseasonalization on a controlled splice (3.10 vs 1.19); but did NOT replicate on real *natural* change. | ★★★★★ the richest: close the splice→natural gap; multi-year gradual change; the D_N "model-the-normal" idea that did work | **Flagship future-work.** The most engaging arc (a method that almost beat the standard + exactly why it didn't). |
| **3** | **Factorized local change characterization: mean \| dispersion \| covariance-ORIENTATION** (interpretable "what kind of change", Codex Ch1) | **NEAR-MISS.** The factorization mechanism worked synthetically (orientation isolated scale-invariantly); the correlation matrix beat the *subspace* version on real spectra. | ★★★★ interpretable change decomposition; tie dispersion/orientation to physical damage vs recovery stages | **Genuinely interesting** — a new way to *read* change; ties to the post-disaster motive. |
| **4** | **Multi-dimensional damage/RECOVERY characterization** (stages/rate/heterogeneity from a time series) | **MIXED → a real non-geometry positive.** Trajectory *geometry* near-chance (0.40); but **multi-band recovery characterization beat the standard single index (NBR/NDVI) 1.0 vs 0.67.** | ★★★★ multi-dimensional recovery monitoring; real recovery data; the Gaza recovery motive | **Applied + motive tie-in.** A genuine positive (just not the subspace). |

## Methodology contribution (transferable)

| # | Problem statement | Result | Future work | Seminar value |
|---|---|---|---|---|
| **5** | **A pre-registered killer-null protocol for evaluating geometric CD methods** (systematically exposes representational redundancy) | Used throughout; repeatedly caught over-claims | ★★★ turn into a reusable evaluation standard | Methods-minded slide; strengthens credibility. |

## Evidence rows (the negatives that BUILD the diagnostic — supporting, not standalone)

| # | Problem statement | Result |
|---|---|---|
| 6 | Hyperspectral spectral-DS **dimensionality threshold** (does DS−SAM grow at 200+ bands?) | NEGATIVE on real Hermiston (DS 0.53 near-chance vs CVA 0.98; gap negative, shrinks). Decisive real-data row. |
| 7 | Bet 1: **small-sample (N≪B)** set-subspace + **registration robustness** | NEGATIVE (subspace flat/worst; correlation & patch-mean-SAM win; registration robustness shared by all patch methods). |
| 8 | Material-signature **subspace (MSM/GDS)** CD | NEGATIVE (SAM-to-mean beats it; worse with more dims). |
| 9 | Contiguous-band **attribution** (S3CCA / DS-basis, Codex Ch2) | NEGATIVE (DS-basis near-random vs per-band diff). |
| 10 | H-C: geometry on **deep/autoencoder features** | NEGATIVE (latent subspace < latent correlation; deep features don't beat raw). |
| 11 | H-A: invariance-residual CD (SFA/GDS) | NEGATIVE for novelty (owned by IR-MAD/SFA/ACD). |
| 12 | H-B: change-trajectory dynamics (2nd-order DS) | NEGATIVE (redundant with mean-vector Δ²m). |
| 13 | Spectral-subspace for **abrupt damage** | NEGATIVE (modest/fragile vs SAM/CVA; beaten by NBR). |
| 14 | Bi-temporal DS / crude temporal DS (the origin) | NEGATIVE (DS≡SAM; seasonality). The method-forcing we corrected. |

## Recommended seminar arc (using #1–#4)
1. **Question + motivation:** the field/lab assumes subspace geometry is a better CD family; post-disaster
   reconstruction needs robust, interpretable change. *Is geometry actually better — and if not, where is it
   redundant, and what IS the right tool?*
2. **The map (the contribution, #1):** Representation × Operator × Decision; the regime table (which simple
   statistic wins per change-type and the structural reason).
3. **The most promising directions + honest near-misses (#2, #3):** seasonality-robust D_N temporal CD (what
   worked, what didn't); the factorized change-characterization (a new way to read change).
4. **A real applied positive (#4):** multi-dimensional recovery characterization > single-index monitoring.
5. **Future work (rich):** close the splice→natural gap for D_N; physical interpretation of the factorization;
   SAR/PolSAR (geometry-native domain); kernel/nonlinear & tensor subspace variants; a CD-evaluation benchmark.

## Honest one-liner for "the best to present"
**#1 (the diagnostic) is the defensible backbone; #2 (seasonality-robust D_N temporal CD) is the most genuinely
interesting story with the richest future-work section.** Present #1 as the result and #2/#3/#4 as the promising
directions — that combination is "so-what"-proof and gives a real future-work slide.
