# KB 02 — Temporal / sequence / warping / SSA mechanisms

Reading-depth legend in [`README.md`](README.md). These papers turn a *sequence* into a subspace and compare
sequences geometrically — the machinery for satellite **time-series** CD.

---

## Slow Feature Analysis for Change Detection — Wu, Du, Zhang, TGRS 2013 `[P]`
- **Method+math.** SFA finds projections $g_j(x)=w_j^\top x$ minimizing the temporal variance of the output on
  *unchanged* pixels: generalized eigenproblem $A\,W=B\,W\Lambda$ with $A=\Sigma_\Delta$ (covariance of the
  bitemporal difference $\hat x-\hat y$) and $B=\tfrac12(\Sigma_x+\Sigma_y)$. Slowest features = most temporally
  invariant = the no-change manifold; **change = the residual** $\mathrm{SFA}_j=w_j^\top\hat x-w_j^\top\hat y$,
  aggregated by chi-square $T=\sum_j(\mathrm{SFA}_j/\sigma_j)^2\sim\chi^2(N)$. Variants: **USFA** (unsupervised,
  whole image), **SSFA** (supervised on unchanged samples), **ISFA** (iterative reweighting by no-change
  probability — the IR-MAD analogue).
- **Signal/subspace idea.** The canonical **"model the invariant, the residual is change"** method on the
  *slow/temporally-invariant* rung of the invariance ladder.
- **Novelty / justify.** First SFA formulation for CD; suppresses radiometric/illumination drift so real change
  separates. Justified on Landsat ETM (Taizhou, Kunshan) with AUC + transformed divergence.
- **Baselines.** CVA, PCA-diff, MAD, **IR-MAD** — ISFA matches/beats IRMAD.
- **Transfer to CD.** Directly bitemporal-multispectral, real-labels testable. **Project status (H-A):** SFA-CD
  held AUC ~1.0 under global-affine nuisance where SAM/CVA/DS collapsed — but IR-MAD already does this, so the
  *detection* novelty is owned; survives only as attribution. **It is the mandatory invariance baseline any
  subspace-invariance claim must beat.**

## Feature-Sequence / Slow Feature Subspace (SFS) for action recognition — `[S]`
("Feature Sequence Representation Via SFA For Action Classification"; "Slow feature subspace: a video
representation based on SFA")
- **Method+math.** Apply SFA to a video's frame-feature sequence; the **span of the slow features** = a
  "slow feature subspace" representing the action; compare actions by canonical angles / MSM on these subspaces.
- **Signal/subspace idea.** SFA gives the *invariant* axes; their span is a compact, nuisance-robust subspace
  representation of a *temporal* set.
- **Novelty/justify/baselines.** Action recognition (KTH, etc.); baselines = SFA features, subspace/MSM, deep.
- **Transfer to CD.** A satellite analogue: per-region temporal slow-feature subspace; change = a shift in the
  slow-feature subspace (a *structural* seasonality change), not a per-date residual. Untested; data-blocked.

## Enhanced Grassmann Discriminant Analysis with Randomized Time Warping (RTW) — Hayashi/Souza/Fukui, Pattern Recognition 2019 `[S]`
(+ "Attention Mechanism in Randomized Time Warping", Hiraoka et al., arXiv:2508.16366 — Sensei asked the
student to review it and to consider incorporating RTW.)
- **Method+math.** **RTW** = a *randomized* generalization of dynamic time warping that represents a whole
  motion **sequence as a low-dimensional subspace**: sample many random monotonic time-warpings of the
  sequence, stack the warped feature vectors, take their principal subspace. Comparing two sequences →
  comparing two subspaces by canonical angles on the **Grassmann manifold**. **eGDA** projects class subspaces
  onto a **GDS** before Grassmann mapping (removes overlap, near-orthogonalizes → more discriminant). Recent
  result: RTW's mechanism is interpretable as a **self-attention** operation.
- **Signal/subspace idea.** **The RTW subspace is invariant to temporal warping (speed/phase of the sequence).**
  This is a *built-in temporal invariance the scalar nulls lack.*
- **Novelty/justify.** Sequence-as-subspace + warp-invariance + GDS-discriminant projection; validated on
  Cambridge gestures, KTH, UCF sports vs DTW, HMM, subspace/MSM, deep baselines.
- **Transfer to CD (the open lever).** **RTW has never been applied to satellite time series or CD.** Its
  warp-invariance maps *exactly* onto the recurring failure mode: **phenological phase shift** (the same
  seasonal cycle arriving earlier/later year-to-year or pixel-to-pixel is *nuisance*, not change). A
  warp-invariant subspace would detect a change in the **shape** of the seasonal cycle while ignoring **when**
  it happens — something harmonic deseasonalization (fixed phase) cannot do. Closest competitor: **DTW-based
  CD** (TWDTW / "Multiannual CD in SITS based on DTW") which *aligns then compares* rather than building a
  warp-*invariant* subspace, and is classification-framed, not subspace-geometric.

## Spotting Fingerspelled Words via Temporally-Regularized CCA (TRCCA) — Fukui et al., FG 2016 `[P]`
- **Method+math.** CCA between a query gesture sequence and a reference word sequence with a **temporal-
  smoothness (Laplacian) regularizer** on the canonical weights + orthogonalization of the shared background;
  the min canonical angle over a sliding window = a match score for **partial** sequence alignment.
- **Signal/subspace idea.** Find the *smooth, contiguous temporal interval* where two sequences correlate —
  partial pattern matching with structured (smooth) weights.
- **Novelty/justify.** Spotting (not just classifying) words in continuous sign video; baselines = CCA, DTW, HMM.
- **Transfer to CD.** Attribution over **time**: *which date interval* drives a change, with smooth weights.
  Sensei-mandated ("research S3CCA and TRCCA for satellite"). Interpretability lever, not a detector.

## S3CCA — Smoothly Structured Sparse CCA for Partial Pattern Matching — Kobayashi, ICPR 2014 `[P]`
- **Method+math.** CCA with a **Laplacian-smoothness + overlapping structured-sparsity** penalty on the
  canonical weight vectors along the *signal axis* (wavelength or time): selects a smooth, contiguous *window*
  of features; min canonical angle = match score, the nonzero weights = the contiguous interval driving the match.
- **Signal/subspace idea.** Structured sparsity → an *interpretable contiguous band/date interval*, not a
  scattered weight pattern.
- **Novelty/justify/baselines.** Partial pattern matching; baselines = CCA, sparse CCA, DTW.
- **Transfer to CD.** **Attribution over wavelength**: *which contiguous absorption bands* drive a change
  (red-edge vs SWIR-moisture vs mineral). Sensei-mandated. The interpretable-attribution contribution (Top-10 #4).

## Singular Spectrum Analysis — "Particularities and commonalities of SSA" (Golyandina-style review) `[S]`
- **Method+math.** Embed a 1-D series into a trajectory (Hankel) matrix; SVD → grouping → diagonal averaging
  (reconstruction). Components = trend + oscillations + noise; the leading singular vectors span the **signal
  subspace**. Model-free decomposition.
- **Transfer to CD.** The substrate for Kanai's signal-subspace DS and for "spectral-SSA along wavelength"
  (Hankel over bands). Lab pedigree (Sensei: Lincoln/Bernardo/Maha used SSA; Suto extended SSA via DS). Gives a
  genuinely multi-dimensional subspace from a single series → the dimensionality condition DS needs.

## Analysis of Temporal Tensor Datasets on the Product Grassmann Manifold — `[P]`
- **Method+math.** A temporal tensor (e.g. video / multi-mode sequence) is factorized into per-mode subspaces;
  each point lives on a **product Grassmann manifold** $\prod_k \mathrm{Gr}(d_k,n_k)$; similarity =
  $\sqrt{\sum_k s_k^2}$ over per-factor canonical-angle similarities.
- **Signal/subspace idea.** Fuse independent subspace factors (spectral, spatial, temporal) **without flattening**
  them into one vector — preserves structure.
- **Novelty/justify/baselines.** Tensor sequence classification; baselines = vectorized PCA/MSM, single-Grassmann.
- **Transfer to CD.** The substrate for Top-10 #8 (spectral×spatial×temporal product-manifold CD) — directly
  answers Sensei's "subspaces break spatial information" concern. Risk: spectral/spatial factors are correlated,
  weakening the "independent factors" selling point.

## Temporal-Stochastic Tensor Features for Action Recognition — `[S]`
- **Method+math.** Stochastic/temporal augmentation of tensor (multi-mode) features for action recognition on a
  Grassmann/tensor representation; akin to RTW's randomization applied to tensors.
- **Transfer to CD.** Reinforces the RTW idea (randomized temporal augmentation → robust subspace) in a
  multi-mode setting. Secondary.

## Grassmann Singular Spectrum Analysis for Bioacoustics Classification — `[P]`
- **Method+math.** SSA signal subspaces per audio segment → points on a Grassmann manifold → Grassmann
  classifier. SSA-subspace + Grassmann geometry end-to-end.
- **Transfer to CD.** A worked template for "SSA signal subspace → Grassmann comparison" on a real 1-D signal —
  the exact pipeline for per-pixel spectral-SSA or per-region temporal-SSA CD.

## Signal Latent Subspace (SLS) for Environmental Sound Classification — Mahyub et al. 2024 `[S]`
- **Method+math.** Build subspaces from **learned/deep latent features** (not raw signal); compare by canonical
  angles / product-Grassmann (SLS-PGM lineage). Rich features make the subspace meaningful + small-sample robust.
- **Signal/subspace idea.** Geometry **on top of** deep features — the "subspace ⇄ DNN" endpoint of the lab roadmap.
- **Transfer to CD.** Substrate for H-C: subspaces from a remote-sensing foundation-model's features; illumination-
  robust, small-sample. Senpai (Mahyub) method, first satellite application would be novel. Needs an extractor.

## Tensor Analysis with n-Mode Generalized Difference Subspace — `[S]`
- **Method+math.** Extends GDS to tensors: a per-mode (n-mode) generalized difference subspace separating
  classes along each tensor mode.
- **Transfer to CD.** GDS that respects the band/height/width/time modes of satellite data — the n-mode
  counterpart of product-Grassmann CD. Future track; complex.

---
### Honest novelty verdict for this cluster
The strongest *open* transfer is **RTW's warp-invariance for phenological-phase-invariant CD** (genuinely
unused in RS, Sensei-endorsed, attacks the exact failure mode) and the **S3CCA/TRCCA attribution** lever
(which bands/dates, Sensei-mandated). SFA-CD is strong but *owned* (it is a baseline, not our novelty).
### Next falsifiable step
Define a phenological-phase-shift nuisance test: does an RTW/warp-invariant subspace stay flat under a pure
phase shift of the seasonal cycle while a harmonic-deseasonalization residual false-alarms — and does it still
fire on a genuine cycle-*shape* change? If a fixed-phase harmonic model with a phase term matches it, the lever
collapses to harmonic deseasonalization.
