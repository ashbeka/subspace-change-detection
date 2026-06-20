# KB 03 — Remote-sensing change detection & hyperspectral baselines

Reading-depth legend in [`README.md`](README.md). These are the *competitors and the niche*: the methods any
subspace CD must be benchmarked against, plus the multitemporal-HSI-CD reviews and their open challenges.

---

## The "model-the-invariant, flag-the-residual" baselines (the invariance ladder, with citations)
- **CVA** — change vector analysis: magnitude+direction of the raw difference vector. No invariance (rung 0).
  Amplitude-sensitive; needs radiometric normalization. `[P-internal]`
- **SAM** — spectral angle: amplitude-invariant direction. Rung 1 (amplitude). **First-order DS collapses to
  this** at low band count. `[P-internal]`
- **MAD / IR-MAD** — Nielsen: CCA between the two dates; canonical variates' differences (MAD) are
  **affine/linear-invariant**; IR-MAD iteratively reweights by no-change probability + chi-square. Rung 2
  (affine/decorrelated). **Owns affine-invariant detection** (AUC 0.98–1.00 for distinct change in this
  project's gating). `[P-internal]`
- **SFA / ISFA / KSFA** — slow/temporally-invariant residual (see KB 02). Rung 3. `[P]`
- **Anomalous Change Detection (ACD) family** `[S]` — the canonical *nuisance-invariant* HSI CD family, and the
  **direct competitor set for any subspace-invariance claim**:
  - **Chronochrome (CC)** — Schaum & Stocker: global least-squares linear predictor of date-2 from date-1; large
    residual = anomalous change. Models the pervasive (nuisance) change, flags the rest.
  - **Covariance Equalization (CE)** — whitening/dewhitening so the two dates share statistics; residual = change.
  - **(Whitened) TLSQ / hyperbolic ACD** — Theiler: equivalent to subspace-RX in the stacked space; **whitened
    TLSQ is coordinate-invariant, and special cases equal CCA and optimized CE.** This is the rigorous
    "subspace + invariance" CD that already exists.
  - **SFA-ACD** (Wu/Du, Neurocomputing 2014); **Sketched Multi-view Subspace Learning (SMSL)** for HACD
    (arXiv:2210.04271) — subspace learning for anomalous change.
  - **Benchmark:** Viareggio 2013 (airborne, 127-band, ACD-designed; gated access).
- **Law (grounded + now externally cited):** methods beat CVA/SAM chiefly by climbing this ladder — *more
  invariance, not more angles*. Plain shape-DS sits at rung 1 and cannot beat SAM on its home turf.
  **⇒ the invariance-*detection* lever is CLOSED; it is owned by IR-MAD/SFA/ACD.**

## A Subspace-Based Change Detection Method for Hyperspectral Images — Wu/Du et al., IEEE 2013 (doc 6459550) `[S]`
- **Method.** Treats the date-2 pixel as a target and builds a **background subspace** from the date-1 pixel +
  neighbourhood; change = the residual outside the background subspace (a subspace-RX / target-detection form).
- **Transfer/novelty boundary.** **Subspace CD for HSI already exists** — but as *background-subspace target
  detection*, NOT as canonical-angle / difference-subspace / Grassmann-geodesic comparison. The DS/GDS/geodesic
  angle is still unoccupied (see `closest_methods_novelty.md`).

## Hyperspectral Image CD Based on the Balanced Metric / band-selection / unmixing CD `[S]`
- Recent HSI-CD threads: balanced metric (PMC11858855), end-to-end band-selection nets, spectral-unmixing +
  CNN CD, multiobjective sparse-unmixing CD. **Theme:** the open problem is *exploiting* the high spectral
  dimension (band redundancy) — which is exactly where a genuinely multi-dimensional subspace could matter.

## Spatial Context Awareness for Unsupervised CD in Optical Satellite Images (SiROC-lineage) — Kondmann et al. `[P-internal/S]`
- **Method.** Models each pixel by its **spatial context** (a ring/neighbourhood prediction); the prediction
  residual across dates = change. Unsupervised, registration/illumination-robust via spatial modelling.
- **Baselines.** PCA-Kmeans, CVA, deep unsupervised CD; strong on OSCD/multisensor.
- **Transfer to CD.** The spatial-context competitor (and a SiROC-style baseline the student bookmarked). Any
  "spatially-aware subspace" must beat it. Shows spatial modelling, not spectral geometry, carries OSCD.

## Battle Damage Detection via Pixel-Wise T-Test on Sentinel-1 (PWTT) — open-access `[S]`
- **Method.** Per-pixel two-sample t-test on a stack of pre vs post SAR intensities; map damage by the t-statistic.
  Simple, robust, label-free; built for conflict/disaster (the project's deep motive).
- **Baselines.** Log-ratio, coherence change.
- **Transfer to CD.** The **damage-motive scalar null** for abrupt change: a subspace damage detector must beat
  a per-pixel t-test on a temporal stack. (For abrupt damage the project found a crude band-subspace velocity
  localizes a fire cleanly — but it was never compared to PWTT/SAM/CVA on real damage.)

## Adaptive Regularized Low-Rank Tensor Decomposition for HSI Denoising & Destriping `[S]`
- **Method.** Low-rank tensor model of the HSI cube + regularization separates signal (low-rank) from
  noise/stripes (residual).
- **Transfer to CD.** Not CD, but (a) a preprocessing step that could make spectral subspaces cleaner, and (b)
  evidence that the **low-rank-signal + residual** decomposition is the dominant HSI prior — the same "model the
  structured part, the residual is the anomaly" logic the invariance lever uses.

## Multitemporal-HSI-CD reviews — Liu et al. "A Review of CD in Multitemporal Hyperspectral Images" (IGRSM 2019); 2024 DL-CD reviews; 2025 HSI-LCCD survey `[S]`
- **Method families catalogued.** Image-algebra (diff/ratio/CVA), transform (PCA/MNF/MAD/IR-MAD/SFA),
  classification/post-classification, spectral-unmixing CD, tensor/low-rank, deep (Siamese/transformer/SSL),
  ACD.
- **Open challenges (the ranked list in `challenges_ranked.md` is drawn from these):**
  1. **Exploiting high spectral dimensionality** without band-redundancy noise (band selection vs subspace).
  2. **Scarcity of labeled multitemporal HSI** (no public labeled HSI *time series*; bitemporal only).
  3. **Pseudo-change / nuisance** (registration, illumination, BRDF, atmosphere, seasonality/phenology).
  4. **Subtle / material-level & sub-pixel change** (mixed pixels; mean-preserving distributional change).
  5. **Interpretability / attribution** (which bands/materials/process; change-*type*).
  6. **Change *characterization* beyond binary** (abrupt vs gradual; trajectory; recovery).
- **Roster to mirror.** github.com/wenhwu/awesome-remote-sensing-change-detection (#traditional-methods):
  PWTT, SiROC, CCDC, BFAST, SFA/ISFA, CVAPS, PCA-Kmeans, IR-MAD, OBCD. These are the fair traditional bars.

---
### Honest novelty verdict for this cluster
The CD niche is **mature and competitive on detection**: IR-MAD/SFA/ACD own invariance, deep nets own
supervised accuracy, SiROC/PWTT own unsupervised spatial/damage. **No room for "a subspace detects change
better."** The genuine openings the reviews themselves name are **#1 (use the high spectral dimension), #5
(attribution), and #6 (change characterization)** — precisely where subspace geometry has a *structural*, not
incremental, argument.
### Next falsifiable step
On a real labeled **bitemporal HSI** benchmark (Bay Area / Hermiston), measure DS/canonical-angle CD vs
SAM/CVA/IR-MAD **as a function of band count** — does the DS−SAM gap turn positive at 200+ bands (challenge
#1)? If not, detection is fundamentally closed and the contribution is characterization (#5/#6) + the diagnostic.
