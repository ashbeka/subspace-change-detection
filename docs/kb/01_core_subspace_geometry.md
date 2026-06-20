# KB 01 — Core subspace geometry (the lab's home mechanisms)

Reading-depth legend in [`README.md`](README.md). Every entry: method+math · signal/subspace idea · novelty ·
how-justified · baselines · transfer to HSI×temporal×CD.

---

## Difference Subspace & its Generalization (DS/GDS/KDS/KGDS) — Fukui & Maki, TPAMI 2015 `[P-internal]`
- **Method+math.** Two subspaces $\mathcal{S}_1=\mathrm{span}(\Phi_1)$, $\mathcal{S}_2=\mathrm{span}(\Phi_2)$.
  Canonical cosines $\sigma_i=\mathrm{svd}(\Phi_1^\top\Phi_2)=\cos\theta_i$. **DS** = eigenvectors of
  $G=P_1+P_2$ ($P_j=\Phi_j\Phi_j^\top$) with eigenvalue in $(0,1)$ = the directions in which the two subspaces
  *disagree*. **Magnitude** $\mathrm{Mag(DS)}=\mathrm{Trace}(2(I-\Sigma))=\sum_i 2(1-\cos\theta_i)$.
  **GDS** = the difference subspace generalized to $N$ class subspaces (sum of projectors, keep the
  small-eigenvalue eigenvectors): the subspace that maximally separates classes; projecting onto it
  orthogonalizes/decorrelates them. **KDS/KGDS** = the same in an RKHS via the kernel trick (kernel PCA
  subspaces + kernel canonical angles).
- **Signal/subspace idea.** A *class/set* is a subspace; the DS is the geometric generalization of the
  difference *vector* $u-v$ to the difference *subspace* between two spans.
- **Novelty.** A principled, basis-level "what differs" object (not just a scalar distance), with a kernel
  generalization; GDS as a feature-extraction/decorrelation transform for subspace-based classifiers.
- **How justified.** Image-set face/object recognition (Venus 3-D sculptures, 300 views → one image-set
  subspace); shows GDS-projection improves class separability of MSM/CMSM.
- **Baselines.** MSM/CMSM, subspace method, kernel PCA, normalized correlation.
- **Transfer to CD.** The lab's anchor. **Hard finding (this project): first-order DS ≡ spectral angle (SAM)
  whenever the per-pixel spectrum is low-dimensional** (13-band S2: ρ=0.9988 with SAM). DS only becomes
  non-trivial when the *set* spanning the subspace is genuinely multi-dimensional → needs 100+ bands, a real
  image-set, or a temporal/SSA embedding. KDS gives nonlinear room but is memory-heavy at scene scale.

## Second-order Difference Subspace — Fukui, Valois, Souza, Kobayashi, arXiv:2409.08563 (2024) `[P]`
- **Method+math.** Three sequential subspaces $\mathcal{S}_1,\mathcal{S}_2,\mathcal{S}_3$. **Principal-
  component (Karcher mean) subspace** $\mathcal{M}(\mathcal{S}_1,\mathcal{S}_3)$ = eigenvectors of $P_1+P_3$
  with eigenvalue $>1$. **Second-order DS** $\mathcal{D}^2(\mathcal{S}_1,\mathcal{S}_2,\mathcal{S}_3)=
  \mathcal{D}(\mathcal{S}_2,\mathcal{M}(\mathcal{S}_1,\mathcal{S}_3))$ — the DS between the middle subspace and
  the Karcher midpoint of its neighbours. Motivated by the 2nd-order central difference
  $\ddot{f}\approx f(t{+}1)-2f(t)+f(t{-}1)$. On the Grassmann manifold, **1st/2nd-order DS = velocity /
  acceleration** of the subspace trajectory; $\mathcal{D}^2=0$ ⇔ constant-velocity (geodesic) motion.
  **Orthogonal decomposition** (their §4.2, via subspace projection $\omega$ onto the geodesic-spanning sum
  space): $\mathrm{Mag}(\mathcal{D}^2)\simeq$ (component **orthogonal** to geodesic = off-path/structural break)
  + (component **along** geodesic = speed change). Self-consistency: $\mathrm{Mag}(\mathcal{D}^2)\approx
  |\frac{d}{dt}\mathrm{Mag}(\mathcal{D}^1)|$ (corr 0.948 on walking).
- **Signal/subspace idea.** Smooth motion (walking) ⇒ along-geodesic component dominates; abrupt motion
  (jumping) ⇒ both components comparable. **The along/orthogonal split is a smooth-vs-abrupt discriminator.**
- **Novelty.** First higher-order extension of DS; ties subspace dynamics to differential geometry; the paper
  flags the projection/decomposition proof as future work (Sensei's Slack: *now proved* — usable with ease).
- **How justified.** Numerical validity on (i) deforming 3-D shape (walking/jumping, CMU mocap), (ii) biometric
  signal via SSA signal subspaces. Demonstration of validity/naturalness, **not** a benchmark leaderboard.
- **Baselines.** Self-consistency (derivative of 1st-order DS); no competing-method comparison (it is a
  representation paper).
- **Transfer to CD.** Sensei's flagship; he explicitly wants it on satellite time series as the "first and
  unique trial." **Project status:** the faithful implementation is verified (`SUBSPACE_CONSTRUCTION_LEDGER`
  §3) but read ~0 in early tests because those constructions had *no genuine acceleration* (uniform rotation =
  constant velocity). The along/orthogonal split as an **abrupt-vs-gradual change-TYPE** descriptor is
  **untested for its real purpose** — a live, defensible, characterization (not detection) lever.

## Time-series Anomaly Detection via DS between Signal Subspaces — Kanai, Sogi, Maki, Fukui, arXiv:2303.17802 (2023) `[P]`
- **Method+math.** SSA: 1-D series $h(t)\to$ trajectory (Hankel) matrix $H_t\in\mathbb{R}^{w\times M}$; **signal
  subspace** $P_t$ = top-$r$ eigenvectors of $H_tH_t^\top$. Past $P_{t-\tau}$ and present $P_t$ overlap → use
  the *generalized* DS (exclude the intersection). Score = **direction index** $\delta(D_{\mathrm{in}},D_N)=
  \frac{1}{c}\sum(1-\lambda_i)$ (canonical relation of the *current* DS to a **learned non-anomalous reference
  DS $D_N$**) × **magnitude index** $\beta=(\mu(D_{\mathrm{in}})-\mu(D_N))^2$, $\mu(D)=\sum\log\cos\theta_i$.
- **Signal/subspace idea.** Generalize the conventional SSA min-angle anomaly score to the *whole* DS between
  signal subspaces, and **model the normal change-direction ($D_N$), flag the residual** = the invariance lever.
- **Novelty.** DS replaces the single min canonical angle in SSA-based change-point/anomaly detection; reports
  AUC 0.923 vs 0.829 for min-angle. (Sensei flagged the paper has "several mistakes" but a valuable reference list.)
- **How justified.** Time-series anomaly benchmarks (not satellite).
- **Baselines.** Classical SSA min-angle change-point detection.
- **Transfer to CD.** This is the lab's *actual* CD-relevant temporal method. **Project status (the SSDS arc):**
  faithful $D_N$ implementation L0-validated (AUC 0.98), and it is **seasonality-robust** (corr with seasonal
  transitions 0.055 vs scalar NDVI-diff 0.58 — the scalar false-alarms every green-up, the subspace does not).
  But on **real** S2 it **comprehensively failed as a detector** (abrupt fire, gradual reservoir, phenology
  onset 0/3) — beaten by harmonic deseasonalization + trivial NDVI mean-shift. The *only* win was an artificial
  splice. ⇒ $D_N$ is real seasonality-robustness machinery whose **detection** edge does not survive on real data.

## Generalized MSM / Mutual Subspace Methods for Image-Set Classification — `[S]`
- **Method+math.** Represent each image *set* as a subspace; classify by canonical angles between the input
  subspace and class subspaces (MSM); CMSM/GDS-projection orthogonalizes class subspaces first; generalized
  versions add kernels, constraints, probabilistic angles.
- **Signal/subspace idea.** "Set ⇄ set" comparison is more robust than "image ⇄ image" (the
  Geometry-of-subspace-set lineage: correlation → SM → MSM → Grassmann).
- **Novelty / justify / baselines.** Standard image-set recognition (faces, gestures); baselines = SM, NN,
  affine/convex-hull set methods.
- **Transfer to CD.** The natural home for "a **material/region** is a subspace; change = a shift in
  canonical-angle *membership*." Caveat (closest-methods doc): "material = subspace" already exists in HSI
  *classification* (endmember/abundance subspace) — so the novelty for CD is the **change operator**, not the
  representation.

## "Geometry of subspace set and its application to machine learning" — Fukui, C-Air talk, 2024 `[P]`
- **Content.** A survey of the lab's program: vector→correlation method→**Subspace Method** (Watanabe 1969,
  projection-based)→**Multiple Similarity Method** (Iizima 1969, min-angle)→**MSM** (set⇄set via canonical
  angles)→**Grassmann learning**→**DS/GDS**→**integration of subspace representation with DNNs**. Canonical
  angles defined by examples (1-D/1-D, 1-D/2-D, 2-D/2-D in $\mathbb{R}^3$).
- **Transfer to CD.** Confirms the *mechanism* framing (`CD_TAXONOMY` Improvement 3): subspace geometry is a
  (representation, operator) pair recurring across families, and the lab's own roadmap ends at
  **subspace ⇄ deep features** — the structural argument for H-C (geometry on foundation-model features).

---
### Honest novelty verdict for this cluster
The DS machinery is **mathematically rich but, as a bare detector, redundant with the spectral angle** unless
the underlying set is genuinely multi-dimensional. The live, non-redundant uses are **(a)** the 2nd-order
geodesic split as a *change-type* descriptor (untested for its real purpose), and **(b)** DS-basis
*attribution* (which modes/bands). 
### Next falsifiable step
Test the geodesic along/orthogonal split as an abrupt-vs-gradual classifier on the real S2 series already in
hand, against the scalar 2nd-difference of the mean spectrum (the null that detects but cannot decompose).
