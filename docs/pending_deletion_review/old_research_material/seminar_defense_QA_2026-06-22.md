# Seminar defense Q&A — every recurring Sensei/senpai question, answered by evidence (2026-06-22)

Purpose: so no question from the 2025/12/16 seminar (or earlier) can be answered with a hunch. Each item maps a
real question from the research notes to (A) a precise answer and (B) the experiment/code that backs it. Gaps
that still need a run are marked **[EXPERIMENT]** with status.

Status legend: ✅ answered by an existing experiment/code · 🔬 experiment running/queued this session · 📋 cite

---

## PART 1 — "Understand what you're doing": the exact subspace implementation (THE #1 2025/12/16 ask)

Sensei's central demand was to know *precisely* how the subspace is built — not to rely on AI to "do the whole
job." Here is the exact construction, from the code (`phase1/ds/pca_utils.py`,
`phase1/scripts/compare_oscd_spatial_subspaces.py`).

**Q: How is the subspace implemented — one subspace per channel? one per whole image (13-D pixels)? one vector
per channel (channel-as-vector)?**
We implement and distinguish **two** constructions, and we now champion the second (it is the one senpai Jang
suggested: "flatten on the Z-axis — 13 vectors, each with x-dimension"):

| | **global_pixel DS** (original / naive) | **band_image DS** (spatially-faithful, championed) |
|---|---|---|
| One sample is | one pixel's 13-band spectrum | one whole flattened band-image |
| PCA input `(d,n)` | `d=13 bands`, `n=N_pixels` | `d=N_valid positions`, `n=13 band-images` |
| sklearn sees | `N_pixels` samples × 13 features | 13 samples × `N_valid` features |
| Subspace lives in | `R^13` (spectral space) | `R^{N_valid}` (spatial-position space) |
| Centering (PCA) | across pixels | across the 13 band-images |
| Rank `r` (#PCs) | ≤13 (we used 5–6) | ≤12 (= 13−1; we use 12) |
| Spatial position | **discarded** (pixels unordered) | **preserved** (position = the ambient axis) |

So in **band_image DS**: each *channel is a vector* (a sample); the 13 channel-vectors of a date define **one**
subspace. This directly answers "13 channels — each a vector or a subspace?": **each channel = one vector
(sample); the date = one subspace** spanned by them.

**Q: What is the subspace, and what is its dimension (number of PCs, not the ambient dimension)?** ✅
The subspace = the span of the top-`r` principal components (an `r`-dimensional linear subspace). The
"dimension" Sensei means is `r` = number of PCs. band_image DS on OSCD: ambient `N_valid≈10^5`, only 13
samples ⇒ max rank 12 ⇒ **r = 12**. global_pixel: ambient 13 ⇒ r = 5–6. (Code: `fit_pca_basis` returns
`basis (d,r)`; `r` is the PC count.)

**Q: Are we losing the information of each pixel vector / spatial position?** ✅ (this is THE crux)
- global_pixel DS: spatial **position** is lost (pixels are an unordered set) — this is exactly the loss Sensei
  warned about.
- band_image DS: position is the ambient axis, so spatial layout is **kept**, and the change score is produced
  **per pixel**. The band_image construction is our direct fix to Sensei's concern.

**Q: 5 PCs in 13-D → shape (5,13)?** ✅ That was global_pixel (basis `(13,5)` = 5 PCs in 13-D spectral space).
band_image is `(N_valid, 12)`.

**Q: How is the projection back to the image done (the math)?** ✅
For band_image DS, the difference subspace `D ∈ R^{N_valid×k}` — *each column is itself a spatial map*. For each
band `b`, the band-difference image `δ_b = post_b − pre_b ∈ R^{N_valid}`. Project and reconstruct:
`c_b = Dᵀ δ_b` (k-vector), `δ̂_b = D c_b ∈ R^{N_valid}`. The **per-pixel change score** is
`s(pixel) = Σ_b δ̂_b(pixel)²` reshaped to `(H,W)`. There is no separate "project back" step — the ambient space
*is* the image grid. (Code: `band_image_ds_score`.) For global_pixel, `D∈R^{13×k}`, per-pixel `δ∈R^13`,
`s = ||Dᵀδ||²` reshaped by pixel index.

**Q: What is "projection energy" / "projection magnitude" and how is it used?** ✅
`projection energy = ||P_D δ||² = ||Dᵀδ||²` (D orthonormal) = squared length of the change vector's component
lying *inside* the difference subspace. `magnitude = sqrt(energy)` (the `energy_norm` score). High value = the
change aligns with the difference subspace = structural change. It is the per-pixel change score.

**Q: DS vs GDS — which did we implement and what's the difference?** ✅
We use **first-order canonical DS** (Fukui–Maki TPAMI 2015): with `ΦᵀΨ = UΣVᵀ` (Σ = canonical correlations =
cosθᵢ), `D = (ΦU − ΨV)[2(I−Σ)]^{-1/2}`, dropping near-shared directions; pair magnitude `= 2(1−cosθᵢ)` and total
DS magnitude `= Σ 2(1−cosθᵢ)`. **GDS (Generalized DS)** is the *multi-class* generalization (one DS summarizing
N≥2 subspaces); for a *bitemporal* pair the pairwise canonical DS is the right object. Our multi-*date*
extension is the **2nd-order DS / geodesic decomposition** (`phase1/subspace/temporal_band_images.py`). Code
variants: `canonical` (used), `eig`, `legacy_residual_stack`.

**Q: # of (datapoints, features) before PCA?** ✅ band_image: `n=13` datapoints, `d=N_valid` features. global_pixel:
`n=N_pixels`, `d=13`.

**Q: "In method 1, each image is one vector, right?"** ✅ Yes — in band_image DS each band-image is one vector
(sample); 13 of them define the date subspace.

---

## PART 2 — U-Net training, thresholding, normalization (Sensei's direct asks)

**Q: Have you trained the U-Net on raw data with NO Otsu / no thresholding?** ✅
Yes. The FC-EF U-Net (`phase1/experiments/unet_ds_prior.py`) trains directly on **raw normalized bands**,
pixel-level masked **BCE+Dice**, with **no Otsu and no thresholding during training**; a threshold is applied
only at evaluation to report F1/IoU. Test AP (3 seeds, 10 official cities): bands 0.4825, bands+DS-fusion 0.5128.

**Q: Normalize first (e.g., 0–1), then train raw without thresholding.** 🔬 **[EXPERIMENT queued]**
Default uses per-band z-score (`oscd_band_stats.json`). Sensei asked for 0–1; queued a `--norm minmax`
variant of M0/M4 to confirm the result is robust to the normalization choice.

**Q: Otsu thresholding before vs after U-Net training.** 🔬 **[EXPERIMENT queued]**
For the *supervised* U-Net no thresholding is needed (it emits probabilities). For the *teacher–student* path,
pseudo-labels need a threshold; default is percentile (top-10%/bottom-50%). Queued an **Otsu-threshold**
pseudo-label variant to directly answer the Otsu question.

---

## PART 3 — Kernel PCA / KDS (Sensei explicitly asked twice)

**Q: Try the Kernel-PCA / KDS trick.** 📋 + 🔬
Codex ran **kernel-DS (RFF/RBF nonlinear)** on global pixels — **CLOSED** (commit `0dfecec`): the nonlinear lift
restores amplitude (Hermiston 0.55→0.97) but still **< CVA 0.985**, and it fails the controlled distributional
test (0.54 vs correlation 0.82). So the kernel trick does **not** beat the simple baseline. To answer Sensei on
*our* construction directly, a **band-image KDS-as-prior** run is queued (the exact analogue). Honest expectation:
LOW (consistent with the global kernel-DS null).

---

## PART 4 — Naive baselines / "why simple methods suck" + covariance/eigendecomposition

**Q: Apply naive methods (Euclidean difference, least-squares) to establish why they're insufficient.** ✅
The naive→complex ladder is explicit (all 24 OSCD cities, AP): **raw Euclidean/CVA 0.226** (the floor) → PCA-diff
0.254 → smoothed-PCA 0.268 → IR-MAD 0.214 → DS 0.241 → fusion 0.278 → CNN 0.483 → CNN+DS-fusion 0.513. Least-
squares fit ≈ PCA reconstruction (the cross-reconstruction control). So "why simple sucks" is shown, rung by rung.

**Q: Represent data as a covariance matrix / eigendecomposition to find hidden patterns.** ✅
The **matched geometric controls** are exactly this: normalized spatial **Gram** (full second-moment, all modes;
AP 0.142), **projector**-row distance (eigenspace orientation; 0.102), cross-reconstruction (0.215). DS beats all
of them — i.e., the canonical DS carries information the raw covariance/eigenstructure does not.

---

## PART 5 — Band selection / combinations (recurring senpai idea)

**Q: Do we need all 13 bands? Which band combinations (RGB / +NDVI / +SWIR) work best for change?** 🔬 **[EXPERIMENT queued]**
Not yet systematic. Queued a targeted band-subset study for the DS prior and the CNN: **RGB-only**, **all-13**,
and **RGB+NIR+SWIR** — to report which bands carry the change signal and whether the DS-specific gain survives
band reduction. (Full 1/2/3/4-band combinatorial sweep is future work; the targeted version answers the question
with evidence.)

---

## PART 6 — CCA / S3CCA / temporally-regularized CCA (Sensei named specific papers)

**Q: Experiment with S3CCA and temporally-regularized CCA; look into CCA.** ✅/📋
- CCA itself: **IR-MAD = iteratively-reweighted CCA** — it is a baseline *and* a fusion component throughout.
- **S3CCA / contiguous-band DS-basis attribution**: tested — **NEGATIVE** (DS-basis attribution near-random vs
  per-band difference; `seminar_ranked_problems` #9).
- Temporally-regularized CCA ≈ the temporal-warp angle: **RTW** tested on BreizhCrops + MultiSenGE (Codex).
We did not run the *exact* S3CCA reference implementation; the attribution question it targets was answered
negative by the DS-basis test.

---

## PART 7 — The big "why subspace / what setting" questions (notes lines 16–21, 274–276)

**Q: What problem in satellite CD is there a genuine place for subspace methods? Why should they help? What
weakness do they address? What setting (low labels / interpretability / temporal / disaster)?** ✅ (this is our thesis)
**Answer, evidence-backed:** subspace geometry is *not* a better standalone detector (proved across the
diagnostic), but it is a genuinely useful **label-free, DS-specific complementary prior** for **label-scarce /
disaster** CD. Evidence: (1) matched controls — DS beats all matched geometric nulls and adds significant,
DS-specific fusion evidence (p=0.0098); (2) learned rung — adding DS to a prior fusion lifts a trained CNN
**0.480→0.513**, DS-specifically (matched-null fusion only 0.487), and the gain **holds under extreme label
scarcity** (n=2 cities: +0.040); (3) teacher–student (running). Setting = low labels, interpretability,
complementarity to deep models — exactly the disaster/reconstruction driver.

---

## PART 8 — Geodesic / smooth change (Sensei's written post-talk feedback)

**Q: How do we use geodesic information between subspaces for smooth change? (geodesic projection proven.)** 📋
We implemented 2nd-order DS + geodesic decomposition (`temporal_band_images.py`); the mechanism works (matches
Sensei's "geodesic projection works as expected") and cleanly separates along-geodesic (smooth drift) from
orthogonal (abrupt) components. Result: **no standalone detection gain** but a valid **interpretable change-type
/ smooth-vs-abrupt descriptor** — present visually, not as a performance claim. (Also: VLM-explanation-as-subspace
idea from the senpai is a future direction.)

---

## PART 9 — Pseudo-change vs real change; seasonality (rice/snow)

**Q: Re-frame around pseudo-change vs real change; how to treat repeatable/seasonal change?** ✅/📋
OSCD itself labels only **artificialization** (man-made) change and ignores vegetation/tide — so the benchmark
encodes the pseudo-vs-real distinction. IR-MAD (affine-invariant) and the seasonal-observation stress test
(Codex) address nuisance/seasonality. The teacher–student + the priors inherit IR-MAD's nuisance handling.

---

## PART 10 — Datasets: OSCD / xBD / MultiSenGE (what each is actually FOR)

- **OSCD** — primary benchmark (14 train / 10 test, official split). All learned-rung experiments here.
- **xBD-S12** — external **disaster** transfer (Codex, frozen protocol): does the DS complementarity generalize
  to building-damage on an event-disjoint split? This is the publication gate.
- **MultiSenGE** — **recommended new role:** the large **unlabeled corpus for teacher–student pre-training**
  (its size is wasted on "testing only"). Use a teacher (DS-fusion) to pseudo-label MultiSenGE → pre-train the
  student → fine-tune on OSCD's few labels. This finally *uses* the big dataset for what it's good at, and is
  the natural home of the label-scarce story. **[discuss before building]**

---

## PART 11 — Semantic / change-type interpretation (clustering, SSC, SAM)

**Q: Make the change interpretable — cluster change types (urban/veg/industrial), SSC/SAM semantics.** 📋 (future work)
Roadmap: DS/fusion change map → crop changed regions → cluster (SSC) / describe (SAM/VLM) → assign rough
semantic labels. Not part of the current *binary*-CD seminar, but the strongest "future work" slide, and it ties
to the disaster damage-state framing.

---

## Gap experiments being run/queued this session (so nothing is hunch-only)
1. 🔬 fusion-scarcity sweep (DS-specific gain vs label scarcity) — RUNNING; n=2 already shows +0.040.
2. 🔬 teacher–student (DS-fusion teacher vs classical; DS-specificity) — RUNNING.
3. 🔬 normalization variant (0–1 vs z-score) for M0/M4 — queued (Sensei).
4. 🔬 band-subset (RGB / all-13 / +NIR+SWIR) for DS prior + CNN — queued (senpai).
5. 🔬 band-image KDS-as-prior (kernel trick on our construction) — queued (Sensei).
6. 🔬 Otsu vs percentile pseudo-label thresholding in teacher–student — queued (Sensei).
7. 📋 cite: global kernel-DS CLOSED (Codex), S3CCA/attribution negative, geodesic mechanism, matched controls.
