# MASTER NARRATIVE — the complete project knowledge base (2026-06-22)

**This is THE comprehensive reference** — the deep version behind the simplified seminar. If a slide simplifies
something, the full story (method, math, every experiment, the path, the numbers) is here. The audience-facing,
plain-language slide kit is `SEMINAR_PREP.md`; the concept intuition/whiteboard scripts are in
`CONCEPTS_EXPLAINED.md`; the exact implementation answers are in `seminar_defense_QA_2026-06-22.md`. This doc
consolidates findings from Abood, Claude, and Codex.

Reading order for full understanding: §1 thesis → §2 method → **§11 full math** → **§12 the research path
(chronology of everything tried)** → §3 the experimental ledger (results) → **§13 consolidated results tables** →
§7 honest scope → §9 the distilled story.

## 0. Status (all experiments complete as of 2026-06-22)
All runs below are finished and saved. Code in `phase1/experiments/`; outputs in `phase1/outputs/`; figures in
`phase1/outputs/seminar_figures/`. Nothing is in flight.

---

## HEADLINE UPDATE (2026-06-23) — Successive Saab-DS is the new centerpiece
Codex's `oscd_successive_subspace_learning_ds_2026-06-23.md` supersedes the earlier "DS as complementary prior"
headline. **Successive Saab-DS** (PixelHop/Green-Learning successive Saab local features → per-hop Band-Image DS →
fused) is the first construction where spatially-faithful subspace geometry **beats the classical unsupervised
detectors** on a frozen held-out OSCD set: **AP 0.342 (pair-adaptive) / 0.338 (train-fitted, leakage-free)**,
AUROC 0.886–0.891, vs smoothed-PCA 0.314, PCA-diff 0.307, IR-MAD 0.265, raw-CVA 0.279. **DS beats L2/PCA on the
same features** (vs matched PCA +0.037 AP, p=0.002, 10/10 cities). **Train-fitted ≈ pair-adaptive → the
representation generalizes (not transductive).** Honest limits: vs smoothed-PCA p=0.084 (10 cities, marginal);
xBD disaster transfer is mixed (raw L2 wins damage-vs-intact; projector wins localization) → keep xBD as the
band-image-projector supporting result. This directly satisfies Sensei's Green-Learning/PixelHop/Saab ask **and**
the spatial-information concern, with clean negatives for the wavelet/pyramid ideas he also suggested. The
construction journey (global pixel 0.06 → band-image 0.24 → Saab-DS 0.34) is the seminar spine.
**Seminar working file: `SEMINAR_2026_FINAL.md`.** Math/path/tables below remain the deep reference.

## 1. Thesis + real-world motive
**Motive (Abood):** post-disaster / post-conflict reconstruction (Gaza) needs fast, reliable change & damage
maps from satellite — but a new event has **no labels**, only free medium-res **Sentinel-2**, so the
operational tool is **unsupervised** change detection, which is per-pixel and **spatially blind**.

**Thesis (evidence-backed):** subspace geometry is **not** a better standalone detector, but a **spatially-
faithful Difference Subspace** carries **DS-specific, label-free complementary** change information that
classical maps, matched geometric controls, **and a trained CNN** do not capture on their own — useful as a
prior / teacher for **label-scarce** CD, and as an interpretable temporal analysis tool.

---

## 2. The method, precisely (defense-ready; Sensei's #1 ask = "understand what you're doing")
Two constructions; we champion the second (= Jang's "flatten on the Z-axis"):

| | global_pixel DS (naive) | **band_image DS (spatially-faithful)** |
|---|---|---|
| one sample = | one 13-band pixel | one whole flattened band-image |
| PCA input (d,n) | (13, N_pixels) | **(N_valid, 13)** |
| subspace in | R¹³ (spectral) | **R^N_valid (spatial)** |
| rank r (#PCs) | 5–6 | **12** |
| spatial position | **lost** | **preserved** |

- Each channel = a vector (sample); 13 of one date span **one** subspace; "dimension" = r = #PCs.
- **Canonical first-order DS** (Fukui–Maki): `ΦᵀΨ=UΣVᵀ`(Σ=cosθᵢ) → `D=(ΦU−ΨV)[2(I−Σ)]^(−1/2)`; magnitude
  `Σ2(1−cosθᵢ)`. GDS = multi-class generalization; 2nd-order/geodesic DS = multi-date extension.
- **Score:** per pixel `s = Σ_bands ‖P_D δ_band‖²` (`δ=post−pre`); ambient axis = image grid (score is already spatial).

---

## 3. The complete cross-agent experimental ledger

### A. The diagnostic (the spine) ✅ 📋
Across detection / characterization / warp / material-subspace / orientation / attribution / deep-feature /
kernel / small-sample / recovery, on real data and against the correct nulls, **subspace geometry is redundant
or worse than a simpler standard statistic** (SAM, CVA, IR-MAD, harmonic deseasonalization, DTW, correlation,
patch-mean SAM, smoothed PCA). Corroborated by two independent research-minings. *This is the rigorous backbone;
the positives below sit on top of it.* Frame = REPRESENTATION × COMPARISON-OPERATOR × DECISION map.

### B. Matched-controls positive (Codex, OSCD, all 24 cities, AP) ✅
Band-Image DS **beats every matched geometric null** (Gram 0.142, projector 0.102, cross-recon 0.215; DS
0.241) but **loses standalone to smoothed PCA** (0.268). In percentile-rank **fusion** it adds **significant,
DS-specific** evidence: DS-fusion 0.278 vs matched cross-substitute fusion, +0.0115 AP, **p=0.0098**, 9/10
test cities. (Report: `oscd_band_image_matched_spatial_controls_2026-06-22.md`.)

### C. The learned rung — DS as a CNN prior (Claude, OSCD, FC-EF U-Net, 3 seeds) ✅/🔬
| config | test AP | note |
|---|---:|---|
| bands (M0) | 0.4825 | strong learned baseline |
| bands+DS (M1) | 0.4788 | a single DS channel is **redundant** to a trained CNN |
| bands+cross / +sPCA | ~0.48 | any single prior redundant |
| **bands+DS+sPCA+IR-MAD (M4)** | **0.5128** | the win |
| bands+sPCA+IR-MAD (no DS, M5) | 0.4804 | fusion **without** DS = no gain |
| bands+cross+sPCA+IR-MAD (M6) | 0.4870 | matched null **can't substitute** for DS |
**→ at full data the fusion gain is DS-specific (+0.032 over no-DS, +0.026 over matched-null), replicating the
classical-rung result at the learned rung.** Code: `phase1/experiments/unet_ds_prior.py`.

**Fusion-scarcity across budgets (honest nuance — DS-specificity is NOT uniform):** DS-fusion vs no-DS vs
cross-fusion (matched null), test AP, 3 seeds:
| n_train | DS-fusion | no-DS | cross-fusion | DS vs cross |
|---:|---:|---:|---:|---:|
| 2 | 0.266 | 0.226 | 0.235 | **+0.031 (DS wins)** |
| 4 | 0.391 | 0.382 | 0.396 | -0.005 (tie) |
| 7 | 0.414 | 0.407 | 0.421 | -0.007 (tie) |
| 14 | 0.513 | 0.480 | 0.487 | **+0.026 (DS wins)** |
DS-fusion ≥ no-DS at every budget; DS clearly beats the matched cross-null **at n=2 and n=14**, but **ties it at
n=4/7** (within ±0.01–0.02 std). Honest claim: *DS-specific advantage at full data and extreme scarcity, neutral
at mid budgets* — not a uniform DS-specificity. The robust part is full-data (matched-controls p=0.0098 + n=14 CNN).

**Normalization (Sensei's 0-1 ask) ✅:** per-image min-max [0,1] is **far worse** than global per-band z-score
(bands 0.218 vs 0.483; fusion 0.282 vs 0.513). Z-score wins decisively (it preserves cross-image radiometry);
the DS-fusion prior still helps under min-max (+0.064). Concrete evidence-based answer.

**Band-subset (senpai "do we need 13 bands?") ✅:** RGB-only bands 0.450; **RGB+NIR+SWIR (6 bands) 0.484 ≈
all-13 0.483**; the DS-fusion prior helps at every subset (rgb +0.025, 6-band +0.010, all-13 +0.030). So ~6
well-chosen bands capture most of the change signal; all-13 + DS-fusion is still the best (0.513).

### D. Teacher-student / pseudo-label distillation (Claude, OSCD) 🔬
Teacher (unsupervised map) → confidence-filter → pseudo-labels → student (raw bands only). Partial:
students track teacher *quality* (spca 0.256 > cva 0.227 > irmad 0.200 > ds 0.228) but land **below** the raw
teacher maps — **naive distillation does not beat the raw map on *tiny* OSCD.** DS-fusion vs no-DS/cross teacher
comparison completing now. **Implication:** the value needs a LARGE unlabeled pool → **MultiSenGE pre-training**
(§3D2). Code: `phase1/experiments/ds_teacher_student.py`.

### D2. MultiSenGE pre-training → OSCD fine-tuning (the label-scarce payoff) 🔬 (1 seed; multi-seed confirming)
Pre-train an FC-EF student on **1200 MultiSenGE bitemporal pairs** with a DS-fusion teacher's pseudo-labels
(no human labels), fine-tune on OSCD at label budgets; common 10 bands. 1-seed result (test AP):

| budget | scratch | pretrain **DS-fusion** teacher | pretrain spca teacher |
|---|---:|---:|---:|
| n=2 | 0.2528 | **0.2664** | 0.1903 |
| n=14 | 0.4924 | 0.4742 | 0.4043 |

**The first place the label-scarce story lands positively *and* DS-specifically at the learned/transfer rung:**
at n=2, DS-fusion-teacher pre-training beats scratch (+0.014) and **massively beats the classical spca-teacher**
(+0.076; spca-teacher pre-training *hurts*). With full labels (n=14) pre-training is neutral-to-negative — a
genuine **label-efficiency** effect (helps only when labels are scarce), exactly the disaster driver. Caveats:
1 seed (the +0.014 vs scratch is within noise; the DS-fusion≫spca gap is large); MultiSenGE is intra-year
seasonal (domain gap to OSCD urban).
**FINAL multi-seed verdict (3 seeds) — CLEAN NEGATIVE (the seed-0 win was an artifact):**
| budget | scratch | pretrain DS-fusion | pretrain spca |
|---:|---:|---:|---:|
| 2 | 0.240 | 0.237 | 0.223 |
| 4 | **0.405** | 0.377 | 0.370 |
| 7 | **0.416** | 0.400 | 0.396 |
| 14 | **0.480** | 0.457 | 0.439 |
Training from scratch beats MultiSenGE-pseudo-label pre-training at **every** budget — **negative transfer**
(MultiSenGE is intra-year seasonal; OSCD is urban artificialization). DS-fusion is consistently the *less-bad*
teacher than spca, but neither helps. **Honest verdict: the label-scarce-via-MultiSenGE-pretraining route does
NOT work, and DS is not specific here.** This sharpens — rather than weakens — the contribution: it shows we
tried the obvious label-scarce route and it failed, leaving the **full-data DS-specific input-prior fusion** +
the **diagnostic** + the **temporal characterization** as the genuine, robust results.
Code: `phase1/experiments/multisenge_pretrain.py`.

### E. Temporal 1st/2nd DS + geodesic (Claude, real S2 sequences) ✅ — Sensei's most-repeated ask
On al_wakrah(Qatar)/beijing_airport/piraeus (20 real S2 dates each), spatially-faithful per-date subspaces:
first-DS magnitude ("velocity") tracks change but is **only partly the trivial mean-shift** (corr 0.27–0.69 →
carries real temporal structure); 2nd-DS + along/orthogonal **geodesic** split localizes abrupt vs smooth
periods. A first/unique **analysis** trial (not a detector), exactly Sensei's framing + the ACCV angle.
Code: `phase1/experiments/temporal_ds_sequence.py`. (4-band RGBI; 13-band GEE sequence = natural extension.)

### F. External disaster transfer (Codex, xBD-S12) ✅ — narrow positive
Frozen protocol, 1,577 official test patches, 5 unseen disasters. The **spatial subspace geometry TRANSFERS**:
the **projector distance** (subspace-orientation change) is the strongest geometry — AUROC 0.734, beats PCA-diff
in **5/5** events (+0.0118), beats IR-MAD on building localization (+0.031 AP, 11/11 train events, p=0.00293).
Canonical DS beats its matched cross-null on the 5 test events (DS−cross +0.0045, 5/5, p=0.0625) but **not**
consistently on training events (5/11) — so DS-specificity is **event-dependent, not universal**. Operationally
a label-free **candidate-localization prior**: top-5% of ranked pixels retrieves ~25% of damaged pixels (≈5×
lift). Honest scope (Codex): **not** SOTA, **not** damage-severity (raw L2 is better at damage-vs-intact),
**not** significant beyond 5 events. A real but narrow external positive — geometry as a disaster-damage
*candidate-localization* prior. (`xbd_s12_external_validation_2026-06-22.md`.)

### G3. (Geometry × interpretability) change-TYPE — NEGATIVE (the interpretability hope, tested) ✅
On Benton (real bitemporal 159-band HSI, **6 change-type GT classes** — the only real multiclass change GT on
disk; the assumed "Hermiston 5-class" does NOT exist locally), cluster changed pixels into 6 types and compare
representations (NMI / ARI / purity vs GT): **polar-CVA direction 0.832 / 0.880 / 0.932** (field baseline, best)
> raw change vector 0.787 > **DS canonical directions 0.467 / 0.349 / 0.702** > PCA-of-diff 0.422. The DS-mode
representation is **far worse** than the simple change *direction* — the subspace projection conflates the
per-pixel direction that discriminates types. So geometry loses even at interpretable change-typing, against the
exact baseline the literature flagged (Bovolo–Bruzzone polar CVA). The DS directions remain *visually*
interpretable spectral change-modes, but they do **not** recover change types better than a normalized
difference. Code: `phase1/experiments/ds_change_modes.py`.

### G. Closed leads (cite as the diagnostic's evidence) 📋
kernel/RFF DS (< CVA); deep autoencoder-feature subspace (< latent correlation); recovery-trajectory geometry
(near-chance, but multi-band recovery > single index — a non-geometry positive); RTW on BreizhCrops/MultiSenGE;
MSM material-subspace (< SAM-to-mean); S3CCA/contiguous-band attribution (near-random); Kanai signal-subspace DS
+ SSA (seasonality-robust on splice, falsified on real natural change); HSI orientation/moment-geometry (< direct
controls); 2nd-order DS detection (redundant with mean-vector Δ²m).

---

## 4. Sensei's asks — coverage
Of ~18 concrete Slack asks: **15 experimented (mostly rigorous, multi-dataset), 2 were gaps now closed** (temporal
DS/geodesic = §3E done; kernel-DS = closed globally by Codex), **1 honest future work** (VLM-as-subspace, which
Sensei flagged as distant). Full map: `sensei_asks_coverage_2026-06-22.md`.

---

## 5. Datasets and their roles
- **OSCD** — primary benchmark (14 train / 10 test). All learned-rung + matched-control work.
- **xBD-S12** — external **disaster** transfer (Codex); the publication gate.
- **MultiSenGE** — **intended role:** large **unlabeled corpus for teacher-student pre-training** (its only
  principled use; "testing-only" wastes it). ⚠️ **not downloaded yet** (code skeleton only, 0 tiles on disk).
  Decision pending: use it this way or drop it.
- **ipol416 sequences** — real multi-date S2 (al_wakrah/beijing/piraeus) for temporal DS (§3E).
- **HSI** (Hermiston/Salinas/Bay Area…) — diagnostic / kernel / moment-geometry experiments.

---

## 6. Novelty & positioning
First use of a **spatially-faithful Difference Subspace as a label-free, global geometric prior/teacher for
data-scarce change detection**, proven **DS-specific** via matched controls at both the score and learned rungs.
Closest work: deep-feature change priors (CGNet, arXiv:2404.09179 — cites the CNN's "insufficient receptive
field," which our *global* subspace fills); hand-crafted/affinity change priors (arXiv:2001.04271); pre-detection
pseudo-labeling incl. **DSFA** (a subspace-family teacher); Grassmann deep nets (GrNet / arXiv:2511.08628 — heavy
manifold-valued, our future work). Bar = Sensei's "novel, honest first trial," not SOTA.

---

## 7. Honest scope (defend these)
- Standalone Band-Image DS is **not** a new SOTA detector (loses to smoothed PCA).
- The positives are **complementarity** results (significant, DS-specific) + an **interpretable temporal analysis**
  + a **diagnostic** — modest in magnitude, rigorous in evidence.
- Learned-rung gain is an **interaction** (DS needs the classical maps to express it); mechanism not yet dissected.
- One architecture (FC-EF), mostly OSCD; **external confirmation (xBD-S12) pending**. Priors are transductive
  (unsupervised, no labels) — stated plainly.

---

## 8. Pending / active experiments (so nothing is forgotten)
1. 🔬 full teacher-student DS-specificity (DS-fusion vs no-DS vs cross teacher).
2. 🔬 fusion-scarcity n=4,7 (does the +0.03 hold across label budgets).
3. 🔬 0-1 normalization + band-subset (RGB / +NIR+SWIR) — Sensei/senpai asks.
4. ⏳ Codex xBD-S12 external transfer results.
5. ⏳ MultiSenGE teacher-student **pre-training** (needs download; the make-or-break for the label-scarce story).
6. 📋/🔬 OSCD-specific kernel-DS confirmation (quick; closes Sensei's "run it yourself" benchmark-matched).
7. ⏳ qualitative panel (where DS-fusion fixes the CNN) — needs GPU after the consolidated job.
8. ⏳ 13-band GEE temporal sequence (upgrade §3E from 4-band RGBI).

---

## 9. DISTILLED — what to actually present (the ~16-slide arc)
1–4. **Real stakes → operational reality (no labels, S2, speed) → the per-pixel blind spot → Sensei's spatial
objection.** 5–7. **The idea (spatially-faithful band-image DS) → the precise construction (§2) → the question.**
8. **The discipline: naive→complex ladder + matched controls.** 9–10. **Result A (matched controls + DS-specific
fusion, p=0.0098).** 11–13. **The hardest test: DS-prior in a CNN — single redundant, but DS-specific fusion gain
that holds under scarcity (§3C); the mechanism (global vs local + sample-efficiency).** 14. **Temporal DS +
geodesic on real S2 (§3E) — the first/unique analysis trial.** 15. **What we tried that did NOT work (honest
rigor): teacher-student + MultiSenGE pre-training (negative transfer), single-channel prior, kernel-DS — these
sharpen *where* DS helps.** 16. **Future: xBD-S12 external transfer (Codex), mechanism/interpretability of the
complementarity, manifold-valued (Grassmann) fusion, 13-band temporal sequence.**

**Headline = the DS-specific complementarity (score + learned rung), anchored by the diagnostic; the temporal
DS/geodesic is the second pillar and the ACCV "first trial." Not "geometry wins" — "where geometry genuinely
contributes, measured honestly."**

---

## 10. Document index
- This file = master. Audience slide kit: `SEMINAR_PREP.md`. Concept intuition + whiteboard: `CONCEPTS_EXPLAINED.md`.
  Defense Q&A: `seminar_defense_QA_2026-06-22.md`. Sensei coverage: `sensei_asks_coverage_2026-06-22.md`.
  References: `REFERENCES.md`. Cross-agent audit: `../../../sccd-claude/docs/research/SUPREME_AUDIT_TABLE.md`.
  Reports: `../experiment_reports/`. Code: `phase1/experiments/{unet_ds_prior, ds_teacher_student,
  multisenge_pretrain, temporal_ds_sequence, ds_change_modes, cca_change, otsu_ablation, ds_prior_panel,
  make_seminar_figures, make_framework_fig, make_real_panels, build_deck}.py`.

---

## 11. Full math (complete — the deep version of §2)

### 11.1 Notation
One date $t$: a cube $C_t \in \mathbb{R}^{B \times H \times W}$ ($B$ bands, $H\times W$ pixels). $N=$ number of
valid pixels. A common valid mask is shared across the two dates so the constructions are comparable.

### 11.2 Two ways to form the per-date subspace
- **global_pixel (naive):** sample = one pixel's spectrum $x_i \in \mathbb{R}^B$. Stack as $X_t^{px}\in\mathbb{R}^{B\times N}$
  (B rows, N columns). PCA over the $N$ pixel-columns → basis $U_t\in\mathbb{R}^{B\times r}$, a subspace in
  $\mathbb{R}^B$ (spectral space). Rank $r\le B$ (we used 5–6). **Spatial position is discarded** (pixels are an
  unordered set).
- **band_image (spatially-faithful, championed):** sample = one whole flattened band-image
  $b_k \in \mathbb{R}^N$. Stack as $X_t \in \mathbb{R}^{N\times B}$ (N rows = pixel positions, B columns = the B
  band-images). PCA over the $B$ band-image-columns (center across columns) → basis $\Phi_t\in\mathbb{R}^{N\times r}$,
  a subspace in $\mathbb{R}^N$ (spatial/position space). Max centered rank $= B-1 = 12$. **Spatial layout is
  preserved** (the ambient axis *is* the pixel grid). #datapoints $=B=13$; #features $=N$.

### 11.3 First-order canonical Difference Subspace (Fukui–Maki, TPAMI 2015)
Given two orthonormal bases $\Phi$ (pre), $\Psi$ (post) of the same ambient space:
1. SVD of the cross-Gram: $\Phi^\top\Psi = U\,\Sigma\,V^\top$, with $\Sigma=\mathrm{diag}(\cos\theta_1,\dots,\cos\theta_r)$.
   The $\theta_i$ are the **canonical (principal) angles**. $\cos\theta_i\to1$: shared direction; $\cos\theta_i$
   small: a difference direction.
2. **Difference subspace basis:** $D = (\Phi U - \Psi V)\,[\,2(I-\Sigma)\,]^{-1/2}$, keeping only pairs with
   $1-\cos\theta_i > \epsilon$ (drop near-shared directions). $D$ is orthonormal and spans "what changed."
3. **DS magnitude (scalar summary):** $m_{DS} = \sum_i 2(1-\cos\theta_i)$.

### 11.4 Per-pixel change score
Band-difference image $\delta_b = X_{post,b} - X_{pre,b}\in\mathbb{R}^N$. Projector $P_D = DD^\top$. Per pixel $i$:
$$ s_i = \sum_{b=1}^{B} \big\lVert (P_D\,\delta_b) \big\rVert^2_{\text{at }i} = \sum_b (P_D\delta_b)_i^2 .$$
`energy_sq` = $s_i$; `energy_norm` = $\sqrt{s_i}$. The map $\{s_i\}$ reshaped to $(H,W)$ is the change map (no
separate "project back" step — the ambient axis is the grid).

### 11.5 Projector distance (the xBD-transfer geometry)
Per-pixel row-norm of the projector difference: $d_i = \big\lVert (P_\Phi - P_\Psi) \big\rVert_{\text{row }i}$,
$P_\Phi=\Phi\Phi^\top$. Computed without materializing $N\times N$:
$d_i = \sqrt{\,\mathrm{lev}_\Phi(i) + \mathrm{lev}_\Psi(i) - 2\,c_i\,}$, where $\mathrm{lev}_\Phi(i)=\sum_k \Phi_{ik}^2$
and $c_i = \sum_{k,l}\Phi_{ik}(\Phi^\top\Psi)_{kl}\Psi_{il}$. Interpretation: how much the spatial subspace
*rotated* at pixel $i$. Strongest geometry on disaster building damage.

### 11.6 Matched geometric nulls (the rigor — same spatial sample axis as DS)
- **Spatial Gram row distance:** $\lVert (A_{pre}A_{pre}^\top - A_{post}A_{post}^\top) \rVert_{\text{row }i}$,
  $A_t = X_t/\lVert X_t\rVert_F$ (trace-normalized; retains ALL singular modes, no rank truncation).
- **Projector row distance:** §11.5 (rank-matched; isolates subspace orientation, ignores the observed Δ).
- **Cross-reconstruction:** reconstruct each date by the other's rank-$r$ subspace; score = symmetric *excess*
  cross-reconstruction residual over self-reconstruction. (A non-DS "look-alike" — the control DS must beat.)

### 11.7 Second-order DS + geodesic (multi-date; Fukui et al. 2024)
For a sequence of date-subspaces $S_1,S_2,\dots$ (points on the Grassmannian $\mathrm{Gr}(r,N)$): first-order =
adjacent DS magnitude / geodesic distance ("velocity"); second-order DS combines the mean subspace with the
along- vs orthogonal-geodesic split ("acceleration"; smooth drift vs abrupt off-geodesic deviation). Code:
`phase1/subspace/temporal_band_images.py`.

### 11.8 Baselines (what geometry is compared against)
- **CVA / raw L2:** $\lVert\delta_i\rVert_2$ (amplitude). **SAM:** angle between pre/post pixel vectors (direction).
- **PCA-diff (Celik):** PCA of the difference image, top-component magnitude. **smoothed/multiscale PCA:**
  + Gaussian $\sigma\in\{0,1,2\}$.
- **IR-MAD (Nielsen):** iteratively-reweighted CCA between the two dates; per-pixel chi-square of the MAD variates.
- **polar-CVA (Bovolo–Bruzzone):** magnitude + direction; direction clusters → change types.
- **Fusion:** equal-weight percentile-rank average of selected maps.
- **Metrics:** AUROC, Average Precision (AP — primary, change is rare ~1–5%), best-F1 / IoU, Otsu-threshold F1.

---

## 12. The research path (chronology — everything tried, in order, and why we pivoted)

1. **Last year (baseline):** bitemporal first-order DS + U-Net with DS/PCA priors on OSCD. Found DS gives
   reasonable but not best change maps (PCA-diff AUROC ~0.81 > DS ~0.75); raw U-Net best IoU. Sensei: "understand
   the construction; you broke the spatial information." → motivated this year.
2. **"Is geometry forced?"** Derived that first-order DS ≡ spectral angle (SAM) for low-dim spectra
   (`finding-ds-equals-spectral-angle`). → reframed: geometry is a *mechanism* (representation × operator ×
   decision), not a CD family.
3. **Temporal-DS detector arc (Kanai signal-subspace DS + SSA + learned non-anomalous reference $D_N$):** worked
   on a synthetic seasonal splice (beat harmonic deseasonalization) but **falsified on real natural change**
   (fire/reservoir/irrigation). → the subspace-as-better-*detector* thesis is dead.
4. **The 3 post-hoc bets, all closed:** (1) N≪B small-sample + registration robustness — subspace flat/worst;
   (2) recovery-trajectory geometry — near-chance 0.40 vs multi-band 1.0 (but multi-band recovery > single index:
   a non-geometry positive); (3) deep autoencoder-feature subspace — latent subspace 0.67 < latent correlation 0.89.
5. **Kernel/nonlinear DS (RFF):** nonlinear lift restores amplitude (Hermiston 0.55→0.97) but still < CVA 0.985;
   fails distributional. Closed.
6. **The diagnostic crystallized:** geometry redundant/worse than a simpler statistic in *every* tested cell, on
   real data, against correct nulls; two independent research-minings concur. → THE contribution.
7. **Matched-controls (Codex, OSCD, 24 cities):** Band-Image DS beats every matched geometric null (Gram/
   projector/cross-recon) but loses standalone to smoothed PCA; in fusion it adds **DS-specific** evidence
   (p=0.0098). → first defensible positive.
8. **The learned rung (Claude, OSCD U-Net):** single DS channel redundant to a trained CNN; but the DS-fusion
   prior beats the no-DS and matched-null fusions at full data (0.513 vs 0.480/0.487); holds at n=2; budget-
   dependent at n=4/7. → complementarity replicates at the learned rung.
9. **Teacher-student (pseudo-labels on OSCD):** students < raw map on tiny OSCD; DS not special at the pseudo-
   label rung (spca teaches as well). Negative.
10. **MultiSenGE pre-training (label-scarce):** pre-train on a large unlabeled corpus → fine-tune OSCD; multi-seed
    shows **negative transfer** (scratch wins every budget). Negative — sharpens where DS helps.
11. **Temporal 1st/2nd-DS + geodesic on real S2 sequences (Sensei's repeated ask):** demonstrated; DS magnitude
    carries structure beyond the trivial mean-shift (corr 0.27–0.69); geodesic separates smooth/abrupt. A
    characterization, not a detector.
12. **xBD-S12 external transfer (Codex):** the spatial projector geometry **transfers** to real disaster damage —
    beats PCA-diff 5/5 events, beats IR-MAD at building localization (p=0.003 train), label-free candidate
    localization (top-5% → ~25% of damage). The narrow, real, motive-aligned positive. → the headline.
13. **Gap-closers (Sensei's named methods):** CCA/S3CCA family (Benton: CVA 0.978 > IR-MAD 0.958 > sparse-CCA
    0.906 > DS 0.508); Otsu (no-threshold U-Net best, Otsu halves F1); change-TYPE interpretability on Benton
    6-class (polar-CVA NMI 0.83 >> DS 0.47 — geometry loses even here). All tested, honest.

**Net:** geometry is redundant for raw detection and change-typing; the defensible positives are the diagnostic,
the DS-specific fusion complementarity (OSCD), and the xBD disaster-localization transfer.

---

## 13. Consolidated results tables (every number in one place)

**OSCD matched controls (24 cities, mean AP):** raw CVA 0.226 · PCA-diff 0.254 · smoothed PCA 0.268 · multiscale
PCA 0.268 · IR-MAD 0.214 · Band-Image DS 0.241 · cross-recon 0.215 · spatial Gram 0.142 · projector 0.102 ·
**rank fusion (sPCA+DS+IR-MAD) 0.278**. DS−cross +0.026 (p=0.0008); DS-fusion − cross-fusion +0.0115 (p=0.0098).

**OSCD U-Net DS-prior (3 seeds, 10 test cities, AP):** bands 0.4825 · +DS 0.4788 · +cross 0.4808 · +sPCA 0.4780 ·
+sPCA+IRMAD (no DS) 0.4804 · +cross+sPCA+IRMAD 0.4870 · **+DS+sPCA+IRMAD 0.5128**.

**Fusion-scarcity (DS-fusion / no-DS / cross-fusion, AP):** n=2: 0.266/0.226/0.235 · n=4: 0.391/0.382/0.396 ·
n=7: 0.414/0.407/0.421 · n=14: 0.513/0.480/0.487. (DS-specific at n=2 & n=14; ties at n=4/7.)

**xBD-S12 (5 unseen events, mean event AUROC | AP):** raw L2 0.443|0.0083 · PCA-diff 0.591|0.0184 · DS
0.626|0.0212 · IR-MAD 0.730|0.0265 · **projector 0.734|0.0302**. Triage recall@top-5% = 0.247 (≈5× lift).
DS−cross +0.0045 (5/5 events, p=0.0625).

**MultiSenGE pre-train → OSCD fine-tune (3 seeds, AP):** n=2 scratch 0.240 / DS-fusion-teacher 0.237 / spca 0.223
· n=14 scratch 0.480 / 0.457 / 0.439. (Negative transfer — scratch wins.)

**CCA/S3CCA family (Benton HSI, AUC):** CVA 0.978 · IR-MAD 0.958 · SAM 0.919 · sparse-CCA(S3CCA) 0.906 · plain
CCA 0.707 · DS 0.508.

**Otsu ablation (OSCD test):** threshold-free best-F1 ≈ 0.34–0.37 vs Otsu hard-threshold F1 ≈ 0.14–0.15 vs
no-threshold U-Net F1 ≈ 0.49–0.51. **Normalization:** z-score AP 0.483 vs 0–1 min-max 0.218.

**Change-type clustering (Benton, 6 classes, NMI | ARI | purity):** polar-CVA 0.832|0.880|0.932 · raw-Δ
0.787|0.804|0.902 · **DS directions 0.467|0.349|0.702** · PCA-of-Δ 0.422.

**Temporal DS (real S2 sequences):** first-DS-magnitude vs trivial mean-spectrum-L2 correlation = 0.27–0.69
(al_wakrah 0.571, beijing 0.686, piraeus 0.269) — DS carries structure beyond the trivial null.
