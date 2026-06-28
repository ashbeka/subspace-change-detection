# Seminar narrative, refined problem statement, and the DS-prior pipeline (2026-06-22)

Author: Claude (inline synthesis; cross-checked against Codex's matched-controls + cross-branch
evidence). Status: living document for the upcoming seminar + slide build.

This document does three things the user asked for:
1. Refines the **problem statement** so it is *driven by a real-world problem*, not only a theoretical
   hypothesis — while keeping the hook the user liked ("what hidden patterns can a Difference Subspace
   surface that classical change maps cannot?").
2. Lays out the **motivation narrative** (the story arc) end to end.
3. Confirms the **naive -> complex baseline ladder** and adds the missing top rung (a learned CNN), then
   specifies the **DS-as-CNN-prior pipeline** and the **seminar skeleton** (slide by slide).

---

## 1. The refined problem statement (real-world driven)

### 1.1 The real-world driver (lead with this)
After a disaster or armed conflict, the response and the *reconstruction that follows* depend on fast,
trustworthy maps of **what changed and what was damaged**, derived from satellite imagery. But the moment
those maps are needed most is exactly when the usual machine-learning assumptions break:

- **Labels are scarce or absent.** A *new* event has no training mask. You cannot wait months to hand-label
  a city to map the city that was just destroyed. The literature is blunt about this: supervised deep models
  "will not be useful when applied to satellite images of geohazards with only few labeled data," and label
  scarcity "remains a fundamental bottleneck" for deep-learning disaster monitoring.
- **Only free, medium-resolution imagery is reliably available everywhere, repeatedly.** Sentinel-2 (10–60 m,
  13 bands, global, 5-day revisit, free) is the workhorse for rapid, repeatable, global damage assessment.
- **Speed matters.** Operational pipelines target a damage map in *minutes*, which in practice means
  *unsupervised* change detection (no per-event training).

So the operational tool of choice is **unsupervised, multispectral change detection** — and its quality
directly bounds how fast and how well a damaged area can be triaged and, later, how reconstruction is
monitored over time.

### 1.2 The technical gap inside that driver
The unsupervised change maps that responders actually use are **per-pixel**: band differencing / CVA, PCA of
the difference image, IR-MAD. They are label-free and fast — but they treat each pixel independently and
therefore **discard the spatial arrangement of the scene**. Structured, spatially-organized change (a
collapsed building block, a cleared lot, a damaged street grid) is exactly the change with the strongest
*spatial* signature, and exactly what a per-pixel statistic is blind to.

The lab's own tradition offers a candidate fix — the **Difference Subspace (DS)** — but the *conventional*
DS construction also builds its subspace over **pixels** (per-pixel spectra as samples), so it inherits the
same blindness. (This is the limitation Sensei raised: a per-pixel DS throws away spatial information.)

### 1.3 The research question (keeps the hook, now grounded)
> **Can a *spatially-faithful* Difference Subspace — one whose samples are the band-images themselves, so the
> subspace encodes the scene's spatial structure — surface structured-change information that (a) classical
> per-pixel maps and matched geometric controls cannot, and (b) even a data-scarce CNN cannot learn on its
> own from a handful of labeled scenes — so that it can serve as a label-free geometric *prior* that makes
> rapid, low-label change detection measurably better?**

This is one statement with three testable clauses, each a real experiment:
- **(a) vs classical + controls** -> the matched-controls study (done; see §3).
- **(b) vs a learned model** -> the DS-as-CNN-prior experiment (running; see §4).
- **the payoff** -> a label-free prior that helps exactly in the low-label regime the real-world driver names.

---

## 1b. The method, precisely (defense-ready — embed this; Sensei's #1 ask was "understand what you're doing")

Two constructions; we champion the second (= Jang's "flatten on the Z-axis" idea):

| | global_pixel DS (naive) | **band_image DS (spatially-faithful)** |
|---|---|---|
| one sample = | one 13-band pixel | one whole flattened band-image |
| PCA input (d,n) | (13, N_pixels) | **(N_valid, 13)** |
| subspace lives in | R¹³ (spectral) | **R^N_valid (spatial)** |
| rank r (#PCs) | 5–6 | **12** |
| spatial position | **lost** (pixels unordered) | **preserved** (position = ambient axis) |

- Each **channel is a vector (sample)**; the 13 of one date span **one** subspace. "Dimension" = r = #PCs (12).
- **Canonical first-order DS** (Fukui–Maki): `ΦᵀΨ=UΣVᵀ` (Σ=cosθᵢ) → `D=(ΦU−ΨV)[2(I−Σ)]^(−1/2)`; magnitude `Σ2(1−cosθᵢ)`. GDS = multi-class generalization; 2nd-order/geodesic DS = multi-date extension.
- **Projection / score:** per pixel `s = Σ_bands ‖P_D δ_band‖²` (`δ=post−pre`); in band_image DS the ambient axis *is* the image grid, so the score map is already spatial (no separate "project-back" step).

Full Q&A for every recurring question: [seminar_defense_QA_2026-06-22.md](seminar_defense_QA_2026-06-22.md).
Sensei-ask-by-ask coverage: [sensei_asks_coverage_2026-06-22.md](sensei_asks_coverage_2026-06-22.md).

## 2. The motivation narrative (the story arc, for the talk)

1. **Hook (real stakes).** Reconstruction after disaster/conflict needs fast, reliable change & damage maps
   from satellite — but with no local labels and only free medium-res imagery. *(Show a Sentinel-2 before/after
   of a real damaged city.)*
2. **What we actually use, and its blind spot.** The practical tool is unsupervised, per-pixel change
   detection. It is fast and label-free, but per-pixel = spatially blind. Structured change is where it leaks.
3. **The lab's instinct, and Sensei's objection.** Subspace geometry (DS) is our tradition — but the textbook
   DS is also per-pixel, so it inherits the blind spot. *(This is the honest pivot: we took the objection
   seriously and rebuilt the representation.)*
4. **The idea.** Make the DS *spatially faithful*: let each **band-image** be a sample, so the subspace lives
   in spatial-position space and encodes the scene's layout. Ask the sharp question — *does this surface change
   information the per-pixel tools and a learned CNN miss?*
5. **The discipline (the part senpais will respect).** We answer it the only honest way: a **naive -> complex
   baseline ladder** with **matched controls**. Every claim is measured against the simplest thing that could
   explain it.
6. **What we found.** (a) DS carries *DS-specific* information beyond matched geometric controls and adds
   *statistically significant* complementary evidence in fusion — but standalone it does not beat a smoothed
   PCA map. (b) Injected as a label-free prior, it *measurably improves a CNN* in the low-label regime — and
   the gain is DS-specific (the control channels do not reproduce it). *(pending final numbers)*
7. **Why it matters & what's next.** A label-free geometric prior for data-scarce change detection — the
   disaster/reconstruction setting — plus a rich research program (mechanism, attention fusion, manifold-valued
   nets, temporal/recovery, external disaster benchmark).

The arc is **so-what-proof**: it opens with real stakes, takes the strongest objection seriously, answers with
rigor, and ends with both an honest result and a real future-work program. It is explicitly *not* "subspace
geometry doesn't work."

---

## 3. The naive -> complex baseline ladder (confirmed) + matched controls

Sensei's standing requirement (from the last seminar): **start from the most naive method and climb complexity
one rung at a time, proving each rung against the one below.** This is done, on all 24 OSCD cities, AP (mean):

| Rung | Method | What it adds over the rung below | OSCD AP |
|---|---|---|---:|
| 0 (naive) | **Raw band difference / CVA magnitude** | the simplest possible change signal | 0.226 |
| 1 | **PCA-diff** | denoise the difference image with PCA | 0.254 |
| 2 | **Smoothed / multiscale PCA-diff** | add spatial smoothing | **0.268** (best single map) |
| 2 | **IR-MAD** | affine-invariant canonical-variate change | 0.214 |
| 3 (ours) | **Band-Image DS** | spatial-faithful subspace orientation change | 0.241 |
| 4 | **Rank fusion (smoothedPCA + DS + IR-MAD)** | combine complementary evidence | **0.278** |
| 5 (top, NEW) | **CNN (FC-EF U-Net)** and **CNN + DS prior** | a learned model; then + our label-free prior | see §4 |

**Matched geometric controls** (the rigor that defines this project) — same spatial sample axis as DS:

| Matched control | What it isolates | OSCD AP |
|---|---|---:|
| Normalized spatial Gram row distance | full second-moment change (all modes) | 0.142 |
| Projector-row distance | subspace *orientation* change only | 0.102 |
| Cross-reconstruction novelty | cross-date prediction residual | 0.215 |
| **Band-Image DS** | the canonical DS projection | **0.241** |

**Reads.** (i) DS *beats every matched geometric null decisively* (Gram +0.099, projector +0.139,
cross-reconstruction +0.026; all significant) — the canonical DS projection carries information those controls
do not. (ii) DS does *not* beat the strongest simple spatial map (smoothed PCA, 0.268) standalone. (iii) In a
percentile-rank fusion, DS adds **statistically significant, DS-specific** evidence: the DS three-way fusion
beats the *matched cross-reconstruction substitute* fusion by +0.0115 AP (9/10 test cities, p=0.0098), and
beats plain PCA-diff by +0.0238 (p=0.0195). The complementarity is real and is *specific to the DS operation*.

> Honest scope for the talk: standalone, Band-Image DS is *not* a new SOTA detector. The defensible positives
> are (1) it carries information matched geometric controls do not, and (2) it contributes significant,
> DS-specific complementary evidence — and, if §4 holds, (3) it helps a learned model in the low-label regime.

---

## 4. The missing top rung: DS as a label-free prior for a CNN

### 4.1 Why this experiment, and why it is the right one
The obvious senpai/Sensei objection to any hand-crafted change feature is: *"A CNN trained end-to-end on the
raw bands can learn whatever spatial feature it needs — your subspace map is redundant once you have deep
learning."* This is the **strongest baseline** and the **top of the ladder**. If the DS prior helps even here,
the contribution is real; if it does not, that is itself a clean, honest diagnostic result.

It also converts the abstract "complementarity" into the concrete thing the real-world driver cares about:
**a better change map in the low-label regime.**

### 4.2 The mechanism (why a global geometric prior could help a local learner) — not magic
- **Global vs local.** Band-Image DS is a PCA over the *whole image* (a global spatial basis from the 13
  band-images). A U-Net is local (bounded receptive field) and, on 14 training cities, cannot cheaply learn a
  global image-wide subspace decomposition. The DS map injects a **global scene-level geometric statistic** the
  CNN does not otherwise have. (CGNet, arXiv:2404.09179, motivates change priors precisely by the
  "insufficient receptive field" of CNNs.)
- **Sample efficiency.** Deep OSCD models degrade sharply as labels shrink (FC-EF "performance rapidly drops"
  from 100% to 10% of training data). A *label-free* geometric prior is a domain inductive bias that should
  help most in exactly the low-label regime — the disaster/reconstruction setting.

### 4.3 The pipeline (early-fusion prior channel) and why this design
Same small **FC-EF U-Net** (the canonical OSCD deep baseline); the **only** thing that changes between
conditions is the set of input channels. This is deliberately the *cleanest controlled test* of the hypothesis
(one variable: the prior), and it preserves the matched-control philosophy at the learned rung:

| Config | Input channels | Role |
|---|---|---|
| `bands` (M0) | [pre 13, post 13] | the strong learned baseline |
| `bands_ds` (M1) | base + **DS map** | **our method** |
| `bands_cross` (M2) | base + cross-reconstruction map | **DS-specificity control** (matched geometric null channel) |
| `bands_spca` (M3) | base + smoothed-PCA map | strongest *classical* map as a prior |
| `bands_fusion` (M4) | base + DS + smoothed-PCA + IR-MAD | full prior fusion |

All prior maps are computed on the **same normalized 13-band stacks** as the matched-controls study, use **no
labels**, and are **percentile-rank normalized to [0,1]** (identical treatment for every prior, so the
comparison is fair). Decisive comparisons:
- **M1 vs M0** — does the DS prior help the CNN at all?
- **M1 vs M2** — is the help *DS-specific*, or does any geometric channel do it? (the matched control)
- **M1 vs M3** — does DS beat the strongest classical map as a prior?

> Why early-fusion concat rather than a two-branch / attention / manifold-valued net? Because it isolates the
> scientific variable (the prior) with a single ablation, it is interpretable, and it is trainable by seminar
> time. The fancier fusions (attention gating, Grassmannian deep nets à la GrNet / topology-driven
> multi-subspace fusion, arXiv:2511.08628) are **future work**, not the claim.

### 4.4 Result (full training: 5 configs × 3 seeds × 2500 steps, 10 official test cities)

| Config | channels | AP (mean±std) | best-F1 | AUROC | IoU |
|---|---|---:|---:|---:|---:|
| `bands` (M0) | base | 0.4825 ±0.020 | 0.4928 | 0.914 | 0.346 |
| `bands_ds` (M1) | +DS | 0.4788 ±0.003 | 0.4868 | 0.923 | 0.342 |
| `bands_cross` (M2) | +cross | 0.4808 ±0.021 | 0.4938 | 0.922 | 0.351 |
| `bands_spca` (M3) | +sPCA | 0.4780 ±0.005 | 0.4833 | 0.923 | 0.342 |
| **`bands_fusion` (M4)** | +DS+sPCA+IR-MAD | **0.5128 ±0.007** | **0.5130** | **0.934** | **0.367** |

Two honest findings:
1. **A *single* DS prior channel does NOT help a well-trained CNN** (M1 0.479 ≈ M0 0.483). The large 30-step,
   single-seed early gap (0.253->0.325) **vanished** at full training: given enough optimization the CNN learns
   whatever a single DS channel supplied. The same is true for every single-prior channel (cross, sPCA all
   ~0.48). *This refutes the simple "DS prior helps a CNN" claim — and it is exactly why we test at full
   training and against controls.*
2. **The multi-prior fusion clearly and consistently helps** (M4 0.513 vs M0 0.483; +0.030 AP, +0.020 F1,
   every seed, far lower variance). This **mirrors the classical rung exactly**: a subspace map is redundant
   *alone* but valuable *in fusion* — at both the score level (rank-fusion 0.278 > best single 0.268) and now
   the learned level (fusion-prior 0.513 > bands 0.493).

### 4.4b Decisive controls: the fusion gain IS DS-specific (the positive headline)

| Config | channels | test AP (3 seeds) | best-F1 |
|---|---|---:|---:|
| `bands` (M0) | base | 0.4825 | 0.493 |
| `bands_pca_irmad` (M5) | base+sPCA+IR-MAD (**no DS**) | 0.4804 ±0.012 | 0.489 |
| `bands_cross_pca_irmad` (M6) | base+**cross**+sPCA+IR-MAD (DS→matched null) | 0.4870 ±0.008 | 0.499 |
| **`bands_fusion` (M4)** | base+**DS**+sPCA+IR-MAD | **0.5128 ±0.007** | **0.513** |

**The fusion gain is DS-specific.** Adding DS to (sPCA+IR-MAD) lifts AP 0.480 -> 0.513 (+0.032); replacing DS
with the *matched* cross-reconstruction null reaches only 0.487 (per-seed ranges of M4 vs M5/M6 do not
overlap). Note the interaction structure (important and honest): *DS alone* added to bands does nothing
(M1 0.479 ~ M0 0.483), and *sPCA+IR-MAD alone* added to bands does nothing (M5 0.480 ~ M0); only the **DS +
classical-maps combination** helps, and the matched null cannot substitute for DS. So the contribution is a
**DS-specific complementarity that a trained CNN does not capture on its own** — the same shape as the
classical rank-fusion result, now replicated at the learned rung. This is the seminar's positive headline
(modest, +0.03 AP, but consistent and DS-specific; first-trial bar, not SOTA).

> Honest caveats: one architecture (FC-EF), one dataset (OSCD), 3 seeds; the gain is an *interaction* effect
> (DS needs the classical maps to express it) whose mechanism is not yet dissected; external/multi-architecture
> confirmation is future work.

### 4.4c Label-scarcity sweep (single-channel) — NOISY, do not over-claim
Single-channel priors at n_train in {2,4,7,14}, 3 seeds (test AP):

| n_train | bands | bands_ds | bands_cross |
|---:|---:|---:|---:|
| 2 | 0.202 | 0.216 | 0.229 |
| 4 | 0.384 | 0.363 | 0.367 |
| 7 | 0.399 | 0.434 | (running) |
| 14 | 0.483 | 0.479 | 0.481 |

There is **no clean monotonic "single DS channel helps more as labels shrink" trend** (n=2 helps, n=4 hurts,
n=7 helps, n=14 neutral) — single-channel scarcity is inconclusive and must not be over-claimed. The
meaningful scarcity question is whether the **DS-specific FUSION** gain (M4 vs M5) holds/grows as labels
shrink; that sweep (`bands` vs `bands_fusion` vs `bands_pca_irmad` at n=2,4,7) is the next run.

### 4.5 Honest threats to validity (state these in the talk)
- **Transductive priors.** The DS map is computed per image (unsupervised, no labels), including test-image
  pixels — standard for unsupervised CD priors, but we say so plainly.
- **Seed variance.** OSCD deep models are seed-sensitive; hence 3 seeds + mean±std, and the *control channels*
  as the real test (a DS win must exceed the cross/PCA channel wins, not just M0).
- **Small test set (10 cities).** We report per-city wins + spread, and treat results as internal evidence
  until an external disaster benchmark (xBD-S12) confirms.

---

### 4.6 The fully-unsupervised counterpart: subspace-teacher self-training (running)
The input-prior test (above) puts DS *inside* the model. The label-scarce driver's *native* paradigm is
**pre-detection pseudo-labeling / teacher->student**: an unsupervised TEACHER makes a change map, it is
confidence-filtered into pseudo-labels, and a deep STUDENT trains on them with **no human labels** (SemiSiROC
2023; CVA / PCA+KMeans / DCVA / **DSFA** pre-detection; survey arXiv:2502.02835). Note DSFA (Deep Slow Feature
Analysis) is itself a *subspace-family* pre-detector — so "subspace as teacher" is a natural, under-explored slot.

Experiment (`phase1/experiments/ds_teacher_student.py`): teacher in {CVA, IR-MAD, smoothed-PCA, DS, **DS-fusion**,
fusion-without-DS, fusion-with-cross} -> double-threshold confidence filter (top 10% positive / bottom 50%
negative / middle ignored) -> FC-EF student trained on pseudo-labels seeing **only raw bands** (no teacher
leakage) -> evaluated on the 10 official test cities with real labels. Reads: (i) does the student *distill/
denoise above* the raw teacher map? (ii) does the **DS-fusion teacher** train the best student? (iii) is it
**DS-specific** (DS-fusion vs fusion-without-DS vs fusion-with-cross teacher)? This is the experiment that most
directly matches the disaster/low-label driver. *(numbers pending.)*

> This sidesteps the "input-stacking = information leakage" critique entirely: the student never sees the
> teacher map, so it must learn raw-image features; the teacher only supplies the (label-free) training target.

### 4.7 Codex's complementary external transfer (xBD-S12)
Codex froze a pre-registered protocol testing whether the *same* matched-controls result (DS beats its
cross-reconstruction substitute; DS-fusion beats cross-fusion) **transfers to an independent disaster dataset**
— xBD-S12 (co-registered pre/post Sentinel-2 + xBD building-damage masks, event-disjoint split, AP primary).
Division of labour: **Codex = does the DS complementarity transfer to disaster damage (score level);
this work = does it survive/help a learned model and can it teach a label-scarce student (OSCD).** Together
these are the two pillars a paper needs: external transfer + a learned-model + label-scarce use case.

### 4.8 Time-sequential 1st/2nd DS + geodesic (Sensei's most-repeated ask; the ACCV "first/unique trial")
On 3 real S2 sequences (al_wakrah/Qatar, beijing_airport, piraeus; 20 dates each;
`phase1/experiments/temporal_ds_sequence.py`) using the spatially-faithful band-image subspace per date, we
compute the **first-DS magnitude** (change "velocity"), **second-DS magnitude** (change "acceleration"/
abruptness) and the **along- vs orthogonal-geodesic** split across the sequence. Result: the first-DS magnitude
tracks change over time but is only **partially** explained by the trivial mean-spectrum-L2 (corr 0.27–0.69), so
DS carries temporal structure beyond the trivial null; the 2nd-DS/geodesic localizes abrupt vs smooth periods.
Honest framing (per Sensei): a first/unique *analysis* trial, not a SOTA detector — a strong characterization
slide and a direct answer to his repeated request. (Note: ipol416 = 4-band RGBI; a 13-band GEE S2 sequence is
the natural extension.)

## 5. Novelty & positioning (mirror the field's rhetoric)

The field *does* inject change priors into CNNs — but the closest precedents differ from ours on the axis that
matters:
- **Deep-feature priors** (CGNet, arXiv:2404.09179): the prior is generated *from the network's own deep
  features*. Ours is **computed from raw images, label-free, before any learning**, and is **global/geometric**.
- **Hand-crafted / affinity change priors** (arXiv:2001.04271; synthetic-change methods): use predefined
  change rules / affinity indicators. Ours is a **subspace-geometry** change indicator, and — uniquely — we
  **prove with matched controls that the contribution is DS-specific**, not generic.
- **Grassmannian deep networks** (GrNet; topology-driven multi-subspace fusion, arXiv:2511.08628): heavy
  manifold-valued architectures. Ours is a **lightweight, interpretable** injection of a subspace-derived map
  into a standard CNN — feasible, and a stepping stone to the manifold-valued version.

> **The gap we fill (the one-line novelty):** *the first use of a spatially-faithful Difference Subspace as a
> label-free, global geometric prior for data-scarce change detection — rigorously shown, via matched
> controls, to carry change information that classical maps, matched geometric nulls, and a learned CNN do not
> capture on their own.* This is squarely Sensei's bar: a novel, honest *first trial*, not a SOTA chase.

---

## 6. Future work (rich, real)
1. **Mechanism / interpretability.** *What* spatial structure does the DS-specific complementarity capture that
   PCA/IR-MAD and the CNN miss? Map where the DS channel flips the CNN's decision; relate to building/road/
   structural change.
2. **Better geometry+NN fusion.** Attention/gating on the DS map; two-branch fusion; **manifold-valued**
   (Grassmannian) layers that consume the subspace directly rather than its scalar map.
3. **External disaster validation.** xBD-S12 (co-registered pre/post Sentinel-2 + building-damage masks, event
   split) — the frozen gate that would turn "internal evidence" into a publishable claim.
4. **Temporal / recovery.** Extend the spatially-faithful subspace across a time series for
   damage-then-recovery monitoring (the reconstruction half of the motive); ties to the multi-band recovery
   positive already found.
5. **Spatial × spectral × temporal product-Grassmann** for full structured change.
6. **2nd-order DS geodesic decomposition** as an interpretable change-*type* descriptor (shown visually).

---

## 7. Seminar skeleton (slide by slide)

> Target: ~14–16 slides, ~20–25 min. Tone: real stakes -> sharp question -> disciplined answer -> honest
> result -> rich future work. One idea per slide; one figure per claim.

1. **Title** — Spatially-faithful Difference Subspaces for label-scarce satellite change detection.
2. **Real-world stakes** — disaster/conflict reconstruction needs fast change & damage maps; before/after
   Sentinel-2 of a real damaged area. *(the driver)*
3. **The operational reality** — no labels for a new event; free medium-res Sentinel-2; speed -> unsupervised
   change detection is the tool.
4. **The blind spot** — the unsupervised tools (CVA / PCA-diff / IR-MAD) are *per-pixel* -> spatially blind;
   structured damage is exactly what they miss. *(figure: per-pixel map missing structured change)*
5. **Our lab's instinct & Sensei's objection** — DS is our tradition, but textbook DS is also per-pixel.
   *(honest framing)*
6. **The idea** — spatially-faithful DS: each *band-image* is a sample; the subspace encodes scene layout.
   *(small schematic: pixels-as-samples vs band-images-as-samples)*
7. **The question** — can it surface change that per-pixel maps *and* a learned CNN miss? (the 3 clauses)
8. **The method: discipline** — the naive->complex ladder + matched controls (the project's spine).
9. **Result A: matched controls** — DS beats every matched geometric null; not the smoothed-PCA map standalone.
   *(the AP table / bar chart)*
10. **Result A: fusion complementarity** — DS adds significant, DS-specific evidence in fusion (p=0.0098 vs the
    matched substitute). *(forest/CI plot)*
11. **The hardest test** — "a CNN makes hand-crafted features redundant" — so we test it: DS as a label-free
    prior channel in an FC-EF U-Net; controls = cross / smoothed-PCA channels.
12. **Why a global prior could help a local learner** — global-vs-local + sample-efficiency (the mechanism).
13. **Result B** — M1 vs M0 (does it help?), M1 vs M2/M3 (is it DS-specific?). *(the ladder table incl. CNN
    rung; bar chart with error bars)* *(pending)*
14. **Honest scope** — what we do *not* claim (not standalone SOTA; internal evidence; transductive prior).
15. **Future work** — mechanism, attention/manifold fusion, xBD-S12, temporal recovery. *(roadmap)*
16. **Takeaway** — a label-free, spatially-faithful *geometric prior* for label-scarce change detection;
    rigor-first; first trial, real future program.

---

## 8. Reproduction
- Matched controls: see `docs/experiment_reports/oscd_band_image_matched_spatial_controls_2026-06-22.md`.
- DS-as-CNN-prior: `phase1/experiments/unet_ds_prior.py` (this branch's working tree).
  - quick: `python -m phase1.experiments.unet_ds_prior --configs bands,bands_ds --seeds 0 --steps 1500`
  - full:  `... --configs bands,bands_ds,bands_cross,bands_spca,bands_fusion --seeds 0,1,2 --steps 2500 --tag main`
  - cache + results under `phase1/outputs/unet_ds_prior/`.
