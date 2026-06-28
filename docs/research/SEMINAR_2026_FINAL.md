# SEMINAR 2026 — elected idea, clear narrative, script, concepts, math (the working file)

This is the **primary seminar file** as of 2026-06-23. It supersedes the xBD-centered draft because Codex's new
**Successive Saab-DS** result is a stronger, more Sensei-aligned, *statistically supported* positive. Audience =
CS master's students with no remote-sensing/subspace background → **every term defined in plain words**. Designed
to be **edited, not rebuilt**, when Codex's train-fitted + xBD-transfer numbers land (see Part 6).

Companions: `MASTER_NARRATIVE_2026-06-22.md` (deep everything-reference + full math + research path),
`CONCEPTS_EXPLAINED.md` (whiteboard intuition), `REFERENCES.md`. Primary evidence:
`docs/experiment_reports/oscd_successive_subspace_learning_ds_2026-06-23.md`.

---

## PART 0 — The elected best idea (and why)

**Title:** *Spatially-faithful subspace construction for unsupervised multispectral change detection — from
global pixels to successive Saab features.*

**The one-sentence story:** *How you turn a satellite image into a "subspace" decides everything; we show, step by
step, that a successive local-feature construction (Green-Learning / PixelHop "Saab") finally makes Difference-
Subspace geometry beat the classical unsupervised change detectors on held-out cities — and that the subspace
adds information the features alone don't.*

**Why this is the best of everything we did:**
- It is the **only construction with a clean positive that beats the baselines** on a *frozen held-out* set with
  *statistical support* (AP 0.342 vs smoothed-PCA 0.314 / PCA-diff 0.307; p<0.05 vs raw/PCA/L2).
- It **directly answers Sensei's two biggest asks at once**: the spatial-information concern ("you broke the
  spatial information") *and* the Green-Learning / PixelHop / wavelet suggestion.
- It has the **perfect honesty arc**: the things Sensei suggested that *didn't* work (literal pyramids, wavelet
  coefficients) are reported as clean negatives, and the version that *did* work (successive Saab) is the result.
- It comes with **rigor** (matched controls isolating the subspace from the representation; 3 seeds; frozen
  protocol) and **honest limits** (transductive fitting; 10-city significance vs the very best baseline is
  marginal; external/disaster transfer pending).

**How it satisfies Sensei (the "I did what you asked" map):**
| Sensei asked for | What we show |
|---|---|
| Understand & fix the spatial-information loss | the construction journey: global pixel (loses it) → band-image (keeps it) → successive Saab (best) |
| Green Learning / PixelHop / Saab / wavelets | implemented Saab successive subspace (works) + wavelets & pyramids (tested, **don't** help) |
| Difference Subspace (1st-order) | canonical DS at each hop; DS beats L2/PCA on the *same* features |
| 1st/2nd-order DS + geodesic on sequences | temporal extension slide (characterization, his repeated ask) |
| Kernel DS / KDS | tested — stays below CVA |
| CCA / S3CCA | tested (IR-MAD = CCA; sparse-CCA = S3CCA) |
| matched controls / honest evaluation | matched L2/PCA/cross-reconstruction controls, frozen split |

**Status (2026-06-23):** the OSCD frozen result is complete. Codex is running (a) leakage-free **train-fitted**
transform vs pair-adaptive, and (b) **event-disjoint xBD-S12** external transfer. If they land in time → add 1-2
slides (Part 6). If not → present the OSCD result as-is; it stands on its own.

---

## PART 1 — Concept primer (understand this cold; teach it on the board if asked)

**A) What's the task?** Two satellite photos of the same place, before and after. Output: a per-pixel map of what
changed. We do it *unsupervised* — no human-labeled examples (because a brand-new disaster has none).

**B) What's a satellite image?** Not 3 channels (RGB) but **13 "bands"** (incl. infrared). Each pixel = a vector
of 13 numbers (its spectral fingerprint).

**C) What's a subspace?** Run PCA on a cloud of vectors → the few main directions they spread along; their span is
the **subspace** — a compact summary of the cloud's shape. Its *dimension* = how many directions you keep.

**D) The Difference Subspace (DS), in one breath.** Build a subspace for the *before* image and one for the
*after*. Measure the **canonical angles** between them (how differently the two "point"). Bundle the directions
that changed into their own small space = the **Difference Subspace**. For each pixel, score how much its
before→after change lies along those changed directions. That score map is the change map.

**E) The problem with the naive subspace (Sensei's point).** If a "sample" is one pixel's 13 numbers, then PCA
treats all pixels as an unordered bag — it throws away *where* each pixel is. The subspace knows the scene's
*colors* but not its *layout*. So global pixel DS is "spatially blind." We need a construction that keeps layout.

**F) Successive Subspace Learning / the Saab transform (Sensei's Green-Learning suggestion) — plain version.**
This is the key new piece. Think of it as a *training-free, label-free* feature extractor that builds richer
local features in stages ("hops"):
- **A hop** looks at each pixel's **3×3 neighborhood** (so it sees local spatial context, not just one pixel).
- The **Saab transform** is just a clever PCA of those neighborhoods: it separates the **DC** part (the local
  average brightness — one direction) from the **AC** part (the local *patterns* — the leading PCA directions of
  what's left). Keep ~16 channels. ("Saab" = Subspace Approximation with Adjusted Bias; here the bias cancels
  because we take differences, so it's essentially "DC + AC PCA of local patches".)
- **Pool** (shrink 2×2) and repeat for a **second hop** — now each value summarizes a *larger* region. So hop 1 =
  fine local detail, hop 2 = coarser context. This is exactly PixelHop's idea, borrowed from Green Learning.
- Crucially, the **same** transform is applied to both dates (so a difference is meaningful), and it's fit with
  **no labels**.
Result: at each hop you have a stack of feature-maps that **preserve spatial layout** and encode local
spatial-spectral patterns — a far better input for the Difference Subspace than raw pixels.

**G) Putting it together — "Successive Saab-DS".** At each hop, treat each feature-map as a sample (the
spatially-faithful "band-image" trick), build the per-date subspace, take the canonical Difference Subspace, score
every pixel, and **average the two hops**. That's the method.

**H) How we measure success (metrics).** Changed pixels are *rare* (a few %), so plain accuracy is useless. We use
**AP** (Average Precision — area under precision-recall, good for rare targets), **AUROC** (how well the score
ranks changed above unchanged; 0.5 = random), and **F1/IoU** (overlap of predicted vs true change). Higher = better.

**I) "Matched controls."** To prove the *subspace* (DS) helps and not just the *features*, we score the same Saab
features three other ways — plain distance (L2), PCA-of-difference, and a look-alike "cross-reconstruction". If DS
beats all of them on the same features, the DS step genuinely adds something.

---

## PART 2 — The slide-by-slide narrative (content + clear script)

Format per slide: **(1) title · (2) what goes on the slide · (3) "Say:" the spoken script (teaches every term).**

### S1 — Title
**On slide:** Spatially-faithful subspace construction for unsupervised multispectral change detection — from
global pixels to successive Saab features · Abushbeka & Fukui · Subspace-Methods Lab.
**Say:** "My project: compare two satellite images, before and after, and automatically find what changed —
without any human labels. The key question turns out to be how you turn an image into a 'subspace'. I'll show a
journey: the naive way fails exactly as my advisor warned, a spatially-faithful way is better, and a
Green-Learning-style 'successive' construction finally beats the classical change detectors on held-out cities."

### S2 — Outline
**On slide:** Motivation · what a subspace is · the construction journey (3 steps) · the result · the methods
Sensei asked me to try · temporal extension · honest limits · contributions.
**Say:** "I'll motivate the problem, define the few terms you need, then walk the construction journey that is the
heart of the talk, show the result, show the methods I was asked to try (some worked, some didn't), the temporal
extension, and finish with honest limits and contributions."

### S3 — Motivation (+ primer: what a satellite image is)
**On slide:** • After disasters, responders need fast damage/change maps over huge areas. • Sentinel-2: free,
global, **13 bands** (incl. infrared), ~5-day revisit. • A new event has **no labels** → can't train a supervised
model in time. • So: **unsupervised** (label-free) change detection.
**Say:** "Background first. A normal photo has three channels — red, green, blue. A Sentinel-2 satellite image,
which is free and global, has *thirteen* bands, including infrared we can't see, that reveal vegetation, water,
moisture. So each pixel is a vector of thirteen numbers. Now the problem: after a disaster, responders need change
and damage maps fast, over a whole region, and a brand-new event has no labeled examples — so a normal supervised
network can't be trained in time. That forces label-free methods that work straight from the two images."

### S4 — Why it's hard + the spatial-information problem (Sensei's challenge)
**On slide:** • Naive subtraction fails: *everything* changes between two dates (sun, season, misalignment). • To
use geometry we turn each image into a 'subspace' (a compact summary via PCA). • But the naive subspace treats
pixels as an unordered bag → it **loses where things are**. • Sensei (last year): "your algorithm makes a
subspace, but it breaks the spatial information." → this talk fixes that.
**Say:** "Why not just subtract? Because everything changes between two days, so subtraction lights up the whole
image. Our lab uses 'subspace' geometry — you summarize each image by the main directions its data spread along,
found by PCA. But here's the catch my advisor pointed out: if each sample is one pixel, PCA treats all pixels as
an unordered bag and throws away *where* each pixel is. The subspace knows the colors but not the map. His exact
words were 'your algorithm makes a subspace but breaks the spatial information.' Fixing that is the spine of this
talk."

### S5 — The construction journey (the whole framework, one picture)
**On slide:** (title only) — show the framework figure.
**Figure:** `fig_saab_framework.svg/png` (built in Part 5).
**Say:** "Here's the whole method, and the story is a three-step journey shown left to right. Step one, the naive
global-pixel subspace — spatially blind. Step two, the band-image construction that keeps spatial layout. Step
three, the new piece — a 'successive Saab' feature extractor, a label-free, training-free Green-Learning-style
stage that builds rich local features at two scales, and *then* we apply the Difference Subspace on top. The
output is an unsupervised change map. I'll unpack each step."

### S6 — Step 1: the naive global-pixel subspace (and why it fails)
**On slide:** • Sample = one 13-band pixel → subspace in 13-D 'color space'. • PCA over all pixels → ignores
position. • Difference Subspace score per pixel. • Result on OSCD: AP **0.06** — far worse than simple baselines.
→ confirms the spatial-information loss.
**Say:** "Step one, the textbook way: each pixel's thirteen numbers is a sample, PCA gives a subspace in
thirteen-dimensional color space, and the Difference Subspace compares before and after. On the OSCD city
benchmark this scores an Average Precision of just 0.06 — far below even raw subtraction. This is the
spatial-information loss made concrete: the method literally can't see structure."

### S7 — Step 2: the spatially-faithful 'band-image' subspace
**On slide:** • Flip the construction: each *whole band* is one sample (a long vector of all its pixels). • Now
the subspace lives in *pixel-location space* → layout preserved. • Result: AP **0.241** — beats the matched
geometric nulls, but still below smoothed PCA (0.268). • Progress, not victory.
**Say:** "Step two, the fix my labmate Jang suggested: flip it. Instead of each pixel being a sample, make each
whole *band-image* a sample — a long vector listing that band's value at every pixel. Now PCA's directions are
spatial patterns, and position is preserved. This jumps from 0.06 to 0.24, and it beats matched geometric control
maps — but it's still below the best simple method, smoothed PCA at 0.27. So: real progress, but not yet a win.
Something is still missing — richer local features."

### S8 — Step 3: successive Saab features + DS (the result)
**On slide:** • Add a label-free Green-Learning 'Saab' feature stage: 3×3 neighborhoods → DC+AC PCA → ~16
channels → pool → 2nd hop. • Same transform for both dates; apply the Difference Subspace at each hop; average.
• Result on frozen held-out OSCD (10 unseen cities): AP **0.342, AUROC 0.886 — best of all methods**.
**Figure:** `fig_saab_results_bar` (Part 5) + the LaTeX table.
**Say:** "Step three is the new contribution. Before doing geometry, we build better *features* with a label-free,
training-free stage borrowed from Green Learning, called successive Saab. Each hop looks at a pixel's 3×3
neighborhood — so it sees local context — and does a simple PCA that splits the local average (DC) from the local
patterns (AC), keeping about sixteen channels; then it pools and repeats once, so the second hop sees a larger
region. The same transform is applied to both dates with no labels. *Then* we run the Difference Subspace on these
features at each scale and average. On ten completely unseen OSCD test cities, this reaches Average Precision
0.342 and AUROC 0.886 — the best of every method we tried, classical or geometric."

### S9 — Two things this result proves
**On slide:** • **(1)** Successive Saab features are a much better spatial support for DS than global pixels,
fixed windows, literal pyramids, or wavelets. • **(2)** The Difference Subspace adds beyond the features: it beats
plain L2 and PCA on the *same* Saab features (vs matched PCA: +0.037 AP, **p=0.002, 10/10 cities**). → the
geometry earns its place.
**Say:** "Two findings, and the matched controls make them solid. First, the successive features are a much better
input for the subspace than any earlier construction. Second — and this is the key one — the Difference Subspace
step adds information beyond the features themselves: on the *same* Saab features, DS beats plain distance and
plain PCA, by 0.037 Average Precision, winning in all ten cities with p equal to 0.002. So it's not just better
features; the subspace geometry genuinely contributes."

### S10 — The methods Sensei asked me to try (honest: some worked, some didn't)
**On slide:** • Green Learning / Saab → **worked** (the result above). • Literal pyramids (2×2/4×4) → AP 0.207,
**below** PCA-diff — didn't help. • Wavelets (Haar/db2) → AP 0.207; detail coefficients are mostly
nuisance/misregistration — didn't help. • Kernel-DS → below CVA. • CCA / S3CCA → tested (IR-MAD is CCA; sparse-CCA
is S3CCA).
**Say:** "My advisor suggested several specific techniques, so I ran all of them and I'm honest about the outcome.
The Green-Learning successive Saab idea is the one that worked — that's the result you just saw. But the *literal*
multi-scale pyramid, with two-by-two and four-by-four grids, scored 0.21 — below plain PCA differencing. And
wavelets — both Haar and Daubechies — also scored 0.21; their high-frequency 'detail' parts mostly captured
misregistration noise, not real change. I also ran the kernel (nonlinear) version of the subspace — still below
the simplest baseline — and the CCA family he named. So: I did everything he asked, and I can tell him exactly
which ideas paid off and which didn't, with numbers."

### S11 — The diagnostic (the honest backbone)
**On slide:** A change map = how you **represent** × how you **compare** × how you **decide**. For each change
type, geometry often *reduces to* a simpler statistic (brightness→CVA; spread→IR-MAD; type→change-direction). The
contribution is knowing *when* the construction matters — which is exactly what the journey shows.
**Say:** "Underneath all this is a general lesson I call the diagnostic: a change detector is a representation
times a comparison times a decision, and for many change types the fancy geometry collapses into a simpler
statistic. That's why naive subspaces lose. The value of this project is showing *precisely* when the
*construction* — the representation — is what makes geometry finally useful. The journey from 0.06 to 0.34 is that
lesson made concrete."

### S12 — Temporal extension: 1st/2nd DS + geodesic (Sensei's repeated ask)
**On slide:** • Same per-date subspaces, but across a *sequence* of dates. • First-order DS = 'speed' of change;
second-order DS = 'acceleration', split into smooth seasonal drift vs abrupt events (geodesic). • First time
Sensei's 2024 second-order-DS is applied to satellites. • A *characterization* tool (his framing: "first/unique
trial").
**Figure:** `seq_al_wakrah.png`.
**Say:** "My advisor's most repeated request was to extend this across *time*, using his own 2024 second-order
Difference-Subspace paper. The same per-date subspaces become points on a curved surface; how fast you move is the
first-order magnitude — the speed of change — and how the path bends is the second-order — the acceleration —
which separates smooth seasonal drift from sudden events. This is the first time it's applied to satellite data.
As he framed it, it's a 'first and unique trial' — a way to characterize change over time — and I show the curves
on a real sequence."

### S13 — Disaster transfer (supporting): does the spatial subspace help on real damage?
**On slide:** • On xBD-S12 (real disasters + building-damage labels), the spatial-subspace 'projector' map beats
classical maps at *localizing* damaged buildings; reviewing the top-5% flagged pixels finds ~25% of damage (5×
lift). • Label-free triage tool. • (Codex is now testing whether *Successive Saab-DS* itself transfers here.)
**Figure:** `fig4b_xbd_top5_overlay.png`.
**Say:** "Does any of this help on *real* disasters? On xBD-S12 — actual hurricanes with human-marked building
damage — the spatial-subspace map beats the classical maps at locating damaged buildings: if a human reviews only
the most-suspicious five percent of pixels, they already find about a quarter of the damage, five times better
than random, with no labels. 'Triage' here means prioritizing what a person checks first. We're currently testing
whether the full successive-Saab method transfers to this disaster setting too."

### S14 — Honest scope (what I do NOT claim)
**On slide:** • Not SOTA; not beating modern supervised deep detectors. • **It generalizes — a transform fit only
on training cities reaches AP 0.338 on unseen test cities (≈ the per-pair 0.342), still beating all baselines.**
• 10-city significance vs the *very best* baseline (smoothed PCA) is marginal (p=0.084); vs raw/PCA it's
significant. • Disaster transfer (xBD) is **mixed** — Saab-DS doesn't beat raw L2 there; the band-image projector
remains the disaster tool. • Not full PixelHop; a scoped adaptation.
**Say:** "Honest limits. This is not state-of-the-art and doesn't beat modern supervised deep networks — it's a
label-free method. One worry I checked directly: was the feature stage just adapting to each test pair? No — when
I fit the transform *only on the training cities* and freeze it, it still reaches Average Precision 0.338 on the
unseen test cities, essentially the same as per-pair fitting, and still beats every baseline. So it's a general
representation, not a per-pair trick. With only ten test cities, the win over the single strongest baseline,
smoothed PCA, is a clear trend but not yet formally significant — though it *is* significant against raw
differencing and PCA. And on real disaster data, this exact method does not beat simple methods — so I present the
disaster result as the simpler band-image projector, and Saab-DS disaster transfer as future work. Being precise
here is what makes the positive credible."

### S15 — Contributions & future work
**On slide:** Contributions: (1) the construction journey — *how* the satellite sample definition decides whether
DS works; (2) a successive-Saab + DS method that beats classical unsupervised detectors on held-out OSCD, with DS
adding beyond the features; (3) honest negatives for pyramids/wavelets/kernel; (4) the diagnostic + temporal
extension. Future: train-fitted transform; external disaster transfer; neural prior.
**Say:** "Contributions. First, the construction journey: I show exactly how the choice of 'sample' decides
whether subspace geometry helps. Second, a concrete method — successive Saab features plus Difference Subspace —
that beats the classical unsupervised detectors on held-out cities, with the geometry adding beyond the features.
Third, clean negatives for the pyramid and wavelet ideas. Fourth, the diagnostic and the temporal extension.
Future work: a label-free transform fit only on training data, transfer to disaster data, and using the map as a
prior for a network."

### S16 — Conclusion
**On slide:** • How you build the subspace decides everything. • A Green-Learning 'successive Saab' construction
finally makes Difference-Subspace geometry beat classical unsupervised change detectors on held-out cities — and
the geometry adds beyond the features. • A careful, honest study that does exactly what Sensei asked. / Thank you.
**Say:** "To conclude: how you turn a satellite image into a subspace decides whether the geometry is useless or
useful. A label-free, Green-Learning-style successive construction finally makes the Difference Subspace beat the
classical unsupervised detectors on unseen cities — and the geometry adds beyond the features. It's a careful,
honest study that does exactly what my advisor asked, including the parts that didn't work. Thank you — I'm happy
to take questions and to derive the construction on the board."

---

## PART 3 — Rich math (for slide S6/S8 and the board)

**Global pixel DS (step 1):** sample = pixel $x\in\mathbb{R}^B$; $X^{px}\in\mathbb{R}^{B\times N}$; PCA →
$U\in\mathbb{R}^{B\times r}$; score $s_i=\lVert D^\top(x^{post}_i-x^{pre}_i)\rVert^2$. Position lost.

**Band-image DS (step 2):** sample = band-image $b_k\in\mathbb{R}^N$; $X\in\mathbb{R}^{N\times B}$; PCA →
$\Phi\in\mathbb{R}^{N\times r}$ (spatial eigenimages, $r=12$). Canonical DS: $\Phi^\top\Psi=U\Sigma V^\top$,
$\Sigma=\mathrm{diag}(\cos\theta_i)$; $D=(\Phi U-\Psi V)[2(I-\Sigma)]^{-1/2}$; score
$s_i=\sum_b\lVert P_D\delta_b\rVert^2_i$, $P_D=DD^\top$.

**Successive Saab-DS (step 3):** per hop $h$, 3×3 neighborhood $z^{(h)}(p)\in\mathbb{R}^{9K_{h-1}}$, $K_0=13$.
DC: $a_0=\tfrac{1}{\sqrt{9K_{h-1}}}\mathbf{1}$; remove DC; AC eigenvectors of the joint pre/post neighborhood
covariance, keep until 95% AC energy or 15 channels (≤16 total). Same transform both dates → 2×2 max-pool → next
hop (2 hops). At hop $h$: flatten each response map → $X_t^{(h)}\in\mathbb{R}^{N_h\times K_h}$; PCA →
$U_t^{(h)}\in\mathbb{R}^{N_h\times r_h}$, $r_h=\min(12,K_h-1)$; canonical DS $D_h$; hop score
$s_h(p)=\lVert \text{row}_p(D_hD_h^\top\Delta_h)\rVert_2$, $\Delta_h=X^{(h)}_{post}-X^{(h)}_{pre}$; normalize by
99.5th pct, clip [0,1], **average the hops**.

**Matched controls (isolate DS from the representation):** on the *same* $\Delta_h$ — direct L2 $\lVert\Delta_h\rVert$;
PCA-diff; symmetric excess cross-reconstruction between $U^{(h)}_{pre},U^{(h)}_{post}$.

---

## PART 4 — Q&A bank (clear answers; full board scripts in CONCEPTS_EXPLAINED.md)
1. *Naive baseline first?* Yes — global pixel DS (0.06) and raw CVA (0.28) are the floor; we climb to Saab-DS (0.34).
2. *What's a Saab transform?* A label-free local PCA on 3×3 patches that splits DC (local mean) from AC (local
   patterns), done in stages ("hops"), from Green Learning / PixelHop. We use it only as feature extraction.
3. *Why does the construction matter so much?* Because a subspace summarizes whatever you feed it; pixels lose
   position, band-images keep it, Saab features add local context — and that's what makes DS finally work.
4. *Does the subspace (DS) actually help, or is it just the features?* DS beats L2/PCA on the *same* Saab features
   (p=0.002, 10/10 cities) — so the geometry adds beyond the representation.
5. *Significance?* vs raw/PCA/L2: p<0.05. vs the strongest baseline (smoothed PCA): +0.028 AP, 8/10 cities, p=0.084
   — a clear trend, not yet formal significance with only 10 cities.
6. *Is it transductive?* The Saab transform is currently fit per image-pair (no labels). A train-fitted version is
   being tested to separate adaptation from a general representation.
7. *Wavelets / pyramids?* Tested (Sensei's suggestion) — both ~0.21, below PCA-diff; wavelet detail = nuisance.
8. *Kernel DS / CCA / S3CCA?* Tested; kernel-DS < CVA; IR-MAD = CCA; sparse-CCA = S3CCA.
9. *1st/2nd DS + geodesic?* Slide 12 — first satellite trial of Sensei's 2024 paper; characterization, not a detector.
10. *Disaster relevance?* xBD-S12 (slide 13): spatial-subspace map localizes damaged-building candidates, top-5% →
    ~25% of damage; we're testing whether Saab-DS itself transfers.
11. *Why not deep learning?* It's the strong default with labels; this is the label-free / interpretable / prior
    use case. We don't claim to beat supervised nets.
12. *What's AP / AUROC?* AP = precision-recall area (for rare targets); AUROC = ranking quality; higher = better;
    change is rare so plain accuracy is misleading.

---

## PART 5 — Figures & tables to use (and how to make them)
- **`fig_saab_framework`** — the 3-step construction-journey pipeline (editable SVG; built this session, see
  `phase1/outputs/seminar_figures/`). The complex, color-coded version.
- **`fig_saab_results_bar`** — bar chart of the frozen held-out OSCD table (Saab-DS best). Build with the LaTeX
  table below or a matplotlib bar.
- **LaTeX results table** — `docs/research/tables/oscd_frozen_results.tex` (compile in your slides/Overleaf).
- Reuse: `fig9_math_reference` (math), `seq_al_wakrah.png` (temporal), `fig4b_xbd_top5_overlay.png` (xBD triage),
  `fig8_diagnostic_matrix.png` (diagnostic), `fig5_cca_family.png`/`fig6_otsu_ablation.png` (methods asked).
- Codex's own figures: `docs/experiment_reports/assets/multiresolution_subspace_2026-06-23/` (frozen comparison,
  city deltas, Chongqing/Norcia qualitative) — use these for the result + qualitative slides.

---

## PART 6 — Codex's follow-up results (LANDED 2026-06-23) + remaining slot-ins
- ✅ **Train-fitted (leakage-free) OSCD:** `frozen_successive_ds` AP **0.3381**, AUROC 0.891 — ≈ pair-adaptive
  (0.342), still beats all baselines; DS beats L2/PCA on the same features (+0.032/+0.033). **→ the method is a
  general representation, not transductive.** Folded into S14. Source:
  `phase1/outputs/successive_transfer_oscd_trainfit_official_20260623_084003/summary.csv`.
- ⚠️ **xBD-S12 transfer (event-disjoint, 5 events):** Saab-DS is **mixed** — on damage-vs-intact, raw L2
  (AP 0.356) and successive-L2 (0.358) beat successive-DS (0.329); on building localization, the band-image
  *projector* (AP 0.086) wins. **→ Saab-DS does not automatically transfer to disaster; keep the projector as the
  disaster result (S13) and Saab-DS disaster transfer as future work.** Source:
  `successive_transfer_oscd12_to_xbd_native_.../summary.csv`.
- Still possibly pending from Codex: seed/rank/failure analyses + a standalone report + commit/push. If a cleaner
  xBD-S12 Saab-DS gate or a second labeled multispectral benchmark lands, slot it into S13/S14 — edit those
  slides only; the narrative arc (the construction journey + the OSCD win) is unchanged.
