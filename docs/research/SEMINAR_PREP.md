# SEMINAR build kit — clear, audience-aware narrative + per-slide script (2026 talk)

**Audience model:** fellow CS master's students. They know basic ML (CNN, PCA, train/test, accuracy) and some
linear algebra, but they do **NOT** know: remote sensing (satellite bands, change detection), the Fukui subspace
tradition (difference subspace, canonical angles), the CD baselines (CVA, IR-MAD), or the metrics conventions
(AP/AUROC). **So every term is defined the first time it appears, in plain words.** Better to run a bit long and
be understood than to be slick and lose the room.

How to read each slide block: **(1) title · (2) what goes ON the slide (tight) · (3) "Say:" the spoken script
that teaches it.** The deck mirrors the scripts in its Notes pane. Deep reference (full math + every experiment):
`MASTER_NARRATIVE_2026-06-22.md` (see the bottom of this file). Plain-English glossary at the very end.

> Recommended: add two short **primer** moments (folded into S3 and S7 below) — "what a satellite image is" and
> "how we score success" — so nobody is lost. They're written into the scripts.

---

## Slide 1 — Title
**On slide:** Spatially-faithful Difference Subspaces — for label-free satellite change & disaster-damage
detection · Abdelrahman I. A. Abushbeka, Kazuhiro Fukui · Subspace-Methods Lab, U-Tsukuba.
**Say:** "My project is about looking at two satellite photos of the same place — one before, one after — and
automatically finding what changed: a new building, a burned forest, a damaged neighborhood. I do it *without*
any human-labeled examples. Last year I tried our lab's geometric tool — the 'difference subspace' — for this.
This year I did something more honest and more useful: I mapped out *exactly where* that geometry actually helps
and where a much simpler method already wins — and I got a real result on actual disaster data. That balance —
geometry, measured honestly — is the talk."

## Slide 2 — Outline
**On slide:** Motivation · the method + its math (plain) · how we evaluate fairly · the main finding (the
'diagnostic') · two real results · the methods Sensei asked me to try · honest limits · contributions & future.
**Say:** "Here's the route. First why this problem matters. Then the method and just enough math, explained
simply. Then how I make sure my comparisons are fair. Then my main scientific finding. Then two concrete results
— one on a city-change benchmark, one on real disaster data. Then the specific techniques my advisor asked me to
test. Then an honest list of what I do *not* claim, and where this goes next."

## Slide 3 — Motivation (+ 30-second primer: what a satellite image is)
**On slide:** • After a disaster, responders need damage maps over huge areas — fast. • Satellites give free,
repeated before/after imagery (Sentinel-2: 13 'bands', ~5-day revisit). • A *brand-new* disaster has *no labels*
→ can't train a supervised model in time. • So we need *label-free* (unsupervised) change detection.
**Say:** "Quick background first. A normal photo has three channels — red, green, blue. A Sentinel-2 satellite
image, which is free and covers the whole Earth every few days, has **thirteen** channels, called *bands* —
including infrared light our eyes can't see, which reveals vegetation, water, and moisture. So each pixel is not
three numbers but a **vector of thirteen numbers** — its 'spectral fingerprint'. Now the problem: after an
earthquake or flood, rescue teams need to know which buildings are damaged, fast, over a whole region. Checking
on the ground is slow and dangerous. The catch is that a *brand-new* disaster has no labeled examples, so you
can't train a normal supervised neural network in time. That's why we need *label-free* methods — also called
*unsupervised* — that work straight from the before and after images with no human marking. My deeper motivation
is post-disaster reconstruction."

## Slide 4 — Why it's hard, and my advisor's challenge
**On slide:** • Naive 'spot-the-difference' fails: *everything* changes between two dates (sun angle, season,
slight misalignment). • Standard fixes look at each pixel alone → they ignore *where* things are (spatial layout).
• Our lab's tool — the 'difference subspace' — but the textbook version is *also* per-pixel. • Sensei (last year):
"your algorithm makes a subspace, but it can break the spatial information." → this year I fixed that.
**Say:** "Why can't you just subtract the two images? Because *everything* changes between two days — the sun is
at a different angle, the season changed, the satellite isn't perfectly aligned. A naive subtraction lights up
the whole image, not just the real change. The standard fixes treat each pixel completely on its own, so they
throw away *where* things are — the spatial layout. Our lab specializes in 'subspace' geometry, and last year I
applied it here — but the textbook version also treats pixels independently. My advisor's exact feedback was:
'your algorithm can make a subspace, but it breaks the spatial information.' So this year's first job was to
rebuild the method so it *keeps* the spatial layout — and then, crucially, to test honestly whether the geometry
actually helps."

## Slide 5 — The whole framework (one picture)
**On slide:** (title only) — show the color-coded framework figure.
**Figure:** `fig1c_framework.png`.
**Say:** "This is the whole pipeline, color-coded, with real Sentinel-2 examples. Read it left to right. We take
the before and after images and normalize them. The **blue** path is the geometry: I'll explain the boxes in a
moment, but the output is a 'change-score map'. The **green** path is the classical, simpler methods. The
**amber** box combines them. Then two outputs: the **coral** one is for disaster response — a map that flags
suspicious pixels for a human to check; the **purple** one feeds the geometry into a neural network. Every box
here produced a real result — those colored maps on the right are actual outputs. The next slide unpacks the blue
boxes."

## Slide 6 — The method, in plain words then symbols
**On slide:** • Each *band-image* is one sample → subspace lives in *pixel-location* space (layout kept).
• Per-date subspace via PCA: $\Phi,\Psi$ (top 12 directions). • Compare them by *canonical angles*:
$\Phi^\top\Psi=U\Sigma V^\top$. • 'Difference Subspace' $D$ = the directions that changed. • Change score per
pixel: $s_i=\sum_b\lVert P_D\,\delta_b\rVert^2$.
**Figure:** `fig9_math_reference.png`.
**Say:** "Let me define 'subspace' simply. If you have a cloud of points, PCA finds the few main directions they
spread along; the flat slice spanned by those directions is the *subspace* — a compact summary of the cloud's
shape. Its *dimension* is just how many directions you keep — here, twelve. Now the trick that keeps spatial
layout: instead of treating each pixel as a sample, I treat each whole *band-image* as one sample — so the
subspace lives in 'pixel-location space', and position is preserved. I build one subspace for the before image
(call it Phi) and one for the after (Psi). To compare two subspaces, I measure the *angles* between them — like
asking how tilted two planes are. A near-zero angle means that direction is shared, i.e. unchanged; a large angle
means a new direction, i.e. change. I bundle the changed directions into their own small space — the **Difference
Subspace**, D. Finally, for each pixel I take its before-to-after change and measure how much of it points along
those changed directions; that number is the pixel's change score. Two things my advisor asked: 'dimension' means
number of components — twelve — not the image size; and there's no separate 'project back to the image' step,
because in my construction the subspace axis *is* the image grid."

## Slide 7 — How I evaluate fairly (+ primer: the metrics)
**On slide:** • Build a 'ladder': start from the simplest method, add complexity one rung at a time. • Compare
each new method to the simplest thing that could explain its result. • Use 'matched controls' (look-alike maps)
to check a gain is *really* from the subspace. • Metrics: AUROC, Average Precision (AP), F1 — higher = better.
**Figure:** `fig2_ladder.png`.
**Say:** "Two things make my comparisons trustworthy. First, a 'ladder': I start from the most naive method —
literally subtracting the images — and climb complexity one step at a time, always asking 'does the fancier
method beat the simpler one?'. Second, 'matched controls': I make a look-alike map that uses the same kind of
math but without the real subspace content; if my method beats a competitor but the look-alike doesn't, the gain
is genuinely from the subspace, not from just adding any extra map. Now the metrics, because changed pixels are
*rare* — only a few percent of an image — so plain accuracy is misleading (a model saying 'nothing changed'
scores 97%). So I use: **AUROC**, which measures how well the score ranks changed pixels above unchanged ones —
0.5 is random, 1.0 is perfect; **Average Precision (AP)**, which is similar but designed for rare targets; and
**F1**, the overlap between predicted and true change. For all of them, higher is better."

## Slide 8 — My main finding: the diagnostic
**On slide:** A change map = how you **represent** a pixel × how you **compare** before/after × how you
**decide**. For each change type, the geometry *reduces to* a simpler statistic: • brightness change → just the
size of the difference (CVA) • simple color-shift → the angle of the difference • change spread over all bands →
a covariance/correlation statistic (IR-MAD) • change *type* → the direction of the difference. Geometry only wins
when the data is 'natively curved' (radar, brain signals) — optical satellite data is not.
**Figure:** `fig8_diagnostic_matrix.png`.
**Say:** "This is my central scientific contribution, and it's a characterization, not a horse-race win. Think of
any change detector as three choices: how you represent each pixel, how you compare before and after, and how you
decide changed-or-not. What I found is that for each *type* of change, the fancy subspace geometry mathematically
**collapses** into a much simpler statistic. If the change is just brightness — a fire, a flood — the plain size
of the difference vector (called CVA, Change Vector Analysis) already captures it. If it's a simple color shift,
the *angle* of the difference captures it. If the change is spread across all bands, a classical statistical
method called IR-MAD captures it. So geometry doesn't beat these simpler tools at raw detection. And there's a
deep reason: geometry only beats simple math when the data is *natively curved* — like radar covariance matrices
or brain-signal covariances, which live on a curved surface where straight-line math is actually wrong. Ordinary
optical satellite pixels are just vectors in flat space, so a simple statistic already does the job. Mapping
*exactly when and why* this happens is the contribution — it tells the whole field where geometry is and isn't
worth it."

## Slide 9 — Result A: the subspace adds unique evidence (city benchmark)
**On slide:** • Benchmark: OSCD — 24 city before/after Sentinel-2 pairs with human 'changed/not' labels.
• A single subspace map doesn't help a trained neural net. • But *combined* with the classical maps it adds
evidence a look-alike control can't: 0.513 (with subspace) vs 0.480 (without) vs 0.487 (look-alike). • Difference
is statistically significant (p = 0.0098). → redundant alone, valuable in combination.
**Figure:** `fig3_oscd_fusion_dsspecific.png`.
**Say:** "First concrete result, on a standard city-change benchmark called OSCD — 24 before/after Sentinel-2
image pairs where humans marked which pixels changed, so we can score ourselves. I fed the images to a U-Net — a
standard image segmentation neural network — and asked: does adding my subspace map as an extra input help? On
its own: no, the network learns that information anyway. But when I *combine* several maps together — the classical
ones plus my subspace map — and compare it to combining them with a *look-alike* control map instead, the version
with the real subspace wins: accuracy 0.51 versus 0.48, and that difference is statistically significant, p equals
0.0098, meaning very unlikely to be luck. So the honest statement is: the subspace is redundant by itself, but it
contributes something unique when used together with other evidence."

## Slide 10 — Result B: it works on real disaster damage (the headline)
**On slide:** • Dataset: xBD-S12 — real disasters (hurricanes, etc.), before/after Sentinel-2 + human-marked
*building damage*; tested on 5 *unseen* disasters. • My geometry map beats the simpler methods at finding damaged
buildings (beats PCA-diff on all 5; beats IR-MAD at localization). • Practical payoff: if a human reviews only
the most-suspicious 5% of pixels, they find ~25% of all damage — 5× better than random. • All label-free.
**Figure:** `fig4_xbd_headline.png`.
**Say:** "The headline result, on real disasters. The dataset is xBD-S12: actual hurricanes and other events,
with before and after Sentinel-2 images and human-marked *building damage*. I froze my method beforehand and
tested on five disasters it had never seen. Here, the spatial subspace geometry actually *beats* the simpler
methods at locating damaged buildings — it wins on all five events against PCA-based differencing and beats the
classical IR-MAD at localization. The practical payoff is on the right: my method gives every pixel a 'how
suspicious' score; if a human reviewer only looks at the most-suspicious five percent of pixels, they already
find about a quarter of all the damaged pixels — five times better than checking a random five percent — and
remember, no labels were used to make this map. So on the disaster task, where structure matters, the geometry
earns its place."

## Slide 11 — What that looks like (the triage picture)
**On slide:** 'Triage' = decide what a human checks first. Left: after image. Middle: the real damaged buildings.
Right: the top-5% most-suspicious pixels my method flags (red). They land on the damage.
**Figure:** `fig4b_xbd_top5_overlay.png`.
**Say:** "Let me make 'review the top 5%' concrete, because it's the whole point. The word *triage* comes from
emergency medicine — when you can't treat everyone at once, you decide who to look at first. Same idea here: a
human analyst can't inspect every pixel of a huge disaster scene, so my method ranks pixels by suspicion and says
'check these first'. Left panel: the after image. Middle: the buildings humans marked as damaged — minor, major,
destroyed. Right: in red, the 5% of pixels my method flags as most suspicious. You can see the red marks land on
the damaged-building clusters — in this scene they catch 44% of the damage — and, again, the method never saw any
labels. That's a usable disaster-response tool: it tells responders where to look first."

## Slide 12 — What I do NOT claim (honesty = credibility)
**On slide:** • Not state-of-the-art damage mapping; not pixel-perfect segmentation. • It *locates candidates*; it
does not *grade* severity (a simpler method does that better). • Only 5 disaster events → the result is a
consistent trend, not yet formal statistical significance. • The subspace's advantage is clearest with full data.
**Say:** "I want to be precise about the limits, because that's what makes the rest believable. This is not the
best-ever damage system and not pixel-perfect outlines. It finds *candidate* damaged areas to review; it does not
*grade* how severe the damage is — a simpler method actually does that better. And because there are only five
independent disaster events to test on, the strongest statistical statement possible is 'a consistent trend
across all five', not a formal p-below-0.05. I'd rather tell you that honestly than overclaim and have it fall
apart under one hard question — and being a careful, honest first attempt is exactly what my advisor values."

## Slide 13 — The methods Sensei asked me to try (all done)
**On slide:** • CCA family (a classic 'find the most-correlated combination' method): IR-MAD is iterative CCA; I
also ran sparse-CCA (the 'S3CCA' he named). The simplest method still wins; the subspace is weakest here.
• Kernel trick (nonlinear version): tested — still below the simple method. • Thresholding: I trained the network
on raw data with *no* thresholding (best); hard 'Otsu' thresholding roughly halves the score.
**Figure:** `fig5_cca_family.png` + `fig6_otsu_ablation.png`.
**Say:** "My advisor named specific techniques, so I ran every one. On the left: the 'CCA' family — Canonical
Correlation Analysis, which finds the combination of bands that's most stable between two dates and flags
deviations; the classic IR-MAD is the iterative version of this, and I also implemented the sparse variant he
named, called S3CCA. The bar chart shows that on this dataset the simplest method still wins and the subspace is
the weakest — an honest negative. I also tried the 'kernel trick', which makes the subspace nonlinear; it still
stays below the simple baseline. On the right: a question about 'thresholding' — turning a continuous map into a
yes/no map. I trained the network directly on the raw images with no thresholding, which works best, and I tested
the classic 'Otsu' automatic threshold, which roughly halves the quality. So: every requested method was tried
and reported, win or lose."

## Slide 14 — Tracking change over time (Sensei's own method, first time on satellites)
**On slide:** • For a *sequence* of dates, each date = a point on a curved surface of subspaces. • First-order
difference subspace = 'speed' of change; second-order = 'acceleration' (abruptness), split into smooth seasonal
drift vs sudden events. • First time this 2024 method is applied to satellite imagery. • A *characterization*
tool, not a detector.
**Figure:** `seq_al_wakrah.png`.
**Say:** "This part uses my advisor's own 2024 paper, and it's the first time it's been applied to satellite data.
The idea: with not two but *many* dates, each date becomes a subspace — a point on a curved surface. How fast you
move across that surface is the 'first-order difference subspace' — think of it as the speed of change. How sharply
the path bends is the 'second-order' version — the acceleration — and we can split that into smooth, gradual drift,
like seasons, versus sudden jumps, like a disaster. On a real Sentinel-2 time series you can see the curve track
the change over time. As my advisor framed it, this is a 'first and unique trial' — a way to *characterize* and
visualize change over time, not a detector that has to beat anything. It's a promising direction, not a finished
result."

## Slide 15 — Does the geometry reveal change *types*? (tested honestly)
**On slide:** • Question: can the subspace's directions tell apart *kinds* of change (e.g. vegetation vs building)?
• Tested on real data with 6 labeled change types (Benton, hyperspectral). • The simple 'direction of change'
groups the types far better (score 0.83) than the subspace directions (0.47). • Geometry loses here too — and I
report it.
**Figure:** `fig7_changetype_interpretability.png`.
**Say:** "A natural hope is that the subspace gives something a simple map can't: not just *that* a pixel changed
but *what kind* of change — vegetation turning to building, say. I tested this on real data that has six labeled
change types. I clustered pixels by their change pattern and checked how well the clusters match the true types,
using a standard agreement score where higher is better. The result: the simple 'direction of the change vector'
groups the types much better, 0.83, than the subspace directions, 0.47. So even on interpretability, the simple
method wins. I'm showing you this because checking the obvious hope and reporting the honest negative is what
makes the positive results credible."

## Slide 16 — Contributions & future work
**On slide:** Contributions: (1) the diagnostic — *when and why* subspace geometry is redundant for optical change
detection; (2) the subspace adds unique, significant evidence in combination (city benchmark); (3) it transfers
to real disaster damage as a label-free 'where to look first' tool. Future: more disaster events for statistical
proof; feed the map into a network properly; geometry-native ('manifold') networks; richer time series.
**Say:** "To summarize the contributions. First, the diagnostic: a clear map of when and why subspace geometry is
redundant for optical change detection — useful to the whole field. Second, that the subspace adds unique,
statistically-significant evidence when combined with other maps on the city benchmark. Third, that it transfers
to real disaster damage as a label-free tool that tells responders where to look first. Future work: test more
disaster events to turn the trend into formal significance; build the prior into a network more cleverly; explore
networks that are geometry-native; and richer multi-date time series for the time-tracking analysis."

## Slide 17 — Key references
**On slide:** the ~15 essentials (lab subspace papers; classical CD baselines; the datasets; the network
backbones). Full list in `REFERENCES.md`.
**Say:** "These are the key references — my lab's subspace papers, the classical change-detection methods I
compared against, the datasets, and the standard networks. The full grouped list with links is in my references
file."

## Slide 18 — Conclusion
**On slide:** • For raw detection and naming change-types, a simpler statistic wins — that's the diagnostic.
• But the subspace adds unique evidence in combination, and it transfers to real disaster-damage triage.
• A careful, honest first study — built to be defended. / Thank you — questions welcome.
**Say:** "To conclude: for raw change detection and for naming the type of change, a simpler statistic wins — and
knowing exactly why is my main finding. But the subspace genuinely adds unique evidence when combined with other
maps, and it transfers to real disaster damage as a label-free tool that prioritizes what a human should review
first. It's a careful, honest first study, built to be defended rather than oversold. Thank you — I'm happy to
take questions, and I can derive the difference subspace on the whiteboard if that's useful."

---

## Plain-English glossary (say these if anyone looks lost)
- **Change detection (CD):** comparing two images of the same place at different times to find what changed.
- **Band:** one color/wavelength channel. Sentinel-2 has 13 (incl. infrared). A pixel = a vector of 13 numbers.
- **Sentinel-2:** a free EU satellite imaging the whole Earth every ~5 days.
- **Label-free / unsupervised:** no human-marked examples needed; works straight from the two images.
- **Pixel as a vector:** each pixel is a point in 13-dimensional space (one axis per band).
- **PCA:** finds the few main directions a cloud of points spreads along.
- **Subspace:** the flat slice of space spanned by those main directions — a compact summary of a set's shape.
- **Canonical (principal) angles:** the angles between two subspaces — how differently two image-dates 'point'.
- **Difference Subspace (DS):** the bundle of directions that changed between the two dates.
- **Projection / change score:** how much a pixel's before→after change lies along the 'changed directions'.
- **CVA (Change Vector Analysis):** the length of the before→after change vector — the simplest change measure.
- **PCA-diff:** subtract the two images, run PCA, keep the main components.
- **IR-MAD:** a classic statistical change detector robust to lighting — it's a form of CCA.
- **CCA / S3CCA:** Canonical Correlation Analysis (most-correlated combination of bands); S3CCA is a sparse version.
- **AUROC / AP / F1:** ranking/overlap quality scores; changed pixels are rare so we don't use plain accuracy. Higher = better.
- **Fusion:** averaging several change-maps' rankings into one combined map.
- **'DS-specific':** a gain that a matched look-alike control cannot reproduce → genuinely from the subspace.
- **U-Net:** a standard neural network for per-pixel image segmentation.
- **Prior / prior channel:** an extra precomputed map fed into the network as a hint.
- **xBD-S12:** a real-disaster dataset (before/after Sentinel-2 + human-marked building damage).
- **Projector distance:** how much the whole subspace 'rotated' between dates, measured per pixel.
- **Triage / top-5% flagged:** ranking pixels by suspicion so a human checks the most-suspicious first.
- **Recall:** of all truly-damaged pixels, the fraction your flagged set catches.
- **Geodesic / second-order DS:** measuring 'speed' and 'acceleration' of change across a sequence of dates.
- **Diagnostic:** my main finding — the map of when/why geometry reduces to a simpler statistic.

## Deep reference (study this to understand everything behind the simplified talk)
**`docs/research/MASTER_NARRATIVE_2026-06-22.md`** — the single comprehensive document: full method + math,
the complete experimental path (everything tried, in order, with results), every results table, datasets, honest
scope, and the distilled story. Drill-downs: `CONCEPTS_EXPLAINED.md` (intuition + whiteboard scripts),
`seminar_defense_QA_2026-06-22.md` (exact implementation answers), `sensei_asks_coverage_2026-06-22.md`,
`REFERENCES.md`, `BOARD_CHEATSHEET.md`.
