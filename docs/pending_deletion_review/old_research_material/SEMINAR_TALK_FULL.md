# SEMINAR TALK — the full expressive track (you build the slides; this is the amalgamation)

You construct the slides; this gives you, for each one: **the slide # · what goes ON the slide (kept light) ·
which figure to screenshot from which paper · and the SCRIPT — written expressively, the way you'd actually
explain it to people who have never heard of any of this.** Every term (subspace, band-image, Saab, Green
Learning, difference subspace, …) is *explained in the script itself*, with the chain of thought, and with the
honest "they asked me, here's what I did" framing.

**The "good presentation" rules I took from Pedro's defense (don't copy it literally — it's a 67-slide PhD
defense; we keep 20 min):** start from the big picture; then **teach the background concepts one at a time, each
on its own slide with one figure**, before showing your method; only then the method, the result, the honest
limits. Assume the audience knows *nothing* about remote sensing or subspaces. Build up.

Target: **~20 slides, ~20 min** (~1 min each; the teaching slides can be 40–60 s). Cut the ones marked
*(optional)* if you run long.

---

## SLIDE 1 — Title
**On slide:** "Spatially-faithful subspace construction for unsupervised change detection — from global pixels to
successive Saab features." Your name, Fukui lab, date.
**Script:** "Hello everyone. My work is about comparing two satellite photos of the same place — a 'before' and an
'after' — and automatically finding what changed, using no human-labeled examples. Last year I presented a first
attempt with our lab's 'difference subspace' tool. This year I went much deeper, and I found that the single most
important decision is something subtle: *how* you turn a satellite image into a mathematical object called a
'subspace'. I'll walk you through that as a journey, and show that the right construction finally makes the
geometry beat the standard methods."

## SLIDE 2 — What I want you to leave with (the outline as a promise)
**On slide:** 3 bullets: (1) what the problem is and why it's hard; (2) the key idea — the construction of the
subspace decides everything; (3) the result — a Green-Learning + difference-subspace method that wins on unseen
cities, plus an honest account of what didn't work.
**Script:** "Three things I want you to leave with. First, the problem and why it's genuinely hard. Second, the one
big idea: how you build the subspace from the image is what decides whether the geometry is useless or useful.
Third, a concrete result — and I'll be honest about the parts my advisor asked me to try that did *not* work,
because that's part of the story too. I'll teach every concept as we go, so don't worry if some words are new."

## SLIDE 3 — Motivation: why automatic change detection from satellites
**On slide:** before/after of a disaster area (use `panel_montpellier.png` or a disaster pair). Bullets: disasters
need fast maps over huge areas; ground survey is slow/dangerous; satellites are free and global.
**Script:** "Let me motivate this with the real-world reason I care. After an earthquake, a flood, or a conflict,
the people responding need to know — fast, over a whole region — which areas and buildings are damaged, so they
can send help and plan reconstruction. Going building by building on the ground takes weeks and is dangerous.
Satellites, on the other hand, photograph the entire Earth for free every few days. So the dream is: take the
satellite picture from before and the one from after, and automatically highlight what changed. My deeper personal
motivation is post-conflict reconstruction."

## SLIDE 4 — Background concept ①: what a satellite image actually is
**On slide:** a simple diagram: a normal photo = 3 channels (R,G,B); a Sentinel-2 image = 13 'bands' (incl.
infrared). One pixel = a vector of 13 numbers.
**Screenshot (optional):** a Sentinel-2 band illustration — e.g. from the OSCD dataset page (Daudt et al.) or any
"Sentinel-2 bands" figure; or keep it as a simple hand-made graphic.
**Script:** "First piece of background, and it's important. A normal photo has three channels — red, green, blue.
A Sentinel-2 satellite image, which is the free European satellite I use, has *thirteen* channels, which we call
'bands'. Some of those bands are infrared light our eyes can't see — and those are gold, because they reveal
vegetation health, water, moisture, soil. The practical consequence: a pixel is no longer three numbers, it's a
**vector of thirteen numbers** — think of it as that pixel's 'spectral fingerprint'. Hold on to that picture,
because everything I do is geometry on these thirteen-dimensional pixel vectors."

## SLIDE 5 — Background concept ②: the task, and why naive subtraction fails
**On slide:** two images → minus sign → a messy difference that lights up everywhere. Caption: "everything
changes between two dates (sun, season, alignment)."
**Script:** "The task, concretely: output a map labeling each pixel 'changed' or 'not changed'. The naive idea is
to just subtract the after image from the before image. But here's why that fails: between two days, *everything*
is a little different — the sun is at a different angle, the season changed the vegetation, and the satellite
isn't perfectly re-aligned to the same pixels. So a raw subtraction lights up the whole image, not just the real
change. The entire field exists to separate *meaningful* change from this background noise. And because a brand-new
disaster has no labeled examples, I work *unsupervised* — no training labels, just the two images."

## SLIDE 6 — Background concept ③: what a 'subspace' is (the lab's language)
**On slide:** a 3D dot-cloud lying near a tilted plane, with 2 arrows on the plane. Caption: "PCA → main
directions → their span = a subspace; dimension = how many directions."
**Screenshot (optional):** any PCA/subspace illustration, or draw it.
**Script:** "Now the core tool of our lab, explained from scratch. Suppose I have a big cloud of vectors — say all
the pixel-fingerprints of an image. If I run PCA, Principal Component Analysis, it finds the few directions along
which that cloud spreads out the most. Picture a flat, tilted sheet of paper that best fits a cloud of dots in 3D
— that sheet is a two-dimensional **subspace**. So a subspace is just a *compact summary of the shape of a set*:
'this data mainly lives along these few directions.' The number of directions you keep is the subspace's
*dimension*. Our lab compares things — faces, motions, images — by comparing their subspaces. That's the whole
philosophy."

## SLIDE 7 — Background concept ④: the Difference Subspace (DS)
**On slide:** two tilted planes (before-subspace, after-subspace); mark the shared direction (angle ≈ 0) and a
sticking-out direction (large angle); bundle the changed directions = D. Add the tiny scoring line:
`pixel score = length of before-after change projected onto D`.
**Screenshot:** **Fukui & Maki, "Difference Subspace and its Generalization for Subspace-Based Methods," IEEE
TPAMI 2015** — the figure illustrating two subspaces and their difference subspace (canonical angles / difference
directions). Also good: **Fukui et al., "Second-order Difference Subspace," arXiv:2409.08563 (2024), Figure 1**.
**Script:** "Building on that: how do we compare a 'before' subspace and an 'after' subspace? We measure the angles
between them — called *canonical angles*. Imagine two tilted sheets of paper meeting in a room; they do not differ
by one angle only. They differ direction by direction. A near-zero angle means this direction is shared by before
and after, so I treat it as stable. A large angle means this direction is different between before and after, so it
is a candidate change direction.

Professor Fukui's Difference Subspace collects those changed directions into one new subspace, which I call D. The
important thing is how D becomes an image map. For each pixel, I first compute its before-to-after feature change.
Then I ask: how much of that change lies inside D? If the projection is large, the pixel receives a high change
score. If the projection is small, the pixel is probably stable or only changed in a common nuisance direction. So
DS is not just a global number between two images; it gives a per-pixel changed-area evidence map."

## SLIDE 8 — The problem my advisor pointed out (the honest setup)
**On slide:** "Sensei (last year): your algorithm makes a subspace, but it breaks the spatial information." A
picture of pixels being thrown into an unordered bag. Add caption: "same 13-D values, but coordinates are ignored
during PCA."
**Script:** "Now the honest pivot that drove this whole year. Last year, when I built the subspace, each *pixel*
was a sample. For example, Beirut gave about 1.26 million samples, and each sample was only thirteen band values.
The problem — and my advisor said this to me directly — is that PCA then treats the pixels as an *unordered bag*.
The pixel from a road, the pixel from a sea region, and the pixel from a building block are all mixed only by their
thirteen values. The fitting step does not know their row and column coordinates.

That is what 'breaking spatial information' means here. The score can still be reshaped back into an image later,
but the subspace itself was learned without knowing image layout. His exact words were: 'your algorithm can make a
subspace, but it breaks the spatial information.' He was right, and fixing that is the spine of today's talk."

## SLIDE 9 — Step 1: the naive 'global pixel' subspace (and proof it fails)
**On slide:** diagram: pixel (13 numbers) → PCA over all pixels → subspace in 13-D 'color space' → "position
LOST". Add matrix cue: `X_pre: 13 x N`, `N = pixel locations`. Big number: **AP = 0.06**.
**Script:** "Step one is exactly the naive way. Each pixel's thirteen numbers is a sample; PCA gives a subspace in
thirteen-dimensional color space; the difference subspace compares before and after. In matrix language, I build
`X_pre` and `X_post` as `13 x N`, where N is the number of valid pixel locations. This construction answers one
question only: did the distribution of thirteen-band pixel values change? It does *not* answer: where are the roads,
blocks, borders, or local structures?

On the standard city-change benchmark, this scores an Average Precision of 0.06 — far below even raw subtraction.
This is my advisor's warning made into a number: with no sense of spatial structure, the global pixel subspace is
basically blind. So we have to change the construction."

## SLIDE 10 — Background concept ⑤: 'band-image', and Step 2 that fixes the space
**On slide:** Left: define 'band-image' — one band viewed as a whole grayscale picture; an image = 13 stacked
band-images. Right: Step 2 — make each *band-image* a sample → subspace lives in spatial space, position KEPT.
Add matrix cue: `X_pre: N x 13`, rows = pixel positions, columns = band-images. If someone asks about notation:
"this is the same data as `13 x N` transposed; here I write it this way to show that the spatial dimension is
being modeled." Number: **AP = 0.24**.
**Script:** "To fix it I need one new word: **band-image**. A Sentinel-2 image has thirteen bands. A band-image is
one of those bands viewed as a whole grayscale image. For example, the near-infrared band-image is the full picture
you get if you keep only the near-infrared channel.

Now the sample definition changes. Instead of one pixel being one sample, one entire feature map is one sample. So
for a thirteen-band image, I have thirteen image-shaped samples. If I flatten each band-image, the matrix becomes
`N x 13`: rows are spatial positions and columns are whole band-images. PCA directions now live over the image
grid, so a basis vector is a spatial pattern, not only a thirteen-dimensional color direction. If you prefer the
usual subspace notation, this is just the transpose of the same data; the important point is that the basis lives
in the spatial dimension.

That is what I mean by more spatially faithful: the coordinate layout is inside the vector being modeled. This
jumps the score from 0.06 to 0.24, and it beats matched geometric controls. But it is still below the best simple
method. So it fixes the construction problem, but the raw band-images are still weak features."

## SLIDE 11 — A quick word on how we score (the metric, so the numbers mean something)
**On slide:** "Changed pixels are rare (a few %) → plain accuracy is useless. We use AP and AUROC; higher =
better." A tiny precision-recall sketch.
**Script:** "Quick aside so the numbers mean something. Changed pixels are *rare* — only a few percent of an image
— so plain accuracy is misleading: a lazy model that says 'nothing changed' is 97% accurate and 100% useless. So
we use two ranking scores. AUROC asks: if I rank pixels by my score, how well do the truly-changed ones float to
the top? Half is random, one is perfect. Average Precision, or AP, is similar but tuned for rare targets. For both,
higher is better. When I say 0.34 beats 0.31, that's on these scales."

## SLIDE 12A — Why Green Learning / PixelHop enters the story
**On slide:** "Problem after Step 2: the subspace keeps image position, but the raw bands are still weak local
features." Then a compact comparison:

- CNN: learns filters with labels + backpropagation.
- Green Learning / PixelHop: learns local image filters from unlabeled image statistics.
- What I borrow: the label-free local feature extractor, not the classifier.

**Best screenshot:** use **Image #1**, but crop or highlight only the top "PixelHop Unit -> pooling -> PixelHop
Unit" row. Do **not** focus on the LAG/classifier blocks, because my method does not use those supervised parts.
If the slide becomes crowded, replace Image #1 with a simplified hand-made version of the top row.

**Script:** "Step two fixed one problem: it kept the spatial layout, because each band-image was treated as a
whole image-shaped sample. But it still used the raw Sentinel-2 bands directly. Raw bands are useful, but they are
not rich local features. They do not explicitly describe small structures like edges, corners, texture, and local
neighborhood patterns. So the question became: can I build better local features before applying our subspace
geometry?

This is where Green Learning and PixelHop enter. A normal CNN also learns local image filters, but it learns them
by backpropagation and usually needs labels. Green Learning, from Professor C.-C. Jay Kuo's group, asks a different
question: can we design useful image features in a feed-forward, interpretable, label-free way, using image
statistics? PixelHop is one concrete method in that family. It processes the image in successive stages, or
'hops'. Each hop looks at a local neighborhood, transforms it into feature maps, optionally shrinks the spatial
size, then repeats at a coarser scale.

The important sentence for my work is this: I am not using PixelHop as a classifier. I borrow only its
unsupervised local feature-extraction idea, because it gives me spatially richer feature maps before I apply the
Difference Subspace."

## SLIDE 12B — The PixelHop unit: local neighborhood -> Saab transform -> feature maps
**On slide:** "One PixelHop unit = collect local neighborhoods, then apply Saab." Put the three-step chain:

```text
3x3 neighborhood around each pixel -> vector of local values -> Saab transform -> new feature maps
```

Under it:

- Local neighborhood vector:

$$
\mathbf{x} \in \mathbb{R}^{9K}
$$

- DC component = local average / constant direction:

$$
y_0 = \mathbf{a}_0^\top \mathbf{x},
\qquad
\mathbf{a}_0 =
\frac{1}{\sqrt{9K}}
\begin{bmatrix}
1 & 1 & \cdots & 1
\end{bmatrix}^{\top}
$$

- AC components = learned local variation patterns:

$$
\mathbf{r} = \mathbf{x} - y_0\mathbf{a}_0,
\qquad
y_j = \mathbf{u}_j^\top \mathbf{r}
$$

where \(\mathbf{u}_j\) are PCA directions learned from unlabeled neighborhood residuals.
- A hop = applying this unit once; successive hops = repeat after pooling.

**Best screenshot:** use **Image #2**. It is the cleanest figure for explaining a PixelHop unit: 3x3 neighborhood
construction, vectorization into `9 x K_{i-1}`, and dimension reduction to `K_i` by the Saab transform.
**Optional backup:** Image #3 is useful only if you want to show the Saab DC/AC filter idea from the feed-forward
CNN paper. Image #4 is just LeNet architecture and should not be used here.

**Script:** "Now let me unpack the technical word 'Saab', because this is where the feature maps come from. At one
pixel, take the 3 by 3 neighborhood around it. If the current image has K feature channels, that neighborhood
becomes one vector \(\mathbf{x}\) with \(9K\) numbers. Saab first projects this vector onto the constant direction:
\(y_0 = \mathbf{a}_0^\top\mathbf{x}\). Here \(\mathbf{a}_0\) is just the all-ones vector, normalized. That is the
**DC component**: roughly the local average level.

Then Saab subtracts that average part:
\(\mathbf{r} = \mathbf{x} - y_0\mathbf{a}_0\). What remains is local variation. PCA is applied to those residuals,
and each AC coefficient is \(y_j = \mathbf{u}_j^\top\mathbf{r}\). These **AC components** behave like learned local
pattern filters: edges, corners, texture, and band-neighborhood contrast. So DC tells me 'what is the local level?',
and AC tells me 'what is the local structure around that level?'

The reason this matters for my project is that Difference Subspace should not compare only raw band values. It
should compare feature maps that already contain local spatial context. After Saab, each date image becomes a stack
of DC/AC feature maps. I then build band-image subspaces from those feature maps and let DS compare the before and
after geometry. One Saab stage is one hop; after pooling, a second hop captures coarser local structure."

## SLIDE 13 — Step 3: putting it together — Successive Saab-DS (the method)
**On slide:** the pipeline figure (`fig_saab_pipeline.png`):
`13-band images -> 3x3x13 local vectors -> Saab DC/AC feature maps -> pool -> Hop 2 -> per-hop Band-Image DS -> fused change map`.
Use a small side note: "feature response = one scalar filter output; feature map = that response over all pixels."
**Screenshot:** none — this is your own figure.
**Script:** "Now I can put the pieces together. The input is a before image and an after image, each with thirteen
Sentinel-2 bands. The first part of the pipeline is only feature extraction. Around every pixel, I take a 3 by 3 by
13 local neighborhood, flatten it into 117 numbers, and apply the Saab transform. A feature response is just one
number produced by one learned Saab direction at that pixel. If I compute that same response at every pixel, I get
one feature map. In my run, Hop 1 produces fourteen feature maps, so each pixel location now has a fourteen-number
local feature vector instead of only thirteen raw band values.

So after Hop 1, I no longer have only raw bands. I have a stack of local feature maps that still have the same
image layout. Then I apply a 2 by 2 pooling step, which shrinks the feature maps, and I repeat the same Saab idea
again. This second hop looks at neighborhoods of the Hop-1 feature maps, so it captures coarser local structure.
In my run, Hop 2 produces twelve feature maps.

Only after this feature stage do I apply the lab's geometry. At each hop, I treat each feature map as one
band-image sample. So for Hop 1, the pre-change image gives a matrix whose rows are spatial pixel positions and
whose columns are the fourteen Saab feature maps. The post-change image gives the same kind of matrix. I fit one
subspace for the pre image and one subspace for the post image. Then Difference Subspace finds the directions where
the before-subspace and after-subspace separate.

Finally, every pixel gets a score: how much does its before-to-after feature change point in the Difference
Subspace direction? Hop 1 gives a fine-scale score map. Hop 2 gives a coarser score map. I resize the Hop-2 map
back to the original size and average the two maps to get the final changed-area evidence.

The bottom branch of the figure is the fairness check. I use the exact same Saab features, but score them with
three simpler alternatives. Raw feature distance asks only 'how large is the feature difference?' PCA difference
asks 'does the change lie in the main PCA directions?' Cross-reconstruction asks 'can the before-subspace
reconstruct the after-features?' These controls ask a very important question: is the improvement coming only from
better features, or does the Difference Subspace geometry add something beyond those features?"

## SLIDE 14 — The result: it finally beats the baselines (frozen, held-out)
**On slide:** the result bar (`fig_saab_results.png`) and/or Codex's `frozen_test_method_comparison.png`. Big
numbers: AP **0.342**; train-fitted **0.338**.
**Script:** "Here is the payoff, on ten cities the method never saw, with the whole thing frozen in advance. The
successive Saab plus difference subspace reaches Average Precision 0.342 in the pair-adaptive setting, which is the
best OSCD result among my unsupervised classical and geometric comparisons. It beats smoothed PCA, plain PCA
difference, raw difference, and IR-MAD in this benchmark.

But there is an important concern: maybe the feature extractor is secretly adapting to each test image. So I ran a
stricter check. I fit the Saab feature extractor only on the training cities, froze those filters, and applied the
same fixed filters to the unseen test cities. Under that stricter train-fitted protocol, the score is still 0.338.
That means the representation is not only a per-image trick; it transfers across OSCD cities."

## SLIDE 15 — Why it's solid (the two findings + controls)
**On slide:** (1) Saab features beat global pixels, windows, pyramids, wavelets as a support for DS. (2) The DS
step adds *beyond* the features: it beats plain L2/PCA on the *same* features (+0.037 AP, p=0.002, 10/10 cities).
**Script:** "This result matters because of the controls. First, the Saab feature construction is clearly a better
support for DS than the earlier constructions I tried: global pixels, local windows, fixed pyramids, and wavelets.
That answers the representation question.

Second, I tested whether DS itself adds anything. I took the exact same Saab feature maps and compared four scoring
rules. If plain L2 won, then my result would just mean 'Saab features are good.' If PCA difference won, then the DS
geometry would not be necessary. But Difference Subspace wins on the same features: it improves AP by about 0.037
over the matched plain feature distance/PCA controls, and it wins in all ten test cities. So the result is not only
'better local features'; the subspace comparison step contributes measurable information."

## SLIDE 16 — "I did what Sensei asked" (honest: what worked, what didn't)
**On slide:** a small table: Green Learning/Saab → WORKED · pyramids (2×2/4×4) → 0.21, didn't help · wavelets
(Haar/db2) → 0.21, detail = nuisance · kernel-DS → below CVA · CCA/S3CCA → tested. (Use `fig5_cca_family.png` +
`fig6_otsu_ablation.png`.)
**Screenshot (optional):** a wavelet 2-D decomposition figure (LL/LH/HL/HH) — e.g. from **Mallat, "A Theory for
Multiresolution Signal Decomposition," IEEE TPAMI 1989**, or any standard wavelet-transform illustration.
**Script:** "I want to be completely transparent on this slide, because it matters to my advisor. Over the past
year he suggested several specific techniques, and honestly, last year I had *not* tried all of them. This year I
made a point of running every single one, so I can stand here and show exactly what I did — including the things
that failed. He suggested Green Learning; that's the Saab idea, and it worked — that's the result you just saw. He
suggested *wavelets* and *multi-scale pyramids* as other ways to add spatial structure; I implemented both, and
I'll be honest — they did *not* help. They scored around 0.21, below plain PCA differencing, because the
high-frequency 'detail' parts mostly captured misalignment noise, not real change. He asked me to try the *kernel
trick* and *Canonical Correlation Analysis*; I ran those too — the kernel version stayed below the simplest
baseline, and IR-MAD, which I already use, *is* a form of CCA. So this slide is me closing the loop: I did what I
was asked, and I can tell you with numbers what paid off and what didn't."

## SLIDE 17 — The bigger lesson: the diagnostic *(optional / can shorten)*
**On slide:** a change detector = representation × comparison × decision; for many change types the fancy geometry
*reduces to* a simpler statistic; the win comes from the *construction*. (`fig8_diagnostic_matrix.png`.)
**Script:** "Stepping back, the general lesson is simple. A change detector has three parts: representation,
comparison, and decision. Representation means what object I compare: raw pixels, band-images, Saab features, or a
time sequence. Comparison means the score: L2, PCA, IR-MAD, DS, or another geometry. Decision means how I threshold
or rank pixels.

The early failure taught me that a sophisticated comparison method cannot rescue a weak representation. Global
pixel DS used a valid mathematical DS, but the representation had already destroyed spatial structure. The later
success taught the opposite: once the representation contains local spatial features, the DS comparison can finally
add value. So the lesson is not 'DS always wins.' The lesson is: the construction decides whether DS has useful
change directions to read."

## SLIDE 18 — Extending to time: 1st/2nd-order DS + geodesic (Sensei's repeated request)
**On slide:** a curved line (subspaces over time); speed = 1st-order DS, bend = 2nd-order DS, split into smooth
drift vs abrupt. (`seq_al_wakrah.png`.)
**Screenshot:** **Fukui et al., "Second-order Difference Subspace," arXiv:2409.08563 (2024), Figure 1** (the
geodesic / second-order picture).
**Script:** "One more direction, because it's the thing my advisor asked about most often, and it uses his own 2024
paper. So far I compared two dates. But if you have *many* dates, each date becomes a subspace — a point on a
curved surface. How *fast* you move across that surface is the 'first-order' difference subspace — the speed of
change. How sharply the path *bends* is the 'second-order' — the acceleration — and you can split that into smooth,
gradual drift, like seasons, versus a sudden jump, like a disaster. This is the first time his second-order
construction is applied to satellite images. As he himself framed it, it's a 'first and unique trial' — a way to
*characterize* change over time. I show the curves on a real sequence; it's a promising direction, not a finished
detector, and I'm honest about that."

## SLIDE 19 — Does it matter for real disasters? (xBD) *(optional)*
**On slide:** xBD triage overlay (`fig4b_xbd_top5_overlay.png`): after / actual damage / top-5% flagged.
**Script:** "Quick reality check on real disasters. On a dataset called xBD-S12 — actual hurricanes with
human-marked building damage — the spatial-subspace map helps locate damaged buildings: if a human reviews only the
five percent most-suspicious pixels, they already find about a quarter of the damage, five times better than
random, with no labels. 'Triage', like in a hospital, just means deciding what to check first. I'll be honest that
my full successive method doesn't yet transfer cleanly to disasters — that's future work — so here I show the
simpler version. But it points at the real use."

## SLIDE 20 — Honest limits (this is a strength)
**On slide:** not SOTA / not beating supervised deep nets; it generalizes (train-fitted 0.338); vs the single best
baseline p=0.084 with only 10 cities (clear trend, significant vs raw/PCA); disaster transfer mixed; a scoped
adaptation of PixelHop.
**Script:** "Let me be precise about what I do *not* claim, because honesty is what makes the rest believable.
This is not state-of-the-art and it doesn't beat modern supervised deep networks — it's a label-free method, a
different goal. The good news is it generalizes — the train-fitted version still scores 0.338 on unseen cities. But
with only ten test cities, my win over the *single* strongest baseline is a clear trend, not yet formally
significant — though it *is* significant against raw differencing and PCA. And on disasters, this exact method
doesn't beat simple methods yet. It's a careful first study, not a finished product."

## SLIDE 21 — Contributions & future work
**On slide:** Contributions: the construction journey; successive Saab + DS beats classical detectors on held-out
OSCD, with DS adding beyond the features; honest negatives for pyramids/wavelets/kernel; the diagnostic + first
satellite trial of 2nd-order DS. Future: more cities → significance; disaster transfer; neural prior; temporal.
**Script:** "To summarize. My contributions: the construction journey, showing that *how* you build the subspace
decides whether geometry helps; a concrete method, successive Saab plus difference subspace, that beats the
classical unsupervised detectors on unseen cities, with the geometry adding beyond the features; clean, honest
negatives for the pyramid, wavelet, and kernel ideas; and the diagnostic, plus the first satellite trial of my
advisor's second-order subspace. For future work: more cities to reach formal significance, transferring to
disaster data, using the map to help a neural network with few labels, and the multi-date temporal analysis."

## SLIDE 22 — References + Thank you
**On slide:** the key papers (Fukui-Maki TPAMI 2015; Fukui 2024 2nd-order DS; Kuo PixelHop 2019; Kuo Green Learning
2022; Daudt OSCD/FC-EF 2018; Nielsen IR-MAD; Celik PCA; Bovolo-Bruzzone CVA; Mallat wavelets). "Thank you —
questions welcome."
**Script:** "These are the key references — Professor Fukui's difference-subspace papers, the Green-Learning and
PixelHop papers behind the Saab features, the classical change-detection baselines, and the dataset. Thank you very
much — I'm happy to take questions, and I'm glad to derive the difference subspace on the board if that helps."

---

## SCREENSHOT SHOPPING LIST (grab these figures for your slides)
| Concept | Paper | Which figure |
|---|---|---|
| Difference Subspace (canonical angles, two subspaces) | Fukui & Maki, *Difference Subspace and its Generalization*, IEEE TPAMI 2015 ([link](https://ieeexplore.ieee.org/document/7053916)) | the figure with two subspaces + their difference directions |
| 2nd-order DS / geodesic | Fukui et al., *Second-order Difference Subspace*, arXiv:2409.08563 (2024) | Figure 1 (the geodesic/velocity-acceleration picture) — your last-year deck already used this |
| Green Learning / PixelHop / successive hops | Kuo et al., *PixelHop*, arXiv:1909.08190 (2019) | **Figure 2** — the neighborhood → Saab → pooling successive-hop diagram |
| Saab transform (DC/AC) | Kuo et al., *Interpretable CNNs via Feedforward Design*, 2018 | the Saab DC/AC decomposition figure |
| Green Learning overview | Kuo & Madni, *Green learning: introduction, examples and outlook*, JVCIR 2022 ([DOI](https://doi.org/10.1016/j.jvcir.2022.103685)) | the system-overview figure |
| Sentinel-2 bands / OSCD example | Daudt et al., OSCD (IGARSS 2018) + page ([link](https://rcdaudt.github.io/oscd/)) | a before/after/GT city example (or a "13 bands" graphic) |
| Wavelet 2-D decomposition (LL/LH/HL/HH) | Mallat, *A Theory for Multiresolution Signal Decomposition*, IEEE TPAMI 1989 | the 2-D subband decomposition figure (to show the idea you tested) |

Your own figures (already made, in `phase1/outputs/seminar_figures/`): `fig_saab_journey`, `fig_saab_pipeline`,
`fig_saab_results`, `fig8_diagnostic_matrix`, `fig5_cca_family`, `fig6_otsu_ablation`, `fig4b_xbd_top5_overlay`;
`temporal_ds_sequence/seq_al_wakrah.png`; Codex's `…/multiresolution_subspace_2026-06-23/{frozen_test_method_comparison,
chongqing_qualitative_comparison,frozen_test_city_ap_deltas}.png`.

## ONE-LINE DEFINITIONS (keep on a card; say these if anyone looks lost)
- **Band-image:** one of the 13 channels viewed as a whole picture (e.g., "the infrared band-image").
- **Subspace:** the few main directions a set of vectors spreads along (from PCA) — a compact summary of its shape.
- **Difference Subspace:** the bundle of directions that changed between the before- and after-subspaces.
- **Canonical angles:** the angles between two subspaces; small = shared, large = changed.
- **Green Learning / PixelHop:** label-free, no-backprop image features built in stages ("hops").
- **Saab transform:** the building block — split each local patch into its average (DC) + PCA patterns (AC).
- **Successive Subspace Learning:** doing the Saab feature step in successive hops, fine → coarse.
- **AP / AUROC:** ranking-quality scores for rare targets; higher = better (change is rare, so not plain accuracy).
- **Unsupervised / label-free:** no human-labeled training examples; works straight from the two images.
- **Triage:** ranking pixels so a human checks the most-suspicious first.
