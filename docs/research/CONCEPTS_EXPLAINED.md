# Concepts explained — plain language + whiteboard scripts (for the seminar)

Every hard concept, explained three ways: (1) a one-line **intuition**, (2) the **math** (what to write), (3) a
**board script** (what to draw if asked). Read this until you can teach each one cold.

---

## 1. What is a subspace? (the foundation)
**Intuition.** A subspace is the set of all directions you can build by mixing a few "basis" directions. A flat
sheet of paper inside a room is a 2-D subspace of 3-D space — every point on it is a mix of two directions.
"Dimension" = how many independent directions you need = number of basis vectors (we call them principal
components, PCs).

**Math.** Take many data vectors, run PCA, keep the top $r$ directions $u_1,\dots,u_r$. Their span is an
$r$-dimensional subspace; the matrix $U=[u_1\,\cdots\,u_r]$ is its (orthonormal) basis.

**Board.** Draw 3-D axes, scatter a cloud of dots that mostly lie near a tilted plane, draw the plane, draw 2
arrows on it ($u_1,u_2$). Say: "the cloud is 3-D but really lives on this 2-D sheet — that sheet is the
subspace; its dimension is 2."

---

## 2. Two ways to build the subspace (Sensei's #1 question — know this cold)
We have one image per date, with $B$ bands (colors) and $N=H\times W$ pixels. There are two ways to turn it into
"a set of vectors" for PCA:

**(a) Per-pixel (the naive way).** Each *pixel* is one vector of its $B$ band-values. PCA over all $N$ pixels →
a subspace in $\mathbb{R}^B$ (e.g. 13-D color space). **Problem:** pixels are treated as an unordered bag — the
*position* of each pixel is thrown away. So the subspace knows the scene's *colors* but not its *layout*. This is
the "breaks the spatial information" issue Sensei flagged.

**(b) Band-image (our spatially-faithful way; senpai Jang's "flatten on the Z-axis").** Each *whole band-image*
is one vector of length $N$ (the band laid out flat). PCA over the $B$ band-images → a subspace in
$\mathbb{R}^N$ (spatial space). Now **each channel is a vector, the date is one subspace**, and the *spatial
layout is preserved* (position is the axis itself). Rank ≤ $B-1 = 12$.

**Board.** Draw a small cube labeled "$B$ bands, $H\times W$". Arrow A: "slice into $N$ pixel-vectors of length
$B$" → loses position. Arrow B: "slice into $B$ band-vectors of length $N$" → keeps position. Circle B: "this is
ours."

**One-liners for Q&A:** "Dimension = number of PCs = 12." "Datapoints × features before PCA: band-image = 13 ×
$N$; per-pixel = $N$ × 13." "Each channel is a vector; the date is one subspace."

---

## 3. Canonical (principal) angles — how two subspaces differ
**Intuition.** Two tilted sheets of paper meeting in a room: how different are they? Not one angle but a *set* of
angles — one per direction. If a direction lies in both sheets, its angle is 0 (shared). If a direction is in one
but not the other, its angle is large (a genuine difference). These are the **canonical/principal angles**
$\theta_i$, and $\cos\theta_i$ tells you how "shared" each direction is.

**Math.** For subspaces with bases $\Phi,\Psi$: $\Phi^\top\Psi = U\Sigma V^\top$ (an SVD).
$\Sigma=\mathrm{diag}(\cos\theta_1,\dots)$ = the canonical correlations. $\cos\theta_i\approx 1$ → shared;
$\cos\theta_i$ small → a difference direction.

**Board.** Draw two overlapping tilted planes; mark their line of intersection ($\theta=0$, shared) and a
direction sticking out of one ($\theta$ large, different).

---

## 4. The Difference Subspace (DS) — the "directions of change"
**Intuition.** Collect exactly the directions where the two date-subspaces *disagree* (the large-angle ones), and
package them into their own little subspace. That package is the Difference Subspace — it literally spans "what
changed in the scene's structure."

**Math.** $D = (\Phi U - \Psi V)\,[\,2(I-\Sigma)\,]^{-1/2}$. Total magnitude $=\sum_i 2(1-\cos\theta_i)$ (small if
the subspaces are nearly identical, large if very different). First-order canonical DS, Fukui–Maki TPAMI 2015.

**Board.** From the two planes, draw the 2 "difference directions" (the ones sticking out), bundle them with a
brace labeled "$D$ = difference subspace."

**DS vs GDS:** DS = between *two* subspaces (our bitemporal case). GDS = generalized to *many* subspaces
(multi-class). The multi-*date* version is the 2nd-order/geodesic DS (concept 9).

---

## 5. Projection energy / the change score — how we turn $D$ into a map
**Intuition.** For each pixel, take its before→after change vector, and ask: *how much of that change points along
the "directions of change" $D$?* That amount is the pixel's score. High = the change matches the scene-level
structural change = likely real change.

**Math.** Change at pixel $i$, band $b$: $\delta_b = X_{post,b}-X_{pre,b}$. Project onto $D$ with $P_D=DD^\top$.
Score $s_i = \sum_b \|P_D\,\delta_b\|^2$ (the "projection energy"; its square root is the "magnitude"). Reshape
$\{s_i\}$ to the image → the change map. (In the band-image construction the axis *is* the image grid, so no
separate "project back to image" step — that answers Sensei's projection question.)

**Board.** Draw a vector $\delta$ and the line $D$; drop a perpendicular from $\delta$ onto $D$; label the shadow
"$P_D\delta$" and say "its length² is the score."

---

## 6. Projector distance — the xBD headline geometry
**Intuition.** Instead of "how much did the change align with $D$", ask "how much did the whole subspace *rotate*
between before and after?" — measured separately at each pixel. Where the scene's spatial structure rotated most
(e.g. a building footprint changed) gets the highest score. On disaster data this localizes damaged structures
best.

**Math.** $P_\Phi=\Phi\Phi^\top$ (projector onto the before-subspace). Per-pixel score
$d_i = \|(P_\Phi-P_\Psi)\|_{\text{row }i}$ — the row-norm of the difference of the two projectors.

**Board.** Two planes again; show one rotating to the other; "how far each point moves under that rotation = the
score."

---

## 7. The diagnostic — why geometry usually ties the simple statistic (your backbone)
**Intuition.** A change map is REPRESENTATION × OPERATOR × DECISION. For most change types, the fancy geometric
operator collapses to a simple one:
- change is **brightness/amplitude** (fire, flood) → magnitude/CVA already captures it.
- change is **low-dimensional spectral** → the DS *equals* the spectral angle (SAM).
- change is **spread across all bands** → a covariance/correlation statistic (IR-MAD) captures it.
- change is **a type** (vegetation→building) → the change *direction* (polar-CVA) clusters it.
Geometry only adds something when the data object is *natively curved* (PolSAR covariance matrices, EEG) — which
optical reflectance is not. So our honest finding: **geometry is redundant for raw optical detection** — and that
*map of when/why* is the contribution, not a failure.

**Board.** A 3-column table: change-type | what wins | why. Fill 3 rows live.

---

## 8. Fusion complementarity — where DS *does* add value
**Intuition.** A weak teammate can still help a team if they're good at something the others aren't. Alone, the DS
map loses; but mixed (rank-averaged) with smoothed-PCA and IR-MAD, the combination beats the same mix where DS is
replaced by a look-alike control. So DS contributes a *distinct* piece of evidence — only visible in concert.

**Math.** rank-fuse(sPCA, DS, IR-MAD) > rank-fuse(sPCA, matched-null, IR-MAD); on OSCD, +0.0115 AP, p=0.0098.

**Board.** Venn-ish: three overlapping circles (sPCA, IR-MAD, DS); shade the small DS-only sliver = "the unique
evidence."

---

## 9. Temporal 1st/2nd DS + geodesic — Sensei's most-requested (and his own paper)
**Intuition.** With a *sequence* of dates, each date is a subspace = a point on a curved surface (the
Grassmannian). Move along it: the *speed* of movement = first-order DS ("change velocity"); the *bending* of the
path = second-order DS ("change acceleration"). Split the bend into "along the smooth path" (seasonal drift) vs
"off the path" (an abrupt event). A single scalar number can't express this — you need ≥3 dates and the geometry.

**Math.** First-order: $\theta(t)$ between consecutive subspaces. Second-order/geodesic: along- vs
orthogonal-geodesic components (Fukui et al. 2024, arXiv:2409.08563 — we apply it to satellite for the first time).

**Board.** A curved line (the path of subspaces over time); a dot moving on it; an arrow = velocity; a curving
arrow = acceleration; mark a kink = "abrupt event."

**The honest framing (Sensei gave it to you):** *"the first and unique trial in this research topic, although it
may not achieve top performance."* Present it as characterization, not a detector.

---

## 10. The 30-second "why this matters" you can always fall back on
"After a disaster you have before/after satellite images but no labels and no time. Naive differencing fails
because everything changes between two days. We make the lab's subspace idea spatially faithful, then we
*rigorously map* where that geometry helps: it's redundant for raw detection (simple statistics win — that's the
diagnostic), but it adds unique evidence in fusion and it transfers to real disaster damage as a label-free
candidate-localization prior that beats classical change maps and finds a quarter of the damage in the top 5% of
flagged pixels. A novel, honest first trial — exactly what Sensei asked for."
