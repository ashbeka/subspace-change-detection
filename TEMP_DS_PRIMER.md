# TEMP: Difference Subspace (DS) Primer (for seminar)

Goal of this document:
- Explain subspaces and subspace methods in plain terms.
- Explain Difference Subspace (DS) as used in the Fukui/Maki papers.
- Explain how DS can be used for change detection (not only classification).
- Point to the exact places DS is implemented and used in this repo.

This is based on:
- the DS papers referenced in this repo (Fukui & Maki 2015; Fukui et al. 2024),
- the included reference code under `references/`,
- and the implementation under `phase1/` and `phase2/`.

---

## 0) The 30-second explanation (what to say out loud)

"Each Sentinel-2 pixel is a vector of 13 spectral bands. For a given city tile, the set of pre-event pixel vectors lies near a low-dimensional subspace, and the post-event pixels lie near another subspace. Difference Subspace (DS) is a geometric way to extract the directions where those two subspaces differ. Then, for each pixel we score change by projecting its spectral difference (post - pre) onto the DS directions. That gives an unsupervised change map. In Phase 2 we optionally feed DS and PCA-diff maps as extra channels (priors) into a supervised segmentation model, to test whether these interpretable priors improve OSCD change segmentation."

---

## 1) Subspaces: the basic concept

### 1.1 Linear subspace (math)
A linear subspace S of R^d is any set of vectors closed under:
- addition: if u,v in S then u+v in S
- scaling: if u in S and a is a scalar then a*u in S

Any r-dimensional subspace can be represented by an orthonormal basis matrix:
- B in R^(d x r), columns orthonormal (B^T B = I)
- S = span(B) = { B a : a in R^r }

### 1.2 Subspace as a data model (why PCA is used)
Real data often lies near a low-dimensional structure:
- features/bands are correlated
- there are a few dominant modes of variation

PCA learns a subspace that explains most variance:
- data matrix X in R^(d x n), each column is a sample vector in R^d
- PCA returns principal directions; the top r define a subspace basis B

Remote sensing mapping to this idea:
- a pixel in a multispectral image is a vector (bands are features)
- Sentinel-2 has 13 bands -> d = 13
- an image tile provides many samples (one sample per valid pixel)

---

## 2) Subspace methods (why they show up in vision papers)

Subspace methods are a family of techniques where you:
1) represent a class/condition by a subspace learned from examples, then
2) compare subspaces (or compare a sample to a subspace).

Examples:
- Eigenfaces: images of a person lie near a subspace.
- Mutual Subspace Method (MSM): each class has a subspace; classify an input subspace by subspace similarity (principal angles).
- Anomaly/change: "normal" condition has a subspace; large residual from that subspace indicates something changed.

Key point:
Subspace geometry is not limited to classification. It is a general tool to model distributions and measure differences.

---

## 3) Tools for comparing subspaces: projectors and principal angles

### 3.1 Projector and residual projector
For an orthonormal basis B:
- projector onto span(B): P = B B^T
- residual projector (orthogonal complement): R = I - P

For a vector x:
- Px is the part explained by the subspace
- Rx is the unexplained part
- ||Rx||^2 is a natural distance-to-subspace score

In this repo:
- `phase1/ds/pca_utils.residual_projector` computes R = I - B B^T

### 3.2 Principal (canonical) angles
For two subspaces with orthonormal bases Phi and Psi:
  Phi^T Psi = U * Sigma * V^T   (SVD)

The singular values satisfy:
  Sigma_i = cos(theta_i)
where theta_i are the principal angles between subspaces.

Interpretation:
- theta near 0 degrees: shared / very similar directions
- theta near 90 degrees: orthogonal / very different directions

Principal angles are the core object behind many DS derivations.

---

## 4) Difference Subspace (DS): what it is and how it is built

### 4.1 Intuition
Given two subspaces S1 and S2, DS aims to capture the directions that represent "how they differ",
separating:
- common/shared variation (things both subspaces contain),
from
- difference-specific variation (things present in one but not the other).

Why DS appears in classification papers:
- if each class is represented by a subspace, DS provides discriminative directions between class subspaces.

### 4.2 DS via canonical angles (paper-faithful construction)
This repo contains a canonical-angle DS implementation:
- `phase1/subspace/second_order_ds.py` -> `difference_subspace_canonical`

Note: despite the filename, `difference_subspace_canonical` is the *first-order* (two-subspace) DS.
It is also used as a building block for the optional *second-order* DS utilities in the same module.

It does:
1) SVD: U, s, V^T = SVD(Phi^T Psi)
2) Exclude near-shared directions where s ~ 1 (cos(theta) ~ 1)
3) Form DS basis from a normalized difference of rotated bases:
   D = (Phi U - Psi V) * ( 1 / sqrt(2(1 - s)) )
4) Orthonormalize D using QR

You do not need to memorize the formula. The key story is:
"DS uses principal angles to construct basis vectors that exist in one subspace but not the other."

### 4.3 DS via residual stacking (simple, robust construction used in Phase 1)
Phase 1 DS scoring in this repo uses a simpler construction by default:
- `phase1/ds/pca_utils.difference_subspace`

Given Phi and Psi (same ambient dimension d):
  R_phi = I - Phi Phi^T
  R_psi = I - Psi Psi^T
  D = orth( [ R_psi Phi , R_phi Psi ] )

Interpretation:
- R_psi Phi: "part of Phi not explained by Psi"
- R_phi Psi: "part of Psi not explained by Phi"
- stacking + orthonormalization yields a basis spanning difference directions

This is consistent with the DS idea: isolate what is in one subspace but not the other.

### 4.4 Turning DS into a scalar score
DS gives you a subspace basis D. To get a change score, you project a vector onto D:
  score(x) = || D^T x ||^2

In change detection we apply it to the per-pixel spectral difference:
  delta = x_post - x_pre
  score(pixel) = || D^T (x_post - x_pre) ||^2

In this repo:
- `phase1/ds/ds_scores.py` -> `_compute_ds_matrix_scores`
  - diff = x2_mat - x1_mat
  - projection_energy = sum( (D^T diff)^2 )

### 4.5 DS "cross residual" score (also used here)
This repo also computes:
  cross_residual = ||R_psi x_post||^2 + ||R_phi x_pre||^2

Meaning:
- post looks unusual under the pre subspace, and/or
- pre looks unusual under the post subspace

In this repo:
- `phase1/ds/ds_scores.py` -> `_compute_ds_matrix_scores`

---

## 5) "DS was used for classification" -> why change detection is a valid use

Your friend is right: DS is often introduced in classification settings.
But DS is not inherently a classification algorithm. It is a geometric construction between two subspaces.

### 5.1 The key reinterpretation for change detection
For change detection, we treat:
- pre-event pixels as samples from condition A
- post-event pixels as samples from condition B

Then:
1) Learn Phi = PCA basis of pre pixels
2) Learn Psi = PCA basis of post pixels
3) Compute D = DS(Phi, Psi)
4) Score each pixel by projecting its delta = (post - pre) onto D

So we apply the same DS geometry, but the downstream task changes:
- classification: DS helps discriminate class subspaces
- change detection: DS quantifies how two time conditions differ, per pixel

### 5.2 Why this is especially sensible for multispectral satellite data
Simple per-band L2 differences are sensitive to:
- illumination / atmospheric changes
- seasonal vegetation cycles
- radiometric shifts

DS learns a structured "difference direction set" between the two conditions and then measures how much a pixel's change lies in those directions.
That is why DS maps can be interpretable: "where does the pre-vs-post subspace geometry disagree?"

---

## 6) A concrete toy example (easy mental model)

Ambient space is R^3 (three "bands").

Imagine a region where:
- before event: pixel spectra vary mostly along directions e1 and e2
- after event: pixel spectra vary mostly along directions e1 and e3

So:
- Phi spans {e1, e2}
- Psi spans {e1, e3}
- common direction is e1
- difference directions are e2 (pre-only) and e3 (post-only)

Now a pixel changes:
  x_pre  = [1, 2, 0]^T
  x_post = [1, 0, 2]^T
  delta  = x_post - x_pre = [0, -2, 2]^T

If DS basis D spans {e2, e3}, then D^T delta is large, so ||D^T delta||^2 is large -> "change".

If a pixel changes only along the shared direction e1:
  delta = [0.1, 0, 0]
then projection onto {e2,e3} is near 0 -> "not a DS-type difference change".

This is the sentence that maps to the toy example:
"DS scores how much the change vector lies in directions that separate pre and post distributions."

---

## 7) How DS is used in THIS project (exact pipeline)

### 7.1 Phase 1: DS as unsupervised change detection
Entry point:
- `phase1/eval/run_oscd_eval.py`

Per city/tile:
1) Load pre/post cubes (C,H,W) from OSCD:
   - `phase1/data/oscd_dataset.py` -> `OSCDEvaluatorDataset`
2) Build valid mask using both pre and post:
   - `phase1/data/oscd_dataset.py` builds valid_pre & valid_post, then ANDs them
3) Normalize each band (z-score) to avoid domination by high-range bands:
   - `phase1/data/preprocessing.apply_normalization`
   - stats path is configurable (default config uses `data/oscd_band_stats.json`, resolved relative to `phase1/` -> `phase1/data/oscd_band_stats.json`; created automatically if missing)
4) Vectorize valid pixels into matrices:
   - `phase1/data/preprocessing.vectorize_cube`
   - X1 in R^(C x N), X2 in R^(C x N)
5) Fit PCA bases:
   - `phase1/ds/pca_utils.fit_pca_basis` -> Phi, Psi
6) Compute DS basis:
   - default: residual-stacked DS (`phase1/ds/pca_utils.difference_subspace`)
   - optional: eig DS (`phase1/ds/pca_utils.difference_subspace_eig`)
7) Score every pixel:
   - projection energy: ||D^T (x2 - x1)||^2
   - cross residual: ||R_psi x2||^2 + ||R_phi x1||^2
   - `phase1/ds/ds_scores.compute_ds_scores`
8) Optionally use local/sliding-window DS:
   - `phase1/ds/ds_scores.sliding_window_ds`
9) Evaluate vs OSCD GT:
   - AUROC for score maps, IoU/F1 after thresholding
10) Save score maps for Phase 2 priors:
   - `phase1/eval/run_oscd_eval.py --save_change_maps`
   - output structure:
     `phase1/outputs/<run_name>/oscd_change_maps/<split>/<method>/<city>_score.npy`

Config controlling this in Phase 1:
- `phase1/configs/oscd_priors_fast.yaml`
  - `ds.rank_r`, `ds.subspace_variant`, `ds.window`, etc.

### 7.2 Phase 2: DS maps as priors for supervised segmentation
Entry point:
- `phase2/train/train_oscd_seg.py`

Dataset loader:
- `phase2/data/oscd_seg_dataset.py` -> `OSCDSegmentationDataset`

What happens:
1) Base input features:
   - raw Sentinel-2 pre+post bands stacked (26 channels)
2) Optional prior channels (loaded from Phase 1 outputs):
   - DS projection map, DS cross residual map, PCA-diff map, etc.
   - `_load_priors()` loads `.../<split>/<method>/<city>_score.npy` and rescales to [0,1]
3) Train segmentation model:
   - U-Net, ResNet-U-Net, or PriorsFusionUNet
   - configs under `phase2/configs/`

Important: DS is not used as a label in Phase 2.
It is an extra feature channel.

---

## 8) What DS is NOT (avoid overclaiming)

- DS by itself is not "damage detection". It detects spectral change.
  Spectral change can be vegetation, illumination, water, construction, etc.
- OSCD labels only certain changes, so DS may highlight real changes that are not labeled.
  This is why DS masks can look dense and why AUROC and F1 can behave differently.

This is a good discussion point for the seminar:
"DS shows where there is spectral change; supervised segmentation learns what the dataset calls change."

---

## 9) Common questions you might get (quick answers)

Q: Why PCA? Why not directly learn DS from raw pixels?
A: PCA gives a stable low-dimensional subspace estimate from many pixels and reduces noise; DS is defined on subspaces, so we need bases.

Q: Why DS instead of just using post-pre difference?
A: DS uses the geometry of the two learned subspaces to focus on directions that differ structurally, not all per-band differences equally.

Q: Why can priors hurt F1 in Phase 2?
A: Priors highlight many spectral changes, but OSCD labels only certain changes. Adding priors can push the network toward unlabeled changes, raising AUROC but not necessarily improving thresholded mask quality (F1).

---

## 10) References (PDFs included in this repo)

Core DS papers:
- `references/reference_papers/Fukui and Maki - 2015 - Difference Subspace and Its Generalization for Subspace-Based Methods.pdf`
- `references/reference_papers/Fukui et al. - 2024 - Second-order difference subspace.pdf`

Helpful "DS beyond classification" example (good for answering your friend's point):
- `references/reference_papers/Kanai et al. - 2023 - Time-series Anomaly Detection based on Difference Subspace between Signal Subspaces.pdf`

Background on subspace methods and baselines:
- `references/reference_papers/Subspace Methods.pdf`
- `references/reference_papers/Unsupervised Change Detection in Satellite Images Using Principal Component Analysis and k-Means Clustering.pdf`
