# Remote-Sensing Change-Detection Taxonomy — improved framework

An expansion/correction of a flat method tree. The base tree (6 families: direct-comparison, post-analysis,
joint-multitemporal, time-series, hybrid, cross-cutting descriptors) is good and comprehensive. The four
structural improvements below make it a tool for *locating novelty* rather than just cataloguing.

## Improvement 1 — Factorize "direct comparison" as REPRESENTATION × COMPARISON-OPERATOR × DECISION
A flat list hides that almost every pre-classification CD method is a *composition* of three orthogonal choices.
Naming the axes exposes the combinatorial space — and the empty cells are where novelty lives.

| REPRESENTATION (what you compare) | COMPARISON OPERATOR (how) | DECISION (binarize) |
|---|---|---|
| raw spectrum/band | difference / ratio / log-ratio | threshold (manual) |
| spectral index (NDVI/NBR/...) | Euclidean / Mahalanobis | Otsu / adaptive |
| transform (PCA/MNF/ICA/Tasseled-Cap) | **angle (SAM) / SID / SCM** | k-means / FCM / GMM |
| correlation transform (CCA/MAD/IR-MAD) | **canonical angle / Difference-Subspace (DS/GDS/KDS)** | graph-cut |
| spatial/texture/morphological feature | projection / Grassmann distance | MRF / CRF |
| **subspace** (PCA / patch / image-set / signal) | learned metric (contrastive) | — |
| **deep / foundation feature** | chi-square / likelihood-ratio | — |

Examples decode cleanly: **SAM** = (raw spectrum) × (angle) × (threshold). **CVA** = (raw spectrum) ×
(Euclidean+direction). **IR-MAD** = (CCA/MAD transform) × (chi-square) × (iterative reweight). **Fukui DS** =
(subspace) × (canonical-angle/DS). **SLS** = (deep feature → subspace) × (canonical angle). Novelty = a *new
cell combination* (e.g. (foundation feature → subspace) × (2nd-order DS)).

## Improvement 2 — INVARIANCE / nuisance-handling is a first-class cross-cutting axis (add to §6)
This single axis predicts real-world CD performance (pseudo-change robustness) better than any other, and is
absent from the base tree. It is the axis on which methods actually beat each other:

| invariance level | what it ignores | example methods |
|---|---|---|
| none | nothing (raw magnitude) | image differencing, CVA |
| amplitude / scale | brightness, illumination | **SAM**, cosine, spectral indices (ratios) |
| affine / decorrelated | linear radiometric transforms | **MAD, IR-MAD**, Mahalanobis |
| slow / temporally-invariant | smooth seasonal/illumination drift | **SFA, ISFA, KSFA, DSFA** |
| learned / nuisance-modeled | whatever a no-change set defines | **GDS-common projection, deep no-change models, anomalous-change-detection** |

**Key law (grounded):** a method beats SAM/CVA on real data chiefly by climbing this ladder — *more invariance*,
not *more angles*. Plain shape-DS sits at "amplitude" (same rung as SAM) and so cannot beat SAM on its home turf.

## Improvement 3 — Re-home "subspace geometry": a MECHANISM, not a sibling family
The base tree lists "1.4 Subspace-geometric comparison" as a sibling of image-algebra/transforms/deep-features.
That is a level error: subspace geometry is a (representation, operator) *pair* that recurs **across** families —
direct (1.4), learned-feature (1.5 "compare via subspace geometry"), temporal (4.6 subspace dynamics), hybrid
(5.2 deep features + subspace). It belongs in the §6 cross-cutting axes (as a representation choice + a comparison
operator), with pointers from each family. Treating it as a standalone silo is what drives method-forcing.

## Improvement 4 — Name the unifying STRATEGY: "model the invariant; the residual is change"
A cross-cutting *strategy* (orthogonal to representation) unifies the strongest unsupervised CD methods and is
not named in the base tree:
```
build the COMMON / INVARIANT / NO-CHANGE component (covariance: MAD/IR-MAD; slow: SFA/ISFA;
   common subspace: GDS projection; predicted background: anomalous-change-detection / chronochrome)
   -> change = the energy/residual OUTSIDE it.
```
This is the principle behind every method that robustly beats SAM/CVA, and the defensible niche for subspace
geometry (GDS-common-projection as the invariant model). **Add an Anomalous-Change-Detection node** (chronochrome,
hyperbolic ACD, covariance-equalization) under direct/hybrid — the canonical nuisance-invariant family, missing
from the base tree (benchmark: Viareggio 2013).

## Where our three live hypotheses sit in this map
- **H-A invariance-residual CD** = climb the invariance ladder via the GDS-common / SFA-slow strategy
  (Improvement 4) → §1.2 (SFA/MAD) ∪ §1.4 (GDS), unsupervised.
- **H-B trajectory characterization** = §4.6 subspace-dynamics (1st/2nd-order DS, velocity/acceleration).
- **H-C geometry on deep features** = §1.5/§5.2 ((deep feature → subspace) × DS).

See `docs/CONTINUITY.md` §4 for the hypotheses and `docs/RESEARCH_DIRECTIONS_TOP10.md` for the full ranked set.
