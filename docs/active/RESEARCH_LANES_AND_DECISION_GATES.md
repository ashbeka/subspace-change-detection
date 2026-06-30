# Research Lanes And Decision Gates

## Quick Links

- [1. Purpose](#purpose)
- [2. Lane Status Labels](#lane-status-labels)
- [3. Ranked Lane Table](#ranked-lane-table)
- [4. Lane Cards](#lane-cards)
- [5. Closed Or Paused Routes](#closed-or-paused-routes)
- [6. Cross-Lane Rules](#cross-lane-rules)

## Purpose

This file turns indecision into a controlled queue. A research lane is one
coherent bet:

```text
problem -> method family -> data -> baseline pressure -> win axis -> gate
```

Only one lane should be active for implementation at a time.

## Lane Status Labels

| Label | Meaning |
|---|---|
| active | run or verify now |
| candidate | promising, but not next |
| paused | evidence weak or missing data |
| closed | current implementation did not pass; reopen only with new mechanism |
| source-only | useful for explanation/literature, not current experiment |

## Ranked Lane Table

| Rank | Lane | Win axis | Current status | Next gate |
|---:|---|---|---|---|
| 1 | Successive Saab-DS for OSCD changed-area evidence | label-free spatial representation + interpretable DS evidence | active | reproduce DS-specific neural-prior claim; find external labeled multispectral test |
| 2 | DS-specific neural-prior fusion | complement supervised learning, label efficiency, or feature fusion | active verification | rerun raw/no-DS/DS/matched-cross controls across seeds |
| 3 | Band-Image/projector candidate localization on xBD-S12 | analyst triage and damaged-building candidate recall | candidate | test another event set or object-level retrieval protocol |
| 4 | Temporal first/second DS and geodesic decomposition | interpretable temporal characterization | candidate | obtain labeled sequence or clear weak-event protocol |
| 5 | Deep subspace / geometry analysis | analyze deep/foundation features geometrically | candidate | choose encoder, define feature object, compare to raw feature baselines |
| 6 | Foundation-model + subspace geometry | interpret/compress/cluster strong foundation features | candidate | define task: post-hoc explanation, prior generation, or low-label adaptation |
| 7 | HSI spectral geometry and wavelength attribution | spectral interpretability and material-change analysis | paused | get real labeled bitemporal HSI benchmark and pressure against HSI baselines |
| 8 | Structured temporal/CCA/SFA lane | invariant/background modeling and attributable temporal change | candidate | define one sequence task and compare SFA/CCA/SSA/DS controls |
| 9 | Tensor/Product-Grassmann satellite cubes | preserve spectral-spatial-temporal modes | future | choose tensor object and prove it beats flattening/control geometry |
| 10 | KDS/KGDS nonlinear satellite change | nonlinear geometry | paused | define specific nonlinear failure of DS/PCA first |
| 11 | Greenhouse/urban infrastructure application | applied object/state monitoring | paused | secure labels and define task: mapping, classification, or change |
| 12 | Diagnostic benchmark paper | honest boundary of subspace methods | fallback | consolidate positives/negatives into one evidence matrix |

## Lane Cards

### 1. Successive Saab-DS

Problem:

```text
Global pixel DS loses spatial structure; can local label-free feature maps make
DS useful for multispectral changed-area maps?
```

Method:

- pair-shared successive Saab transforms;
- local `3x3` neighborhood features;
- DC/AC local PCA-like responses;
- 2-hop scale hierarchy;
- Band-Image DS over response maps.

Evidence:

- OSCD test AP `0.3420`;
- beats raw/PCA/matched-feature L2/PCA controls;
- positive but still limited advantage over matched cross-reconstruction.

Gate:

- reproduce non-transductive/frozen behavior;
- find a second labeled multispectral dataset.

### 2. DS-Specific Neural-Prior Fusion

Problem:

```text
If DS is not the best standalone map, does it still add useful information as a
supervised prior beyond non-DS classical priors?
```

Evidence:

- Claude-side result claims DS-specific multi-prior U-Net gain.
- Not yet independently reproduced in main.

Gate:

- rerun exact comparison with seeds, logs, splits, and matched controls.
- no thesis claim until reproduction succeeds.

### 3. xBD-S12 Candidate Localization

Problem:

```text
Can spatial band-image geometry help rank candidate damaged regions for human
review, even if it is not a damage classifier?
```

Evidence:

- projector distance leads damaged-pixel retrieval at small review budgets;
- raw L2 is better inside known building support for damage-vs-intact;
- high damaged-building recall but weak specificity.

Gate:

- object-level protocol with fixed review budget;
- independent event or no further tuning on inspected events.

### 4. Temporal First/Second DS And Geodesic

Problem:

```text
Can a sequence of per-date satellite subspaces expose change velocity,
acceleration, and along/off-geodesic behavior?
```

Evidence:

- implemented and visualized on real sequences;
- useful for Sensei-aligned characterization;
- current detection/localization evidence is not strong.

Gate:

- labeled or independently annotated sequence;
- compare against BFAST/CCDC/MOSUM/SSA/raw residual controls.

### 5. Deep Subspace / Foundation-Feature Geometry

Problem:

```text
Can subspace geometry explain, compress, or compare representations from strong
deep or foundation models instead of competing with them directly?
```

Possible roles:

- subspaces of patch embeddings;
- DS/GDS on pre/post foundation features;
- clustering changed regions by feature-subspace direction;
- label-efficient priors from frozen features.

Gate:

- choose one encoder and one task;
- compare raw embedding distance, PCA, DS/GDS, and simple classifier controls.

### 6. HSI Spectral Geometry

Problem:

```text
Do many-band images make subspace methods more natural than Sentinel-2's 13
bands?
```

Current status:

- broad transfer was scene-dependent/negative;
- still valuable if a labeled bitemporal HSI dataset is available.

Gate:

- pressure against SAM, CVA, MAD/IR-MAD, unmixing, SMSL/HSI-CD baselines.

### 7. KDS/KGDS

Problem:

```text
Does nonlinear subspace geometry capture change missed by linear DS/PCA?
```

Current status:

- Venus learning/understanding exists;
- satellite application lacks a specific nonlinear hypothesis.

Gate:

- define failure case where linear scores fail and kernel scores should help.

### 8. Structured Temporal / CCA / SFA Lane

Problem:

```text
Can slow/invariant/background subspaces separate real structural change from
seasonal, phase, or radiometric nuisance?
```

Candidate tools:

- SFA / Slow Feature Subspace;
- Singular Spectrum Analysis;
- temporally regularized CCA / S3CCA / KCCA;
- RTW only if a new mechanism beats simple shift/RMS/PCA controls.

Gate:

- define one labeled or weakly labeled sequence task;
- compare against raw residual, global shift, PCA reconstruction, SSA, and
  simple seasonal summaries before claiming novelty.

### 9. Tensor / Product-Grassmann Lane

Problem:

```text
Satellite data are naturally multi-mode: bands, height, width, time, object.
Can tensor/product-Grassmann geometry preserve structure that vector flattening
destroys?
```

Candidate tools:

- n-mode GDS;
- product Grassmann manifold;
- signal latent subspace style deep-feature factors.

Gate:

- define the tensor modes and nulls;
- show a benefit over simpler per-mode PCA/DS or flattened controls.

### 10. Application Lane

Potential applications:

- abandoned greenhouse mapping;
- urban/infrastructure change;
- disaster candidate localization.

Gate:

- define labels, evaluation unit, and baseline before method work.

## Closed Or Paused Routes

| Route | Reason |
|---|---|
| global pixel DS as detector | weak against raw/PCA baselines and breaks spatial information |
| fixed-grid spatial pyramid DS | did not beat strong controls |
| wavelet Band-Image DS | did not pass positive gate |
| RTW as implemented | strong controls beat it on natural crop labels |
| generic HSI transfer | scene-dependent, not current positive route |
| SpaceNet7 RGB transfer | raw L2/cross controls stronger |
| satellite KDS/KGDS | no specific nonlinear question yet |

## Cross-Lane Rules

- A lane must define its win axis before experiments: accuracy,
  interpretability, label efficiency, candidate triage, temporal
  characterization, robustness, or computation.
- A lane must name its strongest unfair-looking baseline before it is trusted.
- AI-generated positive claims are `needs rerun` until reproduced in the active
  code path.
- Sensei-requested temporal/geodesic work remains important even when it is not
  the current best detector.
