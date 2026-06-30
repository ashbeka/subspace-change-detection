# Advisor Feedback And Decisions

## Quick Links

- [1. Purpose](#purpose)
- [2. Sensei Feedback: Core Questions](#sensei-feedback-core-questions)
- [3. Sensei-Requested Temporal Direction](#sensei-requested-temporal-direction)
- [4. Senpai Feedback: Spatial And Green Learning](#senpai-feedback-spatial-and-green-learning)
- [5. Seminar Feedback](#seminar-feedback)
- [6. Sensei/Seminar Coverage Table](#senseiseminar-coverage-table)
- [7. Detailed Human Feedback Preserved From Old Notes](#detailed-human-feedback-preserved-from-old-notes)
- [8. Current Decisions](#current-decisions)

## Purpose

This file answers: **what did Sensei/senpai/seminar feedback require, and what
did we do about it?**

Advisor and human-written notes have higher preservation priority than
AI-generated synthesis.

## Sensei Feedback: Core Questions

Sensei's most important question:

```text
What is the purpose of applying DS to this dataset?
```

Related warning:

```text
The algorithm can make a subspace, but it can break spatial information of the
input image.
```

Decision:

- global pixel DS is no longer the main claim;
- spatially faithful constructions must be tested;
- OSCD is a benchmark, not the whole thesis.

## Sensei-Requested Temporal Direction

Sensei asked about:

- generating a set of time-sequential satellite subspaces;
- frame count and time step;
- first DS magnitude;
- second DS magnitude;
- geodesic projection/decomposition;
- checking with Jang/Aono and related lab code.

Status:

| Task | Status |
|---|---|
| talk with Jang/Aono | done as orientation, not final validation |
| build time-sequential satellite subspaces | first trials done on MultiSenGE/IPOL |
| compute first/second DS and geodesic quantities | implemented and visualized |
| validate as detector | not yet strong; needs labeled sequence |
| Harmonized Sentinel-2 audit | still needed if temporal lane resumes |

Decision:

- temporal DS is a real Sensei-aligned lane;
- present as characterization unless stronger labels/evidence arrive.

## Senpai Feedback: Spatial And Green Learning

Key ideas:

- do not only use one pixel as a 13-D vector;
- try band-image flattening: one band image as one high-dimensional vector;
- try multiscale/local structures inspired by wavelets, image compression,
  Green Learning, PixelHop, and Saab transforms.

Actions taken:

| Idea | Result |
|---|---|
| Band-Image DS | useful boundary/candidate-localization result |
| fixed grid pyramid | negative as primary detector |
| wavelet Band-Image DS | negative as primary detector |
| Successive Saab-DS | strongest OSCD internal result |

Decision:

- the successful part is not literal wavelet/pyramid decomposition;
- the successful part is local label-free successive features before DS.

## Seminar Feedback

Recurring issue:

```text
What axis does the method win on?
```

Current answer:

- not SOTA segmentation accuracy;
- possible win axes are interpretability, label-free local evidence, candidate
  triage, complementarity as a prior, and temporal characterization.

Decision:

- every lane must define its win axis and baseline pressure before new runs.

## Sensei/Seminar Coverage Table

| Ask / concern | Current answer | Status |
|---|---|---|
| What is the subspace? | now documented by construction cards; global pixel, Band-Image, Saab-DS, and temporal date subspaces are different objects | covered, keep explaining |
| Does the current method break spatial information? | yes for global pixel DS; spatial/sample-construction experiments were run | covered |
| Try Venus/nonlinear DS/KDS/KGDS | Venus understanding exists; satellite KDS/KGDS still lacks specific nonlinear hypothesis | partially covered |
| Try first/second DS and geodesic decomposition | implemented on sequences; characterization works, detector evidence weak | covered as characterization |
| Clarify purpose/problem | current purpose is spatially faithful/interpretable multispectral change evidence, not generic damage segmentation | covered, still evolving |
| Compare against simple methods | raw L2, PCA-diff, Celik, IR-MAD, matched controls, and U-Net pressure are tracked | covered |
| Explain Otsu/thresholding/normalization | belongs in method/experiment reports; no method claim should depend on one threshold | covered enough for now |
| Band selection / combinations | preserved as future attribution/HSI lane, not current core | paused |
| CCA/S3CCA/TRCCA | preserved as structured temporal/view-matching lane | future |
| Pseudo-change vs real change | central failure-analysis concern; seasonal/radiometric failures must be reported | active boundary |

## Detailed Human Feedback Preserved From Old Notes

This section keeps the durable human feedback once, so old feedback files do
not need to remain active.

### Jang / Aono Consultations

| Person | What was asked / discussed | Current status |
|---|---|---|
| Jang | multi-channel data, EEG-style/sample-definition analogies, and whether channel/depth-axis construction makes sense | orientation only; not final validation |
| Aono | first/second DS, geodesic decomposition, and lab implementation details | orientation only; not final validation |

### First Seminar Feedback Actions

| Feedback / question | Durable action |
|---|---|
| Explain why ResNet34/backbone choices were used | keep model choices as baselines, not thesis novelty |
| Raw input sometimes beats prior channels | report raw baselines honestly; do not assume priors help |
| PriorsFusionUNet weakness | do not claim fusion until rerun with matched no-DS controls |
| IR-MAD comparison | audit formula and include as classical pressure before DS claims |
| Runtime/resource cost | report if claiming efficiency or label-free practicality |
| Metrics confusion | explain AUROC, AP/PR-AUC, F1/IoU, Otsu threshold separately |
| Related work vs proposed method boundary | separate DS/PixelHop/source methods from project adaptation |
| Alternative prior integration | test prior channels only after standalone prior evidence is clear |

### Submitted QA Report Takeaways

| Topic | Preserved takeaway |
|---|---|
| OSCD labels | OSCD binary labels do not equal semantic/damage truth; spectral/radiometric change can disagree with labels |
| ranking vs thresholding | AUROC/AP and thresholded F1/IoU answer different questions |
| MultiSenGE earliest/latest | can be dominated by seasonality; use with temporal protocol, not as direct OSCD replacement |
| snow/seasonality | robustness/failure-analysis cases, not trivial noise to hide |
| IR-MAD weakness | cannot be claimed unless the implementation and tuning are fair |

### Expanded Sensei Ask Coverage

| Sensei/lab ask | Current handling |
|---|---|
| Kanai SSA / learned normal-reference `D_N` | not implemented as the Kanai detector; temporal DS work is related but not equivalent |
| SFA / Slow Feature Subspace | preserved as structured temporal lane; needs explicit baseline protocol |
| RTW | tested and currently negative; reopen only with phase/shape gate and simple controls |
| CMSM/MSM / local MSM | useful reference family; not current detector claim |
| KGDS/Venus | Venus construction reproduced conceptually; satellite KDS/KGDS needs nonlinear hypothesis |
| VLM/foundation features as subspace | future-only; do not claim without encoder/object definition |
| CCA/S3CCA/TRCCA | preserved for structured temporal/view attribution, not yet detector |
| Google Harmonized Sentinel-2 / time-sequential data | future temporal data source if temporal lane resumes |

## Current Decisions

| Decision | Reason |
|---|---|
| prioritize reproducing DS-specific neural-prior result | most consequential unverified positive claim |
| keep Successive Saab-DS as current internal OSCD anchor | strongest verified OSCD result |
| keep xBD projector as candidate-localization boundary | external evidence exists but not damage classification |
| keep temporal DS/geodesic as Sensei-aligned characterization | implemented but detector evidence weak |
| pause broad method search | too many weak routes already tested |
| collapse AI-generated docs after absorption | reduce noise and improve reading path |
