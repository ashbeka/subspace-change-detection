# Human Notes And Feedback

## Quick Links

- [1. Purpose](#1-purpose)
- [2. Sensei Feedback: Core Questions](#2-sensei-feedback-core-questions)
- [3. Sensei-Requested Temporal Direction](#3-sensei-requested-temporal-direction)
- [4. Senpai Feedback: Spatial And Green Learning](#4-senpai-feedback-spatial-and-green-learning)
- [5. Seminar Feedback](#5-seminar-feedback)
- [6. Sensei/Seminar Coverage Table](#6-senseiseminar-coverage-table)
- [7. Detailed Human Feedback Preserved From Old Notes](#7-detailed-human-feedback-preserved-from-old-notes)
- [8. Current Decisions](#8-current-decisions)

## 1. Purpose

This file answers: **what did Sensei/senpai/seminar feedback require, and what
did we do about it?**

Advisor and human-written notes have higher preservation priority than
AI-generated synthesis.

## 2. Sensei Feedback: Core Questions

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

Other durable Sensei guidance from Slack:

| Guidance | Project consequence |
|---|---|
| Read change-detection survey papers to understand the field and math. | Do not justify the thesis only from lab methods; literature gaps must shape the problem. |
| The field is a red ocean; learn deeper math and develop an original technique or algorithm. | The work needs a clear mathematical/original point, not just an application wrapper. |
| Use subspace representation as an initial tool, but do not stick with it if other approaches are more functional. | DS is a starting point and source of originality, not a doctrine. |
| Increase the number of bands if possible; RGB-only xBD is not ideal for subspace methods. | Multispectral/hyperspectral routes matter because DS has little room in 3-channel data. |
| Conduct simple database evaluation to understand characteristics and verify DS validity. | Before complex models, run fair dataset diagnostics and simple baselines. |
| DS is naive/straightforward but a good starting point for future deep research. | Keep negative DS results honestly; they can guide deeper geometry or hybrid routes. |
| Title and presentation should emphasize the essential item, not many loosely connected ideas. | Every seminar/paper framing needs one main contribution and one win axis. |

## 3. Sensei-Requested Temporal Direction

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

## 4. Senpai Feedback: Spatial And Green Learning

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

## 5. Seminar Feedback

Recurring issue:

```text
What axis does the method win on?
```

Exact questions that should remain answerable from the active docs:

| Question family | Why it matters |
|---|---|
| Why ResNet34 / U-Net-ResNet was used | architecture must be framed as baseline or convenience, not thesis novelty |
| Why raw Sentinel-2 bands sometimes beat DS/PCA-diff priors | forces honest comparison between priors and raw information |
| Why PriorsFusionUNet underperformed | prevents claiming naive prior fusion before matched reruns |
| Why IR-MAD was weak in our results despite being a strong multiband baseline | requires formula/tuning audit before citing negative IR-MAD evidence |
| What Phase 1 objective is, and why PCA-diff can beat DS | requires explaining detection score maps separately from segmentation training |
| Where related work ends and the proposed method begins | must separate DS, PixelHop/Saab, U-Net, IR-MAD, and our satellite adaptation/evaluation |
| Why Sentinel-2/multispectral data are used for disaster assessment | requires a band/value argument over RGB-only imagery |
| Whether non-disaster construction before/after imagery could answer the same technical question | keeps the task problem-driven instead of disaster-themed by habit |
| What the computational-cost/accuracy tradeoff is | matters if claiming efficiency, label-free practicality, or edge deployment |
| Whether priors should be used as loss weighting, curriculum, gating, or attention instead of raw concatenation | preserves the strongest neural-prior improvement ideas |

Current answer:

- not SOTA segmentation accuracy;
- possible win axes are interpretability, label-free local evidence, candidate
  triage, complementarity as a prior, and temporal characterization.

Decision:

- every lane must define its win axis and baseline pressure before new runs.

## 6. Sensei/Seminar Coverage Table

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
| VLM explanations as subspaces | future route: embed explanations for image sets, then analyze first/second/geodesic variation | future |
| More bands than RGB | preserved as multispectral/HSI motivation and dataset-selection rule | active boundary |
| Read survey papers | literature map and baseline pressure now required before claims | active rule |
| What happens after the heatmap? | senpai repeatedly asked what the output is used for: reconstruction, resource allocation, evacuation, damage level, or planning | active decision question |
| Damage assessment vs land-use confusion | combining land-use and damage assessment was criticized as vague unless the method explains the bridge | active scope warning |
| UAV / edge processing angle | personal notes and senpai comments suggested drone imagery and resource-constrained processing as a possible win axis | candidate application |
| Temporal recovery value | first/second DS may be more meaningful for damage/recovery trajectories than one binary changed-area map | candidate route |
| Gaza / reconstruction motivation | personal motivation is reconstruction support, but the thesis still needs a precise task, dataset, and metric | motivation boundary |
| LLM/VLM idea | notes mention LLM/VLM support, but the role must be explicit: explanation, semantic retrieval, or decision support | future-only unless evaluated |
| Alternative prior usage | feedback asked about loss weighting, curriculum, gating, and attention instead of naive concatenation | preserve as neural-prior variants after standalone prior evidence |

## 7. Detailed Human Feedback Preserved From Old Notes

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
| Construction-change alternative | if disaster labels are scarce, consider whether construction before/after data answers the same technical question; do not assume disaster is required unless the research claim needs disaster semantics |
| Sentinel-2 for disaster damage | explain why multispectral bands add value over RGB when discussing damage assessment |

### Submitted QA Report Takeaways

| Topic | Preserved takeaway |
|---|---|
| OSCD labels | OSCD binary labels do not equal semantic/damage truth; spectral/radiometric change can disagree with labels |
| ranking vs thresholding | AUROC/AP and thresholded F1/IoU answer different questions |
| MultiSenGE earliest/latest | can be dominated by seasonality; use with temporal protocol, not as direct OSCD replacement |
| snow/seasonality | robustness/failure-analysis cases, not trivial noise to hide |
| IR-MAD weakness | cannot be claimed unless the implementation and tuning are fair |
| DS/PCA-diff label mismatch | treat them as spectral-change priors/explanatory overlays; OSCD masks are semantic labels and do not mark every radiometric change |

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

## 8. Current Decisions

| Decision | Reason |
|---|---|
| prioritize reproducing DS-specific neural-prior result | most consequential unverified positive claim |
| keep Successive Saab-DS as current internal OSCD anchor | strongest verified OSCD result |
| keep xBD projector as candidate-localization boundary | external evidence exists but not damage classification |
| keep temporal DS/geodesic as Sensei-aligned characterization | implemented but detector evidence weak |
| pause broad method search | too many weak routes already tested |
| collapse AI-generated docs after absorption | reduce noise and improve reading path |
