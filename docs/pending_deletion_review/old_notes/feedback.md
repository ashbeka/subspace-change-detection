# Feedback Notes

## Table Of Contents

- [1. Current Sensei Feedback](#1-current-sensei-feedback)
- [2. Subspace Construction Feedback](#2-subspace-construction-feedback)
- [3. Venus And TPAMI Feedback](#3-venus-and-tpami-feedback)
- [4. Project Framing Feedback](#4-project-framing-feedback)
- [5. Older Advisor And Senpai Feedback](#5-older-advisor-and-senpai-feedback)
- [6. First Seminar Student Feedback](#6-first-seminar-student-feedback)
- [7. First Seminar QA Report Follow-Up](#7-first-seminar-qa-report-follow-up)
- [8. Paused Or Unsafe Claims](#8-paused-or-unsafe-claims)
- [9. Sensei Task Completion Status](#9-sensei-task-completion-status)

## 1. Current Sensei Feedback

Sensei agreed that the current algorithm can form a subspace, but warned that it can break spatial information in the input image.

Core questions from Sensei:

- What is the purpose of applying DS to this dataset?
- Is the task detection of areas that changed between pre and post event imagery?
- Clarify the research purpose and problem again, because different problems require different solutions.

Current honest reply:

```text
The current purpose is detecting changed areas between pre- and post-event Sentinel-2 images, using OSCD as a binary change-detection benchmark.

I understand that the current global pixel-based subspace can break spatial information. I will compare it with spatially aware versions such as patch-vector DS and local-window DS before continuing larger experiments.

The revised question is: can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction preserves the necessary spatial information?
```

Action:

- Run the spatially aware OSCD subspace audit before any more long U-Net sweeps.
- Treat the next research reset as problem clarification, not only method debugging.

Recent Sensei direction from the final source batch:

- Try time-sequential satellite imagery such as Harmonized Sentinel-2 L2A.
- Before applying temporal DS/GDS ideas, report practical sequence facts: number of image frames, date spacing, cloud/no-data rate, and whether the time step is regular.
- First/second DS and geodesic projection may be meaningful when there are multiple temporally ordered observations, not just one pre/post pair.
- Discuss adequate subspace construction with lab members who have worked on related multi-channel or temporal subspace problems.

Sensei-priority task order as of 2026-06-14:

1. Talk to Jang about how to generate an adequate subspace for this problem.
   - Reason: Sensei explicitly recommended this after hearing that the current multi-channel satellite adaptation may not match the original code/data setting.
   - Ask specifically about multi-channel data, because Jang has related work on multi-channel EEG.
   - Bring concrete options: global 13-band pixel subspace, patch-vector subspace, local-window subspace, and time-sequential date subspaces.
   - Status 2026-06-14: discussed the implementation difficulty with Jang. This helped clarify the issue, but did not remove the need to choose and verify the satellite-specific subspace construction ourselves.

2. Talk to Aono-kun about first/second DS and geodesic decomposition theory/implementation.
   - Reason: Sensei explicitly recommended Aono for theoretical aspects and later for implementation of first/second DS and geodesic decomposition.
   - Ask for the analysis tool used in human-motion analysis and what inputs it expects.
   - Status 2026-06-14: discussed the implementation difficulty with Aono. Treat this as helpful orientation, not as complete validation of the current project implementation.

3. Generate a set of time-sequential satellite subspaces from Harmonized Sentinel-2 or another valid multi-date dataset.
   - Reason: Sensei repeatedly asked whether a time-sequential satellite dataset had been obtained, and asked for the Google dataset frame count and time step.
   - Minimum facts to report: area, dates, number of valid frames, nominal revisit interval, cloud/no-data filtering, bands used, and whether the images are spatially aligned.
   - Status 2026-06-19: completed first trials on five 23-date MultiSenGE
     patches and the published 20-frame registered IPOL Las Vegas RGBI
     sequence. Harmonized Sentinel-2 remains the requested independent dataset.
   - Status 2026-06-20: expanded external pressure to four IPOL sequences and
     tested backward/forward date-window subspaces. A labeled or independently
     annotated sequence is still missing; IPOL detector maps are not truth.

4. Calculate first DS magnitude, second DS magnitude, and geodesic-decomposition/projection quantities on the time-sequential subspaces.
   - Reason: Sensei specifically asked whether the changes in magnitudes of the first and second DSs can be calculated soon.
   - Treat this as a lab-aligned exploratory result even if it does not beat modern change-detection baselines.
   - Status 2026-06-19: implemented and formula-tested. First, second, along,
     orthogonal, spatial-contribution, and irregular-cadence diagnostics now
     run through the project CLI. The current whole-scene maps are diffuse and
     registration-sensitive, so local/multiscale and baseline tests are next.
   - Status 2026-06-20: the exact IPOL detector was reproduced on four
     sequences. Second/time-aware quantities show a small sequence-level
     agreement lead over raw interpolation residual, but lower pixel AP.
     Temporal-context DS localization failed; projection novelty and
     registration robustness are the remaining hypotheses.

5. Keep the OSCD spatial-information experiment as the verification track.
   - Reason: Sensei warned that the current global pixel subspace can break spatial information.
   - Use OSCD labels to compare global pixel DS, patch-vector DS, local-window DS, PCA-diff, raw L2/CVA, Celik, and IR-MAD.

6. Keep Venus nonlinear DS/KDS/KGDS as the method-understanding proof.
   - Reason: Sensei explicitly gave Venus data and recommended running nonlinear DS from TPAMI2015.
   - The goal is to show that the original image-set setting is understood before claiming a satellite adaptation.

Actions:

- Add a Harmonized Sentinel-2 or selectively acquired labeled DynamicEarthNet
  feasibility audit before any serious GDS/KGDS or temporal performance claim.
- Prepare concrete questions for Jang/Suto/Pedro/Santos about sample definition, spatial preservation, rank choice, and reference-code behavior.

2026-06-17 source-batch update:

- Treat Sensei's spatial-information concern as the central immediate research problem, not a side concern.
- Treat Jang's channel-wise flattening suggestion as a concrete experiment candidate:
  - current OSCD DS: one pixel is one 13-D spectral vector;
  - Jang-style candidate: one band/channel image is flattened into a spatial vector, then subspaces compare spatial band patterns.
  - Interpret this as advice to test, not ground truth. The key matrix flip is `13 x N_pixels` versus `N_pixels x 13`.
  - The advantage is spatial layout preservation inside each band vector; the risk is that Sentinel-2 supplies only 13 band-image samples.
- Treat Aono/geodesic/first-second DS work as the Sensei-aligned temporal track:
  - first build a real multi-date Harmonized Sentinel-2 sequence;
  - then define one subspace per date;
  - then calculate first DS, second DS, and geodesic projection/decomposition quantities.
- Do not let semantic change, greenhouse mapping, hyperspectral anomaly detection, or foundation-model ideas replace the Sensei-first tasks until their datasets and evaluation protocols are real.
- The latest note batch strengthens the Sensei-first temporal interpretation:
  the temporal axis can supply the real set of related observations that the
  original image-set methods require. Build a subspace from multiple dates of
  the same aligned region/season, rather than assuming that unordered pixels
  from one date are equivalent to multiple views of one object.
- For any event study, report whether first/second magnitudes change at the
  documented event and remain stable during no-event periods. Separate abrupt
  disturbance, gradual drift, recovery, seasonality, and acquisition artifacts.
- IrrMapper transitions are weak labels derived from annual random-forest maps.
  Do not describe them to Sensei as independently annotated switch dates.

## 2. Subspace Construction Feedback

Questions raised by Sensei/senpais:

- What exactly is the subspace in this project?
- Is the subspace built per band, per whole image, per pixel vector, or another way?
- What is the dimension of the subspace?
- Are pixel positions lost?
- How is the projection done?
- What does projection energy mean?
- What does PCA rank 6 mean?
- Is one subspace per satellite date image mathematically meaningful?
- Should local patches, sliding windows, CCA, KPCA, KDS, or KGDS be used instead?

Current answer:

- OSCD currently uses one valid pixel as one 13-D sample vector.
- For each city/date, the matrix is `X in R^(13 x N)`, where rows are Sentinel-2 bands and columns are valid pixels.
- PCA rank 6 gives a `13 x 6` basis for the pre image and a `13 x 6` basis for the post image.
- This is not per-band PCA.
- This is not TPAMI-style whole-image-vector image-set PCA.
- It is one spectral-distribution subspace per date image.
- Pixel coordinates are not used during global PCA fitting. They are only used later to reshape scalar scores back into a map.

Action:

- Compare global pixel DS against patch-vector DS and local-window DS.
- Add valid-mask exclusion checks so real changed pixels are not silently removed.
- Add rank sensitivity experiments instead of treating rank 6 as special.

## 3. Venus And TPAMI Feedback

Sensei provided Venus MATLAB files used in the TPAMI 2015 nonlinear DS/KGDS setting.

Current understanding:

- Venus raw arrays are `(480, 640, 1, 300)`.
- The demo downsamples each view to `63 x 48`.
- One downsampled Venus view becomes one `3024-D` vector.
- The 300 views of one sculpture form `X in R^(3024 x 300)`.
- KDS compares two kernel subspaces.
- KGDS compares three or more kernel subspaces.

Important contrast:

```text
TPAMI Venus:
  one sample = one whole downsampled image view
  vector entries = 3024 grayscale pixel values
  samples = 300 views of one sculpture
  subspace = one image-set subspace

Current OSCD:
  one sample = one valid Sentinel-2 pixel
  vector entries = 13 band values at that location
  samples = all valid pixels in one date image
  subspace = one PCA subspace per date image
```

Action:

- Do not claim the OSCD adaptation is the original TPAMI image-set formulation.
- Use the Venus implementation as a learning and reference audit.
- Implement preimage reconstruction only if TPAMI-style emphasized images become necessary.

## 4. Project Framing Feedback

Current danger:

- The project can become "I have DS, so I force it onto OSCD."
- That is not a good research gap.
- The stronger warning from the final notes batch is: do not solve a problem we invented.

Better framing:

- Remote-sensing change detection faces limited labels, pseudo-changes, registration artifacts, seasonal effects, class imbalance, and interpretability problems.
- Deep learning can perform well when labels and domains match.
- DS is not assumed better.
- The project tests whether interpretable subspace representations can produce useful, spatially meaningful change evidence.

Action:

- Start the next research re-grounding with real remote-sensing change-detection problems, then decide whether DS/subspace tools answer one of them.
- Do not claim DS superiority unless experiments prove it.
- Treat negative results as valid evidence.
- Focus on what subspace construction is appropriate for satellite imagery.

## 5. Older Advisor And Senpai Feedback

Source: nested repo `research-notes/`, especially `notes/sensei_notes.md`, `notes/senpais_notes.md`, `notes/my_notes.md`, `notes/old_notes.md`, and `master/session_audit_2026-03-24.md`.

Status:

- Ingested as historical feedback on 2026-06-07.
- Several old scope decisions are superseded by current evidence, especially the later 3-seed OSCD result and Sensei's spatial-information concern.

Durable Sensei guidance:

- Use subspace representation and DS as an initial tool because it may create an original research point, but do not stick to DS if better methods solve the problem more clearly.
- Read change-detection survey papers and the DS/second-order DS papers to understand the method family, not only this codebase.
- Run simple dataset evaluations first to understand whether DS is valid for the chosen remote-sensing task.
- First-order and second-order DS may be useful for multi-temporal satellite analysis when enough dates exist.
- TPAMI 2015 DS/KDS/KGDS and the Venus experiment are reference material for correct subspace understanding.
- Future directions can include deep features, DS+DNN, DS+LLM, RTW, SSA, and SFA, but only after the core problem is clear.
- Sensei specifically pointed to Hiraoka-san's RTW paper, "Attention Mechanism in Randomized Time Warping" (`arXiv:2508.16366`), accepted to the ICIP Learning Beyond Deep Learning workshop according to the archived note, and asked whether comments had been sent. Keep this as a concrete future-reading/action lead, not current method evidence.

Durable senpai guidance:

- Define the task before defending the method: binary changed-area detection, damage-level classification, land-use mapping, or decision support are different projects.
- Explain what happens after producing maps; vague "reconstruction support" is not enough.
- Do not combine land-use, damage assessment, SSC, U-Net, MCDA, UAV, and LLMs unless the method and purpose are concrete.
- If using subspaces, consider whether multiple images/dates/views should form one subspace instead of one global subspace per date image.
- SSC is only useful if its role is explicit: baseline, pseudo-label source, change-type clustering, compression, or interpretability.

Actions:

- Keep the current problem statement centered on interpretable subspace-based multispectral change detection unless a stronger dataset/task justifies a pivot.
- Treat xBD-S12, MultiSenGE GDS/KGDS, SSC change-type clustering, and operational decision layers as future or warm-extension tracks, not active claims.
- Before adding new methods, write the problem they solve and the evaluation that would prove they help.

## 6. First Seminar Student Feedback

Source: local Excel import `docs/source_records/student_feedback_channel4_2025-11-20.xlsx`.

Status:

- Extracted from the student feedback sheet.
- Japanese comments are translated to English below.
- Treat this as communication and research-framing feedback, not as proof that any method works.

Main themes:

- The presentation was generally clear, well organized, and visually helpful.
- Several students were confused about the practical value of adding DS/PCA-diff when raw Sentinel-2 input performed better.
- Several questions asked why specific baselines or architectures were chosen: ResNet34, PriorsFusionUNet, IR-MAD, DS, and PCA-diff.
- The proposal boundary was unclear to at least one student: what is existing related work, and what is the proposed method?
- Disaster framing created confusion. Students asked why Sentinel-2 is used for disaster damage assessment and whether non-disaster before/after settings such as construction imagery would also work.
- Presentation pacing and metric explanations need improvement.

Extracted feedback and actions:

| row | feedback in English | action |
|---:|---|---|
| 2 | Presentation was very good. | Keep clear structure. |
| 3 | The presentation was careful, easy to understand, and informative. | Keep careful explanation style. |
| 4 | Why use ResNet34 in the supervised stage? | Explain architecture choice and whether it is baseline, ablation, or convenience. |
| 5 | What feature differences do DS and PCA-diff capture? If adding them lowers IoU/F1, why might that happen, and how will they be used going forward? | Prepare a clear DS-vs-PCA-diff explanation and connect it to the negative v5 result. |
| 6 | Looks good. | No specific action. |
| 7 | No comment. | No action. |
| 8 | Images on the slides helped understanding. | Keep visual examples. |
| 9 | Why did raw Sentinel-2 pre/post bands alone achieve higher Phase 2 accuracy? | Add a slide/answer: priors may highlight unlabeled or noisy radiometric change; raw U-Net can learn task-specific filters. |
| 10 | Learned useful applications/effects of subspace methods. | Keep subspace motivation, but avoid overclaiming. |
| 11 | If the method can overcome seasonal change such as snow, it would be valuable. | Add seasonal/pseudo-change robustness as a future evaluation gap. |
| 12 | Good presentation and strong understanding, but time was close to the limit. | Slow down and shorten dense sections. |
| 13 | Rich images and explanation were good; explain AUROC and other metrics better. | Add a metric-definition slide. |
| 14 | E0 raw-only seemed best despite minimal input features. What are possible reasons? | Prepare explanation for why extra priors can hurt. |
| 15 | Why is PriorsFusionUNet less accurate than U-Net/ResNet-U-Net? | Treat fusion architecture as unproven; inspect whether simple fusion overfits or overweights priors. |
| 16 | Why use multispectral satellite data such as Sentinel-2 for disaster damage assessment? | Clarify current scope: Sentinel-2 change detection first; disaster damage is future unless damage labels/pipeline exist. |
| 17 | IR-MAD is generally strong for multiband data. Why was it weak here? Is it due to Sentinel-2 spectral characteristics? | Do not dismiss IR-MAD casually; audit implementation, thresholding, normalization, labels, and Sentinel-2/OSCD fit. |
| 18 | Presentation was extremely clear and well organized. | Keep organization. |
| 19 | Presentation was good, but speaking was sometimes too fast and hard to follow; summarize more for viewers. | Slow down and summarize transitions. |
| 20 | How long does one experiment take? | Record runtime/compute cost for each serious experiment. |
| 21 | What is the benefit of adding DS/PCA-diff for segmentation? | State the hypothesis clearly: interpretability/prior signal, not guaranteed accuracy gain. |
| 22 | Combining classical methods with deep learning seems original and practical. | Useful positive framing, but keep claims evidence-based. |
| 23 | Examples made the presentation easy to understand. | Keep concrete examples. |
| 24 | Presentation was good, but technical material was hard to follow for someone without image-processing background. | Add simpler conceptual explanations before math/results. |
| 25 | What other applications besides disaster damage assessment are possible? | Mention urban change, land-cover change, greenhouse monitoring, construction monitoring, and temporal anomaly detection as candidate applications. |
| 26 | What characteristics does the dataset have? | Add a dataset slide: OSCD, Sentinel-2, 13 bands, binary labels, pre/post dates, limits. |
| 27 | Methodology was clear, but practical applications should be explained more. | Strengthen motivation and application framing. |
| 28 | Figures/tables and slide structure were good. | Keep visual structure. |
| 29 | Clear voice, but content was difficult due to English and field unfamiliarity. The student understood satellite-image disaster assessment, subspace-based change detection/segmentation, and planned Phase 3, but asked where related work ends and the proposed method begins. | Add a "related work vs my contribution" slide. Avoid implying a fixed Phase 3 or completed damage pipeline unless implemented. |
| 30 | What is the trade-off between computational cost and accuracy when moving from classical subspace methods to deep-learning change/damage segmentation? | Track runtime, data needs, interpretability, and accuracy in future comparisons. |
| 31 | Presentation was well done. | No specific action. |
| 32 | If DS/PCA-diff priors sometimes hurt IoU/F1, have alternatives been tried: loss weighting, curriculum, early-epoch-only priors, gating, or attention? | Add alternative prior-use experiments after spatial DS audit if priors remain relevant. |
| 33 | Presenter self-entry. | Ignore as feedback. |
| 34 | What is the objective of the prior-generation stage, and why does PCA-diff perform best among the listed methods? | Clarify objective: unsupervised change evidence/prior generation; explain PCA-diff baseline strength. |
| 35 | If disaster damage labels are scarce, why not use before/after construction imagery instead of disaster pre/post imagery? | Good research-gap question: define whether the project targets general change detection, disaster damage, or broader before/after remote-sensing change. |

Concrete presentation fixes:

- Add a one-slide problem statement: "changed-area detection in multispectral satellite images," with disaster damage as future/application context unless using a damage-labeled dataset.
- Add a one-slide contribution boundary: existing methods vs this project's adaptation/evaluation.
- Add a one-slide metrics glossary: IoU, F1, AUROC, PR-AUC.
- Add a one-slide "why priors can hurt" explanation.
- Add runtime/compute-cost numbers for serious experiments.
- Keep visual examples, but reduce dense text and slow down.

Concrete research tasks from student feedback:

- Explain and test why raw pre/post input can outperform DS/PCA priors.
- Audit IR-MAD before making claims about it being weak.
- Record runtime/compute trade-offs for classical priors versus neural segmentation.
- Consider alternative prior integration: gating/attention, loss weighting, curriculum, or early-epoch-only priors.
- Clarify application scope: general change detection, disaster damage, construction monitoring, greenhouse mapping, or another specific target.
- Add seasonal/pseudo-change robustness to the future experiment list.

## 7. First Seminar QA Report Follow-Up

Source: submitted QA report `docs/source_records/qa_report_response_2025-11-20.pdf`.

Status:

- Preserved as a source record because it was submitted to humans.
- Useful content is extracted below.
- Wording in the report is historical; current claims should follow `docs/PROJECT_BRIEF.md` and `notes/research_paper_plan.md`.

Durable takeaways:

- DS/PCA-diff highlight spectral/radiometric change, while OSCD labels only selected semantic land-cover or land-use changes. This explains why priors can reveal real image differences that count as false positives against OSCD masks.
- This mismatch can improve ranking metrics such as AUROC/PR-AUC while hurting fixed-threshold IoU/F1. The paper should discuss that distinction explicitly.
- MultiSenGE was previously used by pairing earliest valid date versus latest valid date for each spatial patch. This maximizes temporal baseline, but it also creates seasonality risk because the time gap is uncontrolled.
- A stronger MultiSenGE design would compare controlled within-season pairs, use several dates around an event/window, or move to first-/second-order DS for progression of change.
- Snow and seasonal vegetation are not nuisances to ignore; they are expected failure modes for spectral priors and should become explicit robustness checks.
- IR-MAD is a strong established multiband baseline. Weak old IR-MAD results should be treated as implementation/dataset-fit questions, not proof that IR-MAD is weak. A fair audit needs band selection, robust covariance/regularization, and calibrated thresholds.

Actions:

- Add a clear slide/paper paragraph distinguishing spectral change from semantic labeled change.
- Report ranking metrics and threshold metrics separately.
- Add a MultiSenGE pairing audit before making claims from earliest/latest visualizations.
- Add snow/cloud/season-aware preprocessing to the future experimental design.
- Audit IR-MAD before using it as a negative baseline in a paper.

## 8. Paused Or Unsafe Claims

Do not currently claim:

- Completed disaster damage segmentation.
- xBD or xBD-S12 end-to-end training/evaluation.
- Building damage-level prediction.
- DS was invented in this project.
- OSCD binary change proves disaster damage performance.
- Old residual-stack priors prove paper-faithful DS works.
- Current global pixel DS preserves spatial structure during fitting.

## 9. Sensei Task Completion Status

Status after the 2026-06-22 cross-branch review:

| Sensei request | Status | Evidence / remaining gap |
|---|---|---|
| Explain how a multi-channel image becomes a subspace | Completed for pixel, patch, local-window, band-image, and temporal variants | Construction cards are in `notes/methods.md`; Band-Image DS is the best current satellite construction. |
| Address loss of spatial information | Experimentally addressed, not solved | OSCD global/patch/window/band-image/pyramid and matched Gram/projector/cross-reconstruction comparisons completed; external confirmation remains. |
| Run nonlinear DS on the Venus data | Completed as reference verification | KDS/KGDS equations and dimensions reproduced; no satellite KDS claim. |
| Generate sequential subspaces | Completed on MultiSenGE, SpaceNet 7, and BreizhCrops-derived temporal tasks | Specific Harmonized Sentinel-2 evaluation remains optional. |
| Calculate first DS, second DS, and geodesic components | Completed and formula-tested | Controlled behavior is interpretable; SpaceNet 7 detection result is negative. |
| Conduct various tests on first/second DS magnitudes | Completed across controlled nuisances and real temporal data | Report both successful mechanisms and failure under translation/radiometric controls. |
| Investigate SSA/SFA/RTW | Completed for project adaptations | RTW and natural-transfer tests are negative; SFA/SSA synthetic behavior does not establish novelty. |
| Investigate CCA | Partly completed through repaired IR-MAD | Standalone CCA explanation exists; S3CCA/TRCCA and KCCA remain untested faithfully. |
| Investigate KPCA/KDS | Venus construction completed; satellite method incomplete | RFF proxy is not KDS; a nonlinear satellite question must be defined before implementation. |

The next Sensei-facing result should show the subspace constructions, the
first/second/geodesic objects, and the OSCD comparison without claiming that all
requested methods improve detection.
