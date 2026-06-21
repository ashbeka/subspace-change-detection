# Cross-Branch Research Evidence Matrix

## 1. Purpose

This report consolidates the research questions, implementations, datasets,
results, and decision gates produced across the Codex, Claude, Antigravity,
research-mining, and legacy branches. It is an evidence map, not a claim that
every experiment is equally mature or paper-faithful.

The central question is:

> Which subspace constructions provide useful, distinct, and interpretable
> change evidence for multispectral satellite imagery, and which reduce to or
> lose against simpler radiometric, correlation, or temporal controls?

The practical objective is to select a seminar-ready result and the next
publication-oriented experiment without hiding negative evidence.

## 2. Evidence Rules

| Status | Meaning |
|---|---|
| **Positive** | Competitive or complementary on real labeled data under a fair control; still requires the stated validation. |
| **Mixed** | A real or controlled signal exists, but confidence intervals overlap, transfer fails, or a simpler control explains most of it. |
| **Negative** | The tested construction reliably loses to a simpler valid control or fails its intended invariance. |
| **Verification only** | Formula, mechanism, or reference-data behavior was checked, but no satellite-task performance claim is supported. |
| **Untested** | Present in literature/reference code or notes, but no faithful project experiment exists. |

Method names are used strictly:

- **Band-Image DS is pairwise linear DS**, not GDS.
- A random-Fourier-feature lift is a **kernel proxy**, not paper-faithful KDS/KGDS.
- An autoencoder latent-subspace experiment is a **deep-feature proxy**, not
  Fukui's Signal Latent Subspace (SLS).
- A fixed-grid DS pyramid is a **multiscale support experiment**, not PixelHop,
  Green Learning, or a wavelet transform.
- The material common-removal experiment is **GDS-inspired**, not a complete
  implementation of formal multi-subspace GDS.

## 3. Cross-Branch Results

| Research problem | Method and fidelity | Data/protocol | Strongest quantitative evidence | Decision | What it establishes / does not establish |
|---|---|---|---|---|---|
| Preserve spatial information when constructing a DS from one multispectral image per date | **Band-Image DS**, paper-faithful pairwise canonical DS after adapting the sample unit to flattened band images | OSCD, 24 real cities | Rank 12: AUROC `0.8477`, AP `0.2410`, best F1 `0.3021`; PCA-diff AP `0.2541`; AP difference not significant at rank 12 (`p=0.0646`) | **Mixed, strongest DS construction** | Spatial-axis construction is far better than global pixel DS and competitive in ranking. It does not beat PCA-diff or prove a compact DS explanation because the best rank is near the maximum centered rank. |
| Test whether Band-Image DS survives matched spatial nulls | Normalized full spatial Gram, projector-row distance, and symmetric excess cross-reconstruction, all with PCA-matched centering | OSCD, 24 cities plus official split summary | Band-Image AP `0.2410`; Gram `0.1417`; projector `0.1024`; cross-reconstruction `0.2153`; smoothed/multiscale PCA `0.2679/0.2680` | **DS beats all matched geometric nulls; not spatial PCA** | Canonical DS projection contributes evidence beyond full Gram, projector orientation, and cross-prediction. It still does not beat the strongest simple spatial-PCA map. |
| Determine whether subspace scores contain information complementary to classical change maps | Equal-weight percentile-rank fusion of smoothed PCA-diff, Band-Image DS, and repaired IR-MAD | OSCD, 24 cities and official split; no label-fitted weights | Fusion AP `0.2780` vs smoothed PCA `0.2679` (uncertain) and matched cross-reconstruction fusion `0.2657` (24-city interval entirely positive; test `+0.0115`, `p=0.0098`) | **Positive DS-specific ranking complementarity; unproven segmentation/external transfer** | DS cannot be replaced by the matched cross-reconstruction score inside the same fusion. It still does not establish calibrated binary segmentation or improvement over smoothed PCA alone. |
| Test whether simple spatial processing is more important than DS geometry | Spatially smoothed/multiscale PCA-diff control | OSCD, 24-city post-hoc analysis | Smoothed PCA AP `0.267404`; multiscale PCA AP `0.267447`; ordinary PCA-diff `0.254095`. Smoothed delta `+0.013310`, interval `[+0.004860,+0.022220]`, 19/24 wins | **Promising, post-hoc** | Spatial support itself is a strong explanation and pressure baseline. The similar `0.2674` AP to three-way fusion is coincidental; these are separate runs. A frozen rerun is required before confirmation. |
| Test the original global pixel-spectrum adaptation | Global pixel DS; each pixel is one 13-D sample | OSCD, 24 cities | AUROC `0.6270`, AP `0.0625` vs PCA-diff `0.8392/0.2541` | **Negative** | Discarding pixel position during subspace fitting is unsuitable for this task under the tested score. |
| Restore locality by enlarging the sample unit | Patch-vector DS (`3x3`, `5x5`) | OSCD, 24 cities | Patch5 AP `0.1331`; patch3 AP `0.1185` | **Negative to mixed** | Patches improve substantially over global pixel DS but remain well below PCA-diff and Band-Image DS. |
| Fit a separate subspace in each local region | Local-window DS | OSCD, 24 cities | AUROC `0.6302`, AP `0.0598` | **Negative** | Local fitting alone does not make the spectral DS score useful; small-sample instability and score semantics remain problems. |
| Approximate a coarse-to-fine spatial hierarchy | Fixed-grid `1/2/4(/8)` pixel-spectral DS pyramid | OSCD core five | Best pyramid AP about `0.0765` vs global DS `0.0791` | **Negative** | Repeating the same weak spectral DS in grid cells is not Green Learning or a useful multiscale representation. |
| Compare against established classical unsupervised detectors | Repaired IR-MAD; Celik PCA-k-means adaptation; raw L2/CVA; PCA-diff; chronochrome where applicable | OSCD, all cities | IR-MAD AUROC/AP `0.8471/0.2138`; Celik `0.6867/0.1621`; raw L2 `0.7717/0.2261`; PCA-diff `0.8392/0.2541` | **Required controls** | DS must be judged against correlation/CCA and simple difference structure, not against raw L2 alone. |
| Test whether DS priors improve a neural segmenter | U-Net with raw data and old DS/PCA priors; Siamese baseline | OSCD split | Raw U-Net IoU/F1/PR-AUC about `0.2396/0.3588/0.4331`; raw+DS worse; DS+PCA slightly higher IoU/F1 but lower PR-AUC; Siamese PR-AUC about `0.4461` | **Mixed to negative** | Old priors do not show a reproducible neural improvement. The newer Band-Image map has not yet had a clean frozen-prior trial. |
| Characterize sequential change with Sensei's first- and second-order DS | First DS between adjacent subspaces; second DS between first-DS subspaces; geodesic along/orthogonal decomposition | Controlled multispectral sequences and SpaceNet 7 real time series | Controlled local-vs-gain separation can reach AP `1.0`, but one-pixel translation fails; SpaceNet second-DS orthogonal AP `0.1127` vs radiometric rank fusion `0.1910` | **Mechanism verified; real detection negative** | The requested objects and decomposition are implemented and interpretable, but the tested real score does not beat radiometric change. |
| Detect seasonal events while separating magnitude and geometry | Seasonal first DS, second DS, and local temporal eigenspectrum | Controlled MultiSenGE interventions | First DS AP `0.403`; smoothed local eigenspectrum AP `0.688`; NDMI `0.680`; intervals overlap and injected labels are not natural events | **Mixed** | Moment/eigenspectrum change may complement indices under controlled nuisances. It is not yet a natural-event performance claim. |
| Gain temporal invariance to phenological phase shifts | Paper-guided/verified RTW versus snapshot subspace, global shift, DTW-style and reconstruction controls | MultiSenGE controlled patches; BreizhCrops natural labels | MultiSenGE RTW AP `0.8078` vs snapshot `0.8156`; BreizhCrops RTW `0.7052/0.6596` vs PCA cross-reconstruction `0.8128/0.8264` | **Negative for tested RTW** | Warping does not add value beyond simpler order-invariant or reconstruction controls under these protocols. Learned/attention warping is a different untested method. |
| Use signal-subspace analysis to suppress seasonality | SSA/M-SSA Difference Subspace and learned nuisance subspace inspired by Kanai et al. | Synthetic signals and real Sentinel-2 series | Strong synthetic harmonic-dropout/noise behavior; one constructed splice margin improves, but natural-change cases fail | **Mixed to negative transfer** | Signal subspaces can encode controlled dynamics, but current natural-event evidence is insufficient. |
| Use SFA to suppress invariant nuisance | SFA change features and slow-eigenspan proxies | Synthetic affine nuisance; IrrMapper weak-label real data | Synthetic AUC can approach `0.98`, but IR-MAD is already competitive; real irrigation AP about `0.438` vs slowness `0.576` and NDVI `0.729` | **Negative as new detector** | The invariance principle is valid, but the tested implementation neither exceeds the established CCA/MAD control nor transfers strongly. |
| Separate spectral mean, scale, shape, orientation, and DS projection locally | Deterministic local HSI moment geometry with canonical DS, projector attribution, SPD/MMD/unmixing controls | Benton development; Hermiston, Farmland, Shenzhen held out | On holdouts, orientation/DS AP: Hermiston `0.1279/0.2217` vs control `0.3778`; Farmland `0.5743/0.5036` vs `0.6449`; Shenzhen `0.0942/0.0543` vs `0.3056` | **Negative detector; useful verification** | The factors and attribution identities are mathematically testable, but the representation loses against direct controls. A high Farmland ratio was disproved by inverse-score controls. |
| Exploit wavelength-Hankel or high-dimensional spectral subspaces | Spectral SSA/Hankel DS | Hermiston HSI and synthetic controls | Real AP about `0.534` vs SAM `0.848` and CVA `0.981` | **Negative** | More bands do not automatically make DS useful; amplitude and direct spectral angle remain strong. |
| Compare material-class subspaces | MSM and common-component-removal/GDS-inspired material experiment | Salinas synthetic class swaps | Mean spectral-angle controls beat the subspace classifier; performance worsens with unnecessary dimensions | **Negative** | Material subspaces need genuine within-class sample sets and a task where class variation, rather than mean signature, is the nuisance. |
| Detect leading-eigenspace orientation changes | Grassmann/chordal orientation and covariance factorization | Synthetic, Salinas, Hermiston | Synthetic orientation works, but real/semi-synthetic AUC roughly `0.57-0.75` vs correlation controls `0.91-0.98` | **Negative for current task** | Orientation is identifiable in controlled data but a full correlation/covariance statistic subsumes it on tested scenes. |
| Attribute change to bands or dates | DS/projector row-energy and basis attribution | Hermiston and controlled tests | Conservation/invariance checks pass; real attribution AP roughly `0.18-0.26` vs direct per-band scores `0.6-1.0` | **Verification only / negative localization** | Basis-invariant attribution is mathematically valid, but not yet a validated semantic explanation. |
| Use nonlinear feature maps before DS | Random-Fourier-feature kernel proxy | Hermiston/Salinas | One amplitude task recovers AP around `0.97` but remains below CVA `0.985`; distributional task about `0.536` vs correlation `0.816` | **Negative proxy** | This does not evaluate paper-faithful KDS/KGDS and does not justify a kernel claim. |
| Reproduce nonlinear Difference Subspace construction | RKHS KDS/KGDS equations and Venus reference dimensions | Sensei's three Venus image sets | Two 100-D KDS subspaces and three 150-D inputs for a 300-D KGDS object are reproduced as reference constructions | **Verification only** | Demonstrates understanding of the TPAMI setting; no satellite KDS/KGDS performance claim exists. |
| Apply subspaces to learned features | Autoencoder latent-subspace proxy | Hermiston | Latent-subspace score about `0.672` vs latent correlation `0.891` | **Negative proxy** | This is not a faithful SLS experiment and provides no evidence for foundation-feature SLS. |
| Test whether recovery dynamics need geometry | Temporal recovery geometry versus multiband trajectory controls | Controlled/constructed recovery cases | Geometric score about `0.40` vs multiband trajectory `1.0`; on another split `0.342` vs `0.717` | **Negative geometry; positive multiband observation** | Multiband recovery carries useful information, but the tested Grassmann/DS summary discards it. |

## 4. Sensei-Method Completion Matrix

| Method/request | Faithful project status | Real-data status | Current conclusion | Required next action |
|---|---|---|---|---|
| First-order DS | Implemented with canonical principal-vector DS and formula guards | OSCD adaptations and real temporal sequences | Correct object; standalone detection usually weak | Retain as component/characterizer, not winner claim |
| Second-order DS | Implemented over sequential first-DS subspaces | SpaceNet 7 and controlled sequences | Correct construction; real AP below radiometric controls | Use for change-type interpretation only after stronger event labels |
| Geodesic decomposition | Along/orthogonal projection implemented and tested | SpaceNet 7 plus controlled sequences | Mechanism works; no real detection gain | Show decomposition visually; avoid performance claim |
| GDS | Formal multi-subspace implementation exists in reference literature/tooling, but active satellite experiments are proxy/inspired | No decisive faithful real satellite GDS experiment | **Incomplete** | Build exact GDS only for a real multi-date/multi-class question with a valid sample definition |
| KDS/KGDS | RKHS equations and Venus reference demo implemented | No faithful satellite benchmark | **Reference verification only** | Do not equate RFF proxy with KDS; select satellite data only after defining the nonlinear question |
| RTW | Paper-guided implementation and controls tested | MultiSenGE and BreizhCrops | Tested hypothesis rejected | Close this version; reopen only for a materially different learned-warping hypothesis |
| SFA | Core generalized eigenproblem tested | Weak-label irrigation plus synthetic nuisance | Valid principle; below IR-MAD/indices | Retain as baseline/nuisance feature, not novelty |
| SSA/signal subspace DS | Implemented in temporal experiments | Sentinel-2 natural transfer attempted | Synthetic success, real failure | Needs better event labels or should remain closed |
| SLS | Only autoencoder proxy | One HSI scene | Not evaluated faithfully | Requires actual product-Grassmann/frozen-feature construction before any claim |
| SFS | Slow-eigenspan proxy only | Weak-label irrigation | Not a complete SFS/KCMSM pipeline | Implement only if selected by a concrete nuisance problem |
| MSM | Implemented as material/set comparison control | Synthetic/HSI-derived material tests | Simpler spectral means/angles win | No priority without multi-view/material sample sets |
| KMSM | Reference code/literature only | Untested | Open | Low priority under current deadline |
| CCA/IR-MAD | Repaired IR-MAD implements the relevant paired CCA structure | OSCD all cities | Strong required baseline, especially AUROC | Keep as baseline and formula reference |
| KCCA | Reference only | Untested | Open | Evaluate only with a kernel-specific hypothesis |
| S3CCA/TRCCA | Papers/reference concepts only | Untested | **Sensei/senpai reading/experiment gap** | First reproduce the paper task or a small temporal localization toy before satellite adaptation |
| Green Learning/PixelHop/wavelets | Fixed-grid proxy only | OSCD proxy failed | Not actually tested | Do not call the proxy Green Learning; implement a true multiscale transform only as a separate route |

## 5. What Currently Works Well Enough To Present

Ranked by defensibility, not by preference for a method:

1. **Spatial sample definition changes DS behavior materially.** Global pixel
   DS is poor, while Band-Image DS is competitive with strong classical
   methods on 24 OSCD cities. This directly answers Sensei's spatial-information
   criticism.
2. **Band-Image DS contributes complementary ranking information.** Its
   equal-weight fusion with PCA-diff and IR-MAD significantly improves AUROC,
   although AP and thresholded segmentation improvements are not confirmed.
3. **Simple spatial PCA processing is currently the strongest positive map
   result.** This is important pressure against claiming that geometry caused
   the gain.
4. **First/second DS and geodesic decomposition are implemented and verified.**
   They are useful explanatory objects and controlled characterizers, but not
   superior real detectors in the current evidence.
5. **The project has a valuable boundary map.** RTW, SSA, SFA, HSI orientation,
   KDS proxies, and several temporal/geometric summaries reveal when the chosen
   representation discards amplitude, polarity, correlation, or location.

## 6. Ranked Research Problems

### 6.1 Recommended immediate problem

**Does the OSCD spatial sample-construction result transfer under a frozen
protocol to an external multispectral change benchmark, and does Band-Image DS
add information beyond spatial PCA and cross-reconstruction there?**

Why it ranks first:

- it directly follows Sensei's spatial-information question;
- it has real 24-city evidence and the strongest tested DS result;
- the fusion AUROC result is positive but appropriately conditional;
- the matched OSCD null experiment is now complete: Band-Image DS beats full
  normalized Gram, projector rows, and cross-reconstruction but loses to
  spatial PCA;
- OSCD has been examined repeatedly, so external data are required for a
  publication claim;
- the method and all nulls can now be frozen before the new labels are scored;
- either outcome is informative: transfer supports a construction result,
  while failure confines the finding to OSCD and city-specific behavior.

### 6.2 Other defensible problems, in priority order

2. **When does subspace geometry add information beyond sufficient first- and
   second-moment controls?** A focused empirical/theoretical diagnostic using
   paired null models, not a universal claim that geometry fails.
3. **Can first- and second-order DS decompose change type in real sequential
   multispectral events?** Requires event-type or recovery annotations, not
   only changed/unchanged labels.
4. **Can local temporal eigenspectrum features suppress radiometric nuisance
   while retaining real event change?** The controlled near-tie with NDMI
   justifies one natural-event confirmation dataset.
5. **Can subspace scores improve low-label or analyst-triage ranking when used
   as complementary features rather than standalone detectors?** Requires
   frozen spatial splits and ablation against the same radiometric features.
6. **Can a true multiscale wavelet/Green Learning representation produce more
   useful local sample sets for DS?** This is untested; the failed grid proxy
   does not answer it.
7. **Can exact GDS summarize multiple date-specific subspaces for multi-date
   change-type discovery?** Requires a faithful GDS implementation and labels
   or an evaluation protocol that distinguishes temporal event types.
8. **Can paper-faithful KDS/KGDS separate nonlinear material changes that
   linear CVA/CCA cannot?** Requires a selected nonlinear phenomenon; kernels
   must not be introduced merely because they are available.
9. **Can validated band attribution support semantic or material-level change
   explanation?** Requires calibrated wavelengths, known transitions, and
   direct band/unmixing attribution controls.
10. **Can spatially aware subspace features aid an application such as
    abandoned-greenhouse monitoring?** High practical value, but only after
    labels and a greenhouse-specific baseline protocol exist.

## 7. Recommended Seminar Story

Suggested title:

> **How Subspace Construction Changes Spatial and Temporal Change Evidence in
> Multispectral Satellite Imagery**

The story should be positive but conditional:

1. Sensei asked how the subspace is constructed and warned that the global
   pixel sample definition destroys spatial information.
2. We compared explicit sample definitions: pixel spectra, patches, local
   windows, flattened band images, and sequential subspaces.
3. Global pixel DS fails on OSCD, while Band-Image DS reaches AUROC `0.8477`
   and AP `0.2410`, close to PCA-diff AP `0.2541`.
4. Band-Image DS beats matched spatial Gram, projector-row, and
   cross-reconstruction maps, but remains below spatially filtered PCA. A
   smoothed-PCA + Band-Image + IR-MAD fusion reaches AP `0.2780` and beats its
   cross-reconstruction substitute, but gain beyond smoothed PCA is not confirmed.
5. Sensei's first/second DS and geodesic objects were also generated on
   sequential data. They are interpretable but did not outperform simpler
   radiometric controls on the current real event benchmark.
6. Therefore the next decisive question is whether this frozen construction
   and complementarity pattern transfers to external multispectral data.

Real use case: **label-free candidate ranking and analyst triage of changed
regions**. Do not describe the current result as automatic damage segmentation,
semantic interpretation, or a production-ready detector.

## 8. Completed Decisive Experiment And Next Gate

The matched OSCD experiment is complete. Band-Image DS significantly beats
PCA-matched normalized spatial Gram (`+0.0993` AP), projector-row distance
(`+0.1386`), and cross-reconstruction (`+0.0257`), but remains significantly
below smoothed/multiscale PCA on the retrospective 24-city analysis. The DS
three-way fusion also beats the matched cross-reconstruction fusion, but not
smoothed PCA alone. The official
ten-city test subset is too small to confirm the spatial-PCA gain independently.

The next gate is external, frozen validation:

1. choose a labeled multispectral benchmark without inspecting method outputs;
2. freeze preprocessing, Band-Image rank policy, spatial PCA scales, IR-MAD,
   cross-reconstruction, and equal-rank fusion;
3. make AP the primary metric and retain AUROC/F1/IoU/runtime/map analysis;
4. evaluate whether DS improves over spatial PCA and cross-reconstruction, not
   only ordinary PCA;
5. if no compatible dataset is feasible, present the current result as a
   seminar construction study rather than a publication-ready detector.

## 9. Safe And Forbidden Claims

Safe:

- subspace construction strongly affects the usefulness of DS change maps;
- Band-Image DS is the strongest tested DS construction on OSCD;
- Band-Image DS is competitive in ranking but below PCA-diff in mean AP;
- equal-weight DS/PCA/IR-MAD fusion contains complementary AUROC information;
- first/second DS and geodesic decomposition have been implemented and tested;
- several temporal and HSI hypotheses were rejected by simpler controls.

Forbidden:

- Band-Image DS is GDS;
- Band-Image DS beats PCA-diff;
- the fusion solves segmentation;
- projector row energy is validated semantic band attribution;
- RFF DS is paper-faithful KDS/KGDS;
- the autoencoder experiment evaluates SLS;
- subspace geometry universally fails in remote sensing;
- OSCD results establish disaster damage or greenhouse performance.

## 10. Evidence Gaps

- The matched spatial nulls are complete on OSCD; external frozen confirmation
  is now the main evidence gap.
- No clean external dataset/protocol has yet been selected for the exact
  13-band Band-Image construction.
- No external labeled binary change dataset confirms the OSCD construction
  result.
- Formal satellite GDS and paper-faithful satellite KDS/KGDS remain untested.
- S3CCA/TRCCA and a true Green Learning/wavelet construction remain untested.
- The newer HSI implementation is currently branch-local and must be committed
  with its ignored data loader explicitly included.
- Some Claude experiments use polarity-maximized AUC; these are mechanism
  diagnostics and must not be reported as unsupervised detector performance.

## 11. Source Reports

- `docs/experiment_reports/oscd_spatial_ds_baseline_pressure_2026-06-18.md`
- `docs/experiment_reports/oscd_band_image_ds_score_ablation_2026-06-18.md`
- `docs/experiment_reports/multispectral_temporal_difference_subspaces_2026-06-19.md`
- `docs/experiment_reports/seasonal_observation_subspace_stress_test_2026-06-20.md`
- `docs/experiment_reports/spacenet7_temporal_subspace_validation_2026-06-21.md`
- `docs/experiment_reports/multisenge_rtw_invariance_gate_2026-06-21.md`
- `docs/experiment_reports/breizhcrops_rtw_natural_label_transfer_2026-06-21.md`
- `docs/experiment_reports/hsi_local_moment_geometry_2026-06-21.md`
- `docs/experiment_reports/oscd_band_image_matched_spatial_controls_2026-06-22.md`
- `phase1/outputs/spatial_acd_multiscale_allcities_20260619_012510/`
- Claude branch: `docs/research/EXPERIMENT_FINDINGS.md`
- Claude branch: `docs/research/THESIS_CANDIDATE_FINDINGS.md`
- Claude branch: `docs/research/SUPREME_AUDIT_TABLE.md` (audited here; category labels corrected)
