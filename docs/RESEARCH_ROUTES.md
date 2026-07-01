# Research Routes

## Quick Links

- [1. Purpose](#1-purpose)
- [2. Route Rules](#2-route-rules)
- [3. Active Ranked Routes](#3-active-ranked-routes)
- [4. Route Bank](#4-route-bank)
- [5. Closed Or Paused Routes](#5-closed-or-paused-routes)
- [6. Cross-Route Gates](#6-cross-route-gates)

## 1. Purpose

This file prevents scattered ideas from becoming either lost or falsely active.
A research route is one coherent bet:

```text
problem -> method angle -> data/evidence -> baseline pressure -> win axis -> gate
```

The route list is **not fixed**. If a source note implies a real research
problem, it gets added here even if it was not previously named.

## 2. Route Rules

- Only one route should drive implementation at a time.
- A route can be current, candidate, weak, parked, or closed.
- Weak routes are not deleted when they contain a useful idea; they are parked with a gate.
- A route must state what it could win on: accuracy, interpretability, label efficiency, candidate triage, temporal characterization, robustness, computational cost, or theory.
- A route must name the strongest baseline or control before we trust it.
- If a route is only an application idea, it needs labels and an evaluation unit before method work.

## 3. Active Ranked Routes

| Rank | Route | Win axis | Status | Next gate |
|---:|---|---|---|---|
| 1 | Successive Saab-DS for OSCD changed-area evidence | label-free spatial representation plus interpretable DS evidence | current | reproduce DS-specific neural-prior claim; find second labeled multispectral test |
| 2 | Compute-quality DS/GDS route | test whether subspace priors retain useful evidence at lower training/inference cost than frozen VFM features | candidate | compare DS/GDS, DINO feature difference, and DINO+DS with wall-clock/GPU-memory/AP/F1 |
| 3 | Deep/foundation-feature subspace geometry | test DS as a geometry layer over modern dense features, not only over raw bands | candidate | DINOv2/DINOv3 feature-difference vs DINO-feature DS on one suitable benchmark |
| 4 | DS-specific neural-prior fusion | complementary prior for supervised segmentation | verify | rerun raw/no-DS/DS/matched-cross/foundation-feature controls across seeds |
| 5 | Band-Image/projector candidate localization on xBD-S12 | analyst triage and damaged-building candidate recall | candidate | object-level retrieval protocol on fresh events |
| 6 | HSI spectral geometry and wavelength attribution | many-band spectral interpretation | parked | real labeled bitemporal HSI benchmark against SAM/CVA/IR-MAD/HSI-CD baselines |
| 7 | Structured temporal CCA/SFA/S3CCA/TRCCA | invariant/background modeling and attributable temporal change | candidate | one sequence task with raw residual, shift, PCA, SSA, and seasonal controls |
| 8 | Tensor/Product-Grassmann satellite cubes | preserve spectral-spatial-temporal modes | future | define tensor object and prove benefit over flattening |
| 9 | KDS/KGDS nonlinear satellite change | nonlinear spectral/material geometry | parked | define a nonlinear failure case for DS/PCA first |
| 10 | Application-specific object/state monitoring | greenhouse/building/urban infrastructure use case | parked | secure labels and define task: mapping, classification, or change |
| 11 | Diagnostic benchmark paper | honest boundary of subspace methods | fallback | consolidate positive and negative evidence into one defensible matrix |

## 4. Route Bank

These routes are preserved so they remain searchable. They are not all current.

| Route | Problem angle | Possible method | Needed evidence/gate |
|---|---|---|---|
| Spatially faithful Saab-DS | global pixel DS loses spatial context | PixelHop/Saab-like local features plus Band-Image DS | replicate OSCD positive and pressure with matched controls |
| True Green Learning / PixelHop route | local-to-global label-free features may help before DS | source-faithful Saab hops, not fixed-grid proxy | cite Kuo papers; compare to plain PCA/L2 and DS-free Saab controls |
| Wavelet-inspired subspace pyramid | multiscale LL/LH/HL/HH detail may preserve spatial structure | DWT/SWT per band then subspaces per coefficient family | compare against true wavelet energy/L2 and Saab route |
| Senpai multiresolution subspace tree | whole image, 2x2, 4x4, 8x8 subspaces may preserve coarse+fine structure | product of local band-image subspaces or weighted pyramid | must beat simple multiscale L2/PCA; fixed-grid pixel DS already failed |
| Band-image DS | each band/feature map is a high-dimensional spatial sample | flatten band maps, build pre/post subspaces, compare | useful as candidate localization, not current detector |
| Local-window / patch-vector DS | preserve neighborhood context around each pixel | patch samples, sliding windows, local subspaces | compare against smoothed PCA, SiROC-style spatial context, raw local L2 |
| Pixel-spectrum DS | simplest Sentinel-2 adaptation | all valid pixels as 13-D samples | mostly diagnostic; keep as baseline |
| Band-combination search | some band groups may isolate particular changes | test RGB/NIR/SWIR/red-edge/indices combinations | avoid combinatorial overfit; validate on held-out cities/events |
| Spectral feature isolation | subspaces may separate dominant multispectral factors | DS/PCA basis energy, band-group attribution | show stable interpretable factors beyond per-band differences |
| Pseudo-change vs real-change | shadows, seasonality, registration, water, clouds can dominate raw differences | nuisance modeling, IR-MAD/SFA/ACD controls, component filtering | must define what counts as real change and use labels or weak evidence |
| Post-classification/object change | semantic meaning may require pre/post objects/classes first | segment/classify before CD, then compare object states | needs class maps or reliable object proposals |
| Semantic change detection | binary labels are not enough for from-to interpretation | local subspaces, GDS/MSM/CMSM, semantic labels | needs SCD dataset or generated reliable pseudo-labels; LEVIR-CC/Dubai-CC/SECOND-CC are reading leads for caption/from-to framing, not current multispectral evidence |
| Open-vocabulary/foundation-model CD | users may ask for specific object changes, not generic change | CLIP/SAM/RS foundation features plus DS/GDS or clustering | compare to raw foundation-feature distances; avoid claiming FM novelty |
| VLM explanation subspaces | Sensei suggested explanations for image sets as subspaces | embed text explanations, first/second DS, geodesic variation | define text source, stability, and evaluation; future-only |
| Signal Latent Subspace for satellite | SLS suggests subspaces of learned latent features | frozen encoder patch/building embeddings as subspaces | compare to cosine/Euclidean feature distance and linear probes |
| Deep subspace head | geometry may regularize or explain DNN features | subspace loss/head on encoder features | requires controlled neural baseline and ablation |
| DS as neural prior | DS may not win alone but can add complementary channel | bands + DS/PCA/IR-MAD priors in U-Net/Siamese | rerun seeds and matched no-DS priors |
| Low-label prior route | priors may help when labels are scarce | label-budget sweeps with frozen priors | mean/std over seeds; compare no-prior and non-DS priors |
| Teacher/pseudo-label route | DS maps could guide pseudo-label generation | threshold/cluster DS maps then train network | must avoid self-confirming labels; compare simple pseudo-labels |
| xBD-S12 candidate triage | humans may review small budgets after disasters | rank objects/pixels by projector/DS/IR-MAD scores | fixed review budget, object recall, specificity |
| Building-level descriptors | damage is object-level, not just pixel-level | aggregate spectral/texture/context/temporal features per building | needs building masks and classifier baselines |
| Greenhouse mapping | abandoned greenhouses are a real local application | object patches, Sentinel-2/EO fusion, DS/KDS/semantic features | needs labels and task definition: mapping vs abandonment vs change |
| Google Earth / global feature service / DMaaS | automated DS or subspace-feature maps could become a review product, Earth Engine-style demo, or damage-mapping service | pipeline over Sentinel-2/HLS tiles, store feature/change maps, expose review-ready outputs, and connect to object/decision layers | engineering/application route only after method value is shown; do not let the product idea replace the research question |
| MultiSenGE utilization | multi-date dataset should support temporal route, not random testing | first/second DS, GDS, geodesic, seasonality controls | define labels/proxies and avoid vague "testing" |
| Harmonized Sentinel-2 sequence route | Sensei suggested newer Sentinel-2 source | event/date-window construction from HLS/S2 L2A | requires reproducible download and event labels |
| IrrMapper irrigation regime route | irrigation start/stop can add or remove a seasonal vegetation cycle | annual irrigation labels plus Sentinel/HLS time series, first/second DS, SFA/SSA/BFAST-style controls | IrrMapper labels are classifier-derived; use as candidate weak labels and verify switch years independently before making claims |
| OSCD date-neighborhood GDS | around each OSCD pre/post date, gather nearby images | multiple date-specific subspaces; GDS/KGDS over date sets | need clean date access and cloud filtering |
| Temporal first/second DS | Sensei directly asked for first/second DS on sequential satellite subspaces | subspace velocity/acceleration, geodesic split | compare with scalar second differences and temporal baselines |
| Geodesic along/off decomposition | distinguish smooth drift from abrupt off-path change | decompose second DS relative to geodesic path | needs clear abrupt-vs-gradual labels or synthetic-to-real gate |
| RTW warp-invariant satellite route | phenological phase should not count as damage | RTW subspace vs DTW/global shift/seasonal models | current tests negative; reopen only with new mechanism |
| SFA/ISFA route | invariant/slow background can suppress nuisance | slow-feature CD, residual outside invariant subspace | compare to IR-MAD and indices; not novelty unless DS-fusion adds value |
| CCA/KCCA route | structured correlation may align pre/post views | CCA/KCCA similarity and residuals | must beat IR-MAD or serve attribution, not raw detection |
| S3CCA/TRCCA route | Sensei/senpai explicitly pointed to structured/temporal CCA | smooth sparse band/date selection | first reproduce paper toy or small temporal localization task |
| Formal GDS route | multiple subspaces may represent classes/dates/change types | GDS over date/class/object subspaces | needs 3+ meaningful subspaces and labels/proxies |
| KDS/KGDS route | nonlinear DS may fit curved spectral relationships | paper-faithful kernel subspaces | memory-safe sample strategy and nonlinear hypothesis |
| Covariance/SPD route | covariance geometry may model spectral/texture distribution change | SPD distances, covariance equalization, tangent-space features | compare to full covariance/statistical controls |
| PCA reconstruction route | reconstruction error can be a simple anomaly baseline | train PCA on normal/pre features, score residual | baseline or diagnostic, not novelty by itself |
| Low-rank/tensor HSI route | HSI change often uses low-rank/tensor priors | tensor factorization, unmixing, low-rank residuals | only if HSI dataset and baselines are available |
| Material-subspace HSI route | each material/class may span a subspace | MSM/GDS over material spectra | must avoid being beaten by class means/SAM |
| Wavelength attribution route | which wavelengths drive change matters | DS basis energy, sparse CCA, band intervals | multiclass HSI/Sentinel labels or physics validation |
| Anomalous-change detection route | model common/background change, flag residual | chronochrome/CE/TLSQ/SFA/ACD-inspired subspace residual | must compare to ACD family |
| SSC/change-type clustering | changed pixels need grouping into meaningful types | sparse subspace clustering over changed-region features | needs semantic validation or human annotation |
| SSC pseudo-label / auxiliary-target route | label-free change clusters could supervise or regularize a U-Net | SSC over patch/object difference features, then use cluster id as input channel, pseudo-label, or auxiliary target | compare against ordinary k-means/spectral clustering and no-cluster U-Net; avoid self-confirming labels |
| Temporal recovery trajectory clustering | disaster value may be recovery pattern, not one binary change map | first/second DS features plus SSC or trajectory clustering over buildings/regions | needs multi-date objects and interpretable classes: fast recovery, delayed recovery, no recovery, repeated damage |
| Diffusion-pseudotime damage progression | damage/recovery may be a continuum rather than discrete labels | k-NN graph over tile/object descriptors, diffusion map ordering, stage markers | compare against simple severity scores and clustering; needs human or label proxy for progression |
| Component/pseudo-change filtering | remove noisy connected components or known nuisance regions | component features, filtering, calibration | current results mixed; diagnostic only |
| Registration robustness | small misalignment can create false change | shift stress tests, local matching, robust windows | report sensitivity; do not claim invariance without tests |
| Score calibration/thresholding | high AUROC can still give poor AP/Otsu F1 | validation thresholds, calibration, PR curves | required for any segmentation claim |
| UAV / edge-deployable assessment | disaster response may need fast local processing with limited compute | lightweight subspace/Green-Learning features on drone imagery, heavy U-Net only offline | needs UAV dataset or realistic compute budget; win axis is runtime/energy plus acceptable accuracy |
| Disaster-resilience planning / MCDA | maps should support decisions, not just produce heatmaps | combine damage, land-use, roads, terrain, infrastructure and subspace-derived change features in MCDA/MCDM | application route; needs decision objective and validation with planning criteria |
| Critical-infrastructure mapping | disaster output could prioritize hospitals, schools, roads, evacuation centers | object detection/change plus routing or accessibility features | needs infrastructure labels and a concrete decision task |
| Landslide / hazard-specific monitoring | satellite change route may be stronger on one hazard type | multispectral/topographic/SAR features plus DS/SFA/temporal baselines | define hazard dataset and compare to hazard-specific baselines |
| Multi-sensor fusion route | clouds, terrain, SAR, LiDAR, DEM may solve failure modes unavailable to optical-only DS | Sentinel-2 plus SAR/InSAR/LiDAR/DEM/topography descriptors | large preprocessing burden; only pursue after one focused use case |
| Deforestation / climate monitoring route | non-disaster land-cover change may provide clearer labels and less damage-specific ambiguity | Sentinel-2 time series, SFA/SSA/DMD/Fourier/DS, foundation or classical baselines | useful pivot if disaster labels remain weak; must compare to established land-cover CD methods |
| Refugee / population movement proxy route | humanitarian planning may need movement or settlement-change signals | nightlights, built-up change, road/infrastructure context, privacy-safe aggregate features | high ethical and data-risk route; do not pursue without clear safeguards and labels |
| Zero-shot / open-vocabulary object-change route | users may ask "find new buildings, damaged roofs, greenhouses, roads" without task-specific labels | CLIP/SAM/SAM-3-style segmentation or remote-sensing foundation models, VLM/LLM explanations, subspace clustering of embeddings | future only; compare against raw foundation-model retrieval and avoid claiming LLM novelty |
| Explainable seminar route | make subspace construction understandable | construction cards, figures, scripts | presentation support, not a thesis route |
| Negative/diagnostic study | map where subspace geometry fails and why | controlled experiments, failure taxonomy | valid fallback if positives remain narrow |

## 5. Closed Or Paused Routes

| Route | Current reason |
|---|---|
| Global pixel DS as standalone detector | weak against raw/PCA baselines and breaks spatial information |
| Fixed-grid spectral pyramid DS | did not beat strong controls; not true PixelHop or wavelet |
| Wavelet Band-Image proxy | did not pass positive gate |
| RTW as implemented | strong controls beat it on MultiSenGE/BreizhCrops gates |
| Generic HSI transfer | scene-dependent and not a current positive detector |
| SpaceNet7 RGB transfer | raw L2/cross controls stronger |
| RFF kernel proxy | not paper-faithful KDS/KGDS and did not justify nonlinear claim |
| Autoencoder latent proxy | not a faithful SLS/foundation-feature experiment |

## 6. Cross-Route Gates

- Do not call a route novel until the closest prior and strongest baseline are named.
- Do not use OSCD to claim damage, greenhouse, semantic, or object-level performance.
- Do not treat AI-generated positive claims as thesis evidence until rerun in the active code path.
- If a route fails, record the failure mode; it may become part of the diagnostic paper.
- Sensei-requested first/second DS, geodesic projection, CCA/SFA, KDS/KGDS, and deeper math remain important even if the current best empirical route is Saab-DS.
