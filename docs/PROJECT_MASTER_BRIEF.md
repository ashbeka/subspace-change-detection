# Project Master Brief

Generated: 2026-05-03  
Branch: `audit/project-master-brief-20260503`  
Status: current working brief, not ground truth

This document is the current truth-status map for the project. It should supersede older reset, re-entry, and generated summary documents for orientation, but it does not delete or invalidate their historical value. Every claim below should stay tied to one of these evidence labels:

- [code-evidence] Supported by inspected code, configs, scripts, or repository structure.
- [doc-claim] Claimed by project documents, notes, old reports, thesis drafts, or archived AI summaries.
- [experiment-evidence] Supported by a local smoke check or inspected output artifact.
- [external-source] Supported by external papers, dataset pages, or official project pages.
- [unverified] Plausible but not yet proven by a controlled experiment in this repo.
- [risk] A likely source of overclaim, leakage, stale evidence, or thesis weakness.
- [recommendation] A concrete next research, code, or writing move.

## 1. Project in one paragraph

This project currently investigates whether interpretable unsupervised change maps can help a supervised Sentinel-2 binary change segmentation model on OSCD. Phase 1 computes classical and subspace-based change priors from pre/post multispectral imagery, especially Difference Subspace (DS) scores and related baselines. Phase 2 trains U-Net-style segmentation models on raw pre/post Sentinel-2 bands, optionally with those prior maps as extra input channels. The honest current scope is OSCD binary change segmentation, not completed disaster damage segmentation. Damage mapping through xBD or xBD-S12 is a possible future extension, but the implemented and tested core is still OSCD change/no-change. [code-evidence] [experiment-evidence] [risk]

## 2. Current truthful scope

Implemented now:

- Phase 1 loads OSCD Sentinel-2 image pairs and labels, plus partial MultiSenGE temporal data support. [code-evidence]
- Phase 1 computes unsupervised change scores: DS projection, DS cross-residual, pixel difference, CVA, PCA-diff, Celik PCA-kmeans, IR-MAD, and geodesic DS-style priors. [code-evidence]
- Phase 2 loads OSCD patches with raw pre/post 13-band Sentinel-2 stacks, giving 26 raw channels. [code-evidence] [experiment-evidence]
- Phase 2 can append Phase 1 prior maps as extra channels, for example raw plus two priors giving 28 channels in the checked config. [code-evidence] [experiment-evidence]
- Phase 2 includes U-Net, ResNet-backbone U-Net, prior-fusion U-Net, and Siamese U-Net variants. [code-evidence]
- Phase 2 includes OSCD training, stitched evaluation, visualization scripts, and binary metrics such as IoU, F1, AUROC, and PR-AUC. [code-evidence]
- Existing output artifacts report a raw+DS improvement over raw-only in one 150-epoch run: `+0.0495` mean IoU and `+0.0580` mean F1 for E1 over E0. This is artifact evidence, not a fresh reproduction. [experiment-evidence] [risk]

Not implemented now:

- No integrated xBD or xBD-S12 damage segmentation training pipeline is alive in Phase 2. [code-evidence]
- No multi-class building damage metrics, ordinal damage labels, building-instance evaluation, or disaster-event split protocol is implemented. [code-evidence]
- No fresh multi-seed controlled reproduction has yet proven that DS priors reliably improve OSCD segmentation. [unverified]
- No evidence yet shows that the current method transfers from OSCD urban change to disaster damage assessment. [unverified] [risk]
- No current code proves state-of-the-art performance against modern unsupervised or supervised change detection methods. [unverified] [risk]

## 3. Research question

Best current research question:

Can interpretable unsupervised multispectral change priors, especially Difference Subspace priors, improve supervised Sentinel-2 binary change segmentation on OSCD compared with a raw pre/post Sentinel-2 baseline under controlled training and stitched evaluation?

Useful subquestions:

- Does DS add value beyond generic pixel/CVA/PCA/Celik/IR-MAD priors? [unverified]
- Does DS help only threshold-dependent IoU/F1, or also ranking metrics such as AUROC and PR-AUC? [unverified]
- Are any gains stable across seeds, validation choices, and train/test city splits? [unverified]
- Are the prior maps interpretable enough to justify their use beyond marginal metric gains? [unverified]

This is narrower than "damage segmentation" but defensible because it connects an interpretable unsupervised signal to supervised multispectral change segmentation. [recommendation]

## 4. Motivation

Remote sensing change detection often faces limited labels, strong class imbalance, sensor artifacts, seasonal variation, and ambiguity about what counts as meaningful change. OSCD is small enough that inductive bias and interpretable priors may matter. The project motivation is to test whether a human-understandable change cue can complement a supervised neural segmenter rather than asking the network to learn everything from raw pre/post bands. [code-evidence] [external-source] [recommendation]

The disaster response motivation should be written carefully. Sentinel-2 has broad coverage and free access, so a Sentinel-2 change-prior pipeline could eventually support wide-area screening. But the current repo does not yet localize damaged buildings, classify damage levels, or validate operational disaster response behavior. [risk]

## 5. Core idea

Difference Subspace in simple words:

- Represent local pre-event and post-event multispectral patches as low-dimensional subspaces. [code-evidence]
- Compare those subspaces geometrically instead of only subtracting pixels. [code-evidence]
- Extract directions that explain how the local spectral-spatial structure differs between dates. [external-source]
- Project image features through those directions to produce a change-score map. [code-evidence]
- Feed that map to a supervised segmenter as an extra prior channel. [code-evidence]

The central bet is not that DS alone solves change detection. The bet is that DS may provide a structured, interpretable bias that helps a supervised model focus on meaningful pre/post differences. [recommendation]

## 6. Pipeline

Input data:

- OSCD Sentinel-2 pre/post image pairs with 13 bands: `B01`, `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B09`, `B10`, `B11`, `B12`. [code-evidence]
- OSCD binary change labels from official train/test label folders. [code-evidence]
- MultiSenGE temporal data exists locally and has Phase 1 loaders/visualization support, but it is not the main Phase 2 supervised benchmark. [code-evidence]
- xBD files exist locally, but the current Phase 2 damage adapter expects CSV indexes that are absent and is not wired into the OSCD trainer/evaluator. [code-evidence] [experiment-evidence]

Phase 1:

- Load pre/post imagery and valid masks. [code-evidence]
- Normalize bands according to config. [code-evidence]
- Compute unsupervised change maps using DS, DS cross-residual, PCA-diff, pixel difference, CVA, Celik PCA-kmeans, IR-MAD, and geodesic variants. [code-evidence]
- Optionally evaluate unsupervised maps against OSCD binary labels and save per-city maps/summary CSVs. [code-evidence]

Phase 2:

- Build OSCD patches from raw pre/post data. [code-evidence]
- Optionally load saved Phase 1 change maps as prior channels. [code-evidence]
- Train a binary segmentation model, most directly U-Net, with BCE/Dice-style losses and train-only augmentation. [code-evidence]
- Evaluate by stitching patch predictions back to city-level maps and computing binary segmentation metrics. [code-evidence]

Outputs:

- Phase 1 output maps and summary CSVs under ignored `phase1/outputs/`. [code-evidence]
- Phase 2 checkpoints, evaluation JSON/CSV files, prediction maps, and ablation summaries under ignored `phase2/outputs/`. [code-evidence]
- Existing key artifact: `phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv`. [experiment-evidence]

How priors connect to segmentation:

- Priors are not labels. They are extra input features. [code-evidence]
- The supervised model still learns from OSCD labels. [code-evidence]
- The strongest thesis test is whether raw plus DS priors beats raw-only under identical training and evaluation. [recommendation]

## 7. What is trusted

- The code imports for the main Phase 1 and Phase 2 entry points passed a local non-mutating smoke check in `.venv`. [experiment-evidence]
- OSCD data is present locally with 24 image directories, 14 train label cities, and 10 test label cities. [experiment-evidence]
- A baseline OSCD Phase 2 dataset sample with patch size 128 produces `x=(26,128,128)`, `y=(1,128,128)`, and `valid=(1,128,128)`. [experiment-evidence]
- A priors OSCD Phase 2 dataset sample with patch size 128 produces `x=(28,128,128)`, `y=(1,128,128)`, and `valid=(1,128,128)`. [experiment-evidence]
- A U-Net forward pass returns logits with shape `(1,1,128,128)`. [experiment-evidence]
- CUDA is available through the local `.venv` PyTorch install at the time of the smoke check. [experiment-evidence]
- Existing Phase 1 and Phase 2 output CSVs can be read and contain plausible method rows. [experiment-evidence]
- The current code supports OSCD binary change segmentation more strongly than any other thesis storyline. [code-evidence]

## 8. What is unverified

- Whether the existing 150-epoch E0/E1 improvement reproduces from a clean command today. [unverified]
- Whether DS improvement is statistically stable across multiple seeds. [unverified]
- Whether the gain survives a stronger hyperparameter-matched baseline. [unverified]
- Whether raw+DS beats raw+PCA, raw+Celik, raw+IR-MAD, or fusion in a way that supports a DS-specific contribution. [unverified]
- Whether validation city selection or threshold calibration creates optimistic metrics. [risk]
- Whether Phase 1 prior generation and Phase 2 evaluation are fully leakage-free across train/val/test boundaries. [risk]
- Whether MultiSenGE support is mature enough to claim temporal generalization. [unverified]
- Whether xBD/xBD-S12 can be integrated without changing the research question substantially. [unverified]
- Whether the current thesis draft and reports still match the code. Some older docs are stale or broader than the implementation. [doc-claim] [risk]

## 9. What should be paused

Pause or demote these ideas until the OSCD core is reproducible:

- Completed disaster damage segmentation claims. [risk]
- Multi-class xBD damage-level classification. [risk]
- xBD-S12 as an active implementation target. [recommendation]
- UAV, edge deployment, IoT, DMaaS, MCDA, and broad disaster management platform framing. [doc-claim] [recommendation]
- Land-use analysis as a co-equal thesis objective. [doc-claim] [recommendation]
- Foundation-model or SAM/Mamba additions before the current DS-prior question is settled. [recommendation]
- State-of-the-art claims. [risk]
- Any claim that DS is a new method invented by this project. [external-source] [risk]

## 10. Novelty and contribution

Likely defensible contribution:

- A focused empirical study of interpretable unsupervised multispectral change priors as additional channels for supervised OSCD Sentinel-2 binary change segmentation. [recommendation]
- A DS-centered prior pipeline compared against classical unsupervised priors such as pixel difference, CVA, PCA-diff, Celik PCA-kmeans, and IR-MAD. [code-evidence]
- A reproducible codebase that separates prior generation from supervised segmentation and reports stitched city-level OSCD metrics. [code-evidence]
- A thesis framing that values interpretability, ablation, and honest scope over claiming a new general damage segmentation system. [recommendation]

Probably not novel:

- Difference Subspace itself. It is established work by Fukui and Maki. [external-source]
- OSCD binary change segmentation. OSCD and FC/Siamese CNN baselines are established. [external-source]
- U-Net-style segmentation for change detection. [external-source]
- Classical unsupervised priors such as PCA-kmeans and IR-MAD. [external-source]
- The broad idea that prior/semantic/change guidance can help change detection networks. [external-source]

Too weak or unsafe:

- "We solve damage segmentation." The code does not support this. [risk]
- "We propose a new DS theory." The project applies/adapts DS; it does not invent DS. [risk]
- "We are state of the art on OSCD." The repo lacks modern comparison and fresh reproduction. [risk]
- "The artifact proves DS works." A single old artifact is evidence to investigate, not proof. [risk]

Safest thesis phrasing:

This thesis studies whether interpretable unsupervised subspace-based change priors can improve supervised Sentinel-2 binary change segmentation. Using OSCD as the main benchmark, it implements Difference Subspace and classical unsupervised priors, injects them as input channels into U-Net-style segmenters, and evaluates whether they improve raw pre/post baselines under controlled ablations. [recommendation]

## 11. External research context

Difference Subspace:

- Fukui and Maki introduced Difference Subspace and generalized variants for subspace-based methods in IEEE TPAMI 2015. This makes DS an established mathematical/machine-learning method, not a new invention here. [external-source]  
  Link: https://pubmed.ncbi.nlm.nih.gov/26440259/ and https://ieeexplore.ieee.org/document/7053916/
- Fukui et al. later proposed second-order Difference Subspace as an extension for dynamics among subspaces. This supports the project's second-order/geodesic interest, but it also means the theory is moving externally and should be cited carefully. [external-source]  
  Link: https://arxiv.org/abs/2409.08563

OSCD and Sentinel-2 change detection:

- Daudt, Le Saux, Boulch, and Gousseau introduced OSCD for urban change detection with multispectral Sentinel-2 imagery and CNN baselines. OSCD is 24 Sentinel-2 image pairs with pixel-level change labels. [external-source]  
  Link: https://arxiv.org/abs/1810.08468 and https://rcdaudt.github.io/publication/2018-08-22-urban-change-detection
- Daudt, Le Saux, and Boulch also introduced fully convolutional Siamese architectures for change detection. These are important baselines and historical context for Phase 2. [external-source]  
  Link: https://arxiv.org/abs/1810.08462

Classical unsupervised methods:

- Celik 2009 used PCA and k-means for unsupervised satellite-image change detection. Any Celik-style prior in this repo is a baseline, not a project novelty. [external-source]  
  Link: https://doi.org/10.1109/LGRS.2009.2025059
- Nielsen 2007 introduced regularized iteratively reweighted MAD for multi/hyperspectral change detection. IR-MAD is a strong classical comparison point. [external-source]  
  Link: https://orbit.dtu.dk/en/publications/the-regularized-iteratively-reweighted-mad-method-for-change-dete/ and https://doi.org/10.1109/TIP.2006.888195

Modern comparison pressure:

- Metric-CD applies deep metric learning to unsupervised remote sensing change detection and appears in WACV 2025. This raises the bar for any "unsupervised CD" claim. [external-source]  
  Link: https://openaccess.thecvf.com/content/WACV2025/html/Bandara_Deep_Metric_Learning_for_Unsupervised_Remote_Sensing_Change_Detection_WACV_2025_paper.html
- Prior-guided and semantic-guided change detection already exist in the broader literature, including prior semantic information guided CD and change-prior guided networks. This means the project should not claim that using priors in neural CD is novel in itself. [external-source]  
  Links: https://www.mdpi.com/2072-4292/15/6/1655 and https://arxiv.org/abs/2404.09179

Damage mapping context:

- xBD is a large building damage assessment dataset with pre/post satellite imagery and damage labels. It is the right reference if the thesis later moves to damage, but the current code does not yet implement xBD damage evaluation. [external-source]  
  Link: https://arxiv.org/abs/1911.09296 and https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf
- xBD-S12 introduces aligned Sentinel-1/Sentinel-2 image pairs for building damage assessment and is relevant as a future Sentinel-scale damage extension, not current evidence. [external-source] [recommendation]  
  Link: https://arxiv.org/abs/2511.05461 and https://zenodo.org/records/18960454
- ChangeOS is an object-based semantic change detection framework for building damage assessment. It is an important damage-mapping comparison if this project later returns to disaster damage. [external-source]  
  Link: https://www.sciencedirect.com/science/article/pii/S0034425721003564

Must cite or compare before making strong claims:

- DS/GDS: Fukui and Maki 2015; second-order DS if using geodesic/second-order language. [external-source]
- OSCD: Daudt et al. 2018 IGARSS and FC-Siamese ICIP work. [external-source]
- Classical priors: Celik PCA-kmeans, IR-MAD, CVA/pixel differencing baselines. [external-source]
- Modern unsupervised CD pressure: Metric-CD or similarly strong recent unsupervised CD baselines. [external-source]
- Damage future work: xBD, xBD-S12, ChangeOS. [external-source]

## 12. Current repo map

Important active paths:

- `phase1/`: unsupervised prior generation, OSCD/MultiSenGE loaders, DS/PCA/classical baselines, Phase 1 eval and visualization scripts. [code-evidence]
- `phase1/configs/oscd_default.yaml`: main OSCD Phase 1 config. [code-evidence]
- `phase2/`: supervised OSCD segmentation datasets, models, training, evaluation, and visualization. [code-evidence]
- `phase2/configs/oscd_seg_baseline.yaml`: raw pre/post baseline config. [code-evidence]
- `phase2/configs/oscd_seg_priors.yaml`: raw plus DS/PCA prior config. [code-evidence]
- `phase2/configs/damage_dataset_template.yaml`: template for future damage data, not an active training path. [code-evidence]
- `docs/`: current project orientation docs, reset docs, re-entry docs, and now this master brief. [doc-claim]
- `research-notes/master/`: current scope and audit notes. This is ignored by git but important for reasoning. [doc-claim]
- `research-notes/notes/`: advisor/senpai notes and supporting notes. Ignored by git, but high-value context. [doc-claim]
- `data/`: local datasets, ignored by git. [code-evidence]
- `phase1/outputs/` and `phase2/outputs/`: local generated artifacts, ignored by git. [experiment-evidence]
- `references/reference_code/`: external/reference implementations and papers. Useful but should not be confused with active project code. [code-evidence]
- `ds_damage_segmentation.tex`: thesis/report draft, likely broader than current implementation. [doc-claim] [risk]

Main runnable entry points:

- `phase1.eval.run_oscd_eval`: OSCD unsupervised method evaluation. [code-evidence]
- `phase1.eval.run_oscd_geodesic_priors`: geodesic/DS-prior generation and evaluation. [code-evidence]
- `phase1.eval.run_multisenge_viz`: MultiSenGE visualization. [code-evidence]
- `phase1.eval.run_multisenge_temporal_geodesic`: MultiSenGE temporal/geodesic exploration. [code-evidence]
- `phase2.train.train_oscd_seg`: OSCD binary segmentation training. [code-evidence]
- `phase2.eval.evaluate_oscd_seg`: OSCD segmentation evaluation. [code-evidence]
- `phase2.viz.viz_seg_predictions` and `phase2.viz.viz_oscd_combined`: visual inspection scripts. [code-evidence]

## 13. Proposed cleanup/restructure

Do not perform this restructure yet. This is a proposal only.

| current path | action | reason | risk | approval needed |
|---|---|---|---|---|
| `phase1/` | keep | Active prior-generation implementation. | Low. | No for keeping. |
| `phase2/` | keep | Active supervised OSCD segmentation implementation. | Low. | No for keeping. |
| `phase1/configs/` | keep | Needed to reproduce Phase 1 priors. | Low. | No for keeping. |
| `phase2/configs/oscd_seg_baseline.yaml` | keep | Canonical E0 raw baseline. | Low. | No for keeping. |
| `phase2/configs/oscd_seg_priors.yaml` | keep | Canonical prior-channel experiment config. | Low. | No for keeping. |
| `phase2/configs/damage_dataset_template.yaml` | keep but mark future | Useful template, but not active evidence. | Medium if presented as implemented damage support. | No for keeping; yes before expanding. |
| `docs/PROJECT_MASTER_BRIEF.md` | keep | Current truth-status document. | Low. | No. |
| `docs/ADVERSARIAL_REENTRY_AUDIT.md` | merge/archive later | Valuable skeptical audit but should not compete with this brief. | Medium: losing historical context if deleted. | Yes. |
| `docs/IMPLEMENTATION_STATUS.md` | merge/archive later | Useful old status, partially superseded. | Medium: some details may still matter. | Yes. |
| `docs/NEXT_STEP_DECISION_MEMO.md` | merge/archive later | Decision content should be folded into this brief or a new roadmap. | Medium. | Yes. |
| `docs/PROJECT_REENTRY_SYNTHESIS.md` | archive later | Re-entry document, likely superseded. | Low to medium. | Yes. |
| `docs/PROJECT_RESET_DECISION.md` | archive later | Historical reset marker, not day-to-day truth source. | Medium. | Yes. |
| `docs/PROJECT_UNDERSTANDING_GUIDE.md` | merge/archive later | May duplicate this brief. | Medium. | Yes. |
| `phase1/docs/` | merge selected content later | Phase-specific notes may belong in method docs. | Medium. | Yes. |
| `phase2/docs/` | merge selected content later | Contains old reports and experiment summaries. | Medium: old metrics may be misread as reproduced. | Yes. |
| `research-notes/master/` | keep as ignored source notes | Contains current scope and audit reasoning. | Low, but ignored by git means not portable. | Yes before moving/copying into tracked docs. |
| `research-notes/notes/` | keep as ignored archive | Advisor/senpai context is valuable but not polished docs. | Medium privacy/context risk if tracked. | Yes before tracking. |
| `data/` | keep ignored and document expected layout | Local datasets should not be versioned. | Low. | Yes before deleting/moving. |
| `phase1/outputs/` | keep ignored; create artifact index later | Generated outputs should not be tracked, but key runs need documentation. | Medium: stale artifacts can mislead. | Yes before cleanup. |
| `phase2/outputs/` | keep ignored; create artifact index later | Contains important old ablation artifact. | High if deleted before reproducing. | Yes. |
| `references/reference_code/` | keep or move to `references/third_party/` later | Useful external reference material. | Medium: licensing/provenance needs clarity. | Yes. |
| `TEMP_DS_PRIMER.md` | merge/archive later | Likely temporary explanatory doc. | Low. | Yes. |
| `CODEBASE_AUDIT.md` | archive later | Some claims are stale relative to current code. | Medium. | Yes. |
| `RUN_PIPELINE.md` | update later | Should become a current reproducibility guide after E0/E1 rerun. | Medium if outdated commands remain. | Yes before rewriting. |
| `PIPELINE_RERUN_LOG.txt` | archive or replace later | Historical log, not canonical instructions. | Low. | Yes. |
| `ds_damage_segmentation.tex` | keep but revise scope | Thesis draft likely overstates damage scope. | High if used unedited. | Yes before major rewrite. |

Cleaner proposed repo structure later:

```text
docs/
  PROJECT_MASTER_BRIEF.md
  ROADMAP.md
  REPRODUCIBILITY.md
  archive/
phase1/
phase2/
references/
  papers/
  third_party/
scripts/
data/                  # ignored, documented layout only
outputs/               # optional ignored umbrella, or keep phase outputs
```

The cleanup should happen only after the first fresh E0/E1 reproduction, because restructuring before evidence can hide what is alive. [recommendation]

## 14. Next steps

- next reading task: Build a one-page comparison table for Fukui and Maki 2015 DS, second-order DS, Daudt OSCD/FC-Siamese, Celik PCA-kmeans, IR-MAD, Metric-CD, prior-guided CD, xBD, xBD-S12, and ChangeOS. [recommendation]
- next coding task: Add or verify one reproducible command path for E0 raw-only and E1 raw+DS that records config, seed, git commit, data split, checkpoint path, and evaluation output path. [recommendation]
- next experiment: Run a fresh controlled OSCD E0 vs E1 rerun, first short smoke-scale and then full-scale if the smoke run is clean. [recommendation]
- next writing task: Rewrite the thesis/project introduction around "interpretable unsupervised multispectral priors for supervised Sentinel-2 binary change segmentation" and explicitly move damage mapping to future work. [recommendation]

## 15. Forbidden overclaims

Do not claim:

- This project currently performs disaster damage segmentation. [risk]
- This project currently predicts building damage levels. [risk]
- This project currently implements xBD or xBD-S12 training/evaluation end to end. [risk]
- DS was invented in this project. [external-source] [risk]
- Adding priors to neural change detection is novel by itself. [external-source] [risk]
- The current method is state of the art on OSCD. [unverified] [risk]
- One old artifact proves DS reliably improves segmentation. [experiment-evidence] [risk]
- OSCD binary urban change results imply disaster damage performance. [risk]
- MultiSenGE support proves temporal generalization. [unverified] [risk]
- CUDA availability means the training pipeline has been freshly reproduced. [experiment-evidence] [risk]

## Working verdict

The project has a real, narrower core: DS and other interpretable unsupervised priors feeding supervised Sentinel-2 binary change segmentation. That is worth testing. The danger is narrative inflation: "damage segmentation", "new DS", "operational disaster response", or "state of the art" claims are not supported by the current implementation. The next decisive move is not more documentation or broader scope; it is a clean E0 raw-only versus E1 raw+DS reproduction with controlled seeds, saved configs, and a short written interpretation of whether the gain is real. [recommendation]
