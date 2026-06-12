# My Notes

## Table Of Contents

- [1. Purpose](#1-purpose)
- [2. How To Use This File](#2-how-to-use-this-file)
- [3. Current Rough Notes](#3-current-rough-notes)
  - [3.1 Main Research Doubt](#31-main-research-doubt)
  - [3.2 Current Safe Framing](#32-current-safe-framing)
  - [3.3 Subspace Construction Questions](#33-subspace-construction-questions)
  - [3.4 Spatial Information And Green Learning Ideas](#34-spatial-information-and-green-learning-ideas)
  - [3.5 CCA, KCCA, And Structured Matching Ideas](#35-cca-kcca-and-structured-matching-ideas)
  - [3.6 DS, GDS, KDS, KGDS, And Temporal Methods](#36-ds-gds-kds-kgds-and-temporal-methods)
  - [3.7 SSC And Change-Type Clustering](#37-ssc-and-change-type-clustering)
  - [3.8 Dataset And Use-Case Notes](#38-dataset-and-use-case-notes)
  - [3.9 Experiment Ideas To Keep Alive](#39-experiment-ideas-to-keep-alive)
  - [3.10 Paper And Presentation Warnings](#310-paper-and-presentation-warnings)
- [4. Notes To Translate Later](#4-notes-to-translate-later)

## 1. Purpose

This is the personal rough-notes intake file.

Use this file for thoughts written in natural language before they are turned into structured research tasks, method notes, literature notes, experiments, or paper framing.

This file should sound closer to the student's own thinking than the other project notes. The other notes should be more functional and action-oriented.

## 2. How To Use This File

Add new thoughts here first when they are still raw:

- research doubts;
- ideas from Sensei or senpais;
- personal hypotheses;
- questions that are not fully understood yet;
- possible datasets or use cases;
- paper, code, or bookmark leads;
- experiment ideas that are not yet specified.

After new notes are added here, Codex should translate them into the functional notes:

- `feedback.md` for advisor/senpai questions and critique;
- `methods.md` for method understanding and implementation implications;
- `literature.md` for papers, datasets, code references, and citations;
- `experiments.md` for concrete tests, audits, and decision gates;
- `research_paper_plan.md` for paper-facing framing and contribution logic.

Do not treat this file as polished truth. Treat it as the human thinking layer.

Future workflow note:

- If these rough notes later live in Notion or another notes app through MCP, that external note source becomes the intake layer.
- The rule stays the same: keep the raw human note, then translate only stable implications into `feedback.md`, `methods.md`, `literature.md`, `experiments.md`, or `research_paper_plan.md`.
- Do not let Notion plus repo notes become two competing truth sources; the repo notes remain the structured research/action layer unless we explicitly redesign the system.

## 3. Current Rough Notes

These notes are distilled from Apple Notes, Slack notes, Sensei messages, the updated Word research notes, and supporting source files preserved in `docs/source_records/final_organization_2026-06-12/`.

### 3.1 Main Research Doubt

- I do not want to solve a problem that I invented just because I already have a method.
- The project should become problem-driven, not method-forcing.
- The right question is not "How do I force DS onto OSCD?" but "What real problem exists in satellite change detection, and is there a place where subspace methods genuinely help?"
- Possible real weaknesses to investigate:
  - deep models need labels and can be hard to interpret;
  - many apparent changes are pseudo-changes: shadows, seasonality, registration errors, clouds, sensor/radiometric effects;
  - global pixel-level methods can lose spatial context;
  - binary labels such as OSCD may not describe all meaningful spectral or semantic changes;
  - multi-date change and recovery may need temporal structure rather than only pre/post difference.

### 3.2 Current Safe Framing

- The safest current framing is still multispectral satellite change detection, not completed disaster damage segmentation.
- OSCD is useful because it gives labels and lets us test binary change maps, but it may not be the final dataset.
- Harmonized Sentinel-2 and MultiSenGE may be better for multi-date DS/GDS/KGDS questions.
- xBD/xBD-S12 may be useful later if the project becomes disaster damage mapping, but that needs a separate damage pipeline and metrics.
- Abandoned greenhouse mapping could be a real application, but only after we have data, labels, and a clear evaluation protocol.

### 3.3 Subspace Construction Questions

- What is the actual sample used to build the subspace?
- Is the current global pixel construction meaningful enough?
- Does one valid pixel as one 13-D Sentinel-2 band vector lose too much spatial information?
- Should the sample instead be:
  - one local patch;
  - one local region/window;
  - one whole image/tile;
  - one date in a time sequence;
  - one deep feature vector;
  - one object/building/greenhouse patch?
- PCA rank should not be treated as magic. Rank 6, rank 5, variance thresholds, and the meaning of a `13 x r` basis need sensitivity checks.
- Projection back to image space is still a key explanation task. I need to understand what `D D^T delta_x`, projection energy, residual energy, and normalized projection ratios actually mean visually.

### 3.4 Spatial Information And Green Learning Ideas

- Sensei warned that the current algorithm can make a subspace but can break spatial information.
- Senpai suggested Green Learning / PixelHop / PixelHop++ / Successive Subspace Learning as possible ways to preserve multi-scale spatial information.
- Wavelet transform and image compression were mentioned as related intuition:
  - decompose an image into different levels or bands;
  - preserve informative local structure;
  - maybe compare pre/post decomposed components rather than only global pixels.
- These ideas should not become the main thesis automatically, but they are important candidates for spatially aware features before DS/KDS/SSC.

### 3.5 CCA, KCCA, And Structured Matching Ideas

- Sensei/senpai feedback pointed toward CCA-related methods:
  - S3CCA: Smoothly Structured Sparse CCA for partial pattern matching;
  - temporally regularized CCA / KOTRCCA for sequence-like data;
  - CCA/KCCA as ways to compare related views while preserving structure better than unordered PCA.
- Possible satellite views:
  - pre vs post;
  - spectral bands vs spatial patch features;
  - raw features vs deep features;
  - date windows in a time series.
- These are future methods until we define samples, outputs, and evaluation.

### 3.6 DS, GDS, KDS, KGDS, And Temporal Methods

- First-order DS naturally matches two subspaces, such as pre and post.
- GDS/KGDS need multiple subspaces and may fit multi-date datasets better than OSCD.
- Second-order DS and geodesic projection may help analyze smooth progression, recovery, or degradation, but they require more than a simple two-date benchmark.
- RTW, SSA, SFA, DMD, Fourier/time-series tools are future temporal options if we use dense Sentinel-2 sequences.
- The Fukui 2024 subspace overview reinforces that DS/GDS are part of a broader subspace geometry family: canonical angles, mutual subspace methods, GDS projection, kernel variants, and Grassmann views.

### 3.7 SSC And Change-Type Clustering

- Sparse Subspace Clustering could be a strong unsupervised baseline or pseudo-label source.
- It might cluster change features into types such as:
  - no change;
  - vegetation/seasonal change;
  - new construction;
  - building damage;
  - water/soil/moisture change;
  - urban-to-rubble or land-use transition.
- SSC should not be justified vaguely. It is useful only if our feature vectors plausibly lie in multiple low-dimensional change patterns.
- Possible use:
  - compare against DS/PCA-diff;
  - produce pseudo-labels;
  - create auxiliary channels for U-Net;
  - identify change types after detecting changed areas.

### 3.8 Dataset And Use-Case Notes

- OSCD: current labeled binary benchmark; good for controlled experiments, limited for semantic/damage claims.
- Harmonized Sentinel-2 L2A: Sensei recently supported trying this for time-sequential satellite images and asked about number of frames and time step.
- MultiSenGE: possible multi-date/multimodal path; currently not convincing as evidence unless we define evaluation.
- xBD/xBD-S12: possible damage mapping path; xBD-S12 exists partly because matching xBD-style labels to Sentinel-1/Sentinel-2 is hard but valuable.
- Abandoned greenhouses: possible local application in Tsukuba; might be useful because manual labeling is hard and change/abandonment can be subtle.
- Gaza/reconstruction motivation remains personally important, but the thesis must be technically precise and not too broad.

### 3.9 Experiment Ideas To Keep Alive

- Compare global pixel DS, patch-vector DS, and local-window DS.
- Test PCA rank sensitivity and variance thresholds.
- Visualize projection back to image space and band contributions.
- Test raw U-Net without Otsu thresholding and be clear about continuous prior channels.
- Test normalization choices: z-score, 0-1, percentile clipping, per-city vs global.
- Compare raw L2, PCA-diff, corrected DS, KDS/KPCA, CCA/KCCA candidates, SSC, and neural baselines.
- Study pseudo-change explicitly: shadows, seasonal vegetation, registration, clouds, sensor differences.
- For multi-date data, compare first-order DS, second-order DS, GDS/KGDS, geodesic projection, and temporal methods.
- For greenhouse mapping, first ask whether change detection, classification, or object mapping is the actual task.
- For object/building-level descriptors, first define the object unit: building polygon, greenhouse polygon, local patch, connected component, or tile. Then decide whether subspace features describe object state, object change, or neighborhood context.
- For the core research reset, explicitly ask: am I using subspaces because they solve a documented change-detection problem, or because the lab method exists and I am searching for a place to put it?

### 3.10 Paper And Presentation Warnings

- Do not say DS solves disaster damage segmentation.
- Do not say DS improves segmentation unless controlled experiments show it.
- Do not say OSCD proves disaster damage performance.
- Do not make the story only about the method.
- The contribution may become negative or diagnostic: finding when subspace priors help, when they fail, and which subspace construction is appropriate.

## 4. Notes To Translate Later

Use this temporary section for newly added thoughts that have not yet been propagated into the functional notes.
