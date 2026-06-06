# Research Plan

Status: current as of 2026-06-02.

## Current Research Frame

The current project is about interpretable change detection in pre/post Sentinel-2 satellite images. The active benchmark is OSCD binary change detection. Disaster damage mapping and abandoned-greenhouse mapping are future applications unless their own datasets, labels, and evaluation pipelines are implemented.

## Why This Research Is Worth Doing

The project should not be framed as "I have DS, so I will force it onto OSCD." The defensible research motivation is that remote-sensing change detection still has practical weaknesses:

- Deep models can work well, but they need labels and can be hard to interpret.
- Pseudo-changes from shadows, seasonal effects, registration differences, and sensor artifacts can look like real change.
- Classical unsupervised methods are interpretable, but often weaker than supervised models.
- A useful change map should show not only that two images differ, but where the difference is spatially meaningful.

The value of DS/subspace methods is therefore not assumed superiority. The value is to test whether a geometric, interpretable representation of pre/post change can expose useful change evidence in multispectral Sentinel-2 images.

The current method is only the baseline adaptation:

```text
one valid pixel -> one 13-band vector
pre image       -> one spectral subspace
post image      -> one spectral subspace
DS              -> compare the two subspaces
```

The real research question is whether this representation is enough, or whether satellite change detection needs spatially aware subspaces built from patches or local windows.

## What "Interpretable Spatially Meaningful Change" Means

An interpretable subspace representation should let us inspect what kind of change evidence the method is using. For example:

```text
Raw difference:
  A bright shadow or seasonal vegetation shift may produce a high pixel difference.

Global pixel DS:
  The method may say that this pixel's 13-band change lies in a major pre/post spectral-difference direction.

Patch/window DS:
  The method can also consider whether the local neighborhood changed like a coherent object or region,
  such as a new building block, removed greenhouse roof, expanded road, or changed vegetation patch.
```

So "spatially meaningful" means the change map should highlight coherent changed areas rather than isolated spectral noise. If a roof, road segment, field, or greenhouse changes, neighboring pixels should produce a consistent pattern that can be explained and visualized.

This is the reason for testing patch-vector DS and local-window DS instead of stopping at global pixel DS.

## Immediate Priority

The next important research task is a spatially aware subspace audit.

Reason: the current global OSCD subspace treats valid pixels as unordered 13-band vectors. It can produce a subspace, but it does not use pixel position while fitting PCA. Position is only used later when scores are reshaped into an image map.

## Next Experiment

Implement and run:

```text
phase1/scripts/audit_oscd_spatial_subspace.py
```

Compare:

- global pixel DS: one sample is one 13-band pixel.
- patch-vector DS: one sample is a local `3x3` or `5x5` multispectral patch.
- local-window DS: one subspace is fit per local image region, such as `128x128`.

Start with `beirut`, then add at least one dense urban city and one difficult/low-change city.

## Metrics And Outputs

Report for each method:

- AUROC
- PR-AUC
- best F1 and IoU over thresholds
- Otsu-threshold F1 and IoU
- correlation with raw spectral L2
- valid-mask exclusion rate for changed pixels
- runtime
- visual maps beside pre RGB, post RGB, and ground truth

## Decision Gates

If global pixel DS is competitive and stable, it can remain a defensible spectral-distribution baseline.

If patch-vector or local-window DS gives clearer maps or better metrics, the research should pivot toward spatially aware DS.

If corrected DS variants remain weaker than simple raw/PCA-diff baselines, the thesis should not claim DS superiority. It should focus on the negative finding, interpretability, or a revised method.

## Near-Term Writing Task

Rewrite the problem statement around this question:

```text
Can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction preserves the spatial information needed for that task?
```

## Paused Work

Do not expand xBD/xBD-S12 damage mapping, MultiSenGE GDS/KGDS, or abandoned-greenhouse mapping as main work until the OSCD subspace construction question is resolved.
