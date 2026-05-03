# Project Understanding Guide

This is a researcher-facing explanation of the project as it stands now. It is intentionally conservative. The current working project is about Sentinel-2 binary change segmentation on OSCD, not completed disaster damage segmentation.

## 1. What Is This Project Trying To Prove?

The safest claim is:

> Unsupervised multispectral change maps, especially Difference Subspace maps, may be useful as interpretable input priors for supervised Sentinel-2 change segmentation.

In simpler words: instead of giving a neural network only the raw before/after satellite bands, we also give it a hand-crafted "where change might be" map and test whether that helps.

The project is not trying to prove yet that it can map disaster damage from xBD/xBD-S12. That remains future work.

## 2. What Problem Are We Solving?

The immediate problem is binary change segmentation:

- Input: two Sentinel-2 images of the same city, before and after.
- Output: a map of changed vs unchanged pixels.
- Dataset: OSCD.

The broader motivation is disaster or urban-change mapping, but the implemented benchmark is OSCD change detection, not building-level damage assessment.

## 3. Why Use Difference Subspace?

Difference Subspace is used because it gives an interpretable, label-free way to summarize spectral change.

The idea is:

- The pre-image has a spectral structure.
- The post-image has a spectral structure.
- DS compares those structures and scores pixels by how strongly their change aligns with the difference between the two subspaces.

Why that is attractive:

- It does not require labels to produce a change map.
- It works on multispectral bands, not just RGB.
- It gives a prior that can be inspected separately from the neural network.

Important caveat: DS is not currently proven to be the best standalone detector. Fresh Phase 1 results show PCA-diff has higher OSCD test AUROC than DS projection, while DS projection is close to pixel difference.

## 4. What Does Phase 1 Do?

Phase 1 creates unsupervised change maps.

It takes OSCD pre/post Sentinel-2 images and produces score maps such as:

- `ds_projection`
- `ds_cross_residual`
- `pixel_diff`
- `pca_diff`
- other slower or side methods in fuller configs, such as Celik and IR-MAD

These maps are saved under `oscd_change_maps/...` and can later be used by Phase 2 as extra input channels.

Fresh re-entry evidence:

- Phase 1 fast-prior generation ran successfully in `phase1/outputs/reentry_fast_priors_20260502_020139`.
- The broad Phase 1 pattern was reproduced for fast methods: `pca_diff` highest test AUROC, `ds_projection` and `pixel_diff` close, `ds_cross_residual` much weaker.

## 5. What Does Phase 2 Do?

Phase 2 trains supervised OSCD segmentation models.

The baseline model gets:

- 13 pre-event Sentinel-2 bands
- 13 post-event Sentinel-2 bands
- total: 26 raw channels

The prior-assisted model gets:

- the same raw 26 channels
- plus one or more Phase 1 change maps, such as DS projection

So Phase 2 asks: does adding a Phase 1 prior help the neural network segment change?

Fresh re-entry evidence:

- A raw OSCD forward pass succeeded.
- A fresh one-epoch E0 raw vs E1 raw+DS smoke experiment completed end to end on CUDA.
- That one-epoch run proves the pipeline is alive, not that one method is better.

## 6. What Is Not Proven Yet?

Not proven yet:

- DS improves final segmentation in a freshly reproduced full training run.
- The old 150-epoch results are fully reproducible.
- The result is robust across multiple seeds.
- The best thresholding protocol is settled.
- Runtime comparisons are valid.
- CVA is meaningfully different from pixel difference in this code.
- The project performs real disaster damage segmentation.
- xBD/xBD-S12 is implemented end to end.

Allowed narrow statement: the saved 150-epoch artifact reports raw+DS higher than raw-only on test mean IoU and F1. That is artifact evidence, not a fresh reproduction.

## 7. What Are You Probably Misunderstanding?

You may be mixing up these things:

- OSCD change segmentation is not disaster damage segmentation.
- DS priors are not labels. They are extra input features.
- Phase 2 raw U-Net does not train on thresholded Phase 1 masks.
- A good standalone detector is not automatically the best neural-network prior.
- A one-epoch smoke run is not a performance result.
- Old outputs are evidence, not truth.
- "Damage" in the project title is aspirational until a real damage dataset pipeline exists.

## 8. What Should You Understand Before Running More Experiments?

Before more experiments, understand:

- What OSCD measures: binary land-cover/urban change, not damage severity.
- What the 26 raw channels are: 13 bands before plus 13 bands after.
- What a prior channel is: a continuous score map added to the model input.
- How Phase 1 and Phase 2 connect: Phase 1 writes maps; Phase 2 reads them.
- Why metrics can disagree: IoU/F1 depend on thresholded predictions, while AUROC evaluates ranking over thresholds.
- Why one seed is weak evidence.
- Why old output folders need provenance: config, command, seed, epoch count, checkpoint, git hash, and output path.

## 9. What Should You Read First?

Read in this order:

1. `docs/PROJECT_REENTRY_SYNTHESIS.md`  
   Start here for the current direction and the latest fresh smoke run.

2. `docs/IMPLEMENTATION_STATUS.md`  
   Use this to understand what is actually implemented versus planned.

3. `docs/PROJECT_RESET_DECISION.md`  
   Use this to remember what is trusted, unverified, allowed, and forbidden.

4. `phase2/configs/oscd_seg_baseline.yaml` and `phase2/configs/oscd_seg_E1_raw_ds.yaml`  
   Read these to see the raw-only and raw+DS experiment definitions.

5. `phase1/configs/oscd_priors_fast.yaml`  
   Read this to see which prior maps are generated for the core pipeline.

Only after that, read the lower-level code or older narrative docs.

## 10. Whole Pipeline In Simple Words

1. Start with two Sentinel-2 images of the same city: before and after.

2. Phase 1 compares them without using labels and creates change-score maps. One of those maps is DS projection.

3. Phase 2 trains a U-Net to predict the OSCD change mask.

4. The raw baseline gives the U-Net only the before/after Sentinel-2 bands.

5. The prior-assisted version gives the U-Net the before/after bands plus a Phase 1 change map.

6. The research question is whether that extra prior helps the model, and whether it gives a more interpretable route than raw bands alone.

7. So far, the pipeline runs. The old artifact suggests raw+DS can help in a 150-epoch single-seed run. The fresh one-epoch run only proves the pipeline is alive.

The honest thesis direction today is: interpretable unsupervised priors for supervised Sentinel-2 change segmentation. Damage mapping should stay future work until it is actually implemented and evaluated.
