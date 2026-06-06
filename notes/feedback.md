# Feedback Notes

This file keeps advisor, senpai, seminar, and self-critique feedback in one place. Each item should end in a concrete action.

## Current Sensei Feedback

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

## Subspace Construction Feedback

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

## Venus And TPAMI Feedback

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

## Project Framing Feedback

Current danger:

- The project can become "I have DS, so I force it onto OSCD."
- That is not a good research gap.

Better framing:

- Remote-sensing change detection faces limited labels, pseudo-changes, registration artifacts, seasonal effects, class imbalance, and interpretability problems.
- Deep learning can perform well when labels and domains match.
- DS is not assumed better.
- The project tests whether interpretable subspace representations can produce useful, spatially meaningful change evidence.

Action:

- Do not claim DS superiority unless experiments prove it.
- Treat negative results as valid evidence.
- Focus on what subspace construction is appropriate for satellite imagery.

## Paused Or Unsafe Claims

Do not currently claim:

- Completed disaster damage segmentation.
- xBD or xBD-S12 end-to-end training/evaluation.
- Building damage-level prediction.
- DS was invented in this project.
- OSCD binary change proves disaster damage performance.
- Old residual-stack priors prove paper-faithful DS works.
- Current global pixel DS preserves spatial structure during fitting.

