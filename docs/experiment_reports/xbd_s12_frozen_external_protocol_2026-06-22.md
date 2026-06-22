# xBD-S12 Frozen External Validation Protocol

## 1. Question

Does the OSCD Band-Image Difference Subspace result transfer to an independent
multispectral disaster dataset, and does canonical DS contribute changed-area
ranking information beyond spatial PCA, IR-MAD, and a rank-matched
cross-reconstruction substitute?

This protocol was frozen before inspecting any xBD-S12 method output.

## 2. Dataset And Split

- Dataset: official xBD-S12 release, Zenodo record `18960454`.
- Input: co-registered pre/post Sentinel-2 L2A patches, `128 x 128`, twelve
  bands: `B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12`.
- Split: the authors' event-disjoint split from Hafner et al. (2025), using
  the repository's `TRAIN_DISASTERS` and `TEST_DISASTERS` lists.
- Training events may be used only for validating data handling and reporting
  a predeclared method. Test-event labels must not select rank, smoothing,
  fusion membership, preprocessing, or thresholds.

This is building-damage evidence, not ordinary OSCD binary land-cover change.

## 3. Label Views

The original mask values are:

```text
0 background
1 intact building
2 minor damage
3 major damage
4 destroyed
5 unclassified
6 no data
```

Two predeclared evaluations are required:

1. **Full-scene damage retrieval, primary:** positives are `2-4`; negatives
   are `0-1`; values `5-6` are ignored. This tests whether a label-free score
   can rank damaged building pixels in the complete patch.
2. **Building-conditional damage retrieval, secondary:** positives are `2-4`;
   negatives are intact-building pixels `1`; background and values `5-6` are
   ignored. This answers whether change evidence separates damage from intact
   buildings when a building footprint is available.

Building localization (`1-4` versus `0`) is diagnostic only because it is not
the project's change-detection question.

The official categorical downsampling rule is retained. A three-pixel
Euclidean-distance ignore buffer on both sides of labeled building boundaries
is reported in a separate sensitivity run because it changes the evaluation
support; unbuffered results remain the primary transparent view.

## 4. Frozen Preprocessing And Rank

- Use the authors' official `normalization_stats`: per-band 1st/99th
  percentile min-max normalization, clipped to `[0,1]`, applied identically
  to both dates. The Zenodo archive unexpectedly omits the documented
  `normalization.json`; the identical statistics are recovered from the
  official Hugging Face config for `prs-eth/xbd-s12_loc_seed1` and recorded by
  the preparation manifest. The downloaded config SHA-256 is
  `f85fa0b67dd52657c419b141503776c77ae0804b11017eb3c0a3ddc13cc05585`.
- A pixel is input-valid only when all pre/post band values are finite.
- The downloaded Sentinel archive does not redistribute the original xBD
  no-data raster. For the first frozen run, exclude entire patches whose
  metadata reports `N_px_no_data > 0`; do not guess the missing support from
  method scores or labels. A later exact-nodata rerun may use the original VHR
  rasters as a sensitivity check.
- Do not use labels for normalization, masking, rank, or score calibration.
- Band-Image DS receives `X_t in R^(N x 12)`, where each column is one flattened
  band-image sample and rows retain fixed spatial positions.
- Band-Image rank is `11`, the maximum nonzero centered rank from twelve
  band-image samples. This is the direct 12-band analogue of the frozen OSCD
  rank-12 construction.
- Pixel PCA-diff uses the existing fixed-rank policy without label tuning.
- Spatial smoothing is Gaussian `sigma=1`; multiscale PCA uses
  `sigma={0,1,2}`. These are copied from the completed OSCD pressure test.

## 5. Frozen Methods

Individual maps:

1. raw spectral L2 / CVA;
2. spectral angle between paired twelve-band pixel vectors;
3. PCA-diff;
4. smoothed PCA-diff (`sigma=1`);
5. multiscale PCA-diff (`sigma=0,1,2` mean);
6. repaired IR-MAD;
7. Band-Image canonical DS projection magnitude, rank `11`;
8. Band-Image symmetric excess cross-reconstruction, rank `11`;
9. normalized spatial Gram row distance;
10. projector-row distance.

Label-free percentile-rank fusions:

1. smoothed PCA + Band-Image DS;
2. smoothed PCA + IR-MAD;
3. smoothed PCA + Band-Image DS + IR-MAD;
4. smoothed PCA + cross-reconstruction + IR-MAD.

Fusion weights are equal. No xBD-S12 labels may fit weights.

## 6. Metrics And Statistical Unit

- Primary metric: average precision (AP), because damage pixels are sparse.
- Secondary: AUROC, best F1/IoU as an oracle diagnostic, Otsu F1/IoU,
  prevalence, runtime, and qualitative maps.
- Report patch-level, disaster-event-level, and global pixel summaries.
- Statistical independence is defined by disaster event, not pixel or patch.
- Paired comparisons use event-cluster bootstrap intervals and event-level
  wins. Patch bootstrap may be shown only as a secondary stability diagnostic.
- Events and patches without positive damaged pixels remain relevant for
  false-positive/triage analysis; per-patch AP is not averaged blindly when AP
  is undefined.

## 7. Decision Gates

External support for the DS hypothesis requires all of the following:

1. Band-Image DS beats its cross-reconstruction substitute in the same
   direction as OSCD on the event-disjoint test set.
2. The DS three-way fusion beats the cross-reconstruction three-way fusion.
3. The effect is not caused by one disaster event or a label/mask artifact.
4. Maps reveal coherent damage-related evidence rather than cloud, water,
   registration, or broad radiometric pseudo-change alone.

Improvement over smoothed PCA alone is desirable but not required to establish
distinct complementarity. If DS fails the matched controls, the OSCD finding is
dataset-specific and the publication direction must be narrowed or changed.

## 8. Source Trail

- Dataset/code: `https://github.com/prs-eth/xbd-s12`
- Release: `https://zenodo.org/records/18960454`
- Official model/config fallback:
  `https://huggingface.co/prs-eth/xbd-s12_loc_seed1`
- Paper: Dietrich et al., *The Potential of Copernicus Satellites for Disaster
  Response: Retrieving Building Damage from Sentinel-1 and Sentinel-2*, arXiv
  `2511.05461`, ISPRS Congress 2026.
- Project method source: Fukui and Maki, TPAMI 2015 canonical Difference
  Subspace construction, with the Band-Image sample adaptation documented in
  `notes/methods.md`.
