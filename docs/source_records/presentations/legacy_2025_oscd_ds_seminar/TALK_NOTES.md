# Seminar Talk Notes (DS + OSCD Segmentation)

Use this as a memory aid (not a full script). It is written as "what to say out loud" in plain language.

Latest artifacts used for numbers:
- Phase 1 full baselines (OSCD AUROC): `phase1/outputs/oscd_saved_full/oscd_eval_summary.csv`
- Phase 2 extended priors (OSCD mean metrics): `phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv`

---

## 1) The 20-second opening

"We want label-efficient change/damage mapping from satellite imagery. I use Sentinel-2 pre/post images. Phase 1 computes interpretable unsupervised change maps using Difference Subspace (DS) and classical baselines. Phase 2 trains a supervised U-Net-style model on OSCD change masks, and tests whether adding DS maps as extra input channels improves segmentation."

---

## 2) What "raw" means (and why we talk about channels)

### The basic data (OSCD)
- Each OSCD sample is two images of the same place: a pre image and a post image.
- Sentinel-2 is multispectral: each pixel has 13 band values (not just RGB).
- GT is a binary mask: change vs no-change (as defined by OSCD).

### In CNNs, "channels" = features per pixel
- An input patch is a tensor shaped like (channels, height, width).
- RGB has 3 channels. Sentinel-2 has 13 channels.

### "Raw" in this project (important)
"Raw" does NOT mean "raw sensor values straight from the satellite".

In our Phase 2 configs, raw means:
- Use the normalized Sentinel-2 bands directly as inputs.
- We stack BOTH pre and post bands (not only the difference):
  - raw = concat(pre_bands, post_bands)
  - channels = 13 + 13 = 26
- We normalize bands (z-score per band) and ignore invalid pixels via a valid mask.

### "Raw + DS" (why it becomes 27 channels)
In Phase 1 we compute DS score maps (one score per pixel).
That DS score map is like adding one extra grayscale image as a channel.

So:
- raw only: 26 channels (13 pre + 13 post)
- raw + DS projection: 27 channels (26 raw + 1 DS map)
- raw + PCA-diff: 27 channels (26 raw + 1 PCA-diff map)
- raw + DS + PCA-diff: 28 channels (26 raw + 2 priors)
- raw + Celik / IR-MAD / pixel-diff: 27 channels (26 raw + 1 prior map)

One sentence you can say:
"Raw+DS means we feed the network the original pre/post spectral bands plus one extra channel: an unsupervised DS change score map."

---

## 3) What training actually does (end-to-end, no assumptions)

### Step 0: Build training examples
1. Load an OSCD city tile: pre image, post image, GT mask.
2. Preprocess:
   - Build a valid mask (ignore missing pixels).
   - Normalize bands (z-score per band).
3. Build the input tensor x:
   - raw only: x has 26 channels
   - raw+DS: x has 27 channels (add DS map)
4. Cut each city tile into many patches (256x256).

Each training example is:
- x: (C, 256, 256) where C is 26/27/28 depending on the experiment
- y: (1, 256, 256) the GT change mask

### Step 1: Forward pass (make predictions)
For a batch of patches, the model outputs logits of shape (B, 1, 256, 256).
After sigmoid, we get p(change) in [0, 1] per pixel.

### Step 2: Loss (how we tell the model it is wrong)
We compare prediction vs GT and compute a loss:
- BCEWithLogits (pixel-wise classification loss)
- Dice loss (helps overlap + class imbalance)

Loss is one number: "how wrong the model is on this batch".

### Step 3: Backprop + optimizer (how the model improves)
- Backprop computes gradients: how each weight should change to reduce loss.
- AdamW updates weights using those gradients.

Training means repeating:
predict -> compute loss -> update weights
many times.

### Step 4: Epochs and checkpoints
- One epoch = the model has seen all training patches once.
- We train for many epochs (150 in the final run).
- We save checkpoints: best.ckpt (best validation) and last.ckpt (latest).

---

## 4) How we evaluate (IoU/F1/AUROC/PR-AUC)

On test cities:
1. Run the trained model to get probability maps p(change) for each tile.
2. Threshold at 0.5 to get a binary mask.
3. Compute:
   - IoU / F1: quality of the final binary mask (threshold-dependent)
   - AUROC / PR-AUC: ranking quality of probabilities (threshold-free)

Global-scale interpretation sentence:
"All reported test means are averaged over the 10 OSCD test cities (each city weighted equally)."

---

## 5) Slide-by-slide: minimal talking points

### Slide 4 (DS)
- PCA gives a low-dimensional subspace for pre pixels and for post pixels.
- DS captures directions where these subspaces differ.
- DS score map = how strongly (post-pre) aligns with DS directions.

### Slide 11 (Phase 1, OSCD test AUROC)
- PCA-diff 0.813 (best AUROC), pixel-diff/CVA 0.756 ~= DS projection 0.755, IR-MAD 0.704, Celik 0.649, DS cross-residual 0.556.
- Transition: "Now we use these maps as priors for supervised segmentation."

### Slide 13 (Phase 2 pipeline)
- Raw = 26 channels: 13 pre + 13 post.
- Raw+DS = 27 channels: add DS map as one extra channel.
- DS is a feature, not a label.

### Slide 16 (Phase 2 results, OSCD test)
- Best IoU/F1: raw+DS projection (IoU 0.273, F1 0.401).
- Best AUROC: fusion model (AUROC 0.888).
- Best PR-AUC: raw+Celik (PR-AUC 0.448).
- Caveat: single seed + fixed threshold (0.5).

### Slide 18 (why priors can help/hurt)
- DS/PCA highlight broad spectral change; OSCD labels only certain changes.
- So priors can improve ranking/explainability, but may add false positives for strict IoU/F1.

---

## 6) Fast glossary

- Channel: one feature map per pixel (like RGB channels, but here we have 26/27/28).
- Batch: several patches processed together.
- Epoch: one full pass over the training patches.
- Loss: single number measuring error (we minimize it).
- Optimizer (AdamW): updates weights using gradients.
- Backprop: computes gradients (how to change weights to reduce loss).

