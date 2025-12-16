# Slide Edit Notes (What to change in the PNG/PPT)

These notes align the deck with the latest completed runs:
- Phase 1 full baselines: `phase1/outputs/oscd_saved_full/oscd_eval_summary.csv`
- Phase 2 extended priors: `phase2/outputs/runs_gpu_150ep_20251215_233309/oscd_priors_ablation_summary_extended.csv`

---

## High-priority fixes (do these first)

### Slide 2 (Outline)
- Add an explicit Objective bullet at the top (1 sentence).
- Optional: add a Contributions bullet (2-3 items) if your audience expects it.

### Slide 4 (Difference Subspace)
- Replace "between two class subspaces" wording with: "between two subspaces (e.g., pre vs post)."
- Add one line:
  - "Change score per pixel: project (post-pre) onto DS basis; higher energy => more change."

### Slide 11 (Phase 1 results)
Your current bullet "IR-MAD performs worst" is not accurate for the full run.

Update text/table to match OSCD test AUROC:
- PCA-diff: 0.813
- pixel-diff / CVA: 0.756
- DS projection: 0.755
- IR-MAD: 0.704
- Celik: 0.649
- DS cross-residual: 0.556 (worst here)

### Slide 13 (Phase 2 pipeline with priors)
- Add one explicit line on the slide:
  - "Raw input = 26 channels (13 pre + 13 post). Raw+DS = 27 channels (add DS score map)."

### Slide 15 (Experiments / configs)
Your table is currently the "core set" only. Add a small "Extended priors" row/list:
- raw + pixel-diff
- raw + DS cross-residual
- raw + Celik
- raw + IR-MAD

### Slide 16 (Phase 2 results)
Your current slide claims "raw best IoU/F1" and uses older numbers. Replace with the latest results.

Minimum recommended table for the main talk (keep it readable):
- U-Net raw: IoU 0.223, F1 0.343, AUROC 0.869
- U-Net raw + DS projection: IoU 0.273, F1 0.401, AUROC 0.874 (best IoU/F1)
- U-Net raw + Celik: IoU 0.260, F1 0.382, AUROC 0.872 (best PR-AUC 0.448)
- Fusion raw + DS + PCA: IoU 0.243, F1 0.360, AUROC 0.888 (best AUROC)

Put the other priors (pixel-diff, PCA-diff, DS cross-residual, IR-MAD) in a backup slide or as a small "didn't help" note.

### Slide 19 (Discussion and conclusion)
Replace "raw Sentinel-2 segmentation gives the best IoU/F1" with:
- "Best IoU/F1 in this run is U-Net raw + DS projection prior."
Add one limitation bullet:
- "Single seed + fixed threshold; need multi-seed + threshold tuning for final claims."

---

## Optional improvements (if you have time)

### Add a new Objective / Contributions slide (after Slide 1)
If your audience is not already familiar with your project, adding a dedicated slide helps.

### Add a References slide (near the end or as backup)
Use `presentation/REFERENCES.md` as your source list.

### Add one "method in one equation" backup slide
Just show:
- delta = x_post - x_pre
- score = || D^T delta ||^2 where D is a DS basis

