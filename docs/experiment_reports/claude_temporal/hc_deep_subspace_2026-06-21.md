# H-C — subspace geometry on deep features: CLOSED (2026-06-21)

Script: `temporal/experiments/hc_deep_subspace.py` · `temporal/outputs/hc_deep_subspace/`
No pretrained RS foundation model offline → deep features = an autoencoder (242→32) trained on Hermiston;
latent z = encoder(x). Real Hermiston bitemporal. Killer null: does the latent SUBSPACE beat the latent
mean/correlation (geometry over deep features)?

## Result
Per-pixel detection AUC: raw_CVA **0.980**, latent_CVA **0.980**, raw_SAM 0.844, latent_SAM 0.573.
Patch detection AUC: latent_corr **0.891** > raw_patchmean_SAM 0.815 > latent_SUBSPACE 0.672 >
latent_patchmean_SAM 0.583 > raw_SUBSPACE 0.504.

## Verdict — CLOSED (both sub-questions fail)
1. **Deep features do not help over raw** here: latent_CVA = raw_CVA = 0.98 (the amplitude-dominated change is
   equally visible in raw and latent; the AE just compresses).
2. **The latent SUBSPACE (0.67) is redundant/worse than the latent CORRELATION (0.89)** and even raw-patch-mean
   SAM (0.82): geometry adds NOTHING over the deep features. Caveat: a trained AE is not a pretrained foundation
   model, but the geometry-redundancy holds regardless of the feature source — the subspace lost to the latent
   correlation. Any feature-source win would be the FEATURES', not the geometry's (as pre-stated).

## CAMPAIGN COMPLETE — every planned lead closed
Bet 1 (N<<B distributional + registration), Bet 2 (recovery trajectory), H-C (deep-feature geometry) — all
tested rigorously, all closed. Together with detection / characterization / warp / material-subspace /
orientation / attribution, subspace geometry is REDUNDANT or WORSE than a simpler standard statistic
(spectral angle, mean-vector diff, CVA, IR-MAD, harmonic deseasonalization, DTW, correlation matrix,
patch-mean SAM) in EVERY tested cell, on real data, against the correct nulls, corroborated by two independent
research-minings. There is no open positive cell for subspace-as-a-better-CD-method.

THE CONTRIBUTION is the DIAGNOSTIC: a rigorous, real-data-validated, pre-registered-null characterization of
when/why subspace geometry is redundant for satellite change detection — the REPRESENTATION × OPERATOR ×
DECISION map. Next phase (per the user's plan): consolidate the diagnostic + prepare the seminar.
