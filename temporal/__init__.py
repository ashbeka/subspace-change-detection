"""Temporal Difference Subspace analysis for satellite image time series.

Research pivot (see docs/DESIGN_TEMPORAL_DS_ACCV2026.md): move Difference Subspace from the
bi-temporal single-pair setting (where the "set of samples" is fake) to the temporal axis of a
dense Sentinel-2 time series, where a window/sequence of dates is a genuine set.

source -> math object -> satellite adaptation -> code path -> verification -> allowed claim:
  Fukui & Maki, TPAMI 2015 (first-order DS) + Fukui et al. 2024 arXiv:2409.08563 (second-order DS,
  geodesic velocity/acceleration on the Grassmann manifold) + Kanai et al. 2023 arXiv:2303.17802
  (DS between SSA signal subspaces for anomaly detection).
  Reference code cross-checked: references/reference_code/MagTool-main (Jang), cv_motion3d (Soto-san).
"""
