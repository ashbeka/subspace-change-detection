# Phase 1 CLI Cheat Sheet (PowerShell)

Assume:
- You are in repo root: `E:\research_projects\DS_damage_segmentation`
- Root venv is activated: `.\.venv\Scripts\Activate.ps1`
- Data paths:
  - OSCD at `data\OSCD`
  - MultiSenGE at `data\MultiSenGE\s2`

## Core eval/visualization (from repo root)

- OSCD eval (residual DS + baselines):  
  `python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_default.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_run`
- OSCD eval + save change maps:  
  `python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_priors_fast.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_saved_priors_fast --save_change_maps`
- OSCD eval (eig DS, no window, no Celik):  
  `python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_variant_eig.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_eig_nowindow --disable_celik --no_window`
- OSCD eval (eig DS, same settings as residual DS: window + Celik enabled):  
  `python -m phase1.eval.run_oscd_eval --config phase1/configs/oscd_variant_eig.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_eig_full`
- MultiSenGE DS visualization:  
  `python -m phase1.eval.run_multisenge_viz --config phase1/configs/multisenge_default.yaml --multisenge_root data/MultiSenGE/s2 --output_dir phase1/outputs/multisenge_viz`
- OSCD summary figures (all cities):  
  `python -m phase1.eval.visualize_oscd_examples --config phase1/configs/oscd_priors_fast.yaml --oscd_root data/OSCD --output_dir phase1/outputs/oscd_figs_all --cities all --metrics_json phase1/outputs/oscd_saved_priors_fast/oscd_eval_results.json`

### Common variants (flags)
- Disable Celik: add `--disable_celik`
- Disable sliding window: add `--no_window`
- Change window size / stride (if windowing enabled): `--window_size 64 --window_stride 32`
- Turn on IR‑MAD: set `methods.ir_mad: true` in the YAML or use a config that enables it
- Save change maps: `--save_change_maps` (writes to `<output_dir>/oscd_change_maps/...`)

## Helper scripts (from repo root)

- OSCD overview grid (24×3 pre/post/GT):  
  `python phase1/scripts/make_oscd_overview_grid.py`
- MultiSenGE overview grid (pre/post/DS for sample patches):  
  `python phase1/scripts/make_multisenge_overview_grid.py`
- OSCD band grayscale PNGs (per band, pre/post):  
  `python phase1/scripts/make_oscd_band_pngs.py`
- OSCD band pseudocolor PNGs (wavelength-aware):  
  `python phase1/scripts/make_s2_band_pseudocolor.py`
- Pre/post “cube” figure (stacked bands):  
  `python phase1/scripts/make_pre_post_cubes_fig.py`
- Quick RGB preview of a TIFF (MultiSenGE band order):  
  `python phase1/scripts/tiff_to_rgb.py data/MultiSenGE/s2/<file>.tif "Title"`

## Notes
- For all OSCD commands, ensure `data/OSCD` follows the expected structure (images + labels).
- For MultiSenGE commands, ensure `data/MultiSenGE/s2` exists with TIFF patches.
- Change `--output_dir` as needed; `phase1/outputs/` is ignored by git.
