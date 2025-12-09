# Phase 1 CLI Cheat Sheet (PowerShell)

Assume:
- You are in repo root: `E:\research_projects\DS_damage_segmentation`
- Root venv is activated: `.\.venv\Scripts\Activate.ps1`
- Data paths:
  - OSCD at `phase1\data\raw\OSCD`
  - MultiSenGE at `phase1\data\raw\MultiSenGE\s2`

For core eval scripts, `cd phase1` first so the package imports work.

## Core eval/visualization (from repo root)

```powershell
cd phase1
```

- OSCD eval (residual DS + baselines):  
  `python -m eval.run_oscd_eval --config configs/oscd_default.yaml --oscd_root data/raw/OSCD --output_dir outputs/oscd_run`
- OSCD eval + save change maps:  
  `python -m eval.run_oscd_eval --config configs/oscd_default.yaml --oscd_root data/raw/OSCD --output_dir outputs/oscd_saved --save_change_maps`
- OSCD eval (eig DS, no window, no Celik):  
  `python -m eval.run_oscd_eval --config configs/oscd_variant_eig.yaml --oscd_root data/raw/OSCD --output_dir outputs/oscd_eig_nowindow --disable_celik --no_window`
- OSCD eval (eig DS, same settings as residual DS: window + Celik enabled):  
  `python -m eval.run_oscd_eval --config configs/oscd_variant_eig.yaml --oscd_root data/raw/OSCD --output_dir outputs/oscd_eig_full`
- MultiSenGE DS visualization:  
  `python -m eval.run_multisenge_viz --config configs/multisenge_default.yaml --multisenge_root data/raw/MultiSenGE/s2 --output_dir outputs/multisenge_viz`
- OSCD summary figures (all cities):  
  `python -m eval.visualize_oscd_examples --config configs/oscd_default.yaml --oscd_root data/raw/OSCD --output_dir outputs/oscd_figs_all --cities all --metrics_json outputs/oscd_run/oscd_eval_results.json`

### Common variants (flags)
- Disable Celik: add `--disable_celik`
- Disable sliding window: add `--no_window`
- Change window size / stride (if windowing enabled): `--window_size 64 --window_stride 32`
- Turn on IR‑MAD: set `methods.ir_mad: true` in the YAML or use a config that enables it
- Save change maps: `--save_change_maps` (writes to `outputs/oscd_change_maps/...`)

## Helper scripts (from repo root)

```powershell
cd phase1
```

- OSCD overview grid (24×3 pre/post/GT):  
  `python scripts/make_oscd_overview_grid.py`
- MultiSenGE overview grid (pre/post/DS for sample patches):  
  `python scripts/make_multisenge_overview_grid.py`
- OSCD band grayscale PNGs (per band, pre/post):  
  `python scripts/make_oscd_band_pngs.py`
- OSCD band pseudocolor PNGs (wavelength-aware):  
  `python scripts/make_s2_band_pseudocolor.py`
- Pre/post “cube” figure (stacked bands):  
  `python scripts/make_pre_post_cubes_fig.py`
- Quick RGB preview of a TIFF (MultiSenGE band order):  
  `python scripts/tiff_to_rgb.py data/raw/MultiSenGE/s2/<file>.tif "Title"`

## Notes
- For all OSCD commands, ensure `phase1/data/raw/OSCD` follows the expected structure (images + labels).
- For MultiSenGE commands, ensure `phase1/data/raw/MultiSenGE/s2` exists with TIFF patches.
- Change `--output_dir` as needed; `phase1/outputs/` is ignored by git.
