# STATUS — claude/temporal-ds (resume here)

_Last updated 2026-06-18. One-glance state so any session (new subscription, new chat) can continue._

## Where my work lives
- **Branch:** `claude/temporal-ds` · **Worktree:** `E:/research_projects/sccd-claude` (separate from Codex's
  main dir `E:/research_projects/subspace-change-detection`). Run Python with the main venv:
  `cd E:/research_projects/sccd-claude && E:/research_projects/subspace-change-detection/.venv/Scripts/python.exe -m temporal.experiments.<name>`
- **GEE:** live via `ee.Initialize(project='subspace-change-detection')` (one-time `earthengine authenticate` already done).

## The story so far (honest)
1. Pivoted bi-temporal DS → **temporal DS on Sentinel-2 time series** (real "set" = a window of dates).
2. **Verified positive (synthetic):** multivariate-SSA full DS beats conventional min-angle on *distributed
   multi-mode dynamics* change, noise-robust, 100% band attribution. Survived adversarial verification.
3. **Honest limit:** first-order DS ≡ spectral angle unless the change is genuinely multi-dimensional dynamics.
4. **Real-data tests on Tenerife 2023 fire: BOTH negative** — onset (NBR wins; DS smears) and recovery
   (`d_pre` doesn't separate burn from control). A fire is a sharp shift, not a dynamics change.
5. **Current plan (Option A):** test DS on a real phenomenon that IS a phenology/dynamics change
   (agriculture: crop conversion / irrigation / abandonment; or deforestation→regrowth), ideally with REAL
   LABELS. A background workflow is mining notes+web for the best labeled time-series dataset + method
   strengthening. Failsafe = the diagnostic paper (`docs/DRAFT_DIAGNOSTIC_PAPER_ACCV.md`).

## To resume
Say: **"continue Option A on claude/temporal-ds"**. Read order: this file → `docs/METHOD.md` →
`docs/experiment_reports/*2026-06-18.md` (L1, multiband, real_gate0, real_recovery) →
`temporal/configs/L2b_recovery_preregistered.json`. Memory files in `~/.claude/.../memory/` auto-load.

## Key code
`temporal/subspace.py` (DS algebra), `temporal/dynamics.py` (curves), `temporal/gee_fetch.py` (S2 fetch),
`temporal/experiments/{multiband_gate,real_gate0,real_recovery}.py`, `docs/METHOD.md`, `docs/figures/`.
