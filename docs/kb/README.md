# Knowledge Base — per-paper notes (research-mining deliverable)

Created 2026-06-21 (research-mining program, `docs/CRUX_PROMPT.md`). Purpose: a structured, citable
distillation of the provided ~20 papers + the core subspace/CD literature, each reduced to **method+math,
the signal/subspace idea, the novelty, HOW THEY JUSTIFY IT, the BASELINES they compare to, and the concrete
transfer to hyperspectral × temporal × change detection (CD)**. This is the evidence base under
`docs/research/challenges_ranked.md`, `closest_methods_novelty.md`, and `synthesis_specific_tasks.md`.

## Files
- [`01_core_subspace_geometry.md`](01_core_subspace_geometry.md) — DS / GDS / KDS / 2nd-order DS / Kanai
  SSA-DS / MSM / the "Geometry of subspace set" overview. The lab's home mechanisms.
- [`02_temporal_sequence_warping_ssa.md`](02_temporal_sequence_warping_ssa.md) — SFA / SFS, RTW, TRCCA,
  S3CCA, SSA, product-Grassmann, temporal-stochastic tensor, Grassmann-SSA, SLS, n-mode GDS. The
  sequence/temporal/warping mechanisms.
- [`03_remote_sensing_cd_and_hsi.md`](03_remote_sensing_cd_and_hsi.md) — SFA-CD, spatial-context CD, PWTT
  battle damage, the Anomalous-Change-Detection family, subspace CD for HSI, low-rank tensor denoising, the
  multitemporal-HSI-CD reviews, the "awesome-RS-CD" traditional-method roster.

## Reading-depth legend (honesty about source depth — per the reasoning directive)
- `[P]` **primary-read** in this session (key equations/figures read directly).
- `[P-internal]` primary-read previously in this project; verified extraction lives in `docs/METHOD.md`,
  `docs/SUBSPACE_CONSTRUCTION_LEDGER.md`, or an experiment report.
- `[S]` **secondary** — abstract + web-search summary + cross-references; NOT yet read page-by-page. Treat its
  finer claims as provisional and verify before citing a specific number.

## The one-paragraph synthesis these notes support
Across every paper, the recurring engine is the same three-part composition (`docs/CD_TAXONOMY.md`
Improvement 1): **REPRESENTATION (subspace of a *set*) × COMPARISON-OPERATOR (canonical angles / difference
subspace / geodesic) × DECISION**. The papers that *beat* simple baselines do so by climbing the
**invariance ladder** (model the common/slow/normal component, flag the residual) — SFA, GDS-common, Kanai's
reference normal-DS `D_N`, RTW's warp-invariant subspace, ACD's coordinate-invariant background predictor.
The papers that merely *re-describe* a set (plain DS magnitude, min canonical angle) tie or lose to scalars.
The transferable levers for satellite CD are therefore (a) **genuine multi-dimensionality** of the set
(so DS ≠ spectral angle — only real at 100+ bands), (b) a built-in **invariance** the scalar nulls lack
(temporal warp; intra-material variation; the learned normal), and (c) **characterization/attribution**
(which modes/bands/change-type) that a scalar structurally cannot output.
