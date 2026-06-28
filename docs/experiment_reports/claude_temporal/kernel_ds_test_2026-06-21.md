# Kernel / nonlinear Difference Subspace (RFF) — CLOSED, with a clean diagnostic detail (2026-06-21)

Script: `temporal/experiments/kernel_ds_test.py` · `temporal/outputs/kernel_ds_test/`
The untested variant: nonlinear DS via Random Fourier Features (RBF-kernel lift) → linear subspace in phi-space.

## Result (AUC change vs no-change)
| | kernel_DS | linear_DS | corr | CVA | patch_mean_SAM |
|---|---|---|---|---|---|
| A: real Hermiston (amplitude-dominated) | 0.966 | 0.549 | 0.904 | **0.985** | 0.767 |
| B: controlled distributional (mean-preserving) | 0.536 | 0.524 | **0.816** | 0.690 (patch-mean) | — |

## Verdict — redundant, but informative
- The nonlinear RFF lift jumps the subspace from 0.55 (linear) to 0.97 on amplitude-dominated change — it
  RESTORES the amplitude sensitivity the linear angle-subspace discards. This cleanly confirms WHY linear-DS
  failed (it throws away amplitude).
- BUT kernel-DS still does NOT beat the simple nulls: plain CVA (0.985) edges it on Hermiston; the correlation
  matrix (0.816) crushes it on the distributional change (where kernel-DS is near-chance 0.536).
- So the nonlinear lift makes the subspace CATCH UP to the simple baseline on amplitude change, never surpass it,
  and it fails on distributional change. The redundancy pattern holds for the nonlinear variant too.
