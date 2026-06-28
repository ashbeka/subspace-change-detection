# H-A 1a gating — invariance novelty CLOSED (IR-MAD owns it); synthetic unreliable

Date 2026-06-19. Code: temporal/experiments/hsi_hard_nuisance.py. Data: Salinas synthetic bitemporal CD.

## Result (trustworthy part: class-swap change) [experiment-evidence]
IR-MAD AUC under nuisance (large class-swap change): affine 0.98, nonlinear 1.00, spatial 1.00 at alpha=0.4.
=> IR-MAD (the established iteratively-reweighted affine-invariant CD method) does NOT degrade under
nonlinear or spatially-varying nuisance for distinct change. The 1a hypothesis ("linear invariant-residual
degrades under harder nuisance, opening room for a kernel/local method") is FALSIFIED for distinct change.
My one-pass SFA-CD is weaker than IR-MAD (lacks iterative reweighting) -> last turn's "SFA-CD wins" was
"an invariant-residual method wins"; IR-MAD is the stronger instance. Invariance-DETECTION novelty: CLOSED.

## Contaminated part (do not use): subtle additive-Gaussian change
IR-MAD scored 0.27-0.53 (below chance, worsening as change magnitude grows) EVEN WITH NO NUISANCE -> an
inverted-score bug in my IR-MAD's variate-variance normalization for additive change (it is correct for
class-swap: AUC 1.000). Also, additive isotropic change is trivially CVA-detectable (CVA~1.0), so it does
not isolate invariance. SFA-CD's 1.0 there = magnitude-sensitivity, not an invariance advantage. CONCLUSION:
synthetic nuisance+change experiments are NOT reliably adjudicating H-A -- outcomes swing with arbitrary
change/nuisance design and implementation details. A trustworthy H-A answer requires the REAL bitemporal HSI
benchmark (real change + real nuisance), which is data-blocked.

## Honest strategic implication
H-A's DETECTION angle is dominated by IR-MAD (established, near-perfect under hard nuisance for distinct
change) and cannot be reliably tested on synthetic. Across the whole project, the subspace/geometry direction
has not produced a single reliable positive beating established methods on real data (bi-temporal DS failed;
temporal DS 3x real failures; hyperspectral spectral-DS proxy negative; invariance-residual owned by IR-MAD).
This is a consistent signal. The honest, defensible deliverables are now: (1) the DIAGNOSTIC paper (Top-10
#10 -- when does subspace geometry help CD, and when is it dominated by SAM/CVA/IR-MAD) -- most certain;
(2) PIVOT to H-B (change-trajectory characterization via 2nd-order DS -- a DIFFERENT question IR-MAD does not
address) or H-C (geometry on deep features); (3) the real HSI-CD benchmark as the only trustworthy H-A test.
Do NOT keep generating synthetic H-A experiments (drilling).
