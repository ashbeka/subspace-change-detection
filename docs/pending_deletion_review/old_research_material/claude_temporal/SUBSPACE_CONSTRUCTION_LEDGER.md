# SUBSPACE CONSTRUCTION LEDGER

Per Sensei's standing feedback ("understand exactly how we build the subspaces"). For EVERY experiment, record:
what the subspace IS, the data matrix it is built from, how it is built, and what it is supposed to REPRESENT.
Verified against the lab papers (read 2026-06-20): Second-order DS (arXiv:2409.08563), Time-series Anomaly DS
between Signal Subspaces (arXiv:2303.17802), Difference Subspace & its Generalization (TPAMI 2015).

## 1. Verified core constructions (faithful to the papers + MagTool)
- **Set → PCA/autocorrelation subspace.** Data matrix X = (ambient_dim × n_samples), samples as COLUMNS. Subspace
  = top-k left singular vectors. `center=False` (autocorrelation PCA) → top component ≈ dominant *material*
  direction of the set; `center=True` → directions of *variation*. REPRESENTS: the span of the dominant
  directions present in a SET of vectors (e.g. the spectra in a patch).
- **Canonical angles** between span(B1), span(B2): singular values of B1ᵀB2 = cosθ_i ∈[0,1]. (`canonical_cosines`.)
- **First-order Difference Subspace (DS).** Analytical (Fukui-Maki TPAMI'15; 2nd-order paper Eq.4): eigenvectors
  of G = P1+P2 (sum of orthogonal projectors) with eigenvalue in (0,1). REPRESENTS: the directions in which the
  two subspaces DIFFER. **Magnitude** Mag(D) = Trace(2(I−Σ)) = Σ 2(1−cosθ_i). (`difference_subspace`,`magnitude`.)
- **Karcher / principal-component subspace M(S1,S2).** Eigenvectors of G=P1+P2 with eigenvalue >1 (intersection
  dirs have eigenvalue =2, included). REPRESENTS: the "mean" subspace of the two. (`karcher_subspace`.)
- **Second-order DS** (Def 3.1/3.2): D²(S1,S2,S3) = D(S2, M(S1,S3)); Mag(D²) = Mag( D( S2, Karcher(S1,S3) ) ).
  REPRESENTS: ACCELERATION = how far the middle subspace S2 deviates from the constant-velocity geodesic
  midpoint of its neighbours. Constant-velocity (geodesic) motion ⇒ D²=0. **VERIFIED my `second_order_magnitude`
  matches this exactly.** Geodesic decomposition (`second_order_decomposed`) splits it into along-geodesic
  (speed change) vs orthogonal (off-geodesic) components.
- **SSA signal subspace** (Kanai'23 Eq.1–2). 1D series h(t) → trajectory (Hankel) matrix H_t ∈ R^{w×M},
  H[r,c]=h(t−w−M+2+r+c) (window width w, M windows). Signal subspace P_t = top-r eigenvectors of H_tH_tᵀ.
  REPRESENTS: the dominant temporal/oscillatory structure of the signal window (captures periodicity/trend).
- **Kanai DS-between-signal-subspaces change score** (the lab's actual CD-relevant method — NOT yet implemented
  here). Past P_{t−τ}, present P_t (they OVERLAP → R-dim intersection; use generalized DS = eigenvectors of
  P_{t−τ}+P_t with eigenvalue in (0,1−δ), excluding intersection). Learn a **non-anomalous reference DS `D_N`**
  from a normal period (principal subspace of the normal DSs). Score = direction index δ(D_in,D_N)=(1/c)Σ(1−λ_i)
  [canonical cos between current DS and D_N] × magnitude index β=(μ(D_in)−μ(D_N))², μ(D)=Σ log cosθ_i.
  REPRESENTS: departure of the *current change-direction* from the *normal change-direction* — i.e. "model the
  normal dynamics, flag the residual." This is the invariance lever.

## 2. Per-experiment construction log
| exp | subspace object | data matrix | how built | represents |
|---|---|---|---|---|
| bitemporal DS (13-band S2) | per-pixel local subspace | spectral neighbourhood | PCA | local spectral identity (≈ rank-1 → DS≡SAM) |
| temporal DS (S2 series) | per-date window subspace | dates×bands window | PCA / M-SSA | the state at a date — CRUDE (see §3) |
| HSI spectral-SSA | per-pixel spectral subspace | Hankel along wavelength | SSA | spectral structure of one pixel |
| H-A SFA-CD | SFA slow subspace | bitemporal pixels | gen. eig (Cder,Cov) | the slow/invariant component |
| H-B-1/1b (patch) | patch material subspace | patch spectra (B×N) | uncentered PCA k=2 | dominant materials in a patch |

## 3. CORRECTIONS (2026-06-20 — found by reading the papers; both affect prior "closures")
1. **2nd-order DS "never fired" is NOT a bug.** My implementation is faithful (verified). It read ~0 because my
   constructions had NO genuine subspace acceleration: hb1 subspaces barely moved; hb1b/C2 was a UNIFORM
   rotation (constant velocity ⇒ zero acceleration by definition). 2nd-order DS remains UNTESTED for its real
   purpose (detecting a *change in the rate* of subspace motion). Do not cite it as "dead."
2. **My "temporal DS failed on S2" never implemented the lab's actual method.** I used raw date-window subspace
   velocity. Kanai'23's method = SSA signal subspaces + a *learned non-anomalous reference DS `D_N`* +
   direction/magnitude indices (model-the-normal, flag-the-residual). The `D_N` reference is exactly the
   invariance lever that elsewhere is the only thing that beats baselines, and I never built it. ⇒ temporal-DS /
   H-B is NOT fairly closed for the faithful method. This is now the lead experiment.

## 4. Rule going forward
Every new experiment appends a row to §2 BEFORE running, stating the object/data-matrix/build/meaning. If the
"what it represents" is not knowable before running, say so and fill it after. Prefer FAITHFUL lab constructions
over ad-hoc ones; when deviating, justify why.
