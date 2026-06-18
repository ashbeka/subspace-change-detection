# Method — Temporal Difference Subspaces for Satellite Image Time Series

Canonical, exact-math description of the method on branch `claude/temporal-ds`. Every formula maps to a
function in `temporal/subspace.py` / `temporal/dynamics.py` and to the lab reference `MagTool` line it was
derived from. Figures in [`docs/figures/`](figures/).

Companion docs: design `docs/DESIGN_TEMPORAL_DS_ACCV2026.md` · results `docs/experiment_reports/` ·
failsafe framing `docs/DRAFT_DIAGNOSTIC_PAPER_ACCV.md`.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $D$ | ambient dimension of a subspace's vector space |
| $\mathbf{\Phi}\in\mathbb{R}^{D\times k}$ | an orthonormal **basis** (columns) spanning a $k$-dim subspace $\mathcal{S}=\mathrm{span}(\mathbf{\Phi})$ |
| $\theta_1\le\dots\le\theta_k$ | **canonical (principal) angles** between two subspaces |
| $\sigma_i=\cos\theta_i$ | canonical **cosines** = singular values of $\mathbf{\Phi}_1^\top\mathbf{\Phi}_2$ |
| $S(t)$ | the subspace at date/time index $t$ — one point on the **Grassmann manifold** $\mathrm{Gr}(k,D)$ |
| $\tau$ | temporal lag (in acquisitions) used as the finite-difference step |

A subspace is a *point on the Grassmann manifold*; a time series of subspaces $S(0),S(1),\dots$ is a
*trajectory* on it. The whole method is finite-difference calculus of that trajectory.

---

## 1. Construction — from a satellite time series to a subspace trajectory

This is the crux, and the reason the temporal framing works where the bi-temporal one failed.

**Why a temporal subspace is legitimate (the "real set").** A Difference Subspace needs a *set of samples*
that span a subspace. A single pre/post image pair has no such set, so the old project faked one from the
pixels of one image — which collapses to a band re-weighting (≈ spectral angle) and loses spatial
structure. A **temporal window of dates is a genuine set**: its samples are the dated observations, and the
subspace they span captures the region's *dynamics* (trend + harmonics), not just its mean spectrum.

**Multivariate SSA (M-SSA) construction** (verified regime, `temporal/experiments/multiband_gate.py`):

For a region $R$ with a $B$-band index time series $\mathbf{x}_b(t)$, and a window $W$ of dates:

1. **Per-band trajectory (Hankel) matrix** with embedding length $L$:
   $$ H_b = \begin{bmatrix} x_b(t) & x_b(t{+}1) & \cdots \\ x_b(t{+}1) & x_b(t{+}2) & \cdots \\ \vdots & & \ddots \end{bmatrix} \in \mathbb{R}^{L\times (W-L+1)} $$
2. **Stack bands** vertically: $H = [\,H_1;\,H_2;\,\dots;\,H_B\,]\in\mathbb{R}^{BL\times(W-L+1)}$.
3. **Signal subspace** $S = \mathrm{top\text{-}}k$ left singular vectors of $H$ (rank by energy ratio).
   → `pca_subspace(H, dim=k, center=False, energy=0.99)`.

Sliding the window gives the trajectory $S(0),S(1),\dots$. Because $H$ stacks delay-embedded multi-band
data, $S$ is **genuinely multi-dimensional** (trend + multiple harmonics + cross-band structure) — the
property the whole contribution depends on.

> Alternative constructions (per-date spatial-spectral subspace; bi-temporal $T{=}2$ special case) are in
> `DESIGN_TEMPORAL_DS_ACCV2026.md §11`. M-SSA is the one that beats scalars.

See [`figures/fig_mssa_construction.png`](figures/fig_mssa_construction.png).

---

## 2. Subspace algebra (exact, as implemented)

All in `temporal/subspace.py`, re-derived from `MagTool/magnitude.py` and guarded by `tests/`.

**Basis construction** — `pca_subspace`. SVD $X=U\Sigma V^\top$ of the data matrix $X\in\mathbb{R}^{D\times n}$
(samples as columns); keep the smallest $k$ with $\frac{\sum_{i\le k}\sigma_i^2}{\sum_i\sigma_i^2}\ge$ `energy`,
capped at `dim`. Energy-based rank avoids padding a low-rank region with noise directions (the failure mode
that makes DS noise-dominated).

**Canonical cosines** — `canonical_cosines`: $\sigma_i = \mathrm{svd}\!\big(\mathbf{\Phi}_1^\top\mathbf{\Phi}_2\big)$, clipped to $[0,1]$.

**Difference-Subspace magnitude** — `magnitude` (the core scalar; `MagTool magnitude.py:223`):
$$ \boxed{\;\lVert \mathrm{DS}(\mathcal{S}_1,\mathcal{S}_2)\rVert \;=\; 2\Big(\max(k_1,k_2) - \textstyle\sum_i \cos\theta_i\Big)\;=\;2\sum_i\big(1-\cos\theta_i\big)\;} $$
Identical subspaces $\Rightarrow 0$; fully orthogonal $k$-dim $\Rightarrow 2k$ (max). We sum over
$\max(k_1,k_2)$ (not MagTool's $\min$) so a genuinely **new** direction (e.g. a post-event material/mode)
counts at full weight instead of being dropped; identical for equal dims.

**Difference Subspace basis** — `difference_subspace`. Eigendecompose the projector sum
$P_1+P_2=\mathbf{\Phi}_1\mathbf{\Phi}_1^\top+\mathbf{\Phi}_2\mathbf{\Phi}_2^\top$; its eigenvalues are
$1\pm\cos\theta_i$ (plus exact $2$ for shared dims, $1$ for single-cover dims). The **DS basis = eigenvectors
with eigenvalue in $(0,1)$** — the directions in which the two subspaces disagree. Empty ⇔ identical.

**Karcher (principal) mean** — `karcher_subspace`: eigenvectors of $P_1+P_2$ with eigenvalue $\ge 1$ — the
geodesic midpoint subspace of $\mathcal{S}_1,\mathcal{S}_3$.

**First/second-order geodesic decomposition** — `first_order_decomposed`, `second_order_decomposed`.
Project the middle subspace $\mathbf{\Phi}_2$ onto the plane $W=\mathrm{span}(\mathbf{\Phi}_1,\mathbf{\Phi}_3)$:
- `mag_orth` $= 2\big(\lvert s\rvert-\sum s\big)$ where $s=\mathrm{svd}(W^\top\mathbf{\Phi}_2)$ — the component
  **off** the geodesic plane (a regime break / new direction).
- `mag_along` $= \lVert\mathrm{DS}(\mathbf{\Phi}_2^{\text{proj}}, \cdot)\rVert$ — **along** the geodesic;
  first-order measures vs the start $\mathbf{\Phi}_1$, **second-order vs the Karcher mean $M$**.

**Second-order magnitude** — `second_order_magnitude` (Fukui 2024):
$$ \lVert \mathrm{DS}^{(2)}\rVert \;=\; \big\lVert \mathrm{DS}\big(S(t{+}\tau),\; M(S(t),S(t{+}2\tau))\big)\big\rVert $$
Constant-velocity (geodesic) motion ⇒ $S(t{+}\tau)=M$ ⇒ $0$. Large ⇒ the trajectory **curves** = acceleration.

---

## 3. Change descriptors (the curves we read) — `temporal/dynamics.py`

| Curve | Formula | Physical meaning | "A spike" |
|---|---|---|---|
| **velocity** $d_1(t)$ | $\lVert\mathrm{DS}(S(t),S(t{+}\tau))\rVert$ | rate of scene change | **change onset** |
| **acceleration** $d_2(t)$ | $\lVert\mathrm{DS}(S(t{+}\tau),M(S(t),S(t{+}2\tau)))\rVert$ | non-uniform change | **abrupt event** (vs steady drift) |
| **geodesic** $(\,\text{along},\text{orth}\,)$ | `second_order_decomposed` | on-path vs off-path | **regime break** (orth) vs trend (along) |
| **recovery** $d_\text{pre}(t)$ | $\lVert\mathrm{DS}(S(t),S_\text{pre})\rVert$ | distance to pre-event state | **rising = degrading, falling = recovering** |
| **attribution** | per-band energy $\sum_{\text{rows}_b}\lVert \mathrm{DS\,basis}\rVert^2$ | *which band/mode* changed | the band that lights up |

Self-consistency (label-free correctness check, Fukui 2024): $d_2(t)\approx \lvert\frac{d}{dt}d_1(t)\rvert$
(`numeric_derivative`) — acceleration must equal the numerical derivative of velocity. This is a *correctness*
test, not task validity.

---

## 4. Pipeline (end-to-end)

```mermaid
flowchart TD
    A[Sentinel-2 Harmonized L2A<br/>COPERNICUS/S2_SR_HARMONIZED] -->|GEE: per-AOI fetch + cloud mask| B[per-region multi-band<br/>index time series x_b]
    B --> C[sliding window W]
    C --> D[per-band Hankel H_b<br/>embedding L]
    D --> E[stack bands -> H in R^{BL x .}]
    E --> F["S(t) = top-k SVD of H<br/>(energy-based rank)"]
    F --> G[subspace trajectory<br/>S0,S1,... on Grassmann]
    G --> H1["velocity d1(t)<br/>= ||DS(S(t),S(t+tau))||"]
    G --> H2["acceleration d2(t)<br/>vs Karcher mean"]
    G --> H3["geodesic along/orth"]
    G --> H4["recovery d_pre(t)<br/>vs S_pre"]
    F --> H5["attribution<br/>per-band DS-basis energy"]
    H1 --> V{Verification ladder}
    H2 --> V
    H3 --> V
    H4 --> V
    H5 --> V
    V -->|L0 sanity| V0[identical->0; d2~d/dt d1]
    V -->|L1 synthetic| V1[injected change: AUC vs magnitude]
    V -->|L2 known event| V2[spike within +-5 days of date]
    V -->|L3 recovery| V3[d_pre vs NBR curve]
    V -->|L4 baselines| V4[vs min-angle / spectral scalar / trivial null]
```

See [`figures/fig_pipeline.png`](figures/fig_pipeline.png) and
[`figures/fig_velocity_acceleration.png`](figures/fig_velocity_acceleration.png).

---

## 5. Verification ladder (with the verified results)

| Rung | Test | Ground truth | Status |
|---|---|---|---|
| **L0** | identical-date magnitude $=0$; $d_2\approx \lvert d\,d_1/dt\rvert$ | math invariant | ✅ holds |
| **L1** | inject step/ramp into stable series → detection AUC vs magnitude | injection (exact) | ✅ but first-order spectral DS $\equiv$ spectral angle (rank-1 collapse) — see L1 report |
| **multi-dim gate** | distributed multi-mode change (M-SSA): full DS vs min-angle & scalars | synthetic, multi-seed | ✅ **DS beats min-angle +0.45–0.57 (all configs); beats best scalar under noise (+0.15–0.32); attribution 100%** |
| **L2** | real S2: $d_1$ peaks within ±5 days of documented event | MTBS/EFFIS dates | ⏳ next (GEE wired) |
| **L3** | recovery $d_\text{pre}$ vs NBR slope | NBR (guard B8/B12 circularity) | ⏳ |
| **L4** | vs min-angle, spectral scalars, trivial reflectance-diff null | same as rung | ✅ on synthetic, ⏳ on real |

**Honest scope** (verified): the full DS magnitude *equals* aggregating all canonical angles
($d_\text{mag}\equiv\text{mean}_\text{all}\equiv\text{sum}_{1-\cos}$); its genuine edges are (i) decisively
beating the conventional single min-angle SSA, (ii) **noise robustness** vs even the best spectral scalar,
(iii) **attribution** of the change to bands/modes, (iv) the second-order/geodesic recovery axis. On clean
data strong scalars tie it; it is regime-specific (fails on amplitude-redistribution change).

---

## 6. Provenance

```text
MagTool (Jang) references/reference_code/MagTool-main/MagTool-main/magnitude.py
  -> magnitude = 2*sum(1-cos), difference/karcher/sum subspaces, adjustEig snapping
cv_motion3d (Soto-san) references/reference_code/cv_motion3d_public-main
  -> per-frame subspace + first/second/geodesic over a temporal lag (the satellite analogue)
Fukui et al. 2024, "Second-order Difference Subspace" (arXiv:2409.08563)
  -> first/second-order DS = velocity/acceleration on a Grassmann geodesic; d2 ~ d/dt d1
Kanai, Sogi, Maki, Fukui 2023 (arXiv:2303.17802)
  -> DS between SSA signal subspaces > single min-angle (0.923 vs 0.829 AUC)
```
Code-truth: `temporal/subspace.py`, `temporal/dynamics.py`; tests in `tests/`.
