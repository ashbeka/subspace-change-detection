"""T1 L1 — geometric change-TYPE characterization (research-mining's TOP task; docs/research/synthesis_specific_tasks.md).
Pre-registered falsifier: does the 2nd-order DS geodesic ALONG/ORTHOGONAL split classify abrupt-vs-gradual
change-TYPE better than the scalar 2nd-difference of the mean spectrum (||Δ²m|| magnitude AND its along/orth
decomposition relative to the velocity)? Construction = M-SSA signal-subspace trajectory (the PROPER one, joint
cross-band+temporal; not the weak patch subspaces of HB1). ≥8 seeds. If subspace does NOT beat the scalar →
T1 folds into the diagnostic (T4); pre-committed.

CONSTRUCTION (ledger): multi-band seasonal series x(t)∈R^B; per-band Hankel stacked → M-SSA signal subspace
S(t)=top-r left singular vectors (joint spectral-temporal structure); geodesic decomposition of 2nd-order DS
on triples (S(t),S(t+lag),S(t+2lag)) → (along=on-geodesic speed change, orth=off-geodesic structural break).
abrupt step → off-geodesic (orth-dominant); gradual ramp → on-geodesic (along-dominant). Matched endpoints.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.t1_changetype
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import signal_subspace_ds as S
from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "t1_changetype")
os.makedirs(OUT, exist_ok=True)
T, B, P, T0 = 150, 6, 36, 90
W, M, R, LAG, SEEDS = 30, 16, 4, 2, 40


def gen(kind, rng, mag=2.5):
    t = np.arange(T)
    x = np.zeros((T, B))
    for b in range(B):                                          # stable multi-mode seasonal base
        x[:, b] = ((1 + 0.4 * rng.standard_normal()) * np.sin(2 * np.pi * t / P + rng.uniform(0, 2 * np.pi))
                   + 0.5 * np.sin(4 * np.pi * t / P + rng.uniform(0, 2 * np.pi)))
    d = rng.standard_normal(B); d /= np.linalg.norm(d)         # change spectral direction
    g = mag * ((t >= T0).astype(float) if kind == "abrupt" else np.clip((t - T0) / (T - 1 - T0), 0, 1))
    return x + g[:, None] * d[None, :] + 0.1 * rng.standard_normal((T, B))


def features(x):
    span = W + M - 1
    bases = [S.mssa_signal_subspace(x[:t + 1], W, M, R) for t in range(span - 1, T)]
    along, orth = [], []
    for i in range(len(bases) - 2 * LAG):
        a, o = ss.second_order_decomposed(bases[i], bases[i + LAG], bases[i + 2 * LAG])
        along.append(a); orth.append(o)
    along, orth = np.array(along), np.array(orth)
    po, pa = (orth.max() if len(orth) else 0.0), (along.max() if len(along) else 0.0)
    sub_ratio = po / (pa + po + 1e-12)
    # scalar null: Δ²x decomposed relative to the velocity direction (the "direction of ||Δ²m||")
    vel = np.diff(x, axis=0); acc = np.diff(vel, axis=0)
    vh = vel[:-1] / (np.linalg.norm(vel[:-1], axis=1, keepdims=True) + 1e-12)
    al_s = np.abs(np.sum(acc * vh, axis=1))
    orth_s = np.linalg.norm(acc - al_s[:, None] * vh, axis=1)
    sc_ratio = orth_s.max() / (al_s.max() + orth_s.max() + 1e-12)
    return sub_ratio, po, sc_ratio, orth_s.max(), np.linalg.norm(acc, axis=1).max()


def main():
    rng = np.random.default_rng(0)
    data = {"abrupt": [], "gradual": []}
    for k in data:
        for _ in range(SEEDS):
            data[k].append(features(gen(k, np.random.default_rng(rng.integers(1 << 30)))))
    arr = {k: np.array(v) for k, v in data.items()}
    y = np.r_[np.ones(SEEDS), np.zeros(SEEDS)]
    auc = lambda c: float(max(roc_auc_score(y, np.r_[arr["abrupt"][:, c], arr["gradual"][:, c]]),
                              1 - roc_auc_score(y, np.r_[arr["abrupt"][:, c], arr["gradual"][:, c]])))
    names = ["subspace orth/along RATIO", "subspace peak ORTH", "scalar Δ²m orth/along RATIO",
             "scalar Δ²m peak ORTH", "scalar ||Δ²m|| magnitude"]
    print("T1 L1 — classify abrupt vs gradual change-TYPE (AUC; M-SSA, 40 seeds):")
    res = {}
    for c, nm in enumerate(names):
        res[nm] = auc(c); print(f"  {nm:>28}: {res[nm]:.3f}")
    sub = max(res["subspace orth/along RATIO"], res["subspace peak ORTH"])
    sc = max(res["scalar Δ²m orth/along RATIO"], res["scalar Δ²m peak ORTH"], res["scalar ||Δ²m|| magnitude"])
    verdict = "T1 POSITIVE — geodesic split beats the scalar 2nd-difference" if sub > sc + 0.05 else \
        "T1 REDUNDANT — folds into the diagnostic (T4)"
    print(f"\n  best SUBSPACE {sub:.3f}  vs  best SCALAR {sc:.3f}  ->  {verdict}")
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
