"""
(Geometry x Interpretability) — DS canonical directions as interpretable change-MODES.

Question (the user's interpretability ask): a scalar change map gives ONE number
("how much changed") and structurally cannot say WHICH KIND of change. The
Difference Subspace gives DIRECTIONS — canonical change modes. Do those DS
directions recover real change TYPES (unsupervised), and do they beat the field's
direction-based baseline (polar-domain CVA, Bovolo-Bruzzone)?

Data: Benton (real bitemporal hyperspectral, 159 bands, 2004 vs 2007) with a
multiclass change-TYPE ground truth: classes 1-6 are change types, 7 is
background/no-change. The ONLY real multiclass change GT on disk.

Methods (cluster the CHANGED pixels into 6 groups, compare to the 6 GT types):
  ds_mode   : per-date spectral PCA subspaces -> canonical DS basis D (159 x k);
              per-pixel coords c = D^T (post-pre). Geometry's change-mode rep.
  cva_dir   : polar-CVA direction = unit change vector (Bovolo-Bruzzone). FIELD BASELINE.
  raw_delta : the raw change vector (post-pre). Naive baseline.
  pcadiff   : PCA of the change vectors (rank k) coords. Standard reduced rep.

Honest pre-registration: prior is LOW that ds_mode beats cva_dir on cluster
agreement (covariance/correlation has beaten the subspace in every prior cell).
The deliverable is BOTH the quantitative comparison AND the interpretable
artifact (each DS direction = a named spectral change-mode), regardless.

Run: .\.venv\Scripts\python.exe -m phase1.experiments.ds_change_modes --rank 12 --k 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phase1.ds import pca_utils

BENTON = ROOT / "data" / "HSI_change" / "benton"
OUT = ROOT / "phase1" / "outputs" / "ds_change_modes"


def load_benton():
    pre = sio.loadmat(BENTON / "PreImg_2004.mat")["img_2004"].astype(np.float64)
    post = sio.loadmat(BENTON / "PostImg_2007.mat")["img_2007"].astype(np.float64)
    gt = sio.loadmat(BENTON / "Reference_Map_Multiclass.mat")["Ref_map_multiclass"].astype(int)
    return pre, post, gt


def purity(y_true, y_pred):
    labs = np.unique(y_true); clus = np.unique(y_pred)
    M = np.zeros((len(clus), len(labs)))
    for i, c in enumerate(clus):
        for j, l in enumerate(labs):
            M[i, j] = np.sum((y_pred == c) & (y_true == l))
    return M.max(axis=1).sum() / M.sum()


def cluster_eval(feat, y_true, k, seed=0):
    f = StandardScaler().fit_transform(feat)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(f)
    yp = km.labels_
    return {"nmi": float(normalized_mutual_info_score(y_true, yp)),
            "ari": float(adjusted_rand_score(y_true, yp)),
            "purity": float(purity(y_true, yp))}, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=12, help="per-date PCA subspace rank")
    ap.add_argument("--k", type=int, default=6, help="number of change-type clusters (= #GT change classes)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pre, post, gt = load_benton()
    H, W, B = pre.shape
    # per-band standardize across the scene (both dates jointly) for scale stability
    both = np.concatenate([pre.reshape(-1, B), post.reshape(-1, B)], 0)
    mu, sd = both.mean(0), both.std(0) + 1e-6
    Xpre = ((pre.reshape(-1, B) - mu) / sd).T   # (B, N)
    Xpost = ((post.reshape(-1, B) - mu) / sd).T
    delta = Xpost - Xpre                         # (B, N)
    gflat = gt.reshape(-1)

    # canonical DS between per-date spectral subspaces
    pre_basis = pca_utils.fit_pca_basis(Xpre, rank=args.rank, variance_threshold=None, random_state=args.seed)
    post_basis = pca_utils.fit_pca_basis(Xpost, rank=args.rank, variance_threshold=None, random_state=args.seed)
    D = pca_utils.build_difference_subspace(pre_basis.basis, post_basis.basis, variant="canonical")  # (B, kD)
    print(f"[benton] {H}x{W} {B} bands; DS dim = {D.shape[1]}", flush=True)

    # PCA of change vectors (standard reduced rep)
    pcad = pca_utils.fit_pca_basis(delta, rank=args.rank, variance_threshold=None, random_state=args.seed).basis  # (B,k)

    # restrict to CHANGED pixels (GT classes 1..6); class 7 = background/no-change
    changed = (gflat >= 1) & (gflat <= 6)
    y = gflat[changed]
    d_ch = delta[:, changed]                                   # (B, Nc)
    feats = {
        "ds_mode": (D.T @ d_ch).T,                              # (Nc, kD)
        "cva_dir": (d_ch / (np.linalg.norm(d_ch, axis=0, keepdims=True) + 1e-9)).T,  # polar-CVA direction
        "raw_delta": d_ch.T,                                    # (Nc, B)
        "pcadiff": (pcad.T @ d_ch).T,                           # (Nc, k)
    }
    print(f"[benton] changed pixels = {int(changed.sum())} across {len(np.unique(y))} GT change types", flush=True)

    results, preds = {}, {}
    for name, f in feats.items():
        m, yp = cluster_eval(f, y, args.k, args.seed)
        results[name] = m; preds[name] = yp
        print(f"  {name:>10}: NMI {m['nmi']:.4f}  ARI {m['ari']:.4f}  purity {m['purity']:.4f}", flush=True)

    # ---- interpretability figure ----
    fig = plt.figure(figsize=(16, 8))
    # GT map
    axg = fig.add_subplot(2, 3, 1); axg.imshow(gt, cmap="tab10"); axg.set_title("GT change types (1-6) + bg(7)"); axg.axis("off")
    # cluster maps for ds_mode and cva_dir (remapped to full image)
    for i, name in enumerate(["ds_mode", "cva_dir"]):
        cmap_full = np.zeros(H * W); cmap_full[changed] = preds[name] + 1
        axc = fig.add_subplot(2, 3, 2 + i); axc.imshow(cmap_full.reshape(H, W), cmap="tab10")
        axc.set_title(f"{name} clusters (NMI {results[name]['nmi']:.3f})"); axc.axis("off")
    # DS direction spectral profiles (the interpretable "change modes")
    axd = fig.add_subplot(2, 3, 4)
    for i in range(min(6, D.shape[1])):
        axd.plot(D[:, i], label=f"DS dir {i+1}", lw=1)
    axd.set_title("DS canonical directions = spectral change-modes"); axd.set_xlabel("band"); axd.legend(fontsize=6)
    # mean change spectrum per GT class (interpretable reference)
    axm = fig.add_subplot(2, 3, 5)
    for c in range(1, 7):
        mc = d_ch[:, y == c].mean(1)
        axm.plot(mc, label=f"class {c}", lw=1)
    axm.set_title("mean change spectrum per GT class"); axm.set_xlabel("band"); axm.legend(fontsize=6)
    # NMI bar
    axb = fig.add_subplot(2, 3, 6)
    names = list(results.keys()); nmis = [results[n]["nmi"] for n in names]
    axb.bar(names, nmis); axb.set_title("change-type cluster agreement (NMI)"); axb.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fp = OUT / "change_modes_benton.png"
    fig.savefig(fp, dpi=130, bbox_inches="tight"); plt.close(fig)

    import json
    with (OUT / "results.json").open("w") as f:
        json.dump({"rank": args.rank, "k": args.k, "ds_dim": int(D.shape[1]),
                   "n_changed": int(changed.sum()), "results": results}, f, indent=2)
    print(f"\nsaved -> {OUT}  (figure: {fp.name})")
    best = max(results, key=lambda n: results[n]["nmi"])
    print(f"=== best change-type clustering: {best} (NMI {results[best]['nmi']:.4f}); "
          f"ds_mode vs cva_dir NMI: {results['ds_mode']['nmi']:.4f} vs {results['cva_dir']['nmi']:.4f} ===")


if __name__ == "__main__":
    main()
