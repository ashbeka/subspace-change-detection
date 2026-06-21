"""H-C — subspace geometry on DEEP features (SLS-style). No pretrained RS foundation model is installed offline,
so the legitimate deep-feature extractor is an AUTOENCODER trained on the HSI data; latent z=encoder(x) are the
"deep features." Decisive gate on real Hermiston bitemporal: (1) do latent methods beat raw bands? (the AE's
contribution), and (2) THE KILLER — does the latent SUBSPACE beat the latent mean/correlation difference (i.e.
does the GEOMETRY add over just using the deep features)? Honest caveat: any (1)-win is the features' not the
geometry's; only a (2)-win is a geometry contribution. Pre-registered nulls reported.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.hc_deep_subspace
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss

DIR = "data_hsi/ChangeDetectionDataset/Hermiston"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hc_deep_subspace")
os.makedirs(OUT, exist_ok=True)
LAT, W = 32, 5


class AE(nn.Module):
    def __init__(self, B, lat=LAT):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(B, 128), nn.ReLU(), nn.Linear(128, lat))
        self.dec = nn.Sequential(nn.Linear(lat, 128), nn.ReLU(), nn.Linear(128, B))

    def forward(self, x):
        z = self.enc(x); return self.dec(z), z


def sam(a, b):
    return np.arccos(np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9), -1, 1))


def auc(yv, s):
    return float(max(roc_auc_score(yv, s), 1 - roc_auc_score(yv, s)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = lambda f: loadmat(os.path.join(DIR, f))
    X1 = [v for k, v in g("hermiston2004.mat").items() if not k.startswith("__")][0].astype(np.float32)
    X2 = [v for k, v in g("hermiston2007.mat").items() if not k.startswith("__")][0].astype(np.float32)
    GT = [v for k, v in g("rdChangesHermiston_5classes.mat").items() if not k.startswith("__")][0]
    Hh, Ww, B = X1.shape
    f1 = X1.reshape(-1, B); f2 = X2.reshape(-1, B); y = (GT.reshape(-1) > 0)
    mu = np.concatenate([f1, f2]).mean(0); sd = np.concatenate([f1, f2]).std(0) + 1e-6
    f1z = (f1 - mu) / sd; f2z = (f2 - mu) / sd

    # train AE on pooled pixels
    torch.manual_seed(0)
    ae = AE(B).to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    pool = np.concatenate([f1z, f2z]).astype(np.float32)
    rng = np.random.default_rng(0)
    print(f"H-C: training AE (B={B}->{LAT}) on {len(pool)} pixels, dev={dev}")
    for ep in range(60):
        idx = rng.choice(len(pool), 4096, replace=False)
        xb = torch.tensor(pool[idx], device=dev)
        rec, _ = ae(xb); loss = ((rec - xb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        z1 = ae.enc(torch.tensor(f1z, device=dev)).cpu().numpy()
        z2 = ae.enc(torch.tensor(f2z, device=dev)).cpu().numpy()
    print(f"  AE trained (final recon MSE {float(loss):.3f}); latent extracted")

    # balanced per-pixel subset
    nch = np.where(y)[0]; nno = np.where(~y)[0]
    k = min(3000, len(nch), len(nno))
    sel = np.concatenate([rng.choice(nch, k, replace=False), rng.choice(nno, k, replace=False)])
    yv = np.r_[np.ones(k), np.zeros(k)]
    print("\n=== per-pixel detection AUC ===")
    pp = {"raw_CVA": auc(yv, np.linalg.norm(f1z[sel] - f2z[sel], axis=1)),
          "raw_SAM": auc(yv, sam(f1z[sel], f2z[sel])),
          "latent_CVA": auc(yv, np.linalg.norm(z1[sel] - z2[sel], axis=1)),
          "latent_SAM": auc(yv, sam(z1[sel], z2[sel]))}
    for kk, vv in pp.items():
        print(f"  {kk:>12}: {vv:.3f}")

    # patch-based (latent subspace vs latent mean/correlation vs raw) — N=W*W << latent? lat=32, N=25 -> N<lat
    Z1 = z1.reshape(Hh, Ww, LAT); Z2 = z2.reshape(Hh, Ww, LAT)
    R1 = f1z.reshape(Hh, Ww, B); R2 = f2z.reshape(Hh, Ww, B)
    h = W // 2; ksub = min(8, W * W - 1)
    nchp = np.where(GT[h:Hh - h, h:Ww - h].reshape(-1) > 0)[0]
    nnop = np.where(GT[h:Hh - h, h:Ww - h].reshape(-1) == 0)[0]
    Wc = Ww - 2 * h
    def loc(i): return (i // Wc + h, i % Wc + h)
    kp = min(1500, len(nchp), len(nnop))
    selp = [loc(i) for i in rng.choice(nchp, kp, replace=False)] + [loc(i) for i in rng.choice(nnop, kp, replace=False)]
    yp = np.r_[np.ones(kp), np.zeros(kp)]
    methods = {"raw_patchmean_SAM": [], "latent_patchmean_SAM": [], "latent_corr": [], "latent_SUBSPACE": [], "raw_SUBSPACE": []}
    for (r, c) in selp:
        zp1 = Z1[r - h:r + h + 1, c - h:c + h + 1].reshape(-1, LAT); zp2 = Z2[r - h:r + h + 1, c - h:c + h + 1].reshape(-1, LAT)
        rp1 = R1[r - h:r + h + 1, c - h:c + h + 1].reshape(-1, B); rp2 = R2[r - h:r + h + 1, c - h:c + h + 1].reshape(-1, B)
        methods["raw_patchmean_SAM"].append(float(sam(rp1.mean(0)[None], rp2.mean(0)[None])[0]))
        methods["latent_patchmean_SAM"].append(float(sam(zp1.mean(0)[None], zp2.mean(0)[None])[0]))
        Rc1 = np.nan_to_num(np.corrcoef(zp1.T)); Rc2 = np.nan_to_num(np.corrcoef(zp2.T))
        methods["latent_corr"].append(float(np.linalg.norm(Rc1 - Rc2)))
        methods["latent_SUBSPACE"].append(float(ss.magnitude(ss.pca_subspace(zp1.T, ksub, center=True),
                                                             ss.pca_subspace(zp2.T, ksub, center=True))))
        methods["raw_SUBSPACE"].append(float(ss.magnitude(ss.pca_subspace(rp1.T, ksub, center=True),
                                                          ss.pca_subspace(rp2.T, ksub, center=True))))
    print("\n=== patch detection AUC (latent subspace vs latent mean/corr vs raw) ===")
    patch = {kk: auc(yp, np.array(vv)) for kk, vv in methods.items()}
    for kk, vv in patch.items():
        print(f"  {kk:>22}: {vv:.3f}")

    print("\n=== VERDICT ===")
    deep_helps = pp["latent_CVA"] > pp["raw_CVA"] + 0.02 or pp["latent_SAM"] > pp["raw_SAM"] + 0.02
    geom_adds = patch["latent_SUBSPACE"] > max(patch["latent_patchmean_SAM"], patch["latent_corr"]) + 0.03
    print(f"  deep features help over raw (the AE's contribution): {deep_helps}")
    print(f"  latent SUBSPACE beats latent mean/corr (GEOMETRY adds over deep features): {geom_adds}")
    print(f"  => {'H-C POSITIVE (geometry adds on deep features)' if geom_adds else 'H-C: geometry redundant on deep features too (any gain is the features, not the subspace)'}")
    json.dump({"per_pixel": pp, "patch": patch}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
