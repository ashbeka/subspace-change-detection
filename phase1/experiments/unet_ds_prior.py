"""
U-Net with a Band-Image Difference-Subspace (DS) prior channel on OSCD.

Scientific question
-------------------
The matched-controls study showed Band-Image DS adds statistically-significant,
DS-specific complementary *ranking* information to classical change maps in a
percentile-rank fusion, yet standalone it does not beat a smoothed PCA map. The
natural objection (and the top rung of the naive->complex baseline ladder) is:

    "A CNN trained end-to-end on the raw bitemporal bands can learn any spatial
     feature it wants -- a hand-crafted subspace map must be redundant once you
     have a deep model."

This experiment tests that head on. We take a small FC-EF U-Net (Daudt et al.,
the canonical OSCD deep baseline) and ask whether injecting the *label-free,
global, spatially-faithful* Band-Image DS map as an extra input channel improves
change segmentation over the same network without it -- AND whether the gain is
DS-specific by swapping in matched control channels (cross-reconstruction, the
strongest geometric null; smoothed-PCA, the strongest classical map; IR-MAD).

Why a DS prior could help a CNN (the mechanism, not magic)
---------------------------------------------------------
1. GLOBAL vs LOCAL. The Band-Image DS is a PCA over the whole image (a global
   spatial basis from the 13 band-images). A U-Net is local (limited receptive
   field) and, on 14 OSCD training cities, cannot cheaply learn a global,
   image-wide subspace decomposition. The DS map injects a global scene-level
   geometric statistic the CNN does not otherwise have. (CGNet, arXiv:2404.09179,
   explicitly motivates change priors by the "insufficient receptive field" of
   CNNs.)
2. SAMPLE EFFICIENCY. Deep OSCD models degrade sharply as labels shrink. A
   label-free geometric prior is a domain inductive bias that should help most in
   exactly this low-label regime -- the disaster/reconstruction setting where
   labels are scarce or absent.

Configs (same U-Net, same training; only input channels differ)
---------------------------------------------------------------
  bands        : [pre(13), post(13)]                 -- the strong learned baseline (M0)
  bands_ds     : base + DS                            -- our method            (M1)
  bands_cross  : base + cross-reconstruction          -- DS-specificity control (M2)
  bands_spca   : base + smoothed-PCA                  -- strongest classical map prior (M3)
  bands_fusion : base + DS + smoothed-PCA + IR-MAD    -- full prior fusion      (M4)

Key reads: M1 vs M0 (does DS help?), M1 vs M2 (is it DS-specific?), M1 vs M3.

All priors are computed on the SAME normalized 13-band stacks used by the
matched-controls study, then percentile-rank normalized to [0,1] (the same
scale-free convention as the fusion). The priors use NO labels.

Run (quick proof):
  .\.venv\Scripts\python.exe -m phase1.experiments.unet_ds_prior \
      --configs bands,bands_ds,bands_cross,bands_spca --seeds 0 --steps 1500

Run (full):
  ... --configs bands,bands_ds,bands_cross,bands_spca,bands_fusion --seeds 0,1,2 --steps 4000
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn import metrics as skm

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.baselines.ir_mad import ir_mad_score
from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.oscd_dataset import OSCDEvaluatorDataset, OFFICIAL_TRAIN, OFFICIAL_TEST
from phase1.data.preprocessing import apply_normalization, load_band_stats
from phase1.scripts.compare_oscd_spatial_subspaces import (
    BAND_ORDER,
    band_image_ds_score,
    band_image_spatial_control_score,
    parse_method_spec,
)

warnings.filterwarnings("ignore")

OUT_ROOT = ROOT / "phase1" / "outputs" / "unet_ds_prior"
CACHE_DIR = OUT_ROOT / "cache"
DS_RANK = 12

# Map a config name to the extra prior channels appended after [pre,post].
CONFIG_CHANNELS = {
    "bands": [],
    "bands_ds": ["ds"],
    "bands_cross": ["cross"],
    "bands_spca": ["spca"],
    "bands_irmad": ["irmad"],
    "bands_fusion": ["ds", "spca", "irmad"],
    # fusion DS-specificity controls: does DS add to the multi-prior fusion?
    "bands_pca_irmad": ["spca", "irmad"],            # fusion WITHOUT DS
    "bands_cross_pca_irmad": ["cross", "spca", "irmad"],  # DS replaced by matched null
}


# --------------------------------------------------------------------------- #
# Precompute: normalized bands + label-free prior maps, cached per city.
# --------------------------------------------------------------------------- #
def _rank_norm(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Percentile-rank a score map to [0,1] over valid pixels (scale-free)."""
    out = np.zeros_like(x, dtype=np.float32)
    v = x[valid].astype(np.float64)
    if v.size == 0:
        return out
    order = np.argsort(np.argsort(v))
    out[valid] = (order / max(1, v.size - 1)).astype(np.float32)
    return out


def precompute_city(city: str, split: str, oscd_root: Path, stats, seed: int) -> dict:
    cache_fp = CACHE_DIR / f"{city}.npz"
    if cache_fp.exists():
        d = np.load(cache_fp)
        return {k: d[k] for k in d.files}

    ds_set = OSCDEvaluatorDataset(root=oscd_root, split=split, band_order=BAND_ORDER,
                                  nodata_value=0.0, min_valid_bands=3)
    sample = ds_set.load_city(city)
    x1n, valid = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask)
    x2n, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask)
    x1n = x1n.astype(np.float32)
    x2n = x2n.astype(np.float32)
    y = (sample.y[0] if sample.y is not None else np.zeros(valid.shape, np.uint8)).astype(np.uint8)

    args = SimpleNamespace(rank=DS_RANK, seed=seed)
    priors: dict[str, np.ndarray] = {}
    # Band-Image DS (energy_norm) -- our prior.
    ds_map, _, _ = band_image_ds_score(x1n, x2n, valid, args, parse_method_spec("band_image_norm"))
    priors["ds"] = _rank_norm(ds_map, valid)
    # Cross-reconstruction (matched geometric null) -- DS-specificity control.
    cross_map, _ = band_image_spatial_control_score(
        x1n, x2n, valid, args, parse_method_spec("band_image_cross_reconstruction"))
    priors["cross"] = _rank_norm(cross_map, valid)
    # Smoothed PCA-diff (strongest classical map).
    pca_map = pca_diff_score(x1n, x2n, valid, variance_threshold=0.95, random_state=seed)
    pca_sm = gaussian_filter(pca_map.astype(np.float32), sigma=1.0)
    pca_sm[~valid] = 0.0
    priors["spca"] = _rank_norm(pca_sm, valid)
    # IR-MAD.
    try:
        irm = ir_mad_score(x1n, x2n, valid, iters=10, random_state=seed)
        priors["irmad"] = _rank_norm(irm.astype(np.float32), valid)
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] IR-MAD failed on {city}: {exc}")
        priors["irmad"] = np.zeros(valid.shape, np.float32)

    rec = {"x1": x1n, "x2": x2n, "y": y, "valid": valid.astype(np.uint8), **priors}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_fp, **rec)
    return rec


# Band subsets (indices into BAND_ORDER) for the "do we need all 13 bands?" study.
BAND_SUBSETS = {
    "all": list(range(13)),
    "rgb": [3, 2, 1],                 # B04,B03,B02
    "rgbnir": [3, 2, 1, 7, 12],       # + B08, B8A
    "rgbnirswir": [3, 2, 1, 7, 12, 10, 11],  # + B11,B12
}


def _minmax_bands(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-band image-wise min-max to [0,1] over valid pixels (Sensei's 0-1 norm)."""
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        v = x[c][valid]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        if hi > lo:
            out[c][valid] = np.clip((x[c][valid] - lo) / (hi - lo), 0, 1)
    return out


def assemble_input(rec: dict, channels: list[str], bands_idx=None, norm: str = "zscore") -> np.ndarray:
    """[pre(bands), post(bands), <priors>] -> (C,H,W) float32."""
    valid = rec["valid"].astype(bool)
    x1, x2 = rec["x1"], rec["x2"]
    if bands_idx is not None:
        x1 = x1[bands_idx]
        x2 = x2[bands_idx]
    if norm == "minmax":
        x1 = _minmax_bands(x1, valid)
        x2 = _minmax_bands(x2, valid)
    parts = [x1, x2]
    for ch in channels:
        parts.append(rec[ch][None, ...].astype(np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Model: small FC-EF U-Net (early fusion).
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bott = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(b), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)[:, 0]


# --------------------------------------------------------------------------- #
# Training (patch-based, change-aware sampling, masked BCE+Dice).
# --------------------------------------------------------------------------- #
def sample_patch(rec_in: np.ndarray, rec: dict, rng, ps: int, p_change: float):
    _, H, W = rec_in.shape
    y, valid = rec["y"], rec["valid"].astype(bool)
    if H <= ps or W <= ps:
        # pad small images
        ph, pw = max(0, ps - H), max(0, ps - W)
        rec_in = np.pad(rec_in, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        y = np.pad(y, ((0, ph), (0, pw)))
        valid = np.pad(valid, ((0, ph), (0, pw)))
        _, H, W = rec_in.shape
    if rng.random() < p_change and (y & valid).any():
        cr, cc = np.where(y & valid)
        j = rng.integers(len(cr))
        r0 = int(np.clip(cr[j] - ps // 2, 0, H - ps))
        c0 = int(np.clip(cc[j] - ps // 2, 0, W - ps))
    else:
        r0 = int(rng.integers(0, H - ps + 1))
        c0 = int(rng.integers(0, W - ps + 1))
    xi = rec_in[:, r0:r0 + ps, c0:c0 + ps]
    yi = y[r0:r0 + ps, c0:c0 + ps]
    vi = valid[r0:r0 + ps, c0:c0 + ps]
    # augment: flips + rot90
    k = rng.integers(4)
    xi = np.rot90(xi, k, axes=(1, 2)).copy(); yi = np.rot90(yi, k).copy(); vi = np.rot90(vi, k).copy()
    if rng.random() < 0.5:
        xi = xi[:, ::-1].copy(); yi = yi[::-1].copy(); vi = vi[::-1].copy()
    return xi, yi.astype(np.float32), vi.astype(np.float32)


def masked_loss(logits, y, v, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * v).sum() / (v.sum() + 1e-6)
    p = torch.sigmoid(logits)
    inter = (p * y * v).sum()
    denom = (p * v).sum() + (y * v).sum()
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    return bce + dice


def train_one(config: str, seed: int, train_recs, train_inputs, args, device) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    in_ch = train_inputs[0].shape[0]
    model = UNet(in_ch, base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # global pos_weight from train labels (capped)
    tot_pos = sum(int(((r["y"] == 1) & r["valid"].astype(bool)).sum()) for r in train_recs)
    tot_val = sum(int(r["valid"].astype(bool).sum()) for r in train_recs)
    pw = float(np.clip((tot_val - tot_pos) / max(1, tot_pos), 1.0, 50.0))

    model.train()
    n_city = len(train_inputs)
    for step in range(args.steps):
        xs, ys, vs = [], [], []
        for _ in range(args.batch):
            ci = int(rng.integers(n_city))
            xi, yi, vi = sample_patch(train_inputs[ci], train_recs[ci], rng, args.patch, args.p_change)
            xs.append(xi); ys.append(yi); vs.append(vi)
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        v = torch.from_numpy(np.stack(vs)).to(device)
        logits = model(x)
        loss = masked_loss(logits, y, v, pw)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % max(1, args.steps // 5) == 0:
            print(f"    [{config} seed{seed}] step {step+1}/{args.steps} loss {loss.item():.4f}", flush=True)
    return model


@torch.no_grad()
def infer_city(model, rec_in: np.ndarray, device, mult: int = 8) -> np.ndarray:
    model.eval()
    _, H, W = rec_in.shape
    ph = (mult - H % mult) % mult
    pw = (mult - W % mult) % mult
    x = np.pad(rec_in, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    xt = torch.from_numpy(x[None]).to(device)
    prob = torch.sigmoid(model(xt))[0].cpu().numpy()
    return prob[:H, :W]


def eval_city(prob, rec) -> dict:
    valid = rec["valid"].astype(bool)
    y = rec["y"][valid].astype(np.uint8).ravel()
    s = prob[valid].astype(np.float64).ravel()
    if y.size == 0 or np.unique(y).size < 2:
        return {}
    ap = float(skm.average_precision_score(y, s))
    auroc = float(skm.roc_auc_score(y, s))
    prec, rec_, thr = skm.precision_recall_curve(y, s)
    f1 = (2 * prec * rec_) / (prec + rec_ + 1e-12)
    bi = int(np.nanargmax(f1))
    best_f1 = float(f1[bi])
    iou = float(best_f1 / (2 - best_f1 + 1e-12))
    return {"ap": ap, "auroc": auroc, "best_f1": best_f1, "iou": iou,
            "pos_rate": float(y.mean())}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oscd_root", type=Path, default=ROOT / "data" / "OSCD")
    ap.add_argument("--stats_path", type=Path, default=ROOT / "phase1" / "data" / "oscd_band_stats.json")
    ap.add_argument("--configs", default="bands,bands_ds,bands_cross,bands_spca")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p_change", type=float, default=0.6)
    ap.add_argument("--n_train", type=int, default=0,
                    help="If >0, use only the first n_train official train cities (label-scarcity sweep).")
    ap.add_argument("--norm", choices=["zscore", "minmax"], default="zscore",
                    help="Band normalization for the CNN input (Sensei's 0-1 vs z-score test).")
    ap.add_argument("--bands", choices=list(BAND_SUBSETS.keys()), default="all",
                    help="Which band subset to feed the CNN (do-we-need-13-bands study).")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    bands_idx = BAND_SUBSETS[args.bands]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stats = load_band_stats(args.stats_path)

    train_cities = OFFICIAL_TRAIN if args.n_train <= 0 else OFFICIAL_TRAIN[:args.n_train]
    t0 = time.time()
    print(f"[precompute] {len(train_cities)} train + {len(OFFICIAL_TEST)} test cities ...", flush=True)
    train_recs = [precompute_city(c, "train", args.oscd_root, stats, seed=1234) for c in train_cities]
    test_recs = [precompute_city(c, "test", args.oscd_root, stats, seed=1234) for c in OFFICIAL_TEST]
    print(f"[precompute] done in {time.time()-t0:.1f}s", flush=True)

    results = {}
    for config in configs:
        chans = CONFIG_CHANNELS[config]
        tr_in = [assemble_input(r, chans, bands_idx, args.norm) for r in train_recs]
        te_in = [assemble_input(r, chans, bands_idx, args.norm) for r in test_recs]
        per_seed = []
        for seed in seeds:
            ts = time.time()
            model = train_one(config, seed, train_recs, tr_in, args, device)
            city_metrics = []
            for r, xin in zip(test_recs, te_in):
                prob = infer_city(model, xin, device)
                m = eval_city(prob, r)
                if m:
                    m["city"] = r  # placeholder
                    city_metrics.append({k: v for k, v in m.items() if k != "city"})
            agg = {k: float(np.mean([cm[k] for cm in city_metrics]))
                   for k in ["ap", "auroc", "best_f1", "iou"]}
            agg["seed"] = seed
            per_seed.append(agg)
            print(f"  [{config} seed{seed}] TEST  AP {agg['ap']:.4f}  AUROC {agg['auroc']:.4f}  "
                  f"bestF1 {agg['best_f1']:.4f}  IoU {agg['iou']:.4f}  ({time.time()-ts:.0f}s)", flush=True)
        summary = {k: {"mean": float(np.mean([p[k] for p in per_seed])),
                       "std": float(np.std([p[k] for p in per_seed]))}
                   for k in ["ap", "auroc", "best_f1", "iou"]}
        results[config] = {"in_ch": int(tr_in[0].shape[0]), "per_seed": per_seed, "summary": summary}
        print(f"==> {config}: AP {summary['ap']['mean']:.4f}+-{summary['ap']['std']:.4f}  "
              f"bestF1 {summary['best_f1']['mean']:.4f}+-{summary['best_f1']['std']:.4f}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_fp = OUT_ROOT / f"results_{args.tag}.json"
    with out_fp.open("w") as f:
        json.dump({"args": vars(args) | {"oscd_root": str(args.oscd_root),
                                         "stats_path": str(args.stats_path)},
                   "results": results}, f, indent=2, default=str)
    print(f"\nsaved -> {out_fp}")

    # readable ladder table
    print("\n=== TEST-city means (10 official cities) ===")
    print(f"{'config':<16}{'in_ch':>6}{'AP':>10}{'AUROC':>10}{'bestF1':>10}{'IoU':>10}")
    for config in configs:
        s = results[config]["summary"]
        print(f"{config:<16}{results[config]['in_ch']:>6}"
              f"{s['ap']['mean']:>10.4f}{s['auroc']['mean']:>10.4f}"
              f"{s['best_f1']['mean']:>10.4f}{s['iou']['mean']:>10.4f}")


if __name__ == "__main__":
    main()
