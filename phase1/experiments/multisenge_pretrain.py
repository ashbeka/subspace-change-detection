"""
MultiSenGE unsupervised pre-training -> OSCD fine-tuning (the label-scarce story).

Tests the real-world driver directly: a NEW disaster has no labels, but there is
a huge unlabeled Sentinel-2 archive. Recipe (SemiSiROC-style pre-detection
pseudo-labeling, but with a SUBSPACE teacher):
  1. TEACHER (label-free) makes a change map on each MultiSenGE bitemporal pair.
  2. Confidence-filter -> pseudo-labels.
  3. PRE-TRAIN an FC-EF student on the large MultiSenGE pseudo-label set (no human labels).
  4. FINE-TUNE on OSCD's few real labels (n_train cities).
  5. Evaluate on OSCD test.

Questions:
  (i) does MultiSenGE pseudo-label pre-training improve LABEL EFFICIENCY (gain biggest at small n_train)?
  (ii) is the DS-fusion teacher better than a classical (smoothed-PCA) teacher? (DS-specificity)

Channels: MultiSenGE has 10 S2 bands [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12];
OSCD is subset to the SAME 10 bands so the student is channel-compatible across
pre-train and fine-tune.

Stages (so CPU pseudo-labeling runs while a GPU job is busy):
  --stage precompute : cache OSCD 10-band recs + MultiSenGE pseudo-label recs (CPU)
  --stage train      : pre-train + fine-tune + eval (GPU)
  --stage all        : both

Run:
  python -m phase1.experiments.multisenge_pretrain --stage precompute --n_pairs 800
  python -m phase1.experiments.multisenge_pretrain --stage train --teachers fusion_ds,spca --budgets 2,14 --seeds 0
"""
from __future__ import annotations

import argparse, json, time, warnings, copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn import metrics as skm

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.baselines.ir_mad import ir_mad_score
from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.oscd_dataset import OSCDEvaluatorDataset, OFFICIAL_TRAIN, OFFICIAL_TEST
from phase1.data.preprocessing import load_band_stats, reindex_stats, build_valid_mask
from phase1.data.multisenge_dataset import scan_multisenge_s2, select_pairs, load_pair
from phase1.ds import pca_utils
from phase1.experiments.unet_ds_prior import UNet, sample_patch, masked_loss, infer_city, eval_city, _rank_norm

warnings.filterwarnings("ignore")

OUT = ROOT / "phase1" / "outputs" / "multisenge_pretrain"
MS_CACHE = OUT / "ms_cache"
OSCD_CACHE = OUT / "oscd10_cache"
MS_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
OSCD_FULL = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
OSCD_IDX = [OSCD_FULL.index(b) for b in MS_BANDS]  # [1,2,3,4,5,6,7,12,10,11]
RANK = 9  # 10 bands -> max centered rank 9


def band_image_ds_map(x1, x2, valid, seed=1234):
    rows, cols = np.where(valid)
    a = x1[:, rows, cols].T.astype(np.float32)
    b = x2[:, rows, cols].T.astype(np.float32)
    mr = max(1, min(RANK, a.shape[1] - 1))
    pre = pca_utils.fit_pca_basis(a, rank=mr, variance_threshold=None, random_state=seed, use_randomized=True)
    post = pca_utils.fit_pca_basis(b, rank=mr, variance_threshold=None, random_state=seed, use_randomized=True)
    D = pca_utils.build_difference_subspace(pre.basis, post.basis, variant="canonical")
    diff = b - a
    proj = np.zeros_like(diff) if D.shape[1] == 0 else D @ (D.T @ diff)
    out = np.zeros(valid.shape, np.float32)
    out[rows, cols] = np.sqrt(np.maximum((proj * proj).sum(1), 0.0)).astype(np.float32)
    return _rank_norm(out, valid)


def priors(x1, x2, valid, seed=1234):
    ds = band_image_ds_map(x1, x2, valid, seed)
    pca = pca_diff_score(x1, x2, valid, variance_threshold=0.95, random_state=seed).astype(np.float32)
    spca = gaussian_filter(pca, sigma=1.0); spca[~valid] = 0.0; spca = _rank_norm(spca, valid)
    try:
        irm = _rank_norm(ir_mad_score(x1, x2, valid, iters=4, random_state=seed).astype(np.float32), valid)
    except Exception:
        irm = np.zeros(valid.shape, np.float32)
    return {"ds": ds, "spca": spca, "irmad": irm}


def teacher_map(name, pr, valid):
    if name == "spca":
        return pr["spca"]
    if name == "fusion_ds":
        return _rank_norm(np.mean([pr["spca"], pr["ds"], pr["irmad"]], axis=0).astype(np.float32), valid)
    raise ValueError(name)


def make_pseudo(tmap, valid, k_pos=0.10, k_neg=0.50):
    valid = valid.astype(bool)
    pos = valid & (tmap >= 1.0 - k_pos)
    neg = valid & (tmap <= k_neg)
    y = np.zeros(valid.shape, np.uint8); y[pos] = 1
    return y, (pos | neg).astype(np.uint8)


def precompute(args, stats10):
    OSCD_CACHE.mkdir(parents=True, exist_ok=True); MS_CACHE.mkdir(parents=True, exist_ok=True)
    # ---- OSCD 10-band recs (real labels) ----
    for split, cities in (("train", OFFICIAL_TRAIN), ("test", OFFICIAL_TEST)):
        ds = OSCDEvaluatorDataset(root=args.oscd_root, split=split, band_order=OSCD_FULL, nodata_value=0.0, min_valid_bands=3)
        for c in cities:
            fp = OSCD_CACHE / f"{c}.npz"
            if fp.exists():
                continue
            s = ds.load_city(c)
            x1 = (s.x_pre[OSCD_IDX] - stats10.mean[:, None, None]) / (stats10.std[:, None, None] + 1e-6)
            x2 = (s.x_post[OSCD_IDX] - stats10.mean[:, None, None]) / (stats10.std[:, None, None] + 1e-6)
            x1[:, ~s.valid_mask] = 0; x2[:, ~s.valid_mask] = 0
            y = (s.y[0] if s.y is not None else np.zeros(s.valid_mask.shape, np.uint8)).astype(np.uint8)
            np.savez_compressed(fp, x1=x1.astype(np.float32), x2=x2.astype(np.float32),
                                y=y, valid=s.valid_mask.astype(np.uint8))
        print(f"[oscd10] {split} cached", flush=True)
    # ---- MultiSenGE pseudo-label recs ----
    samples = scan_multisenge_s2(args.ms_root)
    pairs = select_pairs(samples, n_dates_required=2, pair_strategy="earliest_latest")
    rng = np.random.default_rng(0); rng.shuffle(pairs)
    pairs = pairs[:args.n_pairs]
    print(f"[ms] {len(pairs)} pairs", flush=True)
    t0 = time.time()
    for i, (key, p1, p2) in enumerate(pairs):
        fp = MS_CACHE / f"{i:05d}.npz"
        if fp.exists():
            continue
        try:
            x1, x2, valid = load_pair(p1, p2, nodata_value=0.0, min_valid_bands=3)
        except Exception:
            continue
        if x1.shape[0] != 10 or int(valid.sum()) < 200:
            continue
        x1n = (x1 - stats10.mean[:, None, None]) / (stats10.std[:, None, None] + 1e-6); x1n[:, ~valid] = 0
        x2n = (x2 - stats10.mean[:, None, None]) / (stats10.std[:, None, None] + 1e-6); x2n[:, ~valid] = 0
        pr = priors(x1n, x2n, valid)
        rec = {"x1": x1n.astype(np.float16), "x2": x2n.astype(np.float16), "valid": valid.astype(np.uint8)}
        for tname in ("fusion_ds", "spca"):
            py, pm = make_pseudo(teacher_map(tname, pr, valid), valid)
            rec[f"y_{tname}"] = py; rec[f"m_{tname}"] = pm
        np.savez_compressed(fp, **rec)
        if (i + 1) % 100 == 0:
            print(f"  [ms] {i+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[ms] pseudo-label precompute done ({time.time()-t0:.0f}s)", flush=True)


def load_oscd(cities):
    recs = []
    for c in cities:
        d = np.load(OSCD_CACHE / f"{c}.npz")
        recs.append({k: d[k] for k in d.files})
    return recs


def load_ms(teacher):
    recs = []
    for fp in sorted(MS_CACHE.glob("*.npz")):
        d = np.load(fp)
        if f"y_{teacher}" not in d.files:
            continue
        recs.append({"x1": d["x1"].astype(np.float32), "x2": d["x2"].astype(np.float32),
                     "y": d[f"y_{teacher}"], "valid": d[f"m_{teacher}"]})
    return recs


def to_input(rec):
    return np.concatenate([rec["x1"], rec["x2"]], 0).astype(np.float32)


def train_loop(model, recs, inputs, steps, lr, device, seed, batch=16, patch=96, p_change=0.6):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    tot_pos = sum(int(((r["y"] == 1) & r["valid"].astype(bool)).sum()) for r in recs)
    tot_val = sum(int(r["valid"].astype(bool).sum()) for r in recs)
    pw = float(np.clip((tot_val - tot_pos) / max(1, tot_pos), 1.0, 50.0))
    n = len(inputs); model.train()
    for step in range(steps):
        xs, ys, vs = [], [], []
        for _ in range(batch):
            ci = int(rng.integers(n))
            xi, yi, vi = sample_patch(inputs[ci], recs[ci], rng, patch, p_change)
            xs.append(xi); ys.append(yi); vs.append(vi)
        x = torch.from_numpy(np.stack(xs)).to(device); y = torch.from_numpy(np.stack(ys)).to(device); v = torch.from_numpy(np.stack(vs)).to(device)
        loss = masked_loss(model(x), y, v, pw)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def evaluate(model, test_recs, device):
    cm = [eval_city(infer_city(model, to_input(r), device), r) for r in test_recs]
    cm = [m for m in cm if m]
    return {k: float(np.mean([c[k] for c in cm])) for k in ["ap", "auroc", "best_f1", "iou"]}


def train(args, device):
    test_recs = load_oscd(OFFICIAL_TEST)
    te_in = [to_input(r) for r in test_recs]  # noqa (infer uses to_input)
    budgets = [int(b) for b in args.budgets.split(",")]
    teachers = [t.strip() for t in args.teachers.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    results = {}
    OUT.mkdir(parents=True, exist_ok=True)
    out_fp = OUT / f"results_{args.tag}.json"

    for seed in seeds:
        # pre-train one student per teacher (shared across budgets)
        pretrained = {}
        for tname in teachers:
            ms = load_ms(tname)
            if not ms:
                print(f"[warn] no MS recs for {tname}; run precompute first"); continue
            ms_in = [to_input(r) for r in ms]
            m = UNet(ms_in[0].shape[0], base=32).to(device)
            ts = time.time()
            m = train_loop(m, ms, ms_in, args.pretrain_steps, args.lr, device, seed)
            pretrained[tname] = copy.deepcopy(m.state_dict())
            ev = evaluate(m, test_recs, device)
            print(f"  [pretrain {tname} seed{seed}] MS-pretrained OSCD-test AP {ev['ap']:.4f} (no finetune; {time.time()-ts:.0f}s)", flush=True)
            results.setdefault(f"pretrained_only:{tname}", {})[f"seed{seed}"] = ev
        # fine-tune at each budget, + scratch baseline
        for nb in budgets:
            tr = load_oscd(OFFICIAL_TRAIN[:nb] if nb > 0 else OFFICIAL_TRAIN)
            tr_in = [to_input(r) for r in tr]
            # scratch
            m = UNet(tr_in[0].shape[0], base=32).to(device)
            m = train_loop(m, tr, tr_in, args.finetune_steps, args.lr, device, seed)
            results.setdefault(f"scratch:n{nb}", {})[f"seed{seed}"] = evaluate(m, test_recs, device)
            print(f"  [scratch n{nb} seed{seed}] AP {results[f'scratch:n{nb}'][f'seed{seed}']['ap']:.4f}", flush=True)
            # fine-tune from each pretrain
            for tname, sd in pretrained.items():
                m = UNet(tr_in[0].shape[0], base=32).to(device); m.load_state_dict(sd)
                m = train_loop(m, tr, tr_in, args.finetune_steps, args.lr * 0.5, device, seed)
                ev = evaluate(m, test_recs, device)
                results.setdefault(f"pretrain_{tname}:n{nb}", {})[f"seed{seed}"] = ev
                print(f"  [pretrain_{tname} -> ft n{nb} seed{seed}] AP {ev['ap']:.4f}", flush=True)
            with out_fp.open("w") as f:
                json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_fp}")
    # summary table
    print("\n=== MultiSenGE pretrain -> OSCD finetune (test AP, mean over seeds) ===")
    def mean_ap(key):
        if key not in results: return float("nan")
        return float(np.mean([v["ap"] for v in results[key].values()]))
    for nb in budgets:
        print(f"  n_train={nb}: scratch {mean_ap(f'scratch:n{nb}'):.4f}  "
              + "  ".join(f"pretrain_{t} {mean_ap(f'pretrain_{t}:n{nb}'):.4f}" for t in teachers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["precompute", "train", "all"], default="all")
    ap.add_argument("--oscd_root", type=Path, default=ROOT / "data" / "OSCD")
    ap.add_argument("--ms_root", type=Path, default=ROOT / "data" / "MultiSenGE" / "s2")
    ap.add_argument("--stats_path", type=Path, default=ROOT / "phase1" / "data" / "oscd_band_stats.json")
    ap.add_argument("--n_pairs", type=int, default=800)
    ap.add_argument("--teachers", default="fusion_ds,spca")
    ap.add_argument("--budgets", default="2,14")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--pretrain_steps", type=int, default=3000)
    ap.add_argument("--finetune_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    stats_full = load_band_stats(args.stats_path)
    stats10 = reindex_stats(stats_full, OSCD_FULL, MS_BANDS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.stage in ("precompute", "all"):
        precompute(args, stats10)
    if args.stage in ("train", "all"):
        train(args, device)


if __name__ == "__main__":
    main()
