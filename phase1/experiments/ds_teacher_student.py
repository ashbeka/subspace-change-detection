"""
Subspace-teacher self-training for label-free change detection on OSCD.

Motivation (the real-world driver)
----------------------------------
For a *new* disaster there are no labels, only free Sentinel-2. The established
label-scarce paradigm is teacher->student / pre-detection pseudo-labeling: an
unsupervised algorithm (the TEACHER) produces a change map, it is confidence-
filtered into pseudo-labels, and a deep STUDENT is trained on them with NO human
labels (e.g. SemiSiROC 2023; CVA/PCA+KMeans/DCVA/DSFA pre-detection;
survey arXiv:2502.02835).

Question
--------
Does a *subspace-informed* teacher -- specifically the DS-fusion map
(smoothed-PCA + Band-Image DS + IR-MAD), our strongest unsupervised map -- train
a better student than classical teachers (CVA, IR-MAD, smoothed-PCA), and is any
advantage DS-SPECIFIC (fusion_ds vs fusion_nods vs fusion_cross)?

This is the fully-unsupervised counterpart to unet_ds_prior.py: there the DS map
was an INPUT feature; here it is the TEACHER. The student only ever sees raw
bands [pre,post], so there is no input-leakage of the teacher map -- the teacher
signal must be *distilled* into a model that works on raw imagery.

Teachers (all label-free)
-------------------------
  cva          raw L2 / CVA magnitude                 (naive teacher)
  irmad        IR-MAD                                  classical
  spca         smoothed PCA-diff (sigma=1)             strongest single classical map
  ds           Band-Image DS                           our spatial subspace map
  fusion_ds    rank-fuse(spca, ds, irmad)              OUR best map / teacher
  fusion_nods  rank-fuse(spca, irmad)                  fusion WITHOUT DS (control)
  fusion_cross rank-fuse(spca, cross, irmad)           DS replaced by matched null (DS-specificity)

Reads: student(fusion_ds) vs student(spca/cva/irmad)  -> does a subspace teacher help?
       student(fusion_ds) vs student(fusion_nods/cross) -> is it DS-specific?
       student vs raw teacher-map AP                   -> does distillation denoise/improve?

Run:
  .\.venv\Scripts\python.exe -m phase1.experiments.ds_teacher_student \
      --teachers cva,irmad,spca,ds,fusion_ds,fusion_nods,fusion_cross --seeds 0,1,2 --steps 2500
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
from sklearn import metrics as skm

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data.oscd_dataset import OFFICIAL_TRAIN, OFFICIAL_TEST
from phase1.data.preprocessing import load_band_stats
from phase1.experiments.unet_ds_prior import (
    OUT_ROOT, UNet, eval_city, infer_city, masked_loss, precompute_city,
    sample_patch, _rank_norm,
)

warnings.filterwarnings("ignore")

TEACHERS = ["cva", "irmad", "spca", "ds", "fusion_ds", "fusion_nods", "fusion_cross"]


def cva_map(rec: dict) -> np.ndarray:
    valid = rec["valid"].astype(bool)
    diff = rec["x2"] - rec["x1"]
    m = np.sqrt(np.sum(diff * diff, axis=0)).astype(np.float32)
    return _rank_norm(m, valid)


def teacher_map(name: str, rec: dict) -> np.ndarray:
    """Return a label-free [0,1] rank-normalized teacher score map for a city."""
    valid = rec["valid"].astype(bool)
    if name == "cva":
        return cva_map(rec)
    if name in ("irmad", "spca", "ds", "cross"):
        return rec[name].astype(np.float32)
    if name in ("fusion_ds", "fusion_nods", "fusion_cross"):
        comps = {"fusion_ds": ["spca", "ds", "irmad"],
                 "fusion_nods": ["spca", "irmad"],
                 "fusion_cross": ["spca", "cross", "irmad"]}[name]
        avg = np.mean([rec[c].astype(np.float32) for c in comps], axis=0)
        return _rank_norm(avg, valid)
    raise ValueError(f"unknown teacher {name}")


def make_pseudo(tmap: np.ndarray, valid: np.ndarray, k_pos: float, k_neg: float):
    """Double-threshold confidence filter: top k_pos -> 1, bottom k_neg -> 0, middle -> ignore."""
    valid = valid.astype(bool)
    pos = valid & (tmap >= (1.0 - k_pos))
    neg = valid & (tmap <= k_neg)
    pseudo_y = np.zeros(valid.shape, np.uint8)
    pseudo_y[pos] = 1
    pseudo_mask = (pos | neg)  # confident pixels only; uncertain middle ignored in loss
    return pseudo_y, pseudo_mask.astype(np.uint8)


def train_student(seed, pseudo_recs, inputs, args, device) -> torch.nn.Module:
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet(inputs[0].shape[0], base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    tot_pos = sum(int(((r["y"] == 1) & r["valid"].astype(bool)).sum()) for r in pseudo_recs)
    tot_val = sum(int(r["valid"].astype(bool).sum()) for r in pseudo_recs)
    pw = float(np.clip((tot_val - tot_pos) / max(1, tot_pos), 1.0, 50.0))
    n = len(inputs)
    model.train()
    for step in range(args.steps):
        xs, ys, vs = [], [], []
        for _ in range(args.batch):
            ci = int(rng.integers(n))
            xi, yi, vi = sample_patch(inputs[ci], pseudo_recs[ci], rng, args.patch, args.p_change)
            xs.append(xi); ys.append(yi); vs.append(vi)
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        v = torch.from_numpy(np.stack(vs)).to(device)
        loss = masked_loss(model(x), y, v, pw)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def map_ap(tmap: np.ndarray, rec: dict) -> dict:
    valid = rec["valid"].astype(bool)
    y = rec["y"][valid].astype(np.uint8).ravel()
    s = tmap[valid].astype(np.float64).ravel()
    if y.size == 0 or np.unique(y).size < 2:
        return {}
    return {"ap": float(skm.average_precision_score(y, s)),
            "auroc": float(skm.roc_auc_score(y, s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oscd_root", type=Path, default=ROOT / "data" / "OSCD")
    ap.add_argument("--stats_path", type=Path, default=ROOT / "phase1" / "data" / "oscd_band_stats.json")
    ap.add_argument("--teachers", default=",".join(TEACHERS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p_change", type=float, default=0.6)
    ap.add_argument("--k_pos", type=float, default=0.10, help="top fraction -> pseudo positive")
    ap.add_argument("--k_neg", type=float, default=0.50, help="bottom fraction -> pseudo negative")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stats = load_band_stats(args.stats_path)

    train_recs = [precompute_city(c, "train", args.oscd_root, stats, seed=1234) for c in OFFICIAL_TRAIN]
    test_recs = [precompute_city(c, "test", args.oscd_root, stats, seed=1234) for c in OFFICIAL_TEST]
    # student inputs are RAW bands only (no teacher leakage)
    tr_in = [np.concatenate([r["x1"], r["x2"]], 0).astype(np.float32) for r in train_recs]
    te_in = [np.concatenate([r["x1"], r["x2"]], 0).astype(np.float32) for r in test_recs]
    targs = SimpleNamespace(steps=args.steps, batch=args.batch, patch=args.patch,
                            base=args.base, lr=args.lr, p_change=args.p_change)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_fp = OUT_ROOT / f"teacher_student_{args.tag}.json"
    results = {}
    for tname in teachers:
        # reference: raw teacher-map AP on test (no student)
        tmap_ap = [map_ap(teacher_map(tname, r), r) for r in test_recs]
        tmap_ap = [m for m in tmap_ap if m]
        ref = {"ap": float(np.mean([m["ap"] for m in tmap_ap])),
               "auroc": float(np.mean([m["auroc"] for m in tmap_ap]))}
        # build pseudo-labelled train recs
        pseudo_recs = []
        pos_rates = []
        for r in train_recs:
            tm = teacher_map(tname, r)
            py, pm = make_pseudo(tm, r["valid"], args.k_pos, args.k_neg)
            pseudo_recs.append({"y": py, "valid": pm})
            pr = float((py & pm.astype(bool)).sum()) / max(1, int(pm.sum()))
            pos_rates.append(pr)
        per_seed = []
        for seed in seeds:
            ts = time.time()
            model = train_student(seed, pseudo_recs, tr_in, targs, device)
            cm = [eval_city(infer_city(model, xin, device), r) for r, xin in zip(test_recs, te_in)]
            cm = [m for m in cm if m]
            agg = {k: float(np.mean([c[k] for c in cm])) for k in ["ap", "auroc", "best_f1", "iou"]}
            per_seed.append(agg)
            print(f"  [{tname} seed{seed}] student AP {agg['ap']:.4f} F1 {agg['best_f1']:.4f} "
                  f"(teacher-map AP {ref['ap']:.4f}; {time.time()-ts:.0f}s)", flush=True)
        summ = {k: {"mean": float(np.mean([p[k] for p in per_seed])),
                    "std": float(np.std([p[k] for p in per_seed]))}
                for k in ["ap", "auroc", "best_f1", "iou"]}
        results[tname] = {"teacher_map": ref, "student": summ,
                          "pseudo_pos_rate": float(np.mean(pos_rates)), "per_seed": per_seed}
        print(f"==> teacher={tname:<12} teacher-map AP {ref['ap']:.4f} | "
              f"student AP {summ['ap']['mean']:.4f}+-{summ['ap']['std']:.4f} "
              f"F1 {summ['best_f1']['mean']:.4f}", flush=True)
        # checkpoint after every teacher so a crash/sleep never wipes progress
        with out_fp.open("w") as f:
            json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "results": results}, f, indent=2)

    print(f"\nsaved -> {out_fp}")

    print("\n=== Teacher->Student (10 official test cities) ===")
    print(f"{'teacher':<14}{'pseudoPos%':>11}{'teacherMapAP':>14}{'studentAP':>12}{'studentF1':>12}{'gain(stu-map)':>14}")
    for tname in teachers:
        if tname not in results:
            continue
        r = results[tname]
        gain = r["student"]["ap"]["mean"] - r["teacher_map"]["ap"]
        print(f"{tname:<14}{100*r['pseudo_pos_rate']:>10.1f}%{r['teacher_map']['ap']:>14.4f}"
              f"{r['student']['ap']['mean']:>12.4f}{r['student']['best_f1']['mean']:>12.4f}{gain:>+14.4f}")


if __name__ == "__main__":
    main()
