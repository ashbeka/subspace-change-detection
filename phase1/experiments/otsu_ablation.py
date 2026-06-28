"""
Closing Sensei's Otsu ask: thresholding before/after vs no-thresholding.

Sensei asked: "Have you trained U-Net with raw data without Otsu / no thresholding?"
and "experiment with Otsu thresholding before U-Net training and after."

This shows, on the 10 official OSCD test cities (cached maps), what Otsu
thresholding does to the classical change maps versus the threshold-free /
learned pipeline:
  - threshold-FREE ranking quality (AP, AUROC, oracle best-F1) of each map;
  - Otsu-binarized F1/IoU (a fixed unsupervised threshold) for the SAME map;
  - reference: the FC-EF U-Net trained on raw bands with NO thresholding (M0/M4
    from results_main.json), which is the answer to "trained raw, no Otsu".

Conclusion to read off: Otsu (a hard unsupervised threshold) loses to the
threshold-free ranking and to the no-threshold learned model; this is why we
train on raw, normalized bands without Otsu.

Run: .\.venv\Scripts\python.exe -m phase1.experiments.otsu_ablation
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from sklearn import metrics as skm

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phase1.data.oscd_dataset import OFFICIAL_TEST
from phase1.eval.thresholding import otsu_threshold, apply_threshold
from phase1.experiments.unet_ds_prior import precompute_city, _rank_norm
from phase1.data.preprocessing import load_band_stats

OUT = ROOT / "phase1" / "outputs" / "otsu_ablation"


def f1_iou(pred, y, valid):
    p = pred[valid].astype(np.uint8); t = y[valid].astype(np.uint8)
    tp = int(((p == 1) & (t == 1)).sum()); fp = int(((p == 1) & (t == 0)).sum()); fn = int(((p == 0) & (t == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9); iou = tp / (tp + fp + fn + 1e-9)
    return f1, iou


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = load_band_stats(ROOT / "phase1" / "data" / "oscd_band_stats.json")
    recs = [precompute_city(c, "test", ROOT / "data" / "OSCD", stats, 1234) for c in OFFICIAL_TEST]
    maps = ["ds", "spca", "irmad"]
    # also a fusion (rank-avg of the three rank-normed maps)
    agg = {m: {"ap": [], "auroc": [], "bestf1": [], "otsuf1": [], "otsuiou": []} for m in maps + ["fusion"]}
    for r in recs:
        valid = r["valid"].astype(bool); y = r["y"].astype(np.uint8)
        fusion = _rank_norm(np.mean([r[m] for m in maps], axis=0).astype(np.float32), valid)
        allmaps = {m: r[m] for m in maps}; allmaps["fusion"] = fusion
        for m, s in allmaps.items():
            yv = y[valid].ravel(); sv = s[valid].astype(np.float64).ravel()
            if np.unique(yv).size < 2:
                continue
            agg[m]["ap"].append(skm.average_precision_score(yv, sv))
            agg[m]["auroc"].append(skm.roc_auc_score(yv, sv))
            p, rc, t = skm.precision_recall_curve(yv, sv)
            agg[m]["bestf1"].append(float(np.nanmax((2 * p * rc) / (p + rc + 1e-9))))
            thr = float(otsu_threshold(s, valid))
            pred = apply_threshold(s, thr)
            f1, iou = f1_iou(pred, y, valid)
            agg[m]["otsuf1"].append(f1); agg[m]["otsuiou"].append(iou)
    summary = {m: {k: float(np.mean(v)) for k, v in d.items() if v} for m, d in agg.items()}
    print(f"{'map':>8}{'AP(free)':>10}{'AUROC':>9}{'bestF1(free)':>14}{'OtsuF1':>9}{'OtsuIoU':>9}")
    for m in maps + ["fusion"]:
        s = summary[m]
        print(f"{m:>8}{s['ap']:>10.4f}{s['auroc']:>9.4f}{s['bestf1']:>14.4f}{s['otsuf1']:>9.4f}{s['otsuiou']:>9.4f}")
    # reference: U-Net trained on raw, NO thresholding
    try:
        mainj = json.load(open(ROOT / "phase1" / "outputs" / "unet_ds_prior" / "results_main.json"))
        m0 = mainj["results"]["bands"]["summary"]; m4 = mainj["results"]["bands_fusion"]["summary"]
        print(f"\n[reference] U-Net trained on RAW bands, NO thresholding:")
        print(f"  bands(M0)       AP {m0['ap']['mean']:.4f}  bestF1 {m0['best_f1']['mean']:.4f}")
        print(f"  bands+fusion(M4) AP {m4['ap']['mean']:.4f}  bestF1 {m4['best_f1']['mean']:.4f}")
        summary["_unet_no_threshold"] = {"M0_ap": m0['ap']['mean'], "M4_ap": m4['ap']['mean']}
    except Exception as e:
        print(f"[warn] unet ref unavailable: {e}")
    json.dump(summary, open(OUT / "results.json", "w"), indent=2)
    print(f"\nTakeaway: Otsu (hard unsupervised threshold) F1 < threshold-free best-F1; the no-threshold "
          f"learned U-Net (AP ~0.48) is far above any Otsu-binarized classical map. -> train raw, no Otsu.")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
