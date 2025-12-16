"""
Evaluate trained OSCD segmentation models (spec Section 6.2).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

try:  # pragma: no cover
  from rasterio.errors import NotGeoreferencedWarning

  warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except Exception:  # pragma: no cover
  pass

from phase1.data.oscd_dataset import OSCDEvaluatorDataset

from phase2.data.oscd_seg_dataset import OSCDSegmentationDataset
from phase2.models.unet2d import UNet2D
from phase2.models.unet2d_resnet_backbone import UNet2DResNetBackbone
from phase2.models.priors_fusion_heads import PriorsFusionUNet
from phase2.models.siamese_unet import SiameseUNet2D
from phase2.eval.metrics_segmentation import auroc_score, binary_stats, pr_auc_score


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", type=Path, required=True)
  ap.add_argument("--oscd_root", type=Path, required=True)
  ap.add_argument("--phase1_change_maps_root", type=Path, required=True)
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--output_dir", type=Path, required=True)
  ap.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to use: cuda, cuda:0, cpu, or auto (default: cuda).",
  )
  return ap.parse_args()


def load_config(path: Path) -> Dict:
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f)

def git_hash() -> str:
  try:
    import subprocess

    return (
      subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
      .decode("utf-8")
      .strip()
    )
  except Exception:
    return "unknown"

def _default_band_order() -> list[str]:
  return ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]

def resolve_device(device_str: str) -> torch.device:
  s = str(device_str or "").strip().lower()
  if s == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(device_str)

def ensure_cuda_available(device: torch.device) -> None:
  if device.type != "cuda":
    return
  if torch.cuda.is_available():
    return
  raise RuntimeError(
    "CUDA device requested but torch.cuda.is_available() is False. "
    "This usually means CPU-only PyTorch is installed (e.g., torch==...+cpu). "
    "Reinstall a CUDA-enabled PyTorch build: https://pytorch.org/get-started/locally/"
  )


def build_model(cfg: Dict, in_channels: int) -> torch.nn.Module:
  model_cfg = cfg["model"]
  model_type = model_cfg.get("type", "unet2d")
  num_classes = model_cfg.get("num_classes", 1)
  if model_type == "unet2d":
    base_ch = model_cfg.get("base_channels", 64)
    depth = model_cfg.get("depth", 4)
    return UNet2D(in_channels=in_channels, base_channels=base_ch, depth=depth, num_classes=num_classes)
  if model_type == "unet2d_resnet":
    pretrained = model_cfg.get("pretrained", False)
    return UNet2DResNetBackbone(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
  if model_type == "priors_fusion_unet":
    counts = infer_channel_counts(cfg)
    n_raw, n_priors = counts["n_raw"], counts["n_priors"]
    base_ch = model_cfg.get("base_channels", 64)
    depth = model_cfg.get("depth", 4)
    return PriorsFusionUNet(n_raw=n_raw, n_priors=n_priors, base_channels=base_ch, depth=depth, num_classes=num_classes)
  if model_type == "siamese_unet":
    band_order = cfg.get("dataset", {}).get("band_order") or _default_band_order()
    n_bands = int(len(band_order))
    base_ch = model_cfg.get("base_channels", 64)
    depth = model_cfg.get("depth", 4)
    return SiameseUNet2D(n_bands=n_bands, base_channels=base_ch, depth=depth, num_classes=num_classes)
  raise NotImplementedError(f"Unknown model type: {model_type}")


def infer_channel_counts(cfg: Dict) -> Dict[str, int]:
  band_order = cfg.get("dataset", {}).get("band_order") or _default_band_order()
  n_bands = int(len(band_order))
  feats = cfg["features"]
  n_raw = 0
  if feats.get("use_raw_s2", True):
    if feats.get("use_pre_post_stack", True):
      n_raw += 2 * n_bands
    else:
      n_raw += n_bands
  priors = feats.get("priors", {})
  n_priors = 0
  for _, enabled in priors.items():
    if enabled:
      n_priors += 1
  return {"n_raw": n_raw, "n_priors": n_priors, "total": n_raw + n_priors}

def _stitch_city_probs(city_preds: list[tuple[np.ndarray, int, int]], shape_hw: tuple[int, int]) -> np.ndarray:
  h, w = shape_hw
  acc = np.zeros((h, w), dtype=np.float32)
  counts = np.zeros((h, w), dtype=np.float32)
  for probs, y0, x0 in city_preds:
    ph, pw = probs.shape
    ys = slice(y0, min(y0 + ph, h))
    xs = slice(x0, min(x0 + pw, w))
    h_eff = ys.stop - ys.start
    w_eff = xs.stop - xs.start
    if h_eff <= 0 or w_eff <= 0:
      continue
    acc[ys, xs] += probs[:h_eff, :w_eff]
    counts[ys, xs] += 1.0
  counts[counts == 0] = 1.0
  return acc / counts


def main():
  args = parse_args()
  cfg = load_config(args.config)
  out_dir = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  device = resolve_device(args.device)
  ensure_cuda_available(device)
  if device.type == "cuda":
    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    print(f"Using device: {device} ({name})")
  else:
    print(f"Using device: {device}", file=sys.stderr)
  use_cuda = device.type == "cuda"

  counts = infer_channel_counts(cfg)
  in_ch = counts["total"]
  model = build_model(cfg, in_ch).to(device)

  ckpt = torch.load(args.checkpoint, map_location=device)
  model.load_state_dict(ckpt["model_state"])
  model.eval()

  results: Dict[str, Dict[str, Dict]] = {}

  eval_thr = float(cfg.get("evaluation", {}).get("threshold", 0.5))
  band_order = cfg.get("dataset", {}).get("band_order") or _default_band_order()

  for split in ["val", "test"]:
    # Full-tile loader for GT + masks.
    full_ds = OSCDEvaluatorDataset(
      args.oscd_root,
      split,
      band_order,
      val_cities=cfg["dataset"]["split"].get("val", []),
      val_from_train=0,
    )
    city_info: dict[str, dict] = {}
    for city in full_ds.cities:
      sample = full_ds.load_city(city)
      h, w = sample.x_pre.shape[1:]
      y_full = sample.y[0].astype(np.uint8) if sample.y is not None else np.zeros((h, w), dtype=np.uint8)
      city_info[city] = {"shape": (h, w), "y": y_full, "valid": sample.valid_mask.astype(bool)}

    # Patch dataset for inference. Note: OSCDSegmentationDataset applies random
    # augmentations only when split == "train" (evaluation is deterministic).
    patch_ds = OSCDSegmentationDataset(
      args.oscd_root,
      split,
      cfg,
      phase1_change_maps_root=args.phase1_change_maps_root,
    )
    eval_bs = int(cfg.get("evaluation", {}).get("batch_size", 4))
    city_to_indices: dict[str, list[int]] = {}
    for i, (c, _, _) in enumerate(patch_ds.patches):
      city_to_indices.setdefault(c, []).append(i)

    split_res: dict[str, dict] = {}
    with torch.no_grad():
      for city, idxs in city_to_indices.items():
        info = city_info.get(city)
        if info is None:
          continue
        h, w = info["shape"]
        acc = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        subset = Subset(patch_ds, idxs)
        loader = DataLoader(subset, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=use_cuda)
        for bi, batch in enumerate(loader):
          x = batch["x"].to(device, non_blocking=use_cuda)
          logits = model(x)
          if logits.shape[-2:] != x.shape[-2:]:
            logits = torch.nn.functional.interpolate(
              logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
          probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)  # (B,H,W)
          base = bi * eval_bs
          for j in range(probs.shape[0]):
            _, y0, x0 = patch_ds.patches[idxs[base + j]]
            ph, pw = probs[j].shape
            ys = slice(y0, min(y0 + ph, h))
            xs = slice(x0, min(x0 + pw, w))
            h_eff = ys.stop - ys.start
            w_eff = xs.stop - xs.start
            if h_eff <= 0 or w_eff <= 0:
              continue
            acc[ys, xs] += probs[j][:h_eff, :w_eff]
            counts[ys, xs] += 1.0

        counts[counts == 0] = 1.0
        prob_full = acc / counts
        pred_mask = (prob_full >= eval_thr).astype(np.uint8)
        stats = binary_stats(pred_mask, info["y"], info["valid"])
        stats["auroc"] = auroc_score(prob_full, info["y"], info["valid"])
        stats["pr_auc"] = pr_auc_score(prob_full, info["y"], info["valid"])
        split_res[city] = stats

    if split_res:
      mean_iou = float(np.mean([v["iou"] for v in split_res.values()]))
      mean_f1 = float(np.mean([v["f1"] for v in split_res.values()]))
      mean_auroc = float(np.mean([v["auroc"] for v in split_res.values()]))
      mean_pr_auc = float(np.mean([v["pr_auc"] for v in split_res.values()]))
    else:
      mean_iou = mean_f1 = mean_auroc = mean_pr_auc = float("nan")
    results[split] = {
      "per_city": split_res,
      "summary": {
        "mean_iou": mean_iou,
        "mean_f1": mean_f1,
        "mean_auroc": mean_auroc,
        "mean_pr_auc": mean_pr_auc,
        "threshold": eval_thr,
      },
    }

  with (out_dir / "oscd_seg_eval_results.json").open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

  model_type = cfg.get("model", {}).get("type", "unknown")
  exp_tag = cfg.get("experiment_tag", "")
  with (out_dir / "oscd_seg_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=["split", "model", "experiment_tag", "threshold", "mean_iou", "mean_f1", "mean_auroc", "mean_pr_auc"],
    )
    writer.writeheader()
    for split in ["val", "test"]:
      s = results.get(split, {}).get("summary", {})
      writer.writerow(
        {
          "split": split,
          "model": model_type,
          "experiment_tag": exp_tag,
          "threshold": s.get("threshold", eval_thr),
          "mean_iou": s.get("mean_iou"),
          "mean_f1": s.get("mean_f1"),
          "mean_auroc": s.get("mean_auroc"),
          "mean_pr_auc": s.get("mean_pr_auc"),
        }
      )

  meta = {
    "config_path": str(args.config),
    "config": cfg,
    "oscd_root": str(args.oscd_root),
    "phase1_change_maps_root": str(args.phase1_change_maps_root),
    "checkpoint": str(args.checkpoint),
    "device": str(device),
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "python_version": sys.version,
    "git_hash": git_hash(),
  }
  with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)


if __name__ == "__main__":
  main()
