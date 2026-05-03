"""
Train OSCD segmentation model for Phase 2 (spec Section 5).
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:  # pragma: no cover
  from rasterio.errors import NotGeoreferencedWarning

  warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except Exception:  # pragma: no cover
  pass

try:
  from tqdm import tqdm
  _TQDM_AVAILABLE = True
except Exception:  # pragma: no cover
  def tqdm(x, *args, **kwargs):
    return x
  _TQDM_AVAILABLE = False

from phase2.data.oscd_seg_dataset import OSCDSegmentationDataset
from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase2.models.unet2d import UNet2D
from phase2.models.unet2d_resnet_backbone import UNet2DResNetBackbone
from phase2.models.priors_fusion_heads import PriorsFusionUNet
from phase2.models.siamese_unet import SiameseUNet2D
from phase2.train.losses import BCEDiceLoss
from phase2.train.optimizer_schedules import build_optimizer, build_scheduler
from phase2.train.callbacks import ModelCheckpoint, MetricsLogger
from phase2.eval.metrics_segmentation import auroc_score, binary_stats, pr_auc_score


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", type=Path, required=True)
  ap.add_argument("--oscd_root", type=Path, required=True)
  ap.add_argument("--phase1_change_maps_root", type=Path, required=True)
  ap.add_argument("--output_dir", type=Path, required=True)
  ap.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to use: cuda, cuda:0, cpu, or auto (default: cuda).",
  )
  ap.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override number of epochs (also updates cosine scheduler T_max to match).",
  )
  ap.add_argument("--max_epochs", type=int, default=None, help="Optional cap on number of epochs (for quick tests).")
  ap.add_argument(
    "--resume",
    action="store_true",
    help="Resume from output_dir/last.ckpt if present (else best.ckpt).",
  )
  ap.add_argument(
    "--resume_ckpt",
    type=Path,
    default=None,
    help="Explicit checkpoint path to resume from (overrides --resume).",
  )
  ap.add_argument(
    "--overwrite_output_dir",
    action="store_true",
    help="Allow writing into a non-empty output_dir without resuming (will overwrite logs/checkpoints).",
  )
  ap.add_argument(
    "--progress_style",
    choices=["tqdm", "none"],
    default="tqdm",
    help="Training progress display. Use 'tqdm' for animated batch progress bars or 'none' for quiet training.",
  )
  ap.add_argument(
    "--progress_leave",
    action="store_true",
    help="Keep completed tqdm bars in the terminal. By default only the current bar is kept to reduce terminal overflow.",
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

def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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

def _default_band_order() -> list[str]:
  return ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]


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


def compute_train_patch_weights(
  oscd_root: Path,
  cfg: Dict,
  train_patches: list[tuple[str, int, int]],
  patch_size: int,
  alpha: float = 5.0,
  power: float = 1.0,
  min_pos_fraction: float = 0.0,
) -> list[float]:
  band_order = cfg.get("dataset", {}).get("band_order") or _default_band_order()
  full_train = OSCDEvaluatorDataset(
    oscd_root,
    "train",
    band_order,
    val_cities=cfg["dataset"]["split"].get("val", []),
    val_from_train=0,
  )
  city_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  for city in cfg["dataset"]["split"]["train"]:
    sample = full_train.load_city(city)
    y = sample.y[0].astype(np.uint8) if sample.y is not None else np.zeros(sample.x_pre.shape[1:], dtype=np.uint8)
    city_cache[city] = (y, sample.valid_mask.astype(bool))

  weights = []
  for city, y0, x0 in train_patches:
    y, vm = city_cache[city]
    ys = slice(y0, y0 + patch_size)
    xs = slice(x0, x0 + patch_size)
    vm_p = vm[ys, xs]
    denom = float(vm_p.sum())
    if denom <= 0:
      pos_frac = 0.0
    else:
      pos = float((y[ys, xs][vm_p] > 0).sum())
      pos_frac = pos / denom
    pos_frac = max(float(min_pos_fraction), pos_frac)
    w = (1.0 + float(alpha) * pos_frac) ** float(power)
    weights.append(float(w))
  return weights


def evaluate_tiles(
  model: torch.nn.Module,
  split: str,
  cfg: Dict,
  patch_ds: OSCDSegmentationDataset,
  city_to_indices: dict[str, list[int]],
  city_info: dict[str, dict],
  device: torch.device,
) -> Dict[str, float]:
  model.eval()
  eval_thr = float(cfg.get("evaluation", {}).get("threshold", 0.5))
  eval_bs = int(cfg.get("evaluation", {}).get("batch_size", cfg.get("training", {}).get("batch_size", 4)))

  per_city = []
  use_cuda = device.type == "cuda"
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
          logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
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
      per_city.append(stats)

  model.train()
  if not per_city:
    return {"f1": float("nan"), "iou": float("nan"), "auroc": float("nan"), "pr_auc": float("nan")}
  return {
    "f1": float(np.mean([m["f1"] for m in per_city])),
    "iou": float(np.mean([m["iou"] for m in per_city])),
    "auroc": float(np.mean([m["auroc"] for m in per_city])),
    "pr_auc": float(np.mean([m["pr_auc"] for m in per_city])),
  }


def main():
  args = parse_args()
  cfg = load_config(args.config)
  out_dir = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  run_label = out_dir.name or str(cfg.get("experiment_tag", "run"))

  epochs = int(cfg.get("training", {}).get("epochs", 50))
  if args.epochs is not None:
    epochs = int(args.epochs)
  if args.max_epochs is not None:
    epochs = min(epochs, args.max_epochs)

  # If user overrides epochs, keep cosine schedule consistent without forcing
  # changes to the tracked YAML configs.
  scheduler_cfg = cfg.get("training", {}).get("scheduler", {}) or {}
  if args.epochs is not None and str(scheduler_cfg.get("name", "cosine")).lower() == "cosine":
    scheduler_cfg = dict(scheduler_cfg)
    scheduler_cfg["T_max"] = int(epochs)
    cfg.setdefault("training", {})
    cfg["training"]["scheduler"] = scheduler_cfg

  seed = int(cfg.get("training", {}).get("seed", cfg.get("seed", 1234)))
  set_seed(seed)

  progress_style = str(args.progress_style or "tqdm").lower()
  if progress_style == "tqdm" and not _TQDM_AVAILABLE:
    raise RuntimeError(
      "Progress bars requested with --progress_style tqdm, but tqdm is not installed. "
      "Install dependencies with: .\\.venv\\Scripts\\python.exe -m pip install -r phase2/requirements.txt"
    )

  device = resolve_device(args.device)
  ensure_cuda_available(device)
  if device.type == "cuda":
    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    print(f"Using device: {device} ({name})")
  else:
    print(f"Using device: {device}", file=sys.stderr)
  print(f"Run: {run_label} | seed: {seed} | epochs: {epochs}")
  use_cuda = device.type == "cuda"

  counts = infer_channel_counts(cfg)
  in_ch = counts["total"]
  model = build_model(cfg, in_ch).to(device)

  train_ds = OSCDSegmentationDataset(
    args.oscd_root,
    "train",
    cfg,
    phase1_change_maps_root=args.phase1_change_maps_root,
  )
  val_ds = OSCDSegmentationDataset(
    args.oscd_root,
    "val",
    cfg,
    phase1_change_maps_root=args.phase1_change_maps_root,
  )

  num_workers = cfg["training"].get("num_workers", 0)
  g = torch.Generator()
  g.manual_seed(seed)

  def _worker_init_fn(worker_id: int):
    set_seed(seed + worker_id + 1)

  sampler_cfg = cfg["training"].get("sampler", {}) or {}
  sampler_name = str(sampler_cfg.get("name", "")).lower().strip()
  if sampler_name == "weighted_pos":
    weights = compute_train_patch_weights(
      args.oscd_root,
      cfg,
      train_ds.patches,
      train_ds.patch_size,
      alpha=float(sampler_cfg.get("alpha", 5.0)),
      power=float(sampler_cfg.get("power", 1.0)),
      min_pos_fraction=float(sampler_cfg.get("min_pos_fraction", 0.0)),
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=g)
    shuffle = False
  else:
    sampler = None
    shuffle = True

  train_loader = DataLoader(
    train_ds,
    batch_size=cfg["training"]["batch_size"],
    shuffle=shuffle if sampler is None else False,
    sampler=sampler,
    num_workers=num_workers,
    generator=g,
    worker_init_fn=_worker_init_fn if num_workers else None,
    pin_memory=use_cuda,
    persistent_workers=bool(num_workers),
  )

  optimizer = build_optimizer(model.parameters(), cfg["training"]["optimizer"])
  scheduler = build_scheduler(optimizer, cfg["training"]["scheduler"])

  loss_cfg = cfg["training"]["loss"]
  criterion = BCEDiceLoss(
    bce_weight=loss_cfg.get("bce_weight", 1.0),
    dice_weight=loss_cfg.get("dice_weight", 1.0),
    smooth=loss_cfg.get("smooth", 1.0),
    pos_weight=loss_cfg.get("pos_weight"),
  )

  ckpt_cb = ModelCheckpoint(out_dir, monitor="val_iou", mode="max", save_last=True)
  log_cb = MetricsLogger(out_dir)

  resume_path = None
  if args.resume_ckpt is not None:
    resume_path = args.resume_ckpt
  elif args.resume:
    last = out_dir / "last.ckpt"
    best = out_dir / "best.ckpt"
    if last.exists():
      resume_path = last
    elif best.exists():
      resume_path = best
    else:
      raise FileNotFoundError(f"--resume specified but no checkpoint found in {out_dir}")

  if resume_path is None and not args.overwrite_output_dir:
    existing = [out_dir / "best.ckpt", out_dir / "last.ckpt", out_dir / "train_log.json"]
    if any(p.exists() for p in existing):
      raise FileExistsError(
        f"Refusing to overwrite existing run artifacts in {out_dir}. "
        f"Use --resume to continue, --overwrite_output_dir to overwrite, or choose a new --output_dir."
      )

  start_epoch_num = 1
  if resume_path is not None:
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    opt_state = ckpt.get("optimizer_state")
    if opt_state is not None:
      optimizer.load_state_dict(opt_state)
    if scheduler is not None:
      sch_state = ckpt.get("scheduler_state")
      if sch_state is not None:
        scheduler.load_state_dict(sch_state)
      else:
        sch_name = cfg["training"]["scheduler"].get("name", "cosine").lower()
        if sch_name != "plateau":
          for _ in range(int(ckpt.get("epoch", 0) or 0)):
            scheduler.step()
        else:
          print("Warning: checkpoint has no scheduler_state; plateau scheduler will restart.", file=sys.stderr)

    ckpt_cb.best = ckpt.get("best_value")
    ckpt_cb.best_epoch = ckpt.get("best_epoch")
    if ckpt_cb.best is not None:
      ckpt_cb.best = float(ckpt_cb.best)
    if ckpt_cb.best_epoch is not None:
      ckpt_cb.best_epoch = int(ckpt_cb.best_epoch)

    start_epoch_num = int(ckpt.get("epoch", 0) or 0) + 1

    log_cb.load_existing()
    if log_cb.records:
      log_cb.records = [r for r in log_cb.records if int(r.get("epoch", 0) or 0) < start_epoch_num]

    print(f"Resuming from {resume_path} (start_epoch={start_epoch_num}, target_epochs={epochs})")

  # Prebuild validation inputs once to avoid reloading tiles each epoch.
  band_order = cfg.get("dataset", {}).get("band_order") or _default_band_order()
  val_full_ds = OSCDEvaluatorDataset(
    args.oscd_root,
    "val",
    band_order,
    val_cities=cfg["dataset"]["split"].get("val", []),
    val_from_train=0,
  )
  val_city_info: dict[str, dict] = {}
  for city in val_full_ds.cities:
    sample = val_full_ds.load_city(city)
    h, w = sample.x_pre.shape[1:]
    y_full = sample.y[0].astype(np.uint8) if sample.y is not None else np.zeros((h, w), dtype=np.uint8)
    val_city_info[city] = {"shape": (h, w), "y": y_full, "valid": sample.valid_mask.astype(bool)}
  val_city_to_indices: dict[str, list[int]] = {}
  for i, (c, _, _) in enumerate(val_ds.patches):
    val_city_to_indices.setdefault(c, []).append(i)

  val_every = int(cfg["training"].get("val_every", 1) or 1)
  if val_every < 1:
    val_every = 1
  if start_epoch_num > epochs:
    print(f"Nothing to do: start_epoch={start_epoch_num} > target_epochs={epochs}")
    return
  for epoch in range(start_epoch_num, epochs + 1):
    model.train()
    epoch_t0 = time.perf_counter()
    train_loss_vals = []
    if progress_style == "tqdm":
      batch_iter = enumerate(
        tqdm(
          train_loader,
          desc=f"{run_label} ep {epoch}/{epochs}",
          leave=bool(args.progress_leave),
          dynamic_ncols=True,
          ascii=True,
          mininterval=0.5,
        ),
        start=1,
      )
    else:
      batch_iter = enumerate(train_loader, start=1)

    for batch_idx, batch in batch_iter:
      x = batch["x"].to(device, non_blocking=use_cuda)
      y = batch["y"].to(device, non_blocking=use_cuda)
      valid = batch["valid"].to(device, non_blocking=use_cuda)
      optimizer.zero_grad()
      logits = model(x)
      if logits.shape[-2:] != y.shape[-2:]:
        logits = torch.nn.functional.interpolate(
          logits, size=y.shape[-2:], mode="bilinear", align_corners=False
        )
      loss, _, _ = criterion(logits, y, valid)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      train_loss_vals.append(loss.item())
    train_loss = float(sum(train_loss_vals) / max(len(train_loss_vals), 1))

    # Tile-level validation to match the final evaluation methodology.
    if (epoch % val_every) == 0 or epoch == epochs:
      val_metrics = evaluate_tiles(
        model,
        "val",
        cfg,
        val_ds,
        val_city_to_indices,
        val_city_info,
        device,
      )
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
      val_metrics = {"f1": float("nan"), "iou": float("nan"), "auroc": float("nan"), "pr_auc": float("nan")}
    if scheduler is not None:
      sch_name = cfg["training"]["scheduler"].get("name", "cosine").lower()
      if sch_name == "plateau":
        if not np.isnan(val_metrics.get("iou", float("nan"))):
          scheduler.step(val_metrics["iou"])
      else:
        scheduler.step()
    epoch_time = time.perf_counter() - epoch_t0
    train_metrics = {"loss": train_loss}
    log_cb.log(epoch, train_metrics, val_metrics)
    ckpt_cb.step({"val_iou": val_metrics.get("iou", float("nan"))}, model, optimizer, epoch, scheduler=scheduler)

    print(
      f"{run_label} epoch {epoch}/{epochs} "
      f"- train_loss: {train_loss:.4f} "
      f"- val_iou: {val_metrics['iou']:.3f} "
      f"- val_f1: {val_metrics['f1']:.3f} "
      f"- time: {epoch_time:.1f}s"
    )

  meta = {
    "config_path": str(args.config),
    "config": cfg,
    "oscd_root": str(args.oscd_root),
    "phase1_change_maps_root": str(args.phase1_change_maps_root),
    "resume_from": str(resume_path) if resume_path is not None else None,
    "start_epoch": int(start_epoch_num),
    "seed": seed,
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
