"""
Simple training callbacks: checkpointing and metric logging.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ModelCheckpoint:
  def __init__(
    self,
    out_dir: Path,
    monitor: str = "val_iou",
    mode: str = "max",
    save_last: bool = True,
  ):
    self.out_dir = Path(out_dir)
    self.monitor = monitor
    self.mode = mode
    self.best = None
    self.best_epoch = None
    self.save_last = bool(save_last)
    self.out_dir.mkdir(parents=True, exist_ok=True)

  def _save(
    self,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
  ) -> None:
    state: Dict[str, Any] = {
      "epoch": int(epoch),
      "model_state": model.state_dict(),
      "optimizer_state": optimizer.state_dict(),
      "monitor": self.monitor,
      "best_value": self.best,
      "best_epoch": self.best_epoch,
      "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if scheduler is not None:
      state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, Path(path))

  def step(
    self,
    metrics: Dict[str, float],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
  ) -> None:
    value = metrics.get(self.monitor) if metrics else None
    if value is not None and (value != value):  # NaN
      value = None
    if value is not None:
      if self.best is None:
        improved = True
      elif self.mode == "max":
        improved = value > self.best
      elif self.mode == "min":
        improved = value < self.best
      else:
        raise ValueError(f"Unknown mode: {self.mode}")
      if improved:
        self.best = float(value)
        self.best_epoch = int(epoch)
        self._save(self.out_dir / "best.ckpt", model, optimizer, epoch, scheduler=scheduler)
    if self.save_last:
      self._save(self.out_dir / "last.ckpt", model, optimizer, epoch, scheduler=scheduler)


class MetricsLogger:
  def __init__(self, out_dir: Path):
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.records = []

  def load_existing(self) -> None:
    path = self.out_dir / "train_log.json"
    if not path.exists():
      return
    try:
      with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
      if isinstance(data, list):
        self.records = data
    except Exception:
      return

  def log(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
    rec = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
    self.records.append(rec)
    with (self.out_dir / "train_log.json").open("w", encoding="utf-8") as f:
      json.dump(self.records, f, indent=2)

