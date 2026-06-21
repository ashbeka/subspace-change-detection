"""Read official BreizhCrops Sentinel-2 field time series for RTW transfer tests.

Source/provenance
-----------------
Russwurm et al., *BreizhCrops: A Time Series Dataset for Crop Type Mapping*,
ISPRS Archives (2020), https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020.
The official dataset/code repository is https://github.com/dl4sits/BreizhCrops.

The project reads the published HDF5 files directly instead of importing the
legacy training package. In the L2A files each row stores acquisition time,
ten parcel-mean Sentinel-2 bands (B2--B8A, B11, B12), and cloud/edge/saturation
flags. Rows failing the reported quality flags are removed. This loader does
not alter crop labels or create change labels.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


BAND_NAMES = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")


@dataclass(frozen=True)
class BreizhCropsSequence:
  """One labeled parcel-mean Sentinel-2 temporal sequence."""

  region: str
  field_id: str
  culture_code: str
  class_id: int
  class_name: str
  times: np.ndarray
  values: np.ndarray


def _class_mapping(path: Path) -> dict[str, tuple[int, str]]:
  with path.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
  required = {"id", "classname", "code"}
  columns = set(rows[0]) if rows else set()
  if not rows or not required.issubset(columns):
    raise ValueError(f"Class mapping lacks {sorted(required - columns)}.")
  return {
    str(row["code"]): (int(row["id"]), str(row["classname"]))
    for row in rows
  }


def _day_of_year(nanoseconds: np.ndarray) -> np.ndarray:
  timestamps = np.asarray(nanoseconds, dtype=np.int64).astype("datetime64[ns]")
  days = timestamps.astype("datetime64[D]")
  year_start = timestamps.astype("datetime64[Y]").astype("datetime64[D]")
  return (days - year_start).astype(np.int64).astype(np.float64) + 1.0


def _deduplicate_dates(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  unique, inverse = np.unique(times, return_inverse=True)
  if unique.size == times.size:
    return times, values
  accumulated = np.zeros((unique.size, values.shape[1]), dtype=np.float64)
  counts = np.zeros(unique.size, dtype=np.int64)
  for row, group in enumerate(inverse):
    accumulated[group] += values[row]
    counts[group] += 1
  return unique, accumulated / counts[:, None]


def _load_h5_sequence(
  database: h5py.File,
  key: str,
  *,
  quality_threshold: float,
  min_steps: int,
) -> tuple[np.ndarray, np.ndarray] | None:
  raw = np.asarray(database[str(key).lstrip("/")], dtype=np.float64)
  if raw.ndim != 2 or raw.shape[1] < 14:
    return None
  bands = raw[:, 1:11] * 1e-4
  flags = raw[:, 11:14]
  valid = (
    np.all(np.isfinite(raw[:, :14]), axis=1)
    & np.all(flags <= float(quality_threshold), axis=1)
    & np.all(bands >= 0.0, axis=1)
  )
  if int(np.sum(valid)) < int(min_steps):
    return None
  times = _day_of_year(raw[valid, 0])
  values = np.clip(bands[valid], 0.0, 1.5)
  order = np.argsort(times)
  times, values = _deduplicate_dates(times[order], values[order])
  if times.size < int(min_steps):
    return None
  return times, values


def sample_breizhcrops_region(
  root: Path,
  region: str,
  *,
  max_per_class: int,
  seed: int,
  min_steps: int = 12,
  quality_threshold: float = 0.5,
) -> tuple[list[BreizhCropsSequence], dict]:
  """Load a deterministic class-balanced sample from one geographic region."""
  root = Path(root)
  region = str(region).lower()
  database_path = root / "2017" / "L2A" / f"{region}.h5"
  index_path = root / "2017" / "L2A" / f"{region}.csv"
  mapping_path = root / "classmapping.csv"
  for required in (database_path, index_path, mapping_path):
    if not required.exists():
      raise FileNotFoundError(required)
  mapping = _class_mapping(mapping_path)
  with index_path.open(newline="", encoding="utf-8-sig") as handle:
    raw_index = list(csv.DictReader(handle))
  index = [row for row in raw_index if str(row["CODE_CULTU"]) in mapping]
  by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
  for row in index:
    by_class[mapping[str(row["CODE_CULTU"])][0]].append(row)
  rng = np.random.default_rng(int(seed))
  records: list[BreizhCropsSequence] = []
  rejected = 0
  with h5py.File(database_path, "r") as database:
    for class_id, candidates in sorted(by_class.items()):
      order = rng.permutation(len(candidates))
      accepted = 0
      for position in order:
        row = candidates[int(position)]
        loaded = _load_h5_sequence(
          database,
          str(row["path"]),
          quality_threshold=quality_threshold,
          min_steps=min_steps,
        )
        if loaded is None:
          rejected += 1
          continue
        times, values = loaded
        code = str(row["CODE_CULTU"])
        mapped_id, class_name = mapping[code]
        records.append(BreizhCropsSequence(
          region=region,
          field_id=str(row["id"]),
          culture_code=code,
          class_id=mapped_id,
          class_name=class_name,
          times=times,
          values=values,
        ))
        accepted += 1
        if accepted >= int(max_per_class):
          break
  class_counts = {
    int(class_id): sum(record.class_id == class_id for record in records)
    for class_id in sorted({record.class_id for record in records})
  }
  metadata = {
    "region": region,
    "database": str(database_path),
    "index_rows": len(index),
    "loaded_sequences": len(records),
    "rejected_for_quality_or_length": int(rejected),
    "class_counts": class_counts,
    "min_steps": int(min_steps),
    "quality_threshold": float(quality_threshold),
    "band_names": list(BAND_NAMES),
  }
  return records, metadata
