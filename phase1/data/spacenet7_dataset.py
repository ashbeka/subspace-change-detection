"""SpaceNet 7 monthly-image and persistent-building-label loader.

Source/provenance:
- Van Etten et al. (2021), *The Multi-Temporal Urban Development SpaceNet
  Dataset*, defines the MUDS/SpaceNet 7 data: monthly 4 m RGB mosaics,
  persistent building identifiers, and unusable-data masks.
- The official public objects are hosted at
  ``s3://spacenet-dataset/spacenet/SN7_buildings/``.

This module does not implement the SpaceNet SCOT challenge metric.  It exposes
the dated imagery, validity masks, and first appearance of matched building
IDs needed by this project's exploratory temporal change-localization study.
Treating an ID's first labeled appearance as a construction event is a project
evaluation choice and is recorded explicitly in experiment metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom


Array = np.ndarray
DATE_RE = re.compile(r"global_monthly_(?P<date>\d{4}_\d{2})_mosaic_")


@dataclass(frozen=True)
class SpaceNet7Observation:
  date: str
  image_path: Path
  matched_label_path: Path
  udm_label_path: Path | None = None


@dataclass(frozen=True)
class SpaceNet7Image:
  rgb: Array
  valid_mask: Array
  transform: object
  crs: object


def _date_from_name(path: Path) -> str | None:
  match = DATE_RE.search(path.name)
  return match.group("date") if match else None


def scan_spacenet7_aoi(root: Path) -> list[SpaceNet7Observation]:
  """Return dated observations with matched labels in chronological order."""
  root = Path(root)
  image_by_date = {
    date: path
    for path in sorted((root / "images").glob("*.tif"))
    if (date := _date_from_name(path)) is not None
  }
  label_by_date = {
    date: path
    for path in sorted((root / "labels_match_pix").glob("*_Buildings.geojson"))
    if (date := _date_from_name(path)) is not None
  }
  udm_by_date = {
    date: path
    for path in sorted((root / "labels").glob("*_UDM.geojson"))
    if (date := _date_from_name(path)) is not None
  }
  dates = sorted(set(image_by_date) & set(label_by_date))
  if not dates:
    raise FileNotFoundError(
      f"No matched SpaceNet 7 image/label dates under {root}. Expected images/ and labels_match_pix/."
    )
  missing_images = sorted(set(label_by_date) - set(image_by_date))
  missing_labels = sorted(set(image_by_date) - set(label_by_date))
  if missing_images or missing_labels:
    raise ValueError(
      f"SpaceNet 7 date mismatch: missing_images={missing_images}, missing_labels={missing_labels}"
    )
  return [
    SpaceNet7Observation(
      date=date,
      image_path=image_by_date[date],
      matched_label_path=label_by_date[date],
      udm_label_path=udm_by_date.get(date),
    )
    for date in dates
  ]


def read_geojson_features(path: Path) -> list[dict]:
  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  features = payload.get("features", [])
  if not isinstance(features, list):
    raise ValueError(f"Invalid GeoJSON feature collection: {path}")
  return features


def _read_geojson_with_crs(path: Path) -> tuple[list[dict], str | None]:
  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  features = payload.get("features", [])
  if not isinstance(features, list):
    raise ValueError(f"Invalid GeoJSON feature collection: {path}")
  crs = payload.get("crs", {})
  name = crs.get("properties", {}).get("name") if isinstance(crs, dict) else None
  return features, str(name) if name else None


def _reproject_features(features: Sequence[dict], source_crs: str, target_crs: object) -> list[dict]:
  """Reproject GeoJSON geometries while preserving feature properties."""
  output = []
  for feature in features:
    geometry = feature.get("geometry")
    if not geometry:
      output.append(feature)
      continue
    transformed = dict(feature)
    transformed["geometry"] = transform_geom(source_crs, target_crs, geometry)
    output.append(transformed)
  return output


def load_spacenet7_image(observation: SpaceNet7Observation) -> SpaceNet7Image:
  """Load RGB in ``[0,1]`` and combine alpha and optional UDM validity."""
  with rasterio.open(observation.image_path) as dataset:
    if dataset.count < 3:
      raise ValueError(f"Expected at least RGB bands, got {dataset.count}: {observation.image_path}")
    rgb = dataset.read([1, 2, 3]).astype(np.float32) / 255.0
    alpha = dataset.read(4) if dataset.count >= 4 else np.full(dataset.shape, 255, dtype=np.uint8)
    valid = alpha > 0
    transform = dataset.transform
    crs = dataset.crs

  if observation.udm_label_path is not None and observation.udm_label_path.exists():
    udm_features, udm_crs = _read_geojson_with_crs(observation.udm_label_path)
    if udm_features:
      if udm_crs and crs:
        udm_features = _reproject_features(udm_features, udm_crs, crs)
      unusable = rasterize_feature_mask(
        udm_features,
        out_shape=valid.shape,
        transform=transform,
        all_touched=True,
      )
      valid &= ~unusable
  valid &= np.all(np.isfinite(rgb), axis=0)
  return SpaceNet7Image(rgb=rgb, valid_mask=valid, transform=transform, crs=crs)


def building_ids(features: Iterable[dict]) -> set[int]:
  output: set[int] = set()
  for feature in features:
    properties = feature.get("properties", {})
    if "Id" not in properties:
      raise ValueError("Matched SpaceNet 7 building feature is missing persistent Id.")
    output.add(int(properties["Id"]))
  return output


def first_appearance_features(
  feature_sequences: Sequence[Sequence[dict]],
) -> list[list[dict]]:
  """Select features whose persistent ID has not appeared at an earlier date."""
  seen: set[int] = set()
  output: list[list[dict]] = []
  for features in feature_sequences:
    current = []
    for feature in features:
      identifier = int(feature.get("properties", {}).get("Id"))
      if identifier not in seen:
        current.append(feature)
    seen.update(building_ids(features))
    output.append(current)
  return output


def rasterize_feature_mask(
  features: Sequence[dict],
  *,
  out_shape: tuple[int, int],
  transform: object,
  all_touched: bool = False,
) -> Array:
  """Rasterize feature geometries to a Boolean image mask."""
  geometries = [feature.get("geometry") for feature in features if feature.get("geometry")]
  if not geometries:
    return np.zeros(out_shape, dtype=bool)
  values = rasterize(
    [(geometry, 1) for geometry in geometries],
    out_shape=out_shape,
    transform=transform,
    fill=0,
    dtype=np.uint8,
    all_touched=bool(all_touched),
  )
  return values.astype(bool, copy=False)
