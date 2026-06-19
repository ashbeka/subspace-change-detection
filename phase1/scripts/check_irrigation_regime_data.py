"""Check IrrMapper and Harmonized Sentinel-2 feasibility for regime-change DS.

Sources:
- Earth Engine IrrMapper v1.2 catalog:
  https://developers.google.com/earth-engine/datasets/catalog/UMT_Climate_IrrMapper_RF_v1_2
- Harmonized Sentinel-2 L2A catalog:
  https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

The public STAC metadata check works without Earth Engine authentication.  An
actual AOI/frame query requires an Earth Engine-enabled Google Cloud project.
This script does not download imagery and does not call IrrMapper transitions
ground truth; it only establishes data availability and extraction readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


IRRMAPPER_STAC = (
  "https://storage.googleapis.com/earthengine-stac/catalog/UMT/"
  "UMT_Climate_IrrMapper_RF_v1_2.json"
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--ee_project", default="")
  parser.add_argument("--bbox", default="-112.60,45.20,-112.40,45.35")
  parser.add_argument("--start", default="2017-01-01")
  parser.add_argument("--end", default="2025-01-01")
  parser.add_argument("--max_cloud", type=float, default=60.0)
  return parser.parse_args()


def parse_bbox(value: str) -> tuple[float, float, float, float]:
  values = tuple(float(part.strip()) for part in value.split(","))
  if len(values) != 4:
    raise ValueError("bbox must contain west,south,east,north")
  west, south, east, north = values
  if not west < east or not south < north:
    raise ValueError("bbox must satisfy west < east and south < north")
  return west, south, east, north


def public_catalog_metadata() -> dict:
  with urlopen(IRRMAPPER_STAC, timeout=30) as response:  # noqa: S310 - fixed trusted URL
    payload = json.load(response)
  return {
    "id": payload["id"],
    "title": payload["title"],
    "temporal_extent": payload["extent"]["temporal"]["interval"][0],
    "spatial_bbox": payload["extent"]["spatial"]["bbox"][0],
    "license": payload.get("license"),
    "bands": payload.get("summaries", {}).get("eo:bands", []),
    "catalog_url": (
      "https://developers.google.com/earth-engine/datasets/catalog/"
      "UMT_Climate_IrrMapper_RF_v1_2"
    ),
  }


def query_earth_engine(
  project: str,
  bbox: tuple[float, float, float, float],
  start: str,
  end: str,
  max_cloud: float,
) -> dict:
  import ee

  ee.Initialize(project=project)
  region = ee.Geometry.Rectangle(list(bbox), geodesic=False)
  irrigation = (
    ee.ImageCollection("UMT/Climate/IrrMapper_RF/v1_2")
    .filterBounds(region)
    .filterDate(start, end)
    .sort("system:time_start")
  )
  sentinel = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate(start, end)
    .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(max_cloud)))
    .sort("system:time_start")
  )
  irrigation_times = irrigation.aggregate_array("system:time_start").getInfo()
  sentinel_times = sentinel.aggregate_array("system:time_start").getInfo()
  sentinel_cloud = sentinel.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
  return {
    "status": "ready",
    "project": project,
    "irrigation_images": int(irrigation.size().getInfo()),
    "irrigation_time_start_ms": irrigation_times,
    "sentinel2_images_after_metadata_cloud_filter": int(sentinel.size().getInfo()),
    "sentinel2_time_start_ms": sentinel_times,
    "sentinel2_cloud_percent": sentinel_cloud,
    "next_step": (
      "Inspect annual transition prevalence and monthly clear-pixel coverage; "
      "do not download until the patch/year sampling protocol is fixed."
    ),
  }


def main() -> None:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  bbox = parse_bbox(args.bbox)
  report = {
    "purpose": "IrrMapper plus Sentinel-2 seasonal-regime feasibility",
    "bbox": bbox,
    "start": args.start,
    "end": args.end,
    "max_cloud": args.max_cloud,
    "public_catalog": public_catalog_metadata(),
    "label_warning": (
      "IrrMapper images are annual random-forest classifications. A transition "
      "between classes is a weak derived label, not an independently annotated event."
    ),
  }
  if args.ee_project:
    try:
      report["earth_engine_query"] = query_earth_engine(
        args.ee_project,
        bbox,
        args.start,
        args.end,
        args.max_cloud,
      )
    except Exception as exc:  # preserve a reproducible readiness report
      report["earth_engine_query"] = {
        "status": "blocked",
        "error_type": type(exc).__name__,
        "error": str(exc),
      }
  else:
    report["earth_engine_query"] = {
      "status": "blocked_missing_project",
      "next_step": (
        "Register/choose an Earth Engine-enabled Google Cloud project, then rerun "
        "this command with --ee-project PROJECT_ID."
      ),
    }
  path = args.output_dir / "irrigation_regime_data_feasibility.json"
  path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(json.dumps(report, indent=2))
  print(f"Saved: {path}")


if __name__ == "__main__":
  main()
