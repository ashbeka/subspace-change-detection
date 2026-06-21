"""Download verified official BreizhCrops 2017 L2A partitions.

URLs and expected uncompressed HDF5 sizes come from the official
``dl4sits/BreizhCrops`` repository's ``breizhcrops/datasets/urls.py``. Archives
are opened explicitly and only the expected HDF5 member is copied, avoiding
unsafe whole-archive extraction.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen


BASE = "https://breizhcrops.s3.eu-central-1.amazonaws.com"
EXPECTED_H5_SIZE = {
  "frh01": 987259904,
  "frh02": 803457960,
  "frh03": 890027448,
  "frh04": 639215848,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=Path, default=Path("data/BreizhCrops"))
  parser.add_argument("--regions", default="frh01,frh02,frh03,frh04")
  parser.add_argument("--keep_archives", action="store_true")
  return parser.parse_args()


def _download(url: str, destination: Path) -> None:
  if destination.exists() and destination.stat().st_size > 0:
    print(f"[EXISTS] {destination}")
    return
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_suffix(destination.suffix + ".part")
  print(f"[DOWNLOAD] {url} -> {destination}")
  with urlopen(url) as source, temporary.open("wb") as target:
    shutil.copyfileobj(source, target, length=1024 * 1024)
  temporary.replace(destination)


def _extract_h5(archive: Path, destination: Path, region: str) -> None:
  expected = EXPECTED_H5_SIZE[region]
  if destination.exists() and destination.stat().st_size == expected:
    print(f"[VERIFIED] {destination}")
    return
  temporary = destination.with_suffix(".h5.part")
  with tarfile.open(archive, "r:gz") as bundle:
    matches = [
      member for member in bundle.getmembers()
      if member.isfile() and Path(member.name).name == f"{region}.h5"
    ]
    if len(matches) != 1:
      raise RuntimeError(f"Expected one {region}.h5 member, found {len(matches)}.")
    source = bundle.extractfile(matches[0])
    if source is None:
      raise RuntimeError(f"Could not read {matches[0].name}.")
    with source, temporary.open("wb") as target:
      shutil.copyfileobj(source, target, length=1024 * 1024)
  if temporary.stat().st_size != expected:
    raise RuntimeError(
      f"Unexpected {region}.h5 size {temporary.stat().st_size}; expected {expected}."
    )
  temporary.replace(destination)
  print(f"[VERIFIED] {destination}")


def main() -> None:
  args = parse_args()
  regions = [item.strip().lower() for item in args.regions.split(",") if item.strip()]
  unknown = sorted(set(regions) - set(EXPECTED_H5_SIZE))
  if unknown:
    raise ValueError(f"Unknown regions: {unknown}")
  folder = args.data_root / "2017" / "L2A"
  folder.mkdir(parents=True, exist_ok=True)
  for name in ("classmapping.csv", "codes.csv"):
    _download(f"{BASE}/{name}", args.data_root / name)
  for region in regions:
    _download(f"{BASE}/2017/L2A/{region}.csv", folder / f"{region}.csv")
    archive = folder / f"{region}.h5.tar.gz"
    _download(f"{BASE}/2017/L2A/{region}.h5.tar.gz", archive)
    _extract_h5(archive, folder / f"{region}.h5", region)
    if not args.keep_archives:
      archive.unlink(missing_ok=True)


if __name__ == "__main__":
  main()
