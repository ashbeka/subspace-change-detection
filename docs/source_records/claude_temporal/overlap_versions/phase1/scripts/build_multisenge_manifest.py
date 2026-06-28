"""
Build a small MultiSenGE time-series manifest for temporal/geodesic experiments.

This avoids scanning hundreds of thousands of TIFFs by reading labels JSON files
which list the corresponding S1/S2 timestamps per patch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phase1.data.multisenge_manifest import load_multisenge_records, select_records, write_manifest


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--multisenge_root", type=Path, required=True, help="Path to data/MultiSenGE (contains labels/, s2/, ...).")
  ap.add_argument("--output_path", type=Path, required=True, help="Manifest JSON output path.")
  ap.add_argument("--min_s2_dates", type=int, default=5)
  ap.add_argument("--max_patches", type=int, default=50)
  ap.add_argument("--seed", type=int, default=1234)
  ap.add_argument("--include_s1", action="store_true", help="Also store S1 timestamps in the manifest (slower).")
  ap.add_argument("--no_require_ground_reference", action="store_true", help="Allow patches without ground_reference/*.tif.")
  return ap.parse_args()


def main():
  args = parse_args()
  records = load_multisenge_records(args.multisenge_root, include_s1=bool(args.include_s1))
  subset = select_records(
    records,
    min_s2_dates=args.min_s2_dates,
    max_patches=args.max_patches,
    seed=args.seed,
    require_ground_reference=not args.no_require_ground_reference,
    multisenge_root=args.multisenge_root,
  )
  write_manifest(
    args.output_path,
    multisenge_root=args.multisenge_root,
    records=subset,
    min_s2_dates=args.min_s2_dates,
    max_patches=args.max_patches,
    seed=args.seed,
  )
  print(f"Wrote {len(subset)} patches -> {args.output_path}")


if __name__ == "__main__":
  main()
