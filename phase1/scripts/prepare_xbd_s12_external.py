"""Prepare the frozen xBD-S12 external-validation inputs.

The script verifies the official Zenodo archive, extracts only Sentinel-2 plus
metadata/normalization, and extracts post-disaster JSON labels from the user's
licensed original xBD archives. It never modifies the source archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


EXPECTED_SIZE = 9_523_531_926
EXPECTED_MD5 = "891b86000caa6019c20a95fc2f3f4e68"
ORIGINAL_ARCHIVES = (
    "train_images_labels_targets.tar",
    "test_images_labels_targets.tar",
    "hold_images_labels_targets.tar",
    "tier3.tar",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and selectively prepare xBD-S12 for external validation."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/xbd_s12_download/xbd_s12.tar.gz"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/xbd_s12"))
    parser.add_argument("--original-xbd-root", type=Path, default=Path("data/xbd"))
    parser.add_argument(
        "--labels-output", type=Path, default=Path("data/xbd_s12_original_labels")
    )
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--skip-release-extraction", action="store_true")
    parser.add_argument("--skip-label-extraction", action="store_true")
    return parser.parse_args()


def md5sum(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_release_path(name: str) -> Path | None:
    parts = PurePosixPath(name).parts
    if not parts:
        return None
    if "xbd_s12" in parts:
        index = parts.index("xbd_s12")
        parts = parts[index + 1 :]
    if not parts:
        return None
    if parts[0] == "s2" or parts[-1] in {
        "normalization.json",
        "xbd_s12_metadata.geojson",
    }:
        return Path(*parts)
    return None


def _safe_destination(root: Path, relative: Path) -> Path:
    destination = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root != destination and resolved_root not in destination.parents:
        raise RuntimeError(f"Unsafe archive member path: {relative}")
    return destination


def extract_release(archive: Path, output_root: Path) -> dict[str, int]:
    counts = {"files": 0, "bytes": 0}
    output_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            if not member.isfile():
                continue
            relative = _relative_release_path(member.name)
            if relative is None:
                continue
            destination = _safe_destination(output_root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == member.size:
                counts["files"] += 1
                counts["bytes"] += member.size
                continue
            source = package.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member {member.name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            counts["files"] += 1
            counts["bytes"] += member.size
            if counts["files"] % 500 == 0:
                print(f"Extracted {counts['files']} release files...", flush=True)
    return counts


def extract_original_labels(original_root: Path, labels_output: Path) -> dict[str, int]:
    labels_output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for archive_name in ORIGINAL_ARCHIVES:
        archive = original_root / archive_name
        if not archive.exists():
            raise FileNotFoundError(f"Missing original xBD archive: {archive}")
        tier = archive_name.split("_", 1)[0] if archive_name != "tier3.tar" else "tier3"
        destination_folder = labels_output / tier
        destination_folder.mkdir(parents=True, exist_ok=True)
        count = 0
        with tarfile.open(archive, "r|gz") as package:
            for member in package:
                if not member.isfile() or not member.name.endswith("_post_disaster.json"):
                    continue
                if "/labels/" not in member.name.replace("\\", "/"):
                    continue
                destination = _safe_destination(destination_folder, Path(member.name).name)
                source = package.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read {member.name} from {archive}")
                if not destination.exists() or destination.stat().st_size != member.size:
                    with source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
                else:
                    source.close()
                count += 1
        counts[tier] = count
        print(f"Extracted/indexed {count} post-disaster labels from {archive_name}.")
    return counts


def main() -> None:
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    size = args.archive.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"Archive size mismatch: {size} != {EXPECTED_SIZE}")
    checksum = "skipped"
    if not args.skip_checksum:
        checksum = md5sum(args.archive)
        if checksum != EXPECTED_MD5:
            raise RuntimeError(f"Archive MD5 mismatch: {checksum} != {EXPECTED_MD5}")
    release = (
        {"files": 0, "bytes": 0, "status": "skipped"}
        if args.skip_release_extraction
        else extract_release(args.archive, args.output_root)
    )
    labels = (
        {"status": "skipped"}
        if args.skip_label_extraction
        else extract_original_labels(args.original_xbd_root, args.labels_output)
    )
    manifest = {
        "archive": str(args.archive),
        "archive_size": size,
        "archive_md5": checksum,
        "expected_md5": EXPECTED_MD5,
        "output_root": str(args.output_root),
        "release_extraction": release,
        "labels_output": str(args.labels_output),
        "label_counts": labels,
    }
    manifest_path = args.output_root / "project_preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote preparation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
