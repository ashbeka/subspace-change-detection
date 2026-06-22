"""Load xBD-S12 Sentinel-2 pairs and derive damage masks from xBD labels.

Source/provenance
-----------------
- Dataset schema, event split, band order, and normalization follow the
  official ``prs-eth/xbd-s12`` repository and its ``DATASET.md``,
  ``src/constants.py``, and ``src/training/dataset.py`` files.
- Original xBD WKT damage polygons are rasterized with the class mapping used
  by the official xBD-S12 ``create_masks.py`` script.
- Categorical masks are reduced from 1024 to 128 pixels by block-majority
  voting, equivalent to the official one-hot average-pooling plus ``argmax``.

Project adaptation
------------------
This loader exposes two explicit binary damage views for unsupervised change
scores.  It does not relabel xBD-S12 as ordinary land-cover change and does not
use labels for image normalization or subspace construction.
"""
from __future__ import annotations

import json
import math
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from affine import Affine
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely import wkt


Array = np.ndarray
S2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
TRAIN_DISASTERS = (
    "lower-puna-volcano",
    "palu-tsunami",
    "mexico-earthquake",
    "socal-fire",
    "woolsey-fire",
    "portugal-wildfire",
    "pinery-bushfire",
    "midwest-flooding",
    "moore-tornado",
    "joplin-tornado",
    "hurricane-florence",
    "hurricane-harvey",
    "hurricane-michael",
)
TEST_DISASTERS = (
    "nepal-flooding",
    "tuscaloosa-tornado",
    "guatemala-volcano",
    "sunda-tsunami",
    "santa-rosa-wildfire",
    "hurricane-matthew",
)
DAMAGE_CLASS = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 5,
}


@dataclass(frozen=True)
class XBDS12Record:
    uid: str
    disaster: str
    event_split: str
    xbd_tier: str
    no_data_pixels: int
    properties: dict[str, object]


@dataclass(frozen=True)
class XBDS12Patch:
    record: XBDS12Record
    pre: Array
    post: Array
    mask: Array
    input_valid: Array


@dataclass(frozen=True)
class XBDObject:
    object_id: str
    damage_class: int
    mask: Array


def _properties_from_geojson(path: Path) -> list[dict[str, object]]:
    package = json.loads(path.read_text(encoding="utf-8"))
    features = package.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"Invalid GeoJSON features in {path}.")
    return [dict(feature.get("properties") or {}) for feature in features]


def _int_or_zero(value: object) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    return int(numeric) if math.isfinite(numeric) else 0


def load_metadata(root: Path) -> list[XBDS12Record]:
    """Load metadata without requiring geopandas."""
    path = root / "xbd_s12_metadata.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Missing xBD-S12 metadata: {path}")
    records: list[XBDS12Record] = []
    for properties in _properties_from_geojson(path):
        uid = str(properties.get("xbd_uid") or properties.get("uid") or "")
        disaster = str(properties.get("disaster") or "")
        if not uid or not disaster:
            raise ValueError("xBD-S12 metadata entry lacks xbd_uid or disaster.")
        split = str(properties.get("event_split") or "").lower()
        if split not in {"train", "test"}:
            if disaster in TRAIN_DISASTERS:
                split = "train"
            elif disaster in TEST_DISASTERS:
                split = "test"
            else:
                split = "unknown"
        records.append(
            XBDS12Record(
                uid=uid,
                disaster=disaster,
                event_split=split,
                xbd_tier=str(properties.get("xbd_tier") or ""),
                no_data_pixels=_int_or_zero(properties.get("N_px_no_data")),
                properties=properties,
            )
        )
    return records


def select_records(
    records: Iterable[XBDS12Record],
    *,
    split: str,
    exclude_metadata_nodata: bool = True,
    disasters: set[str] | None = None,
    maximum: int | None = None,
) -> list[XBDS12Record]:
    """Select records deterministically without consulting method scores."""
    selected = [record for record in records if split == "all" or record.event_split == split]
    if exclude_metadata_nodata:
        selected = [record for record in selected if record.no_data_pixels == 0]
    if disasters:
        selected = [record for record in selected if record.disaster in disasters]
    selected.sort(key=lambda record: (record.disaster, record.uid))
    return selected if maximum is None else selected[: max(0, int(maximum))]


def deterministic_event_sample(
    records: Iterable[XBDS12Record],
    *,
    per_event: int,
    seed: int,
) -> list[XBDS12Record]:
    """Select a reproducible UID-hash sample independently inside each event.

    The selection uses no image values, labels, or method scores. Hash ranking
    avoids the geographic/order bias of taking the first records in metadata.
    """
    grouped: dict[str, list[XBDS12Record]] = {}
    for record in records:
        grouped.setdefault(record.disaster, []).append(record)
    output: list[XBDS12Record] = []
    for event in sorted(grouped):
        ranked = sorted(
            grouped[event],
            key=lambda record: hashlib.sha256(
                f"{int(seed)}:{record.uid}".encode("utf-8")
            ).digest(),
        )
        output.extend(ranked[: max(0, int(per_event))])
    return sorted(output, key=lambda record: (record.disaster, record.uid))


def load_normalization(root: Path) -> tuple[Array, Array]:
    path = root / "normalization.json"
    package = json.loads(path.read_text(encoding="utf-8"))
    low = np.asarray(package["s2"]["1st"], dtype=np.float32)
    high = np.asarray(package["s2"]["99th"], dtype=np.float32)
    if low.shape != (len(S2_BANDS),) or high.shape != low.shape:
        raise ValueError(f"Unexpected Sentinel-2 normalization shapes: {low.shape}, {high.shape}")
    return low, high


def normalize_s2(cube: Array, low: Array, high: Array) -> Array:
    if cube.ndim != 3 or cube.shape[0] != len(S2_BANDS):
        raise ValueError(f"Expected 12 x H x W Sentinel-2 cube, got {cube.shape}.")
    scale = np.maximum(high - low, np.float32(1e-8))[:, None, None]
    return np.clip((cube.astype(np.float32) - low[:, None, None]) / scale, 0.0, 1.0)


def read_s2_pair(root: Path, uid: str, low: Array, high: Array) -> tuple[Array, Array, Array]:
    folder = root / "s2"
    pre_path = folder / f"{uid}_pre_disaster_s2.tif"
    post_path = folder / f"{uid}_post_disaster_s2.tif"
    with rasterio.open(pre_path) as source:
        pre_raw = source.read().astype(np.float32)
    with rasterio.open(post_path) as source:
        post_raw = source.read().astype(np.float32)
    if pre_raw.shape != post_raw.shape:
        raise ValueError(f"Unmatched xBD-S12 pair shapes for {uid}: {pre_raw.shape}, {post_raw.shape}")
    valid = np.all(np.isfinite(pre_raw), axis=0) & np.all(np.isfinite(post_raw), axis=0)
    pre = normalize_s2(pre_raw, low, high)
    post = normalize_s2(post_raw, low, high)
    valid &= np.all(np.isfinite(pre), axis=0) & np.all(np.isfinite(post), axis=0)
    return pre, post, valid


def rasterize_xbd_label(label_path: Path, output_size: int = 128) -> Array:
    """Rasterize one original xBD post-disaster JSON and majority-downsample."""
    package = json.loads(label_path.read_text(encoding="utf-8"))
    shapes: list[tuple[object, int]] = []
    for feature in package.get("features", {}).get("xy", []):
        subtype = str((feature.get("properties") or {}).get("subtype") or "")
        if subtype not in DAMAGE_CLASS:
            continue
        geometry = wkt.loads(str(feature["wkt"]))
        shapes.append((geometry, DAMAGE_CLASS[subtype]))
    with warnings.catch_warnings():
        # xBD WKT coordinates are image pixels, not map coordinates.
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        full = (
            rasterize(
                shapes,
                out_shape=(1024, 1024),
                fill=0,
                transform=Affine.identity(),
                dtype=np.uint8,
            )
            if shapes
            else np.zeros((1024, 1024), dtype=np.uint8)
        )
    if 1024 % output_size:
        raise ValueError(f"output_size={output_size} must divide 1024.")
    factor = 1024 // output_size
    blocks = full.reshape(output_size, factor, output_size, factor).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(output_size, output_size, factor * factor)
    counts = np.stack([(blocks == value).sum(axis=-1) for value in range(7)], axis=-1)
    return np.argmax(counts, axis=-1).astype(np.uint8)


def rasterize_xbd_instances(label_path: Path, categorical_mask: Array) -> list[XBDObject]:
    """Rasterize visible xBD building instances at Sentinel output scale.

    Each polygon is sampled at output-pixel centers, then intersected with the
    official project categorical mask for its damage class. Objects that do
    not survive the 1024-to-output reduction are excluded. This prevents tiny
    VHR-only polygons from being counted as retrievable Sentinel-2 objects.
    """
    if categorical_mask.ndim != 2 or categorical_mask.shape[0] != categorical_mask.shape[1]:
        raise ValueError("categorical_mask must be a square 2-D array.")
    output_size = int(categorical_mask.shape[0])
    if 1024 % output_size:
        raise ValueError(f"Output size {output_size} must divide 1024.")
    factor = 1024 // output_size
    package = json.loads(label_path.read_text(encoding="utf-8"))
    output: list[XBDObject] = []
    for index, feature in enumerate(package.get("features", {}).get("xy", [])):
        properties = feature.get("properties") or {}
        subtype = str(properties.get("subtype") or "")
        if subtype not in DAMAGE_CLASS:
            continue
        damage_class = DAMAGE_CLASS[subtype]
        geometry = wkt.loads(str(feature["wkt"]))
        with warnings.catch_warnings():
            # xBD WKT coordinates are image pixels, not map coordinates.
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            sampled = rasterize(
                [(geometry, 1)],
                out_shape=categorical_mask.shape,
                fill=0,
                transform=Affine.scale(factor, factor),
                all_touched=False,
                dtype=np.uint8,
            ).astype(bool)
        visible = sampled & (categorical_mask == damage_class)
        if not np.any(visible):
            continue
        object_id = str(
            properties.get("uid")
            or properties.get("feature_id")
            or properties.get("id")
            or index
        )
        output.append(
            XBDObject(
                object_id=object_id,
                damage_class=damage_class,
                mask=visible,
            )
        )
    return output


def build_label_index(labels_root: Path) -> dict[str, Path]:
    """Index extracted post-disaster xBD JSON labels by UID."""
    index: dict[str, Path] = {}
    for path in labels_root.rglob("*_post_disaster.json"):
        uid = path.name.removesuffix("_post_disaster.json")
        if uid in index:
            raise RuntimeError(f"Duplicate post-disaster label for {uid}: {index[uid]} and {path}")
        index[uid] = path
    return index


def load_patch(
    root: Path,
    record: XBDS12Record,
    *,
    labels: dict[str, Path],
    low: Array,
    high: Array,
    mask_cache_root: Path | None = None,
) -> XBDS12Patch:
    if record.uid not in labels:
        raise FileNotFoundError(f"No original xBD post-disaster label for {record.uid}.")
    pre, post, valid = read_s2_pair(root, record.uid, low, high)
    cache_path = (
        mask_cache_root / f"{record.uid}_mask.npy" if mask_cache_root is not None else None
    )
    if cache_path is not None and cache_path.exists():
        mask = np.load(cache_path, allow_pickle=False).astype(np.uint8, copy=False)
    else:
        mask = rasterize_xbd_label(labels[record.uid], output_size=pre.shape[1])
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, mask, allow_pickle=False)
    return XBDS12Patch(record=record, pre=pre, post=post, mask=mask, input_valid=valid)


def damage_evaluation_mask(
    mask: Array,
    input_valid: Array,
    view: str,
    *,
    boundary_buffer: int = 0,
) -> tuple[Array, Array]:
    """Return binary target and evaluation support for one declared label view.

    ``boundary_buffer`` removes pixels within that Euclidean distance, in
    4-metre output pixels, of any labeled building boundary. It is a
    sensitivity check for xBD-to-Sentinel alignment uncertainty; unbuffered
    evaluation remains primary.
    """
    if int(boundary_buffer) < 0:
        raise ValueError("boundary_buffer must be non-negative.")
    damage = (mask >= 2) & (mask <= 4)
    if view == "full_scene_damage":
        support = input_valid & (mask <= 4)
    elif view == "building_conditional_damage":
        support = input_valid & (mask >= 1) & (mask <= 4)
    elif view == "building_localization_diagnostic":
        support = input_valid & (mask <= 4)
        damage = (mask >= 1) & (mask <= 4)
    else:
        raise ValueError(f"Unknown xBD-S12 label view: {view!r}")
    if boundary_buffer:
        # Unclassified objects are still building footprints, so they define
        # boundaries even though class 5 is excluded from evaluation.
        building = (mask >= 1) & (mask <= 5)
        if np.any(building):
            inside_distance = distance_transform_edt(building)
            outside_distance = distance_transform_edt(~building)
            boundary_zone = np.where(building, inside_distance, outside_distance) <= float(
                boundary_buffer
            )
            support &= ~boundary_zone
    return damage.astype(np.uint8), support
