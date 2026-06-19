"""Temporal subspaces built from aligned multispectral band images.

Source/provenance:
- The first-order magnitude and second-order/geodesic quantities follow Fukui
  et al., *Second-order Difference Subspace* (2024), Definitions 2.6, 3.1,
  3.2, and 4.1.
- The sample construction is a satellite adaptation of the paper's shape
  subspace: spatial locations are the ambient coordinates and each aligned
  spectral band image is one high-dimensional column vector.

For one date, a cube with ``B`` bands and a fixed valid mask of ``N`` pixels is
converted to ``X_t in R^(N x B)``.  The leading left singular vectors span the
date subspace ``S_t in Gr(r, N)``.  A common mask is mandatory across a
sequence so that every date uses the same ambient spatial coordinates.

This construction is experimental.  Formula tests establish dimensional and
geometric consistency; they do not establish remote-sensing usefulness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np

from phase1.ds.pca_utils import difference_subspace_canonical_components
from phase1.subspace.geodesic import (
  geodesic_projection,
  grassmann_geodesic_distance,
  grassmann_geodesic_interpolate,
  sum_subspace,
  subspace_magnitude,
)
from phase1.subspace.second_order_ds import second_order_difference_subspace

Array = np.ndarray


@dataclass(frozen=True)
class BandImageSubspace:
  basis: Array
  singular_values: Array
  explained_energy_ratio: Array
  n_spatial_locations: int
  n_bands: int
  rank: int
  preprocessing: str


@dataclass(frozen=True)
class TiledBandImageSubspace:
  """One date subspace fitted inside one fixed spatial grid cell."""

  grid_size: int
  tile_row: int
  tile_col: int
  row_start: int
  row_end: int
  col_start: int
  col_end: int
  valid_mask: Array
  fitted: BandImageSubspace

  @property
  def key(self) -> tuple[int, int, int]:
    return self.grid_size, self.tile_row, self.tile_col


@dataclass(frozen=True)
class TiledTemporalDecomposition:
  """Spatial maps and per-tile magnitudes for one temporal triple."""

  paper_total: Array
  orthogonal: Array
  along: Array
  time_aware: Array
  tile_rows: tuple[dict[str, float | int], ...]


def sequence_common_valid_mask(
  cubes: Sequence[Array],
  *,
  nodata_value: float = 0.0,
  require_all_bands: bool = True,
) -> Array:
  """Return spatial locations valid at every date in a sequence."""
  if not cubes:
    raise ValueError("At least one cube is required.")
  shape = cubes[0].shape
  if len(shape) != 3:
    raise ValueError(f"Expected cubes shaped (bands, height, width), got {shape}")
  common = np.ones(shape[1:], dtype=bool)
  for cube in cubes:
    if cube.shape != shape:
      raise ValueError(f"All cubes must have the same shape: {shape} vs {cube.shape}")
    finite = np.isfinite(cube)
    non_nodata = cube != float(nodata_value)
    valid_per_band = finite & non_nodata
    valid = np.all(valid_per_band, axis=0) if require_all_bands else np.any(valid_per_band, axis=0)
    common &= valid
  return common


def build_band_image_subspace(
  cube: Array,
  valid_mask: Array,
  *,
  rank: int,
  preprocessing: str = "centered_band_l2",
  eps: float = 1e-12,
) -> BandImageSubspace:
  """Construct ``S_t`` from flattened band-image vectors.

  ``preprocessing`` controls what information the subspace can use:

  - ``uncentered``: raw band-image columns; brightness and spatial pattern.
  - ``centered``: remove each band's spatial mean; spatial deviations only.
  - ``band_l2``: normalize each raw band-image column to unit length.
  - ``centered_band_l2``: center then normalize each band column; the default
    shape-focused construction used for the first temporal experiment.
  """
  if cube.ndim != 3:
    raise ValueError(f"Expected cube shaped (bands, height, width), got {cube.shape}")
  if valid_mask.shape != cube.shape[1:]:
    raise ValueError(f"Mask shape {valid_mask.shape} does not match cube {cube.shape}")
  if not np.any(valid_mask):
    raise ValueError("The common valid mask contains no spatial locations.")

  key = str(preprocessing).strip().lower().replace("-", "_")
  valid_modes = {"uncentered", "centered", "band_l2", "centered_band_l2"}
  if key not in valid_modes:
    raise ValueError(f"Unknown preprocessing={preprocessing!r}; expected {sorted(valid_modes)}")

  # Rows retain fixed spatial coordinates; columns are complete band images.
  matrix = cube[:, valid_mask].T.astype(np.float64, copy=False)
  if key in {"centered", "centered_band_l2"}:
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
  if key in {"band_l2", "centered_band_l2"}:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    matrix = matrix / np.maximum(norms, float(eps))

  max_rank = min(matrix.shape)
  effective_rank = max(1, min(int(rank), max_rank))
  left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
  energy = singular_values * singular_values
  total_energy = float(np.sum(energy))
  explained = energy[:effective_rank] / total_energy if total_energy > 0.0 else np.zeros(effective_rank)
  basis = left[:, :effective_rank]
  return BandImageSubspace(
    basis=basis.astype(np.float32, copy=False),
    singular_values=singular_values[:effective_rank].astype(np.float32, copy=False),
    explained_energy_ratio=explained.astype(np.float32, copy=False),
    n_spatial_locations=int(matrix.shape[0]),
    n_bands=int(matrix.shape[1]),
    rank=effective_rank,
    preprocessing=key,
  )


def spatial_grid_bounds(shape: tuple[int, int], grid_size: int) -> list[tuple[int, int, int, int, int, int]]:
  """Partition an image into ``grid_size x grid_size`` non-overlapping cells."""
  rows, cols = (int(shape[0]), int(shape[1]))
  grid = int(grid_size)
  if rows < 1 or cols < 1 or grid < 1:
    raise ValueError(f"Invalid shape/grid combination: shape={shape}, grid_size={grid_size}")
  row_edges = np.linspace(0, rows, grid + 1, dtype=int)
  col_edges = np.linspace(0, cols, grid + 1, dtype=int)
  return [
    (tile_row, tile_col, int(row_edges[tile_row]), int(row_edges[tile_row + 1]),
     int(col_edges[tile_col]), int(col_edges[tile_col + 1]))
    for tile_row in range(grid)
    for tile_col in range(grid)
  ]


def build_tiled_band_image_subspaces(
  cube: Array,
  common_valid_mask: Array,
  *,
  grid_size: int,
  rank: int,
  preprocessing: str = "centered_band_l2",
  min_valid_locations: int = 64,
) -> list[TiledBandImageSubspace]:
  """Fit one band-image subspace per fixed grid cell.

  Every date in a sequence must use the same ``common_valid_mask`` and grid.
  This makes corresponding tile bases points on the same local Grassmann
  manifold.  Spatial coordinates are preserved within each tile; relations
  across tile boundaries are not represented at that scale.

  The construction is related to the multi-scale patch strategy in Dagobert
  et al. (IPOL 2022), but the fitted object and score are different: this code
  fits band-image subspaces and compares their Grassmann/DS geometry, whereas
  the IPOL method fits temporal novelty bases and applies NFA testing.
  """
  if common_valid_mask.shape != cube.shape[1:]:
    raise ValueError("common_valid_mask must match the cube spatial shape.")
  tiles: list[TiledBandImageSubspace] = []
  for tile_row, tile_col, row_start, row_end, col_start, col_end in spatial_grid_bounds(
    common_valid_mask.shape, grid_size
  ):
    local_mask = common_valid_mask[row_start:row_end, col_start:col_end]
    if int(np.sum(local_mask)) < int(min_valid_locations):
      continue
    local_cube = cube[:, row_start:row_end, col_start:col_end]
    fitted = build_band_image_subspace(
      local_cube,
      local_mask,
      rank=rank,
      preprocessing=preprocessing,
    )
    tiles.append(TiledBandImageSubspace(
      grid_size=int(grid_size),
      tile_row=tile_row,
      tile_col=tile_col,
      row_start=row_start,
      row_end=row_end,
      col_start=col_start,
      col_end=col_end,
      valid_mask=local_mask.copy(),
      fitted=fitted,
    ))
  if not tiles:
    raise ValueError(
      f"No grid cells retained for grid_size={grid_size}; "
      f"lower min_valid_locations={min_valid_locations}."
    )
  return tiles


def tiled_temporal_decomposition(
  left: Sequence[TiledBandImageSubspace],
  middle: Sequence[TiledBandImageSubspace],
  right: Sequence[TiledBandImageSubspace],
  *,
  output_shape: tuple[int, int],
  time_fraction: float,
) -> TiledTemporalDecomposition:
  """Compute local paper/geodesic DS maps for corresponding spatial tiles."""
  left_by_key = {tile.key: tile for tile in left}
  middle_by_key = {tile.key: tile for tile in middle}
  right_by_key = {tile.key: tile for tile in right}
  shared_keys = sorted(set(left_by_key) & set(middle_by_key) & set(right_by_key))
  if not shared_keys:
    raise ValueError("The three dates have no corresponding retained tiles.")
  maps = {
    key: np.zeros(output_shape, dtype=np.float32)
    for key in ("paper_total", "orthogonal", "along", "time_aware")
  }
  rows: list[dict[str, float | int]] = []
  for key in shared_keys:
    first = left_by_key[key]
    current = middle_by_key[key]
    last = right_by_key[key]
    basis1 = first.fitted.basis
    basis2 = current.fitted.basis
    basis3 = last.fitted.basis
    if basis1.shape != basis2.shape or basis2.shape != basis3.shape:
      raise ValueError(f"Tile {key} has inconsistent basis shapes across dates.")

    result = second_order_difference_subspace(basis1, basis2, basis3, decompose=True)
    endpoint_sum = sum_subspace(basis1, basis3)
    projected_middle = geodesic_projection(endpoint_sum, basis2)
    expected_middle = grassmann_geodesic_interpolate(basis1, basis3, time_fraction)
    targets = {
      "paper_total": result.mean_subspace,
      "orthogonal": projected_middle,
      "along": result.mean_subspace,
      "time_aware": expected_middle,
    }
    sources = {
      "paper_total": basis2,
      "orthogonal": basis2,
      "along": projected_middle,
      "time_aware": basis2,
    }
    slices = (
      slice(current.row_start, current.row_end),
      slice(current.col_start, current.col_end),
    )
    magnitudes: dict[str, float] = {}
    for map_name in maps:
      local = spatial_difference_contribution(
        sources[map_name], targets[map_name], current.valid_mask
      )
      maps[map_name][slices] = local
      magnitudes[map_name] = float(np.sum(local))
    rows.append({
      "grid_size": int(current.grid_size),
      "tile_row": int(current.tile_row),
      "tile_col": int(current.tile_col),
      "valid_locations": int(np.sum(current.valid_mask)),
      "rank": int(current.fitted.rank),
      "paper_total": magnitudes["paper_total"],
      "orthogonal": magnitudes["orthogonal"],
      "along": magnitudes["along"],
      "time_aware": magnitudes["time_aware"],
    })
  return TiledTemporalDecomposition(
    paper_total=maps["paper_total"],
    orthogonal=maps["orthogonal"],
    along=maps["along"],
    time_aware=maps["time_aware"],
    tile_rows=tuple(rows),
  )


def parse_dates(dates: Iterable[str]) -> list[datetime]:
  return [datetime.strptime(str(value), "%Y%m%d") for value in dates]


def temporal_difference_measurements(bases: Sequence[Array], dates: Sequence[str]) -> dict[str, list[float]]:
  """Measure paper-faithful first/second DS quantities over a sequence.

  The raw second-order definition assumes equally spaced samples.  We retain
  its output for every triple but also expose interval balance so downstream
  analysis can restrict claims to approximately equal-gap triples.
  """
  if len(bases) != len(dates):
    raise ValueError(f"Expected one date per basis, got {len(bases)} and {len(dates)}")
  if len(bases) < 3:
    raise ValueError("At least three sequential subspaces are required.")
  parsed = parse_dates(dates)

  adjacent_geo: list[float] = []
  adjacent_mag: list[float] = []
  adjacent_days: list[float] = []
  adjacent_geo_per_day: list[float] = []
  adjacent_sqrt_mag_per_day: list[float] = []
  for left, right, date_left, date_right in zip(bases[:-1], bases[1:], parsed[:-1], parsed[1:]):
    days = float((date_right - date_left).days)
    if days <= 0:
      raise ValueError("Dates must be strictly increasing.")
    geo = grassmann_geodesic_distance(left, right)
    mag = subspace_magnitude(left, right)
    adjacent_geo.append(geo)
    adjacent_mag.append(mag)
    adjacent_days.append(days)
    adjacent_geo_per_day.append(geo / days)
    adjacent_sqrt_mag_per_day.append(float(np.sqrt(max(mag, 0.0))) / days)

  span_mag: list[float] = []
  second_mag: list[float] = []
  second_along: list[float] = []
  second_orth: list[float] = []
  decomposition_residual: list[float] = []
  first_speed_change: list[float] = []
  gap_left_days: list[float] = []
  gap_right_days: list[float] = []
  gap_ratio: list[float] = []
  time_fraction: list[float] = []
  time_aware_deviation_magnitude: list[float] = []
  time_aware_deviation_geodesic: list[float] = []
  time_aware_acceleration_proxy: list[float] = []
  for index in range(1, len(bases) - 1):
    left_days = float((parsed[index] - parsed[index - 1]).days)
    right_days = float((parsed[index + 1] - parsed[index]).days)
    ratio = max(left_days, right_days) / min(left_days, right_days)
    fraction = left_days / (left_days + right_days)
    expected_middle = grassmann_geodesic_interpolate(
      bases[index - 1], bases[index + 1], fraction
    )
    deviation_magnitude = subspace_magnitude(bases[index], expected_middle)
    deviation_geodesic = grassmann_geodesic_distance(bases[index], expected_middle)
    result = second_order_difference_subspace(
      bases[index - 1], bases[index], bases[index + 1], decompose=True
    )
    along = float(result.mag_along or 0.0)
    orth = float(result.mag_orth or 0.0)
    total = float(result.mag_total)
    span_mag.append(subspace_magnitude(bases[index - 1], bases[index + 1]))
    second_mag.append(total)
    second_along.append(along)
    second_orth.append(orth)
    decomposition_residual.append(total - along - orth)
    first_speed_change.append(abs(adjacent_sqrt_mag_per_day[index] - adjacent_sqrt_mag_per_day[index - 1]))
    gap_left_days.append(left_days)
    gap_right_days.append(right_days)
    gap_ratio.append(ratio)
    time_fraction.append(fraction)
    time_aware_deviation_magnitude.append(deviation_magnitude)
    time_aware_deviation_geodesic.append(deviation_geodesic)
    # For a Euclidean quadratic trajectory, interpolation error at the middle
    # sample is 0.5 * h_left * h_right * acceleration.  This is a local
    # Grassmann-distance analogue, not a paper-defined DS magnitude.
    time_aware_acceleration_proxy.append(
      2.0 * deviation_geodesic / (left_days * right_days)
    )

  return {
    "first_adjacent_geodesic": adjacent_geo,
    "first_adjacent_magnitude": adjacent_mag,
    "adjacent_gap_days": adjacent_days,
    "first_geodesic_per_day": adjacent_geo_per_day,
    "first_sqrt_magnitude_per_day": adjacent_sqrt_mag_per_day,
    "first_endpoint_span_magnitude": span_mag,
    "first_speed_change": first_speed_change,
    "second_magnitude": second_mag,
    "second_along_geodesic": second_along,
    "second_orthogonal_geodesic": second_orth,
    "second_decomposition_residual": decomposition_residual,
    "second_gap_left_days": gap_left_days,
    "second_gap_right_days": gap_right_days,
    "second_gap_ratio": gap_ratio,
    "second_time_fraction": time_fraction,
    "time_aware_geodesic_deviation_magnitude": time_aware_deviation_magnitude,
    "time_aware_geodesic_deviation": time_aware_deviation_geodesic,
    "time_aware_acceleration_proxy": time_aware_acceleration_proxy,
  }


def spatial_difference_contribution(
  basis1: Array,
  basis2: Array,
  valid_mask: Array,
) -> Array:
  """Map a first-order DS magnitude back to its ambient spatial coordinates.

  For canonical principal-vector pairs ``u_i, v_i``, the paper defines total
  magnitude as ``sum_i ||u_i-v_i||^2``.  The returned map stores each spatial
  coordinate's contribution ``sum_i (u_i[j]-v_i[j])^2``; consequently its sum
  matches ``subspace_magnitude(basis1, basis2)`` up to numerical precision.
  """
  if valid_mask.ndim != 2:
    raise ValueError(f"Expected a 2D spatial mask, got {valid_mask.shape}")
  if basis1.shape[0] != int(np.sum(valid_mask)) or basis2.shape[0] != int(np.sum(valid_mask)):
    raise ValueError("Basis ambient dimension must equal the number of valid spatial locations.")
  components = difference_subspace_canonical_components(basis1, basis2)
  contribution = np.zeros(basis1.shape[0], dtype=np.float64)
  if components.basis.shape[1]:
    contribution = np.sum(
      (components.basis.astype(np.float64) ** 2)
      * components.squared_pair_magnitudes.astype(np.float64)[None, :],
      axis=1,
    )
  output = np.zeros(valid_mask.shape, dtype=np.float32)
  output[valid_mask] = contribution.astype(np.float32, copy=False)
  return output
