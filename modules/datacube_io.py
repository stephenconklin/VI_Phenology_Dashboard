"""
datacube_io.py — File discovery, lazy dataset loading, pixel extraction,
and coordinate reprojection for the VI Phenology Dashboard.

Memory contract
---------------
- The full datacube array is NEVER loaded into memory.
- Basemap: computed via Dask lazy reduce + spatial coarsening.
- Pixel time series: direct HDF5 hyperslab read via netCDF4-python.
- Dataset handles: lru_cache so files are opened once per session.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATACUBE_ROOT,
    FLIGHT_BOX_SUBDIR,
    DEFAULT_VI_VAR,
    DATACUBE_CRS_EPSG,
    TARGET_CRS_EPSG,
    BASEMAP_MAX_DIM,
    VI_VALID_RANGE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegionPaths:
    """Paths for one LVIS flight box region."""
    region_id: str
    nc_path: Path
    zarr_path: Path | None      # None if not yet converted
    metrics_path: Path | None   # None if pixel_metrics.nc not present


class PixelTimeSeries(NamedTuple):
    """Raw time series data for a single pixel."""
    dates: np.ndarray       # datetime64[D], shape (n_time,)
    raw_vi: np.ndarray      # float32, shape (n_time,); NaN where masked
    valid_mask: np.ndarray  # bool, shape (n_time,)
    x_coord: float          # UTM easting (m)
    y_coord: float          # UTM northing (m)
    lon: float              # WGS84 longitude
    lat: float              # WGS84 latitude


# ---------------------------------------------------------------------------
# Region discovery
# ---------------------------------------------------------------------------

def discover_regions(root: Path = DATACUBE_ROOT) -> dict[str, RegionPaths]:
    """
    Scan DATACUBE_ROOT/LVIS_flightboxes_final/ for all G5_xx subdirectories.

    For each region, checks for:
        {VI}_{region}_datacube.nc          (required)
        {VI}_{region}_datacube.zarr/       (optional — fast path)
        {VI}_{region}_pixel_metrics.nc     (optional — precomputed metrics)

    Returns a dict keyed by region_id (e.g. "G5_1") sorted naturally.
    Raises FileNotFoundError if the root directory does not exist.
    """
    flight_dir = root / FLIGHT_BOX_SUBDIR
    if not flight_dir.exists():
        raise FileNotFoundError(
            f"Datacube directory not found: {flight_dir}\n"
            f"Set VI_DATACUBE_ROOT env var or edit config.py."
        )

    vi = DEFAULT_VI_VAR
    regions: dict[str, RegionPaths] = {}

    for subdir in sorted(flight_dir.iterdir(), key=lambda p: _natural_sort_key(p.name)):
        if not subdir.is_dir() or not subdir.name.startswith("G5_"):
            continue
        region_id = subdir.name
        nc_path = subdir / f"{vi}_{region_id}_datacube.nc"
        if not nc_path.exists():
            continue

        zarr_path  = subdir / f"{vi}_{region_id}_datacube.zarr"
        metrics_path = subdir / f"{vi}_{region_id}_pixel_metrics.nc"

        regions[region_id] = RegionPaths(
            region_id=region_id,
            nc_path=nc_path,
            zarr_path=zarr_path if zarr_path.exists() else None,
            metrics_path=metrics_path if metrics_path.exists() else None,
        )

    return regions


def _natural_sort_key(name: str):
    """Sort 'G5_10' after 'G5_9' (numeric-aware)."""
    import re
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


# ---------------------------------------------------------------------------
# Dataset handles — lru_cache so files are opened only once per session
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _open_datacube_cached(nc_path_str: str) -> xr.Dataset:
    """
    Open a NetCDF4 datacube with xarray + Dask using the file's native
    HDF5 chunk layout (chunks={}).

    Using chunks={} avoids the UserWarning about misaligned chunks that
    occurs when the requested chunk sizes don't match what's stored on disk.
    The time axis is rechunked to -1 inside compute_basemap_metric() where
    needed for efficient temporal reduction.

    String argument (not Path) required for lru_cache hashability.
    """
    return xr.open_dataset(
        nc_path_str,
        engine="netcdf4",
        chunks={},
        mask_and_scale=True,
    )


def get_dataset(paths: RegionPaths) -> xr.Dataset:
    """
    Return a lazily opened dataset for the region.
    Prefers ZARR (faster pixel access) when available.
    """
    if paths.zarr_path is not None:
        return xr.open_zarr(str(paths.zarr_path))
    return _open_datacube_cached(str(paths.nc_path))


# ---------------------------------------------------------------------------
# Coordinate reprojection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _get_transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(
        f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True
    )


def utm_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    src_epsg: int = DATACUBE_CRS_EPSG,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert UTM easting/northing arrays to WGS84 (lon, lat)."""
    tf = _get_transformer(src_epsg, TARGET_CRS_EPSG)
    lon, lat = tf.transform(x, y)
    return lon, lat


def latlon_to_utm(
    lon: float,
    lat: float,
    dst_epsg: int = DATACUBE_CRS_EPSG,
) -> tuple[float, float]:
    """Convert WGS84 lon/lat to UTM easting/northing."""
    tf = _get_transformer(TARGET_CRS_EPSG, dst_epsg)
    x, y = tf.transform(lon, lat)
    return float(x), float(y)


def detect_crs_epsg(ds: xr.Dataset) -> int:
    """
    Read the EPSG code from the CF grid_mapping 'spatial_ref' variable.
    Falls back to DATACUBE_CRS_EPSG if not present or unparseable.
    """
    try:
        from pyproj import CRS
        sr = ds["spatial_ref"]
        wkt = sr.attrs.get("crs_wkt") or sr.attrs.get("spatial_ref", "")
        if wkt:
            epsg = CRS.from_wkt(wkt).to_epsg()
            if epsg:
                return int(epsg)
    except Exception:
        pass
    return DATACUBE_CRS_EPSG


def build_display_coords(
    ds: xr.Dataset,
    src_epsg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lon/lat 2-D arrays from the dataset's x/y coordinate vectors.
    Used to label Plotly heatmap axes in geographic coordinates.

    Returns (lon_2d, lat_2d), each shape (ny, nx).
    Only reads coordinate arrays — never the data variable.
    """
    if src_epsg is None:
        src_epsg = detect_crs_epsg(ds)
    x_vals = ds["x"].values  # (nx,)
    y_vals = ds["y"].values  # (ny,)
    xx, yy = np.meshgrid(x_vals, y_vals)
    lon, lat = utm_to_latlon(xx.ravel(), yy.ravel(), src_epsg)
    return lon.reshape(yy.shape), lat.reshape(yy.shape)


# ---------------------------------------------------------------------------
# Date cache — shared structure for all pixels in a datacube
# ---------------------------------------------------------------------------

def build_date_cache(ds: xr.Dataset) -> dict:
    """
    Build the date_cache dict required by _extract_pixel_metrics().
    Reads only the time coordinate — never the data array.

    Returns a dict with keys:
        n_days      : int — calendar days from first to last observation
        year_arr    : np.int32 (n_days,) — year of each day on the daily grid
        doy_arr     : np.int16 (n_days,) — day-of-year on the daily grid
        years       : np.ndarray — unique years present
        year_masks  : dict[int, np.ndarray] — boolean masks per year
        day_offsets : np.int32 (n_time,) — index of each obs on the daily grid
    """
    # Decode time from "days since 1970-01-01" to pd.DatetimeIndex
    time_raw = ds["time"].values
    if np.issubdtype(time_raw.dtype, np.datetime64):
        times = pd.DatetimeIndex(pd.to_datetime(time_raw))
    else:
        # int32 days since 1970-01-01
        origin = pd.Timestamp("1970-01-01")
        times = pd.DatetimeIndex(
            [origin + pd.Timedelta(days=int(d)) for d in time_raw]
        )

    n_days = (times[-1] - times[0]).days + 1
    all_dates = pd.date_range(start=times[0], periods=n_days, freq="D")
    year_arr  = all_dates.year.values.astype(np.int32)
    doy_arr   = all_dates.dayofyear.values.astype(np.int16)
    years     = np.unique(year_arr)

    day_offsets = np.array(
        [(t - times[0]).days for t in times],
        dtype=np.int32,
    )

    return {
        "n_days":      n_days,
        "year_arr":    year_arr,
        "doy_arr":     doy_arr,
        "years":       years,
        "year_masks":  {int(yr): (year_arr == yr) for yr in years},
        "day_offsets": day_offsets,
    }


# ---------------------------------------------------------------------------
# WGS84 regridding helper
# ---------------------------------------------------------------------------

def _regrid_to_wgs84(
    z: np.ndarray,
    lon_2d: np.ndarray,
    lat_2d: np.ndarray,
    x_c: np.ndarray,
    y_c: np.ndarray,
    src_epsg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-project z values from a regular UTM grid onto a regular WGS84 grid.

    Problem
    -------
    go.Heatmap uses only lon[0,:] and lat[:,0] as axis vectors, implicitly
    assuming a purely rectangular WGS84 grid.  UTM Zone 34S grids are rotated
    slightly relative to geographic north, so each row has a different longitude
    offset and each column a different latitude offset.  Passing the raw UTM
    meshgrid lon/lat to Plotly causes visible misalignment against any WGS84
    basemap.

    Solution
    --------
    1. Set up a RegularGridInterpolator on the original UTM grid.
    2. Define a regular WGS84 target grid with the same pixel count.
    3. Convert each target (lon, lat) back to UTM and look up its z-value.

    The returned arrays are truly regular: lon_grid[i,j] == lon_grid[0,j]
    and lat_grid[i,j] == lat_grid[i,0] for all i, j.
    """
    from scipy.interpolate import RegularGridInterpolator

    ny, nx = z.shape

    # RegularGridInterpolator requires strictly increasing axis arrays.
    # y_c (UTM northing) is often decreasing in raster convention.
    y_order = np.argsort(y_c)
    x_order = np.argsort(x_c)
    z_sorted = z[y_order, :][:, x_order]

    interp = RegularGridInterpolator(
        (y_c[y_order], x_c[x_order]),
        z_sorted,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Regular WGS84 target grid — lat increases south→north (Plotly default).
    lon_reg = np.linspace(float(lon_2d.min()), float(lon_2d.max()), nx)
    lat_reg = np.linspace(float(lat_2d.min()), float(lat_2d.max()), ny)
    lon_grid, lat_grid = np.meshgrid(lon_reg, lat_reg)

    # Convert target WGS84 positions → UTM for the interpolator lookup.
    tf_inv = _get_transformer(TARGET_CRS_EPSG, src_epsg)
    x_tgt, y_tgt = tf_inv.transform(lon_grid.ravel(), lat_grid.ravel())

    z_reg = interp(
        np.column_stack([y_tgt, x_tgt])   # interpolator axis order: (northing, easting)
    ).reshape(ny, nx)

    return z_reg, lon_grid, lat_grid


# ---------------------------------------------------------------------------
# Basemap spatial metrics — lazy Dask compute, downsampled
# ---------------------------------------------------------------------------

def compute_basemap_metric(
    ds: xr.Dataset,
    metric: str,
    vi_var: str = DEFAULT_VI_VAR,
    max_dim: int = BASEMAP_MAX_DIM,
    src_epsg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a simple spatial summary metric using Dask and return downsampled
    arrays suitable for Plotly heatmap display.

    The full data array is NEVER loaded into memory — Dask reads only the
    chunks needed for the spatial coarsening + reduce operation.

    Supported metric keys:
        "peak_ndvi_mean"  → max over time axis
        "mean_ndvi"       → mean over time axis
        "std_ndvi"        → std  over time axis
        "data_coverage"   → fraction of non-NaN observations

    Returns (z_2d, lon_2d, lat_2d) numpy arrays at display resolution.
    """
    da = ds[vi_var]  # Dask-backed DataArray (time, y, x)
    ny, nx = da.sizes["y"], da.sizes["x"]

    # Rechunk time to -1 so each spatial tile holds the full time series.
    # This makes the subsequent temporal max/mean a single in-memory reduce
    # per tile rather than a multi-chunk aggregation.
    da = da.chunk({"time": -1})

    # Compute coarsening factors to stay within max_dim × max_dim
    cf_y = max(1, ny // max_dim)
    cf_x = max(1, nx // max_dim)

    # Coarsen spatially first (still lazy), then reduce over time
    da_c = da.coarsen(y=cf_y, x=cf_x, boundary="trim").mean()

    if metric == "peak_ndvi_mean":
        z = da_c.max(dim="time").compute().values
    elif metric == "mean_ndvi":
        z = da_c.mean(dim="time").compute().values
    elif metric == "std_ndvi":
        z = da_c.std(dim="time").compute().values
    elif metric == "data_coverage":
        z = da_c.notnull().mean(dim="time").compute().values
    else:
        raise ValueError(f"Unknown on-the-fly basemap metric: {metric!r}")

    # Build WGS84 coordinates and reproject onto a regular grid so that
    # go.Heatmap axes (lon[0,:] / lat[:,0]) are truly axis-aligned.
    if src_epsg is None:
        src_epsg = detect_crs_epsg(ds)
    x_c = da_c["x"].values
    y_c = da_c["y"].values
    xx, yy = np.meshgrid(x_c, y_c)
    lon_2d, lat_2d = utm_to_latlon(xx.ravel(), yy.ravel(), src_epsg)
    lon_2d = lon_2d.reshape(yy.shape)
    lat_2d = lat_2d.reshape(yy.shape)

    return _regrid_to_wgs84(z, lon_2d, lat_2d, x_c, y_c, src_epsg)


def load_metrics_for_basemap(
    metrics_path: Path,
    metric: str,
    max_dim: int = BASEMAP_MAX_DIM,
    src_epsg: int = DATACUBE_CRS_EPSG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast path: load one 2-D metric slice from a precomputed pixel_metrics.nc.
    Returns (z_2d, lon_2d, lat_2d) at display resolution.
    """
    with xr.open_dataset(str(metrics_path), mask_and_scale=True) as ds_m:
        if metric not in ds_m:
            raise KeyError(
                f"Metric {metric!r} not found in {metrics_path}. "
                f"Available: {list(ds_m.data_vars)}"
            )
        da = ds_m[metric]
        ny, nx = da.sizes["y"], da.sizes["x"]
        cf_y = max(1, ny // max_dim)
        cf_x = max(1, nx // max_dim)
        if cf_y > 1 or cf_x > 1:
            da = da.coarsen(y=cf_y, x=cf_x, boundary="trim").mean()
        z = da.values
        x_c = da["x"].values
        y_c = da["y"].values

    xx, yy = np.meshgrid(x_c, y_c)
    lon_2d, lat_2d = utm_to_latlon(xx.ravel(), yy.ravel(), src_epsg)
    lon_2d = lon_2d.reshape(yy.shape)
    lat_2d = lat_2d.reshape(yy.shape)
    return _regrid_to_wgs84(z, lon_2d, lat_2d, x_c, y_c, src_epsg)


# ---------------------------------------------------------------------------
# Single-pixel time series — direct HDF5 hyperslab read
# ---------------------------------------------------------------------------

def extract_pixel_timeseries(
    nc_path: Path,
    yi: int,
    xi: int,
    vi_var: str = DEFAULT_VI_VAR,
) -> PixelTimeSeries:
    """
    Extract the full time series for pixel (yi, xi) using a direct netCDF4
    hyperslab read.  This reads only T floats (≈ 5 KB) — never the full array.

    Parameters
    ----------
    nc_path : path to the .nc datacube file
    yi, xi  : zero-based array indices (row = y direction, col = x direction)
    vi_var  : name of the VI variable in the file

    Returns
    -------
    PixelTimeSeries namedtuple
    """
    with nc4.Dataset(str(nc_path), mode="r") as ds:
        # Decode time to datetime64
        time_var = ds.variables["time"]
        time_vals = np.array(time_var[:])
        units = getattr(time_var, "units", "days since 1970-01-01")

        if "days since" in units.lower():
            origin_str = units.lower().replace("days since", "").strip().split()[0]
            origin = np.datetime64(origin_str, "D")
            dates = origin + time_vals.astype("timedelta64[D]")
        else:
            # Fall back to netCDF4 num2date
            import cftime
            dts = nc4.num2date(time_vals, units, only_use_cftime_datetimes=False)
            dates = np.array(
                [np.datetime64(str(d)[:10], "D") for d in dts]
            )

        # Hyperslab read — only the pixel's time series
        vi_data = ds.variables[vi_var][:, yi, xi]
        vi_arr = np.array(vi_data, dtype=np.float32)

        # Mask fill value → NaN
        fill = getattr(ds.variables[vi_var], "_FillValue", None)
        if fill is not None and not np.isnan(fill):
            vi_arr[vi_arr == fill] = np.nan

        # Spatial coordinates
        x_val = float(ds.variables["x"][xi])
        y_val = float(ds.variables["y"][yi])

    # Apply valid-range mask
    vi_min, vi_max = VI_VALID_RANGE.get(vi_var, (-1.0, 2.0))
    valid_mask = (
        ~np.isnan(vi_arr)
        & (vi_arr >= vi_min)
        & (vi_arr <= vi_max)
    )

    lon_arr, lat_arr = utm_to_latlon(np.array([x_val]), np.array([y_val]))

    return PixelTimeSeries(
        dates=dates,
        raw_vi=vi_arr,
        valid_mask=valid_mask,
        x_coord=x_val,
        y_coord=y_val,
        lon=float(lon_arr[0]),
        lat=float(lat_arr[0]),
    )


# ---------------------------------------------------------------------------
# Click coordinate → array index mapping
# ---------------------------------------------------------------------------

def click_to_array_index(
    click_lon: float,
    click_lat: float,
    ds: xr.Dataset,
    src_epsg: int | None = None,
) -> tuple[int, int]:
    """
    Convert a Plotly heatmap click's (lon, lat) display coordinates to
    zero-based (yi, xi) array indices via nearest-neighbour lookup.

    Parameters
    ----------
    click_lon, click_lat : WGS84 coordinates from the Plotly click event
    ds                   : lazily opened xarray Dataset for the region
    src_epsg             : UTM EPSG of the datacube; auto-detected if None

    Returns
    -------
    (yi, xi) clamped to valid array bounds
    """
    if src_epsg is None:
        src_epsg = detect_crs_epsg(ds)

    x_click, y_click = latlon_to_utm(click_lon, click_lat, src_epsg)

    x_vals = ds["x"].values  # (nx,) — coordinate arrays are tiny
    y_vals = ds["y"].values  # (ny,)

    xi = int(np.argmin(np.abs(x_vals - x_click)))
    yi = int(np.argmin(np.abs(y_vals - y_click)))

    xi = max(0, min(xi, len(x_vals) - 1))
    yi = max(0, min(yi, len(y_vals) - 1))

    return yi, xi
