#!/usr/bin/env python3
"""
tools/pixel_phenology_extract.py — Batch per-pixel phenological metric extraction.

For every pixel in one or more VI datacubes this script:
  1. Reads the pixel time series via a direct netCDF4 hyperslab read.
  2. Applies Whittaker smoothing (same lambda as the dashboard default).
  3. Computes all 19 phenological metrics using the logic in
     modules/phenology_metrics.py.
  4. Writes the results to a pixel_metrics.nc file alongside the datacube.

Once pixel_metrics.nc exists the dashboard can display any of the 19 metrics
as a spatial basemap without re-computing them on the fly.

Output file location
--------------------
  {DATACUBE_ROOT}/LVIS_flightboxes_final/{region}/{VI}_{region}_pixel_metrics.nc

Usage
-----
  # Single region:
  python tools/pixel_phenology_extract.py --region G5_1

  # All discovered regions (skips any that already have pixel_metrics.nc):
  python tools/pixel_phenology_extract.py --all

  # Custom smoothing lambda and minimum-observation threshold:
  python tools/pixel_phenology_extract.py --region G5_14 --lambda 100 --min-obs 20

  # Overwrite an existing file:
  python tools/pixel_phenology_extract.py --region G5_1 --overwrite

  # Control parallelism (default: all logical CPUs):
  python tools/pixel_phenology_extract.py --region G5_14 --workers 6
  python tools/pixel_phenology_extract.py --region G5_14 --workers 1  # serial

Performance notes
-----------------
  - Row chunks are distributed across worker processes via ProcessPoolExecutor.
  - Each worker opens its own netCDF4 file handle and rebuilds the Whittaker
    matrix once; no shared state between processes.
  - Default parallelism: all logical CPUs (os.cpu_count()).  Use --workers to
    override (e.g. --workers 1 for serial, --workers 6 to leave headroom).
  - Typical speed with 10 cores: 500×500 grid → 1–3 min;
    G5_14 (2222×409) → 10–15 min.
  - Memory per worker: ~(chunk_rows × nx × n_days × 8 B) for the row slab
    plus ~(chunk_rows × nx × 19 × 4 B) for the partial result arrays.
  - Progress is shown chunk-by-chunk via tqdm when it is installed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make modules/ and config.py importable regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import netCDF4 as nc4
import numpy as np

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from config import (
    ALL_19_METRICS,
    DEFAULT_VI_VAR,
    LAMBDA_DEFAULT,
    PIXEL_METRIC_CONFIG,
    VI_VALID_RANGE,
)
from modules.datacube_io import build_date_cache, discover_regions, get_dataset
from modules.phenology_metrics import _cached_whittaker_system, _extract_pixel_metrics


# Standard NetCDF fill value for float32 (replaced by NaN when xarray reads
# the file with mask_and_scale=True).
_NC_FILL_F32 = np.float32(9.96921e+36)


# ---------------------------------------------------------------------------
# Worker function (must be at module level for multiprocessing "spawn")
# ---------------------------------------------------------------------------

def _worker_process_rows(
    nc_path: str,
    row_start: int,
    row_end: int,
    nx: int,
    n_days: int,
    lam: float,
    config: dict,
    date_cache: dict,
    vi_var: str,
    fill_val: float | None,
) -> tuple[int, dict[str, np.ndarray]]:
    """
    Process rows [row_start, row_end) of one datacube in a worker process.

    Returns
    -------
    row_start : int
        First row index processed (used by the main process to place results).
    partial : dict[str, np.ndarray]
        Mapping of metric name → float32 array of shape (chunk_ny, nx).
        Pixels without sufficient valid observations remain at _NC_FILL_F32.
    """
    # Rebuild the Whittaker penalty matrix once per worker (cheap; lru_cache
    # avoids redundant work if the same worker handles multiple chunks).
    lam_DTD = _cached_whittaker_system(n_days, float(lam))

    vi_min    = config.get("vi_min",        -0.1)
    vi_max    = config.get("vi_max",         1.0)
    min_valid = config.get("min_valid_obs",   20)
    chunk_ny  = row_end - row_start

    partial: dict[str, np.ndarray] = {
        m: np.full((chunk_ny, nx), _NC_FILL_F32, dtype=np.float32)
        for m in ALL_19_METRICS
    }

    with nc4.Dataset(nc_path, mode="r") as nc_ds:
        vi_ncvar = nc_ds.variables[vi_var]

        for local_i in range(chunk_ny):
            yi = row_start + local_i
            # Hyperslab: full time series for all x in this row.
            row_data = np.array(vi_ncvar[:, yi, :], dtype=np.float64)

            if fill_val is not None and not np.isnan(fill_val):
                row_data[row_data == fill_val] = np.nan

            for xi in range(nx):
                pixel_ts = row_data[:, xi]

                valid = (
                    ~np.isnan(pixel_ts)
                    & (pixel_ts >= vi_min)
                    & (pixel_ts <= vi_max)
                )
                if int(valid.sum()) < min_valid:
                    continue  # partial stays at fill_value

                metrics = _extract_pixel_metrics(
                    pixel_ts, lam_DTD, config, date_cache
                )
                for m in ALL_19_METRICS:
                    v = metrics.get(m, np.nan)
                    partial[m][local_i, xi] = (
                        _NC_FILL_F32 if np.isnan(float(v)) else np.float32(v)
                    )

    return row_start, partial


# ---------------------------------------------------------------------------
# Per-region orchestrator
# ---------------------------------------------------------------------------

def _process_region(
    region_id: str,
    paths,
    lam: float,
    config: dict,
    overwrite: bool,
    vi_var: str,
    n_workers: int,
) -> None:
    """
    Compute per-pixel metrics for one region and write pixel_metrics.nc.

    Parameters
    ----------
    region_id : e.g. "G5_1"
    paths     : RegionPaths dataclass from discover_regions()
    lam       : Whittaker smoothing lambda
    config    : PIXEL_METRIC_CONFIG dict (may have modified min_valid_obs)
    overwrite : if False, skip regions that already have pixel_metrics.nc
    vi_var    : name of the VI variable inside the datacube (e.g. "NDVI")
    n_workers : number of parallel worker processes
    """
    out_name = f"{vi_var}_{region_id}_pixel_metrics.nc"
    out_path = paths.nc_path.parent / out_name

    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name} already exists "
              f"(pass --overwrite to regenerate)")
        return

    print(f"  Opening {paths.nc_path.name} …")
    t0 = time.time()

    # Build date_cache from the time coordinate (cheap: reads ~5 KB).
    ds_xr      = get_dataset(paths)
    date_cache = build_date_cache(ds_xr)
    n_days     = date_cache["n_days"]

    lam_DTD = _cached_whittaker_system(n_days, float(lam))
    status  = "ready" if lam_DTD is not None else "skipped (n_days < 3)"
    print(f"  n_days={n_days}, lambda={lam:.0f}, Whittaker matrix: {status}")

    y_vals = ds_xr["y"].values   # (ny,) UTM northings
    x_vals = ds_xr["x"].values   # (nx,) UTM eastings
    ny     = len(y_vals)
    nx     = len(x_vals)
    print(f"  Grid: {ny} rows × {nx} cols = {ny * nx:,} pixels")
    print(f"  Workers: {n_workers}")

    # Read fill value from the source file.
    with nc4.Dataset(str(paths.nc_path), mode="r") as nc_ds:
        vi_ncvar = nc_ds.variables[vi_var]
        fill_val = getattr(vi_ncvar, "_FillValue", None)
        if fill_val is not None:
            fill_val = float(fill_val)

    # ------------------------------------------------------------------
    # Build row chunks — one per worker for maximum contiguous I/O.
    # Each chunk is a half-open interval [row_start, row_end).
    # ------------------------------------------------------------------
    actual_workers = min(n_workers, ny)
    base, remainder = divmod(ny, actual_workers)
    chunks: list[tuple] = []
    row = 0
    for i in range(actual_workers):
        end = row + base + (1 if i < remainder else 0)
        chunks.append((
            str(paths.nc_path),
            row, end,
            nx, n_days, lam,
            config, date_cache,
            vi_var, fill_val,
        ))
        row = end

    # Allocate full output arrays.
    results: dict[str, np.ndarray] = {
        m: np.full((ny, nx), _NC_FILL_F32, dtype=np.float32)
        for m in ALL_19_METRICS
    }

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------
    if actual_workers == 1:
        # Single-worker: run inline to avoid subprocess spawn overhead.
        iter_chunks = [chunks[0]]
        if _HAS_TQDM:
            iter_chunks = tqdm(
                iter_chunks,
                desc=f"  {region_id}",
                unit="chunk",
                leave=True,
                dynamic_ncols=True,
            )
        for chunk in iter_chunks:
            row_start, partial = _worker_process_rows(*chunk)
            chunk_ny = partial[ALL_19_METRICS[0]].shape[0]
            for m in ALL_19_METRICS:
                results[m][row_start:row_start + chunk_ny, :] = partial[m]
    else:
        pbar = (
            tqdm(
                total=ny,
                desc=f"  {region_id}",
                unit="row",
                leave=True,
                dynamic_ncols=True,
            )
            if _HAS_TQDM else None
        )
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_map = {
                executor.submit(_worker_process_rows, *chunk): chunk
                for chunk in chunks
            }
            for future in as_completed(future_map):
                row_start, partial = future.result()
                chunk_ny = partial[ALL_19_METRICS[0]].shape[0]
                for m in ALL_19_METRICS:
                    results[m][row_start:row_start + chunk_ny, :] = partial[m]
                if pbar is not None:
                    pbar.update(chunk_ny)
        if pbar is not None:
            pbar.close()

    # ------------------------------------------------------------------
    # Write output NetCDF4
    # ------------------------------------------------------------------
    print(f"  Writing {out_path.name} …")
    with nc4.Dataset(str(out_path), mode="w", format="NETCDF4") as out:
        out.description = (
            f"Per-pixel phenological metrics — {region_id} "
            f"(VI={vi_var}, lambda={lam:.0f})"
        )
        out.source_datacube = str(paths.nc_path)
        out.lambda_value    = float(lam)
        out.min_valid_obs   = int(config.get("min_valid_obs", 20))

        out.createDimension("y", ny)
        out.createDimension("x", nx)

        yv           = out.createVariable("y", "f8", ("y",))
        yv[:]        = y_vals
        yv.units     = "m"
        yv.long_name = "UTM northing"

        xv           = out.createVariable("x", "f8", ("x",))
        xv[:]        = x_vals
        xv.units     = "m"
        xv.long_name = "UTM easting"

        for m in ALL_19_METRICS:
            var = out.createVariable(
                m, "f4", ("y", "x"),
                fill_value=_NC_FILL_F32,
                zlib=True,
                complevel=4,
            )
            var[:]          = results[m]
            var.long_name   = m.replace("_", " ")

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"  Finished in {mins}m {secs}s  →  {out_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    _cpu_count = os.cpu_count() or 1

    parser = argparse.ArgumentParser(
        prog="pixel_phenology_extract",
        description=(
            "Compute per-pixel phenological metrics for VI datacubes and "
            "save pixel_metrics.nc files for use by the dashboard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        "--region",
        metavar="ID",
        help="Process a single region, e.g. --region G5_1",
    )
    region_group.add_argument(
        "--all",
        action="store_true",
        help="Process all regions discovered in DATACUBE_ROOT",
    )

    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=LAMBDA_DEFAULT,
        metavar="N",
        help=(
            f"Whittaker smoothing lambda.  Use the same value as the "
            f"dashboard lambda slider for consistency.  Default: {LAMBDA_DEFAULT}"
        ),
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=PIXEL_METRIC_CONFIG["min_valid_obs"],
        metavar="N",
        help=(
            f"Minimum valid observations required to compute metrics for a "
            f"pixel.  Default: {PIXEL_METRIC_CONFIG['min_valid_obs']}"
        ),
    )
    parser.add_argument(
        "--vi",
        default=DEFAULT_VI_VAR,
        metavar="VAR",
        help=f"VI variable name inside the datacube.  Default: {DEFAULT_VI_VAR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite pixel_metrics.nc if it already exists.  "
            "Without this flag existing files are skipped."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_cpu_count,
        metavar="N",
        help=(
            f"Number of parallel worker processes.  "
            f"Default: {_cpu_count} (all logical CPUs on this machine).  "
            f"Use --workers 1 to run serially without subprocesses."
        ),
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    config = dict(PIXEL_METRIC_CONFIG)
    config["min_valid_obs"] = args.min_obs

    try:
        all_regions = discover_regions()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not all_regions:
        print(
            "No regions found. Check VI_DATACUBE_ROOT or config.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.all:
        targets = list(all_regions.items())
    else:
        if args.region not in all_regions:
            available = ", ".join(all_regions)
            print(
                f"Region {args.region!r} not found.\n"
                f"Available regions: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        targets = [(args.region, all_regions[args.region])]

    print(
        f"VI Phenology Metric Extraction\n"
        f"  Regions  : {len(targets)}\n"
        f"  Lambda   : {args.lam}\n"
        f"  Min obs  : {args.min_obs}\n"
        f"  VI var   : {args.vi}\n"
        f"  Workers  : {args.workers}\n"
        f"  Overwrite: {args.overwrite}\n"
    )

    total_t0 = time.time()
    errors = []

    for region_id, paths in targets:
        print(f"[{region_id}]")
        try:
            _process_region(
                region_id=region_id,
                paths=paths,
                lam=args.lam,
                config=config,
                overwrite=args.overwrite,
                vi_var=args.vi,
                n_workers=args.workers,
            )
        except Exception as exc:
            msg = f"  ERROR processing {region_id}: {exc}"
            print(msg, file=sys.stderr)
            errors.append(region_id)
        print()

    total_elapsed = time.time() - total_t0
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    print(f"All done in {total_mins}m {total_secs}s.  "
          f"Errors: {errors if errors else 'none'}")


if __name__ == "__main__":
    main()
