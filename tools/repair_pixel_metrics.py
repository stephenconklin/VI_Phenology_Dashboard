#!/usr/bin/env python3
"""
tools/repair_pixel_metrics.py — Reset nodata pixels in existing pixel_metrics.nc files.

Existing *_pixel_metrics.nc files may contain computed values for pixels that
are nodata in the source datacube.  This script identifies those pixels by
re-reading the source datacube and resets them to the NetCDF fill value.

Root cause: pixel_phenology_extract.py had min_valid_obs=0 (from config.py),
which disabled the observation-count guard and allowed the Whittaker smoother
to produce spurious near-zero values for pixels with no valid observations.

Algorithm per file:
  1. Read source_datacube path from the pixel_metrics.nc global attributes.
  2. Read one row at a time from the source datacube; apply fill-value → NaN
     replacement and VI valid-range filter; record pixels below min_valid_obs.
  3. Open the pixel_metrics.nc in append mode and set all 19 metric variables
     to the fill value for every identified nodata pixel.

Usage
-----
  # Dry run — report counts without writing:
  python tools/repair_pixel_metrics.py --all --dry-run

  # Repair all regions (default paths from config.py):
  python tools/repair_pixel_metrics.py --all

  # Single region:
  python tools/repair_pixel_metrics.py --region G5_1

  # Custom observation threshold:
  python tools/repair_pixel_metrics.py --all --min-obs 10

  # Metrics files in a different directory:
  python tools/repair_pixel_metrics.py --all \
      --source-dir /path/to/my/pixel_metrics

  # Metrics files AND source datacubes in different locations:
  python tools/repair_pixel_metrics.py --all \
      --source-dir /path/to/my/pixel_metrics \
      --datacube-root /path/to/my/datacubes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import netCDF4 as nc4
import numpy as np

from config import (
    ALL_19_METRICS,
    DATACUBE_ROOT,
    VI_VALID_RANGE,
    DEFAULT_VI_VAR,
)

_NC_FILL_F32 = np.float32(9.96921e+36)

# ---------------------------------------------------------------------------
# Nodata mask builder
# ---------------------------------------------------------------------------

def _build_nodata_mask(
    nc_path: Path,
    vi_var: str,
    vi_min: float,
    vi_max: float,
    min_valid_obs: int,
) -> np.ndarray:
    """
    Return a bool (ny, nx) array where True = pixel is nodata (valid obs < min_valid_obs).

    Iterates over time slices rather than rows so that each read loads exactly
    one HDF5 chunk.  The source datacubes are chunked as [1, ny, nx] (one full
    spatial grid per time step), so reading [t, :, :] is perfectly aligned with
    the storage layout.  Reading row-by-row ([: , yi, :]) would force HDF5 to
    decompress all n_time chunks per row, making it ny × slower.
    """
    with nc4.Dataset(str(nc_path), mode="r") as ds:
        vi_ncvar = ds.variables[vi_var]

        fill_val = getattr(vi_ncvar, "_FillValue", None)
        if fill_val is not None:
            fill_val = float(fill_val)

        n_time, ny, nx = vi_ncvar.shape
        valid_count = np.zeros((ny, nx), dtype=np.int32)

        for t in range(n_time):
            # One read = one HDF5 chunk (perfectly aligned with [1, ny, nx] layout)
            slc = np.array(vi_ncvar[t, :, :], dtype=np.float64)  # (ny, nx)

            if fill_val is not None and not np.isnan(fill_val):
                slc[slc == fill_val] = np.nan

            valid_count += (
                ~np.isnan(slc) & (slc >= vi_min) & (slc <= vi_max)
            ).astype(np.int32)

    return valid_count < min_valid_obs


# ---------------------------------------------------------------------------
# Per-file repair
# ---------------------------------------------------------------------------

def _resolve_source_datacube(
    src_path_str: str,
    metrics_path: Path,
    datacube_root: Path | None,
) -> Path | None:
    """
    Resolve the source datacube path from the stored attribute string.

    Resolution order:
      1. The stored path as-is (if it exists).
      2. If --datacube-root was given, remap: strip the stored path's root
         up to and including the first component that also appears in
         DATACUBE_ROOT, then prepend datacube_root.
         e.g. stored = /old/root/LVIS_flightboxes_final/G5_1/file.nc
              datacube_root = /new/root
              result = /new/root/LVIS_flightboxes_final/G5_1/file.nc
      3. Look for the datacube filename alongside the metrics file.
    """
    stored = Path(src_path_str)
    if stored.exists():
        return stored

    if datacube_root is not None:
        # Find the longest suffix of stored that exists under datacube_root
        parts = stored.parts
        for i in range(1, len(parts)):
            candidate = datacube_root.joinpath(*parts[i:])
            if candidate.exists():
                return candidate

    # Co-location fallback: same directory as the metrics file
    sibling = metrics_path.parent / stored.name
    if sibling.exists():
        return sibling

    return None


def _repair_file(
    metrics_path: Path,
    min_valid_obs: int,
    dry_run: bool,
    vi_range_override: tuple[float, float] | None,
    datacube_root: Path | None,
) -> dict:
    """
    Repair one pixel_metrics.nc file.  Returns a status dict.
    """
    result = {"file": metrics_path.name, "nodata_pixels": 0, "status": "OK"}

    # Read metadata from the metrics file
    try:
        with nc4.Dataset(str(metrics_path), mode="r") as ds_m:
            src_path_str = getattr(ds_m, "source_datacube", None)
            # Try to infer vi_var from global attribute or filename
            vi_var = getattr(ds_m, "vi_var", None)
    except Exception as exc:
        result["status"] = f"ERROR opening metrics file: {exc}"
        return result

    if not src_path_str:
        result["status"] = "ERROR: source_datacube attribute missing"
        return result

    src_path = _resolve_source_datacube(src_path_str, metrics_path, datacube_root)
    if src_path is None:
        result["status"] = f"ERROR: source datacube not found: {src_path_str}"
        return result

    # Infer vi_var from filename if not in attributes (e.g. NDVI_G5_1_pixel_metrics.nc)
    if not vi_var:
        stem = metrics_path.stem  # e.g. "NDVI_G5_1_pixel_metrics"
        for candidate in VI_VALID_RANGE:
            if stem.upper().startswith(candidate.upper()):
                vi_var = candidate
                break
        if not vi_var:
            vi_var = DEFAULT_VI_VAR

    # Determine VI valid range
    if vi_range_override:
        vi_min, vi_max = vi_range_override
    else:
        vi_min, vi_max = VI_VALID_RANGE.get(vi_var, VI_VALID_RANGE[DEFAULT_VI_VAR])

    # Build the nodata mask from the source datacube
    try:
        nodata_mask = _build_nodata_mask(src_path, vi_var, vi_min, vi_max, min_valid_obs)
    except Exception as exc:
        result["status"] = f"ERROR building nodata mask: {exc}"
        return result

    n_nodata = int(nodata_mask.sum())
    result["nodata_pixels"] = n_nodata

    if n_nodata == 0:
        result["status"] = "OK (no nodata pixels found)"
        return result

    if dry_run:
        result["status"] = f"DRY RUN ({n_nodata:,} pixels would be reset)"
        return result

    # Apply the mask: reset nodata pixels to fill value in all 19 metrics
    try:
        with nc4.Dataset(str(metrics_path), mode="a") as ds_m:
            for m in ALL_19_METRICS:
                if m not in ds_m.variables:
                    continue
                var = ds_m.variables[m]
                data = var[:].data.copy() if hasattr(var[:], "data") else np.array(var[:])
                data[nodata_mask] = _NC_FILL_F32
                var[:] = data
    except Exception as exc:
        result["status"] = f"ERROR writing repairs: {exc}"
        return result

    result["status"] = f"REPAIRED ({n_nodata:,} pixels reset)"
    return result


# ---------------------------------------------------------------------------
# Region discovery (mirrors discover_regions logic for metrics files)
# ---------------------------------------------------------------------------

def _discover_metrics_files(
    region_id: str | None = None,
    source_dir: Path | None = None,
) -> list[Path]:
    """
    Find all *_pixel_metrics.nc files under source_dir (or DATACUBE_ROOT).
    If region_id is given, return only files matching that region.
    """
    root = source_dir if source_dir is not None else DATACUBE_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")

    all_files = sorted(root.rglob("*_pixel_metrics.nc"))

    if region_id:
        all_files = [f for f in all_files if f"_{region_id}_pixel_metrics" in f.name]

    return all_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="repair_pixel_metrics",
        description=(
            "Reset nodata pixels in existing pixel_metrics.nc files by "
            "re-checking valid observation counts in the source datacube."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        "--region",
        metavar="ID",
        help="Repair a single region, e.g. --region G5_1",
    )
    region_group.add_argument(
        "--all",
        action="store_true",
        help="Repair all *_pixel_metrics.nc files found under --source-dir",
    )

    parser.add_argument(
        "--source-dir",
        dest="source_dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory to search for *_pixel_metrics.nc files.  "
            f"Default: DATACUBE_ROOT from config.py ({DATACUBE_ROOT})"
        ),
    )
    parser.add_argument(
        "--datacube-root",
        dest="datacube_root",
        metavar="DIR",
        default=None,
        help=(
            "Root directory where source datacubes are stored.  Only needed "
            "when the source_datacube path stored inside a metrics file no "
            "longer matches the actual datacube location (e.g. files were "
            "moved).  The script remaps stored paths by replacing their "
            "leading components with this directory."
        ),
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Minimum valid observations required for a pixel to keep its "
            "metrics.  Pixels below this threshold are reset to nodata.  "
            "Default: 1 (reset only pixels with zero valid observations — "
            "true nodata).  Use a higher value (e.g. --min-obs 20) to also "
            "reset sparsely-observed pixels."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many pixels would be reset per file without writing.",
    )

    args = parser.parse_args()

    source_dir    = Path(args.source_dir)    if args.source_dir    else None
    datacube_root = Path(args.datacube_root) if args.datacube_root else None

    try:
        targets = _discover_metrics_files(
            region_id=args.region if not args.all else None,
            source_dir=source_dir,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    search_root = source_dir or DATACUBE_ROOT
    if not targets:
        print(
            f"No *_pixel_metrics.nc files found"
            + (f" for region {args.region!r}" if args.region else "")
            + f" under {search_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"VI Phenology Metrics Repair\n"
        f"  Source dir   : {search_root}\n"
        + (f"  Datacube root: {datacube_root}\n" if datacube_root else "")
        + f"  Files        : {len(targets)}\n"
        f"  Min obs      : {args.min_obs}\n"
        f"  Dry run      : {args.dry_run}\n"
    )

    total_t0 = time.time()
    total_reset = 0
    errors = []

    for metrics_path in targets:
        print(f"  {metrics_path.name} …", end=" ", flush=True)
        t0 = time.time()
        result = _repair_file(
            metrics_path=metrics_path,
            min_valid_obs=args.min_obs,
            dry_run=args.dry_run,
            vi_range_override=None,
            datacube_root=datacube_root,
        )
        elapsed = time.time() - t0
        print(f"{result['status']}  ({elapsed:.1f}s)")
        if result["status"].startswith("ERROR"):
            errors.append(metrics_path.name)
        else:
            total_reset += result["nodata_pixels"]

    total_elapsed = time.time() - total_t0
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    action = "would be reset" if args.dry_run else "reset"
    print(
        f"\nDone in {total_mins}m {total_secs}s  |  "
        f"Total pixels {action}: {total_reset:,}  |  "
        f"Errors: {errors if errors else 'none'}"
    )


if __name__ == "__main__":
    main()
