#!/usr/bin/env python3
"""
tools/add_crs_to_pixel_metrics.py — Add CRS and CF-1.8 compliance attributes
to existing pixel_metrics.nc files.

The pixel_phenology_extract.py script writes UTM x/y coordinates but does not
encode the coordinate reference system.  This script:

  1. Discovers all *_pixel_metrics.nc files under SOURCE_DIR.
  2. Reads the CRS from each file's corresponding source datacube
     (via the ``source_datacube`` global attribute), mirroring the
     detect_crs_epsg() logic in modules/datacube_io.py.
  3. Copies each file to OUT_DIR (flat, no sub-folders).
  4. Appends CF-1.8 CRS metadata to each copy without rewriting data.
  5. Runs an inline CF-1.8 compliance check and prints a report.

Usage
-----
  # Process all files with default paths:
  python tools/add_crs_to_pixel_metrics.py

  # Specify custom directories:
  python tools/add_crs_to_pixel_metrics.py \\
      --source-dir /path/to/LVIS_flightboxes_final \\
      --out-dir    /path/to/pixel_metrics_with_CRS

  # Overwrite any existing output files:
  python tools/add_crs_to_pixel_metrics.py --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Make project root importable regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import netCDF4 as nc4
from pyproj import CRS

from config import DATACUBE_CRS_EPSG, METRIC_LABELS

# ---------------------------------------------------------------------------
# Default paths (can be overridden via CLI)
# ---------------------------------------------------------------------------
_DATACUBE_ROOT = Path(
    "/Volumes/ConklinGeospatialData/Data/BioSCape_SA_LVIS/VI_Phenology/netcdf_datacube"
)
_DEFAULT_SOURCE_DIR = _DATACUBE_ROOT / "LVIS_flightboxes_final"
_DEFAULT_OUT_DIR    = _DATACUBE_ROOT / "pixel_metrics_with_CRS"

# CF-1.8 coordinate attributes to add
_X_ATTRS = {"standard_name": "projection_x_coordinate", "axis": "X"}
_Y_ATTRS = {"standard_name": "projection_y_coordinate", "axis": "Y"}


# ---------------------------------------------------------------------------
# CRS discovery
# ---------------------------------------------------------------------------

class CRSResult(NamedTuple):
    wkt: str
    epsg: int | None
    source: str  # human-readable description of where it came from


def _discover_crs(metrics_path: Path) -> CRSResult:
    """
    Read the CRS from the source datacube referenced by the pixel_metrics file.

    Priority order (mirrors datacube_io.detect_crs_epsg):
      1. source datacube's spatial_ref.crs_wkt attribute
      2. source datacube's spatial_ref.spatial_ref attribute
      3. Fallback: DATACUBE_CRS_EPSG from config.py
    """
    # Step 1: read source_datacube path from the metrics file
    try:
        with nc4.Dataset(str(metrics_path), mode="r") as ds_m:
            src_path_str = getattr(ds_m, "source_datacube", None)
    except Exception as exc:
        return _fallback_crs(f"could not open metrics file ({exc})")

    if not src_path_str:
        return _fallback_crs("source_datacube attribute is missing or empty")

    src_path = Path(src_path_str)
    if not src_path.exists():
        return _fallback_crs(f"source datacube not found: {src_path}")

    # Step 2: try to read spatial_ref from the source datacube
    try:
        with nc4.Dataset(str(src_path), mode="r") as ds_src:
            if "spatial_ref" not in ds_src.variables:
                return _fallback_crs(
                    f"no spatial_ref variable in {src_path.name}"
                )
            sr_var = ds_src["spatial_ref"]
            wkt = (
                getattr(sr_var, "crs_wkt", None)
                or getattr(sr_var, "spatial_ref", None)
            )
            if not wkt:
                return _fallback_crs(
                    f"spatial_ref in {src_path.name} has no crs_wkt / spatial_ref attr"
                )
            crs = CRS.from_wkt(wkt)
            epsg = crs.to_epsg()
            return CRSResult(
                wkt=wkt,
                epsg=epsg,
                source=f"discovered from {src_path.name}",
            )
    except Exception as exc:
        return _fallback_crs(f"error reading {src_path.name}: {exc}")


def _fallback_crs(reason: str) -> CRSResult:
    crs = CRS.from_epsg(DATACUBE_CRS_EPSG)
    return CRSResult(
        wkt=crs.to_wkt(),
        epsg=DATACUBE_CRS_EPSG,
        source=f"FALLBACK to EPSG:{DATACUBE_CRS_EPSG} (config.DATACUBE_CRS_EPSG) — {reason}",
    )


# ---------------------------------------------------------------------------
# CRS / CF attribute writer
# ---------------------------------------------------------------------------

def _add_crs_attributes(dst_path: Path, crs_result: CRSResult) -> None:
    """
    Open dst_path in append mode and write all missing CF-1.8 CRS metadata.
    Data variables and compression are not touched.
    """
    crs = CRS.from_wkt(crs_result.wkt)
    cf_params = crs.to_cf()
    grid_mapping_name = cf_params.get("grid_mapping_name", "transverse_mercator")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with nc4.Dataset(str(dst_path), mode="a") as ds:
        # ---- global attributes ----------------------------------------
        if not getattr(ds, "Conventions", None):
            ds.Conventions = "CF-1.8"
        existing_history = getattr(ds, "history", "")
        new_entry = (
            f"{timestamp}: Added CRS (EPSG:{crs_result.epsg or 'unknown'}, "
            f"{grid_mapping_name}; {crs_result.source}) and CF-1.8 "
            f"compliance attributes via add_crs_to_pixel_metrics.py."
        )
        ds.history = (existing_history + "\n" + new_entry).strip()

        # ---- spatial_ref scalar variable --------------------------------
        if "spatial_ref" not in ds.variables:
            sr = ds.createVariable("spatial_ref", "i4", ())
            sr[:] = 0
        else:
            sr = ds["spatial_ref"]

        sr.grid_mapping_name = grid_mapping_name
        sr.crs_wkt           = crs_result.wkt
        sr.spatial_ref       = crs_result.wkt   # GDAL compatibility alias

        # Copy remaining CF parameters from pyproj (semi-major axis, etc.)
        for attr_name, attr_val in cf_params.items():
            if attr_name != "grid_mapping_name":
                try:
                    setattr(sr, attr_name, attr_val)
                except Exception:
                    pass  # skip any parameter pyproj returns that netCDF4 rejects

        # ---- coordinate variable attributes ----------------------------
        if "y" in ds.variables:
            yv = ds["y"]
            for k, v in _Y_ATTRS.items():
                if not getattr(yv, k, None):
                    setattr(yv, k, v)

        if "x" in ds.variables:
            xv = ds["x"]
            for k, v in _X_ATTRS.items():
                if not getattr(xv, k, None):
                    setattr(xv, k, v)

        # ---- data variable attributes ----------------------------------
        coord_var_names = {"x", "y", "spatial_ref"}
        for varname in ds.variables:
            if varname in coord_var_names:
                continue
            var = ds[varname]
            # grid_mapping
            if not getattr(var, "grid_mapping", None):
                var.grid_mapping = "spatial_ref"
            # units — add from METRIC_LABELS if not already present
            if not getattr(var, "units", None) and varname in METRIC_LABELS:
                _, unit_str = METRIC_LABELS[varname]
                if unit_str:
                    var.units = unit_str


# ---------------------------------------------------------------------------
# CF-1.8 compliance checker
# ---------------------------------------------------------------------------

class CheckResult(NamedTuple):
    check: str
    passed: bool
    detail: str


def _check_cf_compliance(path: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    with nc4.Dataset(str(path), mode="r") as ds:
        # 1. Conventions
        conv = getattr(ds, "Conventions", "")
        results.append(CheckResult(
            "Conventions = CF-1.8",
            "CF-1.8" in conv,
            repr(conv) if conv else "MISSING",
        ))

        # 2. spatial_ref variable with crs_wkt
        has_sr = "spatial_ref" in ds.variables
        if has_sr:
            sr = ds["spatial_ref"]
            has_wkt = bool(getattr(sr, "crs_wkt", None))
            results.append(CheckResult(
                "spatial_ref.crs_wkt present",
                has_wkt,
                "OK" if has_wkt else "crs_wkt attribute missing",
            ))
        else:
            results.append(CheckResult(
                "spatial_ref variable present",
                False,
                "MISSING",
            ))

        # 3. x / y standard_name
        for dim, expected in [("x", "projection_x_coordinate"),
                               ("y", "projection_y_coordinate")]:
            if dim in ds.variables:
                sn = getattr(ds[dim], "standard_name", "")
                results.append(CheckResult(
                    f"{dim}.standard_name",
                    sn == expected,
                    repr(sn) if sn else "MISSING",
                ))
            else:
                results.append(CheckResult(
                    f"{dim}.standard_name",
                    False,
                    f"coordinate '{dim}' not found",
                ))

        # 4. x / y axis attribute
        for dim, expected in [("x", "X"), ("y", "Y")]:
            if dim in ds.variables:
                ax = getattr(ds[dim], "axis", "")
                results.append(CheckResult(
                    f"{dim}.axis",
                    ax == expected,
                    repr(ax) if ax else "MISSING",
                ))

        # 5. x / y units
        for dim in ("x", "y"):
            if dim in ds.variables:
                u = getattr(ds[dim], "units", "")
                results.append(CheckResult(
                    f"{dim}.units",
                    bool(u),
                    repr(u) if u else "MISSING",
                ))

        # 6. data variables: grid_mapping present
        coord_names = {"x", "y", "spatial_ref"}
        no_gm = [
            v for v in ds.variables
            if v not in coord_names and not getattr(ds[v], "grid_mapping", None)
        ]
        results.append(CheckResult(
            "all data vars have grid_mapping",
            len(no_gm) == 0,
            "OK" if not no_gm else f"missing: {no_gm}",
        ))

        # 7. data variables: _FillValue present
        no_fv = [
            v for v in ds.variables
            if v not in coord_names and ds[v].get_fill_value() is None
        ]
        results.append(CheckResult(
            "all data vars have _FillValue",
            len(no_fv) == 0,
            "OK" if not no_fv else f"missing: {no_fv}",
        ))

        # 8. file format is NETCDF4
        fmt = ds.file_format
        results.append(CheckResult(
            "format is NETCDF4",
            fmt == "NETCDF4",
            fmt,
        ))

    return results


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    src: Path,
    out_dir: Path,
    overwrite: bool,
    verbose: bool,
) -> dict:
    """Copy src to out_dir, add CRS, run compliance check. Returns status dict."""
    dst = out_dir / src.name

    if dst.exists() and not overwrite:
        return {"file": src.name, "status": "SKIPPED (already exists)", "crs_source": "-", "checks": []}

    # -- discover CRS before copying so we can report without touching dst yet
    crs_result = _discover_crs(src)
    is_fallback = crs_result.source.startswith("FALLBACK")

    # -- copy
    shutil.copy2(src, dst)

    # -- append attributes
    _add_crs_attributes(dst, crs_result)

    # -- compliance check
    checks = _check_cf_compliance(dst)
    all_pass = all(c.passed for c in checks)

    if verbose:
        print(f"\n  {src.name}")
        print(f"    CRS: {crs_result.source}")
        for c in checks:
            mark = "PASS" if c.passed else "FAIL"
            print(f"    [{mark}] {c.check}: {c.detail}")

    return {
        "file": src.name,
        "status": "OK" if all_pass else "WARN",
        "crs_source": "fallback" if is_fallback else "datacube",
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="add_crs_to_pixel_metrics",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        metavar="DIR",
        default=str(_DEFAULT_SOURCE_DIR),
        help=f"Root directory to search for *_pixel_metrics.nc files "
             f"(default: {_DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default=str(_DEFAULT_OUT_DIR),
        help=f"Output directory for CRS-annotated copies "
             f"(default: {_DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file check details; show summary table only",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir    = Path(args.out_dir)
    verbose    = not args.quiet

    if not source_dir.exists():
        print(f"ERROR: source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = sorted(source_dir.rglob("*_pixel_metrics.nc"))
    if not metrics_files:
        print(f"No *_pixel_metrics.nc files found under {source_dir}")
        sys.exit(0)

    print(f"Found {len(metrics_files)} pixel_metrics.nc file(s)")
    print(f"Output directory: {out_dir}\n")

    results = []
    for src in metrics_files:
        result = process_file(src, out_dir, args.overwrite, verbose)
        results.append(result)
        if not verbose:
            mark = "[OK]  " if result["status"] == "OK" else (
                   "[SKIP]" if result["status"].startswith("SKIPPED") else "[WARN]"
            )
            print(f"  {mark}  {result['file']}")

    # ---- summary table ---------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'File':<45} {'Status':<8} {'CRS from'}")
    print("-" * 70)
    n_ok   = 0
    n_warn = 0
    n_skip = 0
    n_fail_checks: list[tuple[str, list[CheckResult]]] = []

    for r in results:
        status = r["status"]
        if status == "OK":
            n_ok += 1
        elif status.startswith("SKIPPED"):
            n_skip += 1
        else:
            n_warn += 1
        print(f"  {r['file']:<43} {status:<8} {r['crs_source']}")
        failed = [c for c in r["checks"] if not c.passed]
        if failed:
            n_fail_checks.append((r["file"], failed))

    print("=" * 70)
    print(
        f"\nProcessed: {n_ok + n_warn} file(s)  |  "
        f"OK: {n_ok}  |  Warnings: {n_warn}  |  Skipped: {n_skip}"
    )

    if n_fail_checks:
        print("\nFailed compliance checks:")
        for fname, checks in n_fail_checks:
            print(f"  {fname}")
            for c in checks:
                print(f"    [FAIL] {c.check}: {c.detail}")
    else:
        print("\nAll compliance checks PASSED.")


if __name__ == "__main__":
    main()
