# BioSCape Phenology Explorer — Claude Code Context

## Project Overview

Interactive Shiny for Python dashboard for exploring vegetation index (VI) phenology
datacubes from the BioSCape / LVIS South Africa flight campaign.  Lets users select
a flight-box region (via dropdown or by clicking a shapefile polygon), view a spatial
metric basemap, click any 30 m pixel, and instantly see its Whittaker-smoothed VI
time series plus 19 per-pixel phenological metrics.

## Data

**Default data root:** `/Volumes/ConklinGeospatialData/Data/BioSCape_SA_LVIS/VI_Phenology/netcdf_datacube`
Override with `export VI_DATACUBE_ROOT=/your/path` or edit `DATACUBE_ROOT` in `config.py`.

**Supported VIs (auto-detected from filename):** NDVI, EVI2, NIRv
**CRS:** UTM Zone 34S (EPSG:32734); reprojected to Web Mercator (EPSG:3857) for display
**18 flight-box regions** (G5_1 – G5_25, subset); ~30 m, 2016–2019

## Running

```bash
conda activate vi_phenology_dashboard
shiny run app.py            # http://127.0.0.1:8000
shiny run app.py --reload   # development mode
```

## Key Files

| File | Role |
|---|---|
| `app.py` | Shiny Core API entry point; reactive graph |
| `config.py` | All constants — data paths, VI ranges, shapefile paths, metric config |
| `environment.yml` | Conda environment (`vi_phenology_dashboard`) |
| `modules/datacube_io.py` | File discovery, lazy Dask loading, Zarr fast path, pixel extraction, basemap cache, reprojection |
| `modules/phenology_metrics.py` | Whittaker smoothing + 19 per-pixel phenological metrics |
| `modules/visualization.py` | ipyleaflet map factory, Plotly figure helpers, shapefile overlay |
| `tools/convert_to_zarr.py` | One-time CLI to rechunk NC → Zarr (fast pixel reads) |
| `tools/cache_basemaps.py` | One-time CLI to pre-compute and save basemap `.npz` caches |
| `tools/pixel_phenology_extract.py` | Batch per-pixel metric extraction → `pixel_metrics.nc` |
| `shapefiles/` | LVIS_Flightboxes.geojson, BioSCape_HLS_Tiles.geojson |
| `docs/methods.md` | Scientific and architectural methods reference |

## Architecture Notes

### Never load the full datacube
Always use Dask lazy reads for basemap computation or HDF5 hyperslab reads for pixel
extraction.  The largest file (G5_14) is 2.6 GB compressed / ~23 GB decompressed.

### Reactive graph (Shiny Core API)
- `selected_idx` — `reactive.Value` storing `(region_id, yi, xi)`
- `effective_selected_idx` — `@reactive.Calc` that returns `(yi, xi)` only when the
  stored region matches the current input; single choke-point for stale-selection detection
- `narrowed_timeseries()` — applies both the VI amplitude filter and year range filter;
  every downstream reactive reads from here, never from the raw pixel series.
  The colorscale amplitude filter is only applied when the active basemap metric is
  VI-scaled (one of the four quick metrics); phenology metrics (DOY, days, etc.) have
  colorscale values in different units and must not gate VI observations.
- `pixel_date_cache()` — rebuilds date metadata from the clipped time window so the
  Whittaker grid is sized exactly to the selected years (no extrapolation)
- `_sf_select_region` — `reactive.Value[str | None]`; set by the GeoJSON `on_click`
  callback (non-reactive context); a `@reactive.Effect` picks it up and calls
  `ui.update_select` in a proper reactive context
- `_sf_notify` — `reactive.Value[str | None]`; same pattern for no-data warnings

### Shapefile click → region navigation
- On initial app load the map zooms to the union bounding box of all configured
  shapefiles (`_SHAPEFILE_OVERVIEW_BOUNDS`, computed at startup from stdlib JSON).
- Clicking any `LVIS_Flightboxes.geojson` polygon navigates to the matching region
  (matched by `box_nr` field against `ALL_REGIONS` keys).  Clicking a polygon with
  no matching datacube shows a warning notification.
- Clicking the *currently active* region's polygon is silently ignored so it never
  re-triggers a region reload while the user is selecting pixels.
- All other shapefiles (e.g. HLS tiles) are display-only — no click handler.
- `_on_interaction` (map-level pixel click) checks `_sf_select_region() is not None`
  at the top and bails out early when a region change is already queued; this prevents
  spurious phenology computations and concurrent image-overlay updates that can
  destabilise Leaflet rendering.
- `_set_map_view(m, bounds, zoom_offset)` sets `m.center` / `m.zoom` as widget traits
  (not `fit_bounds()` which uses `send()` and requires an open comm channel).

### Basemap reprojection pipeline
- Source data is in UTM Zone 34S (EPSG:32734), a rotated grid relative to geographic north.
- `_regrid_to_mercator()` in `datacube_io.py` reprojects to a regular Web Mercator (EPSG:3857)
  grid via nearest-neighbour lookup (pyproj EPSG:3857 → EPSG:32734, then scipy
  `RegularGridInterpolator`).  WGS84 lat/lon is derived algebraically from the Mercator grid
  for the `ImageOverlay` bounds — no extra transform step.
- Coarsening factors (`cf_y`, `cf_x`) are computed independently per axis so each dimension
  stays within `BASEMAP_MAX_DIM` without degrading the finer axis.
- The returned `(z, lon, lat)` arrays are stored in `.npz` caches; loading a cache skips
  both the Dask compute and the reprojection entirely.

### Three-tier basemap load order
1. Disk `.npz` cache (< 1 s) — pre-populate with `python tools/cache_basemaps.py --all`
2. Dask on-the-fly compute (3–25 s) — shows a browser progress bar
3. Precomputed `pixel_metrics.nc` (for phenology metrics) — generated by `pixel_phenology_extract.py`

### Zarr fast pixel reads
With a Zarr store (`time=-1, y=10, x=10` chunks) a pixel click reads ~580 KB instead
of decompressing the entire NC file (~23 GB for G5_14).  Convert with
`python tools/convert_to_zarr.py --all`.

### pixel_metrics.nc naming convention
`{vi_var}_{region_id}_pixel_metrics.nc` — e.g. `NDVI_G5_25_pixel_metrics.nc`.
`discover_regions()` checks this canonical name first, then falls back to
`{nc_stem}_pixel_metrics.nc`.

## Config Shortcuts

```python
# config.py — shapefile overlays (space-separated paths)
SHAPEFILE_PATHS        = "shapefiles/flight_boxes.geojson shapefiles/hls_tiles.geojson"
SHAPEFILE_LABEL_FIELDS = "box_nr Name"   # one per file; last reused if fewer than files

# Add a new VI type:
VI_VALID_RANGE["MYVI"] = (-0.2, 1.5)
```

## Known Display Issues (Fixed)

| Issue | Root cause | Fix |
|---|---|---|
| Season Length, Peak DOY std, Peak Separation showing ~-9.2e18 on basemap | xarray decodes variables with `units="days"` as `timedelta64`; float32 fill value overflows to int64 min | `decode_timedelta=False` in `load_metrics_for_basemap` (`modules/datacube_io.py`) |
| Negative lower colorscale bound for non-negative metrics (DOY, days, counts) | `colorscale_limits()` returns `mean - N*std < 0` | Floor `zmin = max(0.0, zmin)` for `NONNEGATIVE_METRICS` in `app.py:colorscale_limits()` |
| Std band shading extends below zero in annual metric trend plots | `lo = mean_val - std_val` unclamped | `lo = max(0.0, lo)` for `NONNEGATIVE_METRICS` in `modules/visualization.py` |
| Data Coverage basemap shows full bounding box instead of flight strip only | `notnull().mean()` returns 0.0 (not NaN) for all-nodata pixels | `z[z == 0.0] = np.nan` applied after every basemap load path in `app.py:basemap_data()` and `modules/datacube_io.py:compute_basemap_metric()` |

`NONNEGATIVE_METRICS` is defined in `config.py` — the frozenset of metrics bounded below by zero.
Excluded (can be legitimately negative): `peak_ndvi_mean`, `integrated_ndvi_mean`, `floor_ndvi_mean`, `ceiling_ndvi_mean`.

## Coding Conventions

- All configuration constants live in `config.py` — do not hardcode paths or thresholds elsewhere
- VI variable name is always dynamic (`vi_var` from region metadata), never hardcoded as `"NDVI"`
- `lru_cache(maxsize=18)` on dataset-open functions — all region handles cached for session lifetime
- `compute_pixel_with_annual()` returns a 5-tuple and is called once per (pixel, λ, range, year-range);
  all four plot tabs derive from that single result
- Prefer in-place mutation of ipyleaflet layer properties over remove/re-add to avoid map resets
