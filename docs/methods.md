# VI Phenology Dashboard — Scientific & Technical Methods

## 1. Data Sources

### 1.1 Input Datacubes

| Parameter | Value |
|---|---|
| Campaign | BioSCape (Biodiversity Survey of the Cape), South Africa |
| Instrument | NASA G-V (N95ND) HLS / HLS-2 satellite imagery |
| Spatial footprint | LVIS flight box regions (G5_1 through G5_25, 18 regions used) |
| Temporal coverage | January 2016 – June 2019 (~3.5 years) |
| Temporal resolution | Irregular (1–3 day cadence; cloud/quality masking applied) |
| Spatial resolution | 30 m (HLS Sentinel-2 and Landsat-8/9 tiles) |
| VI variable | NDVI, EVI2, or NIRv — auto-detected from filename |
| Valid range | Per-VI (NDVI: −0.1–1.0; EVI2: −1.0–2.0; NIRv: −0.5–1.0) |

### 1.2 File Format

Datacubes are CF-1.8 compliant NetCDF4 (HDF5 backend) files with dimensions
`(time, y, x)` and coordinates stored as:

- `time` — `int32`, "days since 1970-01-01" (proleptic Gregorian calendar)
- `y` — `float64`, UTM northing in metres
- `x` — `float64`, UTM easting in metres
- `spatial_ref` — scalar variable carrying the full CRS WKT string

### 1.3 Coordinate Reference System

All datacubes in this dataset use **WGS 84 / UTM Zone 34S (EPSG:32734)**.
This projection covers the area between 18°E and 24°E in the southern hemisphere,
encompassing the Cape Floristic Region of South Africa.

For display in the Plotly heatmap, coordinates are reprojected to
**WGS84 geographic (EPSG:4326)** using `pyproj.Transformer`.

---

## 2. Vegetation Indices

The dashboard supports three vegetation indices, auto-detected from the datacube filename.
All share the same Whittaker smoothing and 19-metric computation pipeline; only the
valid observation range and axis labelling differ.

### 2.1 NDVI

$$
\text{NDVI} = \frac{\rho_\text{NIR} - \rho_\text{RED}}{\rho_\text{NIR} + \rho_\text{RED}}
$$

Valid range: **−0.1 to 1.0**.  Values outside this range are masked to NaN.
Values above ~0.2 indicate photosynthetically active vegetation.

### 2.2 EVI2

$$
\text{EVI2} = 2.5 \cdot \frac{\rho_\text{NIR} - \rho_\text{RED}}{\rho_\text{NIR} + 2.4 \cdot \rho_\text{RED} + 1}
$$

A two-band Enhanced Vegetation Index that does not require a blue band.
Valid range: **−1.0 to 2.0**.  Less sensitive to soil background in sparse canopies.

### 2.3 NIRv

$$
\text{NIRv} = \text{NDVI} \times \rho_\text{NIR}
$$

Near-infrared reflectance of vegetation, proportional to the fraction of absorbed
photosynthetically active radiation (fAPAR).  Valid range: **−0.5 to 1.0**.

---

## 3. Whittaker Penalized Least-Squares Smoothing

### 3.1 Motivation

Satellite VI time series are unevenly sampled in time due to cloud cover, sensor
revisit cycles, and quality masking.  Direct fitting of annual phenology models to
gapped observations is unstable.  The Whittaker smoother interpolates gaps and
removes high-frequency noise while preserving phenological shape.

### 3.2 Mathematical Formulation

Observations are first mapped onto a **regular daily calendar grid**.  Let:

- **y** ∈ ℝⁿ — NDVI values on the daily grid (0 where no observation)
- **w** ∈ {0,1}ⁿ — binary weight vector (1 = observed, 0 = gap)
- **W** = diag(**w**) — diagonal weight matrix
- **D** ∈ ℝ^{(n−2)×n} — second-order finite difference matrix
- **λ** > 0 — smoothing parameter

The Whittaker smoother minimises the penalised weighted sum of squares:

$$
\hat{z} = \arg\min_z \left[ \sum_{i:w_i=1}(y_i - z_i)^2 + \lambda \sum_{j=3}^{n}(\Delta^2 z_j)^2 \right]
$$

where Δ²zⱼ = zⱼ − 2zⱼ₋₁ + zⱼ₋₂ is the second difference.

This is solved as a sparse linear system:

$$
(\mathbf{W} + \lambda \mathbf{D}^\top \mathbf{D})\hat{z} = \mathbf{W}\mathbf{y}
$$

using `scipy.sparse.linalg.spsolve`.

### 3.3 Lambda Parameter

| λ value | Effect |
|---|---|
| 10 – 50 | Tight fit to observations; short-period noise retained |
| 100 – 200 | Balanced smoothing; seasonal shape preserved |
| 500 (default) | Smooth seasonal envelope; intra-seasonal fluctuations suppressed |
| 1000 | Very smooth; only broad annual signal retained |

The dashboard exposes λ via a slider (10–1000, step 10).

### 3.4 Implementation Detail

The penalty matrix **λ D^T D** is precomputed once per (n_days, λ) combination
and cached via `functools.lru_cache`.  **n_days** is the calendar span from the
first to the last observation in the datacube (~1,262 days for 2016–2019).
This matrix is shared across all pixel clicks at the same λ setting, making
repeated pixel selections nearly instantaneous after the first.

The `spsolve` is called **once** per (pixel, λ, data-range) combination.
`compute_pixel_with_annual()` returns the smoothed daily curve alongside the
19 aggregated metrics and per-year arrays; the Raw VI and Annual Cycles displays
derive from this single result rather than running a separate solve.

### 3.5 Daily Grid Construction

Each observation at time step t is placed at integer position
`day_offset[t] = (date[t] - date[0]).days` on the daily grid.
If multiple observations fall on the same calendar day (rare with HLS), their
values are averaged before smoothing.

---

## 4. Phenological Metric Definitions

All metrics are computed per-pixel from the Whittaker-smoothed daily time series.
Per-year values are computed within annual calendar windows; multi-year statistics
are then computed across valid years.

A year is included in the multi-year statistics only if it contains at least
**5 valid observations** (`min_valid_obs_per_year = 5`).

### 4.1 Peak Group

**peak_ndvi_mean** — Mean annual peak NDVI across years:

$$
\overline{\text{NDVI}}_{\text{peak}} = \frac{1}{N_y}\sum_{y} \max_{d \in y} \hat{z}_d
$$

**peak_ndvi_std** — Interannual standard deviation of peak NDVI.

**peak_doy_mean** — Mean day-of-year of peak NDVI:

$$
\overline{\text{DOY}}_{\text{peak}} = \frac{1}{N_y}\sum_{y} \arg\max_{d \in y}(\hat{z}_d)
$$

**peak_doy_std** — Interannual standard deviation of peak DOY (in days).

### 4.2 Productivity Group

**integrated_ndvi_mean** — Mean annual integrated NDVI (area under the smoothed
curve, using the trapezoidal rule):

$$
\overline{\text{iNDVI}} = \frac{1}{N_y}\sum_y \int_{y} \hat{z}(d)\,\text{d}d \approx \frac{1}{N_y}\sum_y \text{trapz}(\hat{z}_{d \in y})
$$

Units: NDVI·days yr⁻¹.

**integrated_ndvi_std** — Interannual standard deviation of integrated NDVI.

**greenup_rate_mean** — Mean slope from the annual NDVI floor to the annual peak:

$$
\overline{\text{GUR}} = \frac{1}{N_y}\sum_y \frac{\hat{z}_{\text{peak},y} - \hat{z}_{\text{floor},y}}{\Delta d_{\text{floor}\to\text{peak},y}}
$$

Units: NDVI day⁻¹.

**greenup_rate_std** — Interannual standard deviation of green-up rate.

### 4.3 Seasonality Group

**floor_ndvi_mean** — Mean dry-season NDVI floor (annual minimum of smoothed curve):

$$
\overline{\text{NDVI}}_{\text{floor}} = \frac{1}{N_y}\sum_y \min_{d \in y} \hat{z}_d
$$

**ceiling_ndvi_mean** — Mean wet-season NDVI ceiling (annual maximum):

$$
\overline{\text{NDVI}}_{\text{ceiling}} = \frac{1}{N_y}\sum_y \max_{d \in y} \hat{z}_d
$$

**season_length_mean** — Mean number of days per year during which NDVI exceeds
the phenological threshold:

$$
\text{threshold} = \hat{z}_{\text{floor}} + \theta \cdot (\hat{z}_{\text{ceiling}} - \hat{z}_{\text{floor}})
$$

where θ = 0.20 (20% of annual amplitude) by default.  Season length is the count
of days above this threshold within the year.

**season_length_std** — Interannual standard deviation of season length (days).

### 4.4 Variability Group

**cv** — Coefficient of variation computed from all raw valid observations across
the full time series (not per year):

$$
\text{CV} = \frac{\sigma_{\text{raw}}}{\bar{y}_{\text{raw}}}
$$

This is the only metric computed from raw (un-smoothed) observations.

**interannual_peak_range** — Range of annual peak NDVI across years:

$$
\text{range} = \max_y(\hat{z}_{\text{peak},y}) - \min_y(\hat{z}_{\text{peak},y})
$$

**interannual_peak_std** — Standard deviation of annual peak NDVI across years.

### 4.5 Bimodality Group

These metrics characterise pixels with two distinct growing seasons per year.
Peak detection uses `scipy.signal.find_peaks` with:
- `prominence ≥ 0.05` (minimum peak prominence as fraction of range)
- `distance ≥ 45` days (minimum inter-peak separation)

**n_peaks_mean** — Mean number of detected peaks per year.

**peak_separation_mean** — Mean separation in days between the two highest peaks
per year (NaN if fewer than 2 peaks detected in a year).

**relative_peak_amplitude_mean** — Mean ratio of the smaller to the larger of the
two highest peaks:

$$
\text{rPA} = \frac{\min(\hat{z}_{\text{peak1}},\hat{z}_{\text{peak2}})}{\max(\hat{z}_{\text{peak1}},\hat{z}_{\text{peak2}})}
$$

Ranges from 0 (strongly asymmetric bimodality) to 1 (equal peak heights).

**valley_depth_mean** — Mean normalised depth of the valley between the two
highest peaks:

$$
\text{VD} = 1 - \frac{\hat{z}_{\text{valley}}}{\frac{\hat{z}_{\text{peak1}} + \hat{z}_{\text{peak2}}}{2}}
$$

Values near 0 indicate a shallow valley; values near 1 indicate a deep
inter-peak trough.

---

## 5. Dashboard Memory Architecture

### 5.1 Design Principle

The full datacube array is **never loaded into memory**.  The largest file
(G5_14) contains ~1.17 billion float32 values (~4.36 GB uncompressed after
decompression of a 2.6 GB compressed file).  Loading such a file would exceed
typical workstation RAM budgets and make multi-region sessions impossible.

### 5.2 Basemap Computation

The spatial basemap metric follows a three-tier load order (fastest first):

**Tier 1 — Disk cache (< 1 s).** On first compute the result is serialised to a
compressed `.npz` file alongside the datacube:

    {nc_stem}_basemap_{metric}_d{max_dim}.npz

On all subsequent loads (new browser tabs, app restarts, region switches) the
arrays are deserialized directly, bypassing all Dask computation.
The `tools/cache_basemaps.py` CLI pre-populates caches for every region offline.

**Tier 2 — Dask on-the-fly compute.** When no cache exists the metric is
computed lazily:

1. `xr.open_dataset(..., chunks={})` — opens the file respecting its native HDF5
   chunk layout; no data is read.
2. `da.chunk({'time': -1})` — rechunks the time axis so each spatial tile holds
   the full time series; still lazy.
3. `da.coarsen(y=cf_y, x=cf_x).mean()` — adds spatial averaging to the graph;
   still lazy.
4. `da_coarse.max(dim='time').compute()` — triggers execution.  Dask reads and
   decompresses each tile once, aggregates along time, then discards the tile.
   Peak memory ≈ one tile × time axis ≈ 200 × 200 × 1287 × 4 bytes ≈ 200 MB.
5. The result (≤ 500 × 500 float32) is written to the disk cache and returned.

A browser progress bar (`ui.Progress`) is shown while the Dask compute runs.
For the largest files this takes 10–25 seconds on the first load; the pre-warm
background thread computes and caches the default region's four fast metrics at
app startup so the first user generally sees cached data.

**Tier 3 — Precomputed pixel_metrics.nc.** For phenology metrics (e.g.,
season length) the dashboard reads directly from `{vi_var}_{region_id}_pixel_metrics.nc`
if present — no Dask computation is needed.  Generated by
`tools/pixel_phenology_extract.py`.

### 5.2.1 Basemap Display (ipyleaflet)

The metric array is reprojected from the UTM grid onto a regular WGS84 grid
(nearest-neighbour via `scipy.interpolate.RegularGridInterpolator`), then
rendered as a PNG image overlay on an `ipyleaflet.Map`.  A satellite tile
layer (default: ESRI World Imagery) is displayed underneath.  The tile service, overlay opacity, and **Data range** clipping (full / Mean ± N SD)
are all adjustable in the sidebar without triggering a Dask recompute.

The Data range control also acts as an **analysis filter** (see §5.2.2).

Pixel selection is handled by the `Map.on_interaction` callback: on a click event,
the (lat, lon) coordinates from ipyleaflet are converted back to UTM and
nearest-neighbour matched to an array index `(yi, xi)`.

### 5.2.2 Data Range Filter

The **Data range** sidebar control defines a VI sub-range [z_min, z_max].  It does
two things simultaneously:

1. **Colorbar clipping** — the colour ramp on the basemap overlay is stretched to
   [z_min, z_max] so spatial variation within the range is maximally visible.

2. **Observation filter** — all pixel-level analysis is restricted to observations
   within the range.  This is implemented via a `narrowed_timeseries()` reactive
   calculation that copies the `PixelTimeSeries` struct with a modified `valid_mask`:

   ```
   mask = original_valid_mask
         & (raw_vi >= z_min)
         & (raw_vi <= z_max)
   ```

   Every downstream reactive — Whittaker smoothing, all 19 phenological metrics,
   the raw-VI scatter, the Annual Cycles overlay, and the Metric Trends plots —
   reads from `narrowed_timeseries()` rather than the full pixel time series.

The `pixel_metric_config()` reactive additionally clamps the per-VI physical valid
range (from `VI_VALID_RANGE`) to [z_min, z_max], so metric-quality checks (e.g.,
`min_valid_obs`) are evaluated against the filtered observation count.

When no range is set, `narrowed_timeseries()` returns the original series unchanged
and there is no performance penalty.

The sidebar shows both the total valid observation count and the in-range count
(e.g., "In range: 782 / 842 valid (93%)") so the effect of the filter is immediately
visible.

### 5.3 Pixel Time Series Extraction

When a Zarr store is available the dashboard uses xarray's `.isel()` on the
Zarr dataset:

```python
ts = zarr_dataset[vi_var].isel(y=yi, x=xi).values
```

With Zarr chunks `{time: -1, y: 10, x: 10}` the selected pixel falls in
exactly one chunk (worst case: four chunks at a spatial chunk boundary).
Each chunk covers the full time axis for a 10 × 10 pixel block.  For G5_14
(1465 time steps) this decompresses ≈ **580 KB** to return ≈ 5 KB of values —
approximately 4 000× less I/O than the NC path for the same file.

When only a NetCDF4 file is available, the direct HDF5 hyperslab API is used:

```python
vi_arr = nc4_dataset.variables[vi_var][:, yi, xi]
```

The HDF5 library must decompress every spatial chunk that contains any part of
the requested hyperslab.  Because datacubes use chunk layout `[1, full_y, full_x]`
(one complete spatial frame per time step), reading a single pixel's time series
requires decompressing all T spatial chunks — effectively the entire file.
For G5_14 this is ≈ **23 GB** decompressed.  ZARR conversion is therefore
strongly recommended for large files (see §5.4).

All opened dataset handles are cached via `functools.lru_cache(maxsize=18)` so
re-opening the same region within a session is free.

### 5.4 ZARR Optimisation

All datacubes in this dataset use HDF5 chunk layout `[1, full_y, full_x]` — one
complete 2-D spatial frame per time step.  This layout is efficient for writing
time-stacked rasters but requires decompressing every spatial chunk to extract
a single pixel's time series (see §5.3).

Converting to Zarr with `{'time': -1, 'y': 10, 'x': 10}` places each 10 × 10
pixel block's full time series in a single Blosc-compressed chunk.  The dashboard
auto-detects `.zarr` directories and uses them for both basemap display (via Dask)
and pixel extraction (via `xr.Dataset.isel()`).

| NC file size | Pixel read (NC, no Zarr) | Pixel read (Zarr) | Speedup |
|---|---|---|---|
| 6.5 MB (G5_25) | < 100 ms | < 50 ms | ~2× |
| 509 MB (G5_1) | 2–5 s | < 100 ms | ~50× |
| 2.6 GB (G5_14) | 10–20 s | < 100 ms | ~150× |

Run `python tools/convert_to_zarr.py --all` once to convert all regions.
Disk space: ≈ 1–1.5× the NC file size per Zarr store.

---

## 6. Software Dependencies

| Package | Version | Role |
|---|---|---|
| Python | 3.11 | Runtime |
| shiny | ≥ 0.10 | Web framework and reactive system |
| shinywidgets | ≥ 0.3 | Plotly FigureWidget ↔ Shiny bridge |
| plotly | ≥ 5.17 | Interactive heatmap and time series |
| xarray | current | Lazy datacube access via Dask |
| dask | current | Parallel lazy computation |
| netCDF4 | current | Direct HDF5 hyperslab reads |
| scipy | current | Sparse linear solver (`spsolve`) for Whittaker |
| numpy | current | Array operations |
| pandas | current | Date arithmetic |
| pyproj | current | CRS reprojection (UTM → WGS84) |
| zarr | current | Rechunked storage format |
| pillow | current | Image utilities |

### 6.1 Core Metric Functions

The three Whittaker and metric functions are reproduced verbatim from the
VI_Phenology pipeline and live in `modules/phenology_metrics.py`:

| Function | Purpose |
|---|---|
| `_build_whittaker_system(n_days, lam)` | Builds sparse λ D^T D penalty matrix |
| `_whittaker_smooth_pixel(daily_y, daily_w, lam_DTD)` | Solves (W + λ D^T D) z = W y |
| `_extract_pixel_metrics(pixel_ts, lam_DTD, config, date_cache)` | Computes all 19 metrics |

They are copied rather than imported to avoid the heavy top-level dependencies
(matplotlib, tqdm, io_utils) in the original pipeline module.  The mathematical
logic is identical to the upstream source.

The dashboard is fully self-contained — no external VI_Phenology source tree is
required at runtime.

### 6.2 Time-Series Visualisation Panels

After a pixel is selected, three panels are available:

| Tab | Content |
|---|---|
| Raw VI | Raw observations (grey scatter) + Whittaker-smoothed daily curve (green line) over the full date range |
| Annual Cycles | Per-calendar-year DOY overlay: each year's smoothed curve plotted on a 1–366 axis, with a black cross-year mean trace |
| Metric Trends | Per-year scatter for each of the 11 trackable metrics, with a dashed mean line and grey ± std band |

All three panels update when λ changes or the Data range filter changes, as they
all share `narrowed_timeseries()` → `smoothed_result` → `pixel_annual_data` in the
reactive graph.  The y-axis bounds and label are set dynamically from `VI_VALID_RANGE`
for the current region's VI type.

---

## 7. Data Quality and Limitations

- **Cloud masking** — Quality-masked pixels appear as NaN.  Pixels with fewer
  than 20 valid observations (`min_valid_obs`) across the full time series produce
  `NaN` for all metrics.
- **Data range filter** — Setting a Data range further reduces the observation
  pool.  If the filtered count falls below `min_valid_obs`, all metrics show N/A
  for that pixel.  The sidebar shows both the full valid count and the in-range
  count to make this visible.
- **Bimodality metrics** — `peak_separation_mean`, `relative_peak_amplitude_mean`,
  and `valley_depth_mean` are `NaN` for pixels where fewer than 2 peaks are
  detected in most years.
- **Coordinate reprojection accuracy** — UTM→WGS84 reprojection introduces sub-pixel
  error at display resolution (< 0.5 pixel for regions < 200 km across).
- **Basemap downsampling** — The displayed heatmap is downsampled to ≤ 500 × 500 pixels
  for performance.  Sub-pixel spatial detail is not visible; the full-resolution
  pixel is used for time series extraction.
- **Lambda range** — λ must be > 0.  The slider minimum of 10 prevents near-zero
  values that would cause numerical instability in the sparse solver.

---

## 8. References

Eilers, P.H.C. (2003). A perfect smoother. *Analytical Chemistry*, 75(14), 3631–3636.
https://doi.org/10.1021/ac034173t

Sentinel-2 HLS product: Claverie, M. et al. (2018). The Harmonized Landsat and
Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145–161.

BioSCape campaign: https://www.bioscape.io
