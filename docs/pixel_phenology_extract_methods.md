# Pixel Phenology Extraction — Scientific Methods and Product Documentation

`tools/pixel_phenology_extract.py`

---

## 1. Purpose and Overview

`pixel_phenology_extract.py` computes 19 per-pixel phenological metrics from
multi-year vegetation index (VI) time series and writes the results to a
spatially-referenced NetCDF4 file (`pixel_metrics.nc`).  These output files
are the primary data product intended for downstream spatial analysis and
predictive modelling.

Each metric is derived from a Whittaker-smoothed daily VI time series and
summarises a distinct phenological property of the annual vegetation cycle:
peak greenness, productivity, seasonality, interannual variability, and
bimodality.  Multi-year statistics (mean, standard deviation, range) are
computed across all calendar years that pass quality thresholds; the resulting
values represent pixel-level summaries of the 2016–2019 observation record.

---

## 2. Input Data

### 2.1 Source Datacubes

| Property | Value |
|---|---|
| Format | CF-1.8 NetCDF4 (HDF5 backend) |
| Dimensions | `(time, y, x)` |
| Spatial resolution | 30 m |
| CRS | WGS 84 / UTM Zone 34S (EPSG:32734) |
| `y` coordinate | UTM northing (m), `float64` |
| `x` coordinate | UTM easting (m), `float64` |
| `time` coordinate | `int32`, "days since 1970-01-01" |
| VI variable | One of NDVI, EVI2, NIRv — auto-detected from filename |
| Fill value | Per-file `_FillValue` attribute; treated as NaN |

Temporal coverage is January 2016 – June 2019 (approximately 3.5 years,
irregular cadence of 1–3 days depending on cloud cover and quality masking).

### 2.2 Valid Observation Ranges

Observations outside these per-VI ranges are treated as invalid (set to NaN)
and excluded from all computations:

| VI | Valid range |
|---|---|
| NDVI | −0.1 to 1.0 |
| EVI2 | −1.0 to 2.0 |
| NIRv | −0.5 to 1.0 |

The practical effect is that cloud-contaminated retrievals, water surfaces,
and deep shadow pixels are suppressed before any smoothing or metric extraction.

---

## 3. Processing Pipeline

For each pixel the following steps are applied in order.

### 3.1 Time Series Extraction

A direct HDF5 hyperslab read retrieves the full time-axis vector for the
pixel location `(yi, xi)`:

```
pixel_ts[t]  =  VI_datacube[t, yi, xi]
```

The per-file `_FillValue` is replaced with NaN.  Observations outside the
valid VI range are also masked to NaN at this stage.

### 3.2 Quality Gate

Pixels with fewer than `min_valid_obs` (default: **20**) valid observations
across the entire time series are skipped; all output metrics are set to the
NetCDF fill value for those pixels.  This threshold excludes heavily cloud-
masked locations and permanently shadowed terrain.

### 3.3 Daily Grid Construction

Irregular satellite observations are mapped onto a **regular daily calendar
grid** spanning from the date of the first observation to the date of the
last observation (`n_days` grid points).

Each observation at time step *t* is placed at integer position:

```
day_offset[t]  =  (date[t] − date[0]).days
```

If multiple observations fall on the same calendar day (rare), their values
are averaged.  Grid cells with no observation receive weight 0; observed
cells receive weight 1.  This produces:

- **y** ∈ ℝⁿ — daily VI values (averaged where needed; 0 elsewhere)
- **w** ∈ {0,1}ⁿ — binary weight vector

### 3.4 Whittaker Penalised Least-Squares Smoothing

Observations are smoothed using a second-order Whittaker smoother, which
minimises the penalised weighted sum of squares:

$$
\hat{z} = \arg\min_z \left[
    \sum_{i:\,w_i=1}(y_i - z_i)^2
    \;+\; \lambda \sum_{j=3}^{n}(\Delta^2 z_j)^2
\right]
$$

where Δ²zⱼ = zⱼ − 2zⱼ₋₁ + zⱼ₋₂ is the second finite difference, and
λ is the smoothing parameter.  Equivalently, the solution is:

$$
(\mathbf{W} + \lambda\,\mathbf{D}^\top\mathbf{D})\,\hat{z} = \mathbf{W}\,\mathbf{y}
$$

where **W** = diag(**w**) and **D** ∈ ℝ^{(n−2)×n} is the second-order finite
difference matrix.  This sparse linear system is solved with
`scipy.sparse.linalg.spsolve`.

**Lambda (λ):**  Controls the trade-off between fidelity to observations and
smoothness of the resulting curve.  Higher λ produces a smoother curve that
tracks only the broad seasonal envelope; lower λ retains more intra-seasonal
variation.

| λ | Character |
|---|---|
| 10–50 | Tight fit; high-frequency noise retained |
| 100–200 | Balanced; seasonal shape preserved |
| **500 (default)** | Smooth seasonal envelope; sub-seasonal fluctuations suppressed |
| 1000 | Very smooth; only broad annual signal retained |

The default λ = 500 is the same value used by the interactive dashboard
so that spatial basemaps and pixel-level displays are consistent with the
precomputed metrics.

The sparse penalty matrix **λ D^T D** is precomputed once per unique
(n_days, λ) combination and reused across all pixels in a region; it is
not recomputed per pixel.

After solving, the smoothed curve is clipped to the valid VI range
[vi_min, vi_max].

### 3.5 Annual Loop — Per-Year Metric Computation

The smoothed daily series is split into **calendar-year windows**.  For each
year *y* the following per-year values are computed (subject to the per-year
quality gate described in §3.6):

| Symbol | Description |
|---|---|
| ẑ(y) | Smoothed daily VI values within year *y* |
| DOY(y) | Day-of-year index for each day in year *y* |
| d_peak | Day index of max(ẑ(y)) within year *y* |
| d_floor | Day index of min(ẑ(y)) within year *y* |

### 3.6 Per-Year Quality Gate

A year is included in the annual statistics only if both conditions are met:

1. The year window contains at least **30 calendar days** of smoothed data.
2. The year has at least `min_valid_obs_per_year` (default: **0**) raw valid
   observations within it.

The default of 0 for `min_valid_obs_per_year` means the only active gate for
the batch extraction is the 30-day calendar span check.  Users requiring a
stricter per-year observation threshold can override this with `--min-obs`
(which sets the global `min_valid_obs`) and by editing `PIXEL_METRIC_CONFIG`
in `config.py`.

### 3.7 Multi-Year Aggregation

After the annual loop, each per-year list is aggregated across all valid years
N_y to produce the output metrics.  Safe aggregation is used:  NaN values
within a per-year list are silently excluded before computing the mean or
standard deviation, so a single bad year does not nullify the multi-year
aggregate.

---

## 4. Metric Definitions

All 19 metrics are written as 2-D spatial arrays (`y × x`) in the output
NetCDF4 file.

### 4.1 Peak Group

**`peak_ndvi_mean`** — Mean annual peak VI (dimensionless, VI units)

$$
\overline{\text{VI}}_{\text{peak}} = \frac{1}{N_y}\sum_{y} \max_{d \in y}\,\hat{z}_d
$$

The maximum of the smoothed curve within each calendar year, averaged across
valid years.

---

**`peak_ndvi_std`** — Interannual standard deviation of annual peak VI
(VI units)

$$
\sigma_{\text{peak}} = \text{std}\left\{\max_{d \in y}\,\hat{z}_d\right\}_{y=1}^{N_y}
$$

A measure of year-to-year fluctuation in peak greenness.

---

**`peak_doy_mean`** — Mean day-of-year of annual peak VI (DOY, 1–366)

$$
\overline{\text{DOY}}_{\text{peak}} = \frac{1}{N_y}\sum_{y}\,\arg\max_{d \in y}(\hat{z}_d)
$$

The average calendar timing of the annual vegetation peak.  Values near 1
indicate early-January peaks; values near 182 indicate mid-year peaks.

---

**`peak_doy_std`** — Interannual standard deviation of peak DOY (days)

Year-to-year variability in the timing of peak greenness.

---

### 4.2 Productivity Group

**`integrated_ndvi_mean`** — Mean annual integrated VI (VI·days yr⁻¹)

$$
\overline{\text{iVI}} = \frac{1}{N_y}\sum_y \int_y \hat{z}(d)\,\text{d}d
\;\approx\; \frac{1}{N_y}\sum_y \text{trapz}\!\left(\hat{z}_{d \in y}\right)
$$

The area under the smoothed daily curve within each calendar year, computed
via the trapezoidal rule and averaged across years.  Proportional to
cumulative photosynthetic activity (a proxy for gross primary productivity
under stable atmospheric and biophysical conditions).

---

**`integrated_ndvi_std`** — Interannual standard deviation of integrated VI
(VI·days)

---

**`greenup_rate_mean`** — Mean green-up rate (VI day⁻¹)

$$
\overline{\text{GUR}} = \frac{1}{N_y}\sum_y
    \frac{\hat{z}_{\text{peak},y} - \hat{z}_{\text{floor},y}}
         {\text{DOY}_{\text{peak},y} - \text{DOY}_{\text{floor},y}}
$$

The slope of the smoothed curve from the annual minimum to the annual maximum.
Only computed for years where the annual minimum (floor) occurs **before** the
annual maximum in calendar time; years where the maximum precedes the minimum
are excluded from this metric.  This constraint ensures the rate captures the
ascending limb of the seasonal growth cycle.

---

**`greenup_rate_std`** — Interannual standard deviation of green-up rate
(VI day⁻¹)

---

### 4.3 Seasonality Group

**`floor_ndvi_mean`** — Mean annual VI floor (VI units)

$$
\overline{\text{VI}}_{\text{floor}} = \frac{1}{N_y}\sum_y \min_{d \in y}\,\hat{z}_d
$$

Average dry-season minimum.  Reflects baseline photosynthetic activity during
the period of lowest greenness.

---

**`ceiling_ndvi_mean`** — Mean annual VI ceiling (VI units)

$$
\overline{\text{VI}}_{\text{ceiling}} = \frac{1}{N_y}\sum_y \max_{d \in y}\,\hat{z}_d
$$

Equivalent to `peak_ndvi_mean`.  Retained as a distinct variable to pair with
`floor_ndvi_mean` when characterising the full seasonal amplitude.

---

**`season_length_mean`** — Mean phenological season length (days)

The number of days per year during which VI exceeds the phenological threshold:

$$
\text{threshold}_y = \hat{z}_{\text{floor},y} + \theta \cdot
    \left(\hat{z}_{\text{ceiling},y} - \hat{z}_{\text{floor},y}\right)
$$

where θ = **0.20** (20 % of the annual amplitude) by default.

Season length is the span (in days) between the first and last day exceeding
this threshold within the calendar year:

$$
\text{SL}_y = d_{\text{last above}} - d_{\text{first above}}
$$

The metric is undefined (NaN) for years where the annual amplitude is below
1 × 10⁻⁶ (effectively zero-amplitude years) or where fewer than two days
exceed the threshold.

---

**`season_length_std`** — Interannual standard deviation of season length
(days)

---

### 4.4 Variability Group

**`cv`** — Coefficient of variation (dimensionless ratio)

$$
\text{CV} = \frac{\sigma_{\text{raw}}}{\bar{y}_{\text{raw}}}
$$

Computed from all **raw (unsmoothed) valid observations** across the full
multi-year time series (not per-year).  This is the only metric that bypasses
Whittaker smoothing.  Undefined (NaN) when the mean of raw observations is
zero or negative.

---

**`interannual_peak_range`** — Range of annual peak VI across years (VI units)

$$
\text{range} = \max_y\!\left(\hat{z}_{\text{peak},y}\right)
              - \min_y\!\left(\hat{z}_{\text{peak},y}\right)
$$

The difference between the highest and lowest annual peak observed across
the study period.

---

**`interannual_peak_std`** — Standard deviation of annual peak VI across years
(VI units)

Numerically identical to `peak_ndvi_std`; retained as a distinct variable
within the Variability metric group for model feature organisation.

---

### 4.5 Bimodality Group

These metrics quantify pixels with two distinct growing seasons per year.
Peak detection is performed on the smoothed annual curve using
`scipy.signal.find_peaks` with the following thresholds:

| Parameter | Default | Description |
|---|---|---|
| `peak_prominence` | 0.05 | Minimum peak prominence (absolute VI units) |
| `peak_min_distance_days` | 45 | Minimum separation between detected peaks (days) |

A peak is a local maximum whose height above the nearest surrounding minimum
exceeds 0.05 VI units, separated from any other peak by at least 45 days.
These thresholds prevent noisy inflections from being counted as bimodal peaks.

---

**`n_peaks_mean`** — Mean number of detected peaks per year (peaks yr⁻¹)

Values near 1 indicate unimodal seasonality; values near 2 indicate consistent
bimodality.  Non-integer values reflect interannual variation in peak count.

---

**`peak_separation_mean`** — Mean separation between the two tallest peaks
per year (days)

$$
\overline{\text{PS}} = \frac{1}{N_y}\sum_y\,
    \left|\text{DOY}_{p_1,y} - \text{DOY}_{p_2,y}\right|
$$

where p₁ and p₂ are the two highest detected peaks ranked by VI value.
Set to NaN for years with fewer than two detected peaks.

---

**`relative_peak_amplitude_mean`** — Mean ratio of secondary to primary peak
(dimensionless, 0–1)

$$
\overline{\text{rPA}} = \frac{1}{N_y}\sum_y
    \frac{\min\!\left(\hat{z}_{p_1,y},\,\hat{z}_{p_2,y}\right)}
         {\max\!\left(\hat{z}_{p_1,y},\,\hat{z}_{p_2,y}\right)}
$$

A value of 1.0 indicates equal peak heights (symmetric bimodality); a value
near 0 indicates that one peak strongly dominates.  NaN when fewer than two
peaks are detected.

---

**`valley_depth_mean`** — Mean normalised valley depth between the two tallest
peaks (dimensionless, 0–1)

$$
\overline{\text{VD}} = \frac{1}{N_y}\sum_y
    \frac{\bar{z}_{\text{peaks},y} - \hat{z}_{\text{valley},y}}
         {\bar{z}_{\text{peaks},y}}
$$

where $\bar{z}_{\text{peaks},y} = (\hat{z}_{p_1,y} + \hat{z}_{p_2,y}) / 2$
is the mean of the two peak heights, and $\hat{z}_{\text{valley},y}$ is the
minimum of the smoothed curve between the two peaks.

Values near 0 indicate a shallow inter-peak trough (weak bimodality); values
near 1 indicate a deep trough approaching zero (strong bimodality with
near-complete senescence between seasons).  NaN when fewer than two peaks are
detected.

---

## 5. Output File Specification

### 5.1 File Naming and Location

```
{DATACUBE_ROOT}/LVIS_flightboxes_final/{region_id}/{VI}_{region_id}_pixel_metrics.nc
```

Example: `NDVI_G5_14_pixel_metrics.nc`

### 5.2 File Format and Global Attributes

| Attribute | Value |
|---|---|
| Format | NetCDF4 (HDF5 backend) |
| Conventions | CF-1.8 |
| `description` | Per-pixel phenological metrics — {region_id} (VI={vi_var}, lambda={λ}) |
| `source_datacube` | Absolute path to the source NC4 file |
| `lambda_value` | Whittaker smoothing λ used to generate the file (float) |
| `min_valid_obs` | Minimum valid observations threshold used (integer) |

### 5.3 Dimensions and Coordinate Variables

| Variable | Dims | Type | Units | Description |
|---|---|---|---|---|
| `y` | (y) | float64 | m | UTM northing |
| `x` | (x) | float64 | m | UTM easting |
| `spatial_ref` | scalar | int32 | — | CF-1.8 grid mapping variable |

The `spatial_ref` variable carries the full CRS WKT string (copied from the
source datacube's `spatial_ref` variable, or derived from EPSG:32734 as
fallback) in two attributes:

- `crs_wkt` — CF-1.8 standard attribute
- `spatial_ref` — GDAL/rasterio compatibility alias (identical WKT string)

All metric variables carry `grid_mapping = "spatial_ref"`.

### 5.4 Metric Variables

All 19 metric variables are written as `float32` arrays with shape `(y, x)`:

| Variable name | Long name | Units |
|---|---|---|
| `peak_ndvi_mean` | Peak NDVI (mean) | NDVI |
| `peak_ndvi_std` | Peak NDVI (std) | NDVI |
| `peak_doy_mean` | Peak DOY (mean) | DOY |
| `peak_doy_std` | Peak DOY (std) | days |
| `integrated_ndvi_mean` | Integrated NDVI (mean) | NDVI·days yr⁻¹ |
| `integrated_ndvi_std` | Integrated NDVI (std) | NDVI·days |
| `greenup_rate_mean` | Green-up Rate (mean) | NDVI day⁻¹ |
| `greenup_rate_std` | Green-up Rate (std) | NDVI day⁻¹ |
| `floor_ndvi_mean` | Floor NDVI | NDVI |
| `ceiling_ndvi_mean` | Ceiling NDVI | NDVI |
| `season_length_mean` | Season Length (mean) | days |
| `season_length_std` | Season Length (std) | days |
| `cv` | Coeff. of Variation | ratio |
| `interannual_peak_range` | Interannual Peak Range | NDVI |
| `interannual_peak_std` | Interannual Peak Std | NDVI |
| `n_peaks_mean` | N Peaks (mean) | peaks yr⁻¹ |
| `peak_separation_mean` | Peak Separation (mean) | days |
| `relative_peak_amplitude_mean` | Relative Peak Amplitude | ratio |
| `valley_depth_mean` | Valley Depth (mean) | normalized |

### 5.5 Fill Value and Masking

The NetCDF standard float32 fill value (`9.96921e+36`, IEEE 754 OFV) is used
for:

- Pixels with fewer than `min_valid_obs` valid raw observations
- Pixels where all raw observations are NaN (no-data pixels, e.g. ocean,
  permanent cloud, or outside the flight strip)
- Individual metrics that are conditionally undefined (e.g. bimodality metrics
  for unimodal pixels; green-up rate when the floor occurs after the peak)

When the file is opened with xarray or any CF-aware reader (`mask_and_scale=True`),
fill values are automatically decoded as `NaN`.  When read with raw NetCDF4 or
GDAL, apply the `_FillValue` attribute to mask nodata pixels.

**Important:** The variable `units` attribute for `peak_doy_mean` and
`season_length_mean` is set to `"DOY"` and `"days"` respectively, but the
variables are written as plain `float32` scalars — they are **not** encoded as
`timedelta64`.  The file is opened with `decode_timedelta=False` by the
dashboard to prevent xarray from misinterpreting these variables.  Any workflow
reading these files should do the same.

### 5.6 Compression

All metric variables are written with zlib compression at level 4.
Typical compressed file sizes: 1–20 MB per region depending on grid dimensions.

---

## 6. Extraction Parameters and Defaults

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| Smoothing lambda | `--lambda` | 500 | Whittaker λ (higher = smoother) |
| Min valid obs (global) | `--min-obs` | 20 | Min valid obs across full time series |
| VI variable | `--vi` | NDVI | Variable name in source datacube |
| Overwrite | `--overwrite` | False | Re-generate if file already exists |
| Workers | `--workers` | all CPUs | Parallel worker process count |
| Chunk rows | `--chunk-rows` | 20 | Rows per work unit (progress granularity) |

### 6.1 Fixed Parameters (config.py)

| Parameter | Value | Description |
|---|---|---|
| `min_valid_obs_per_year` | 0 | Per-year minimum valid obs (0 = only 30-day span check) |
| `peak_prominence` | 0.05 | Minimum peak prominence for bimodality detection (VI units) |
| `peak_min_distance_days` | 45 | Minimum inter-peak separation (days) |
| `season_threshold` (θ) | 0.20 | Fractional amplitude threshold for season length |

These parameters are defined in `PIXEL_METRIC_CONFIG` in `config.py` and are
applied uniformly to all pixels and all regions.  Changing them requires
regenerating the output files.

---

## 7. Usage

```bash
# Single region
python tools/pixel_phenology_extract.py --region G5_1

# All regions (skips existing files)
python tools/pixel_phenology_extract.py --all

# All regions, starting from a specific region (resumes interrupted run)
python tools/pixel_phenology_extract.py --all --start-from G5_8

# Custom lambda (must match dashboard slider for consistent spatial display)
python tools/pixel_phenology_extract.py --region G5_14 --lambda 100

# Raise minimum observations threshold
python tools/pixel_phenology_extract.py --region G5_1 --min-obs 30

# Overwrite an existing output file
python tools/pixel_phenology_extract.py --region G5_1 --overwrite

# Serial execution (for debugging or single-CPU environments)
python tools/pixel_phenology_extract.py --region G5_14 --workers 1
```

### 7.1 Lambda Consistency

When using these metric files in conjunction with the BioSCape Phenology
Explorer dashboard, the `--lambda` value used for extraction **must match**
the lambda slider value in the dashboard for spatial basemap display to be
consistent with the per-pixel time series panels.  The default (λ = 500) is
the dashboard default.  The `lambda_value` global attribute in the output
NetCDF4 file records the value used at extraction time.

---

## 8. Quality Considerations for Downstream Modelling

### 8.1 Observation Density

The 2016–2019 BioSCape HLS time series provides approximately 250–400 valid
observations per pixel under typical cloud cover conditions.  Pixels with fewer
than 20 valid observations (the `min_valid_obs` threshold) are masked in all
output variables.  Sparsely observed pixels near the edges of flight boxes or
in persistently cloudy areas may still have valid metrics but with higher
uncertainty due to fewer years contributing to the multi-year aggregates.

### 8.2 Number of Contributing Years

All metrics that report interannual statistics (means, standard deviations,
range) are computed across at most 3–4 calendar years for most regions (2016,
2017, 2018, and partial 2019).  Some edge regions with late start or early end
dates may have fewer contributing years.  The per-pixel year count is not
written to the output file; it can be inferred as:
`n_years ≈ integrated_ndvi_std is finite AND ≥2 years contributed`.

### 8.3 Bimodality Metrics and Unimodal Pixels

`peak_separation_mean`, `relative_peak_amplitude_mean`, and `valley_depth_mean`
are `NaN` (fill value) for pixels where the annual smoothed curve does not
exhibit two distinct peaks meeting the prominence and distance thresholds.
For modelling purposes these NaN values should be treated as "feature absent"
rather than as missing data — they identify unimodal pixels, which are the
majority in most LVIS regions.

### 8.4 Green-up Rate and Bimodal Pixels

`greenup_rate_mean` is only computed for years where the annual floor (minimum)
precedes the annual peak (maximum) in calendar time.  In bimodal or inverted-
seasonality pixels, this condition may fail for some or all years, resulting in
NaN.  The metric is most reliable for pixels with a single, clearly defined
growing season.

### 8.5 Smoothing Lambda and Metric Sensitivity

The choice of λ affects all smoothed-curve-derived metrics.  Higher λ values
produce more gradual seasonal envelopes:

- `peak_ndvi_mean` decreases slightly (peak is smoothed down)
- `integrated_ndvi_mean` is relatively stable (area under smooth curve is
  conservative)
- `season_length_mean` increases (smoother curve spends more time above
  threshold)
- `n_peaks_mean` decreases (small secondary peaks are smoothed out)
- `greenup_rate_mean` decreases (slower ascent in smoothed curve)

If comparing metrics generated with different λ values, or combining metric
files from different extraction runs, verify that the `lambda_value` global
attribute is the same across all files.

### 8.6 CRS and Coordinate Alignment

All spatial coordinates are in **WGS 84 / UTM Zone 34S (EPSG:32734)**.  The
`y` and `x` arrays in each output file are copied directly from the source
datacube's coordinate arrays.  Spatial joins and raster overlays with other
datasets should reproject to a common CRS before sampling or extracting values.

---

## 9. Software and References

### 9.1 Software

| Package | Role |
|---|---|
| Python 3.11 | Runtime |
| netCDF4 | HDF5 hyperslab reads; output file writing |
| numpy | Array operations |
| scipy (sparse, signal) | Whittaker solver; peak detection |
| pyproj | CRS introspection (WKT, CF parameters) |
| xarray | Time coordinate access |
| tqdm (optional) | Progress reporting |

### 9.2 Method References

**Whittaker smoothing:**
Eilers, P.H.C. (2003). A perfect smoother. *Analytical Chemistry*, 75(14), 3631–3636.
https://doi.org/10.1021/ac034173t

**Input data (HLS):**
Claverie, M. et al. (2018). The Harmonized Landsat and Sentinel-2 surface
reflectance data set. *Remote Sensing of Environment*, 219, 145–161.
https://doi.org/10.1016/j.rse.2018.09.002

**BioSCape campaign:**
https://www.bioscape.io
