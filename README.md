# VI Phenology Dashboard

An interactive Shiny for Python dashboard for exploring vegetation index (VI)
phenology datacubes from the BioSCape / LVIS flight campaign.

---

## Overview

The dashboard reads NetCDF4 VI datacubes (one per LVIS flight box region),
displays a spatial metric map, and lets you click any pixel to instantly
compute and visualise its full Whittaker-smoothed phenology time series plus
19 per-pixel phenological metrics.

**Key features**

| Feature | Description |
|---|---|
| Region selector | Dropdown of all G5_xx LVIS flight box regions |
| Spatial basemap | Plotly heatmap of peak NDVI or other spatial metrics |
| Click-to-select | Click any pixel on the map to load its time series |
| Time series plot | Raw NDVI observations + Whittaker-smoothed curve |
| Lambda slider | Adjust smoothing (λ = 10–1000) and see results update live |
| 19-metric sidebar | All phenological metrics for the selected pixel, grouped by category |
| ZARR acceleration | Optional one-time rechunking for fast pixel reads on large files |

---

## Prerequisites

- **macOS or Linux** (Windows untested)
- **Conda / Miniconda** — [install Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **External data drive** mounted containing the datacubes
  (default: `/Volumes/ConklinGeospatialData/...`)

The Whittaker smoothing and phenology metric functions are self-contained in
`modules/phenology_metrics.py` — no external VI_Phenology source tree is required.

---

## Setup

### 1. Create the conda environment

```bash
cd /path/to/VI_Phenology_Dashboard
conda env create -f environment.yml
conda activate vi_phenology_dashboard
```

### 2. Verify the data path

The default datacube root is:
```
/Volumes/ConklinGeospatialData/Data/BioSCape_SA_LVIS/VI_Phenology/netcdf_datacube
```

Override it without editing any files:
```bash
export VI_DATACUBE_ROOT=/your/custom/path
```

Or edit `DATA_ROOT` directly in `config.py`.

### 3. Run the dashboard

```bash
shiny run app.py
# Open http://127.0.0.1:8000 in a browser

# Development mode (auto-reload on save):
shiny run app.py --reload
```

---

## Data Directory Layout

The dashboard expects this structure under `DATACUBE_ROOT`:

```
netcdf_datacube/
└── LVIS_flightboxes_final/
    ├── G5_1/
    │   ├── NDVI_G5_1_datacube.nc          ← required
    │   ├── NDVI_G5_1_datacube.zarr/       ← optional (fast pixel reads)
    │   └── NDVI_G5_1_pixel_metrics.nc     ← optional (fast basemap display)
    ├── G5_2/
    │   └── NDVI_G5_2_datacube.nc
    ...
    └── G5_25/
        └── NDVI_G5_25_datacube.nc
```

The dashboard discovers all regions automatically at startup.

---

## Dashboard Layout

```
┌─ Sidebar (310 px) ──────────┐  ┌─ Main panel ─────────────────────────────────────────┐
│  Region     [G5_1 ▼]        │  │  ┌─ Spatial map (ipyleaflet) ───────────────────────┐ │
│  Basemap    [Peak NDVI ▼]   │  │  │  Satellite basemap + metric colour overlay        │ │
│  Basemap style [Imagery ▼]  │  │  │  Click any pixel → red marker + time series load  │ │
│  Metric opacity [─●──]      │  │  └─────────────────────────────────────────────────┘ │
│  Color scale [Mean±2SD ▼]   │  │                                                       │
│  [colorbar]                 │  │  ┌─ Tabs ──────────────────────────────────────────┐  │
│  λ smoothing [────●────]    │  │  │  Raw NDVI │ Annual Cycles │ Metric Trends        │  │
│  10          500       1000 │  │  │  ─────────────────────────────────────────────  │  │
│  ─────────────────────────  │  │  │  Raw obs (grey) + Whittaker-smooth (green)       │  │
│  Selected pixel             │  │  │    OR  per-DOY curves by year                   │  │
│  Lat -33.4821°              │  │  │    OR  per-year metric scatter + mean ± std      │  │
│  Lon  19.2341°              │  │  └─────────────────────────────────────────────────┘  │
│  Valid obs: 842/1287 (65%)  │  │                                                       │
│  Date range: 2016-01 →      │  │                                                       │
│  NDVI range: 0.11 – 0.94    │  │                                                       │
│  ─────────────────────────  │  │                                                       │
│  PEAK                       │  │                                                       │
│    Peak NDVI (mean)  0.7821 │  │                                                       │
│    Peak DOY (mean)   127    │  │                                                       │
│  PRODUCTIVITY               │  │                                                       │
│    Integrated NDVI   44.2   │  │                                                       │
│    Green-up Rate     0.0042 │  │                                                       │
│  SEASONALITY ...            │  │                                                       │
│  VARIABILITY ...            │  │                                                       │
│  BIMODALITY ...             │  │                                                       │
└─────────────────────────────┘  └───────────────────────────────────────────────────────┘
```

---

## Workflow

1. **Select a region** from the dropdown — the basemap metric overlay loads (2–20 s for large files).
2. **Choose a basemap metric** to recolour the overlay (e.g., Peak NDVI, Season Length).
3. **Choose a basemap style** (satellite imagery, OpenStreetMap, etc.) and adjust **opacity**.
4. **Click any pixel** on the map — a red marker appears and the time series loads.
5. **Adjust the lambda slider** to change Whittaker smoothing — the curve and all 19 metrics update instantly.
6. The **19 phenological metrics** appear in the sidebar, grouped by category.
7. Switch between **three time-series tabs**:
   - **Raw NDVI** — raw observations + Whittaker-smoothed curve, full date range
   - **Annual Cycles** — per-DOY overlay by calendar year (seasonal shape comparison)
   - **Metric Trends** — annual scatter plots for each metric with mean ± std bands

---

## Basemap Metrics

These four "quick" metrics are always computed on-the-fly via Dask (no extra files needed):

| Dropdown label | Description |
|---|---|
| Peak NDVI | Maximum NDVI value across all observations |
| Mean NDVI | Temporal mean across all valid observations |
| Std Dev NDVI | Temporal standard deviation |
| Data Coverage | Fraction of timesteps with valid (non-NaN) observations |

If a precomputed `pixel_metrics.nc` file exists alongside the datacube (generated by
`tools/pixel_phenology_extract.py`), the full suite of 19 phenological metrics becomes
available as basemap options too.

### Basemap Display Controls

| Control | Description |
|---|---|
| Basemap style | Tile service: World Imagery (default), OpenStreetMap, Topo, Light Gray |
| Metric layer opacity | 0–1 slider controlling transparency of the colour overlay |
| Color scale range | Clip colorbar to full range, Mean ± 1/2/3 SD |

---

## Performance Notes

| File size (compressed) | Basemap load | Pixel read |
|---|---|---|
| < 100 MB (G5_16, G5_25) | < 3 s | < 50 ms |
| 500 MB – 1 GB | 5–10 s | 50–200 ms |
| 2–3 GB (G5_7, G5_10, G5_14) | 10–25 s | 200–500 ms |

For the largest files, ZARR conversion dramatically improves pixel read speed.

---

## ZARR Conversion (Recommended for Large Files)

Convert once, then the dashboard automatically uses the ZARR store:

```bash
conda activate vi_phenology_dashboard

# Convert the largest files first:
python tools/convert_to_zarr.py --region G5_14
python tools/convert_to_zarr.py --region G5_10
python tools/convert_to_zarr.py --region G5_7

# Or convert everything at once:
python tools/convert_to_zarr.py --all

# Preview what would be converted (no files written):
python tools/convert_to_zarr.py --all --dry-run
```

ZARR stores appear as `NDVI_G5_xx_datacube.zarr/` directories next to the `.nc` files.
Disk space: expect ~1–1.5× the NC file size per ZARR store.

---

## Phenology Metrics Reference

See [docs/methods.md](docs/methods.md) for full mathematical definitions.

| Group | Metric | Description |
|---|---|---|
| **Peak** | peak_ndvi_mean | Mean annual peak NDVI across years |
| | peak_ndvi_std | Interannual std of peak NDVI |
| | peak_doy_mean | Mean day-of-year of peak NDVI |
| | peak_doy_std | Interannual std of peak DOY |
| **Productivity** | integrated_ndvi_mean | Mean integrated NDVI (area under curve) |
| | integrated_ndvi_std | Interannual std of integrated NDVI |
| | greenup_rate_mean | Mean green-up slope (floor → peak) |
| | greenup_rate_std | Interannual std of green-up rate |
| **Seasonality** | floor_ndvi_mean | Mean dry-season NDVI floor |
| | ceiling_ndvi_mean | Mean wet-season NDVI ceiling |
| | season_length_mean | Mean days above phenological threshold |
| | season_length_std | Interannual std of season length |
| **Variability** | cv | Coefficient of variation (whole series) |
| | interannual_peak_range | Max − min peak NDVI across years |
| | interannual_peak_std | Std of annual peak NDVI across years |
| **Bimodality** | n_peaks_mean | Mean number of seasonal peaks per year |
| | peak_separation_mean | Mean day-gap between the two highest peaks |
| | relative_peak_amplitude_mean | Mean amplitude ratio of lower to higher peak |
| | valley_depth_mean | Mean normalised depth of inter-peak valley |

---

## Configuration

All settings are in `config.py`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATACUBE_ROOT` | (external drive path) | Root directory for datacubes |
| `DATACUBE_CRS_EPSG` | 32734 | CRS of the datacubes (UTM Zone 34S) |
| `LAMBDA_DEFAULT` | 500.0 | Default Whittaker smoothing λ |
| `BASEMAP_MAX_DIM` | 500 | Max pixels per axis for display downsampling |
| `PIXEL_METRIC_CONFIG` | (dict) | Thresholds for metric computation |

Override `DATACUBE_ROOT` without editing code:
```bash
export VI_DATACUBE_ROOT=/new/path
```

---

## Project Structure

```
VI_Phenology_Dashboard/
├── app.py                          # Shiny Core API entry point
├── config.py                       # All configuration constants
├── environment.yml                 # Conda environment specification
├── modules/
│   ├── datacube_io.py              # File discovery, lazy loading, pixel extraction
│   ├── phenology_metrics.py        # Whittaker smoothing + 19-metric computation
│   └── visualization.py           # ipyleaflet map + Plotly figure factory functions
├── tools/
│   ├── convert_to_zarr.py         # One-time ZARR rechunking utility
│   └── pixel_phenology_extract.py  # Batch per-pixel metrics → pixel_metrics.nc
└── docs/
    └── methods.md                  # Scientific methods documentation
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `shiny` | Python web framework |
| `shinywidgets` | Plotly FigureWidget integration with Shiny |
| `plotly` | Interactive heatmap and time series charts |
| `xarray` + `dask` | Lazy datacube loading and spatial computation |
| `netcdf4` | Direct HDF5 hyperslab reads for single-pixel access |
| `scipy` | Sparse linear solver for Whittaker smoothing |
| `pyproj` | UTM → WGS84 coordinate reprojection |
| `zarr` | Optional rechunked format for fast pixel access |
| `numpy` + `pandas` | Array and date operations |

---

## Troubleshooting

**"Data directory not found"** — Check that the external drive is mounted and
`DATACUBE_ROOT` in `config.py` is correct.

**Basemap metric overlay takes > 30 s** — Normal for G5_14 (2.6 GB compressed). Run ZARR
conversion to speed up future loads, or wait.

**Click on the map does nothing** — Click directly on the coloured metric overlay
(not on ocean or background with no data coverage).

**Metrics show all N/A** — The selected pixel may have fewer than 20 valid
observations (`min_valid_obs` in `PIXEL_METRIC_CONFIG`). Try a pixel in the
high-coverage area of the basemap (use "Data Coverage" as the basemap metric to find dense areas).

**Annual Cycles / Metric Trends tab is empty** — The selected pixel may have
insufficient observations per year (`min_valid_obs_per_year = 5` in `PIXEL_METRIC_CONFIG`).

**Phenology metrics not available as basemap options** — Run
`python tools/pixel_phenology_extract.py --region G5_xx` to generate
`pixel_metrics.nc` for that region first.
