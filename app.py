"""
app.py — VI Phenology Dashboard
Shiny for Python, Core API.

Run with:
    shiny run app.py
    shiny run app.py --reload   (development mode)

Environment variables:
    VI_DATACUBE_ROOT   — override the root data directory
    VI_PHENOLOGY_SRC   — override path to VI_Phenology/src

Architecture notes
------------------
- go.FigureWidget (not go.Figure) is used for all Plotly outputs so that
  Python-side on_click() callbacks work via shinywidgets.
- reactive.isolate() is required inside the on_click callback to prevent
  accidental reactive dependency creation.
- The full datacube array is NEVER loaded — see modules/datacube_io.py.
- dataset_date_cache is computed once per region from the time coordinate.
"""

from __future__ import annotations

import numpy as np
import threading
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget

from config import (
    ALL_19_METRICS,
    BASEMAP_MAX_DIM,
    DEFAULT_VI_VAR,
    FAST_BASEMAP_METRICS,
    LAMBDA_DEFAULT,
    LAMBDA_MAX,
    LAMBDA_MIN,
    LAMBDA_STEP,
    METRIC_GROUPS,
    METRIC_LABELS,
    PIXEL_METRIC_CONFIG,
    SHAPEFILE_LABEL_FIELDS,
    SHAPEFILE_PATHS,
    VI_VALID_RANGE,
)
from modules.datacube_io import (
    PixelTimeSeries,
    RegionPaths,
    basemap_cache_path,
    build_date_cache,
    build_date_cache_from_dates,
    click_to_array_index,
    compute_basemap_metric,
    detect_crs_epsg,
    discover_regions,
    extract_pixel_timeseries,
    get_dataset,
    load_basemap_cache,
    load_metrics_for_basemap,
    save_basemap_cache,
    utm_to_latlon,
)
from modules.phenology_metrics import (
    compute_pixel_with_annual,
    source_available,
    source_error,
)
from modules.visualization import (
    LEAFLET_TILE_SERVICES,
    make_annual_cycle_figure,
    make_colorbar_html,
    make_empty_timeseries_figure,
    make_leaflet_map,
    make_metrics_annual_figure,
    make_metrics_table,
    make_phenology_scatter_figure,
    make_timeseries_figure,
    update_leaflet_map,
)

# Tile service choices for the basemap dropdown (defined in visualization.py).
_TILE_CHOICES: dict[str, str] = {k: v["label"] for k, v in LEAFLET_TILE_SERVICES.items()}

# Shapefile overlay names for the checkbox UI.
# Keys are string indices ("0", "1", ...), values are file stems.
from pathlib import Path as _P
_SHAPEFILE_CHOICES: dict[str, str] = (
    {str(i): _P(p).stem for i, p in enumerate(SHAPEFILE_PATHS.split())}
    if SHAPEFILE_PATHS else {}
)


# ---------------------------------------------------------------------------
# Startup: discover regions (fast directory scan)
# ---------------------------------------------------------------------------

_STARTUP_ERROR: str | None = None
try:
    ALL_REGIONS: dict[str, RegionPaths] = discover_regions()
    # Dict choices: key = region_id (returned by input.region()),
    # value = display label with data-source tag.
    # Both nc and zarr present → "[zarr]" (zarr is always preferred by get_dataset).
    # Only nc present           → "[nc]".
    # Precomputed metrics file  → "+ metrics" appended.
    def _region_label(p: RegionPaths) -> str:
        fmt = "zarr" if p.zarr_path else "nc"
        extra = " + metrics" if p.metrics_path else ""
        return f"{p.region_id}  [{fmt}{extra}]"
    REGION_CHOICES: dict[str, str] = {
        rid: _region_label(paths) for rid, paths in ALL_REGIONS.items()
    }
except FileNotFoundError as _e:
    ALL_REGIONS = {}
    REGION_CHOICES = {}
    _STARTUP_ERROR = str(_e)

# Basemap metric dropdown choices — values are internal metric keys.
# Groups: "Quick metrics" (always available on-the-fly) +
#         "Phenology metrics" (requires precomputed pixel_metrics.nc).
_BASEMAP_CHOICES: dict[str, dict[str, str]] = {
    "── Quick metrics ──": {v: k for k, v in FAST_BASEMAP_METRICS.items()},
    "── Phenology metrics (precomputed .nc required) ──": {
        key: METRIC_LABELS[key][0]
        for key in ALL_19_METRICS
        if key in METRIC_LABELS
    },
}
_DEFAULT_BASEMAP_KEY: str = FAST_BASEMAP_METRICS["Mean VI"]  # "mean_ndvi"


# ---------------------------------------------------------------------------
# Background pre-warm: compute + cache basemap for the default region
# ---------------------------------------------------------------------------

def _prewarm_default_basemap() -> None:
    """
    Called once in a background daemon thread at app startup.
    Silently computes and caches the basemap for all 4 fast metrics on the
    first (default) region.  By the time a user opens the app the cache is
    already warm, so the first region loads instantly.

    Errors are swallowed — this is a best-effort optimisation only.
    """
    if not ALL_REGIONS:
        return
    default_paths = next(iter(ALL_REGIONS.values()))
    for metric in FAST_BASEMAP_METRICS.values():
        cache = basemap_cache_path(default_paths.nc_path, metric, BASEMAP_MAX_DIM)
        if cache.exists():
            continue
        try:
            ds = get_dataset(default_paths)
            result = compute_basemap_metric(ds, metric, vi_var=default_paths.vi_var)
            save_basemap_cache(cache, *result)
        except Exception:
            pass


if ALL_REGIONS:
    threading.Thread(target=_prewarm_default_basemap, daemon=True).start()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_source_warning = "" if source_available() else (
    ui.div(
        ui.tags.b("⚠ VI_Phenology source not found."),
        ui.tags.br(),
        "Per-pixel metrics will be unavailable. "
        f"Error: {source_error()}",
        style="color:#cc4400;font-size:0.82em;padding:6px;",
    )
)

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("VI Phenology Dashboard", style="margin-bottom:4px"),
        ui.tags.hr(style="margin:4px 0"),

        # --- Region selector ---
        ui.input_select(
            id="region",
            label="Region",
            choices=REGION_CHOICES,
            selected=next(iter(REGION_CHOICES), None),
        ),

        # --- Basemap metric ---
        ui.input_select(
            id="basemap_metric_label",
            label="Basemap metric",
            choices=_BASEMAP_CHOICES,
            selected=_DEFAULT_BASEMAP_KEY,
        ),

        # --- Layer display controls ---
        ui.input_select(
            id="basemap_type",
            label="Basemap style",
            choices=_TILE_CHOICES,
            selected="World_Imagery",
        ),
        ui.input_slider(
            id="metric_opacity",
            label="Metric layer opacity",
            min=0.0, max=1.0, value=0.75, step=0.05,
            ticks=False,
        ),
        ui.input_select(
            id="colorscale_range",
            label="Data range",
            choices={
                "full": "Full range (min – max)",
                "3sd":  "Mean ± 3 SD  (~99.7 %)",
                "2sd":  "Mean ± 2 SD  (~95 %)",
                "1sd":  "Mean ± 1 SD  (~68 %)",
            },
            selected="3sd",
        ),
        ui.output_ui("year_range_slider"),
        ui.output_ui("shapefile_toggles"),
        ui.output_ui("colorbar_panel"),

        # --- Lambda slider ---
        ui.input_slider(
            id="lambda_val",
            label="Whittaker λ (smoothing)",
            min=LAMBDA_MIN,
            max=LAMBDA_MAX,
            value=LAMBDA_DEFAULT,
            step=LAMBDA_STEP,
            ticks=False,
        ),

        ui.tags.hr(style="margin:8px 0"),

        # --- Selected pixel info ---
        ui.output_ui("selected_pixel_info"),

        ui.tags.hr(style="margin:8px 0"),

        # --- Metrics table ---
        ui.h6("Pixel Phenology Metrics", style="margin-bottom:2px"),
        ui.output_ui("pixel_stats_panel"),

        _source_warning,

        width=420,
        bg="#f4f7f4",
    ),

    # ---------------------------------------------------------------------------
    # Global CSS: resizable map card + drag-handle indicator
    # ---------------------------------------------------------------------------
    ui.tags.style("""
        /* Map card: user-resizable by dragging the bottom edge */
        .vipd-map-card {
            resize: vertical;
            overflow: auto !important;
            min-height: 280px;
        }
        /* Resize hint — right-aligned to match the native corner gripper */
        .vipd-map-card::after {
            content: "drag corner to resize ↘";
            display: block;
            text-align: right;
            padding-right: 20px;
            height: 16px;
            line-height: 16px;
            font-size: 10px;
            color: #9aaa9a;
            background: #eef3ee;
            border-top: 1px solid #ccdacc;
            user-select: none;
        }
        /* Card bodies fill available height */
        .vipd-map-card .card-body,
        .vipd-ts-card  .card-body {
            padding: 4px;
            height: calc(100% - 40px);
        }
        /* Time-series Plotly widget fills card */
        .vipd-ts-card .card-body > * {
            height: 100% !important;
            width:  100% !important;
        }
        /* ipyleaflet map: make every wrapper div fill the card body */
        .vipd-map-card .card-body,
        .vipd-map-card .card-body > div,
        .vipd-map-card .card-body .widget-output-container,
        .vipd-map-card .card-body .shiny-html-output,
        .vipd-map-card .card-body .jupyter-widget-output-area {
            height: 100% !important;
            width:  100% !important;
        }
        /* Leaflet container itself */
        .vipd-map-card .leaflet-container {
            height: 100% !important;
            width:  100% !important;
        }
        /* Render the metric overlay with hard pixel edges (no interpolation) */
        .leaflet-image-layer {
            image-rendering: pixelated;
            image-rendering: crisp-edges;
        }
    """),

    # ---------------------------------------------------------------------------
    # ResizeObserver: trigger Plotly autosize whenever the map card is resized
    # ---------------------------------------------------------------------------
    ui.tags.script("""
        (function () {
            function attachObserver() {
                var cards = document.querySelectorAll('.vipd-map-card');
                if (!cards.length || !window.Plotly) {
                    setTimeout(attachObserver, 500);
                    return;
                }
                var ro = new ResizeObserver(function (entries) {
                    entries.forEach(function (entry) {
                        entry.target
                            .querySelectorAll('.js-plotly-plot')
                            .forEach(function (g) {
                                Plotly.relayout(g, {autosize: true});
                            });
                    });
                });
                cards.forEach(function (el) { ro.observe(el); });
            }
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', attachObserver);
            } else {
                attachObserver();
            }
        })();
    """),

    # --- Main panel: map above (resizable), time series below ---
    ui.card(
        ui.card_header("Spatial map — click to select pixel"),
        output_widget("basemap_widget"),
        full_screen=True,
        fill=True,
        class_="vipd-map-card",
        style="height: 62vh;",
    ),
    ui.navset_card_tab(
        ui.nav_panel(
            "Raw VI",
            output_widget("timeseries_widget"),
        ),
        ui.nav_panel(
            "Annual Cycles",
            output_widget("annual_cycle_widget"),
        ),
        ui.nav_panel(
            "Metric Trends",
            ui.tags.div(
                output_widget("metrics_annual_widget"),
                style="height:100%;overflow-y:auto;",
            ),
        ),
        ui.nav_panel(
            "Phenology Scatter",
            output_widget("phenology_scatter_widget"),
        ),
        id="ts_tabs",
        selected="Raw VI",
        full_screen=True,
    ),

    title="VI Phenology Dashboard",
    fillable=True,
)

if _STARTUP_ERROR:
    app_ui = ui.page_fluid(
        ui.card(
            ui.card_header("⚠ Data directory not found"),
            ui.p(_STARTUP_ERROR),
            ui.p(
                "Set the ",
                ui.tags.code("VI_DATACUBE_ROOT"),
                " environment variable, or edit ",
                ui.tags.code("config.py"),
                ".",
            ),
        )
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):

    # -----------------------------------------------------------------------
    # Reactive state for pixel selection
    # -----------------------------------------------------------------------

    selected_idx:    reactive.Value[tuple[str, int, int] | None] = reactive.Value(None)
    selected_coords: reactive.Value[tuple[float, float] | None] = reactive.Value(None)

    # Non-reactive per-session reference to the current basemap FigureWidget.
    # Set by basemap_widget on (re-)render; read by _update_basemap_inplace.
    _fig_ref: list = [None]

    # -----------------------------------------------------------------------
    # Reactive calculations — cached, invalidated only when deps change
    # -----------------------------------------------------------------------

    @reactive.Calc
    def region_paths() -> RegionPaths:
        return ALL_REGIONS[input.region()]

    @reactive.Calc
    def active_dataset():
        """
        Lazily open the datacube for the selected region.
        File handle is lru_cached in datacube_io, so reopening is free.
        """
        return get_dataset(region_paths())

    @reactive.Calc
    def dataset_date_cache() -> dict:
        """
        Build date_cache from the dataset's time coordinate.
        Cheap (reads ~5 KB of time values), but cached per region.
        """
        return build_date_cache(active_dataset())

    @reactive.Calc
    def native_pixel_step() -> tuple[float, float]:
        """
        Native pixel footprint in WGS84 degrees: (lat_step, lon_step).
        Reads only the x/y coordinate arrays — never the data variable.
        Used to size the selection rectangle to exactly one 30 m pixel.
        """
        ds   = active_dataset()
        x_v  = ds["x"].values
        y_v  = ds["y"].values
        dx_m = abs(float(x_v[1] - x_v[0]))   # metres per native pixel (x)
        dy_m = abs(float(y_v[1] - y_v[0]))   # metres per native pixel (y)
        x_mid = float((x_v.min() + x_v.max()) / 2)
        y_mid = float((y_v.min() + y_v.max()) / 2)
        _, lat_mid = utm_to_latlon(
            np.array([x_mid]), np.array([y_mid]),
            detect_crs_epsg(ds),
        )
        lat_deg = float(lat_mid[0])
        return (
            dy_m / 111_320.0,                                          # lat_step
            dx_m / (111_320.0 * np.cos(np.radians(lat_deg))),         # lon_step
        )

    @reactive.Calc
    def basemap_metric_key() -> str:
        """Return the selected internal metric key (dropdown value is already the key)."""
        return input.basemap_metric_label()

    @reactive.Calc
    def basemap_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute (z, lon, lat) for the basemap.
        Triggered by: region change OR basemap_metric change.

        Load order (fastest first):
          1. Precomputed pixel_metrics.nc (phenology metrics only)
          2. Disk cache (.npz alongside the NC file) — milliseconds
          3. Dask on-the-fly compute — seconds to tens of seconds

        On-the-fly results are written to the disk cache so subsequent
        loads (new tabs, app restarts) skip the Dask step entirely.
        A progress bar is shown in the browser while Dask computes.
        """
        paths  = region_paths()
        metric = basemap_metric_key()
        ds     = active_dataset()

        # Fast path 1: precomputed metrics file + phenology metric requested
        if paths.metrics_path is not None and metric in ALL_19_METRICS:
            try:
                return load_metrics_for_basemap(paths.metrics_path, metric)
            except (KeyError, Exception):
                pass  # fall through

        # Resolve the actual metric key to compute (fall back for phenology w/o file)
        _fast_keys = set(FAST_BASEMAP_METRICS.values())
        if metric not in _fast_keys:
            ui.notification_show(
                f"No precomputed pixel_metrics.nc found for region "
                f"'{input.region()}'. Showing Peak {paths.vi_var} instead. "
                "Run tools/pixel_phenology_extract.py to generate pixel_metrics.nc.",
                type="warning",
                duration=10,
            )
            metric = "peak_ndvi_mean"

        # Fast path 2: disk cache
        cache = basemap_cache_path(paths.nc_path, metric, BASEMAP_MAX_DIM)
        hit = load_basemap_cache(cache)
        if hit is not None:
            return hit

        # Slow path: Dask compute with progress indicator
        with ui.Progress(min=0, max=3) as p:
            p.set(1, message=f"Computing basemap…",
                  detail=f"{paths.region_id} · {metric}")
            result = compute_basemap_metric(ds, metric, vi_var=paths.vi_var)
            p.set(2, detail="Saving cache…")
            save_basemap_cache(cache, *result)
            p.set(3)
        return result

    @reactive.Calc
    def colorscale_limits() -> tuple[float | None, float | None]:
        """
        Compute (zmin, zmax) for the heatmap colorbar based on the SD-clipping
        selection.  Depends only on basemap_data + the dropdown so it is
        recomputed when the metric or region changes, but NOT on opacity/
        satellite inputs.
        """
        z, _, _ = basemap_data()
        sel = input.colorscale_range()
        if sel == "full":
            return None, None
        z_mean = float(np.nanmean(z))
        z_std  = float(np.nanstd(z))
        n_sd   = {"1sd": 1, "2sd": 2, "3sd": 3}.get(sel, 2)
        return z_mean - n_sd * z_std, z_mean + n_sd * z_std

    @reactive.Calc
    def effective_selected_idx() -> tuple[int, int] | None:
        """
        Return (yi, xi) for the selected pixel IFF it belongs to the current
        region and is within bounds; otherwise None.

        Reading input.region() here means this calc invalidates in the SAME
        reactive flush as region_paths() when the region changes.  No separate
        _reset_selection effect is needed to clear selected_idx, so
        pixel_timeseries (and pixel_stats_panel downstream) receive exactly one
        invalidation per region switch instead of two.
        """
        idx = selected_idx()
        if idx is None:
            return None
        stored_region, yi, xi = idx
        if stored_region != input.region():
            return None
        ds = active_dataset()
        ny, nx = ds.sizes["y"], ds.sizes["x"]
        if yi >= ny or xi >= nx:
            return None
        return (yi, xi)

    @reactive.Calc
    def pixel_timeseries() -> PixelTimeSeries | None:
        """
        Extract raw time series for the selected pixel via HDF5 hyperslab.
        Triggered by: pixel click OR region change.
        """
        idx = effective_selected_idx()
        if idx is None:
            return None
        yi, xi = idx
        paths = region_paths()
        try:
            return extract_pixel_timeseries(
                paths.nc_path, yi, xi,
                vi_var=paths.vi_var,
                zarr_path=paths.zarr_path,
            )
        except Exception as exc:
            ui.notification_show(f"Pixel read error: {exc}", type="error", duration=6)
            return None

    @reactive.Calc
    def narrowed_timeseries() -> PixelTimeSeries | None:
        """
        PixelTimeSeries with valid_mask restricted to observations that fall
        within the active data range (colorscale_limits) and year range
        (year_range_slider).  All smoothing and metric computation uses this
        rather than the full physical series, so that both filter controls
        genuinely affect the analysis.
        """
        ts = pixel_timeseries()
        if ts is None:
            return None
        zmin, zmax = colorscale_limits()
        mask = ts.valid_mask.copy()
        if zmin is not None:
            mask &= ts.raw_vi >= float(zmin)
        if zmax is not None:
            mask &= ts.raw_vi <= float(zmax)
        try:
            yr_start, yr_end = input.year_range()
            obs_years = ts.dates.astype("datetime64[Y]").astype(int) + 1970
            year_keep = (obs_years >= yr_start) & (obs_years <= yr_end)
            mask &= year_keep
            # Clip the arrays to the year range so that the Whittaker daily
            # grid (built from ts.dates[0]…ts.dates[-1]) only spans the
            # selected years — preventing extrapolation outside them.
            return PixelTimeSeries(
                dates=ts.dates[year_keep],
                raw_vi=ts.raw_vi[year_keep],
                valid_mask=mask[year_keep],
                x_coord=ts.x_coord,
                y_coord=ts.y_coord,
                lon=ts.lon,
                lat=ts.lat,
            )
        except Exception:
            pass  # slider not yet rendered (dynamic UI not initialized)
        return PixelTimeSeries(
            dates=ts.dates,
            raw_vi=ts.raw_vi,
            valid_mask=mask,
            x_coord=ts.x_coord,
            y_coord=ts.y_coord,
            lon=ts.lon,
            lat=ts.lat,
        )

    @reactive.Calc
    def smoothed_result() -> tuple[np.ndarray, np.ndarray] | None:
        """
        Whittaker-smoothed daily time series + daily date array.
        Derived from pixel_annual_data so the Whittaker solve runs only once
        per (pixel, lambda, data-range) combination.
        """
        result = pixel_annual_data()
        if result is None:
            return None
        _metrics, _valid_years, _annual_data, smoothed_daily, daily_dates = result
        return smoothed_daily, daily_dates

    @reactive.Calc
    def pixel_metric_config() -> dict:
        """
        Per-region metric config with vi_min/vi_max clamped to both the
        physical VI valid range and the active data range (colorscale_limits).
        Ensures observation masking and curve clipping inside the Whittaker
        smoother match the data range filter applied to the time series.
        """
        vi = region_paths().vi_var
        vi_min, vi_max = VI_VALID_RANGE.get(vi, VI_VALID_RANGE[DEFAULT_VI_VAR])
        zmin, zmax = colorscale_limits()
        if zmin is not None:
            vi_min = max(vi_min, float(zmin))
        if zmax is not None:
            vi_max = min(vi_max, float(zmax))
        return {**PIXEL_METRIC_CONFIG, "vi_min": vi_min, "vi_max": vi_max}

    @reactive.Calc
    def pixel_date_cache() -> dict:
        """
        Date cache scoped to the active (year-filtered) pixel time series.
        When year filtering clips ts.dates, this cache covers only those dates,
        so the Whittaker daily grid — and thus smoothing — is bounded to the
        selected range with no extrapolation outside it.
        Falls back to the full region date cache when no pixel is selected.
        """
        ts = narrowed_timeseries()
        if ts is None or len(ts.dates) < 2:
            return dataset_date_cache()
        return build_date_cache_from_dates(ts.dates)

    @reactive.Calc
    def pixel_annual_data():
        """
        Full annual breakdown: (metrics, valid_years, annual_data).
        Triggered by: narrowed_timeseries, lambda_val, or data range change.
        """
        ts = narrowed_timeseries()
        if ts is None:
            return None
        dc = pixel_date_cache()
        return compute_pixel_with_annual(
            ts, dc, float(input.lambda_val()), pixel_metric_config()
        )

    @reactive.Calc
    def pixel_metrics() -> dict[str, float] | None:
        """
        All 19 aggregated phenological metrics for the selected pixel.
        Derived from pixel_annual_data so both share the same computation.
        """
        result = pixel_annual_data()
        if result is None:
            return None
        metrics, _valid_years, _annual_data, _smoothed, _dates = result
        return metrics

    # -----------------------------------------------------------------------
    # Basemap widget
    # -----------------------------------------------------------------------

    @render_widget
    def basemap_widget():
        """
        Build the ipyleaflet Map and register the click callback.

        Re-creates the map ONLY when input.region changes (the sole reactive
        dependency outside the isolate block).  All other display changes
        (metric, opacity, color scale, tile style, pixel selection) are
        handled in-place by _update_basemap_inplace — zoom/pan are preserved.
        """
        region = input.region()   # sole reactive dependency

        with reactive.isolate():
            z, lon, lat = basemap_data()
            metric_key  = basemap_metric_key()
            zmin, zmax  = colorscale_limits()
            tile_svc    = input.basemap_type()
            opacity     = float(input.metric_opacity())
            coords      = selected_coords()
            nat_lat_step, nat_lon_step = native_pixel_step()

        m = make_leaflet_map(
            z=z,
            lon=lon,
            lat=lat,
            metric_key=metric_key,
            tile_service=tile_svc,
            metric_opacity=opacity,
            zmin=zmin,
            zmax=zmax,
            native_lat_step=nat_lat_step,
            native_lon_step=nat_lon_step,
            shapefile_paths=SHAPEFILE_PATHS,
            shapefile_label_fields=SHAPEFILE_LABEL_FIELDS,
        )

        # Show pixel marker if a pixel is already selected (e.g. region switch)
        if coords is not None:
            update_leaflet_map(m, z, lat, metric_key, selected_pixel=coords)

        def _on_interaction(**kwargs):
            if kwargs.get("type") != "click":
                return
            coordinates = kwargs.get("coordinates")
            if not coordinates:
                return
            click_lat = float(coordinates[0])
            click_lon = float(coordinates[1])
            with reactive.isolate():
                ds = active_dataset()
                _z, lon, lat = basemap_data()
            # Ignore clicks outside the current region's data extent
            # (e.g. clicks on shapefile polygons that extend beyond the datacube).
            if not (float(lon.min()) <= click_lon <= float(lon.max())
                    and float(lat.min()) <= click_lat <= float(lat.max())):
                return
            yi, xi = click_to_array_index(click_lon, click_lat, ds)
            selected_idx.set((region, yi, xi))
            selected_coords.set((click_lon, click_lat))

        m.on_interaction(_on_interaction)
        m._region_id = region   # used by guard in _update_basemap_inplace
        _fig_ref[0]  = m
        return m

    @reactive.Effect
    @reactive.event(
        input.basemap_metric_label,
        input.colorscale_range,
        input.metric_opacity,
        input.basemap_type,
        selected_coords,
    )
    def _update_basemap_inplace():
        """
        Update the existing ipyleaflet Map in place — only changed ipywidgets
        properties are sent to the browser, so zoom/pan state is preserved.

        Triggered by: metric, color-scale range, opacity, tile style, or pixel
        selection.  NOT triggered by region change (handled by full re-render).
        """
        m = _fig_ref[0]
        if m is None:
            return

        # Guard: skip if the stored map belongs to a stale region (e.g. when
        # selected_coords reset fires before basemap_widget creates new map).
        with reactive.isolate():
            current_region = input.region()
        if getattr(m, "_region_id", None) != current_region:
            return

        z, _lon, lat = basemap_data()
        metric_key   = basemap_metric_key()
        zmin, zmax   = colorscale_limits()
        tile_svc     = input.basemap_type()
        opacity      = float(input.metric_opacity())
        coords       = selected_coords()

        update_leaflet_map(
            m=m,
            z=z,
            lat=lat,
            metric_key=metric_key,
            tile_service=tile_svc,
            metric_opacity=opacity,
            zmin=zmin,
            zmax=zmax,
            selected_pixel=coords,
        )

    # -----------------------------------------------------------------------
    # Year range slider (dynamic — bounds derived from the active dataset)
    # -----------------------------------------------------------------------

    @render.ui
    def year_range_slider():
        years = dataset_date_cache()["years"]
        yr_min, yr_max = int(years.min()), int(years.max())
        return ui.input_slider(
            id="year_range",
            label="Year range",
            min=yr_min,
            max=yr_max,
            value=[yr_min, yr_max],
            step=1,
            sep="",
            ticks=False,
        )

    # -----------------------------------------------------------------------
    # Shapefile overlay toggles
    # -----------------------------------------------------------------------

    @render.ui
    def shapefile_toggles():
        if not _SHAPEFILE_CHOICES:
            return ui.div()
        return ui.div(
            ui.tags.label("Overlay layers", class_="form-label",
                          style="margin-bottom:4px"),
            ui.input_checkbox_group(
                id="shapefile_visible",
                label=None,
                choices=_SHAPEFILE_CHOICES,
                selected=list(_SHAPEFILE_CHOICES.keys()),
            ),
            style="margin-top:4px",
        )

    if _SHAPEFILE_CHOICES:
        @reactive.Effect
        @reactive.event(input.shapefile_visible)
        def _toggle_shapefile_layers():
            m = _fig_ref[0]
            if m is None or not hasattr(m, "_shapefile_layers"):
                return
            visible_set = set(input.shapefile_visible())
            for i, layer_dict in enumerate(m._shapefile_layers):
                show = str(i) in visible_set
                layer_dict["geojson"].visible = show
                for marker in layer_dict["labels"]:
                    marker.visible = show

    # -----------------------------------------------------------------------
    # Time series / annual-cycle / metric-trend widgets
    # -----------------------------------------------------------------------

    @render_widget
    def timeseries_widget():
        ts = narrowed_timeseries()
        if ts is None:
            return make_empty_timeseries_figure()

        result = smoothed_result()
        if result is None:
            return make_empty_timeseries_figure()

        smoothed_daily, daily_dates = result
        zmin, zmax = colorscale_limits()
        return make_timeseries_figure(
            ts=ts,
            smoothed_daily=smoothed_daily,
            daily_dates=daily_dates,
            region_id=input.region(),
            vi_var=region_paths().vi_var,
            basemap_metric=basemap_metric_key(),
            zmin=zmin,
            zmax=zmax,
        )

    @render_widget
    def annual_cycle_widget():
        result = smoothed_result()
        if result is None:
            return make_empty_timeseries_figure()

        smoothed_daily, daily_dates = result
        zmin, zmax = colorscale_limits()
        return make_annual_cycle_figure(
            smoothed=smoothed_daily,
            daily_dates=daily_dates,
            region_id=input.region(),
            vi_var=region_paths().vi_var,
            basemap_metric=basemap_metric_key(),
            zmin=zmin,
            zmax=zmax,
        )

    @render_widget
    def phenology_scatter_widget():
        ts = narrowed_timeseries()
        if ts is None:
            return make_empty_timeseries_figure()
        zmin, zmax = colorscale_limits()
        return make_phenology_scatter_figure(
            ts=ts,
            vi_var=region_paths().vi_var,
            lam=float(input.lambda_val()),
            region_id=input.region(),
            basemap_metric=basemap_metric_key(),
            zmin=zmin,
            zmax=zmax,
        )

    @render_widget
    def metrics_annual_widget():
        result = pixel_annual_data()
        if result is None:
            return make_empty_timeseries_figure()

        metrics, valid_years, annual_data, _smoothed, _dates = result
        if not valid_years:
            return make_empty_timeseries_figure()

        return make_metrics_annual_figure(
            valid_years=valid_years,
            annual_data=annual_data,
            metrics=metrics,
            region_id=input.region(),
        )

    # -----------------------------------------------------------------------
    # Sidebar outputs
    # -----------------------------------------------------------------------

    @render.ui
    def selected_pixel_info():
        coords = selected_coords()
        idx    = selected_idx()
        if coords is None or idx is None:
            return ui.p(
                "No pixel selected. Click on the map.",
                style="color:#999;font-size:0.82em;font-style:italic",
            )
        lon, lat        = coords
        _, yi, xi       = idx

        # Pull observation stats from both full and narrowed time series
        ts_full = pixel_timeseries()
        ts_nr   = narrowed_timeseries()
        if ts_full is not None and ts_nr is not None:
            n_total   = len(ts_full.valid_mask)
            n_valid   = int(ts_full.valid_mask.sum())
            n_in_range = int(ts_nr.valid_mask.sum())
            pct_valid = 100.0 * n_valid   / n_total if n_total  > 0 else 0.0
            pct_range = 100.0 * n_in_range / n_valid if n_valid > 0 else 0.0
            valid_dates = ts_nr.dates[ts_nr.valid_mask]
            date_first  = str(valid_dates.min())[:10] if n_in_range > 0 else "—"
            date_last   = str(valid_dates.max())[:10] if n_in_range > 0 else "—"
            vi_rng = ts_nr.raw_vi[ts_nr.valid_mask]
            vi_lo  = float(vi_rng.min()) if n_in_range > 0 else float("nan")
            vi_hi  = float(vi_rng.max()) if n_in_range > 0 else float("nan")

            range_line = (
                ui.tags.b("In range: "),
                f"{n_in_range} / {n_valid} valid  ({pct_range:.1f}%)",
                ui.tags.br(),
            ) if n_in_range != n_valid else ()

            obs_block = ui.tags.small(
                ui.tags.b("Valid obs: "),
                f"{n_valid} / {n_total}  ({pct_valid:.1f}%)",
                ui.tags.br(),
                *range_line,
                ui.tags.b("Date range: "),
                f"{date_first} → {date_last}",
                ui.tags.br(),
                ui.tags.b(f"{region_paths().vi_var} range: "),
                f"{vi_lo:.3f} – {vi_hi:.3f}",
                style="color:#444",
            )
        else:
            obs_block = ui.tags.small(
                "Loading observations…",
                style="color:#999;font-style:italic",
            )

        return ui.div(
            ui.tags.b("Selected pixel"),
            ui.tags.br(),
            ui.tags.small(
                f"Lat {lat:.4f}°, Lon {lon:.4f}°",
                ui.tags.br(),
                f"Array index: y={yi}, x={xi}",
                style="color:#444",
            ),
            ui.tags.br(),
            obs_block,
        )

    @render.ui
    def colorbar_panel():
        """Horizontal colorbar for the current metric and color-scale range."""
        z, _, _ = basemap_data()
        metric_key = basemap_metric_key()
        zmin, zmax = colorscale_limits()
        valid = z[~np.isnan(z)]
        if len(valid) == 0:
            return ui.p("No valid data", style="color:#999;font-size:0.8em")
        cb_min = float(valid.min()) if zmin is None else zmin
        cb_max = float(valid.max()) if zmax is None else zmax
        return ui.HTML(make_colorbar_html(metric_key, cb_min, cb_max))

    @render.ui
    def pixel_stats_panel():
        metrics = pixel_metrics()
        if metrics is None:
            return ui.p(
                "Select a pixel to compute metrics.",
                style="color:#999;font-size:0.82em;font-style:italic",
            )
        zmin, zmax = colorscale_limits()
        table_html = make_metrics_table(
            metrics,
            selected_metric=basemap_metric_key(),
            zmin=zmin,
            zmax=zmax,
        )
        return ui.HTML(table_html)

    # -----------------------------------------------------------------------
    # Reset pixel selection on region change
    # -----------------------------------------------------------------------

    @reactive.Effect
    @reactive.event(input.region)
    def _reset_selection():
        # Clear the map marker only. selected_idx is NOT reset here because
        # effective_selected_idx() handles region mismatch atomically in the
        # same reactive flush as the region change, avoiding the double-
        # invalidation that caused client state machine errors on pixel_stats_panel.
        selected_coords.set(None)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

app = App(app_ui, server)
