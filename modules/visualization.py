"""
visualization.py — Plotly FigureWidget factory functions.

IMPORTANT: All figures use go.FigureWidget, NOT go.Figure.
Only FigureWidget supports Python-side click callbacks via on_click().
Using go.Figure silently breaks all pixel-selection interactions.
"""

from __future__ import annotations

import base64
import io
import math
from typing import NamedTuple

import ipyleaflet as ipl
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure
from PIL import Image as PILImage

import pandas as pd
from plotly.subplots import make_subplots as _make_subplots

from config import METRIC_LABELS, METRIC_GROUPS, VI_VALID_RANGE


class SatelliteImage(NamedTuple):
    """
    A satellite raster tile fetched from a tile service, ready for
    use as a Plotly layout.images entry in geographic (lon/lat) coordinates.
    """
    data_uri: str    # "data:image/png;base64,..." — base64-encoded PNG
    lon_min: float   # west edge of the fetched image (WGS84)
    lon_max: float   # east edge
    lat_min: float   # south edge
    lat_max: float   # north edge


def _z_to_json_safe(arr: np.ndarray) -> list:
    """
    Convert a 2-D numpy array to a nested list with NaN/Inf replaced by None.

    shinywidgets serialises FigureWidget data with stdlib json.dumps, which
    raises ValueError on numpy NaN and Inf.  Plotly renders None as a gap
    in heatmaps, which is the correct visual for masked/invalid pixels.
    """
    return [
        [
            None if (v is not None and isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
            for v in row
        ]
        for row in arr.tolist()
    ]


# ---------------------------------------------------------------------------
# Colour scales
# ---------------------------------------------------------------------------

_NDVI_COLORSCALE   = "RdYlGn"
_METRIC_COLORSCALE = "Viridis"
_COVERAGE_COLORSCALE = "Blues"


def _choose_colorscale(metric_key: str) -> str:
    if "coverage" in metric_key:
        return _COVERAGE_COLORSCALE
    if "ndvi" in metric_key or "integrated" in metric_key:
        return _NDVI_COLORSCALE
    return _METRIC_COLORSCALE


# ---------------------------------------------------------------------------
# Basemap heatmap
# ---------------------------------------------------------------------------

def make_basemap_figure(
    z: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    metric_key: str,
    region_id: str,
    selected_pixel: tuple[float, float] | None = None,
    satellite: SatelliteImage | None = None,
    satellite_attribution: str = "© Esri",
    metric_opacity: float = 0.75,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.FigureWidget:
    """
    Build the interactive basemap FigureWidget.

    Parameters
    ----------
    z              : (ny, nx) 2-D metric values (display resolution)
    lon            : (ny, nx) WGS84 longitude array
    lat            : (ny, nx) WGS84 latitude array
    metric_key     : internal metric ID (used for colorscale choice & label)
    region_id      : shown in the title
    selected_pixel : (lon, lat) of the currently selected pixel, or None

    Notes
    -----
    - x-axis = lon[0, :] (longitude along columns)
    - y-axis = lat[:, 0] (latitude along rows)
    - uirevision="basemap" preserves zoom/pan when data is updated
    - The FigureWidget click event fires on 'plotly_click'; the caller
      wires fig.data[0].on_click(callback) after this function returns.
    - If the latitude axis appears inverted (image upside down), set
      yaxis=dict(autorange="reversed") in the layout below.
    """
    label, units = METRIC_LABELS.get(metric_key, (metric_key, ""))
    colorbar_title = f"{label}<br><sub>{units}</sub>" if units else label

    x_axis = lon[0, :]   # longitude vector along x (columns)
    y_axis = lat[:, 0]   # latitude vector along y  (rows)

    heatmap = go.Heatmap(
        z=_z_to_json_safe(z),
        x=x_axis.tolist(),
        y=y_axis.tolist(),
        colorscale=_choose_colorscale(metric_key),
        colorbar=dict(title=colorbar_title, thickness=15, len=0.8),
        hoverongaps=False,
        opacity=metric_opacity,
        zmin=zmin,
        zmax=zmax,
        zauto=(zmin is None and zmax is None),
        hovertemplate=(
            "Lon: %{x:.4f}<br>"
            "Lat: %{y:.4f}<br>"
            "Value: %{z:.4f}<extra></extra>"
        ),
        name=label,
    )

    traces: list[go.BaseTraceType] = [heatmap]

    # Always include the selected-pixel scatter trace — even when no pixel is
    # selected (empty arrays) — so fig.data[1] is always available for
    # in-place updates without needing to add/remove traces later.
    sel_lon = [selected_pixel[0]] if selected_pixel is not None else []
    sel_lat = [selected_pixel[1]] if selected_pixel is not None else []
    traces.append(
        go.Scatter(
            x=sel_lon,
            y=sel_lat,
            mode="markers",
            marker=dict(
                color="red",
                size=12,
                symbol="cross",
                line=dict(width=2, color="white"),
            ),
            name="Selected pixel",
            showlegend=False,
            hovertemplate=(
                "<b>Selected pixel</b><br>"
                "Lon: %{x:.4f}<br>"
                "Lat: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    # Axis ranges: tight to the data extent with a small margin.
    lon_pad = max((float(lon.max()) - float(lon.min())) * 0.02, 0.005)
    lat_pad = max((float(lat.max()) - float(lat.min())) * 0.02, 0.005)
    x_range = [float(lon.min()) - lon_pad, float(lon.max()) + lon_pad]
    y_range = [float(lat.min()) - lat_pad, float(lat.max()) + lat_pad]

    # Geographic aspect ratio: 1° lat ≠ 1° lon on screen at these latitudes.
    # scaleanchor + constrain="domain" keeps correct proportions by shrinking
    # the plot domain (letterbox) rather than extending the axis range (whitespace).
    mid_lat = float(np.mean(lat))
    scale_ratio = 1.0 / math.cos(math.radians(mid_lat))

    # Satellite imagery background (below heatmap) ---------------------------
    layout_images = []
    layout_annotations = []
    if satellite is not None:
        layout_images.append(dict(
            source=satellite.data_uri,
            xref="x", yref="y",
            x=satellite.lon_min,        # left edge in data coords
            y=satellite.lat_max,        # top edge (lat increases upward)
            sizex=satellite.lon_max - satellite.lon_min,
            sizey=satellite.lat_max - satellite.lat_min,
            xanchor="left", yanchor="top",
            sizing="stretch",
            opacity=1.0,
            layer="below",              # render under all traces
        ))
        layout_annotations.append(dict(
            text=f"Basemap: {satellite_attribution}",
            x=0.995, y=0.005, xref="paper", yref="paper",
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=8, color="#555"),
            bgcolor="rgba(255,255,255,0.65)",
            borderpad=2,
        ))

    fig = go.FigureWidget(
        data=traces,
        layout=go.Layout(
            title=dict(text=f"{region_id} — {label}", font=dict(size=14)),
            xaxis=dict(
                title="Longitude", showgrid=False,
                range=x_range, constrain="domain",
            ),
            yaxis=dict(
                title="Latitude", showgrid=False,
                range=y_range,
                scaleanchor="x", scaleratio=scale_ratio,
                constrain="domain",
            ),
            images=layout_images if layout_images else [],
            annotations=layout_annotations if layout_annotations else [],
            margin=dict(l=60, r=20, t=50, b=50),
            autosize=True,
            uirevision="basemap",
            plot_bgcolor="#f8f8f8",
        ),
    )
    return fig


def update_basemap_data(
    fig: go.FigureWidget,
    z: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    metric_key: str,
    selected_pixel: tuple[float, float] | None = None,
) -> None:
    """
    Update an existing basemap FigureWidget in place (avoids full re-render).
    Used when only the colormap metric changes — preserves zoom state.
    """
    label, units = METRIC_LABELS.get(metric_key, (metric_key, ""))
    colorbar_title = f"{label}<br><sub>{units}</sub>" if units else label
    with fig.batch_update():
        fig.data[0].z = _z_to_json_safe(z)
        fig.data[0].x = lon[0, :].tolist()
        fig.data[0].y = lat[:, 0].tolist()
        fig.data[0].colorscale = _choose_colorscale(metric_key)
        fig.data[0].colorbar.title = colorbar_title
        fig.layout.title.text = f"{fig.layout.title.text.split(' — ')[0]} — {label}"
        if selected_pixel is not None and len(fig.data) > 1:
            fig.data[1].x = [selected_pixel[0]]
            fig.data[1].y = [selected_pixel[1]]


def update_basemap_display(
    fig: go.FigureWidget,
    z: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    metric_key: str,
    selected_pixel: tuple[float, float] | None = None,
    satellite: SatelliteImage | None = None,
    satellite_attribution: str = "© Esri",
    metric_opacity: float = 0.75,
    zmin: float | None = None,
    zmax: float | None = None,
) -> None:
    """
    Update all display properties of an existing basemap FigureWidget in place.

    Called instead of a full re-render when metric, opacity, color-scale range,
    or satellite style changes — so the user's zoom/pan state is preserved.
    Assumes fig.data[1] is the selected-pixel Scatter trace (always present
    because make_basemap_figure unconditionally appends it).
    """
    label, units = METRIC_LABELS.get(metric_key, (metric_key, ""))
    colorbar_title = f"{label}<br><sub>{units}</sub>" if units else label

    with fig.batch_update():
        # Heatmap data + style
        fig.data[0].z              = _z_to_json_safe(z)
        fig.data[0].x              = lon[0, :].tolist()
        fig.data[0].y              = lat[:, 0].tolist()
        fig.data[0].colorscale     = _choose_colorscale(metric_key)
        fig.data[0].colorbar.title = colorbar_title
        fig.data[0].opacity        = metric_opacity
        fig.data[0].zmin           = zmin
        fig.data[0].zmax           = zmax
        fig.data[0].zauto          = (zmin is None and zmax is None)

        # Title (preserve "Region —" prefix)
        region_part = (fig.layout.title.text or "").split(" — ")[0].strip()
        fig.layout.title.text = f"{region_part} — {label}"

        # Selected-pixel marker (data[1] always exists)
        if selected_pixel is not None:
            fig.data[1].x = [selected_pixel[0]]
            fig.data[1].y = [selected_pixel[1]]
        else:
            fig.data[1].x = []
            fig.data[1].y = []

        # Satellite basemap image + attribution
        if satellite is not None:
            fig.layout.images = [dict(
                source=satellite.data_uri,
                xref="x", yref="y",
                x=satellite.lon_min,
                y=satellite.lat_max,
                sizex=satellite.lon_max - satellite.lon_min,
                sizey=satellite.lat_max - satellite.lat_min,
                xanchor="left", yanchor="top",
                sizing="stretch", opacity=1.0, layer="below",
            )]
            fig.layout.annotations = [dict(
                text=f"Basemap: {satellite_attribution}",
                x=0.995, y=0.005, xref="paper", yref="paper",
                xanchor="right", yanchor="bottom",
                showarrow=False,
                font=dict(size=8, color="#555"),
                bgcolor="rgba(255,255,255,0.65)",
                borderpad=2,
            )]
        else:
            fig.layout.images      = []
            fig.layout.annotations = []


# ---------------------------------------------------------------------------
# ipyleaflet map — fills its container, native zoom/pan, tile basemaps
# ---------------------------------------------------------------------------

LEAFLET_TILE_SERVICES: dict[str, dict] = {
    "World_Imagery": {
        "url":         "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "© Esri, Maxar, Earthstar Geographics",
        "max_zoom":    18,
        "label":       "Satellite",
    },
    "World_Topo_Map": {
        "url":         "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "© Esri, HERE, Garmin, FAO, USGS, NGA",
        "max_zoom":    18,
        "label":       "Topographic",
    },
    "World_Shaded_Relief": {
        "url":         "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        "attribution": "© Esri, USGS, NOAA",
        "max_zoom":    13,
        "label":       "Shaded Relief",
    },
    "OpenStreetMap": {
        "url":         "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "max_zoom":    19,
        "label":       "OpenStreetMap",
    },
}


def _mpl_cmap_for(metric_key: str) -> str:
    """Return the matplotlib colormap name that matches the Plotly colorscale."""
    if "coverage" in metric_key:
        return "Blues"
    if "ndvi" in metric_key or "integrated" in metric_key:
        return "RdYlGn"
    return "viridis"


def make_metric_overlay_png(
    z: np.ndarray,
    lat: np.ndarray,
    metric_key: str,
    zmin: float | None = None,
    zmax: float | None = None,
    opacity: float = 0.75,
) -> str:
    """
    Render a 2-D metric array as a georeferenced RGBA PNG data URI for
    ipyleaflet ImageOverlay.  NaN pixels are fully transparent; valid pixels
    have alpha = opacity.

    ImageOverlay anchors the PNG top-left corner to the NW corner of the
    bounds rectangle, so row 0 of the PNG must be the northernmost row.
    If lat[0, 0] < lat[-1, 0] (south-to-north storage order) the array is
    flipped vertically before rendering.
    """
    cmap = matplotlib.colormaps[_mpl_cmap_for(metric_key)]

    valid = z[~np.isnan(z)]
    if zmin is None:
        zmin = float(valid.min()) if len(valid) else 0.0
    if zmax is None:
        zmax = float(valid.max()) if len(valid) else 1.0
    if zmax <= zmin:
        zmax = zmin + 1e-6

    rgba = cmap(mcolors.Normalize(vmin=zmin, vmax=zmax)(z))  # (ny, nx, 4)
    rgba[..., 3] = np.where(np.isnan(z), 0.0, opacity)

    # Ensure north-first row order (top of PNG = north)
    if lat[0, 0] < lat[-1, 0]:
        rgba = rgba[::-1, :, :]

    img = PILImage.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def make_colorbar_html(
    metric_key: str,
    zmin: float,
    zmax: float,
) -> str:
    """
    Render a horizontal colorbar as a tiny matplotlib PNG and return HTML
    for display in the Shiny sidebar.  Uses the non-interactive Agg backend
    via matplotlib.figure.Figure so it is thread-safe.
    """
    label, units = METRIC_LABELS.get(metric_key, (metric_key, ""))
    title = f"{label} ({units})" if units else label

    fig = MplFigure(figsize=(3.2, 0.32))
    ax  = fig.add_axes([0.02, 0.05, 0.96, 0.85])
    matplotlib.colorbar.ColorbarBase(
        ax,
        cmap=matplotlib.colormaps[_mpl_cmap_for(metric_key)],
        norm=mcolors.Normalize(vmin=zmin, vmax=zmax),
        orientation="horizontal",
    ).ax.tick_params(labelsize=6)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=96, bbox_inches="tight", transparent=True)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    import matplotlib.pyplot as _plt; _plt.close(fig)  # noqa: E702

    return (
        f'<div style="font-size:0.72em;color:#444;margin-bottom:1px">{title}</div>'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;max-width:280px;height:auto"/>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.68em;color:#666">'
        f'<span>{zmin:.4g}</span><span>{zmax:.4g}</span></div>'
    )


def _auto_zoom(lat_range: float, lon_range: float) -> int:
    """Estimate an initial Leaflet zoom level that fits the bounding box."""
    extent = max(lat_range, lon_range)
    if extent <= 0:
        return 10
    return max(6, min(14, int(math.log2(360.0 / extent)) - 1))


def make_leaflet_map(
    z: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    metric_key: str,
    tile_service: str = "World_Imagery",
    metric_opacity: float = 0.75,
    zmin: float | None = None,
    zmax: float | None = None,
    native_lat_step: float | None = None,
    native_lon_step: float | None = None,
    shapefile_paths: str | None = None,
    shapefile_label_fields: str = "NAME",
) -> ipl.Map:
    """
    Build an ipyleaflet Map with:
      - A TileLayer basemap (ESRI or OpenStreetMap)
      - An ImageOverlay carrying the metric raster as an RGBA PNG
      - A Rectangle that highlights the selected pixel (initially invisible)

    Pixel step sizes (_lon_step, _lat_step) are stored on the map so the
    selection rectangle can be sized exactly to one pixel footprint.

    layout width/height="100%" makes the map fill its Shiny card container.
    The caller registers on_interaction() after this function returns.
    """
    lat_min = float(lat.min())
    lat_max = float(lat.max())
    lon_min = float(lon.min())
    lon_max = float(lon.max())
    center  = [(lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0]
    zoom    = _auto_zoom(lat_max - lat_min, lon_max - lon_min)

    # Pixel step size in degrees (used to size the selection rectangle)
    ny, nx   = z.shape
    lon_step = (lon_max - lon_min) / max(nx - 1, 1)
    lat_step = (lat_max - lat_min) / max(ny - 1, 1)

    svc = LEAFLET_TILE_SERVICES.get(tile_service, LEAFLET_TILE_SERVICES["World_Imagery"])
    tile_layer = ipl.TileLayer(
        url=svc["url"],
        attribution=svc["attribution"],
        max_zoom=svc.get("max_zoom", 18),
        name="Basemap",
    )

    image_overlay = ipl.ImageOverlay(
        url=make_metric_overlay_png(z, lat, metric_key, zmin, zmax, metric_opacity),
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        name="Metric overlay",
    )

    # Selected-pixel rectangle — always in the layer stack, hidden until clicked.
    # Bounds are a placeholder; update_leaflet_map sets them on selection.
    pixel_rect = ipl.Rectangle(
        bounds=[[center[0], center[1]], [center[0], center[1]]],
        color="#ffff00",
        weight=2,
        fill_color="#ffff00",
        fill_opacity=0.0,
        opacity=0.0,
        name="Selected pixel",
    )

    m = ipl.Map(
        center=center,
        zoom=zoom,
        scroll_wheel_zoom=True,
        layers=[tile_layer, image_overlay, pixel_rect],
        layout=dict(width="100%", height="100%"),
    )

    # Attach named references for in-place updates
    m._tile_layer    = tile_layer
    m._image_overlay = image_overlay
    m._pixel_rect    = pixel_rect
    # Use native pixel step for the selection rectangle when available so the
    # yellow highlight outlines exactly one 30 m pixel, not a coarsened block.
    m._lon_step = native_lon_step if native_lon_step is not None else lon_step
    m._lat_step = native_lat_step if native_lat_step is not None else lat_step

    m._shapefile_layers: list[dict] = []
    if shapefile_paths is not None:
        paths  = shapefile_paths.split()
        fields = shapefile_label_fields.split()
        for i, path in enumerate(paths):
            field = fields[i] if i < len(fields) else fields[-1]
            add_shapefile_overlay(m, path, label_field=field)

    return m


def update_leaflet_map(
    m: ipl.Map,
    z: np.ndarray,
    lat: np.ndarray,
    metric_key: str,
    tile_service: str | None = None,
    metric_opacity: float = 0.75,
    zmin: float | None = None,
    zmax: float | None = None,
    selected_pixel: tuple[float, float] | None = None,
) -> None:
    """
    Update the metric overlay (and optionally the tile layer / pixel marker)
    of an existing ipyleaflet Map in place — without resetting zoom or pan.

    Only the changed ipywidgets properties are sent to the browser as deltas.

    Parameters
    ----------
    tile_service   : pass to swap the basemap; None leaves it unchanged.
    selected_pixel : (lon, lat) to show the marker; None to hide it.
    """
    # Re-render the metric PNG and push just the URL delta
    m._image_overlay.url = make_metric_overlay_png(
        z, lat, metric_key, zmin, zmax, metric_opacity
    )

    # Swap tile layer only when the service key actually changes
    if tile_service is not None:
        svc = LEAFLET_TILE_SERVICES.get(tile_service, LEAFLET_TILE_SERVICES["World_Imagery"])
        if m._tile_layer.url != svc["url"]:
            m.remove_layer(m._tile_layer)
            m._tile_layer = ipl.TileLayer(
                url=svc["url"],
                attribution=svc["attribution"],
                max_zoom=svc.get("max_zoom", 18),
                name="Basemap",
            )
            m.add_layer(m._tile_layer)

    # Show / hide the selected-pixel rectangle.
    # Bounds are sized to exactly one pixel footprint using the step sizes
    # computed when the map was built (stored as m._lon_step / m._lat_step).
    if selected_pixel is not None:
        sel_lon, sel_lat = selected_pixel
        hlat = m._lat_step / 2.0
        hlon = m._lon_step / 2.0
        m._pixel_rect.bounds       = [
            [sel_lat - hlat, sel_lon - hlon],
            [sel_lat + hlat, sel_lon + hlon],
        ]
        m._pixel_rect.opacity      = 1.0
        m._pixel_rect.fill_opacity = 0.25
    else:
        m._pixel_rect.opacity      = 0.0
        m._pixel_rect.fill_opacity = 0.0


# ---------------------------------------------------------------------------
# Shapefile overlay
# ---------------------------------------------------------------------------

def add_shapefile_overlay(
    m: ipl.Map,
    shapefile_path,
    label_field: str = "NAME",
) -> None:
    """
    Load a shapefile and add it to an existing ipyleaflet Map.

    Adds two layers:
      - A GeoJSON layer for the polygon/line outlines.
      - One Marker + DivIcon per feature for persistent text labels
        positioned at each feature's centroid.

    Parameters
    ----------
    m              : ipyleaflet.Map to add layers to.
    shapefile_path : Path or str to the .shp file.
    label_field    : Attribute field name used as the label text.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not installed — shapefile overlay skipped.")
        return

    from pathlib import Path as _Path

    path = _Path(shapefile_path)
    if not path.exists():
        print(f"Shapefile not found: {path} — overlay skipped.")
        return

    gdf = gpd.read_file(path)
    # Reproject to WGS84 so coordinates match the Leaflet map
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # --- Shape outlines via GeoJSON layer ---
    geojson_layer = ipl.GeoJSON(
        data=gdf.__geo_interface__,
        style={
            "color": "#ffffff",
            "weight": 2,
            "fillOpacity": 0.0,
        },
        hover_style={
            "color": "#ffff00",
            "weight": 3,
        },
        name="Shapefile overlay",
    )
    m.add_layer(geojson_layer)

    # --- Text labels via Marker + DivIcon at each centroid ---
    label_markers = []
    for _, row in gdf.iterrows():
        label_text = str(row.get(label_field, "")) if label_field in gdf.columns else ""
        if not label_text:
            continue
        centroid = row.geometry.centroid
        icon = ipl.DivIcon(
            html=f'<div style="'
                 f'color:#ffffff;'
                 f'font-size:11px;'
                 f'font-weight:bold;'
                 f'text-shadow:0 0 3px #000,0 0 3px #000;'
                 f'white-space:nowrap;'
                 f'pointer-events:none;'
                 f'">{label_text}</div>',
            icon_size=[0, 0],
            icon_anchor=[0, 0],
        )
        marker = ipl.Marker(
            location=[centroid.y, centroid.x],
            icon=icon,
            draggable=False,
        )
        m.add_layer(marker)
        label_markers.append(marker)

    # Register in the ordered layer list for checkbox-driven visibility toggling.
    from pathlib import Path as _Path
    m._shapefile_layers.append({
        "name":   _Path(shapefile_path).stem,
        "geojson": geojson_layer,
        "labels":  label_markers,
    })


# ---------------------------------------------------------------------------
# Time series plot
# ---------------------------------------------------------------------------

def _ndvi_compatible(metric_key: str) -> bool:
    """True if the metric is in NDVI units and reference lines make sense."""
    return any(s in metric_key for s in ("ndvi", "integrated"))


def make_timeseries_figure(
    ts,                             # PixelTimeSeries
    smoothed_daily: np.ndarray,     # (n_days,) smoothed on daily grid
    daily_dates: np.ndarray,        # datetime64[D] (n_days,)
    region_id: str,
    vi_var: str = "NDVI",
    basemap_metric: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.FigureWidget:
    """
    Build the pixel time series FigureWidget.

    Shows:
    - Grey scatter dots: raw valid observations
    - Green line:        Whittaker-smoothed curve on the daily grid
    - Dashed reference lines at zmin / zmax when the basemap metric is
      NDVI-compatible, so the time series shares the same value range as
      the map colorscale.
    """
    # Raw observations (valid only)
    obs_dates = ts.dates[ts.valid_mask].astype("datetime64[ms]").astype(str)
    obs_vi    = ts.raw_vi[ts.valid_mask].tolist()

    # Smoothed curve on daily grid — filter out NaN segments for clean lines
    daily_date_strs = daily_dates.astype("datetime64[ms]").astype(str)
    smooth_vi = smoothed_daily.tolist()

    raw_trace = go.Scatter(
        x=obs_dates,
        y=obs_vi,
        mode="markers",
        marker=dict(color="#888888", size=4, opacity=0.65),
        name="Raw observations",
        hovertemplate=f"Date: %{{x}}<br>{vi_var}: %{{y:.4f}}<extra></extra>",
    )

    smooth_trace = go.Scatter(
        x=daily_date_strs,
        y=smooth_vi,
        mode="lines",
        line=dict(color="#2ca02c", width=2),
        name="Whittaker smoothed",
        hovertemplate=f"Date: %{{x}}<br>{vi_var}: %{{y:.4f}}<extra></extra>",
        connectgaps=False,
    )

    subtitle = (
        f"Lat {ts.lat:.4f}°, Lon {ts.lon:.4f}° | "
        f"n={int(ts.valid_mask.sum())} valid obs"
    )

    # Y-axis: autoscale to the actual plotted data so the smoothed curve and
    # all observations are always within the plot area regardless of filter.
    all_values = obs_vi + [v for v in smooth_vi if v is not None and not math.isnan(v)]
    if all_values:
        _dmin, _dmax = min(all_values), max(all_values)
        _pad = max((_dmax - _dmin) * 0.08, 0.02)
        y_range = [_dmin - _pad, _dmax + _pad]
    else:
        _vi_lo, _vi_hi = VI_VALID_RANGE.get(vi_var, (-0.15, 1.05))
        y_range = [_vi_lo - 0.05, _vi_hi + 0.05]

    fig = go.FigureWidget(
        data=[raw_trace, smooth_trace],
        layout=go.Layout(
            title=dict(
                text=(
                    f"{region_id} — Pixel time series"
                    f"<br><sup style='font-size:11px'>{subtitle}</sup>"
                ),
                font=dict(size=13),
            ),
            xaxis=dict(title="Date", showgrid=True, gridcolor="#e0e0e0"),
            yaxis=dict(
                title=vi_var,
                range=y_range,
                showgrid=True,
                gridcolor="#e0e0e0",
            ),
            autosize=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=60, r=20, t=70, b=50),
            uirevision=f"timeseries-{zmin}-{zmax}",
            plot_bgcolor="#f8f8f8",
        ),
    )
    return fig


def make_empty_timeseries_figure() -> go.FigureWidget:
    """Placeholder shown before a pixel is selected."""
    fig = go.FigureWidget(
        layout=go.Layout(
            title=dict(
                text="Click a pixel on the map to view its time series",
                font=dict(size=13, color="#888888"),
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            autosize=True,
            plot_bgcolor="#f8f8f8",
            paper_bgcolor="#f8f8f8",
            annotations=[
                dict(
                    text="← Select a pixel",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="#aaaaaa"),
                )
            ],
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Annual-cycle and per-metric trend figures
# ---------------------------------------------------------------------------

# Mean metrics that have per-year data in the annual_data dict
_ANNUAL_MEAN_METRICS: list[str] = [
    "peak_ndvi_mean",
    "peak_doy_mean",
    "integrated_ndvi_mean",
    "greenup_rate_mean",
    "floor_ndvi_mean",
    "ceiling_ndvi_mean",
    "season_length_mean",
    "n_peaks_mean",
    "peak_separation_mean",
    "relative_peak_amplitude_mean",
    "valley_depth_mean",
]

# Maps mean-metric key → key in the annual_data dict
_ANNUAL_KEY: dict[str, str] = {
    "peak_ndvi_mean":               "peak_ndvi",
    "peak_doy_mean":                "peak_doy",
    "integrated_ndvi_mean":         "integrated",
    "greenup_rate_mean":            "greenup",
    "floor_ndvi_mean":              "floor",
    "ceiling_ndvi_mean":            "ceiling",
    "season_length_mean":           "season_len",
    "n_peaks_mean":                 "n_peaks",
    "peak_separation_mean":         "peak_sep",
    "relative_peak_amplitude_mean": "rel_amp",
    "valley_depth_mean":            "valley",
}

# Maps mean-metric key → corresponding std metric key (where one exists)
_STD_KEY: dict[str, str] = {
    "peak_ndvi_mean":       "peak_ndvi_std",
    "peak_doy_mean":        "peak_doy_std",
    "integrated_ndvi_mean": "integrated_ndvi_std",
    "greenup_rate_mean":    "greenup_rate_std",
    "season_length_mean":   "season_length_std",
}

# Consistent year colour palette (up to 8 years)
_YEAR_PALETTE: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _year_color(idx: int) -> str:
    return _YEAR_PALETTE[idx % len(_YEAR_PALETTE)]


def _short_metric_label(metric_key: str) -> str:
    """Compact subplot title: strip '(mean)' / 'Mean', append units."""
    label, units = METRIC_LABELS.get(metric_key, (metric_key, ""))
    for suffix in (" (mean)", " Mean", "(mean)"):
        label = label.replace(suffix, "")
    label = label.strip()
    return f"{label} [{units}]" if units else label


def make_annual_cycle_figure(
    smoothed: np.ndarray,
    daily_dates: np.ndarray,
    region_id: str,
    vi_var: str = "NDVI",
    basemap_metric: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.FigureWidget:
    """
    Seasonal-cycle view of the Whittaker-smoothed NDVI record.

    X-axis: Day of Year (1–366).  One trace per calendar year in the record,
    coloured consistently with make_metrics_annual_figure.  A black 'Mean'
    trace shows the cross-year per-DOY average.  Individual years can be
    toggled via the Plotly legend.

    Dashed reference lines at zmin / zmax are added when the current basemap
    metric is NDVI-compatible.
    """
    dates_dt  = pd.DatetimeIndex(daily_dates.astype("datetime64[ns]"))
    years_arr = dates_dt.year.to_numpy()
    doys_arr  = dates_dt.day_of_year.to_numpy()   # 1–366

    unique_years = sorted(set(years_arr.tolist()))
    traces: list[go.BaseTraceType] = []
    doy_rows: list[pd.DataFrame] = []

    for i_yr, yr in enumerate(unique_years):
        mask   = years_arr == yr
        doy_yr = doys_arr[mask]
        val_yr = smoothed[mask]
        color  = _year_color(i_yr)
        traces.append(go.Scatter(
            x=doy_yr.tolist(),
            y=val_yr.tolist(),
            mode="lines",
            name=str(yr),
            legendgroup=str(yr),
            line=dict(color=color, width=1.5),
            opacity=0.75,
            hovertemplate=f"{yr}, DOY %{{x}}: %{{y:.4f}}<extra></extra>",
        ))
        doy_rows.append(pd.DataFrame({"doy": doy_yr, "ndvi": val_yr}))

    # Cross-year mean (per DOY)
    if doy_rows:
        mean_doy = (
            pd.concat(doy_rows)
            .groupby("doy")["ndvi"]
            .mean()
            .reset_index()
        )
        traces.append(go.Scatter(
            x=mean_doy["doy"].tolist(),
            y=mean_doy["ndvi"].tolist(),
            mode="lines",
            name="Mean",
            legendgroup="mean",
            line=dict(color="#000000", width=2.5),
            hovertemplate="Mean, DOY %{x}: %{y:.4f}<extra></extra>",
        ))

    # Y-axis: autoscale to the actual smoothed data so all traces stay visible.
    valid_vals = smoothed[~np.isnan(smoothed)]
    if len(valid_vals) > 0:
        _dmin, _dmax = float(valid_vals.min()), float(valid_vals.max())
        _pad = max((_dmax - _dmin) * 0.08, 0.02)
        y_range = [_dmin - _pad, _dmax + _pad]
    else:
        _vi_lo, _vi_hi = VI_VALID_RANGE.get(vi_var, (-0.15, 1.05))
        y_range = [_vi_lo - 0.05, _vi_hi + 0.05]

    return go.FigureWidget(
        data=traces,
        layout=go.Layout(
            title=dict(text=f"{region_id} — Annual {vi_var} Cycles", font=dict(size=13)),
            xaxis=dict(title="Day of Year", showgrid=True, gridcolor="#e0e0e0",
                       range=[1, 366]),
            yaxis=dict(
                title=vi_var,
                range=y_range,
                showgrid=True,
                gridcolor="#e0e0e0",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                        font=dict(size=10)),
            autosize=True,
            margin=dict(l=60, r=20, t=70, b=50),
            uirevision=f"annual_cycle-{zmin}-{zmax}",
            plot_bgcolor="#f8f8f8",
        ),
    )


def make_metrics_annual_figure(
    valid_years: list[int],
    annual_data: dict[str, list],
    metrics: dict[str, float],
    region_id: str,
) -> go.FigureWidget:
    """
    2-column subplot grid showing each annual metric across years.

    For each metric in _ANNUAL_MEAN_METRICS:
    - Coloured scatter markers, one per year (same colours as make_annual_cycle_figure)
    - Dashed black horizontal 'Mean' line
    - Grey ±std shaded band (for metrics that have a std counterpart)

    A shared Plotly legend (one entry per year + 'Mean') allows toggling
    individual years across all subplots simultaneously via legendgroup.
    The figure height is set proportionally to the number of rows so that
    the card can scroll vertically to reveal all subplots.
    """
    mean_keys = [k for k in _ANNUAL_MEAN_METRICS if k in _ANNUAL_KEY]
    n_plots   = len(mean_keys)
    n_cols    = 2
    n_rows    = math.ceil(n_plots / n_cols)

    subplot_titles = [_short_metric_label(k) for k in mean_keys]
    subplot_titles += [""] * (n_rows * n_cols - n_plots)   # pad grid

    fig_base = _make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.12,
    )

    x_span = (
        [min(valid_years) - 0.5, max(valid_years) + 0.5]
        if valid_years else [0.5, 1.5]
    )

    for plot_idx, metric_key in enumerate(mean_keys):
        row  = plot_idx // n_cols + 1
        col  = plot_idx %  n_cols + 1
        akey = _ANNUAL_KEY[metric_key]
        yr_vals = annual_data.get(akey, [])

        # Per-year scatter markers
        for i_yr, yr in enumerate(valid_years):
            val = yr_vals[i_yr] if i_yr < len(yr_vals) else float("nan")
            y_val = None if (isinstance(val, float) and np.isnan(val)) else val
            fig_base.add_trace(
                go.Scatter(
                    x=[yr],
                    y=[y_val],
                    mode="markers",
                    name=str(yr),
                    legendgroup=str(yr),
                    showlegend=(plot_idx == 0),
                    marker=dict(color=_year_color(i_yr), size=10, symbol="circle"),
                    hovertemplate=f"{yr}: %{{y:.4f}}<extra></extra>",
                ),
                row=row, col=col,
            )

        # Mean ± std
        mean_val = metrics.get(metric_key)
        std_key  = _STD_KEY.get(metric_key)
        std_val  = metrics.get(std_key) if std_key else None

        def _valid(v) -> bool:
            return v is not None and not np.isnan(float(v))

        if _valid(mean_val):
            # ±std shaded band
            if std_key and _valid(std_val):
                hi = float(mean_val) + float(std_val)
                lo = float(mean_val) - float(std_val)
                fig_base.add_trace(
                    go.Scatter(
                        x=x_span + x_span[::-1],
                        y=[hi, hi, lo, lo],
                        fill="toself",
                        fillcolor="rgba(0,0,0,0.08)",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row, col=col,
                )
            # Mean line
            fig_base.add_trace(
                go.Scatter(
                    x=x_span,
                    y=[float(mean_val), float(mean_val)],
                    mode="lines",
                    name="Mean",
                    legendgroup="mean",
                    showlegend=(plot_idx == 0),
                    line=dict(color="#000000", width=1.8, dash="dash"),
                    hovertemplate=f"Mean: {float(mean_val):.4f}<extra></extra>",
                ),
                row=row, col=col,
            )

    fig = go.FigureWidget(fig_base)
    fig.update_layout(
        title=dict(text=f"{region_id} — Annual Metric Trends", font=dict(size=13)),
        height=n_rows * 200 + 80,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=10), traceorder="normal"),
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=50),
        uirevision="metrics_annual",
    )
    fig.update_xaxes(
        tickformat="d",
        tickmode="array",
        tickvals=valid_years if valid_years else [],
        showgrid=True,
        gridcolor="#e0e0e0",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
    return fig


# ---------------------------------------------------------------------------
# Pixel metrics sidebar table
# ---------------------------------------------------------------------------

def _metric_swatch_html(val: float, metric_key: str, zmin: float, zmax: float) -> str:
    """
    Return an inline HTML color swatch (10×10 px square) whose fill color
    matches the map colorscale position of val within [zmin, zmax].
    Returns an empty string if val is NaN or the range is degenerate.
    """
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return ""
    if np.isnan(fval) or zmax <= zmin:
        return ""
    t = max(0.0, min(1.0, (fval - zmin) / (zmax - zmin)))
    r, g, b, _ = matplotlib.colormaps[_mpl_cmap_for(metric_key)](t)
    hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return (
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'background:{hex_color};border:1px solid #888;'
        f'margin-right:4px;vertical-align:middle"></span>'
    )


def make_metrics_table(
    metrics: dict[str, float],
    selected_metric: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> str:
    """
    Render all 19 phenological metrics as a grouped HTML table.

    Parameters
    ----------
    metrics          : dict from compute_pixel_metrics()
    selected_metric  : the currently selected metric key (highlighted in bold)
    zmin, zmax       : color-scale range from the map; when provided, a color
                       swatch is shown next to the highlighted metric value so
                       the user can see where the pixel falls in the map range.

    Returns
    -------
    HTML string for use with ui.HTML()
    """
    rows: list[str] = []
    for group_name, metric_keys in METRIC_GROUPS.items():
        rows.append(
            f'<tr><td colspan="3" style="'
            f'font-weight:bold;background:#e8f0e8;'
            f'padding:4px 6px;font-size:0.8em;letter-spacing:0.05em;">'
            f'{group_name.upper()}</td></tr>'
        )
        for key in metric_keys:
            val = metrics.get(key, np.nan)
            label, units = METRIC_LABELS.get(key, (key, ""))
            val_str = f"{val:.4f}" if (val is not None and not np.isnan(float(val) if val is not None else float("nan"))) else "N/A"
            is_selected = key == selected_metric
            highlight = "font-weight:bold;background:#fff9c4;" if is_selected else ""

            # Color swatch for the highlighted metric (shows map position)
            swatch = ""
            if is_selected and zmin is not None and zmax is not None and val_str != "N/A":
                swatch = _metric_swatch_html(val, key, zmin, zmax)

            rows.append(
                f'<tr style="{highlight}">'
                f'<td style="padding:2px 6px;font-size:0.78em">{label}</td>'
                f'<td style="padding:2px 6px;font-size:0.78em;text-align:right">'
                f'{swatch}{val_str}</td>'
                f'<td style="padding:2px 4px;font-size:0.72em;color:#666">{units}</td>'
                f'</tr>'
            )

    return (
        '<table style="width:100%;border-collapse:collapse;'
        'font-family:monospace">'
        + "".join(rows)
        + "</table>"
    )
