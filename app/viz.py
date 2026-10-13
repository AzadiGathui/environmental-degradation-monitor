"""
Build a Folium map with toggleable layers:
  • RGB true-colour (T1 and T2)
  • Segmentation mask (T1 and T2)
  • Change / encroachment overlay
  • Protected-area boundaries
  • County boundaries
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds

from app.config import (
    CLASS_COLORS_RGBA, CLASS_HEX, CLASS_NAMES,
    CHANGE_RGBA, COUNTIES_GEOJSON, PA_GEOJSON, TARGET_CRS, WGS84_CRS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wgs84_bounds(src_path: Path) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS-84 for a raster file."""
    with rasterio.open(src_path) as src:
        west, south, east, north = transform_bounds(
            src.crs, WGS84_CRS, *src.bounds
        )
    return west, south, east, north


def _mask_to_rgba(mask: np.ndarray, alpha: int = 160) -> np.ndarray:
    """
    Convert a 2-D uint8 class-index mask to an RGBA image.

    Unknown class values default to fully transparent.
    """
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS_RGBA.items():
        px = mask == cls_idx
        rgba[px] = color   # already includes per-class alpha
    return rgba


def _change_to_rgba(change_mask: np.ndarray) -> np.ndarray:
    H, W = change_mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[change_mask] = CHANGE_RGBA
    return rgba


def _array_to_png_b64(arr: np.ndarray) -> str:
    """Encode an HxWx4 RGBA array as a base64 PNG data URI."""
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _add_raster_overlay(
    fmap: folium.Map,
    array: np.ndarray,
    bounds_wgs84: tuple,
    name: str,
    show: bool = True,
    opacity: float = 0.7,
) -> None:
    """Add a numpy RGBA array as an ImageOverlay layer."""
    west, south, east, north = bounds_wgs84
    data_uri = _array_to_png_b64(array)
    layer = folium.raster_layers.ImageOverlay(
        image=data_uri,
        bounds=[[south, west], [north, east]],
        opacity=opacity,
        name=name,
        show=show,
        cross_origin=False,
    )
    layer.add_to(fmap)


def _load_mask_array(mask_path: Path) -> tuple[np.ndarray, tuple]:
    with rasterio.open(mask_path) as src:
        arr    = src.read(1)
        bounds = transform_bounds(src.crs, WGS84_CRS, *src.bounds)
    return arr, bounds


def _load_rgb_array(rgb_path: Path) -> tuple[np.ndarray, tuple]:
    """Return (H×W×3 uint8, wgs84_bounds)."""
    with rasterio.open(rgb_path) as src:
        bands  = src.read([1, 2, 3]).transpose(1, 2, 0)
        bounds = transform_bounds(src.crs, WGS84_CRS, *src.bounds)
    return bands, bounds


def _center(bounds_wgs84: tuple) -> tuple[float, float]:
    west, south, east, north = bounds_wgs84
    return (south + north) / 2, (west + east) / 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_legend_html() -> str:
    """Return an HTML string for a floating class-colour legend."""
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<div style="width:14px;height:14px;background:{CLASS_HEX[i]};'
        f'border:1px solid #666;flex-shrink:0"></div>'
        f'<span style="font-size:12px">{name}</span></div>'
        for i, name in enumerate(CLASS_NAMES)
    )
    change_item = (
        '<div style="display:flex;align-items:center;gap:6px;margin:6px 0 3px">'
        '<div style="width:14px;height:14px;background:#ff0000;'
        'border:1px solid #666;flex-shrink:0"></div>'
        '<span style="font-size:12px"><b>Encroachment</b></span></div>'
    )
    return (
        '<div style="position:fixed;bottom:30px;right:10px;z-index:1000;'
        'background:#ffffff;color:#222222;padding:10px;border:1px solid #ccc;'
        'border-radius:6px;font-family:sans-serif;min-width:160px">'
        "<b style='font-size:13px;color:#222222'>Land cover</b><br>"
        + items
        + change_item
        + "</div>"
    )


def build_map(
    aoi_geom,
    rgb1_path: Path,
    rgb2_path: Path,
    mask1_path: Path,
    mask2_path: Path,
    change_result: dict | None = None,
) -> folium.Map:
    """
    Build a Folium map with toggleable layers for the analysis result.

    Parameters
    ----------
    aoi_geom       : shapely geometry in WGS-84 (used for zoom)
    rgb1_path      : T1 RGB GeoTIFF
    rgb2_path      : T2 RGB GeoTIFF
    mask1_path     : T1 segmentation mask GeoTIFF
    mask2_path     : T2 segmentation mask GeoTIFF
    change_result  : output of change_detection.detect_change() (may be None)
    """
    bounds1 = _wgs84_bounds(rgb1_path)
    center  = _center(bounds1)

    fmap = folium.Map(
        location=center,
        zoom_start=11,
        tiles="CartoDB positron",
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

    # --- RGB overlays ---
    rgb1, b1 = _load_rgb_array(rgb1_path)
    # Convert to RGBA for ImageOverlay
    rgba1 = np.concatenate([rgb1, np.full((*rgb1.shape[:2], 1), 200, dtype=np.uint8)], axis=-1)
    _add_raster_overlay(fmap, rgba1, b1, name="RGB T1 (true colour)", show=False, opacity=0.9)

    if rgb2_path.exists():
        rgb2, b2 = _load_rgb_array(rgb2_path)
        rgba2 = np.concatenate([rgb2, np.full((*rgb2.shape[:2], 1), 200, dtype=np.uint8)], axis=-1)
        _add_raster_overlay(fmap, rgba2, b2, name="RGB T2 (true colour)", show=False, opacity=0.9)

    # --- Mask overlays ---
    if mask1_path.exists():
        m1, mb1 = _load_mask_array(mask1_path)
        _add_raster_overlay(fmap, _mask_to_rgba(m1), mb1, name="Mask T1 (segmentation)", show=False)

    if mask2_path.exists():
        m2, mb2 = _load_mask_array(mask2_path)
        _add_raster_overlay(fmap, _mask_to_rgba(m2), mb2, name="Mask T2 (segmentation)", show=False)

    # --- Change / encroachment overlay ---
    if change_result and change_result["change_mask"].any():
        cm = change_result["change_mask"]
        # Use mask1 bounds for the change overlay (same grid)
        _, chg_bounds = _load_mask_array(mask1_path)
        _add_raster_overlay(
            fmap, _change_to_rgba(cm), chg_bounds,
            name="Encroachment (new built-up)", show=False, opacity=0.8,
        )

        enc = change_result.get("encroachments")
        if enc is not None and len(enc):
            enc_wgs = enc.to_crs(WGS84_CRS)
            folium.GeoJson(
                enc_wgs.__geo_interface__,
                name="Encroachment polygons (PA overlap)",
                show=True,   # on by default
                style_function=lambda _: {
                    "fillColor": "#ff0000",
                    "color": "#990000",
                    "weight": 1.5,
                    "fillOpacity": 0.5,
                },
                tooltip=folium.GeoJsonTooltip(fields=["name", "area_km2"],
                                               aliases=["Protected area", "Area (km²)"]),
            ).add_to(fmap)

    # --- Protected-area boundaries ---
    if PA_GEOJSON.exists():
        pa = gpd.read_file(PA_GEOJSON).to_crs(WGS84_CRS)
        # Clip to rough AOI extent for performance
        west, south, east, north = bounds1
        pad = 0.5
        pa_clip = pa.cx[west - pad: east + pad, south - pad: north + pad]
        if len(pa_clip):
            folium.GeoJson(
                pa_clip[["name", "geometry"]].__geo_interface__,
                name="Protected areas (WDPA)",
                show=True,   # on by default
                style_function=lambda _: {
                    "fillColor": "none",
                    "color": "#2ecc71",
                    "weight": 2,
                    "fillOpacity": 0,
                },
                tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["PA:"]),
            ).add_to(fmap)

    # --- County boundaries ---
    if COUNTIES_GEOJSON.exists():
        counties = gpd.read_file(COUNTIES_GEOJSON)
        folium.GeoJson(
            counties[["name", "geometry"]].__geo_interface__,
            name="County boundaries",
            style_function=lambda _: {
                "fillColor": "none",
                "color": "#555555",
                "weight": 1,
                "fillOpacity": 0,
                "dashArray": "4 4",
            },
            tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["County:"]),
            show=False,
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    # Floating legend injected as raw HTML
    fmap.get_root().html.add_child(folium.Element(make_legend_html()))

    return fmap
