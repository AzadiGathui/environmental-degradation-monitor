"""
Change detection and encroachment alerting.

Steps
-----
1. Load T1 and T2 class masks (both in TARGET_CRS after inference).
2. Reproject T2 onto T1's exact pixel grid so masks are pixel-aligned.
3. Find pixels where T2 == BUILT_UP and T1 != BUILT_UP  →  encroachment candidates.
4. Vectorise the change pixels and intersect with protected-area polygons.
5. Return a summary dict and a GeoDataFrame of encroachment polygons per PA.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import shape

from app.config import (
    BUILT_UP_IDX, PA_GEOJSON, PA_BUFFER_GEOJSON, PA_BUFFER_M, TARGET_CRS, CLASS_NAMES,
)


def _load_pa_layers():
    """
    Return (pa, pa_buf) GeoDataFrames in TARGET_CRS.

    `pa`     : raw protected-area polygons.
    `pa_buf` : the same polygons grown by PA_BUFFER_M metres. Loaded from the
               pre-built pa_100m_buffer.geojson if present (produced by
               preprocessing.ipynb); otherwise buffered on the fly so the app
               still works before that notebook has been run.
    """
    pa = gpd.read_file(PA_GEOJSON).to_crs(TARGET_CRS)[["name", "geometry"]]

    if PA_BUFFER_GEOJSON.exists():
        pa_buf = gpd.read_file(PA_BUFFER_GEOJSON).to_crs(TARGET_CRS)[["name", "geometry"]]
    else:
        pa_buf = pa.copy()
        pa_buf["geometry"] = pa.buffer(PA_BUFFER_M)   # metric CRS → buffer in metres
    return pa, pa_buf


def _load_aligned(mask1_path: Path, mask2_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read both masks and warp T2 onto T1's grid.
    Returns (mask1, mask2_aligned, t1_profile).
    """
    with rasterio.open(mask1_path) as src1:
        mask1   = src1.read(1)
        profile = src1.profile.copy()
        t       = src1.transform
        crs     = src1.crs
        w, h    = src1.width, src1.height

    with rasterio.open(mask2_path) as src2:
        # WarpedVRT lazily reprojects T2 onto T1's exact grid
        with WarpedVRT(
            src2,
            crs=crs,
            transform=t,
            width=w,
            height=h,
            resampling=Resampling.nearest,   # categorical data — no interpolation
        ) as vrt:
            mask2 = vrt.read(1)

    return mask1, mask2, profile


def _vectorise_change(change_mask: np.ndarray, profile: dict) -> gpd.GeoDataFrame:
    """
    Convert a boolean change raster to a GeoDataFrame of polygons in TARGET_CRS.
    """
    binary = change_mask.astype(np.uint8)
    transform = profile["transform"]
    crs       = profile["crs"]

    polys = [
        shape(geom)
        for geom, val in shapes(binary, transform=transform)
        if val == 1
    ]

    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    return gpd.GeoDataFrame(geometry=polys, crs=crs)


def detect_change(
    mask1_path: Path,
    mask2_path: Path,
    progress_cb=None,
) -> dict:
    """
    Compute change between two segmentation masks and intersect with protected areas.

    Returns a dict with keys:
        change_mask   : np.ndarray (H, W) bool  — pixels that changed to built-up
        encroachments : GeoDataFrame — change polygons that overlap a PA (with PA name)
        stats         : dict — pixel counts and area estimates
        profile       : rasterio profile of the mask (for bounds / CRS)
    """
    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    _log("Aligning masks …")
    mask1, mask2, profile = _load_aligned(mask1_path, mask2_path)

    # Pixels that transitioned TO built-up
    change_mask = (mask2 == BUILT_UP_IDX) & (mask1 != BUILT_UP_IDX)

    px_area_m2 = abs(profile["transform"].a * profile["transform"].e)
    changed_px  = int(change_mask.sum())
    changed_km2 = changed_px * px_area_m2 / 1e6

    _log(f"Change pixels: {changed_px:,} (~{changed_km2:.2f} km²)")

    # Per-class change breakdown (what did those pixels used to be?)
    from_classes = {}
    for cls_idx, name in enumerate(CLASS_NAMES):
        n = int(((mask1 == cls_idx) & change_mask).sum())
        if n:
            from_classes[name] = n

    stats = {
        "changed_px":   changed_px,
        "changed_km2":  round(changed_km2, 3),
        "from_classes": from_classes,
        "t1_built_px":  int((mask1 == BUILT_UP_IDX).sum()),
        "t2_built_px":  int((mask2 == BUILT_UP_IDX).sum()),
    }

    if changed_px == 0:
        _log("No change detected.")
        return {
            "change_mask":   change_mask,
            "encroachments": gpd.GeoDataFrame(geometry=[], crs=profile["crs"]),
            "stats":         stats,
            "profile":       profile,
        }

    _log("Vectorising change pixels …")
    change_gdf = _vectorise_change(change_mask, profile)

    _log("Loading protected-area boundaries (+100 m buffer) …")
    pa, pa_buf = _load_pa_layers()

    # Encroachment inside the protected-area-plus-buffer zone. This is the headline:
    # protected land (or its 100 m margin) newly converted to built-up.
    _log("Intersecting change with protected area + 100 m buffer …")
    encroachments = gpd.overlay(change_gdf, pa_buf, how="intersection")
    encroachments["area_km2"] = (encroachments.geometry.area / 1e6).round(4)

    # Split into inside-the-PA vs. in-the-buffer-margin, per PA.
    inside = gpd.overlay(encroachments, pa, how="intersection")
    inside_by_pa = (
        inside.assign(a=inside.geometry.area / 1e6).groupby("name_1")["a"].sum()
        if len(inside) else {}
    )

    protected_lost_km2 = float(encroachments["area_km2"].sum())
    inside_pa_km2      = float(inside.geometry.area.sum() / 1e6) if len(inside) else 0.0
    buffer_km2         = max(protected_lost_km2 - inside_pa_km2, 0.0)

    stats["protected_lost_km2"] = round(protected_lost_km2, 3)   # headline: PA + buffer
    stats["inside_pa_km2"]      = round(inside_pa_km2, 3)        # strictly inside the PA
    stats["buffer_km2"]         = round(buffer_km2, 3)           # in the 100 m margin only
    # Back-compat alias used elsewhere in the UI
    stats["encroachment_km2"]   = round(protected_lost_km2, 3)
    stats["encroachment_sites"] = len(encroachments)
    stats["affected_pas"]       = sorted(encroachments["name"].unique().tolist())

    _log(
        f"Protected land lost: {protected_lost_km2:.3f} km² "
        f"({inside_pa_km2:.3f} inside PA + {buffer_km2:.3f} in 100 m buffer) "
        f"across {len(stats['affected_pas'])} site(s)"
    )

    return {
        "change_mask":   change_mask,
        "encroachments": encroachments,
        "stats":         stats,
        "profile":       profile,
    }
