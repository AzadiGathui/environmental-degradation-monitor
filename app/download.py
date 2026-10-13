"""
Download Sentinel-2 L2A RGB composites from the Copernicus Data Space Ecosystem (CDSE).

Requires a free CDSE account (dataspace.copernicus.eu) and OAuth credentials
stored as environment variables:
    CDSE_CLIENT_ID
    CDSE_CLIENT_SECRET

For the default Nairobi AOI these are NOT required — pre-downloaded imagery is
seeded into the cache automatically by app/cache.py.
"""
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

from app.config import S2_RESOLUTION, MAX_PX, TARGET_CRS

# Evalscript: true-colour composite (B04=R, B03=G, B02=B), float32 reflectance.
# LEAST_CC mosaicking selects the least-cloudy scene per pixel automatically.
_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input:  [{bands: ["B04", "B03", "B02"], units: "REFLECTANCE"}],
        output: [{id: "default", bands: 3, sampleType: "FLOAT32"}]
    };
}
function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
}
"""


def _sh_config():
    """Build a SentinelHub config pointed at the CDSE endpoint."""
    try:
        from sentinelhub import SHConfig
    except ImportError as e:
        raise ImportError(
            "sentinelhub is not installed. Run: pip install sentinelhub"
        ) from e

    cfg = SHConfig()
    cfg.sh_client_id     = os.environ.get("CDSE_CLIENT_ID", "")
    cfg.sh_client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")
    cfg.sh_base_url      = "https://sh.dataspace.copernicus.eu"
    cfg.sh_token_url     = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )

    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise EnvironmentError(
            "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables are required "
            "for satellite image downloads. Register at https://dataspace.copernicus.eu "
            "and create an OAuth client, then set these variables (or add to a .env file)."
        )
    return cfg


def _fit_resolution(bbox, min_res: float = S2_RESOLUTION, max_px: int = MAX_PX):
    """
    Return the native 10 m resolution and pixel size for *bbox*.

    Resolution is locked to ``S2_RESOLUTION`` (10 m) to match the model's training
    ground-sampling — the app is scoped to AOIs that fit a single request, so no
    downsampling is ever needed. If an AOI is too large to fit in ``max_px`` at
    10 m we raise rather than coarsen, because segmenting at a coarser GSD than the
    model was trained on silently degrades every prediction.
    """
    from sentinelhub import bbox_to_dimensions
    w, h = bbox_to_dimensions(bbox, resolution=min_res)
    if max(w, h) > max_px:
        raise ValueError(
            f"AOI is {w}×{h} px at {min_res} m/px, exceeding the {max_px}px single-request "
            f"limit. This AOI would require tiling+mosaicking to stay at 10 m; the current "
            f"scope only supports AOIs that fit one request (Nairobi NP, Karura)."
        )
    return min_res, (int(w), int(h))


def _save_rgb_geotiff(array: np.ndarray, bounds_wgs84: tuple, out_path: Path) -> None:
    """
    Save float32 (H, W, 3) reflectance array as a uint8 GeoTIFF in TARGET_CRS.

    bounds_wgs84 : (west, south, east, north) in WGS-84
    """
    west, south, east, north = bounds_wgs84
    h, w = array.shape[:2]

    rgb = (np.clip(array, 0, 1) * 255).astype(np.uint8)

    tmp = out_path.with_suffix(".wgs84.tif")
    transform_wgs = from_bounds(west, south, east, north, width=w, height=h)

    with rasterio.open(
        tmp, "w",
        driver="GTiff", height=h, width=w,
        count=3, dtype="uint8",
        crs="EPSG:4326", transform=transform_wgs,
        compress="LZW",
    ) as dst:
        for b in range(3):
            dst.write(rgb[:, :, b], b + 1)

    # Reproject to TARGET_CRS (UTM 37S) for consistent downstream processing
    with rasterio.open(tmp) as src:
        t, tw, th = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=TARGET_CRS, transform=t, width=tw, height=th)

        with rasterio.open(out_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=t,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear,
                )
    tmp.unlink(missing_ok=True)


def download_s2(
    aoi_geom,
    date_start: str,
    date_end: str,
    max_cloud_pct: int,
    out_path: Path,
    progress_cb=None,
) -> Path:
    """
    Download a Sentinel-2 L2A least-cloud-cover composite for *aoi_geom*.

    Parameters
    ----------
    aoi_geom      : shapely geometry **in WGS-84 (EPSG:4326)**
    date_start    : ISO date string, e.g. "2023-01-01"
    date_end      : ISO date string, e.g. "2023-03-31"
    max_cloud_pct : scene cloud-cover filter (0–100)
    out_path      : where to write the output GeoTIFF (TARGET_CRS, uint8, 3-band)
    progress_cb   : optional callable(str) for status messages

    Returns
    -------
    out_path
    """
    from sentinelhub import (
        BBox, CRS, SentinelHubRequest,
        DataCollection, MimeType, MosaickingOrder,
    )

    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    minx, miny, maxx, maxy = aoi_geom.bounds
    bbox = BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)

    config = _sh_config()
    resolution, size = _fit_resolution(bbox)

    _log(f"Requesting {size[0]}×{size[1]} px at {resolution:.1f} m/px …")

    request = SentinelHubRequest(
        evalscript=_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    "s2l2a",
                    service_url=config.sh_base_url,
                ),
                time_interval=(date_start, date_end),
                mosaicking_order=MosaickingOrder.LEAST_CC,
                maxcc=max_cloud_pct / 100.0,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    _log("Downloading from Copernicus Data Space …")
    data = request.get_data()[0]   # (H, W, 3) float32

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_rgb_geotiff(data, (minx, miny, maxx, maxy), out_path)
    _log(f"Saved → {out_path.name}")

    return out_path
