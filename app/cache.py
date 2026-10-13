"""Lightweight file-based cache for downloaded imagery and computed masks."""
import hashlib
import shutil
from pathlib import Path

from app.config import DATA_CACHE, DATA_RAW

DATA_CACHE.mkdir(parents=True, exist_ok=True)

# Durable location for pre-downloaded demo imagery (survives cache clears; can be
# shipped so the bundled AOIs work without CDSE credentials).
DEMO_DIR = DATA_RAW.parent / "demo"


def _key(aoi_name: str, date_start: str, date_end: str, tag: str) -> str:
    raw = f"{aoi_name}|{date_start}|{date_end}|{tag}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def cache_path(aoi_name: str, date_start: str, date_end: str, tag: str) -> Path:
    """
    Return the expected cache file path for a given (AOI, date range, tag) triple.

    tag : one of "rgb", "mask" — distinguishes the imagery from the model output
    """
    return DATA_CACHE / f"{_key(aoi_name, date_start, date_end, tag)}.tif"


def is_cached(aoi_name: str, date_start: str, date_end: str, tag: str) -> bool:
    return cache_path(aoi_name, date_start, date_end, tag).exists()


def demo_path(aoi_label: str, window: tuple, tag: str = "rgb") -> Path:
    """
    Durable filename for a pre-downloaded demo raster, e.g.
    ``data/demo/Nairobi National Park__2016__rgb.tif``.
    """
    safe = aoi_label.replace("/", "-")
    return DEMO_DIR / f"{safe}__{window[0][:4]}__{tag}.tif"


def seed_demo_cache(target_pas: list, early_window: tuple, late_window: tuple) -> None:
    """
    Copy any pre-downloaded demo imagery from ``data/demo/`` into the runtime
    cache so the bundled AOIs appear as cache hits on first run — no CDSE
    credentials needed. Missing demo files are simply skipped (the app then
    downloads on demand if credentials are available).
    """
    if not DEMO_DIR.exists():
        return
    for spec in target_pas:
        label = spec["label"] if isinstance(spec, dict) else spec
        for window in (early_window, late_window):
            src = demo_path(label, window, "rgb")
            dst = cache_path(label, *window, "rgb")
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
