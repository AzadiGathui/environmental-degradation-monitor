"""
Segment a Sentinel-2 RGB GeoTIFF using the trained U-Net.

Pipeline
--------
1. Read the GeoTIFF and normalise to float32 [0, 1]
2. Reproject to TARGET_CRS (UTM 37S) if needed
3. Tile into overlapping 64×64 chips (CHIP_STRIDE=48 → 16 px overlap)
4. Run batched model inference
5. Average overlapping predictions and take argmax → class mask
6. Write mask GeoTIFF in the same CRS/transform as the (normalised) input
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from app.config import (
    IMG_SIZE, CHIP_STRIDE, N_CLASSES, N_CHANNELS,
    INFER_BATCH, MODEL_PATH, TARGET_CRS,
)


@lru_cache(maxsize=1)
def load_model():
    """Load the Keras model once and cache it in-process."""
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def _normalise_to_utm(src_path: Path) -> tuple[np.ndarray, dict]:
    """
    Read a GeoTIFF, take the first 3 bands, reproject to TARGET_CRS, and
    return (rgb_uint8 H×W×3, rasterio profile).
    """
    with rasterio.open(src_path) as src:
        if str(src.crs) == TARGET_CRS or src.crs.to_string() == TARGET_CRS:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)   # H×W×3
            profile = src.profile.copy()
            return rgb, profile

        # Reproject to TARGET_CRS
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=TARGET_CRS, transform=transform,
                       width=width, height=height, count=3, dtype="uint8")

        bands = []
        for band_idx in range(1, min(src.count, 3) + 1):
            dst_band = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst_band,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
            )
            bands.append(dst_band)

        # If source has < 3 bands (e.g. grayscale), repeat the last band
        while len(bands) < 3:
            bands.append(bands[-1])

        rgb = np.stack(bands, axis=-1)   # H×W×3
        return rgb, profile


def _pad(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Pad H and W so every chip fits without out-of-bounds reads.

    Padded size along each axis:
        n_chips = ceil((size - IMG_SIZE) / CHIP_STRIDE) + 1
        padded  = (n_chips - 1) * CHIP_STRIDE + IMG_SIZE
    """
    H, W = arr.shape[:2]

    def _padded_size(size):
        n = math.ceil(max(size - IMG_SIZE, 0) / CHIP_STRIDE) + 1
        return (n - 1) * CHIP_STRIDE + IMG_SIZE

    pH = _padded_size(H)
    pW = _padded_size(W)
    pad_h = pH - H
    pad_w = pW - W

    if arr.ndim == 3:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")

    return padded, (H, W)


def _tile_positions(H: int, W: int) -> list[tuple[int, int]]:
    """Return (row, col) top-left corners for all overlapping chips."""
    positions = []
    r = 0
    while True:
        c = 0
        while True:
            positions.append((r, c))
            if c + IMG_SIZE >= W:
                break
            c += CHIP_STRIDE
        if r + IMG_SIZE >= H:
            break
        r += CHIP_STRIDE
    return positions


def predict_scene(
    rgb_path: Path,
    out_mask_path: Path,
    progress_cb=None,
) -> Path:
    """
    Segment a Sentinel-2 GeoTIFF and write a single-band uint8 mask GeoTIFF.

    Mask pixel values are 0-indexed class indices (0=water … 6=fallow land).

    Parameters
    ----------
    rgb_path      : input GeoTIFF (any CRS, ≥3 bands, uint8)
    out_mask_path : where to write the output mask GeoTIFF
    progress_cb   : optional callable(str) for status messages
    """
    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    _log("Loading model …")
    model = load_model()

    _log("Reading and reprojecting image …")
    rgb, profile = _normalise_to_utm(rgb_path)
    H_orig, W_orig = rgb.shape[:2]

    # Normalise to [0, 1]
    img = rgb.astype(np.float32) / 255.0

    # Pad so chip positions are clean
    img_pad, (H_orig, W_orig) = _pad(img)
    pH, pW = img_pad.shape[:2]

    positions = _tile_positions(pH, pW)
    n_chips = len(positions)
    _log(f"Running inference on {n_chips} chips …")

    # Accumulators for averaged softmax predictions
    pred_sum   = np.zeros((pH, pW, N_CLASSES), dtype=np.float32)
    pred_count = np.zeros((pH, pW),            dtype=np.float32)

    # Batch inference
    for batch_start in range(0, n_chips, INFER_BATCH):
        batch_pos = positions[batch_start: batch_start + INFER_BATCH]
        chips = np.stack([
            img_pad[r: r + IMG_SIZE, c: c + IMG_SIZE]
            for r, c in batch_pos
        ])                                                  # (B, 64, 64, 3)

        preds = model.predict(chips, verbose=0)             # (B, 64, 64, N_CLASSES)

        for (r, c), pred in zip(batch_pos, preds):
            pred_sum  [r: r + IMG_SIZE, c: c + IMG_SIZE] += pred
            pred_count[r: r + IMG_SIZE, c: c + IMG_SIZE] += 1.0

        done = min(batch_start + INFER_BATCH, n_chips)
        _log(f"  {done}/{n_chips} chips …")

    # Average overlapping predictions and take argmax
    pred_avg = pred_sum / np.maximum(pred_count[..., None], 1e-6)
    mask_pad = np.argmax(pred_avg, axis=-1).astype(np.uint8)  # (pH, pW)

    # Trim padding back to original size
    mask = mask_pad[:H_orig, :W_orig]

    # Write mask GeoTIFF reusing the (now-UTM) profile
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", compress="LZW", nodata=None)
    with rasterio.open(out_mask_path, "w", **profile) as dst:
        dst.write(mask, 1)

    _log(f"Mask saved → {out_mask_path.name}")
    return out_mask_path
