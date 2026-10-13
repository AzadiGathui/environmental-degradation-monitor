from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
DATA_RAW   = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"

# Production model — full-dataset GPU run (test mean IoU 0.49, built-up IoU 0.62).
# The CPU baseline (unet_sen2_lulc_best.keras, IoU ~0.24) is kept only for the
# train.ipynb-vs-train_gpu.ipynb comparison; it is NOT used for inference.
MODEL_PATH = ROOT / "unet_sen2_lulc_best_gpu.keras"

COUNTIES_GEOJSON = ROOT / "ke.geojson"          # static backdrop only (not selectable)
PA_GEOJSON = (
    DATA_RAW
    / "unep-wcmc and iucn protected areas"
    / "protected_conserved_areas_wdpca_polygons.geojson"
)
# PA polygons grown by a 100 m buffer (built by preprocessing.ipynb). Encroachment
# is measured against PA-plus-buffer so boundary breaches are captured too.
PA_BUFFER_GEOJSON = DATA_RAW / "pa_100m_buffer.geojson"
PA_BUFFER_M = 100   # buffer width in metres

# --- Projections ---
TARGET_CRS  = "EPSG:32737"   # UTM zone 37S — all rasters normalised to this
WGS84_CRS   = "EPSG:4326"

# --- Model ---
IMG_SIZE     = 64
N_CHANNELS   = 3
N_CLASSES    = 7
CHIP_STRIDE  = 48   # 16 px overlap on each axis when tiling
INFER_BATCH  = 32
BUILT_UP_IDX = 4    # 0-indexed class for "built up"

CLASS_NAMES = [
    "water",
    "dense forest",
    "sparse forest",
    "barren land",
    "built up",
    "agriculture",
    "fallow land",
]

# RGBA (0-255) used when painting mask overlays
CLASS_COLORS_RGBA = {
    0: (66,  135, 245, 180),   # water         – blue
    1: (26,  122,  26, 180),   # dense forest  – dark green
    2: (92,  184,  92, 180),   # sparse forest – medium green
    3: (200, 169, 110, 180),   # barren land   – tan
    4: (217,  83,  79, 200),   # built up      – red
    5: (240, 173,  78, 180),   # agriculture   – amber
    6: (170, 212, 170, 180),   # fallow land   – pale green
}

CLASS_HEX = {k: "#{:02x}{:02x}{:02x}".format(*v[:3]) for k, v in CLASS_COLORS_RGBA.items()}

# Red overlay for change / encroachment pixels
CHANGE_RGBA = (255, 0, 0, 200)

# --- Area-of-interest scope ---------------------------------------------------
# The app is intentionally scoped to two small protected areas that both fit in a
# single 10 m Sentinel-2 request (no downsampling), so inference runs at exactly
# the 10 m ground-sampling the model was trained on:
#   Nairobi National Park  126 km²  2261×1484 px @10 m
#   Karura Forest Reserve   10 km²   769×346  px @10 m
# `match`/`desig` locate the polygon in the WDPA layer (the WDPA `name` for the
# park is simply "Nairobi", and there are duplicate rows — dedupe by site_id).
TARGET_PAS = [
    {"key": "nairobi_np", "label": "Nairobi National Park",
     "match": "Nairobi", "desig": "National Park"},
    {"key": "karura",     "label": "Karura Forest Reserve",
     "match": "Karura",   "desig": "Forest Reserve"},
]

# --- Sentinel-2 download ---
S2_RESOLUTION = 10      # metres (native S2 10-m bands) — never coarsened; see download.py
MAX_PX        = 2500    # Copernicus Process API per-request limit
MAX_CLOUD_PCT = 20      # scene cloud filter (fixed; no longer user-adjustable)

# Fixed early / late comparison windows. Dates are NOT user-selectable: the whole
# point of the tool is a consistent 2016→2025 change signal, so a full-year window
# is used at each end and LEAST_CC mosaicking picks the least-cloudy pixels.
# CDSE catalog probe (both AOIs) confirmed 2016 has 34 scenes with a 0.0 % cloud
# minimum, so 2016 is retained as the early anchor rather than bumped to 2017/2018.
EARLY_WINDOW = ("2016-01-01", "2016-12-31")
LATE_WINDOW  = ("2025-01-01", "2025-12-31")
EARLY_YEAR = 2016
LATE_YEAR  = 2025
