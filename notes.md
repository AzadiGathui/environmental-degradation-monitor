# Environmental Degradation Monitor — Notes

---

## References

1. Training dataset: Sawant, Suraj; Garg, Rahul Dev; Meshram, Vishal; Mistry, Shrayank (2023), "Sen-2 LULC", Mendeley Data, V3, doi: 10.17632/f4ky6ks248.3 — [source](https://data.mendeley.com/datasets/f4ky6ks248/3)
2. Land-cover reference tiles: UNDP GeoHub 10m Annual Land Use Land Cover (9-class) — [source](https://geohub.data.undp.org/api/datasets/ffdee6dc36a27769c79d8ef6e9ccc7da)
3. Protected areas: UNEP-WCMC & IUCN World Database on Protected Areas (WDPA) polygons
4. Waterways: HOTOSM Kenya Waterways (HDX)
5. Satellite imagery: Copernicus Sentinel-2 L2A via Copernicus Data Space Ecosystem (CDSE)

---

## How This Project Works — Full Workflow

### The Big Picture

The goal is to detect land that transitioned from natural cover (forest, crops, bare ground) to built-up area **inside or near a protected area** between two points in time. To do this, the project:

1. Downloads two satellite images of the same place at different times (T1 and T2)
2. Runs a neural network on each image to classify every pixel by land-cover type
3. Compares the two classification maps — pixels that switched to "built up" are the change signal
4. Overlays that change signal with protected-area boundary polygons to flag encroachment

---

### Step 1 — Training Data

**What:** A labelled dataset of Sentinel-2 satellite image chips paired with land-cover masks.

**Source used here:** Mendeley Sen-2 LULC (India, ~150k chips, 7 classes, 64×64 px RGB). This was used as a starting point because it is publicly available and already formatted as chip+mask pairs.

**7 classes:** water, dense forest, sparse forest, barren land, built up, agriculture, fallow land.

**Why India data:** No comparable Kenya-specific chip dataset with pixel-level masks existed at the time. The India chips were used to build and validate the training pipeline. The intention is to fine-tune on Kenya chips once inference on Kenya imagery has been validated visually.

**Tool:** `train.ipynb` — loads image/mask pairs with TensorFlow `tf.data`, applies augmentation (flips, rotations), computes class weights for imbalanced classes, trains the U-Net, and checkpoints the best model.

---

### Step 2 — Model Architecture

**What:** A U-Net — the standard architecture for semantic segmentation of satellite imagery.

**Why U-Net:** It uses an encoder-decoder structure with skip connections. The encoder (series of convolution + pooling layers) compresses the image into features. The decoder (upsampling layers) reconstructs a full-resolution prediction map. Skip connections carry fine spatial detail from early encoder layers directly to the decoder, which matters because the edges of land-cover classes are important.

**Architecture built here:**
- 4-level encoder: 32 → 64 → 128 → 256 filters
- 3-level decoder with transposed convolutions and skip connections
- Output layer: 1×1 convolution with softmax → 7-class probability map per pixel
- ~1.9 million parameters

**Loss function:** Weighted sparse categorical cross-entropy + Dice loss. The weighting corrects for class imbalance (agriculture and built-up dominate the dataset; water is rare). Dice loss directly optimizes class overlap rather than per-pixel accuracy.

**Metric:** Mean IoU (Intersection over Union) across all 7 classes. Pixel accuracy is misleading on imbalanced data — a model that always predicts "agriculture" gets high accuracy but is useless.

**Current performance:** CPU baseline (`train.ipynb`, 1,200 chips) reached val mean IoU ~0.24. The full-dataset GPU run (`train_gpu.ipynb`, 149,600 chips) reaches **test mean IoU 0.49, built-up IoU 0.62** — this is the production model. On real Kenyan imagery, however, built-up IoU falls to ~0.06 (`evaluation.ipynb`), so Kenya fine-tuning remains the key next step.

**Tools:** `train.ipynb` (baseline, kept for comparison) and `train_gpu.ipynb` → production model `unet_sen2_lulc_best_gpu.keras`

---

### Step 3 — Reference Land-Cover Map (Labels for Kenya)

**What:** UNDP 10m Annual LULC GeoTIFFs covering Kenya, used as ground-truth labels for fine-tuning.

**Why:** These tiles provide pixel-level land-cover labels at the same 10m resolution as Sentinel-2. They use different class definitions (9 classes vs. the model's 7), but can be remapped. They cover all of Kenya across 4 UTM zone tiles (36M, 36N, 37M, 37N).

**What the notebooks do:**
- `prep.ipynb`: Reprojects the 4 tiles to a common CRS (UTM 37S), mosaics them, clips to Kenya, and provides the function `get_label_for_image(image_path)` that reprojects the LULC tiles to match any downloaded satellite chip's exact pixel grid. This is the mechanism for generating chip+label pairs for Kenya fine-tuning.

---

### Step 4 — Sentinel-2 Imagery Download

**What:** Downloading actual satellite images for the area of interest at two timestamps.

**Source:** Copernicus Data Space Ecosystem (CDSE) — the free EU satellite data platform. Sentinel-2 L2A products (atmospherically corrected surface reflectance) at 10m resolution.

**How it works:**
- The user selects an AOI (county or protected area) in the Streamlit app
- The app checks the local cache; if the image is not there, it calls the CDSE API
- CDSE returns a least-cloud-cover composite for the selected date window
- The image is saved as a GeoTIFF in UTM 37S (the common CRS for all rasters in this project)

**Bands downloaded:** B04 (Red), B03 (Green), B02 (Blue) — matches the RGB training data. NIR (B08) would improve performance but requires the model to be retrained on 4-band input.

**Fixed dates:** the early window is **2016** and the late window is **2025**, both
full-year LEAST_CC composites (a CDSE catalog probe confirmed 2016 has ≥34 low-cloud
scenes for both AOIs). Dates are not user-selectable.

**Pre-cached for demo:** the 2016 and 2025 composites for Nairobi National Park and
Karura Forest Reserve are stored in `data/demo/` and seeded into the cache on startup,
so the demo runs without CDSE credentials.

**Tool:** `app/download.py` — wraps the `sentinelhub` Python SDK pointed at the CDSE endpoint

---

### Step 5 — Inference (Segmenting the Image)

**What:** Running the trained model on a full-scene satellite image.

**The tiling problem:** The model was trained on 64×64 pixel chips. A full Nairobi scene is ~2,200×1,800 pixels. You cannot feed the whole image to the model at once.

**Solution — sliding window with overlap:**
1. Pad the image so its dimensions are divisible by the stride
2. Extract overlapping 64×64 chips with a stride of 48 pixels (16 px overlap on each side)
3. Run the model on batches of chips
4. Accumulate softmax probabilities in an array the same size as the full image
5. Where chips overlap, average the probabilities
6. Take `argmax` across the class dimension → one class label per pixel

**Output:** A single-band uint8 GeoTIFF where each pixel value is the class index (0–6).

**Tool:** `app/inference.py`

---

### Step 6 — Change Detection

**What:** Comparing the two segmentation masks to find pixels that transitioned to built-up.

**How it works:**
1. Load both masks and warp T2 onto T1's exact pixel grid (they may have different dimensions or CRS)
2. `change_mask = (T2 == built_up) AND (T1 != built_up)` — pixels that weren't built-up before but are now
3. Convert the change pixels to vector polygons using `rasterio.features.shapes`
4. Load WDPA protected area polygons and reproject to the same CRS
5. Run a geopandas `overlay` (intersection) between the change polygons and PA polygons
6. Sum the intersecting area per PA — this is the encroachment signal

**Output:** A dict with the change mask array, a GeoDataFrame of encroachment polygons per PA, and summary statistics (total changed area, affected PAs, per-class breakdown of what those pixels used to be).

**Tool:** `app/change_detection.py`

---

### Step 7 — Streamlit App

**What:** A browser-based interface for end users.

**User flow:**
1. User selects AOI (county or protected area) from a dropdown
2. User sets T1 and T2 date ranges and cloud cover threshold
3. Clicks Run Analysis
4. App downloads imagery (if not cached), runs inference on both timestamps, runs change detection
5. Displays an interactive folium map with 7 toggleable layers:
   - RGB true colour ×2 (before/after)
   - Segmentation mask ×2 (before/after, coloured by class)
   - Encroachment overlay (red pixels = new built-up)
   - Protected area boundaries (green outlines)
   - County boundaries (grey dashed)
6. Shows summary metrics (changed area, encroachment in PAs, affected sites)
7. Expandable tables with per-class breakdown and per-PA encroachment areas

**Tools:** `streamlit`, `streamlit-folium`, `folium`, `PIL` for raster→PNG conversion, `base64` for embedding images in the map, `app/viz.py`

---

## Tools and Libraries Used

| Tool | What It Does in This Project |
|---|---|
| **TensorFlow / Keras** | Build, train, and run inference with the U-Net model |
| **rasterio** | Read/write GeoTIFFs, reproject rasters, window-based reading, vectorising rasters |
| **geopandas** | Load/reproject vector data (counties, PAs, waterways), spatial intersection |
| **shapely** | Geometry operations (bounds, intersection, buffer) |
| **sentinelhub** | Download Sentinel-2 imagery from Copernicus Data Space Ecosystem |
| **numpy** | Array operations for tiling, stitching, change mask computation |
| **PIL (Pillow)** | Convert numpy arrays to PNG images for map overlays |
| **folium** | Interactive web map; ImageOverlay for raster layers, GeoJson for vector layers |
| **streamlit** | Browser UI framework — widgets, session state, spinner, cache decorators |
| **streamlit-folium** | Embed folium maps inside Streamlit |
| **python-dotenv** | Load CDSE credentials from a `.env` file |

---

## How to Recreate This Project From Scratch

This is a learning path. Each block builds on the previous one.

### Block 1 — Python and Geospatial Basics (1–2 weeks)

Learn to read and write raster and vector geospatial data.

- **Rasterio tutorial:** https://rasterio.readthedocs.io/en/latest/quickstart.html
  - Open a GeoTIFF, read a band as a numpy array, inspect CRS and transform
  - Reproject a raster to a different CRS using `rasterio.warp.reproject`
  - Clip a raster to a polygon using `rasterio.mask.mask`
  - Write a result back to a GeoTIFF

- **GeoPandas tutorial:** https://geopandas.org/en/stable/getting_started/introduction.html
  - Load a shapefile or GeoJSON
  - Filter rows, reproject, plot
  - Spatial join and overlay (intersection between two GeoDataFrames)

- **Practice exercise:** Download a free GeoTIFF of any country's land cover (e.g. from UNDP GeoHub), clip it to one province, count pixels by class value.

### Block 2 — Satellite Imagery (1 week)

Understand what Sentinel-2 is and how to get it.

- Read: https://sentinel.esa.int/web/sentinel/missions/sentinel-2 — what the bands mean
- Register at https://dataspace.copernicus.eu — this gives you access to the full archive
- Try downloading a small area manually using the Copernicus Browser (no code needed, just to see what a GeoTIFF looks like)
- Then replicate it in Python using the `sentinelhub` library:
  - https://sentinelhub-py.readthedocs.io/en/latest/
  - Write a function that takes a bounding box and a date range and returns a numpy array of RGB values

### Block 3 — Neural Networks for Image Segmentation (2–3 weeks)

Build and train the U-Net from scratch.

- **Prerequisite:** Be comfortable with numpy arrays and basic Python. Know what a matrix multiplication is.
- **Start with:** fast.ai Practical Deep Learning course (free, https://course.fast.ai) — covers the intuition before the maths
- **Then learn:** Keras/TensorFlow with the official image segmentation guide: https://www.tensorflow.org/tutorials/images/segmentation
- **Build the U-Net:** The architecture in `train.ipynb` is the reference implementation. Read it line by line.
  - Understand what `Conv2D`, `MaxPooling2D`, and `Conv2DTranspose` do spatially
  - Understand what `concatenate` (skip connection) does and why it matters
  - Understand `softmax` output and `sparse_categorical_crossentropy` loss
- **Understand the metrics:** Why IoU is better than accuracy for segmentation. Compute it manually for one class on a small example.
- **Train it:** Run `train.ipynb` on the Mendeley dataset. Watch the val_mean_iou curve. Understand when it stops improving and why.

### Block 4 — Tiling and Inference on Full Scenes (3–5 days)

Bridge the gap between a model trained on 64×64 chips and a full satellite scene.

- Understand the sliding window concept: a fixed-size window moves across the image, the model predicts each window, predictions are averaged where windows overlap
- Implement `_pad`, `_tile_positions`, and the accumulator loop in `app/inference.py` yourself from scratch
- Test with a known image: segment a small image, verify the output has the right shape and sensible class predictions

### Block 5 — Change Detection (2–3 days)

Compute a difference between two segmentation maps.

- Learn the basics of boolean array masking in numpy: `mask = (arr_a == 4) & (arr_b != 4)`
- Learn `rasterio.features.shapes` to convert a boolean raster to vector polygons
- Learn `geopandas.overlay` with `how='intersection'` to find where those polygons overlap a reference layer (e.g. PA boundaries)
- Read the full `app/change_detection.py` and understand every step

### Block 6 — Streamlit App (1 week)

Build the browser interface.

- **Streamlit quickstart:** https://docs.streamlit.io/get-started — start with a "Hello World" app
- Learn `st.sidebar`, `st.selectbox`, `st.date_input`, `st.button`, `st.spinner`, `st.metric`, `st.expander`
- Learn `@st.cache_data` and `@st.cache_resource` — critical for performance (caches model loading and geodataframe reads)
- Learn `st.session_state` — how to preserve results between reruns triggered by widget interactions
- Learn `streamlit-folium`: `st_folium(map_object)` renders a folium map inline
- Understand `folium.raster_layers.ImageOverlay` — how to convert a numpy array to a base64 PNG and overlay it on the map with lat/lon bounds

### Block 7 — Putting It All Together (ongoing)

- Add fine-tuning on Kenya chips (use `get_label_for_image` from `prep.ipynb` to generate labels for downloaded Nairobi imagery, then run a few epochs of training with the saved model as the starting point)
- Add hyperparameter tuning with Optuna (`pip install optuna`)
- Deploy to Streamlit Cloud (push to GitHub, connect at https://share.streamlit.io, add CDSE credentials as Secrets)

---

## Demo Runbook

### Prerequisites

- The `tf-gpu` Python environment (`/home/hp/tf-gpu`) with all packages installed
  (`sentinelhub`, `streamlit`, `streamlit-folium`, `python-dotenv`, plus the geo/ML stack)
- The repo at `06_capstone/environmental-degradation-monitor`
- No CDSE credentials needed for the two bundled AOIs (imagery pre-cached in `data/demo/`)

### Step 1 — Start the app

```bash
cd 06_capstone/environmental-degradation-monitor
/home/hp/tf-gpu/bin/streamlit run streamlit_app.py
```

The browser opens at http://localhost:8501.

### Step 2 — Run the demo (no download needed)

The app seeds the pre-downloaded 2016/2025 imagery from `data/demo/` into the cache
on startup, so both AOIs work without credentials.

1. Pick a protected area: **Nairobi National Park** or **Karura Forest Reserve**
2. Click **▶ Run Analysis** (dates are fixed at 2016 → 2025; there is no date picker)
3. First run segments both scenes (CPU: ~30 s Nairobi NP, ~5 s Karura); later runs load from cache
4. Read the metric strip: **protected land lost (km²)**, inside-PA vs. in-100 m-buffer, total new built-up
5. Toggle map layers: **Mask (early/late)**, **Encroachment** (red = new built-up), **Protected areas** (green outline)
6. Expand "Change breakdown" and "Affected protected areas" for detail

Reference figures from the verified run:

| AOI | Protected land lost | inside PA | in 100 m buffer |
|---|---|---|---|
| Nairobi National Park | 2.03 km² | 1.92 | 0.11 |
| Karura Forest Reserve | 0.19 km² | 0.13 | 0.06 |

> Read these as model estimates, not ground truth — see `evaluation.ipynb` (built-up
> IoU 0.62 India → ~0.06 Kenya).

### Step 3 — Add a new protected area (requires CDSE credentials)

1. Copy `.env.example` to `.env` and fill in your `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET`
2. Add an entry to `TARGET_PAS` in `app/config.py` (must fit one 10 m request, ≤2500 px/axis)
3. Click **▶ Run Analysis** — imagery downloads on demand and is cached

### Step 4 — Caching

Imagery and masks live in `data/cache/`, keyed by a hash of (AOI, date window, type).
Re-running the same AOI is instant. Delete `data/cache/` to force a fresh run;
`data/demo/` preserves downloaded imagery across cache clears.

### What the output means

| Metric | Interpretation |
|---|---|
| Changed to built-up | Total area that switched from any class to "built up" across the whole AOI |
| Encroachment in PAs | Subset of that change that falls inside a WDPA-listed protected area — the alert signal |
| PA sites affected | Number of distinct protected area polygons with at least one encroachment pixel |
| Built-up growth (+px) | Net change in total built-up pixel count T1→T2 (can be negative if model disagrees between runs) |

### Known limitations for the demo

- The model was trained on India RGB chips, so class predictions for Kenya will have errors. The built-up class tends to be reasonably discriminated (IoU 0.25 on India test set), but forest sub-classes are weaker.
- Cloud cover in the pre-downloaded 2023 image is low; the 2025 image covers the dry season. In practice, cloud-affected pixels would be mis-classified — the model has no cloud-detection output.
- The encroachment numbers are model predictions, not verified ground truth. Visual spot-checking against the RGB overlay is the validation method for now.
