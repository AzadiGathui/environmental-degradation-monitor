# Environmental Degradation Monitor 

## Project Overview

A change-detection tool that uses satellite imagery and semantic segmentation to **quantify how much protected land was lost to built-up encroachment between 2016 and 2025** in Kenya. For a selected protected area the app downloads Sentinel-2 imagery at an early date (**2016**) and a late date (**2025**), segments both with a U-Net, and reports the area (km²) of protected land — **inside the boundary and within a 100 m buffer** — that transitioned to built-up. Newly built-up land is highlighted in red on the map.

**Scope of this build:** protected areas, not counties; two fixed comparison years (2016 → 2025, not user-selectable); native **10 m** resolution to match the model's training; and two bundled AOIs — **Nairobi National Park** and **Karura Forest Reserve** — that both fit a single 10 m Sentinel-2 request. Additional protected areas can be added once these are validated.

## Scope

**In scope (MVP)**
- Multi-class semantic segmentation of satellite imagery (built-up, vegetation, water, bare soil, cropland)
- Change detection between a fixed 2016 (early) and 2025 (late) composite
- Intersection of detected change with protected-area boundaries **and a 100 m buffer** around them
- A headline metric: **km² of protected land converted to built-up, 2016 → 2025**
- Visual output showing before/after masks and newly built-up land in red
- A simple browser-based interface: pick a protected area, click Run (no date picking)

**Out of scope (MVP)**
- User-selectable dates (the tool answers one fixed question: 2016 → 2025)
- Protected areas too large to fit a single 10 m request (Mau complex, Maasai Mara) — these need download tiling/mosaicking, deferred
- Integration with external workflows

**Potential future extensions**
- Temporal trend analysis across multiple timestamps
- Confidence scoring and uncertainty quantification per alert
- All-weather detection
- Automated retraining pipeline to handle model drift
- API integration

## Data Sources

| Layer | Source | Notes |
|---|---|---|
| Satellite imagery | Copernicus / Sentinel-2 | Multispectral bands; two timestamps per target area |
| Labeled training data | Radiant Earth LandCoverNet, ESA CCI Land Cover (Africa), RCMRD Geoportal | Kenya-specific chips preferred to reduce domain shift |
| Protected areas | Existing shapefiles | Starting point; riparian zones approximated from water body proximity |
| Riparian zones | Derived from river shapefiles + buffer | Approximate band on either side of water bodies |

Training data selection must account for geographic leakage — chips from the target test county should be held out of training and validation entirely.

### Sources (Updated):
- downloaded 4 lulc shapefiles from this ([undp dataset '10m Annual Land Use Land Cover (9-class'](https://geohub.data.undp.org/api/datasets/ffdee6dc36a27769c79d8ef6e9ccc7da))

## Methodology

### 1. Data Preparation
- Download labeled Sentinel-2 chips from a Kenya-relevant source
- Exclude chips from the target test county to prevent leakage
- Split remaining data into train / validation sets
- Download two timestamps of unlabeled Sentinel-2 imagery for the test county
- Tile imagery into uniform chip sizes matching the training data
- Apply spectral preprocessing (normalisation, cloud masking, band selection)

### 2. Model Training
- Architecture: U-Net (encoder-decoder with skip connections) as the primary model
- Input: Sentinel-2 multispectral chips
- Output: pixel-wise segmentation mask (binary or multi-class)
- Loss function (TBD)
- Optional comparison model: DeepLabv3 or a simpler fully convolutional network, to document model selection rationale

### 3. Inference & Change Detection
- Run inference on both timestamps for the test county
- Generate segmentation masks for each timestamp
- Subtract masks to isolate pixels that transitioned from natural to built-up
- Intersect change pixels with protected area and riparian zone shapefiles

### 4. Alerting
- Flag intersecting regions as encroachment events
- Output: map overlay showing flagged zones with before/after context

## Deployment Plan

The MVP is deployed as a lightweight browser application using Streamlit (or equivalent).

**User flow:**
1. User selects a protected area (Nairobi National Park or Karura Forest Reserve)
2. Application retrieves cached — or downloads on demand — the 2016 and 2025 Sentinel-2 composites for that PA at 10 m
3. Inference runs on both timestamps; change detection is computed against the PA + 100 m buffer
4. Output displays: the protected-land-lost metric, side-by-side segmentation masks, and newly built-up land highlighted in red against the protected-area boundary

**Infrastructure (MVP):**
- Boundary layers (counties, PAs, 100 m buffer) loaded as static GeoJSON assets
- Satellite imagery pre-cached in `data/demo/` (credential-free demo) or downloaded on demand via the Copernicus API
- Streamlit app hosted on Streamlit Cloud
- Model weights: `unet_sen2_lulc_best_gpu.keras` (full-dataset GPU run, test mean IoU 0.49)

## Limitations & Known Risks

- **Domain shift is the dominant risk.** The model is trained on India RGB chips; `evaluation.ipynb` measures built-up IoU dropping from 0.62 (India test) to ~0.06 on real Kenyan imagery. All encroachment km² are model estimates, not ground truth. Kenya fine-tuning (via `prep.ipynb`'s `get_label_for_image`) is the highest-value next step.
- Cloud cover in Sentinel-2 imagery can be misclassified as built-up/bare (no cloud mask in the RGB pipeline yet).
- Change is a difference of two independent noisy segmentations, which amplifies noise; small signals can be swamped.
- Only AOIs that fit a single 10 m request are supported; larger PAs (Mau, Maasai Mara) need download tiling and are deferred.

## Reproducing this build

Run order (see per-notebook write-ups in `assets/<notebook>/`):

1. `preprocessing.ipynb` — county GeoJSON + **`pa_100m_buffer.geojson`**
2. `prep.ipynb` — Kenya reference LULC mosaic + `get_label_for_image` (only needed for future fine-tuning)
3. `train_gpu.ipynb` — production model (`unet_sen2_lulc_best_gpu.keras`); `train.ipynb` is the CPU baseline kept for comparison
4. `evaluation.ipynb` — train (India) vs. inference (Kenya) performance
5. `streamlit_app.py` — the app (`streamlit run streamlit_app.py`)

## Future extensions

- Fine-tune on Kenyan chips to close the domain-shift gap
- Feed the U-Net features into an LSTM/GRU for multi-date temporal trend analysis
- Cloud masking from the Sentinel-2 Scene Classification band
- Download tiling + mosaicking to support large PAs (Mau complex, Maasai Mara) at 10 m