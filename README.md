# Environmental Degradation Monitor 

## Project Overview

A change-detection tool that uses satellite imagery and semantic segmentation to identify encroachment onto protected areas in Kenya. The system compares segmentation masks from two points in time, intersects detected changes with known protected area boundaries, and surfaces alerts where natural land has transitioned to built-up land within a protected zone.

The initial scope targets a single county (Kajiado or Nairobi TBD).

## Scope

**In scope (MVP)**
- Binary or multi-class semantic segmentation of satellite imagery (built-up, vegetation, water, bare soil, cropland)
- Change detection between two user-selected timestamps
- Intersection of detected change with protected area and riparian zone shapefiles
- Visual output showing before/after masks and flagged encroachment zones
- A simple browser-based interface for region and date selection

**Out of scope (MVP)**
- Real-time or automated satellite data ingestion
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
1. User selects a county
2. Application downloads or retrieves cached Sentinel-2 imagery for the selected area and dates (TBD how to set dates)
3. Inference runs on both timestamps; change detection is computed
5. Output displays: side-by-side segmentation masks, change overlay, and flagged encroachment zones highlighted against protected area boundaries

**Infrastructure (MVP):**
- Shapefiles loaded as static assets
- Satellite imagery downloaded on-demand via Copernicus API or pre-cached for the target county
- Streamlit app hosted Streamlit Cloud
- Model weights (TBD)

## Limitations & Known Risks

- Model trained on one county may generalise imperfectly to others
- Cloud cover in Sentinel-2 imagery can interfere with accuracy
- Model utility decreases over time as construction patterns or vegetation conditions change
- No ground truth verification layer in MVP (visual spot-checks are the primary validation method for now)