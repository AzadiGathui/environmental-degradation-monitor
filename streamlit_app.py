"""
Environmental Degradation Monitor — Streamlit UI

Goal
----
Quantify protected land converted to built-up between a fixed EARLY year (2016)
and a fixed LATE year (2025), for a small set of protected areas.

User flow
---------
1. Pick a protected area (Nairobi National Park or Karura Forest Reserve)
2. Click "Run Analysis"
3. App downloads 2016 & 2025 imagery (if not cached), segments both, computes the
   change, and reports how much protected land (inside the PA and within its 100 m
   buffer) became built-up — with the new built-up highlighted in red on the map.

Dates are fixed, not user-selectable: the tool answers one question (2016→2025),
so a full-year LEAST_CC composite is used at each end.
"""
from __future__ import annotations

import geopandas as gpd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

# Load .env for CDSE credentials (local dev; Streamlit Cloud uses Secrets)
load_dotenv()

from app.cache import cache_path, seed_demo_cache
from app.config import (
    PA_GEOJSON, TARGET_PAS,
    EARLY_WINDOW, LATE_WINDOW, EARLY_YEAR, LATE_YEAR,
    MAX_CLOUD_PCT,
)
from app.download import download_s2
from app.inference import predict_scene
from app.change_detection import detect_change
from app.viz import build_map

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Environmental Degradation Monitor",
    page_icon="🌍",
    layout="wide",
)

# Seed any pre-downloaded demo imagery into the cache so the bundled AOIs work
# without CDSE credentials.
seed_demo_cache(TARGET_PAS, EARLY_WINDOW, LATE_WINDOW)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_target_aois() -> dict:
    """
    Resolve the configured TARGET_PAS to {label: shapely geometry (WGS-84)}.

    The WDPA layer has duplicate rows (e.g. two identical "Nairobi" National Park
    polygons sharing site_id 761), so we dedupe and dissolve per target.
    """
    pa = gpd.read_file(PA_GEOJSON).to_crs("EPSG:4326")
    aois = {}
    for spec in TARGET_PAS:
        sub = pa[pa["name"].str.contains(spec["match"], case=False, na=False)]
        if spec.get("desig") and (sub["desig_eng"] == spec["desig"]).any():
            sub = sub[sub["desig_eng"] == spec["desig"]]
        if len(sub):
            aois[spec["label"]] = sub.geometry.union_all()
    return aois


# ---------------------------------------------------------------------------
# Sidebar — AOI selection only (dates are fixed)
# ---------------------------------------------------------------------------
st.sidebar.title("🌍 Degradation Monitor")
st.sidebar.markdown("**Step 1 — Choose a protected area**")

aois = load_target_aois()
aoi_name = st.sidebar.selectbox("Protected area", list(aois.keys()))
aoi_geom = aois[aoi_name]

st.sidebar.divider()
st.sidebar.markdown(
    f"**Step 2 — Compare fixed periods**\n\n"
    f"Early: **{EARLY_YEAR}** &nbsp;→&nbsp; Late: **{LATE_YEAR}**  \n"
    f"<span style='color:#888;font-size:12px'>Least-cloud annual composites "
    f"(≤{MAX_CLOUD_PCT}% cloud), 10 m resolution.</span>",
    unsafe_allow_html=True,
)

st.sidebar.divider()
st.sidebar.markdown("**Step 3 — Run**")
run_btn = st.sidebar.button("▶ Run Analysis", use_container_width=True, type="primary")

st.sidebar.divider()
st.sidebar.caption(
    "Imagery: Sentinel-2 L2A via Copernicus Data Space. "
    "Protected areas: UNEP-WCMC / IUCN WDPA (+100 m buffer)."
)

# ---------------------------------------------------------------------------
# Helpers: ensure imagery and masks are available
# ---------------------------------------------------------------------------

def _ensure_rgb(aoi_name, aoi_geom, window, label, log):
    """Download RGB for the fixed window if not cached; return path."""
    t_start, t_end = window
    rgb_path = cache_path(aoi_name, t_start, t_end, "rgb")
    if rgb_path.exists():
        log(f"{label} imagery found in cache.")
        return rgb_path

    log(f"Downloading {label} ({t_start[:4]}) imagery for {aoi_name} …")
    try:
        download_s2(
            aoi_geom=aoi_geom,
            date_start=t_start,
            date_end=t_end,
            max_cloud_pct=MAX_CLOUD_PCT,
            out_path=rgb_path,
            progress_cb=log,
        )
    except EnvironmentError as e:
        st.error(
            f"**CDSE credentials missing.** {e}\n\n"
            "Set `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` in a `.env` file or "
            "Streamlit secrets, or ship pre-downloaded imagery in `data/demo/`."
        )
        st.stop()
    return rgb_path


def _ensure_mask(aoi_name, window, rgb_path, label, log):
    """Run inference if mask not cached; return path."""
    t_start, t_end = window
    mask_path = cache_path(aoi_name, t_start, t_end, "mask")
    if mask_path.exists():
        log(f"{label} mask found in cache.")
        return mask_path

    log(f"Segmenting {label} image …")
    predict_scene(rgb_path, mask_path, progress_cb=log)
    return mask_path


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Environmental Degradation Monitor")
st.caption(
    f"AOI: **{aoi_name}** · Early: {EARLY_YEAR} → Late: {LATE_YEAR} · "
    "Metric: protected land (PA + 100 m buffer) converted to built-up"
)

# Initialise session state
if "result" not in st.session_state:
    st.session_state.result = None
if "last_key" not in st.session_state:
    st.session_state.last_key = None

run_key = f"{aoi_name}|{EARLY_YEAR}|{LATE_YEAR}"

if run_btn:
    st.session_state.result = None
    st.session_state.last_key = run_key
    log_box = st.empty()
    messages = []

    def log(msg):
        messages.append(msg)
        log_box.info("\n".join(f"• {m}" for m in messages[-6:]))

    with st.spinner("Running analysis …"):
        rgb1_path  = _ensure_rgb(aoi_name, aoi_geom, EARLY_WINDOW, "Early", log)
        rgb2_path  = _ensure_rgb(aoi_name, aoi_geom, LATE_WINDOW,  "Late",  log)
        mask1_path = _ensure_mask(aoi_name, EARLY_WINDOW, rgb1_path, "Early", log)
        mask2_path = _ensure_mask(aoi_name, LATE_WINDOW,  rgb2_path, "Late",  log)

        log("Computing change …")
        result = detect_change(mask1_path, mask2_path, progress_cb=log)
        result["rgb1_path"]  = rgb1_path
        result["rgb2_path"]  = rgb2_path
        result["mask1_path"] = mask1_path
        result["mask2_path"] = mask2_path

        st.session_state.result = result

    log_box.empty()
    st.success("Analysis complete.")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

result = st.session_state.result

if result is None:
    st.info(
        "Choose a protected area, then click **▶ Run Analysis** to compare "
        f"{EARLY_YEAR} with {LATE_YEAR}."
    )
    import folium
    cx, cy = aoi_geom.centroid.x, aoi_geom.centroid.y
    fmap = folium.Map(location=[cy, cx], zoom_start=12, tiles="CartoDB positron")
    folium.GeoJson(
        aoi_geom.__geo_interface__,
        style_function=lambda _: {"fillColor": "none", "color": "#2ecc71", "weight": 2},
    ).add_to(fmap)
    st_folium(fmap, height=500, use_container_width=True)

else:
    stats = result["stats"]
    cols  = st.columns(4)
    cols[0].metric(
        "Protected land lost",
        f"{stats.get('protected_lost_km2', 0):.3f} km²",
        help="Protected land (inside the PA or within its 100 m buffer) that became built-up.",
    )
    cols[1].metric(
        "Inside PA boundary",
        f"{stats.get('inside_pa_km2', 0):.3f} km²",
    )
    cols[2].metric(
        "In 100 m buffer",
        f"{stats.get('buffer_km2', 0):.3f} km²",
    )
    cols[3].metric(
        "Total new built-up (AOI)",
        f"{stats.get('changed_km2', 0):.2f} km²",
        help="All land in the scene that became built-up, protected or not.",
    )

    fmap = build_map(
        aoi_geom   = aoi_geom,
        rgb1_path  = result["rgb1_path"],
        rgb2_path  = result["rgb2_path"],
        mask1_path = result["mask1_path"],
        mask2_path = result["mask2_path"],
        change_result = result,
    )
    st_folium(fmap, height=560, use_container_width=True)

    with st.expander("Change breakdown — what did those pixels used to be?"):
        fc = stats.get("from_classes", {})
        if fc:
            import pandas as pd
            df = pd.DataFrame(
                [(cls, px, round(px * abs(result["profile"]["transform"].a) ** 2 / 1e6, 3))
                 for cls, px in sorted(fc.items(), key=lambda x: -x[1])],
                columns=["Previous class", "Pixels", "Area (km²)"],
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No land-cover transitions detected.")

    if stats.get("affected_pas"):
        with st.expander(f"Affected protected areas ({len(stats['affected_pas'])})"):
            for pa_name in stats["affected_pas"]:
                subset = result["encroachments"][result["encroachments"]["name"] == pa_name]
                total  = subset["area_km2"].sum()
                st.write(f"**{pa_name}** — {total:.3f} km² of protected land lost")
