import streamlit as st
import pandas as pd
import numpy as np
import folium

from streamlit_folium import st_folium
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.downscaler import WeatherDownscaler
from weather_api import get_block_weather


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SIH26074 Weather Downscaling",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f0f9ff 0%,
        #ecfeff 50%,
        #f0fdf4 100%
    );
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #064e3b 0%,
        #075985 100%
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] input {
    background-color: white !important;
    color: #111827 !important;
}

[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stButton button {
    background: #0284c7 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 48px;
}


/* TITLE */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #075985;
    line-height: 1.15;
}

.main-subtitle {
    font-size: 18px;
    color: #334155;
    margin-top: 8px;
}

.main-description {
    font-size: 14px;
    color: #64748b;
    margin-top: 6px;
}


/* INFO */

.info-banner {
    background: #dbeafe;
    border-left: 5px solid #0284c7;
    padding: 15px 20px;
    border-radius: 10px;
    color: #075985;
    margin-top: 25px;
    margin-bottom: 30px;
}


/* SECTION */

.section-title {
    color: #064e3b;
    font-size: 28px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 18px;
}


/* WORKFLOW */

.workflow-card {
    background: white;
    border: 1px solid #d1fae5;
    border-radius: 16px;
    padding: 25px 10px;
    text-align: center;
    min-height: 135px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
}

.workflow-card .icon {
    font-size: 32px;
    margin-bottom: 8px;
}

.workflow-card .title {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
}

.workflow-card .text {
    font-size: 13px;
    color: #64748b;
    margin-top: 6px;
}


/* EMPTY STATE */

.empty-card {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding-top: 35px;
    padding-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PANCHAYAT DATA
# ============================================================

DEFAULT_PANCHAYATS = [
    {
        "id": "P01",
        "name": "North Hill",
        "lat": 26.18,
        "lon": 91.75,
        "elevation": 650,
        "slope": 22,
        "ndvi": 0.72
    },
    {
        "id": "P02",
        "name": "Central Valley",
        "lat": 26.15,
        "lon": 91.77,
        "elevation": 120,
        "slope": 4,
        "ndvi": 0.45
    },
    {
        "id": "P03",
        "name": "South Plains",
        "lat": 26.10,
        "lon": 91.80,
        "elevation": 70,
        "slope": 2,
        "ndvi": 0.55
    },
    {
        "id": "P04",
        "name": "East Ridge",
        "lat": 26.14,
        "lon": 91.85,
        "elevation": 480,
        "slope": 18,
        "ndvi": 0.68
    }
]


# ============================================================
# SESSION STATE
# ============================================================

if "weather" not in st.session_state:
    st.session_state.weather = None

if "forecasts" not in st.session_state:
    st.session_state.forecasts = None

if "panchayats" not in st.session_state:
    st.session_state.panchayats = DEFAULT_PANCHAYATS


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return WeatherDownscaler()


try:
    engine = load_model()
    model_loaded = True
except Exception as e:
    engine = None
    model_loaded = False
    st.error(f"❌ Model loading failed: {e}")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Simple HTML only — no nested divs
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.12);
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 25px;
        color: white;
    ">
        🌦️<br>
        <b style="font-size:24px;">SIH26074</b><br>
        <span style="font-size:13px;">
            Weather Downscaling Prototype
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Forecast Configuration")

    st.markdown("#### 📍 Block Location")

    block_lat = st.number_input(
        "Latitude",
        value=26.15000,
        format="%.5f"
    )

    block_lon = st.number_input(
        "Longitude",
        value=91.77000,
        format="%.5f"
    )

    st.markdown("#### 🗺️ Panchayat Data")

    uploaded_panchayats = st.file_uploader(
        "Upload Panchayat CSV",
        type=["csv"],
        help="Required: id, name, lat, lon, elevation, slope, ndvi"
    )

    if uploaded_panchayats is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_panchayats
            )

            required = [
                "id",
                "name",
                "lat",
                "lon",
                "elevation",
                "slope",
                "ndvi"
            ]

            missing = [
                col for col in required
                if col not in uploaded_df.columns
            ]

            if missing:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing)
                )

            else:

                st.session_state.panchayats = (
                    uploaded_df[required].to_dict("records")
                )

                st.success(
                    f"Loaded {len(uploaded_df)} Panchayats"
                )

        except Exception as e:

            st.error(f"CSV error: {e}")


    panchayats = st.session_state.panchayats

    st.markdown("#### 🚀 Run Prediction")

    generate_button = st.button(
        "🌐 Get Live Weather & Generate Forecast",
        use_container_width=True
    )

    if generate_button:

        if not model_loaded:

            st.error("ML model is not available.")

        else:

            try:

                with st.spinner(
                    "🌐 Fetching live weather..."
                ):

                    weather = get_block_weather(
                        block_lat,
                        block_lon
                    )

                st.session_state.weather = weather

                coarse_temp = float(
                    weather["current"]["temperature_2m"]
                )

                coarse_rain = float(
                    weather["current"].get(
                        "rain",
                        0
                    ) or 0
                )

                with st.spinner(
                    "🤖 Generating Panchayat-level forecasts..."
                ):

                    forecasts = engine.downscale(
                        coarse_temp=coarse_temp,
                        coarse_rain=coarse_rain,
                        panchayat_features=panchayats
                    )

                st.session_state.forecasts = forecasts

                st.success(
                    "✅ Forecast generated successfully!"
                )

            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="main-title">
🌦️ SIH26074 Weather Downscaling
</div>

<div class="main-subtitle">
🌱 Panchayat-Level Agro-Meteorological Intelligence
</div>

<div class="main-description">
Transforming block-level weather forecasts into localized
Panchayat-level agricultural insights.
</div>
""", unsafe_allow_html=True)


# ============================================================
# INFO
# ============================================================

if st.session_state.forecasts is None:

    st.markdown("""
    <div class="info-banner">
    👉 Click <b>Get Live Weather & Generate Forecast</b>
    from the sidebar to start the prediction pipeline.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# WORKFLOW
# ============================================================

st.markdown(
    '<div class="section-title">🔄 How the Prototype Works</div>',
    unsafe_allow_html=True
)

workflow = [
    ("🌐", "Live Weather", "Current forecast"),
    ("📦", "Block Data", "Coarse forecast"),
    ("🤖", "ML Model", "Downscaling"),
    ("📍", "Panchayat", "Localized forecast"),
    ("🌾", "Advisory", "Farmer insights")
]

cols = st.columns(5)

for col, data in zip(cols, workflow):

    icon, title, description = data

    with col:

        # IMPORTANT:
        # No nested divs.
        st.markdown(
            f"""
            <div class="workflow-card">
            <span class="icon">{icon}</span><br>
            <span class="title">{title}</span><br>
            <span class="text">{description}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CHECK FORECAST
# ============================================================

weather = st.session_state.weather
forecasts = st.session_state.forecasts

if weather is None or forecasts is None:

    st.markdown("""
    <div class="empty-card">
    <h2>🌦️</h2>
    <h3>Ready to Generate a Forecast</h3>
    <p>
    Enter the block location in the sidebar and click
    the forecast button.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# LIVE WEATHER
# ============================================================

st.markdown(
    '<div class="section-title">🌐 Live Block Weather</div>',
    unsafe_allow_html=True
)

current = weather["current"]

current_temp = float(
    current.get("temperature_2m", 0)
)

current_rain = float(
    current.get("rain", 0) or 0
)

current_precip = float(
    current.get("precipitation", 0) or 0
)

cols = st.columns(4)

with cols[0]:
    st.metric(
        "🌡️ Temperature",
        f"{current_temp:.1f} °C"
    )

with cols[1]:
    st.metric(
        "🌧️ Rain",
        f"{current_rain:.1f} mm"
    )

with cols[2]:
    st.metric(
        "💧 Precipitation",
        f"{current_precip:.1f} mm"
    )

with cols[3]:
    st.metric(
        "📍 Panchayats",
        len(panchayats)
    )


# ============================================================
# FORECAST DATAFRAME
# ============================================================

forecast_df = pd.DataFrame(
    forecasts
)

feature_df = pd.DataFrame(
    panchayats
)[
    [
        "id",
        "elevation",
        "slope",
        "ndvi"
    ]
]

forecast_df = forecast_df.merge(
    feature_df,
    on="id",
    how="left",
    suffixes=("", "_feature")
)

if "elevation_feature" in forecast_df.columns:

    forecast_df["elevation"] = (
        forecast_df["elevation"]
        .fillna(
            forecast_df["elevation_feature"]
        )
    )

    forecast_df.drop(
        columns=["elevation_feature"],
        inplace=True
    )


# ============================================================
# RISK
# ============================================================

def get_risk(temp, rain):

    if rain > 35:
        return "🔴 Heavy Rainfall"

    if temp > 36:
        return "🔴 Heat Stress"

    if rain > 15 and temp > 30:
        return "🟠 Fungal Disease"

    if rain > 10:
        return "🟡 Moderate Rain"

    return "🟢 Low Risk"


forecast_df["risk"] = forecast_df.apply(
    lambda row: get_risk(
        row["downscaled_temp"],
        row["downscaled_rain"]
    ),
    axis=1
)


# ============================================================
# SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📊 Panchayat Forecast Summary</div>',
    unsafe_allow_html=True
)

cols = st.columns(4)

with cols[0]:
    st.metric(
        "🌡️ Avg Temperature",
        f"{forecast_df.downscaled_temp.mean():.1f} °C"
    )

with cols[1]:
    st.metric(
        "🌧️ Avg Rainfall",
        f"{forecast_df.downscaled_rain.mean():.1f} mm"
    )

with cols[2]:
    st.metric(
        "🔥 Max Temperature",
        f"{forecast_df.downscaled_temp.max():.1f} °C"
    )

with cols[3]:
    st.metric(
        "🌧️ Max Rainfall",
        f"{forecast_df.downscaled_rain.max():.1f} mm"
    )


# ============================================================
# MAP
# ============================================================

st.markdown(
    '<div class="section-title">🗺️ Panchayat-Level Forecast Map</div>',
    unsafe_allow_html=True
)

map_center = [
    forecast_df.lat.mean(),
    forecast_df.lon.mean()
]

m = folium.Map(
    location=map_center,
    zoom_start=11,
    tiles="OpenStreetMap"
)

for _, row in forecast_df.iterrows():

    if (
        "Heavy" in row.risk
        or "Heat" in row.risk
    ):
        marker_color = "red"

    elif (
        "Fungal" in row.risk
        or "Moderate" in row.risk
    ):
        marker_color = "orange"

    else:
        marker_color = "green"


    popup_html = f"""
    <div style="width:260px;font-family:Arial;">

    <h4 style="color:#075985;">
    📍 {row['name']}
    </h4>

    <hr>

    <b>🌡️ Temperature:</b>
    {row['downscaled_temp']:.2f} °C
    <br>

    <b>🌧️ Rainfall:</b>
    {row['downscaled_rain']:.2f} mm
    <br>

    <b>⛰️ Elevation:</b>
    {row['elevation']:.0f} m
    <br>

    <b>📐 Slope:</b>
    {row['slope']:.1f}°
    <br>

    <b>🌱 NDVI:</b>
    {row['ndvi']:.2f}
    <br>

    <b>⚠️ Risk:</b>
    {row['risk']}

    <br><br>

    <b>🌾 Advisory:</b>
    <br>

    {row['advisory']}

    </div>
    """

    folium.Marker(
        location=[
            row["lat"],
            row["lon"]
        ],
        tooltip=row["name"],
        popup=folium.Popup(
            popup_html,
            max_width=320
        ),
        icon=folium.Icon(
            color=marker_color,
            icon="cloud"
        )
    ).add_to(m)


st_folium(
    m,
    height=550,
    width=None
)


# ============================================================
# PANCHAYAT DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">📍 Panchayat-Level Insights</div>',
    unsafe_allow_html=True
)

for _, row in forecast_df.iterrows():

    with st.expander(
        f"📍 {row['name']} — {row['risk']}"
    ):

        cols = st.columns(4)

        with cols[0]:
            st.metric(
                "Temperature",
                f"{row['downscaled_temp']:.2f} °C"
            )

        with cols[1]:
            st.metric(
                "Rainfall",
                f"{row['downscaled_rain']:.2f} mm"
            )

        with cols[2]:
            st.metric(
                "Elevation",
                f"{row['elevation']:.0f} m"
            )

        with cols[3]:
            st.metric(
                "NDVI",
                f"{row['ndvi']:.2f}"
            )

        st.write(
            f"**📐 Slope:** {row['slope']:.1f}°"
        )

        st.write(
            f"**⚠️ Risk:** {row['risk']}"
        )

        st.info(
            f"🌾 **Agricultural Advisory:** "
            f"{row['advisory']}"
        )


# ============================================================
# TABLE
# ============================================================

st.markdown(
    '<div class="section-title">📋 Forecast Data Table</div>',
    unsafe_allow_html=True
)

display_df = forecast_df[
    [
        "id",
        "name",
        "lat",
        "lon",
        "elevation",
        "slope",
        "ndvi",
        "downscaled_temp",
        "downscaled_rain",
        "risk",
        "advisory"
    ]
].copy()

display_df.columns = [
    "ID",
    "Panchayat",
    "Latitude",
    "Longitude",
    "Elevation (m)",
    "Slope (°)",
    "NDVI",
    "Temperature (°C)",
    "Rainfall (mm)",
    "Risk",
    "Advisory"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = display_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Download Panchayat Forecast CSV",
    data=csv_data,
    file_name="panchayat_weather_forecast.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# 3-DAY FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">📅 3-Day Block Forecast</div>',
    unsafe_allow_html=True
)

try:

    daily = weather["daily"]

    dates = daily.get("time", [])

    max_temp = daily.get(
        "temperature_2m_max",
        []
    )

    min_temp = daily.get(
        "temperature_2m_min",
        []
    )

    rain = daily.get(
        "rain_sum",
        []
    )

    data = []

    for i in range(
        min(3, len(dates))
    ):

        data.append({
            "Date": dates[i],
            "Max Temperature (°C)": max_temp[i],
            "Min Temperature (°C)": min_temp[i],
            "Rainfall (mm)": rain[i]
        })

    if data:

        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True,
            hide_index=True
        )

except Exception as e:

    st.info(
        f"3-day forecast unavailable: {e}"
    )


# ============================================================
# MODEL VALIDATION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 ML Model Validation</div>',
    unsafe_allow_html=True
)

if model_loaded:

    try:

        metrics = engine.get_metrics()

        temp_metrics = metrics["temperature"]

        rain_metrics = metrics["rainfall"]

        st.markdown("#### 🌡️ Temperature Model")

        cols = st.columns(3)

        with cols[0]:
            st.metric(
                "MAE",
                f"{temp_metrics['MAE']:.3f}"
            )

        with cols[1]:
            st.metric(
                "RMSE",
                f"{temp_metrics['RMSE']:.3f}"
            )

        with cols[2]:
            st.metric(
                "R²",
                f"{temp_metrics['R2']:.3f}"
            )

        st.markdown("#### 🌧️ Rainfall Model")

        cols = st.columns(3)

        with cols[0]:
            st.metric(
                "MAE",
                f"{rain_metrics['MAE']:.3f}"
            )

        with cols[1]:
            st.metric(
                "RMSE",
                f"{rain_metrics['RMSE']:.3f}"
            )

        with cols[2]:
            st.metric(
                "R²",
                f"{rain_metrics['R2']:.3f}"
            )

    except Exception as e:

        st.warning(
            f"Model metrics unavailable: {e}"
        )


# ============================================================
# EXTERNAL VALIDATION
# ============================================================

st.markdown(
    '<div class="section-title">📈 External Validation</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload actual and predicted weather values for independent validation."
)

validation_file = st.file_uploader(
    "Upload validation CSV",
    type=["csv"],
    key="validation"
)

if validation_file is not None:

    try:

        validation_df = pd.read_csv(
            validation_file
        )

        required = [
            "actual_temp",
            "predicted_temp",
            "actual_rain",
            "predicted_rain"
        ]

        missing = [
            col
            for col in required
            if col not in validation_df.columns
        ]

        if missing:

            st.error(
                "Missing columns: "
                + ", ".join(missing)
            )

        else:

            actual_temp = validation_df.actual_temp
            predicted_temp = validation_df.predicted_temp

            actual_rain = validation_df.actual_rain
            predicted_rain = validation_df.predicted_rain

            temp_mae = mean_absolute_error(
                actual_temp,
                predicted_temp
            )

            temp_rmse = np.sqrt(
                mean_squared_error(
                    actual_temp,
                    predicted_temp
                )
            )

            temp_r2 = r2_score(
                actual_temp,
                predicted_temp
            )

            rain_mae = mean_absolute_error(
                actual_rain,
                predicted_rain
            )

            rain_rmse = np.sqrt(
                mean_squared_error(
                    actual_rain,
                    predicted_rain
                )
            )

            rain_r2 = r2_score(
                actual_rain,
                predicted_rain
            )

            st.markdown("#### 🌡️ Temperature Validation")

            cols = st.columns(3)

            with cols[0]:
                st.metric(
                    "MAE",
                    f"{temp_mae:.3f}"
                )

            with cols[1]:
                st.metric(
                    "RMSE",
                    f"{temp_rmse:.3f}"
                )

            with cols[2]:
                st.metric(
                    "R²",
                    f"{temp_r2:.3f}"
                )

            st.markdown("#### 🌧️ Rainfall Validation")

            cols = st.columns(3)

            with cols[0]:
                st.metric(
                    "MAE",
                    f"{rain_mae:.3f}"
                )

            with cols[1]:
                st.metric(
                    "RMSE",
                    f"{rain_rmse:.3f}"
                )

            with cols[2]:
                st.metric(
                    "R²",
                    f"{rain_r2:.3f}"
                )

    except Exception as e:

        st.error(
            f"Validation error: {e}"
        )


# ============================================================
# CSV FORMAT
# ============================================================

with st.expander(
    "📄 Panchayat CSV Format"
):

    st.code(
        """id,name,lat,lon,elevation,slope,ndvi
P01,North Hill,26.18,91.75,650,22,0.72
P02,Central Valley,26.15,91.77,120,4,0.45
P03,South Plains,26.10,91.80,70,2,0.55
P04,East Ridge,26.14,91.85,480,18,0.68""",
        language="csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
<b>SIH26074</b> · Panchayat-Level Weather Downscaling
<br>
Agriculture · FoodTech · Rural Development
<br>
Prototype for Smart India Hackathon 2026
</div>
""", unsafe_allow_html=True)