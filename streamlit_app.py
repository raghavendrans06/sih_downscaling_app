import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium

from backend.downscaler import WeatherDownscaler
from weather_api import get_block_weather


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SIH26074 Weather Downscaling",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🌦️ SIH26074 Weather Downscaling")

st.subheader(
    "🌾 Block-Level to Panchayat-Level "
    "Agro-Meteorological Advisory"
)

st.write(
    "Downscale block-level weather forecasts to "
    "Panchayat-level forecasts using machine learning "
    "and local geographical features."
)


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_model():
    return WeatherDownscaler()


engine = load_model()


# ============================================================
# LOAD PANCHAYAT CSV
# ============================================================

PANCHAYAT_FILE = "panchayats.csv"


@st.cache_data
def load_panchayats():

    df = pd.read_csv(PANCHAYAT_FILE)

    required_columns = [
        "id",
        "name",
        "lat",
        "lon",
        "elevation",
        "slope",
        "ndvi"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in panchayats.csv: "
            + ", ".join(missing_columns)
        )

    if df.empty:

        raise ValueError(
            "panchayats.csv is empty."
        )

    return df


# ============================================================
# READ PANCHAYATS
# ============================================================

try:

    panchayat_df = load_panchayats()

except Exception as e:

    st.error(
        f"Unable to load panchayats.csv: {e}"
    )

    st.stop()


panchayats = panchayat_df.to_dict(
    orient="records"
)


# ============================================================
# BLOCK LOCATION
# ============================================================

block_lat = float(
    panchayat_df["lat"].mean()
)

block_lon = float(
    panchayat_df["lon"].mean()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🌤️ Block-Level Weather")

st.sidebar.write(
    f"📍 Panchayats Loaded: {len(panchayats)}"
)


# ============================================================
# FETCH LIVE WEATHER
# ============================================================

if st.sidebar.button(
    "🌐 Fetch Live Weather",
    use_container_width=True
):

    try:

        weather = get_block_weather(
            block_lat,
            block_lon
        )

        st.session_state["weather_data"] = weather

        st.session_state["live_temperature"] = (
            weather["current"]["temperature_2m"]
        )

        st.session_state["live_rain"] = (
            weather["current"]["rain"]
        )

        st.sidebar.success(
            "Live weather fetched."
        )

    except Exception:

        st.sidebar.warning(
            "⚠️ Live weather is temporarily unavailable."
        )

        st.sidebar.info(
            "You can continue using the Block Temperature "
            "and Block Rainfall values manually."
        )


# ============================================================
# TEMPERATURE INPUT
# ============================================================

default_temp = st.session_state.get(
    "live_temperature",
    32.0
)


coarse_temp = st.sidebar.number_input(
    "🌡️ Block Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=float(default_temp),
    step=0.5
)


# ============================================================
# RAINFALL INPUT
# ============================================================

default_rain = st.session_state.get(
    "live_rain",
    20.0
)


coarse_rain = st.sidebar.number_input(
    "🌧️ Block Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=float(default_rain),
    step=1.0
)


# ============================================================
# GENERATE FORECAST
# ============================================================

if st.sidebar.button(
    "🚀 Generate Forecast",
    use_container_width=True
):

    forecasts = engine.downscale(
        coarse_temp=coarse_temp,
        coarse_rain=coarse_rain,
        panchayat_features=panchayats
    )

    st.session_state["forecasts"] = forecasts

    st.session_state["coarse_temp"] = coarse_temp

    st.session_state["coarse_rain"] = coarse_rain


# ============================================================
# PANCHAYAT DATASET
# ============================================================

st.header("📋 Panchayat Dataset")

st.write(
    "Panchayat locations and geographical features "
    "used by the downscaling model."
)

st.dataframe(
    panchayat_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# LIVE BLOCK WEATHER
# ============================================================

if "weather_data" in st.session_state:

    weather = st.session_state["weather_data"]

    st.header("🌦️ Live Block Weather")

    current = weather["current"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🌡️ Temperature",
        f"{current['temperature_2m']:.1f} °C"
    )

    col2.metric(
        "🌧️ Rain",
        f"{current['rain']:.1f} mm"
    )

    col3.metric(
        "💧 Precipitation",
        f"{current['precipitation']:.1f} mm"
    )


    # ========================================================
    # 3-DAY BLOCK FORECAST
    # ========================================================

    st.header("📅 3-Day Block Weather Forecast")

    daily = weather["daily"]

    block_forecast = []

    for i in range(len(daily["time"])):

        block_forecast.append({
            "Date": daily["time"][i],

            "Min Temperature (°C)": round(
                daily["temperature_2m_min"][i],
                1
            ),

            "Max Temperature (°C)": round(
                daily["temperature_2m_max"][i],
                1
            ),

            "Rainfall (mm)": round(
                daily["rain_sum"][i],
                1
            )
        })


    block_df = pd.DataFrame(
        block_forecast
    )


    st.dataframe(
        block_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # 3-DAY PANCHAYAT DOWNSCALING
    # ========================================================

    st.header("📍 3-Day Panchayat-Level Forecast")

    st.write(
        "Block-level daily forecasts are downscaled "
        "for every Panchayat using local geographical features."
    )


    daily_panchayat_forecasts = []


    for i in range(len(daily["time"])):

        date = daily["time"][i]

        daily_max_temp = daily[
            "temperature_2m_max"
        ][i]

        daily_rain = daily[
            "rain_sum"
        ][i]


        day_forecasts = engine.downscale(
            coarse_temp=daily_max_temp,
            coarse_rain=daily_rain,
            panchayat_features=panchayats
        )


        for forecast in day_forecasts:

            daily_panchayat_forecasts.append({
                "Date": date,

                "Panchayat": forecast["name"],

                "Temperature (°C)": (
                    forecast["downscaled_temp"]
                ),

                "Rainfall (mm)": (
                    forecast["downscaled_rain"]
                ),

                "Advisory": (
                    forecast["advisory"]
                )
            })


    daily_panchayat_df = pd.DataFrame(
        daily_panchayat_forecasts
    )


    st.dataframe(
        daily_panchayat_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PANCHAYAT DETAILS
    # ========================================================

    st.header("🌾 Panchayat Details")


    panchayat_names = [
        p["name"]
        for p in panchayats
    ]


    selected_panchayat = st.selectbox(
        "📍 Select Panchayat",
        panchayat_names
    )


    selected_forecast = daily_panchayat_df[
        daily_panchayat_df["Panchayat"]
        == selected_panchayat
    ].copy()


    average_temperature = selected_forecast[
        "Temperature (°C)"
    ].mean()


    total_rainfall = selected_forecast[
        "Rainfall (mm)"
    ].sum()


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "🌡️ Average Temperature",
        f"{average_temperature:.1f} °C"
    )


    col2.metric(
        "🌧️ 3-Day Rainfall",
        f"{total_rainfall:.1f} mm"
    )


    col3.metric(
        "📍 Panchayat",
        selected_panchayat
    )


    st.subheader(
        f"📅 Forecast for {selected_panchayat}"
    )


    st.dataframe(
        selected_forecast,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # AGRICULTURAL ADVISORY
    # ========================================================

    if not selected_forecast.empty:

        latest_advisory = selected_forecast.iloc[-1][
            "Advisory"
        ]

        st.subheader("🌱 Agricultural Advisory")

        st.info(
            latest_advisory
        )


# ============================================================
# CURRENT PANCHAYAT FORECAST
# ============================================================

if "forecasts" in st.session_state:

    forecasts = st.session_state["forecasts"]

    block_temp = st.session_state["coarse_temp"]

    block_rain = st.session_state["coarse_rain"]


    st.header("📊 Current Panchayat-Level Forecast")


    # ========================================================
    # SUMMARY
    # ========================================================

    average_temperature = sum(
        f["downscaled_temp"]
        for f in forecasts
    ) / len(forecasts)


    average_rainfall = sum(
        f["downscaled_rain"]
        for f in forecasts
    ) / len(forecasts)


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "🌡️ Block Temperature",
        f"{block_temp:.1f} °C"
    )


    col2.metric(
        "🌧️ Block Rainfall",
        f"{block_rain:.1f} mm"
    )


    col3.metric(
        "🌡️ Avg Panchayat Temperature",
        f"{average_temperature:.1f} °C"
    )


    col4.metric(
        "🌧️ Avg Panchayat Rainfall",
        f"{average_rainfall:.1f} mm"
    )


    # ========================================================
    # MAP
    # ========================================================

    st.header("🗺️ Panchayat Weather Map")


    map_center_lat = float(
        panchayat_df["lat"].mean()
    )

    map_center_lon = float(
        panchayat_df["lon"].mean()
    )


    weather_map = folium.Map(
        location=[
            map_center_lat,
            map_center_lon
        ],
        zoom_start=11
    )


    for forecast in forecasts:

        popup_html = f"""
        <div style="font-size:14px">

        <b>{forecast["name"]}</b>

        <br><br>

        Temperature:
        <b>{forecast["downscaled_temp"]:.2f} °C</b>

        <br>

        Rainfall:
        <b>{forecast["downscaled_rain"]:.2f} mm</b>

        <br>

        Elevation:
        <b>{forecast["elevation"]} m</b>

        <br>

        Slope:
        <b>{forecast["slope"]}°</b>

        <br>

        NDVI:
        <b>{forecast["ndvi"]}</b>

        <br><br>

        <b>Agricultural Advisory</b>

        <br>

        {forecast["advisory"]}

        </div>
        """


        folium.Marker(
            location=[
                forecast["lat"],
                forecast["lon"]
            ],

            tooltip=(
                f"{forecast['name']} | "
                f"{forecast['downscaled_temp']:.1f} °C | "
                f"{forecast['downscaled_rain']:.1f} mm"
            ),

            popup=folium.Popup(
                popup_html,
                max_width=350
            ),

            icon=folium.Icon(
                icon="info-sign"
            )

        ).add_to(weather_map)


    st_folium(
        weather_map,
        width=None,
        height=500
    )


    # ========================================================
    # FORECAST TABLE
    # ========================================================

    st.header("📋 Current Panchayat Forecast")


    current_forecast_table = []


    for forecast in forecasts:

        current_forecast_table.append({
            "Panchayat":
                forecast["name"],

            "Temperature (°C)":
                round(
                    forecast["downscaled_temp"],
                    2
                ),

            "Rainfall (mm)":
                round(
                    forecast["downscaled_rain"],
                    2
                ),

            "Advisory":
                forecast["advisory"]
        })


    current_forecast_df = pd.DataFrame(
        current_forecast_table
    )


    st.dataframe(
        current_forecast_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📈 Model Performance")


metrics = engine.get_metrics()


col1, col2 = st.columns(2)


# ============================================================
# TEMPERATURE METRICS
# ============================================================

with col1:

    st.subheader("🌡️ Temperature Model")


    temperature_metrics = pd.DataFrame([
        {
            "Metric": "MAE",
            "Value": round(
                metrics["temperature"]["MAE"],
                3
            )
        },

        {
            "Metric": "RMSE",
            "Value": round(
                metrics["temperature"]["RMSE"],
                3
            )
        },

        {
            "Metric": "R²",
            "Value": round(
                metrics["temperature"]["R2"],
                3
            )
        }
    ])


    st.dataframe(
        temperature_metrics,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RAINFALL METRICS
# ============================================================

with col2:

    st.subheader("🌧️ Rainfall Model")


    rainfall_metrics = pd.DataFrame([
        {
            "Metric": "MAE",
            "Value": round(
                metrics["rainfall"]["MAE"],
                3
            )
        },

        {
            "Metric": "RMSE",
            "Value": round(
                metrics["rainfall"]["RMSE"],
                3
            )
        },

        {
            "Metric": "R²",
            "Value": round(
                metrics["rainfall"]["R2"],
                3
            )
        }
    ])


    st.dataframe(
        rainfall_metrics,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("🔎 ML Feature Importance")


st.write(
    "Importance of each input variable in the "
    "Random Forest downscaling models."
)


importance = engine.get_feature_importance()


# ============================================================
# TEMPERATURE FEATURE IMPORTANCE
# ============================================================

temperature_importance_df = pd.DataFrame({
    "Feature": list(
        importance["temperature"].keys()
    ),

    "Importance": list(
        importance["temperature"].values()
    )
})


temperature_importance_df[
    "Importance"
] = temperature_importance_df[
    "Importance"
].round(4)


# ============================================================
# RAINFALL FEATURE IMPORTANCE
# ============================================================

rainfall_importance_df = pd.DataFrame({
    "Feature": list(
        importance["rainfall"].keys()
    ),

    "Importance": list(
        importance["rainfall"].values()
    )
})


rainfall_importance_df[
    "Importance"
] = rainfall_importance_df[
    "Importance"
].round(4)


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "🌡️ Temperature Downscaling"
    )

    st.dataframe(
        temperature_importance_df.sort_values(
            "Importance",
            ascending=False
        ),

        use_container_width=True,

        hide_index=True
    )


with col2:

    st.subheader(
        "🌧️ Rainfall Downscaling"
    )

    st.dataframe(
        rainfall_importance_df.sort_values(
            "Importance",
            ascending=False
        ),

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# INITIAL / FALLBACK MESSAGE
# ============================================================

if "weather_data" not in st.session_state:

    st.info(
        "You can fetch live block-level weather or "
        "enter the Block Temperature and Block Rainfall "
        "manually, then click '🚀 Generate Forecast'."
    )