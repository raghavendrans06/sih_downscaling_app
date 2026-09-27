from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from downscaler import WeatherDownscaler

app = FastAPI(title="SIH26074 Weather Downscaling API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ML engine
engine = WeatherDownscaler()

# Temporary Panchayat spatial data
MOCK_PANCHAYATS = [
    {
        "id": "P01",
        "name": "Panchayat North (Hill Area)",
        "lat": 26.18,
        "lon": 91.75,
        "elevation": 650,
        "slope": 22.0,
        "ndvi": 0.72
    },
    {
        "id": "P02",
        "name": "Panchayat Central (Valley)",
        "lat": 26.15,
        "lon": 91.77,
        "elevation": 120,
        "slope": 4.0,
        "ndvi": 0.45
    },
    {
        "id": "P03",
        "name": "Panchayat South (Plains)",
        "lat": 26.10,
        "lon": 91.80,
        "elevation": 70,
        "slope": 2.0,
        "ndvi": 0.55
    },
    {
        "id": "P04",
        "name": "Panchayat East (Ridge)",
        "lat": 26.14,
        "lon": 91.85,
        "elevation": 480,
        "slope": 18.0,
        "ndvi": 0.68
    }
]


@app.get("/api/forecast")
def get_downscaled_forecast(
    coarse_temp: float = 32.0,
    coarse_rain: float = 20.0
):
    downscaled_data = engine.downscale(
        coarse_temp,
        coarse_rain,
        MOCK_PANCHAYATS
    )

    return {
        "block_name": "Demonstration Block 01",
        "coarse_input": {
            "block_temp": coarse_temp,
            "block_rain": coarse_rain
        },
        "panchayat_forecasts": downscaled_data
    }


# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )