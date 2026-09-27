import requests


def get_block_weather(latitude, longitude):
    """
    Fetch weather forecast for a block location using Open-Meteo.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "rain",
            "precipitation"
        ],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum"
        ],
        "timezone": "auto",
        "forecast_days": 3
    }

    response = requests.get(url, params=params, timeout=15)

    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    # Test location
    # Replace these coordinates with your actual block coordinates later.
    latitude = 26.15
    longitude = 91.77

    weather = get_block_weather(latitude, longitude)

    print("Current Temperature:",
          weather["current"]["temperature_2m"], "°C")

    print("Current Rain:",
          weather["current"]["rain"], "mm")

    print("3-Day Rainfall:",
          weather["daily"]["rain_sum"])