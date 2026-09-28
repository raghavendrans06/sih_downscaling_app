import time
import requests


def get_block_weather(latitude, longitude):

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

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=15
            )

            # Open-Meteo rate limit
            if response.status_code == 429:

                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    time.sleep(wait_time)
                    continue

                raise Exception(
                    "Weather service is temporarily rate-limited. "
                    "Please try again after a few minutes."
                )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.Timeout:

            if attempt < max_retries - 1:
                time.sleep(2)
                continue

            raise Exception(
                "Weather service timed out. "
                "Please try again."
            )

        except requests.exceptions.RequestException as error:

            raise Exception(
                f"Weather service error: {error}"
            )


if __name__ == "__main__":

    latitude = 26.15
    longitude = 91.77

    weather = get_block_weather(
        latitude,
        longitude
    )

    print(
        "Current Temperature:",
        weather["current"]["temperature_2m"],
        "°C"
    )

    print(
        "Current Rain:",
        weather["current"]["rain"],
        "mm"
    )

    print(
        "3-Day Rainfall:",
        weather["daily"]["rain_sum"]
    )