import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class WeatherDownscaler:

    def __init__(self, training_file="training_data.csv"):

        self.training_file = training_file

        self.temp_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        self.rain_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        self.temp_metrics = {}
        self.rain_metrics = {}

        self.train_models()


    # ========================================================
    # TRAIN MODELS
    # ========================================================

    def train_models(self):

        df = pd.read_csv(
            self.training_file
        )

        required_columns = [
            "coarse_temp",
            "coarse_rain",
            "elevation",
            "slope",
            "ndvi",
            "fine_temp",
            "fine_rain"
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns in training data: {missing}"
            )


        features = [
            "coarse_temp",
            "coarse_rain",
            "elevation",
            "slope",
            "ndvi"
        ]


        X = df[features]

        y_temp = df["fine_temp"]

        y_rain = df["fine_rain"]


        # ----------------------------------------------------
        # Temperature train/test split
        # ----------------------------------------------------

        (
            X_train_temp,
            X_test_temp,
            y_train_temp,
            y_test_temp
        ) = train_test_split(
            X,
            y_temp,
            test_size=0.20,
            random_state=42
        )


        # ----------------------------------------------------
        # Rainfall train/test split
        # ----------------------------------------------------

        (
            X_train_rain,
            X_test_rain,
            y_train_rain,
            y_test_rain
        ) = train_test_split(
            X,
            y_rain,
            test_size=0.20,
            random_state=42
        )


        # ----------------------------------------------------
        # Train models
        # ----------------------------------------------------

        self.temp_model.fit(
            X_train_temp,
            y_train_temp
        )

        self.rain_model.fit(
            X_train_rain,
            y_train_rain
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        temp_predictions = (
            self.temp_model.predict(
                X_test_temp
            )
        )

        rain_predictions = (
            self.rain_model.predict(
                X_test_rain
            )
        )


        # ----------------------------------------------------
        # Temperature metrics
        # ----------------------------------------------------

        self.temp_metrics = {

            "MAE": mean_absolute_error(
                y_test_temp,
                temp_predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_test_temp,
                    temp_predictions
                )
            ),

            "R2": r2_score(
                y_test_temp,
                temp_predictions
            )
        }


        # ----------------------------------------------------
        # Rainfall metrics
        # ----------------------------------------------------

        self.rain_metrics = {

            "MAE": mean_absolute_error(
                y_test_rain,
                rain_predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_test_rain,
                    rain_predictions
                )
            ),

            "R2": r2_score(
                y_test_rain,
                rain_predictions
            )
        }


    # ========================================================
    # DOWNSCALE WEATHER
    # ========================================================

    def downscale(
        self,
        coarse_temp,
        coarse_rain,
        panchayat_features
    ):

        forecasts = []


        for panchayat in panchayat_features:

            features = pd.DataFrame([
                {
                    "coarse_temp": coarse_temp,
                    "coarse_rain": coarse_rain,
                    "elevation": panchayat["elevation"],
                    "slope": panchayat["slope"],
                    "ndvi": panchayat["ndvi"]
                }
            ])


            # ------------------------------------------------
            # Temperature prediction
            # ------------------------------------------------

            downscaled_temp = (
                self.temp_model.predict(
                    features
                )[0]
            )


            # ------------------------------------------------
            # Rainfall prediction
            # ------------------------------------------------

            downscaled_rain = (
                self.rain_model.predict(
                    features
                )[0]
            )


            downscaled_rain = max(
                0,
                downscaled_rain
            )


            # ------------------------------------------------
            # Advisory
            # ------------------------------------------------

            advisory = self.generate_advisory(
                downscaled_temp,
                downscaled_rain
            )


            forecasts.append({

                "id": panchayat["id"],

                "name": panchayat["name"],

                "lat": panchayat["lat"],

                "lon": panchayat["lon"],

                "elevation": panchayat["elevation"],

                "downscaled_temp": round(
                    float(downscaled_temp),
                    2
                ),

                "downscaled_rain": round(
                    float(downscaled_rain),
                    2
                ),

                "advisory": advisory
            })


        return forecasts


    # ========================================================
    # AGRICULTURAL ADVISORY
    # ========================================================

    def generate_advisory(
        self,
        temperature,
        rainfall
    ):

        if rainfall > 35:

            return (
                "Heavy rainfall expected. "
                "Postpone pesticide/fertilizer application "
                "and ensure proper field drainage."
            )


        elif rainfall > 15 and temperature > 30:

            return (
                "Warm and wet conditions. "
                "Monitor crops for fungal disease "
                "and inspect fields regularly."
            )


        elif temperature > 36:

            return (
                "High temperature risk. "
                "Use light and frequent irrigation, "
                "preferably during early morning."
            )


        else:

            return (
                "Weather conditions are generally favorable "
                "for normal agricultural activities."
            )


    # ========================================================
    # MODEL METRICS
    # ========================================================

    def get_metrics(self):

        return {
            "temperature": self.temp_metrics,
            "rainfall": self.rain_metrics
        }