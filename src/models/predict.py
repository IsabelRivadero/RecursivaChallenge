import numpy as np
import pandas as pd


class ForecastPredictor:
    """Pipeline de inferencia de pronósticos."""

    def __init__(self, models: dict, config: dict):
        self.models = models
        self.config = config
        self.feature_cols = config["features"]["cols"]

    def predict_df(
        self,
        df_merged: pd.DataFrame,
    ) -> pd.DataFrame:
        """Genera predicciones corregidas o utiliza el modelo base."""

        df_result = df_merged.copy()
        predictions = []

        for _, row in df_result.iterrows():
            cd = row["centro_distribucion"]

            base_prediction = float(
                row["unidades_pronosticadas"]
            )

            model = self.models.get(cd)

            # Fallback para centros sin modelo o sin historia
            # suficiente para generar las features.
            if (
                model is None
                or pd.isna(row.get("rolling_mean_21"))
            ):
                predictions.append(base_prediction)
                continue

            missing = [
                col for col in self.feature_cols
                if col not in row.index
            ]

            if missing:
                raise ValueError(
                    f"Faltan features para {cd}: {missing}"
                )

            X_single = pd.DataFrame(
                [[row[col] for col in self.feature_cols]],
                columns=self.feature_cols,
            )

            X_single = X_single.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            prediction = float(
                model.predict(X_single)[0]
            )

            # Las unidades pronosticadas no deberían ser negativas.
            prediction = max(0.0, prediction)

            predictions.append(prediction)

        df_result["unidades_corregidas"] = predictions

        return df_result