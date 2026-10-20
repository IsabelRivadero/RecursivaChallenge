import numpy as np
import pandas as pd

from src.models.predict import ForecastPredictor


class FakeModel:
    """Modelo simulado para probar el pipeline de inferencia."""

    def predict(self, X):
        return np.array([150.0] * len(X))


def test_predictor_uses_base_forecast_without_trained_model():
    config = {
        "features": {
            "cols": ["rolling_mean_21"]
        }
    }

    predictor = ForecastPredictor(
        models={},
        config=config,
    )

    df = pd.DataFrame({
        "centro_distribucion": ["CD_A"],
        "unidades_pronosticadas": [120.0],
        "rolling_mean_21": [100.0],
    })

    result = predictor.predict_df(df)

    assert result.loc[0, "unidades_corregidas"] == 120.0


def test_predictor_uses_trained_model_when_features_exist():
    config = {
        "features": {
            "cols": ["rolling_mean_21"]
        }
    }

    predictor = ForecastPredictor(
        models={"CD_A": FakeModel()},
        config=config,
    )

    df = pd.DataFrame({
        "centro_distribucion": ["CD_A"],
        "unidades_pronosticadas": [120.0],
        "rolling_mean_21": [100.0],
    })

    result = predictor.predict_df(df)

    assert result.loc[0, "unidades_corregidas"] == 150.0


def test_predictor_falls_back_when_rolling_history_is_missing():
    config = {
        "features": {
            "cols": ["rolling_mean_21"]
        }
    }

    predictor = ForecastPredictor(
        models={"CD_A": FakeModel()},
        config=config,
    )

    df = pd.DataFrame({
        "centro_distribucion": ["CD_A"],
        "unidades_pronosticadas": [120.0],
        "rolling_mean_21": [np.nan],
    })

    result = predictor.predict_df(df)

    assert result.loc[0, "unidades_corregidas"] == 120.0