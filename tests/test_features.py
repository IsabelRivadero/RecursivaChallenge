import pandas as pd

from src.features.builder import FeatureBuilder


def test_feature_builder_creates_calendar_and_lag_features():
    config = {
        "model": {
            "lags": [1, 2],
            "rolling_windows": [2],
        }
    }

    builder = FeatureBuilder(config)

    df = pd.DataFrame({
        "centro_distribucion": ["CD_A"] * 4,
        "fecha_despacho": pd.date_range(
            "2026-01-01", periods=4, freq="D"
        ),
        "unidades_reales": [10, 20, 30, 40],
    })

    result = builder.build_features(df)

    assert "dia_semana" in result.columns
    assert "es_fin_semana" in result.columns
    assert "dia_mes" in result.columns

    assert "lag_1" in result.columns
    assert "lag_2" in result.columns
    assert "rolling_mean_2" in result.columns

    # El primer registro no tiene observación anterior.
    assert pd.isna(result.loc[0, "lag_1"])

    # El segundo registro utiliza el valor del primer día.
    assert result.loc[1, "lag_1"] == 10

    # El tercer registro utiliza el valor de dos días antes.
    assert result.loc[2, "lag_2"] == 10

    # La media móvil utiliza observaciones anteriores,
    # sin incluir el valor del día evaluado.
    assert result.loc[2, "rolling_mean_2"] == 15