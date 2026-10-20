import pandas as pd


class FeatureBuilder:
    """Ingeniería de características temporales y métricas móviles."""

    def __init__(self, config: dict):
        self.config = config
        self.lags = config["model"]["lags"]
        self.windows = config["model"]["rolling_windows"]

    def build_features(self, df_reales_agg: pd.DataFrame) -> pd.DataFrame:
        cds = df_reales_agg["centro_distribucion"].unique()
        features_list = []

        for cd in cds:
            df_cd = (
                df_reales_agg[df_reales_agg["centro_distribucion"] == cd]
                .sort_values("fecha_despacho")
                .copy()
            )

            df_cd["dia_semana"] = df_cd["fecha_despacho"].dt.dayofweek
            df_cd["es_fin_semana"] = (
                df_cd["dia_semana"].isin([5, 6]).astype(int)
            )
            df_cd["dia_mes"] = df_cd["fecha_despacho"].dt.day

            for lag in self.lags:
                df_cd[f"lag_{lag}"] = df_cd["unidades_reales"].shift(lag)

            for w in self.windows:
                df_cd[f"rolling_mean_{w}"] = (
                    df_cd["unidades_reales"].shift(1).rolling(w).mean()
                )
                df_cd[f"rolling_std_{w}"] = (
                    df_cd["unidades_reales"].shift(1).rolling(w).std()
                )
                df_cd[f"rolling_max_{w}"] = (
                    df_cd["unidades_reales"].shift(1).rolling(w).max()
                )

            features_list.append(df_cd)

        return pd.concat(features_list, ignore_index=True)

    def merge_with_forecast(
        self, df_pronostico: pd.DataFrame, df_features: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(
            df_pronostico,
            df_features,
            on=["centro_distribucion", "fecha_despacho"],
            how="left",
        )