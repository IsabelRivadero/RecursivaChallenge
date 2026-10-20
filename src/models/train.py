import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from src.metrics.evaluator import MetricsEvaluator
from src.resilience.gate import ResilienceGate


class ModelTrainer:
    """Entrenamiento y evaluación mediante origen rodante."""

    def __init__(self, config: dict):
        self.config = config
        self.feature_cols = config["features"]["cols"]
        self.random_state = config["model"]["random_state"]
        self.max_iter = config["model"]["max_iter"]

        self.gate = ResilienceGate(
            config["model"]["min_history_days"]
        )
        self.evaluator = MetricsEvaluator()

    def _create_model(self):
        """Construye una instancia nueva del modelo."""

        return HistGradientBoostingRegressor(
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Selecciona las variables configuradas."""

        missing = [
            col for col in self.feature_cols
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Faltan variables configuradas: {missing}"
            )

        X = df[self.feature_cols].copy()

        # El modelo admite valores NaN, pero no infinitos.
        X = X.replace([np.inf, -np.inf], np.nan)

        return X

    def run_rolling_origin_backtest(
        self,
        df_dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Evalúa fechas futuras utilizando únicamente registros
        anteriores a cada fecha de corte.

        El modelo del backtest se entrena globalmente con datos
        de todos los centros de distribución.
        """

        required = [
            "centro_distribucion",
            "fecha_despacho",
            "horizonte",
            "unidades_reales",
            "unidades_pronosticadas",
        ]

        missing = [
            col for col in required
            if col not in df_dataset.columns
        ]

        if missing:
            raise ValueError(
                f"Faltan columnas para backtesting: {missing}"
            )

        df = df_dataset.copy()

        df["fecha_despacho"] = pd.to_datetime(
            df["fecha_despacho"]
        )

        df = df.sort_values(
            ["fecha_despacho", "centro_distribucion"]
        )

        # Excluir el período inicial sin historial suficiente.
        eval_df = df.dropna(
            subset=["rolling_mean_21"]
        ).copy()

        # Se conserva la fecha de inicio de evaluación utilizada
        # en el proyecto original.
        start_date = pd.Timestamp("2026-02-01")

        test_dates = sorted(
            eval_df.loc[
                eval_df["fecha_despacho"] >= start_date,
                "fecha_despacho",
            ].unique()
        )

        predictions = []

        for cutoff_date in test_dates:
            train = eval_df[
                eval_df["fecha_despacho"] < cutoff_date
            ].copy()

            test = eval_df[
                eval_df["fecha_despacho"] == cutoff_date
            ].copy()

            # Evitar entrenamientos con muy pocos registros.
            if len(train) < 50 or test.empty:
                continue

            X_train = self._prepare_features(train)
            y_train = train["unidades_reales"].astype(float)

            X_test = self._prepare_features(test)

            # Evitar que los valores objetivo inválidos
            # entren en el entrenamiento.
            valid_train = y_train.notna()

            X_train = X_train.loc[valid_train]
            y_train = y_train.loc[valid_train]

            if len(X_train) < 50:
                continue

            model = self._create_model()
            model.fit(X_train, y_train)

            result_cols = [
                "centro_distribucion",
                "fecha_despacho",
                "horizonte",
                "unidades_reales",
                "unidades_pronosticadas",
            ]

            test_res = test[result_cols].copy()

            test_res["unidades_modelo"] = model.predict(
                X_test
            )

            predictions.append(test_res)

        if not predictions:
            return pd.DataFrame(
                columns=[
                    "centro_distribucion",
                    "fecha_despacho",
                    "horizonte",
                    "unidades_reales",
                    "unidades_pronosticadas",
                    "unidades_modelo",
                ]
            )

        return pd.concat(
            predictions,
            ignore_index=True,
        )

    def train_final_models(
        self,
        df_pedidos: pd.DataFrame,
        df_dataset: pd.DataFrame,
    ) -> dict:
        """
        Entrena un modelo final por centro de distribución
        cuando supera el control de historia mínima.
        """

        models = {}

        cds = df_dataset[
            "centro_distribucion"
        ].dropna().unique()

        for cd in cds:
            df_pedidos_cd = df_pedidos[
                df_pedidos["centro_distribucion"] == cd
            ].copy()

            gate_eval = self.gate.evaluate_cd_history(
                df_pedidos_cd,
                cd,
            )

            if not gate_eval["habilitado_ml"]:
                print(
                    f"[COLD START] {cd}: "
                    "se utilizará el pronóstico base."
                )
                continue

            df_cd = df_dataset[
                df_dataset["centro_distribucion"] == cd
            ].copy()

            # Requerir suficiente historial para las variables
            # móviles de 21 días.
            df_cd = df_cd.dropna(
                subset=["rolling_mean_21", "unidades_reales"]
            )

            if len(df_cd) < 2:
                print(
                    f"[WARNING] {cd}: datos insuficientes "
                    "para entrenar el modelo."
                )
                continue

            X = self._prepare_features(df_cd)
            y = df_cd["unidades_reales"].astype(float)

            valid = y.notna()

            X = X.loc[valid]
            y = y.loc[valid]

            if len(X) < 2:
                continue

            model = self._create_model()
            model.fit(X, y)

            models[cd] = model

        return models