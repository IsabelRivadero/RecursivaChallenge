import numpy as np
import pandas as pd


class MetricsEvaluator:
    """Cálculo estandarizado de métricas de precisión y sesgo."""

    @staticmethod
    def calculate_wape(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Weighted Absolute Percentage Error."""

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) == 0:
            return float("nan")

        denominator = np.sum(np.abs(y_true))

        if denominator == 0:
            return float("nan")

        return float(
            np.sum(np.abs(y_true - y_pred)) / denominator
        )

    @staticmethod
    def calculate_bias(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Bias normalizado.

        Positivo: sobrepronóstico.
        Negativo: subpronóstico.
        """

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) == 0:
            return float("nan")

        denominator = np.sum(np.abs(y_true))

        if denominator == 0:
            return float("nan")

        return float(
            np.sum(y_pred - y_true) / denominator
        )

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        col_true: str,
        col_pred: str,
    ) -> dict[str, float]:
        """Calcula las métricas sobre las filas válidas."""

        valid = df[[col_true, col_pred]].dropna()

        y_true = valid[col_true].to_numpy(dtype=float)
        y_pred = valid[col_pred].to_numpy(dtype=float)

        return {
            "wape": self.calculate_wape(y_true, y_pred),
            "bias": self.calculate_bias(y_true, y_pred),
        }