import pandas as pd


class ResilienceGate:
    """Mecanismo de control de resguardo de historia mínima (Cold Start)."""

    def __init__(self, min_days: int = 45):
        self.min_days = min_days

    def evaluate_cd_history(
        self, df_pedidos_cd: pd.DataFrame, cd_name: str
    ) -> dict:
        unique_days = df_pedidos_cd["fecha_despacho"].nunique()
        is_active = unique_days >= self.min_days

        return {
            "centro_distribucion": cd_name,
            "dias_disponibles": unique_days,
            "umbral_minimo": self.min_days,
            "habilitado_ml": is_active,
            "estado": "HABILITADO_ML" if is_active else "FALLBACK_BASE",
        }