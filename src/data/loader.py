from pathlib import Path
import pandas as pd


class DataLoader:
    """Carga y validación de esquemas de datos de entrada."""

    def __init__(self, config: dict):
        self.config = config

    def load_pedidos(self, path: str | None = None) -> pd.DataFrame:
        file_path = Path(path or self.config["paths"]["raw_pedidos"])
        df = pd.read_csv(file_path)
        df["fecha_despacho"] = pd.to_datetime(df["fecha_despacho"])
        return df

    def load_pronostico(self, path: str | None = None) -> pd.DataFrame:
        file_path = Path(path or self.config["paths"]["raw_pronostico"])
        df = pd.read_csv(file_path)
        df["fecha_despacho"] = pd.to_datetime(df["fecha_despacho"])
        df["fecha_emision"] = pd.to_datetime(df["fecha_emision"])
        return df

    def load_capacidad(self, path: str | None = None) -> pd.DataFrame:
        file_path = Path(path or self.config["paths"]["raw_capacidad"])
        df = pd.read_csv(file_path)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    def load_plan(self, path: str | None = None) -> pd.DataFrame:
        file_path = Path(path or self.config["paths"]["raw_plan"])
        df = pd.read_csv(file_path)
        df["fecha_plan"] = pd.to_datetime(df["fecha_plan"])
        df["fecha_compra_real"] = pd.to_datetime(df["fecha_compra_real"])
        return df

    def get_aggregated_actuals(self, df_pedidos: pd.DataFrame) -> pd.DataFrame:
        return (
            df_pedidos.groupby(
                ["centro_distribucion", "fecha_despacho"], as_index=False
            )["unidades_reales"]
            .sum()
            .sort_values(["centro_distribucion", "fecha_despacho"])
            .reset_index(drop=True)
        )