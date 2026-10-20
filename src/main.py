from pathlib import Path
import yaml

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.metrics.evaluator import MetricsEvaluator
from src.models.predict import ForecastPredictor
from src.models.train import ModelTrainer


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    loader = DataLoader(config)
    df_pedidos = loader.load_pedidos()
    df_pronostico = loader.load_pronostico()

    df_reales_agg = loader.get_aggregated_actuals(df_pedidos)

    feature_builder = FeatureBuilder(config)
    df_features = feature_builder.build_features(df_reales_agg)
    df_dataset = feature_builder.merge_with_forecast(
        df_pronostico, df_features
    )

    trainer = ModelTrainer(config)
    df_backtest = trainer.run_rolling_origin_backtest(df_dataset)

    evaluator = MetricsEvaluator()
    base_metrics = evaluator.evaluate_dataframe(
        df_backtest, "unidades_reales", "unidades_pronosticadas"
    )
    ml_metrics = evaluator.evaluate_dataframe(
        df_backtest, "unidades_reales", "unidades_modelo"
    )

    print("=== RESULTADOS BACKTESTING ===")
    print(
        f"Base Model - WAPE: {base_metrics['wape']:.4f} | Bias:"
        f" {base_metrics['bias']:.4f}"
    )
    print(
        f"ML Model   - WAPE: {ml_metrics['wape']:.4f} | Bias:"
        f" {ml_metrics['bias']:.4f}"
    )

    trained_models = trainer.train_final_models(df_pedidos, df_dataset)
    predictor = ForecastPredictor(trained_models, config)
    df_final = predictor.predict_df(df_dataset)

    output_dir = Path(config["paths"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_dir / "pronostico_corregido.csv", index=False)
    print(f"Predicciones guardadas en: {output_dir / 'pronostico_corregido.csv'}")


if __name__ == "__main__":
    main()