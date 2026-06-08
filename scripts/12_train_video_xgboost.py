from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import project_root
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.logging_utils import setup_logger
from prop_score_ai.video_training import train_video_xgboost_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dedicated XGBoost model on the video feature tables.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--features-root", default=None, help="Folder containing hot.csv, warm.csv, cold.csv.")
    parser.add_argument("--output-dataset", default=None, help="Path to save the combined video training dataset.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--encoders-path", default=None)
    parser.add_argument("--feature-schema-path", default=None)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--test-size", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    logger = setup_logger("train_video_xgboost", cfg["paths"]["logs_root"] / "12_train_video_xgboost.log")

    features_root = Path(args.features_root or cfg["paths"]["features_root"])
    if not features_root.is_absolute():
        features_root = (project_root() / features_root).resolve()

    result = train_video_xgboost_model(
        features_root=features_root,
        output_dataset_csv=Path(args.output_dataset or (cfg["paths"]["processed_root"] / "video_training_dataset.csv")),
        model_path=Path(args.model_path or (cfg["paths"]["models_root"] / "video_xgboost_model.pkl")),
        encoders_path=Path(args.encoders_path or (cfg["paths"]["models_root"] / "video_encoders.pkl")),
        feature_schema_path=Path(args.feature_schema_path or (cfg["paths"]["models_root"] / "video_feature_schema.json")),
        metrics_dir=Path(args.metrics_dir or cfg["paths"]["metrics_root"]),
        random_state=int(cfg["tabular"]["random_state"]),
        test_size=float(args.test_size if args.test_size is not None else cfg["tabular"]["test_size"]),
        xgb_params=cfg["xgboost"],
        logger=logger,
    )
    logger.info("Video model metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
