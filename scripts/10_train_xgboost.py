from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import project_root
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.logging_utils import setup_logger
from prop_score_ai.training_pipeline import train_xgboost_model


def main():
    parser = argparse.ArgumentParser(description="Train the XGBoost fusion model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--encoders-path", default=None)
    parser.add_argument("--feature-schema-path", default=None)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--test-size", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    logger = setup_logger("train_xgboost", cfg["paths"]["logs_root"] / "10_train_xgboost.log")
    dataset_path = Path(args.dataset or (cfg["paths"]["processed_root"] / "fused_dataset.csv"))
    if not dataset_path.exists():
        dataset_path = cfg["paths"]["processed_root"] / "prospects.csv"
    if not dataset_path.is_absolute():
        dataset_path = (project_root() / dataset_path).resolve()

    multimodal_cfg = cfg.get("multimodal", {})
    features_root = cfg["paths"]["features_root"]
    selected_labels = [
        label
        for label, enabled in [
            ("hot", bool(multimodal_cfg.get("use_hot_csv", True))),
            ("warm", bool(multimodal_cfg.get("use_warm_csv", True))),
            ("cold", bool(multimodal_cfg.get("use_cold_csv", True))),
        ]
        if enabled
    ]
    available_feature_files = []
    if selected_labels:
        for label in selected_labels:
            feature_path = features_root / f"{label}.csv"
            if feature_path.exists() and feature_path.stat().st_size > 0:
                available_feature_files.append(feature_path.name)
            else:
                logger.warning("Missing multimodal feature file: %s", feature_path)
    else:
        logger.warning("No main multimodal class CSVs are enabled in config; training will treat the dataset as tabular-only unless the dataset already contains multimodal columns.")
    ravdess_path = features_root / "ravdess.csv"
    use_ravdess = bool(multimodal_cfg.get("use_ravdess_csv", False))
    ravdess_mode = str(multimodal_cfg.get("ravdess_mode", "auxiliary_only"))
    if use_ravdess and ravdess_mode == "train_with_main" and ravdess_path.exists() and ravdess_path.stat().st_size > 0:
        available_feature_files.append(ravdess_path.name)
    elif use_ravdess and ravdess_mode == "train_with_main":
        logger.warning("Requested RAVDESS support is enabled but %s is missing.", ravdess_path)

    dataset_columns = set()
    try:
        dataset_columns = set(pd.read_csv(dataset_path, nrows=0).columns)
    except Exception:
        pass
    multimodal_column_hints = {
        "dominant_emotion",
        "confidence_emotion",
        "pitch_mean",
        "energy_mean",
        "speech_rate",
        "mfcc_1",
        "vision_score",
        "audio_score",
        "semantic_score",
        "final_score",
        "face_detected",
        "landmark_count",
        "face_center_x",
        "face_center_y",
        "face_width",
        "face_height",
    }
    uses_multimodal_columns = bool(dataset_columns & multimodal_column_hints)
    model_type = "tabular_only"
    if "synthetic_pair_id" in dataset_columns:
        model_type = "demo_synthetic"
    elif uses_multimodal_columns:
        model_type = "hybrid"

    if model_type != "tabular_only" and not {"hot.csv", "warm.csv", "cold.csv"}.issubset(set(available_feature_files)):
        logger.warning(
            "Some main multimodal class files are missing. Current available files: %s",
            available_feature_files,
        )

    training_context = {
        "model_type": model_type,
        "used_multimodal_files": available_feature_files if model_type != "tabular_only" else [],
        "used_ravdess": bool(use_ravdess and ravdess_mode == "train_with_main" and ravdess_path.name in available_feature_files),
        "ravdess_mode": ravdess_mode,
    }
    result = train_xgboost_model(
        dataset_csv=dataset_path,
        model_path=Path(args.model_path or (cfg["paths"]["models_root"] / "xgboost_model.pkl")),
        encoders_path=Path(args.encoders_path or (cfg["paths"]["models_root"] / "encoders.pkl")),
        feature_schema_path=Path(args.feature_schema_path or (cfg["paths"]["models_root"] / "feature_schema.json")),
        metrics_dir=cfg["paths"]["metrics_root"],
        random_state=int(cfg["tabular"]["random_state"]),
        test_size=float(args.test_size if args.test_size is not None else cfg["tabular"]["test_size"]),
        xgb_params=cfg["xgboost"],
        use_score_lead=bool(cfg["tabular"]["use_score_lead"]),
        use_binary_target_as_feature=bool(cfg["tabular"]["use_binary_target_as_feature"]),
        target_column=str(cfg["tabular"]["target_column"]),
        training_context=training_context,
        logger=logger,
    )
    logger.info("Metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
