from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import project_root
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.fusion_pipeline import build_fused_dataset, load_multimodal_features
from prop_score_ai.logging_utils import setup_logger
from prop_score_ai.tabular_pipeline import clean_tabular_dataset, load_or_create_tabular_data
from utils.feature_schema import clean_multimodal_dataframe


def main():
    parser = argparse.ArgumentParser(description="Fuse tabular and multimodal datasets.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tabular-csv", default=None, help="Cleaned prospects.csv path.")
    parser.add_argument("--multimodal-csv", default=None, help="Optional prebuilt multimodal csv.")
    parser.add_argument("--features-root", default=None, help="Folder containing hot.csv/warm.csv/cold.csv.")
    parser.add_argument("--output-fused", default=None)
    parser.add_argument("--output-multimodal", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    logger = setup_logger("merge_datasets", cfg["paths"]["logs_root"] / "09_merge_datasets.log")

    tabular_csv = Path(args.tabular_csv or cfg["tabular"]["input_csv"])
    if not tabular_csv.is_absolute():
        tabular_csv = (project_root() / tabular_csv).resolve()
    tabular_df = load_or_create_tabular_data(
        tabular_csv,
        multimodal_df=None,
        output_mock_path=cfg["paths"]["processed_root"] / "prospects_mock.csv",
        use_mock=bool(cfg["tabular"]["use_mock_data"]),
        random_state=int(cfg["tabular"]["random_state"]),
    )
    tabular_df = clean_tabular_dataset(tabular_df)

    multimodal_df = pd.DataFrame()
    if args.multimodal_csv:
        multimodal_path = Path(args.multimodal_csv)
        if multimodal_path.exists() and multimodal_path.stat().st_size > 0:
            multimodal_df = pd.read_csv(multimodal_path)
            multimodal_df = clean_multimodal_dataframe(multimodal_df, dataset_name=multimodal_path.stem, add_prospect_id=True)
    else:
        features_root = Path(args.features_root or cfg["paths"]["features_root"])
        if not features_root.is_absolute():
            features_root = (project_root() / features_root).resolve()
        multimodal_cfg = cfg.get("multimodal", {})
        selected_labels = [
            label
            for label, enabled in [
                ("hot", bool(multimodal_cfg.get("use_hot_csv", True))),
                ("warm", bool(multimodal_cfg.get("use_warm_csv", True))),
                ("cold", bool(multimodal_cfg.get("use_cold_csv", True))),
            ]
            if enabled
        ]
        if selected_labels:
            multimodal_df = load_multimodal_features(
                features_root,
                labels=selected_labels,
                include_ravdess=bool(multimodal_cfg.get("use_ravdess_csv", False)),
                ravdess_mode=str(multimodal_cfg.get("ravdess_mode", "auxiliary_only")),
                logger=logger,
            )
        else:
            logger.warning("No main multimodal class CSVs are enabled in config; multimodal fusion will be tabular-only.")
            multimodal_df = pd.DataFrame()

    output_multimodal = Path(args.output_multimodal or (cfg["paths"]["processed_root"] / "multimodal_features.csv"))
    output_fused = Path(args.output_fused or (cfg["paths"]["processed_root"] / "fused_dataset.csv"))
    report_path = Path(args.report or (cfg["paths"]["metrics_root"] / "fusion_report.json"))

    if not multimodal_df.empty:
        multimodal_df.to_csv(output_multimodal, index=False)
    else:
        output_multimodal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_multimodal, index=False)

    multimodal_cfg = cfg.get("multimodal", {})
    selected_labels = [
        label
        for label, enabled in [
            ("hot", bool(multimodal_cfg.get("use_hot_csv", True))),
            ("warm", bool(multimodal_cfg.get("use_warm_csv", True))),
            ("cold", bool(multimodal_cfg.get("use_cold_csv", True))),
        ]
        if enabled
    ]
    allow_synthetic_label_pairing = bool(cfg["fusion"]["allow_synthetic_label_pairing"])
    main_feature_files = {f"{label}.csv" for label in selected_labels}
    multimodal_files_available = {f"{label}.csv" for label in selected_labels if (cfg["paths"]["features_root"] / f"{label}.csv").exists()}
    main_files_complete = main_feature_files.issubset(multimodal_files_available)
    ravdess_enabled_for_training = bool(multimodal_cfg.get("use_ravdess_csv", False)) and str(multimodal_cfg.get("ravdess_mode", "auxiliary_only")) == "train_with_main"
    if ravdess_enabled_for_training:
        logger.warning(
            "WARNING: RAVDESS is an external emotion dataset and may not represent real estate prospect behavior."
        )
        allow_synthetic_label_pairing = True
    elif not main_files_complete:
        if cfg["fusion"]["mode"] == "demo_synthetic" or allow_synthetic_label_pairing:
            logger.warning(
                "Main multimodal prospect files are incomplete (%s). Disabling synthetic pairing so the pipeline does not pretend the prospect multimodal model is complete.",
                sorted(multimodal_files_available),
            )
        allow_synthetic_label_pairing = False
        if cfg["fusion"]["mode"] == "demo_synthetic":
            cfg["fusion"]["mode"] = "auto"

    fused_df, report = build_fused_dataset(
        tabular_df,
        multimodal_df,
        output_fused,
        report_path,
        mode=cfg["fusion"]["mode"],
        allow_synthetic_label_pairing=allow_synthetic_label_pairing,
        logger=logger,
    )

    logger.info("Saved fused dataset to %s", output_fused)
    logger.info("Fusion mode used: %s", report.get("fusion_mode_used"))


if __name__ == "__main__":
    main()
