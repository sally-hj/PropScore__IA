from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import project_root
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.io_utils import copy_videos, discover_videos
from prop_score_ai.logging_utils import setup_logger
from prop_score_ai.constants import LABELS


def main():
    parser = argparse.ArgumentParser(description="Run the full PropScore_AI pipeline end-to-end.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tabular-csv", default=None, help="Path to dataset_final_catboost.csv or another raw tabular source.")
    parser.add_argument("--video-root", default=None, help="Source directory containing hot/warm/cold videos.")
    parser.add_argument("--use-mock-tabular", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    logger = setup_logger("full_pipeline", cfg["paths"]["logs_root"] / "run_full_pipeline.log")

    from prop_score_ai.audio_pipeline import process_audio_dataset, transcribe_audio_dataset
    from prop_score_ai.feature_pipeline import build_multimodal_feature_table
    from prop_score_ai.fusion_pipeline import build_fused_dataset, load_multimodal_features
    from prop_score_ai.tabular_pipeline import load_or_create_tabular_data, prepare_tabular_dataset
    from prop_score_ai.training_pipeline import train_xgboost_model
    from prop_score_ai.video_training import train_video_xgboost_model
    from prop_score_ai.video_pipeline import process_emotion_features_dataset, process_face_features_dataset

    tabular_input = Path(args.tabular_csv or cfg["tabular"]["input_csv"])
    if not tabular_input.is_absolute():
        tabular_input = (project_root() / tabular_input).resolve()

    prospects_csv = cfg["paths"]["processed_root"] / "prospects.csv"
    tabular_report = cfg["paths"]["metrics_root"] / "tabular_dataset_report.json"
    if tabular_input.exists():
        prepare_tabular_dataset(tabular_input, prospects_csv, tabular_report, logger=logger)
    elif args.use_mock_tabular or bool(cfg["tabular"]["use_mock_data"]):
        logger.warning("Raw tabular input not found. Falling back to mock tabular data for demo only.")
        mock_df = load_or_create_tabular_data(
            prospects_csv,
            multimodal_df=pd.DataFrame(),
            output_mock_path=cfg["paths"]["processed_root"] / "prospects_mock.csv",
            use_mock=True,
            random_state=int(cfg["tabular"]["random_state"]),
        )
        mock_df.to_csv(prospects_csv, index=False)
    else:
        raise FileNotFoundError(f"Tabular input not found: {tabular_input}")

    import check_dataset_leakage as leakage_script  # type: ignore

    # Call the leakage checker in-process by simulating CLI arguments.
    import sys

    argv_backup = sys.argv[:]
    try:
        sys.argv = ["check_dataset_leakage.py", "--input", str(prospects_csv)]
        leakage_script.main()
    finally:
        sys.argv = argv_backup

    video_root_arg = args.video_root or cfg["paths"]["source_video_root"]
    source_videos = Path(video_root_arg)
    if not source_videos.is_absolute():
        source_videos = (project_root() / source_videos).resolve()

    raw_root = cfg["paths"]["raw_videos_root"]
    available_feature_files: list[str] = []
    has_videos = source_videos.exists() and any(discover_videos(source_videos / label) for label in LABELS)
    if has_videos:
        copy_videos(source_videos, raw_root, LABELS)
        logger.info("Copied videos from %s to %s", source_videos, raw_root)

        process_face_features_dataset(
            video_root=raw_root,
            frames_root=cfg["paths"]["frames_root"],
            output_csv=cfg["paths"]["processed_root"] / "face_features.csv",
            class_names=LABELS,
            sample_fps=float(cfg["pipeline"]["frame_sample_fps"]),
            logger=logger,
        )
        process_emotion_features_dataset(
            video_root=raw_root,
            frames_root=cfg["paths"]["frames_root"],
            output_csv=cfg["paths"]["processed_root"] / "emotion_features.csv",
            class_names=LABELS,
            sample_fps=float(cfg["pipeline"]["frame_sample_fps"]),
            logger=logger,
        )
        process_audio_dataset(
            video_root=raw_root,
            audio_root=cfg["paths"]["audio_root"],
            output_csv=cfg["paths"]["processed_root"] / "audio_features.csv",
            class_names=LABELS,
            sample_rate=int(cfg["audio"]["sample_rate"]),
            logger=logger,
        )
        transcribe_audio_dataset(
            audio_root=cfg["paths"]["audio_root"],
            output_csv=cfg["paths"]["processed_root"] / "transcripts.csv",
            model_size=cfg["whisper"]["model_size"],
            class_names=LABELS,
            logger=logger,
        )
        from prop_score_ai.nlp_pipeline import process_semantic_dataset

        process_semantic_dataset(
            transcripts_csv=cfg["paths"]["processed_root"] / "transcripts.csv",
            output_csv=cfg["paths"]["processed_root"] / "semantic_features.csv",
            model_name=cfg["camembert"]["model_name"],
            logger=logger,
        )
        build_multimodal_feature_table(
            face_csv=cfg["paths"]["processed_root"] / "face_features.csv",
            emotion_csv=cfg["paths"]["processed_root"] / "emotion_features.csv",
            audio_csv=cfg["paths"]["processed_root"] / "audio_features.csv",
            transcripts_csv=cfg["paths"]["processed_root"] / "transcripts.csv",
            semantic_csv=cfg["paths"]["processed_root"] / "semantic_features.csv",
            output_processed_dir=cfg["paths"]["processed_root"],
            features_dir=cfg["paths"]["features_root"],
            class_names=LABELS,
            logger=logger,
        )
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
                cfg["paths"]["features_root"],
                labels=selected_labels,
                include_ravdess=bool(multimodal_cfg.get("use_ravdess_csv", False)),
                ravdess_mode=str(multimodal_cfg.get("ravdess_mode", "auxiliary_only")),
                logger=logger,
            )
        else:
            logger.warning("No main multimodal class CSVs are enabled in config; multimodal fusion will be tabular-only.")
            multimodal_df = pd.DataFrame()
        available_feature_files = []
        for label in selected_labels:
            feature_path = cfg["paths"]["features_root"] / f"{label}.csv"
            if feature_path.exists() and feature_path.stat().st_size > 0:
                available_feature_files.append(feature_path.name)
            else:
                logger.warning("Missing multimodal feature file: %s", feature_path)
        if bool(multimodal_cfg.get("use_ravdess_csv", False)) and str(multimodal_cfg.get("ravdess_mode", "auxiliary_only")) == "train_with_main":
            ravdess_path = cfg["paths"]["features_root"] / "ravdess.csv"
            if ravdess_path.exists() and ravdess_path.stat().st_size > 0:
                available_feature_files.append(ravdess_path.name)
            else:
                logger.warning("Requested RAVDESS support is enabled but %s is missing.", ravdess_path)

        try:
            train_video_xgboost_model(
                features_root=cfg["paths"]["features_root"],
                output_dataset_csv=cfg["paths"]["processed_root"] / "video_training_dataset.csv",
                model_path=cfg["paths"]["models_root"] / "video_xgboost_model.pkl",
                encoders_path=cfg["paths"]["models_root"] / "video_encoders.pkl",
                feature_schema_path=cfg["paths"]["models_root"] / "video_feature_schema.json",
                metrics_dir=cfg["paths"]["metrics_root"],
                random_state=int(cfg["tabular"]["random_state"]),
                test_size=float(cfg["tabular"]["test_size"]),
                xgb_params=cfg["xgboost"],
                logger=logger,
            )
            logger.info("Trained dedicated video model from feature tables.")
        except Exception as exc:
            logger.warning("Video model training skipped: %s", exc)
    else:
        logger.info("No video folder or video files detected. Using existing multimodal feature CSVs if available.")
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
                cfg["paths"]["features_root"],
                labels=selected_labels,
                include_ravdess=bool(multimodal_cfg.get("use_ravdess_csv", False)),
                ravdess_mode=str(multimodal_cfg.get("ravdess_mode", "auxiliary_only")),
                logger=logger,
            )
            available_feature_files = []
            for label in selected_labels:
                feature_path = cfg["paths"]["features_root"] / f"{label}.csv"
                if feature_path.exists() and feature_path.stat().st_size > 0:
                    available_feature_files.append(feature_path.name)
                else:
                    logger.warning("Missing multimodal feature file: %s", feature_path)
            if bool(multimodal_cfg.get("use_ravdess_csv", False)) and str(multimodal_cfg.get("ravdess_mode", "auxiliary_only")) == "train_with_main":
                ravdess_path = cfg["paths"]["features_root"] / "ravdess.csv"
                if ravdess_path.exists() and ravdess_path.stat().st_size > 0:
                    available_feature_files.append(ravdess_path.name)
                else:
                    logger.warning("Requested RAVDESS support is enabled but %s is missing.", ravdess_path)
        else:
            logger.warning("No main multimodal class CSVs are enabled in config; multimodal fusion will be tabular-only.")
            multimodal_df = pd.DataFrame()
            available_feature_files = []

    multimodal_output_csv = cfg["paths"]["processed_root"] / "multimodal_features.csv"
    multimodal_output_csv.parent.mkdir(parents=True, exist_ok=True)
    if multimodal_df.empty:
        pd.DataFrame().to_csv(multimodal_output_csv, index=False)
    else:
        multimodal_df.to_csv(multimodal_output_csv, index=False)

    tabular_df = pd.read_csv(prospects_csv)
    selected_labels = [
        label
        for label, enabled in [
            ("hot", bool(cfg.get("multimodal", {}).get("use_hot_csv", True))),
            ("warm", bool(cfg.get("multimodal", {}).get("use_warm_csv", True))),
            ("cold", bool(cfg.get("multimodal", {}).get("use_cold_csv", True))),
        ]
        if enabled
    ]
    main_feature_files = {f"{label}.csv" for label in selected_labels}
    main_files_complete = main_feature_files.issubset(set(available_feature_files))
    ravdess_enabled_for_training = bool(cfg.get("multimodal", {}).get("use_ravdess_csv", False) and str(cfg.get("multimodal", {}).get("ravdess_mode", "auxiliary_only")) == "train_with_main")
    fusion_mode = cfg["fusion"]["mode"]
    allow_synthetic_label_pairing = bool(cfg["fusion"]["allow_synthetic_label_pairing"])
    if ravdess_enabled_for_training:
        logger.warning(
            "WARNING: RAVDESS is an external emotion dataset and may not represent real estate prospect behavior."
        )
        allow_synthetic_label_pairing = True
    elif not main_files_complete:
        if fusion_mode == "demo_synthetic" or allow_synthetic_label_pairing:
            logger.warning(
                "Main multimodal prospect files are incomplete (%s). Disabling synthetic pairing so the pipeline does not pretend the prospect multimodal model is complete.",
                sorted(available_feature_files),
            )
        allow_synthetic_label_pairing = False
        if fusion_mode == "demo_synthetic":
            fusion_mode = "auto"
    fused_df, fusion_report = build_fused_dataset(
        tabular_df,
        multimodal_df,
        cfg["paths"]["processed_root"] / "fused_dataset.csv",
        cfg["paths"]["metrics_root"] / "fusion_report.json",
        mode=fusion_mode,
        allow_synthetic_label_pairing=allow_synthetic_label_pairing,
        logger=logger,
    )

    train_xgboost_model(
        dataset_csv=cfg["paths"]["processed_root"] / "fused_dataset.csv",
        model_path=cfg["paths"]["models_root"] / "xgboost_model.pkl",
        encoders_path=cfg["paths"]["models_root"] / "encoders.pkl",
        feature_schema_path=cfg["paths"]["models_root"] / "feature_schema.json",
        metrics_dir=cfg["paths"]["metrics_root"],
        random_state=int(cfg["tabular"]["random_state"]),
        test_size=float(cfg["tabular"]["test_size"]),
        xgb_params=cfg["xgboost"],
        use_score_lead=bool(cfg["tabular"]["use_score_lead"]),
        use_binary_target_as_feature=bool(cfg["tabular"]["use_binary_target_as_feature"]),
        target_column=str(cfg["tabular"]["target_column"]),
        training_context={
            "model_type": fusion_report.get("fusion_mode_used", "tabular_only"),
            "used_multimodal_files": available_feature_files if fusion_report.get("fusion_mode_used") != "tabular_only" else [],
            "used_ravdess": bool(cfg.get("multimodal", {}).get("use_ravdess_csv", False) and str(cfg.get("multimodal", {}).get("ravdess_mode", "auxiliary_only")) == "train_with_main"),
            "ravdess_mode": str(cfg.get("multimodal", {}).get("ravdess_mode", "auxiliary_only")),
        },
        logger=logger,
    )

    logger.info("Full pipeline finished. Fusion mode used: %s", fusion_report.get("fusion_mode_used"))


if __name__ == "__main__":
    main()
