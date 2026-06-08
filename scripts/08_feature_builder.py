from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.feature_pipeline import build_multimodal_feature_table
from prop_score_ai.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Build class-level multimodal CSVs.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--face-csv", default=None)
    parser.add_argument("--emotion-csv", default=None)
    parser.add_argument("--audio-csv", default=None)
    parser.add_argument("--transcripts-csv", default=None)
    parser.add_argument("--semantic-csv", default=None)
    parser.add_argument("--ravdess-csv", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    logger = setup_logger("feature_builder", cfg["paths"]["logs_root"] / "08_feature_builder.log")
    build_multimodal_feature_table(
        face_csv=Path(args.face_csv or (cfg["paths"]["processed_root"] / "face_features.csv")),
        emotion_csv=Path(args.emotion_csv or (cfg["paths"]["processed_root"] / "emotion_features.csv")),
        audio_csv=Path(args.audio_csv or (cfg["paths"]["processed_root"] / "audio_features.csv")),
        transcripts_csv=Path(args.transcripts_csv or (cfg["paths"]["processed_root"] / "transcripts.csv")),
        semantic_csv=Path(args.semantic_csv or (cfg["paths"]["processed_root"] / "semantic_features.csv")),
        output_processed_dir=cfg["paths"]["processed_root"],
        features_dir=cfg["paths"]["features_root"],
        class_names=cfg["labels"],
        ravdess_csv=Path(args.ravdess_csv) if args.ravdess_csv else None,
        logger=logger,
    )


if __name__ == "__main__":
    main()
