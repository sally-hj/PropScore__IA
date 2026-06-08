from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.video_pipeline import process_emotion_features_dataset
from prop_score_ai.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run DeepFace emotion analysis over frames.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--sample-fps", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    video_root = Path(args.video_root or cfg["paths"]["source_video_root"])
    frames_root = Path(args.frames_root or cfg["paths"]["frames_root"])
    output_csv = Path(args.output_csv or (cfg["paths"]["processed_root"] / "emotion_features.csv"))
    sample_fps = args.sample_fps if args.sample_fps is not None else float(cfg["pipeline"]["frame_sample_fps"])
    logger = setup_logger("emotion_deepface", cfg["paths"]["logs_root"] / "04_emotion_deepface.log")
    process_emotion_features_dataset(video_root, frames_root, output_csv, class_names=cfg["labels"], sample_fps=sample_fps, logger=logger)


if __name__ == "__main__":
    main()
