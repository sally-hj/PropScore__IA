from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from prop_score_ai.audio_pipeline import process_audio_dataset
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Extract Librosa audio features from videos.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    video_root = Path(args.video_root or cfg["paths"]["source_video_root"])
    audio_root = Path(args.audio_root or cfg["paths"]["audio_root"])
    output_csv = Path(args.output_csv or (cfg["paths"]["processed_root"] / "audio_features.csv"))
    logger = setup_logger("librosa_features", cfg["paths"]["logs_root"] / "06_librosa_features.log")
    process_audio_dataset(video_root, audio_root, output_csv, class_names=cfg["labels"], sample_rate=args.sample_rate, logger=logger)


if __name__ == "__main__":
    main()
