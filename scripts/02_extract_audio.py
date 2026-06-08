from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.audio_pipeline import extract_audio_from_video
from prop_score_ai.io_utils import discover_videos, ensure_dir, video_id_from_path
from prop_score_ai.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Extract audio tracks from raw videos.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    video_root = Path(args.video_root or cfg["paths"]["source_video_root"])
    audio_root = Path(args.audio_root or cfg["paths"]["audio_root"])
    logger = setup_logger("extract_audio", cfg["paths"]["logs_root"] / "02_extract_audio.log")

    rows = []
    for class_name in cfg["labels"]:
        for video_path in discover_videos(video_root / class_name):
            video_id = video_id_from_path(video_path)
            out_dir = ensure_dir(audio_root / class_name)
            audio_path = out_dir / f"{video_id}.wav"
            extracted = extract_audio_from_video(video_path, audio_path, sample_rate=args.sample_rate)
            rows.append(
                {
                    "video_id": video_id,
                    "label": class_name,
                    "audio_path": str(extracted or audio_path),
                    "status": "ok" if extracted else "failed",
                }
            )
            logger.info("Audio extraction %s for %s", "ok" if extracted else "failed", video_path.name)

    pd.DataFrame(rows).to_csv(cfg["paths"]["processed_root"] / "audio_index.csv", index=False)


if __name__ == "__main__":
    main()
