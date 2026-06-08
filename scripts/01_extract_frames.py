from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.io_utils import discover_videos, ensure_dir, prospect_id_from_video_path, video_id_from_path
from prop_score_ai.logging_utils import setup_logger
from prop_score_ai.video_pipeline import extract_frames_from_video


def main():
    parser = argparse.ArgumentParser(description="Extract frames from raw videos.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video-root", default=None, help="Root with hot/warm/cold folders.")
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    video_root = Path(args.video_root or cfg["paths"]["source_video_root"])
    frames_root = Path(args.frames_root or cfg["paths"]["frames_root"])
    sample_fps = args.sample_fps if args.sample_fps is not None else float(cfg["pipeline"]["frame_sample_fps"])
    logger = setup_logger("extract_frames", cfg["paths"]["logs_root"] / "01_extract_frames.log")

    rows = []
    for class_name in cfg["labels"]:
        for video_path in discover_videos(video_root / class_name):
            video_id = video_id_from_path(video_path)
            prospect_id = prospect_id_from_video_path(video_path, class_name)
            out_dir = ensure_dir(frames_root / class_name / video_id)
            frame_paths = extract_frames_from_video(video_path, out_dir, sample_fps=sample_fps, overwrite=args.overwrite)
            rows.append(
                {
                    "prospect_id": prospect_id,
                    "video_id": video_id,
                    "label": class_name,
                    "frames_extracted": len(frame_paths),
                    "frames_dir": str(out_dir),
                    "video_path": str(video_path),
                }
            )
            logger.info("Extracted %d frames from %s", len(frame_paths), video_path.name)

    pd.DataFrame(rows).to_csv(cfg["paths"]["processed_root"] / "frames_index.csv", index=False)


if __name__ == "__main__":
    main()
