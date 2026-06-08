from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from prop_score_ai.audio_pipeline import extract_audio_from_video, extract_librosa_features
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.io_utils import discover_videos, ensure_dir, video_id_from_path
from prop_score_ai.logging_utils import setup_logger
from prop_score_ai.video_pipeline import analyze_video_visual_signals


def main():
    parser = argparse.ArgumentParser(description="Optional RAVDESS audio/emotion feature extraction.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ravdess-root", default=None)
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--sample-fps", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    ravdess_root = Path(args.ravdess_root or (cfg["paths"]["raw_videos_root"] / "ravdess"))
    frames_root = Path(args.frames_root or (cfg["paths"]["frames_root"] / "ravdess"))
    audio_root = Path(args.audio_root or (cfg["paths"]["audio_root"] / "ravdess"))
    output_csv = Path(args.output_csv or (cfg["paths"]["features_root"] / "ravdess.csv"))
    sample_fps = args.sample_fps if args.sample_fps is not None else float(cfg["pipeline"]["frame_sample_fps"])
    logger = setup_logger("ravdess_support", cfg["paths"]["logs_root"] / "11_ravdess_support.log")

    rows = []
    for video_path in discover_videos(ravdess_root):
        video_id = video_id_from_path(video_path)
        frame_dir = ensure_dir(frames_root / video_id)
        audio_dir = ensure_dir(audio_root)
        audio_path = audio_dir / f"{video_id}.wav"
        extract_audio_from_video(video_path, audio_path)
        visual = analyze_video_visual_signals(video_path, frame_dir, sample_fps=sample_fps)
        audio = extract_librosa_features(audio_path)
        rows.append({"video_id": video_id, **visual, **audio})

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved RAVDESS features to %s", output_csv)


if __name__ == "__main__":
    main()
