"""Helpers for training a dedicated video multimodal classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import LABELS
from .io_utils import dataframe_to_csv
from .logging_utils import setup_logger
from .training_pipeline import train_xgboost_model


VIDEO_DROP_COLUMNS = {
    "source_class",
    "final_score",
}


def load_video_feature_tables(features_root: str | Path, labels: Iterable[str] = LABELS) -> pd.DataFrame:
    features_root = Path(features_root)
    frames: list[pd.DataFrame] = []
    for label in labels:
        path = features_root / f"{label}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        df = df.copy()
        df["label"] = label
        df["source_class"] = label
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop(columns=[c for c in VIDEO_DROP_COLUMNS if c in combined.columns], errors="ignore")
    return combined


def train_video_xgboost_model(
    *,
    features_root: str | Path,
    output_dataset_csv: str | Path,
    model_path: str | Path,
    encoders_path: str | Path,
    feature_schema_path: str | Path,
    metrics_dir: str | Path,
    random_state: int = 42,
    test_size: float = 0.2,
    xgb_params: dict | None = None,
    logger=None,
) -> dict:
    """Train a dedicated model on the video-derived feature tables."""
    logger = logger or setup_logger("video_training")
    video_df = load_video_feature_tables(features_root)
    if video_df.empty:
        raise ValueError("No video feature tables found for training.")

    output_dataset_csv = Path(output_dataset_csv)
    output_dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe_to_csv(video_df, output_dataset_csv)

    result = train_xgboost_model(
        dataset_csv=output_dataset_csv,
        model_path=model_path,
        encoders_path=encoders_path,
        feature_schema_path=feature_schema_path,
        metrics_dir=metrics_dir,
        random_state=random_state,
        test_size=test_size,
        xgb_params=xgb_params,
        use_score_lead=False,
        use_binary_target_as_feature=False,
        target_column="label",
        logger=logger,
    )
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    source_files = {
        metrics_dir / "metrics.json": metrics_dir / "video_metrics.json",
        metrics_dir / "classification_report.txt": metrics_dir / "video_classification_report.txt",
        metrics_dir / "confusion_matrix.png": metrics_dir / "video_confusion_matrix.png",
        metrics_dir / "split_report.csv": metrics_dir / "video_split_report.csv",
        metrics_dir / "split_report.json": metrics_dir / "video_split_report.json",
    }
    for src, dst in source_files.items():
        if src.exists():
            dst.write_bytes(src.read_bytes())
    result["video_dataset_rows"] = int(len(video_df))
    return result
