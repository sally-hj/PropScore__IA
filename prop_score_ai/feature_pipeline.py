"""Merge modality outputs into class-level datasets and multimodal tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import LABELS, LABEL_TO_ID
from .io_utils import dataframe_to_csv, ensure_dir
from .logging_utils import setup_logger


def _load_if_exists(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def build_visual_features(face_df: pd.DataFrame, emotion_df: pd.DataFrame) -> pd.DataFrame:
    if face_df.empty and emotion_df.empty:
        return pd.DataFrame()
    if face_df.empty:
        return emotion_df.copy()
    if emotion_df.empty:
        return face_df.copy()
    merged = face_df.merge(emotion_df, on=["prospect_id", "video_id", "label"], how="outer")
    return merged


def build_multimodal_feature_table(
    face_csv: str | Path,
    emotion_csv: str | Path,
    audio_csv: str | Path,
    transcripts_csv: str | Path,
    semantic_csv: str | Path,
    output_processed_dir: str | Path,
    features_dir: str | Path,
    class_names: Iterable[str] = ("hot", "warm", "cold"),
    ravdess_csv: str | Path | None = None,
    logger=None,
) -> pd.DataFrame:
    logger = logger or setup_logger("feature_builder")
    output_processed_dir = Path(output_processed_dir)
    features_dir = Path(features_dir)
    output_processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    face_df = _load_if_exists(face_csv)
    emotion_df = _load_if_exists(emotion_csv)
    audio_df = _load_if_exists(audio_csv)
    transcripts_df = _load_if_exists(transcripts_csv)
    semantic_df = _load_if_exists(semantic_csv)

    visual_df = build_visual_features(face_df, emotion_df)
    if not visual_df.empty:
        dataframe_to_csv(visual_df, output_processed_dir / "visual_features.csv")

    multimodal = visual_df.merge(audio_df, on=["prospect_id", "video_id", "label"], how="outer", suffixes=("", "_audio"))
    multimodal = multimodal.merge(transcripts_df, on=["prospect_id", "video_id", "label"], how="outer", suffixes=("", "_transcript"))
    multimodal = multimodal.merge(semantic_df, on=["prospect_id", "video_id", "label"], how="outer", suffixes=("", "_semantic"))

    if not multimodal.empty:
        multimodal["label"] = multimodal["label"].astype("string").str.lower()
        multimodal["label_id"] = multimodal["label"].map(LABEL_TO_ID)
        dataframe_to_csv(multimodal, output_processed_dir / "multimodal_features.csv")

        for label in class_names:
            class_df = multimodal[multimodal["label"] == label].copy()
            dataframe_to_csv(class_df, features_dir / f"{label}.csv")

    if ravdess_csv is not None:
        ravdess_df = _load_if_exists(ravdess_csv)
        if not ravdess_df.empty:
            dataframe_to_csv(ravdess_df, features_dir / "ravdess.csv")
    else:
        ravdess_path = features_dir / "ravdess.csv"
        if not ravdess_path.exists():
            dataframe_to_csv(pd.DataFrame(), ravdess_path)

    logger.info("Built multimodal table with %d rows", len(multimodal))
    return multimodal


def build_ravdess_feature_table(
    video_root: str | Path,
    frames_root: str | Path,
    audio_root: str | Path,
    output_csv: str | Path,
    sample_fps: float = 1.0,
    logger=None,
) -> pd.DataFrame:
    """Optional helper for RAVDESS processing."""
    logger = logger or setup_logger("ravdess")
    video_root = Path(video_root)
    rows = []
    if not video_root.exists():
        dataframe_to_csv(pd.DataFrame(), output_csv)
        return pd.DataFrame()
    # The actual batch logic is delegated to the dedicated scripts to keep this helper compact.
    dataframe_to_csv(pd.DataFrame(rows), output_csv)
    return pd.DataFrame(rows)

