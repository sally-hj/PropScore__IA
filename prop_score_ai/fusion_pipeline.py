"""Fusion logic for tabular-only, hybrid, and demo synthetic multimodal datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import LABELS
from .io_utils import dataframe_to_csv, save_json
from .logging_utils import setup_logger
from .tabular_pipeline import clean_tabular_dataset, get_tabular_feature_frame
from utils.feature_schema import clean_multimodal_dataframe


VIDEO_FEATURE_HINTS = (
    "dominant_emotion",
    "confidence_emotion",
    "emotion_confidence",
    "face_detected",
    "face_detected_ratio",
    "landmark_count",
    "landmark_quality_score",
    "face_center_x",
    "face_center_y",
    "face_width",
    "face_height",
    "vision_score",
    "mfcc_mean_",
    "mfcc_std_",
    "pitch_mean",
    "pitch_std",
    "energy_mean",
    "energy_std",
    "zero_crossing_rate_mean",
    "spectral_centroid_mean",
    "speech_rate_estimate",
    "audio_score",
    "transcript",
    "language",
    "duration",
    "sentiment_label",
    "sentiment_score",
    "keyword_count",
    "positive_keyword_count",
    "hesitation_keyword_count",
    "negative_keyword_count",
    "keyword_type",
    "semantic_score",
)


def _load_feature_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = clean_multimodal_dataframe(df, dataset_name=dataset_name, add_prospect_id=True)
    return df


def load_multimodal_features(
    features_root: str | Path,
    labels: Iterable[str] = LABELS,
    *,
    include_ravdess: bool = False,
    ravdess_mode: str = "auxiliary_only",
    logger=None,
) -> pd.DataFrame:
    features_root = Path(features_root)
    logger = logger or setup_logger("fusion")
    frames = []
    for label in labels:
        path = features_root / f"{label}.csv"
        if path.exists() and path.stat().st_size > 0:
            df = _load_feature_csv(path, label)
            if "label" not in df.columns:
                df["label"] = label
            frames.append(df)
        else:
            logger.warning("Missing multimodal feature file: %s", path)
    if include_ravdess and ravdess_mode == "train_with_main":
        ravdess_path = features_root / "ravdess.csv"
        if ravdess_path.exists() and ravdess_path.stat().st_size > 0:
            ravdess_df = _load_feature_csv(ravdess_path, "ravdess")
            frames.append(ravdess_df)
        else:
            logger.warning("Missing multimodal feature file: %s", ravdess_path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_mode(mode: str | None) -> str:
    mode = (mode or "auto").strip().lower()
    return mode if mode in {"auto", "tabular_only", "hybrid", "demo_synthetic"} else "auto"


def _multimodal_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if any(hint in col for hint in VIDEO_FEATURE_HINTS):
            cols.append(col)
    return cols


def _merge_on_prospect_id(tabular_df: pd.DataFrame, multimodal_df: pd.DataFrame) -> pd.DataFrame:
    merged = tabular_df.merge(multimodal_df, on="prospect_id", how="inner", suffixes=("", "_video"))
    if "label_video" in merged.columns:
        merged["label"] = merged["label"].fillna(merged["label_video"])
    return merged


def _synthetic_pair_by_label(tabular_df: pd.DataFrame, multimodal_df: pd.DataFrame) -> pd.DataFrame:
    paired_frames = []
    for label in LABELS:
        left = tabular_df[tabular_df["label"] == label].reset_index(drop=True)
        right = multimodal_df[multimodal_df["label"] == label].reset_index(drop=True)
        if left.empty or right.empty:
            continue
        n = max(len(left), len(right))
        left_idx = np.arange(n) % len(left)
        right_idx = np.arange(n) % len(right)
        left_rep = left.iloc[left_idx].copy().reset_index(drop=True)
        right_rep = right.iloc[right_idx].copy().reset_index(drop=True)
        pair_key = [f"{label}_{i:06d}" for i in range(n)]
        left_rep["synthetic_pair_id"] = pair_key
        right_rep["synthetic_pair_id"] = pair_key
        merged = left_rep.merge(right_rep, on=["synthetic_pair_id", "label"], how="inner", suffixes=("", "_video"))
        paired_frames.append(merged)
    if not paired_frames:
        return pd.DataFrame()
    return pd.concat(paired_frames, ignore_index=True)


def fuse_datasets(
    tabular_df: pd.DataFrame,
    multimodal_df: pd.DataFrame | None,
    *,
    mode: str = "auto",
    allow_synthetic_label_pairing: bool = False,
    logger=None,
) -> tuple[pd.DataFrame, dict]:
    logger = logger or setup_logger("fusion")
    tabular_df = clean_tabular_dataset(tabular_df)
    tabular_df = tabular_df.copy()
    multimodal_df = pd.DataFrame() if multimodal_df is None else multimodal_df.copy()
    mode = _normalize_mode(mode)

    report = {
        "requested_mode": mode,
        "allow_synthetic_label_pairing": bool(allow_synthetic_label_pairing),
        "tabular_rows": int(len(tabular_df)),
        "multimodal_rows": int(len(multimodal_df)),
        "matched_prospect_ids": 0,
        "fusion_mode_used": "tabular_only",
        "multimodal_feature_columns": _multimodal_feature_columns(multimodal_df) if not multimodal_df.empty else [],
    }

    if tabular_df.empty:
        return tabular_df, report
    if multimodal_df.empty:
        return tabular_df, report

    matched_ids = set(tabular_df.get("prospect_id", pd.Series(dtype="string")).astype(str)) & set(multimodal_df.get("prospect_id", pd.Series(dtype="string")).astype(str))
    report["matched_prospect_ids"] = int(len(matched_ids))

    if mode == "tabular_only":
        return tabular_df, report

    if matched_ids:
        fused = _merge_on_prospect_id(tabular_df, multimodal_df)
        report["fusion_mode_used"] = "hybrid"
        report["fused_rows"] = int(len(fused))
        return fused, report

    if mode == "hybrid":
        logger.warning("Hybrid fusion requested but no prospect_id matches were found. Falling back to tabular-only training.")
        return tabular_df, report

    if mode == "demo_synthetic" or allow_synthetic_label_pairing:
        logger.warning("Synthetic label-based pairing was used. This is only for demo purposes and is not valid scientific evaluation.")
        fused = _synthetic_pair_by_label(tabular_df, multimodal_df)
        report["fusion_mode_used"] = "demo_synthetic"
        report["fused_rows"] = int(len(fused))
        return fused if not fused.empty else tabular_df, report

    logger.warning("No prospect_id matches were found between tabular and video data. Falling back to tabular-only training.")
    return tabular_df, report


def build_fused_dataset(
    tabular_df: pd.DataFrame,
    multimodal_df: pd.DataFrame | None,
    output_path: str | Path,
    report_path: str | Path | None = None,
    *,
    mode: str = "auto",
    allow_synthetic_label_pairing: bool = False,
    logger=None,
) -> tuple[pd.DataFrame, dict]:
    fused, report = fuse_datasets(
        tabular_df,
        multimodal_df,
        mode=mode,
        allow_synthetic_label_pairing=allow_synthetic_label_pairing,
        logger=logger,
    )
    dataframe_to_csv(fused, output_path)
    if report_path is not None:
        save_json(report, report_path)
    return fused, report
