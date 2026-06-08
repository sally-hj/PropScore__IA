"""Schema and cleaning helpers for multimodal feature CSVs."""

from __future__ import annotations

import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_MULTIMODAL_COLUMNS = [
    "video_path",
    "label",
    "transcript",
    "sentiment",
    "dominant_emotion",
    "confidence_emotion",
    "pitch_mean",
    "energy_mean",
    "speech_rate",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "keyword_count",
    "keyword_type",
    "vision_score",
    "audio_score",
    "semantic_score",
    "final_score",
    "face_detected",
    "landmark_count",
    "face_center_x",
    "face_center_y",
    "face_width",
    "face_height",
]

NUMERIC_MULTIMODAL_COLUMNS = [
    "confidence_emotion",
    "pitch_mean",
    "energy_mean",
    "speech_rate",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "keyword_count",
    "vision_score",
    "audio_score",
    "semantic_score",
    "final_score",
    "face_detected",
    "landmark_count",
    "face_center_x",
    "face_center_y",
    "face_width",
    "face_height",
]

TEXT_MULTIMODAL_COLUMNS = [
    "video_path",
    "label",
    "transcript",
    "sentiment",
    "dominant_emotion",
    "keyword_type",
]

LABEL_NORMALIZATION_MAP = {
    "chaud": "hot",
    "hot": "hot",
    "tiede": "warm",
    "tiède": "warm",
    "warm": "warm",
    "froid": "cold",
    "cold": "cold",
}

VALID_MULTIMODAL_LABELS = {"hot", "warm", "cold"}


def _normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_label(value: Any) -> str:
    text = _normalize_text_value(value).lower()
    if not text:
        return "unknown"
    return LABEL_NORMALIZATION_MAP.get(text, text)


def _normalize_video_path(value: Any) -> str:
    text = _normalize_text_value(value)
    if not text:
        return ""
    if text.lower() in {"unknown", "n/a", "na"}:
        return ""
    text = text.replace("file:///", "")
    if re.match(r"^[A-Za-z]:\\", text) or re.match(r"^[A-Za-z]:/", text):
        win_path = PureWindowsPath(text)
        return str(win_path).replace("\\", "/")
    text = text.replace("\\", "/")
    text = re.sub(r"/{2,}", "/", text)
    return text


def normalize_multimodal_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "label" in result.columns:
        result["label"] = result["label"].map(_normalize_label)
    return result


def clean_video_paths(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "video_path" in result.columns:
        result["video_path"] = result["video_path"].map(_normalize_video_path)
    return result


def _infer_dataset_name(dataset_name: str | None) -> str:
    dataset_name = (dataset_name or "").strip().lower()
    return dataset_name or "multimodal"


def build_prospect_id_from_video_path(df: pd.DataFrame, dataset_name: str | None) -> pd.DataFrame:
    result = df.copy()
    dataset_name = _infer_dataset_name(dataset_name)
    if "prospect_id" in result.columns:
        ids = result["prospect_id"].astype("string")
        missing = ids.isna() | (ids.astype(str).str.strip() == "") | (ids.astype(str).str.lower() == "nan")
        if not missing.any():
            return result
        existing_ids = ids.copy()
    else:
        existing_ids = pd.Series([None] * len(result), index=result.index, dtype="object")
        missing = pd.Series([True] * len(result), index=result.index)

    new_ids: list[str] = []
    for position, (idx, row) in enumerate(result.iterrows(), start=1):
        current = existing_ids.loc[idx] if idx in existing_ids.index else None
        current_text = _normalize_text_value(current)
        if current_text:
            new_ids.append(current_text)
            continue

        if dataset_name == "ravdess":
            new_ids.append(f"ravdess_{position:06d}")
            continue

        video_path = _normalize_video_path(row.get("video_path", ""))
        stem = Path(video_path).stem if video_path else ""
        stem = stem.strip()
        if stem:
            if dataset_name and not stem.lower().startswith(f"{dataset_name}_"):
                stem = f"{dataset_name}_{stem}"
            new_ids.append(stem)
        else:
            new_ids.append(f"{dataset_name}_{position:06d}")

    result["prospect_id"] = new_ids
    return result


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in NUMERIC_MULTIMODAL_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        if col in NUMERIC_MULTIMODAL_COLUMNS or pd.api.types.is_numeric_dtype(result[col]):
            median = pd.to_numeric(result[col], errors="coerce").median()
            if pd.isna(median):
                median = 0.0
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(float(median))
        else:
            result[col] = result[col].map(_normalize_text_value).replace("", "unknown")
    return result


def clean_multimodal_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str | None = None,
    add_prospect_id: bool = False,
) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(col).strip() for col in result.columns]
    result = clean_video_paths(result)
    result = normalize_multimodal_labels(result)
    result = _coerce_numeric_columns(result)
    result = _fill_missing_values(result)
    if add_prospect_id:
        result = build_prospect_id_from_video_path(result, dataset_name)
    return result


def validate_multimodal_schema(df: pd.DataFrame) -> dict[str, Any]:
    normalized_columns = [str(col).strip() for col in df.columns]
    actual_set = set(normalized_columns)
    expected_set = set(EXPECTED_MULTIMODAL_COLUMNS)
    missing_columns = [col for col in EXPECTED_MULTIMODAL_COLUMNS if col not in actual_set]
    extra_columns = [col for col in normalized_columns if col not in expected_set]
    label_distribution = {}
    if "label" in df.columns:
        labels = df["label"].map(_normalize_label).fillna("unknown").astype(str)
        label_distribution = labels.value_counts().to_dict()
    missing_values = int(df.isna().sum().sum())
    return {
        "is_valid": not missing_columns and len(normalized_columns) >= len(EXPECTED_MULTIMODAL_COLUMNS),
        "exact_schema_match": normalized_columns == EXPECTED_MULTIMODAL_COLUMNS,
        "expected_columns": EXPECTED_MULTIMODAL_COLUMNS,
        "actual_columns": normalized_columns,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_values": missing_values,
        "label_distribution": label_distribution,
    }
