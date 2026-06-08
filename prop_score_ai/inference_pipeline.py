"""Inference helpers for uploaded videos and live camera frames."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd

from .constants import ID_TO_LABEL, LABEL_TO_ID
from .io_utils import prospect_id_from_video_path, video_id_from_path


def load_bundle(model_path: str | Path, encoders_path: str | Path) -> tuple[object, dict]:
    model_path = Path(model_path)
    encoders_path = Path(encoders_path)
    model = joblib.load(model_path)
    bundle = joblib.load(encoders_path)
    feature_schema_path = model_path.parent / "feature_schema.json"
    if feature_schema_path.exists():
        try:
            import json

            schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
            bundle["feature_schema_details"] = schema
            bundle.setdefault("model_mode", schema.get("model_mode"))
            bundle.setdefault("feature_schema", schema.get("feature_columns", bundle.get("feature_schema", [])))
            bundle.setdefault("feature_defaults", schema.get("feature_defaults", bundle.get("feature_defaults", {})))
        except Exception:
            pass
    return model, bundle


def _default_raw_row(bundle: dict) -> dict:
    defaults = bundle.get("feature_defaults", {})
    row = {}
    for key, value in defaults.items():
        row[key] = value
    # Add a few common live fields if missing.
    for key in [
        "dominant_emotion",
        "sentiment_label",
        "keyword_type",
        "language",
        "property_type",
        "city",
    ]:
        row.setdefault(key, "unknown")
    for key in [
        "transcript",
        "label",
        "prospect_id",
        "video_id",
    ]:
        row.setdefault(key, "")
    return row


def align_feature_row(row: dict, bundle: dict) -> pd.DataFrame:
    schema = bundle.get("feature_schema", [])
    defaults = _default_raw_row(bundle)
    aligned = {col: row.get(col, defaults.get(col)) for col in schema}
    return pd.DataFrame([aligned])


def predict_from_raw_row(model, bundle: dict, row: dict) -> dict:
    df = align_feature_row(row, bundle)
    preprocessor = bundle["preprocessor"]
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0]
    pred_id = int(np.argmax(prob))
    pred_label = ID_TO_LABEL.get(pred_id, "cold")
    sorted_prob = np.sort(prob)[::-1]
    max_probability = float(sorted_prob[0]) if len(sorted_prob) else 0.0
    probability_gap = float(sorted_prob[0] - sorted_prob[1]) if len(sorted_prob) > 1 else float(sorted_prob[0]) if len(sorted_prob) else 0.0
    return {
        "predicted_id": pred_id,
        "predicted_label": pred_label,
        "probabilities": {ID_TO_LABEL[i]: float(prob[i]) for i in range(len(prob))},
        "max_probability": max_probability,
        "probability_gap": probability_gap,
        "feature_row": df.iloc[0].to_dict(),
    }


def predict_visual_frame(model, bundle: dict, frame_bgr: np.ndarray, visual: dict | None = None) -> dict:
    from .video_pipeline import get_live_visual_features

    visual = visual or get_live_visual_features(frame_bgr)
    row = _default_raw_row(bundle)
    row.update(visual)
    row["transcript"] = ""
    row["language"] = "unknown"
    row["sentiment_label"] = "neutral"
    row["keyword_type"] = "neutral"
    row["sentiment_score"] = 0.0
    row["semantic_score"] = 0.0
    row["audio_score"] = 0.0
    row["prospect_id"] = "live_camera"
    row["video_id"] = "live_camera"
    row["label"] = ""
    return predict_from_raw_row(model, bundle, row)


def process_uploaded_video(
    video_path: str | Path,
    model=None,
    bundle: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Run the full multimodal pipeline on a single uploaded video."""
    from .audio_pipeline import WhisperTranscriber, extract_audio_from_video, extract_librosa_features
    from .nlp_pipeline import CamembertSemanticAnalyzer
    from .online_video_model import DEFAULT_VIDEO_MODEL_NAME, DEFAULT_VIDEO_PROMPTS, score_video_against_prompts
    from .video_pipeline import analyze_video_visual_signals

    video_path = Path(video_path)
    prospect_id = prospect_id_from_video_path(video_path, video_path.parent.name if video_path.parent.name in LABEL_TO_ID else None)
    video_id = video_id_from_path(video_path)
    with tempfile.TemporaryDirectory(prefix="propscore_upload_") as tmpdir:
        tmpdir = Path(tmpdir)
        frames_dir = tmpdir / "frames"
        audio_dir = tmpdir / "audio"
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        sample_fps = float((config or {}).get("pipeline", {}).get("frame_sample_fps", 1.0))
        visual = analyze_video_visual_signals(video_path, frames_dir, sample_fps=sample_fps)
        audio_path = audio_dir / f"{video_id}.wav"
        extract_audio_from_video(video_path, audio_path)
        audio = extract_librosa_features(audio_path)
        transcriber = WhisperTranscriber(model_size=(config or {}).get("whisper", {}).get("model_size", "base"))
        transcript = transcriber.transcribe(audio_path)
        semantic_analyzer = CamembertSemanticAnalyzer(model_name=(config or {}).get("camembert", {}).get("model_name", "camembert-base"))
        semantic = semantic_analyzer.analyze(transcript.get("transcript", ""))
        row = _default_raw_row(bundle or {})
        row.update(
            {
                "prospect_id": prospect_id,
                "video_id": video_id,
                "label": "",
                "transcript": transcript.get("transcript", ""),
                "language": transcript.get("language", "unknown"),
                "duration": transcript.get("duration", 0.0),
                "analysis_method": transcript.get("analysis_method", ""),
            }
        )
        row.update(visual)
        row.update(audio)
        row.update(semantic)
        legacy_prediction = None
        if model is not None and bundle is not None:
            legacy_prediction = predict_from_raw_row(model, bundle, row)

        online_cfg = (config or {}).get("online_video", {})
        online_model_name = online_cfg.get("model_name", DEFAULT_VIDEO_MODEL_NAME)
        online_num_frames = int(online_cfg.get("num_frames", 8))
        online_prompts = online_cfg.get("prompts") or DEFAULT_VIDEO_PROMPTS
        online_error = None
        try:
            online_prediction = score_video_against_prompts(
                video_path,
                model_name=online_model_name,
                prompts=online_prompts,
                num_frames=online_num_frames,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            online_prediction = None
            online_error = str(exc)

        if online_prediction is None and legacy_prediction is not None:
            prediction = {
                **legacy_prediction,
                "prediction_source": "legacy_model",
                "online_error": online_error,
                "legacy_predicted_label": legacy_prediction.get("predicted_label"),
                "legacy_predicted_id": legacy_prediction.get("predicted_id"),
                "legacy_probabilities": legacy_prediction.get("probabilities", {}),
                "legacy_max_probability": legacy_prediction.get("max_probability", 0.0),
                "legacy_probability_gap": legacy_prediction.get("probability_gap", 0.0),
                "raw_predicted_label": legacy_prediction.get("predicted_label"),
                "raw_predicted_id": legacy_prediction.get("predicted_id"),
                "raw_probabilities": legacy_prediction.get("probabilities", {}),
                "raw_max_probability": legacy_prediction.get("max_probability", 0.0),
                "raw_probability_gap": legacy_prediction.get("probability_gap", 0.0),
            }
        elif online_prediction is not None:
            prediction = {
                **online_prediction,
                "prediction_source": "online_video_model",
                "online_error": online_error,
                "legacy_predicted_label": legacy_prediction.get("predicted_label") if legacy_prediction else None,
                "legacy_predicted_id": legacy_prediction.get("predicted_id") if legacy_prediction else None,
                "legacy_probabilities": legacy_prediction.get("probabilities", {}) if legacy_prediction else {},
                "legacy_max_probability": legacy_prediction.get("max_probability", 0.0) if legacy_prediction else 0.0,
                "legacy_probability_gap": legacy_prediction.get("probability_gap", 0.0) if legacy_prediction else 0.0,
                "raw_predicted_label": legacy_prediction.get("predicted_label") if legacy_prediction else online_prediction.get("predicted_label"),
                "raw_predicted_id": legacy_prediction.get("predicted_id") if legacy_prediction else None,
                "raw_probabilities": legacy_prediction.get("probabilities", {}) if legacy_prediction else online_prediction.get("probabilities", {}),
                "raw_max_probability": legacy_prediction.get("max_probability", 0.0) if legacy_prediction else online_prediction.get("max_probability", 0.0),
                "raw_probability_gap": legacy_prediction.get("probability_gap", 0.0) if legacy_prediction else online_prediction.get("probability_gap", 0.0),
            }
        else:
            prediction = {
                "predicted_label": "warm",
                "predicted_id": LABEL_TO_ID["warm"],
                "probabilities": {"cold": 1 / 3, "warm": 1 / 3, "hot": 1 / 3},
                "max_probability": 1 / 3,
                "probability_gap": 0.0,
                "prediction_source": "unavailable",
                "online_error": online_error,
                "legacy_predicted_label": None,
                "legacy_predicted_id": None,
                "legacy_probabilities": {},
                "legacy_max_probability": 0.0,
                "legacy_probability_gap": 0.0,
                "raw_predicted_label": None,
                "raw_predicted_id": None,
                "raw_probabilities": {},
                "raw_max_probability": 0.0,
                "raw_probability_gap": 0.0,
            }
        return {
            "prospect_id": prospect_id,
            "video_id": video_id,
            "visual": visual,
            "audio": audio,
            "transcript": transcript,
            "semantic": semantic,
            "prediction": prediction,
            "legacy_prediction": legacy_prediction,
            "row": row,
        }
