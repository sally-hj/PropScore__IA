"""Video, frame, face mesh, and emotion processing.

DeepFace is intentionally not imported here because its TensorFlow runtime can
crash on machines without AVX support. The live app uses a lightweight fallback
emotion heuristic so the rest of the pipeline remains usable everywhere.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional import guard
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from .constants import ID_TO_LABEL, LABEL_TO_ID
from .io_utils import dataframe_to_csv, discover_videos, ensure_dir, prospect_id_from_video_path, video_id_from_path
from .logging_utils import setup_logger

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional import guard
    mp = None

DEEPFACE_AVAILABLE = False


def _load_haar_cascade(filename: str):
    cascade_path = Path(cv2.data.haarcascades) / filename
    if not cascade_path.exists():
        return None
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return cascade if not cascade.empty() else None


def _frame_step_from_fps(fps: float, sample_fps: float) -> int:
    if fps <= 0 or sample_fps <= 0:
        return 1
    step = int(round(fps / sample_fps))
    return max(step, 1)


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 1.0,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> list[Path]:
    """Extract frames at a regular interval and return their paths."""
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = _frame_step_from_fps(fps, sample_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved: list[Path] = []
    frame_idx = 0
    sample_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            sample_idx += 1
            if max_frames is not None and len(saved) >= max_frames:
                break
            frame_file = output_dir / f"frame_{sample_idx:04d}.jpg"
            if overwrite or not frame_file.exists():
                cv2.imwrite(str(frame_file), frame)
            saved.append(frame_file)
        frame_idx += 1

    cap.release()
    return saved


def _crop_with_margin(frame: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int, margin: float = 0.15) -> np.ndarray:
    h, w = frame.shape[:2]
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    x_pad = int(width * margin)
    y_pad = int(height * margin)
    left = max(0, x_min - x_pad)
    top = max(0, y_min - y_pad)
    right = min(w, x_max + x_pad)
    bottom = min(h, y_max + y_pad)
    crop = frame[top:bottom, left:right]
    return crop if crop.size else frame


def _fallback_emotion(frame_bgr: np.ndarray) -> tuple[str, float]:
    """Lightweight emotion proxy that never depends on TensorFlow.

    The goal is to keep the app responsive on machines where DeepFace cannot
    load (for example, because the bundled TensorFlow build requires AVX).
    This heuristic intentionally returns conservative labels.
    """

    if frame_bgr is None or frame_bgr.size == 0:
        return "unknown", 0.0

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 255.0)

    if brightness >= 0.66:
        emotion = "happy"
    elif brightness <= 0.34:
        emotion = "sad"
    else:
        emotion = "neutral"

    confidence = float(np.clip(0.25 + 0.5 * contrast, 0.0, 0.75))
    return emotion, confidence


def _deepface_emotion(frame_bgr: np.ndarray) -> tuple[str, float]:
    if not DEEPFACE_AVAILABLE:
        return _fallback_emotion(frame_bgr)
    return _fallback_emotion(frame_bgr)


@dataclass
class FrameVisualResult:
    face_detected: bool
    landmark_quality: float
    dominant_emotion: str
    emotion_confidence: float


class VisualSignalAnalyzer:
    """Reusable MediaPipe Face Mesh analyzer with a no-TensorFlow fallback."""

    def __init__(self) -> None:
        self._mp_face_mesh = None
        self._face_mesh = None
        self._haar_face = _load_haar_cascade("haarcascade_frontalface_default.xml")
        self._haar_face_alt = _load_haar_cascade("haarcascade_frontalface_alt2.xml")
        self._haar_profile_face = _load_haar_cascade("haarcascade_profileface.xml")
        self._haar_smile = _load_haar_cascade("haarcascade_smile.xml")
        if mp is not None:
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()

    def _heuristic_emotion(self, frame_bgr: np.ndarray) -> tuple[str, float]:
        """Estimate a face emotion without TensorFlow / DeepFace."""
        if frame_bgr is None or frame_bgr.size == 0:
            return "unknown", 0.0

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray) / 255.0)
        brightness = float(np.mean(gray) / 255.0)

        if self._haar_smile is not None:
            lower_roi = gray[int(gray.shape[0] * 0.45) :, :]
            if lower_roi.size:
                smiles = self._haar_smile.detectMultiScale(
                    lower_roi,
                    scaleFactor=1.20,
                    minNeighbors=18,
                    minSize=(24, 24),
                )
                if len(smiles) > 0:
                    lower_h, lower_w = lower_roi.shape[:2]
                    smile_candidates = [
                        (sw * sh) / float(lower_h * lower_w)
                        for (_, _, sw, sh) in smiles
                        if sw >= int(0.22 * lower_w) and sh >= int(0.10 * lower_h)
                    ]
                    if smile_candidates and max(smile_candidates) >= 0.035:
                        return "happy", float(np.clip(0.70 + 0.18 * contrast, 0.0, 0.92))

        # Keep negative emotion conservative; only very low-light / low-detail
        # faces are interpreted as sad. Most non-smiling faces stay neutral.
        if brightness <= 0.18 and contrast <= 0.18:
            return "sad", float(np.clip(0.30 + 0.25 * (1.0 - brightness), 0.0, 0.75))
        return "neutral", float(np.clip(0.30 + 0.25 * contrast, 0.0, 0.70))

    def _detect_face_bbox(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cascades = [c for c in [self._haar_face, self._haar_face_alt, self._haar_profile_face] if c is not None]
        candidates: list[tuple[int, int, int, int]] = []
        for cascade in cascades:
            try:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(40, 40),
                )
            except Exception:
                faces = ()
            if len(faces) > 0:
                candidates.extend([tuple(map(int, face)) for face in faces])
        if not candidates:
            return None
        return max(candidates, key=lambda f: f[2] * f[3])

    def analyze_frame(self, frame_bgr: np.ndarray) -> FrameVisualResult:
        if self._face_mesh is None:
            bbox = self._detect_face_bbox(frame_bgr)
            if bbox is None:
                return FrameVisualResult(False, 0.0, "unknown", 0.0)
            x, y, w, h = bbox
            crop = _crop_with_margin(frame_bgr, x, y, x + w, y + h)
            emotion, confidence = self._heuristic_emotion(crop)
            area_ratio = (w * h) / float(frame_bgr.shape[0] * frame_bgr.shape[1]) if frame_bgr.size else 0.0
            landmark_quality = float(np.clip(0.35 + 0.55 * area_ratio, 0.0, 1.0))
            return FrameVisualResult(True, landmark_quality, emotion, confidence)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            bbox = self._detect_face_bbox(frame_bgr)
            if bbox is None:
                return FrameVisualResult(False, 0.0, "unknown", 0.0)
            x, y, w, h = bbox
            crop = _crop_with_margin(frame_bgr, x, y, x + w, y + h)
            emotion, confidence = self._heuristic_emotion(crop)
            area_ratio = (w * h) / float(frame_bgr.shape[0] * frame_bgr.shape[1]) if frame_bgr.size else 0.0
            landmark_quality = float(np.clip(0.35 + 0.55 * area_ratio, 0.0, 1.0))
            return FrameVisualResult(True, landmark_quality, emotion, confidence)

        landmarks = results.multi_face_landmarks[0].landmark
        xs = [lm.x for lm in landmarks if 0.0 <= lm.x <= 1.0]
        ys = [lm.y for lm in landmarks if 0.0 <= lm.y <= 1.0]
        visibilities = [getattr(lm, "visibility", 1.0) for lm in landmarks]
        presences = [getattr(lm, "presence", 1.0) for lm in landmarks]
        h, w = frame_bgr.shape[:2]
        x_min = int(max(0, min(xs, default=0.0) * w))
        x_max = int(min(w, max(xs, default=1.0) * w))
        y_min = int(max(0, min(ys, default=0.0) * h))
        y_max = int(min(h, max(ys, default=1.0) * h))
        crop = _crop_with_margin(frame_bgr, x_min, y_min, x_max, y_max)

        emotion, confidence = self._heuristic_emotion(crop)
        area_ratio = 0.0
        if x_max > x_min and y_max > y_min and w > 0 and h > 0:
            area_ratio = ((x_max - x_min) * (y_max - y_min)) / float(w * h)
        landmark_quality = float(np.clip(np.mean(visibilities) * np.mean(presences) * (0.5 + 0.5 * area_ratio), 0.0, 1.0))
        return FrameVisualResult(True, landmark_quality, emotion, confidence)


def _aggregate_visual_results(results: list[FrameVisualResult]) -> dict:
    if not results:
        return {
            "dominant_emotion": "unknown",
            "emotion_confidence_mean": 0.0,
            "emotion_confidence_std": 0.0,
            "face_detected_ratio": 0.0,
            "landmark_quality_score": 0.0,
            "vision_score": 0.0,
        }
    emotions = [r.dominant_emotion for r in results if r.dominant_emotion and r.dominant_emotion != "unknown"]
    dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "unknown"
    confidences = [r.emotion_confidence for r in results]
    face_ratio = sum(1 for r in results if r.face_detected) / len(results)
    qualities = [r.landmark_quality for r in results if r.face_detected]
    quality_mean = float(np.mean(qualities)) if qualities else 0.0
    confidence_mean = float(np.mean(confidences)) if confidences else 0.0
    confidence_std = float(np.std(confidences)) if len(confidences) > 1 else 0.0
    vision_score = float(np.clip(100.0 * (0.45 * face_ratio + 0.30 * quality_mean + 0.25 * confidence_mean), 0.0, 100.0))
    return {
        "dominant_emotion": dominant_emotion,
        "emotion_confidence_mean": confidence_mean,
        "emotion_confidence_std": confidence_std,
        "face_detected_ratio": face_ratio,
        "landmark_quality_score": quality_mean,
        "vision_score": vision_score,
    }


def analyze_video_visual_signals(video_path: str | Path, frames_dir: str | Path, sample_fps: float = 1.0) -> dict:
    """Extract frames and aggregate face mesh + emotion metrics for one video."""
    analyzer = VisualSignalAnalyzer()
    try:
        frames = extract_frames_from_video(video_path, frames_dir, sample_fps=sample_fps)
        results: list[FrameVisualResult] = []
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            results.append(analyzer.analyze_frame(frame))
        return _aggregate_visual_results(results)
    finally:
        analyzer.close()


def process_face_features_dataset(
    video_root: str | Path,
    frames_root: str | Path,
    output_csv: str | Path,
    class_names: Iterable[str] = ("hot", "warm", "cold"),
    sample_fps: float = 1.0,
    logger=None,
) -> pd.DataFrame:
    """Extract frames and compute face mesh quality metrics per video."""
    logger = logger or setup_logger("face_features")
    video_root = Path(video_root)
    frames_root = Path(frames_root)
    rows: list[dict] = []
    analyzer = VisualSignalAnalyzer()

    try:
        for class_name in class_names:
            for video_path in tqdm(discover_videos(video_root / class_name), desc=f"Face mesh {class_name}"):
                prospect_id = prospect_id_from_video_path(video_path, class_name)
                video_id = video_id_from_path(video_path)
                frame_dir = frames_root / class_name / video_id
                frames = extract_frames_from_video(video_path, frame_dir, sample_fps=sample_fps)
                frame_results: list[FrameVisualResult] = []
                for frame_path in frames:
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        continue
                    frame_results.append(analyzer.analyze_frame(frame))
                agg = _aggregate_visual_results(frame_results)
                rows.append(
                    {
                        "prospect_id": prospect_id,
                        "video_id": video_id,
                        "label": class_name,
                        "face_detected_ratio": agg["face_detected_ratio"],
                        "landmark_quality_score": agg["landmark_quality_score"],
                    }
                )
    finally:
        analyzer.close()

    df = pd.DataFrame(rows)
    dataframe_to_csv(df, output_csv)
    logger.info("Saved face features to %s", output_csv)
    return df


def process_emotion_features_dataset(
    video_root: str | Path,
    frames_root: str | Path,
    output_csv: str | Path,
    class_names: Iterable[str] = ("hot", "warm", "cold"),
    sample_fps: float = 1.0,
    logger=None,
) -> pd.DataFrame:
    """Extract frames and compute emotion metrics per video."""
    logger = logger or setup_logger("emotion_features")
    video_root = Path(video_root)
    frames_root = Path(frames_root)
    rows: list[dict] = []
    analyzer = VisualSignalAnalyzer()

    try:
        for class_name in class_names:
            for video_path in tqdm(discover_videos(video_root / class_name), desc=f"Emotion {class_name}"):
                prospect_id = prospect_id_from_video_path(video_path, class_name)
                video_id = video_id_from_path(video_path)
                frame_dir = frames_root / class_name / video_id
                if not frame_dir.exists():
                    frames = extract_frames_from_video(video_path, frame_dir, sample_fps=sample_fps)
                else:
                    frames = sorted(frame_dir.glob("frame_*.jpg"))
                    if not frames:
                        frames = extract_frames_from_video(video_path, frame_dir, sample_fps=sample_fps)
                results: list[FrameVisualResult] = []
                for frame_path in frames:
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        continue
                    results.append(analyzer.analyze_frame(frame))
                agg = _aggregate_visual_results(results)
                rows.append(
                    {
                        "prospect_id": prospect_id,
                        "video_id": video_id,
                        "label": class_name,
                        "dominant_emotion": agg["dominant_emotion"],
                        "emotion_confidence_mean": agg["emotion_confidence_mean"],
                        "emotion_confidence_std": agg["emotion_confidence_std"],
                        "vision_score": agg["vision_score"],
                    }
                )
    finally:
        analyzer.close()

    df = pd.DataFrame(rows)
    dataframe_to_csv(df, output_csv)
    logger.info("Saved emotion features to %s", output_csv)
    return df


def get_live_visual_features(frame_bgr: np.ndarray, analyzer: VisualSignalAnalyzer | None = None) -> dict:
    """Analyze a live frame and return the same visual columns used in training."""
    created = analyzer is None
    analyzer = analyzer or VisualSignalAnalyzer()
    try:
        result = analyzer.analyze_frame(frame_bgr)
        vision_score = float(np.clip(100.0 * (0.45 * float(result.face_detected) + 0.30 * result.landmark_quality + 0.25 * result.emotion_confidence), 0.0, 100.0))
        return {
            "dominant_emotion": result.dominant_emotion,
            "emotion_confidence_mean": result.emotion_confidence,
            "emotion_confidence_std": 0.0,
            "face_detected_ratio": 1.0 if result.face_detected else 0.0,
            "landmark_quality_score": result.landmark_quality,
            "vision_score": vision_score,
        }
    finally:
        if created:
            analyzer.close()
