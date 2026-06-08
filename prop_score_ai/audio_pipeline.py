"""Audio extraction, Librosa feature engineering, and Whisper transcription."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import librosa
except Exception:  # pragma: no cover - optional import guard
    librosa = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional import guard
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from .io_utils import dataframe_to_csv, discover_videos, ensure_dir, prospect_id_from_video_path, video_id_from_path
from .logging_utils import setup_logger

try:
    import whisper
except Exception:  # pragma: no cover - optional import guard
    whisper = None


def extract_audio_from_video(video_path: str | Path, audio_path: str | Path, sample_rate: int = 16000) -> Path | None:
    """Extract a mono WAV audio track using ffmpeg."""
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(audio_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 0 and audio_path.exists():
        return audio_path
    return None


def _safe_stat(values: np.ndarray, fn) -> float:
    if values.size == 0:
        return 0.0
    return float(fn(values))


def _pitch_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            return 0.0, 0.0
        return float(np.mean(f0)), float(np.std(f0))
    except Exception:
        return 0.0, 0.0


def _speech_rate_estimate(y: np.ndarray, sr: int) -> float:
    try:
        intervals = librosa.effects.split(y, top_db=25)
        voiced_duration = float(sum((end - start) for start, end in intervals) / sr)
        duration = max(len(y) / sr, 1e-6)
        return float(voiced_duration / duration)
    except Exception:
        return 0.0


def _audio_score(mfcc_mean: np.ndarray, pitch_mean: float, pitch_std: float, energy_mean: float, zcr_mean: float, speech_rate: float) -> float:
    energy_norm = float(np.clip(energy_mean * 10.0, 0.0, 1.0))
    pitch_norm = float(np.clip(abs(pitch_mean) / 400.0, 0.0, 1.0))
    pitch_variability = float(np.clip(pitch_std / 200.0, 0.0, 1.0))
    zcr_norm = float(np.clip(zcr_mean * 5.0, 0.0, 1.0))
    speech_norm = float(np.clip(speech_rate, 0.0, 1.0))
    return float(np.clip(100.0 * (0.30 * energy_norm + 0.20 * pitch_norm + 0.15 * pitch_variability + 0.15 * zcr_norm + 0.20 * speech_norm), 0.0, 100.0))


def extract_librosa_features(audio_path: str | Path, sr: int = 16000) -> dict:
    """Compute the requested Librosa features for one audio file."""
    audio_path = Path(audio_path)
    if librosa is None:
        return {
            **{f"mfcc_mean_{idx}": 0.0 for idx in range(1, 14)},
            **{f"mfcc_std_{idx}": 0.0 for idx in range(1, 14)},
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "spectral_centroid_mean": 0.0,
            "speech_rate_estimate": 0.0,
            "audio_score": 0.0,
            "audio_missing": True,
        }

    if not audio_path.exists():
        return {"audio_missing": True}

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    if y.size == 0:
        return {"audio_missing": True}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    pitch_mean, pitch_std = _pitch_stats(y, sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    speech_rate = _speech_rate_estimate(y, sr)
    audio_score = _audio_score(mfcc_mean, pitch_mean, pitch_std, float(np.mean(rms)), float(np.mean(zcr)), speech_rate)

    features = {}
    for idx in range(13):
        features[f"mfcc_mean_{idx+1}"] = float(mfcc_mean[idx]) if idx < len(mfcc_mean) else 0.0
        features[f"mfcc_std_{idx+1}"] = float(mfcc_std[idx]) if idx < len(mfcc_std) else 0.0
    features.update(
        {
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "energy_mean": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "spectral_centroid_mean": float(np.mean(centroid)),
            "speech_rate_estimate": float(speech_rate),
            "audio_score": float(audio_score),
        }
    )
    return features


def process_audio_dataset(
    video_root: str | Path,
    audio_root: str | Path,
    output_csv: str | Path,
    class_names: Iterable[str] = ("hot", "warm", "cold"),
    sample_rate: int = 16000,
    logger=None,
) -> pd.DataFrame:
    logger = logger or setup_logger("audio_features")
    video_root = Path(video_root)
    audio_root = Path(audio_root)
    rows: list[dict] = []
    for class_name in class_names:
        for video_path in tqdm(discover_videos(video_root / class_name), desc=f"Audio {class_name}"):
            prospect_id = prospect_id_from_video_path(video_path, class_name)
            video_id = video_id_from_path(video_path)
            audio_dir = ensure_dir(audio_root / class_name)
            audio_path = audio_dir / f"{video_id}.wav"
            if not audio_path.exists():
                extract_audio_from_video(video_path, audio_path, sample_rate=sample_rate)
            features = extract_librosa_features(audio_path, sr=sample_rate)
            features.update({"prospect_id": prospect_id, "video_id": video_id, "label": class_name, "audio_path": str(audio_path)})
            rows.append(features)
    df = pd.DataFrame(rows)
    dataframe_to_csv(df, output_csv)
    logger.info("Saved audio features to %s", output_csv)
    return df


class WhisperTranscriber:
    """Lazy Whisper wrapper for batch transcription."""

    def __init__(self, model_size: str = "base", device: str | None = None):
        self.model_size = model_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if whisper is None:
                raise RuntimeError("whisper is not available. Install openai-whisper.")
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str | Path) -> dict:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return {
                "transcript": "",
                "language": "unknown",
                "duration": 0.0,
                "analysis_method": "missing_audio",
            }
        try:
            if librosa is None:
                return {
                    "transcript": "",
                    "language": "unknown",
                    "duration": 0.0,
                    "analysis_method": "librosa_unavailable",
                }
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            duration = float(len(audio) / sr) if sr else 0.0
            result = self.model.transcribe(str(audio_path), fp16=False)
            return {
                "transcript": (result.get("text") or "").strip(),
                "language": result.get("language", "unknown"),
                "duration": duration,
                "analysis_method": "whisper",
            }
        except Exception as exc:
            return {
                "transcript": "",
                "language": "unknown",
                "duration": 0.0,
                "analysis_method": f"whisper_failed:{type(exc).__name__}",
            }


def transcribe_audio_dataset(
    audio_root: str | Path,
    output_csv: str | Path,
    model_size: str = "base",
    class_names: Iterable[str] = ("hot", "warm", "cold"),
    logger=None,
) -> pd.DataFrame:
    logger = logger or setup_logger("whisper_transcription")
    audio_root = Path(audio_root)
    rows: list[dict] = []
    transcriber = WhisperTranscriber(model_size=model_size)
    for class_name in class_names:
        for audio_path in tqdm(sorted((audio_root / class_name).glob("*.wav")), desc=f"Whisper {class_name}"):
            stem = audio_path.stem
            prospect_id = stem if stem.startswith(f"{class_name}_") else f"{class_name}_{stem}"
            row = transcriber.transcribe(audio_path)
            row.update({"prospect_id": prospect_id, "video_id": stem, "label": class_name})
            rows.append(row)
    df = pd.DataFrame(rows)
    dataframe_to_csv(df, output_csv)
    logger.info("Saved transcripts to %s", output_csv)
    return df
