"""Online pretrained video / frame-text scoring helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    torch = None

try:
    from transformers import AutoModel, AutoProcessor
except Exception:  # pragma: no cover - optional dependency guard
    AutoModel = None
    AutoProcessor = None


# A lighter online zero-shot model that can score either a video clip or
# sampled frames. It is fast enough for interactive uploads and can be
# swapped to a heavier video-text model through config if desired.
DEFAULT_VIDEO_MODEL_NAME = "openai/clip-vit-base-patch32"

DEFAULT_VIDEO_PROMPTS = {
    "hot": "A photo of a real estate prospect who is highly interested, actively engaged, asking to move forward, and ready to buy soon.",
    "warm": "A photo of a real estate prospect who is somewhat interested but hesitant, comparing options, and not fully committed yet.",
    "cold": "A photo of a real estate prospect who is uninterested, disengaged, refusing, or clearly not planning to continue.",
}


def _torch_device() -> str:
    if torch is None:
        return "cpu"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_available") and cuda.is_available():
        return "cuda"
    return "cpu"


def sample_video_frames(video_path: str | Path, num_frames: int = 8) -> List[Image.Image]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        total_frames = num_frames

    indices = np.linspace(0, max(total_frames - 1, 0), num=min(num_frames, max(total_frames, 1)), dtype=int).tolist()

    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


@lru_cache(maxsize=4)
def _load_model_bundle(model_name: str):
    if AutoModel is None or AutoProcessor is None:
        raise RuntimeError("transformers is not available")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = _torch_device()
    if torch is not None:
        model.to(device)
    return processor, model, device


def _to_probability_map(labels: list[str], probs: list[float]) -> dict:
    return {label: float(probs[i]) for i, label in enumerate(labels)}


def _framewise_clip_score(processor, model, device: str, frames: list[Image.Image], text_inputs: list[str]) -> list[float]:
    """Score each sampled frame independently and average the probabilities."""
    if not frames:
        return [1.0 / max(len(text_inputs), 1)] * len(text_inputs)

    frame_probabilities: list[list[float]] = []
    for frame in frames:
        inputs = processor(text=text_inputs, images=[frame], return_tensors="pt", padding=True)
        if torch is not None:
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad() if torch is not None else _nullcontext():
            outputs = model(**inputs)
            logits = outputs.logits_per_image if hasattr(outputs, "logits_per_image") else outputs[0]
            probs = logits.softmax(dim=1)[0].detach().cpu().numpy().tolist()
        frame_probabilities.append(probs)

    return np.mean(np.asarray(frame_probabilities, dtype=float), axis=0).tolist()


def _video_clip_score(processor, model, device: str, frames: list[Image.Image], text_inputs: list[str]) -> list[float]:
    inputs = processor(text=text_inputs, videos=[frames], return_tensors="pt", padding=True)
    if torch is not None:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad() if torch is not None else _nullcontext():
        outputs = model(**inputs)
        logits = outputs.logits_per_video if hasattr(outputs, "logits_per_video") else outputs[0]
        probs = logits.softmax(dim=1)[0].detach().cpu().numpy().tolist()
    return probs


def score_video_against_prompts(
    video_path: str | Path,
    *,
    model_name: str = DEFAULT_VIDEO_MODEL_NAME,
    prompts: Dict[str, str] | None = None,
    num_frames: int = 8,
) -> dict:
    """Score a video against custom class prompts using a pretrained online vision-language model."""
    prompts = prompts or DEFAULT_VIDEO_PROMPTS
    frames = sample_video_frames(video_path, num_frames=num_frames)
    labels = list(prompts.keys())
    text_inputs = [prompts[label] for label in labels]

    if not frames:
        return {
            "model_name": model_name,
            "predicted_label": "warm",
            "probabilities": {"cold": 1 / 3, "warm": 1 / 3, "hot": 1 / 3},
            "max_probability": 1 / 3,
            "probability_gap": 0.0,
            "num_frames": 0,
            "device": _torch_device(),
            "scoring_backend": "empty_video",
        }

    processor, model, device = _load_model_bundle(model_name)

    backend = "video_text"
    try:
        if "xclip" in model_name.lower() or "video" in model_name.lower():
            probs = _video_clip_score(processor, model, device, frames, text_inputs)
        else:
            raise ValueError("framewise_backend")
    except Exception:
        probs = _framewise_clip_score(processor, model, device, frames, text_inputs)
        backend = "framewise_image_text"

    probability_map = _to_probability_map(labels, probs)
    predicted_label = max(probability_map, key=probability_map.get)
    max_probability = float(probability_map[predicted_label])
    sorted_probs = sorted(probability_map.values(), reverse=True)
    probability_gap = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
    return {
        "model_name": model_name,
        "predicted_label": predicted_label,
        "probabilities": probability_map,
        "max_probability": max_probability,
        "probability_gap": probability_gap,
        "num_frames": len(frames),
        "device": device,
        "scoring_backend": backend,
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
