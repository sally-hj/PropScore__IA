"""Configuration helpers for PropScore_AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PATH_KEY_HINTS = (
    "path",
    "root",
    "dir",
    "csv",
    "file",
    "folder",
    "report",
    "metrics",
    "model",
    "output",
    "input",
    "frames",
    "audio",
    "predictions",
    "notebooks",
)


def _resolve_value(value: Any, root: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_value(v, root, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, root, key) for v in value]
    if isinstance(value, str) and key is not None and any(hint in key.lower() for hint in PATH_KEY_HINTS):
        lowered_key = key.lower()
        if lowered_key.endswith("_name") or lowered_key.endswith("name") or lowered_key.endswith("_size") or lowered_key.endswith("size"):
            return value
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        # Hugging Face repo IDs (for example "openai/clip-vit-base-patch32")
        # should stay as-is even though they contain a slash.
        if "/" in value and not value.startswith((".", "~", "..")) and candidate.suffix == "":
            return value
        return (root / candidate).resolve()
    return value


def load_config(config_path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    root = project_root()
    config_file = Path(config_path) if config_path else root / "configs" / "config.yaml"
    if not config_file.is_absolute():
        config_file = (root / config_file).resolve()
    with config_file.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    cfg["project_root"] = root
    cfg["config_path"] = config_file
    cfg = _resolve_value(cfg, root)
    return cfg


def ensure_runtime_dirs(cfg: Dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    for key in [
        "raw_videos_root",
        "frames_root",
        "faces_root",
        "audio_root",
        "processed_root",
        "features_root",
        "models_root",
        "outputs_root",
        "logs_root",
        "metrics_root",
        "predictions_root",
        "notebooks_root",
    ]:
        path = paths.get(key)
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
