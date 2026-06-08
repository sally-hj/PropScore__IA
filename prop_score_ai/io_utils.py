"""General IO helpers for dataset discovery and serialization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .constants import VIDEO_EXTENSIONS


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def discover_videos(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    files = [p for p in root_path.rglob("*") if p.is_file() and is_video_file(p)]
    return sorted(files)


def discover_class_videos(root: str | Path, class_name: str) -> list[Path]:
    return discover_videos(Path(root) / class_name)


def prospect_id_from_video_path(video_path: str | Path, class_label: str | None = None) -> str:
    video_path = Path(video_path)
    stem = video_path.stem
    if class_label and not stem.startswith(f"{class_label}_") and stem != class_label:
        return f"{class_label}_{stem}"
    return stem


def video_id_from_path(video_path: str | Path) -> str:
    return Path(video_path).stem


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def load_csv_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def copy_videos(src_root: str | Path, dst_root: str | Path, class_names: Iterable[str]) -> list[Path]:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    copied: list[Path] = []
    for class_name in class_names:
        src_dir = src_root / class_name
        dst_dir = dst_root / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            continue
        for video_path in discover_videos(src_dir):
            target = dst_dir / video_path.name
            if not target.exists():
                shutil.copy2(video_path, target)
            copied.append(target)
    return copied


def dataframe_to_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

