from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from prop_score_ai.audio_pipeline import transcribe_audio_dataset
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Transcribe extracted audio with Whisper.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--model-size", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    audio_root = Path(args.audio_root or cfg["paths"]["audio_root"])
    output_csv = Path(args.output_csv or (cfg["paths"]["processed_root"] / "transcripts.csv"))
    model_size = args.model_size or cfg["whisper"]["model_size"]
    logger = setup_logger("whisper_transcription", cfg["paths"]["logs_root"] / "05_whisper_transcription.log")
    transcribe_audio_dataset(audio_root, output_csv, model_size=model_size, class_names=cfg["labels"], logger=logger)


if __name__ == "__main__":
    main()
