from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.nlp_pipeline import process_semantic_dataset
from prop_score_ai.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Analyze transcripts with CamemBERT semantic fallback.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--transcripts-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    transcripts_csv = Path(args.transcripts_csv or (cfg["paths"]["processed_root"] / "transcripts.csv"))
    output_csv = Path(args.output_csv or (cfg["paths"]["processed_root"] / "semantic_features.csv"))
    model_name = args.model_name or cfg["camembert"]["model_name"]
    logger = setup_logger("camembert_nlp", cfg["paths"]["logs_root"] / "07_camembert_nlp.log")
    process_semantic_dataset(transcripts_csv, output_csv, model_name=model_name, logger=logger)


if __name__ == "__main__":
    main()
