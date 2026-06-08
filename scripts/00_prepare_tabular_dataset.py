from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import project_root
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.logging_utils import setup_logger
from prop_score_ai.tabular_pipeline import prepare_tabular_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare the structured tabular dataset for PropScore_AI.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="Path to dataset_final_catboost.csv or another raw tabular CSV.")
    parser.add_argument("--output", default=None, help="Output cleaned prospects CSV.")
    parser.add_argument("--report", default=None, help="Output JSON report path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (project_root() / input_path).resolve()
    output_csv = Path(args.output or (cfg["paths"]["processed_root"] / "prospects.csv"))
    report_path = Path(args.report or (cfg["paths"]["metrics_root"] / "tabular_dataset_report.json"))
    logger = setup_logger("prepare_tabular_dataset", cfg["paths"]["logs_root"] / "00_prepare_tabular_dataset.log")
    prepare_tabular_dataset(input_path, output_csv, report_path, logger=logger)


if __name__ == "__main__":
    main()
