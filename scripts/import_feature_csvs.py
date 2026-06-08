from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import project_root
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.io_utils import dataframe_to_csv, save_json
from prop_score_ai.logging_utils import setup_logger
from utils.feature_schema import (
    EXPECTED_MULTIMODAL_COLUMNS,
    clean_multimodal_dataframe,
    validate_multimodal_schema,
)


def _resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    project_candidate = (project_root() / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return path.resolve()


def _process_file(path: Path, *, dataset_name: str, required_label: str | None = None, allowed_labels: set[str] | None = None) -> tuple[pd.DataFrame | None, dict]:
    report: dict = {
        "file_name": path.name,
        "source_path": str(path),
        "destination_path": None,
        "row_count": 0,
        "column_count": 0,
        "missing_values": 0,
        "label_distribution": {},
        "schema_valid": False,
        "exact_schema_match": False,
        "imported_successfully": False,
        "error": None,
    }
    if not path.exists() or path.stat().st_size == 0:
        report["error"] = "File missing or empty"
        return None, report

    try:
        raw_df = pd.read_csv(path)
    except Exception as exc:
        report["error"] = f"Failed to read CSV: {exc}"
        return None, report

    report["row_count"] = int(len(raw_df))
    report["column_count"] = int(raw_df.shape[1])
    report["missing_values"] = int(raw_df.isna().sum().sum())

    cleaned = clean_multimodal_dataframe(raw_df.copy(), dataset_name=dataset_name, add_prospect_id=False)
    validation = validate_multimodal_schema(cleaned)
    report["label_distribution"] = validation.get("label_distribution", {})
    report["schema_valid"] = bool(validation.get("is_valid", False))
    report["exact_schema_match"] = bool(validation.get("exact_schema_match", False))

    if report["column_count"] != len(EXPECTED_MULTIMODAL_COLUMNS) or not report["exact_schema_match"]:
        report["error"] = "Schema does not exactly match expected multimodal columns"
        return None, report

    if required_label is not None:
        labels = set(cleaned["label"].astype(str).str.lower().unique()) if "label" in cleaned.columns else set()
        if labels != {required_label}:
            report["error"] = f"Expected only label='{required_label}' but found {sorted(labels)}"
            return None, report

    if allowed_labels is not None:
        labels = set(cleaned["label"].astype(str).str.lower().unique()) if "label" in cleaned.columns else set()
        if not labels.issubset(allowed_labels):
            report["error"] = f"Found labels outside allowed set: {sorted(labels - allowed_labels)}"
            return None, report

    ordered = cleaned[EXPECTED_MULTIMODAL_COLUMNS].copy()
    report["imported_successfully"] = True
    return ordered, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and validate external multimodal feature CSVs.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--warm", required=True, help="Path to the external warm.csv file.")
    parser.add_argument("--ravdess", default=None, help="Path to the external ravdess.csv file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    logger = setup_logger("import_feature_csvs", cfg["paths"]["logs_root"] / "import_feature_csvs.log")

    features_root = cfg["paths"]["features_root"]
    metrics_root = cfg["paths"]["metrics_root"]
    features_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    reports = []

    warm_path = _resolve_input_path(args.warm)
    warm_df, warm_report = _process_file(warm_path, dataset_name="warm", required_label="warm")
    warm_report["destination_path"] = str(features_root / "warm.csv")
    reports.append(warm_report)
    if warm_df is not None:
        dataframe_to_csv(warm_df, features_root / "warm.csv")

    if args.ravdess:
        ravdess_path = _resolve_input_path(args.ravdess)
        ravdess_df, ravdess_report = _process_file(ravdess_path, dataset_name="ravdess", allowed_labels={"hot", "warm", "cold"})
        ravdess_report["destination_path"] = str(features_root / "ravdess.csv")
        reports.append(ravdess_report)
        if ravdess_df is not None:
            dataframe_to_csv(ravdess_df, features_root / "ravdess.csv")
    else:
        ravdess_report = {
            "file_name": "ravdess.csv",
            "source_path": None,
            "destination_path": str(features_root / "ravdess.csv"),
            "row_count": 0,
            "column_count": 0,
            "missing_values": 0,
            "label_distribution": {},
            "schema_valid": False,
            "exact_schema_match": False,
            "imported_successfully": False,
            "error": "No RAVDESS input provided",
        }
        reports.append(ravdess_report)

    report_payload = {
        "warm": warm_report,
        "ravdess": ravdess_report,
        "overall_success": bool(warm_report["imported_successfully"]) and (not args.ravdess or bool(reports[-1]["imported_successfully"])),
    }
    save_json(report_payload, metrics_root / "import_feature_csvs_report.json")

    logger.info("Import report saved to %s", metrics_root / "import_feature_csvs_report.json")
    if not warm_report["imported_successfully"] or (args.ravdess and not ravdess_report["imported_successfully"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
