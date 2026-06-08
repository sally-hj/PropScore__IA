from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import project_root
from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.constants import LABEL_TO_ID
from prop_score_ai.logging_utils import setup_logger


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "label" in out.columns:
        out["label"] = out["label"].astype("string").str.lower()
    return out


def main():
    parser = argparse.ArgumentParser(description="Check leakage signals in the structured dataset.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="Path to data/processed/prospects.csv")
    parser.add_argument("--output", default=None, help="Leakage report output path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (project_root() / input_path).resolve()
    output_path = Path(args.output or (cfg["paths"]["metrics_root"] / "leakage_report.txt"))
    logger = setup_logger("dataset_leakage", cfg["paths"]["logs_root"] / "check_dataset_leakage.log")

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = _normalize_input(pd.read_csv(input_path))
    if "label" not in df.columns:
        raise ValueError("The input dataset must contain a label column.")

    df["target_multiclasse_id"] = df["label"].map(LABEL_TO_ID)
    lines = []
    lines.append(f"rows: {len(df)}")
    lines.append(f"columns: {len(df.columns)}")
    lines.append(f"missing_values: {int(df.isna().sum().sum())}")
    lines.append(f"class_distribution: {df['label'].value_counts().to_dict()}")
    lines.append("")

    if "score_lead" in df.columns:
        ctab = pd.crosstab(df["score_lead"], df["label"], normalize="index")
        raw_ctab = pd.crosstab(df["score_lead"], df["label"])
        lines.append("score_lead x label crosstab:")
        lines.append(raw_ctab.to_string())
        lines.append("")
        lines.append("score_lead x label normalized by row:")
        lines.append(ctab.to_string())
        lines.append("")
        if not ctab.empty:
            best_row_purity = float(ctab.max(axis=1).mean())
            lines.append(f"mean_row_purity: {best_row_purity:.4f}")
            if best_row_purity >= 0.95:
                lines.append("WARNING: score_lead appears to almost perfectly separate the classes. Do not use it for fair model evaluation.")
                logger.warning("score_lead appears to almost perfectly separate the classes.")
    else:
        lines.append("score_lead column not found.")
        lines.append("")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"target_multiclasse_id"}]
    lines.append("correlation_with_target_id:")
    if numeric_cols:
        corr = df[numeric_cols + ["target_multiclasse_id"]].corr(numeric_only=True)["target_multiclasse_id"].drop("target_multiclasse_id", errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
        lines.append(corr.to_string())
    else:
        lines.append("no_numeric_columns_found")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Leakage report saved to %s", output_path)


if __name__ == "__main__":
    main()
