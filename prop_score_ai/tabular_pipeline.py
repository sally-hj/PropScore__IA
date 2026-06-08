"""Tabular dataset preparation, cleaning, and preprocessing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import (
    CATEGORICAL_TABULAR_COLUMNS,
    EXPECTED_TABULAR_COLUMNS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    TABULAR_CATEGORICAL_FEATURES,
    TABULAR_EXCLUDED_FEATURES,
    TABULAR_NUMERIC_FEATURES,
    TABULAR_RAW_LABEL_MAP,
    TABULAR_RENAME_MAP,
)
from .io_utils import dataframe_to_csv, ensure_dir, save_json
from .logging_utils import setup_logger
from .sklearn_compat import make_one_hot_encoder

TABULAR_FEATURE_COLUMNS = TABULAR_NUMERIC_FEATURES + TABULAR_CATEGORICAL_FEATURES


def _normalize_multiclass_label(value) -> str:
    if pd.isna(value):
        return "cold"
    normalized = str(value).strip().lower()
    normalized = normalized.replace("é", "e")
    normalized = normalized.replace("è", "e")
    normalized = normalized.replace("ê", "e")
    normalized = normalized.replace("ï", "i")
    normalized = normalized.replace("î", "i")
    normalized = normalized.replace("à", "a")
    normalized = normalized.replace("ç", "c")
    normalized = normalized.replace("ô", "o")
    normalized = normalized.replace("û", "u")
    normalized = normalized.replace("ù", "u")
    return TABULAR_RAW_LABEL_MAP.get(normalized, normalized if normalized in LABEL_TO_ID else "cold")


def _ensure_prospect_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "prospect_id" not in out.columns:
        out["prospect_id"] = [f"tabular_{idx:06d}" for idx in range(1, len(out) + 1)]
    else:
        out["prospect_id"] = out["prospect_id"].astype("string")
        missing_mask = out["prospect_id"].isna() | (out["prospect_id"].astype(str).str.strip() == "")
        if missing_mask.any():
            start = 1
            for idx in out.index[missing_mask]:
                out.at[idx, "prospect_id"] = f"tabular_{start:06d}"
                start += 1
    return out


def rename_tabular_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=TABULAR_RENAME_MAP).copy()
    renamed.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(col).strip()).strip("_").lower() for col in renamed.columns]
    return renamed


def clean_tabular_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "label_raw" in out.columns:
        out["label_raw"] = out["label_raw"].astype("string").str.strip()
        out["label"] = out["label_raw"].map(_normalize_multiclass_label)
    elif "label" in out.columns:
        out["label"] = out["label"].map(_normalize_multiclass_label)
    else:
        out["label"] = "cold"

    out["target_multiclasse_id"] = out["label"].map(LABEL_TO_ID).astype(int)

    if "binary_target" in out.columns:
        out["binary_target"] = out["binary_target"].astype("string").str.strip()
    if "score_lead" in out.columns:
        out["score_lead"] = pd.to_numeric(out["score_lead"], errors="coerce")

    for col in TABULAR_NUMERIC_FEATURES + ["score_lead", "target_multiclasse_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in TABULAR_NUMERIC_FEATURES + ["score_lead", "interaction_score", "age", "income", "budget", "num_visits", "total_time_spent_website", "page_views_per_visit", "ratio_budget_income"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    for col in TABULAR_CATEGORICAL_FEATURES + ["source_lead"]:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("unknown").str.strip()
    out = _ensure_prospect_id(out)
    out["label"] = out["label"].astype("string")
    out["label_raw"] = out.get("label_raw", out["label"]).astype("string")
    return out


def prepare_tabular_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    report_path: str | Path,
    logger=None,
) -> pd.DataFrame:
    """Load dataset_final_catboost.csv, clean it, and write prospects.csv."""
    logger = logger or setup_logger("prepare_tabular_dataset")
    input_csv = Path(input_csv)
    if not input_csv.is_absolute():
        input_csv = input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Tabular input file not found: {input_csv}")

    raw = pd.read_csv(input_csv)
    renamed = rename_tabular_columns(raw)
    cleaned = clean_tabular_dataset(renamed)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_csv, index=False)

    report = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "row_count": int(cleaned.shape[0]),
        "column_count": int(cleaned.shape[1]),
        "missing_values": int(cleaned.isna().sum().sum()),
        "class_distribution": cleaned["label"].value_counts(dropna=False).to_dict(),
        "numeric_columns": [c for c in TABULAR_NUMERIC_FEATURES if c in cleaned.columns],
        "categorical_columns": [c for c in TABULAR_CATEGORICAL_FEATURES if c in cleaned.columns],
    }
    report_path = Path(report_path)
    save_json(report, report_path)
    logger.info("Prepared tabular dataset: %s -> %s", input_csv, output_csv)
    return cleaned


def load_or_create_tabular_data(
    prospects_path: str | Path,
    multimodal_df: pd.DataFrame | None = None,
    output_mock_path: str | Path | None = None,
    use_mock: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the cleaned tabular dataset or create a mock fallback for demos."""
    prospects_path = Path(prospects_path)
    if prospects_path.exists():
        df = pd.read_csv(prospects_path)
        if "label" in df.columns:
            df["label"] = df["label"].map(_normalize_multiclass_label)
        if "target_multiclasse_id" not in df.columns and "label" in df.columns:
            df["target_multiclasse_id"] = df["label"].map(LABEL_TO_ID)
        return clean_tabular_dataset(df)

    if not use_mock:
        return pd.DataFrame(columns=EXPECTED_TABULAR_COLUMNS)

    rng = np.random.default_rng(random_state)
    rows = []
    base_rows = list(multimodal_df.itertuples(index=False)) if multimodal_df is not None and not multimodal_df.empty else []
    if base_rows:
        for idx, row in enumerate(base_rows, start=1):
            label = getattr(row, "label", "cold")
            label = _normalize_multiclass_label(label)
            rows.append(
                {
                    "prospect_id": f"tabular_{idx:06d}",
                    "age": int(np.clip(rng.normal(42, 8), 18, 80)),
                    "income": int(np.clip(rng.normal(35000, 9000), 5000, 150000)),
                    "profession": rng.choice(["commercial", "engineer", "teacher", "doctor", "retired", "self_employed"]),
                    "property_type": rng.choice(["apartment", "house", "villa", "plot", "studio"]),
                    "budget": int(np.clip(rng.normal(250000, 70000), 50000, 1000000)),
                    "num_visits": int(np.clip(rng.normal(2, 2), 0, 20)),
                    "source_lead": rng.choice(["website", "whatsapp", "call", "facebook", "referral"]),
                    "binary_target": label,
                    "total_time_spent_website": float(np.clip(rng.normal(180, 120), 0, 4000)),
                    "page_views_per_visit": float(np.clip(rng.normal(4.5, 2), 1, 30)),
                    "city": rng.choice(["Casablanca", "Rabat", "Marrakech", "Tanger", "Fes", "Agadir"]),
                    "lead_origin": rng.choice(["website", "ads", "social", "referral", "organic"]),
                    "last_activity": rng.choice(["viewed_property", "called", "messaged", "follow_up", "no_response"]),
                    "last_notable_activity": rng.choice(["site_visit", "chat", "email_open", "callback", "meeting"]),
                    "score_lead": int(np.clip(rng.normal(55, 18), 0, 100)),
                    "label_raw": {"cold": "Froid", "warm": "Tiede", "hot": "Chaud"}[label],
                    "ratio_budget_income": float(np.clip(rng.normal(6.0, 2.5), 0.1, 25.0)),
                    "interaction_score": float(np.clip(rng.normal(55, 20), 0, 100)),
                    "label": label,
                    "target_multiclasse_id": LABEL_TO_ID[label],
                }
            )
    else:
        labels = ["cold"] * 30 + ["warm"] * 30 + ["hot"] * 30
        for idx, label in enumerate(labels, start=1):
            rows.append(
                {
                    "prospect_id": f"tabular_{idx:06d}",
                    "age": int(np.clip(rng.normal(42 if label == "warm" else (49 if label == "cold" else 36), 8), 18, 80)),
                    "income": int(np.clip(rng.normal(35000 if label == "warm" else (22000 if label == "cold" else 52000), 9000), 5000, 150000)),
                    "profession": rng.choice(["commercial", "engineer", "teacher", "doctor", "retired", "self_employed"]),
                    "property_type": rng.choice(["apartment", "house", "villa", "plot", "studio"]),
                    "budget": int(np.clip(rng.normal(250000 if label == "warm" else (140000 if label == "cold" else 350000), 70000), 50000, 1000000)),
                    "num_visits": int(np.clip(rng.normal(2 if label == "warm" else (0 if label == "cold" else 5), 2), 0, 20)),
                    "source_lead": rng.choice(["website", "whatsapp", "call", "facebook", "referral"]),
                    "binary_target": label,
                    "total_time_spent_website": float(np.clip(rng.normal(180 if label != "cold" else 70, 120), 0, 4000)),
                    "page_views_per_visit": float(np.clip(rng.normal(4.5 if label != "cold" else 2.0, 2), 1, 30)),
                    "city": rng.choice(["Casablanca", "Rabat", "Marrakech", "Tanger", "Fes", "Agadir"]),
                    "lead_origin": rng.choice(["website", "ads", "social", "referral", "organic"]),
                    "last_activity": rng.choice(["viewed_property", "called", "messaged", "follow_up", "no_response"]),
                    "last_notable_activity": rng.choice(["site_visit", "chat", "email_open", "callback", "meeting"]),
                    "score_lead": int(np.clip(rng.normal(35 if label == "cold" else (55 if label == "warm" else 80), 18), 0, 100)),
                    "label_raw": {"cold": "Froid", "warm": "Tiede", "hot": "Chaud"}[label],
                    "ratio_budget_income": float(np.clip(rng.normal(6.0 if label != "cold" else 4.0, 2.5), 0.1, 25.0)),
                    "interaction_score": float(np.clip(rng.normal(35 if label == "cold" else (55 if label == "warm" else 80), 20), 0, 100)),
                    "label": label,
                    "target_multiclasse_id": LABEL_TO_ID[label],
                }
            )
    df = pd.DataFrame(rows)
    if output_mock_path is not None:
        dataframe_to_csv(df, output_mock_path)
    return clean_tabular_dataset(df)


def build_tabular_preprocessor(
    df: pd.DataFrame,
    *,
    use_score_lead: bool = False,
    use_binary_target_as_feature: bool = False,
    target_column: str = "label",
) -> tuple[ColumnTransformer, list[str], list[str], dict, list[str]]:
    """Create a tabular preprocessor and metadata for training/inference."""
    df = clean_tabular_dataset(df)
    numeric_cols = [c for c in TABULAR_NUMERIC_FEATURES if c in df.columns]
    categorical_cols = [c for c in TABULAR_CATEGORICAL_FEATURES if c in df.columns]

    if use_score_lead and "score_lead" in df.columns:
        print("WARNING: score_lead may cause target leakage because it appears to define the multiclass label.")
        numeric_cols = numeric_cols + ["score_lead"]

    if use_binary_target_as_feature and "binary_target" in df.columns:
        categorical_cols = categorical_cols + ["binary_target"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    feature_cols = numeric_cols + categorical_cols
    defaults: dict[str, object] = {}
    for col in numeric_cols:
        defaults[col] = float(pd.to_numeric(df[col], errors="coerce").median()) if col in df.columns else 0.0
    for col in categorical_cols:
        defaults[col] = str(df[col].mode().iloc[0]) if col in df.columns and not df[col].mode().empty else "unknown"

    excluded = sorted(set(TABULAR_EXCLUDED_FEATURES))
    if use_score_lead and "score_lead" in excluded:
        excluded.remove("score_lead")
    if use_binary_target_as_feature and "binary_target" in excluded:
        excluded.remove("binary_target")

    return preprocessor, numeric_cols, categorical_cols, defaults, excluded


def get_tabular_feature_frame(
    df: pd.DataFrame,
    *,
    use_score_lead: bool = False,
    use_binary_target_as_feature: bool = False,
    target_column: str = "label",
) -> pd.DataFrame:
    df = clean_tabular_dataset(df)
    columns = list(TABULAR_NUMERIC_FEATURES) + list(TABULAR_CATEGORICAL_FEATURES)
    if use_score_lead and "score_lead" in df.columns:
        columns = columns + ["score_lead"]
    if use_binary_target_as_feature and "binary_target" in df.columns:
        columns = columns + ["binary_target"]
    return df[[c for c in columns if c in df.columns]].copy()


def save_encoders_bundle(bundle: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return output_path


def _generate_mock_tabular_from_multimodal(multimodal_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    return load_or_create_tabular_data(
        prospects_path="__missing__",
        multimodal_df=multimodal_df,
        output_mock_path=None,
        use_mock=True,
        random_state=random_state,
    )
