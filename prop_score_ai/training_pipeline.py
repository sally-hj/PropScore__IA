"""XGBoost training and evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import ID_TO_LABEL, LABEL_TO_ID
from .io_utils import dataframe_to_csv, save_json
from .logging_utils import setup_logger
from .sklearn_compat import make_one_hot_encoder
from datetime import datetime, timezone

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional import guard
    xgb = None


IDENTIFIER_COLUMNS = {
    "prospect_id",
    "video_id",
    "frame_dir",
    "audio_path",
    "transcript",
    "analysis_method",
    "split",
    "synthetic_pair_id",
}

TARGET_COLUMNS = {
    "label",
    "label_raw",
    "target_multiclasse_id",
    "binary_target",
}

TEXT_OR_PATH_HINTS = ("path", "file", "dir", "transcript", "note", "comment")

VIDEO_FEATURE_HINTS = (
    "dominant_emotion",
    "emotion_confidence",
    "face_detected_ratio",
    "landmark_quality_score",
    "vision_score",
    "mfcc_mean_",
    "mfcc_std_",
    "pitch_mean",
    "pitch_std",
    "energy_mean",
    "energy_std",
    "zero_crossing_rate_mean",
    "spectral_centroid_mean",
    "speech_rate_estimate",
    "audio_score",
    "sentiment_label",
    "sentiment_score",
    "keyword_count",
    "positive_keyword_count",
    "hesitation_keyword_count",
    "negative_keyword_count",
    "keyword_type",
    "semantic_score",
)


def _has_video_features(df: pd.DataFrame) -> bool:
    return any(any(hint in col for hint in VIDEO_FEATURE_HINTS) for col in df.columns)


def _select_feature_columns(
    df: pd.DataFrame,
    *,
    use_score_lead: bool = False,
    use_binary_target_as_feature: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    feature_cols: list[str] = []
    categorical_cols: list[str] = []
    excluded: list[str] = []

    for col in df.columns:
        lower = col.lower()
        if col in IDENTIFIER_COLUMNS or col in TARGET_COLUMNS:
            excluded.append(col)
            continue
        if lower.startswith("label") or lower.startswith("binary_target") or lower.startswith("target_"):
            excluded.append(col)
            continue
        if any(hint in lower for hint in TEXT_OR_PATH_HINTS):
            excluded.append(col)
            continue
        if col == "score_lead" and not use_score_lead:
            excluded.append(col)
            continue
        if col == "binary_target" and not use_binary_target_as_feature:
            excluded.append(col)
            continue
        if col == "score_lead" and use_score_lead:
            print("WARNING: score_lead may cause target leakage because it appears to define the multiclass label.")
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
        else:
            feature_cols.append(col)
            categorical_cols.append(col)

    return feature_cols, categorical_cols, excluded


def _build_preprocessor(
    df: pd.DataFrame,
    *,
    use_score_lead: bool = False,
    use_binary_target_as_feature: bool = False,
) -> tuple[ColumnTransformer, list[str], list[str], dict, list[str]]:
    feature_cols, categorical_cols, excluded_cols = _select_feature_columns(
        df,
        use_score_lead=use_score_lead,
        use_binary_target_as_feature=use_binary_target_as_feature,
    )
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

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

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    defaults: dict[str, object] = {}
    for col in numeric_cols:
        defaults[col] = float(pd.to_numeric(df[col], errors="coerce").median()) if col in df.columns else 0.0
    for col in categorical_cols:
        defaults[col] = str(df[col].mode().iloc[0]) if col in df.columns and not df[col].mode().empty else "unknown"

    return preprocessor, numeric_cols, categorical_cols, defaults, excluded_cols


def _sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    weights = y.map(lambda cls: 1.0 / max(counts.get(cls, 1), 1)).astype(float)
    weights = weights * (len(y) / weights.sum())
    return weights.to_numpy()


def _normalize_xgb_params(xgb_params: dict | None) -> dict:
    if not xgb_params:
        return {}
    normalized: dict = {}
    for key, value in xgb_params.items():
        normalized[key] = str(value) if isinstance(value, Path) else value
    return normalized


def train_xgboost_model(
    dataset_csv: str | Path,
    model_path: str | Path,
    encoders_path: str | Path,
    feature_schema_path: str | Path,
    metrics_dir: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    xgb_params: dict | None = None,
    use_score_lead: bool = False,
    use_binary_target_as_feature: bool = False,
    target_column: str = "label",
    training_context: dict | None = None,
    logger=None,
) -> dict:
    logger = logger or setup_logger("training")
    if xgb is None:
        raise RuntimeError("xgboost is not available. Install the requirements first.")

    dataset_csv = Path(dataset_csv)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_csv}")

    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise ValueError("Training dataset is empty")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    df = df.copy()
    df[target_column] = df[target_column].astype("string").str.lower()
    if "label" not in df.columns:
        df["label"] = df[target_column]
    if "target_multiclasse_id" not in df.columns:
        df["target_multiclasse_id"] = df["label"].map(LABEL_TO_ID)

    y = df[target_column].map(LABEL_TO_ID).astype(int)
    X = df.drop(columns=[target_column], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor, numeric_cols, categorical_cols, defaults, excluded_cols = _build_preprocessor(
        X_train,
        use_score_lead=use_score_lead,
        use_binary_target_as_feature=use_binary_target_as_feature,
    )

    X_train_matrix = preprocessor.fit_transform(X_train)
    X_test_matrix = preprocessor.transform(X_test)

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": random_state,
        "tree_method": "hist",
    }
    params.update(_normalize_xgb_params(xgb_params))

    model = xgb.XGBClassifier(**params)
    sample_weight = _sample_weights(y_train)
    model.fit(X_train_matrix, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test_matrix)
    y_prob = model.predict_proba(X_test_matrix)

    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        target_names=[ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    model_path = Path(model_path)
    encoders_path = Path(encoders_path)
    feature_schema_path = Path(feature_schema_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    feature_schema_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "preprocessor": preprocessor,
        "feature_schema": list(X_train.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_defaults": defaults,
        "excluded_columns": excluded_cols,
        "label_mapping": LABEL_TO_ID,
        "inverse_label_mapping": ID_TO_LABEL,
        "random_state": random_state,
        "training_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": target_column,
        "model_mode": "hybrid" if _has_video_features(X_train) else "tabular_only",
        "use_score_lead": bool(use_score_lead),
        "use_binary_target_as_feature": bool(use_binary_target_as_feature),
        "class_distribution": df[target_column].value_counts().to_dict(),
        "training_date": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(model, model_path)
    joblib.dump(model_bundle, encoders_path)

    feature_schema = {
        "feature_columns": list(X_train.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "excluded_columns": excluded_cols,
        "target_column": target_column,
        "label_mapping": LABEL_TO_ID,
        "model_mode": model_bundle["model_mode"],
        "use_score_lead": bool(use_score_lead),
        "use_binary_target_as_feature": bool(use_binary_target_as_feature),
        "feature_defaults": defaults,
    }
    save_json(feature_schema, feature_schema_path)

    report_path = metrics_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count_before_encoding": len(X_train.columns),
        "feature_count_after_encoding": int(X_train_matrix.shape[1]),
        "label_mapping": LABEL_TO_ID,
        "model_mode": model_bundle["model_mode"],
        "model_type": model_bundle["model_mode"],
        "training_date": model_bundle["training_date"],
        "class_distribution_train": {ID_TO_LABEL[int(k)]: int(v) for k, v in y_train.value_counts().sort_index().items()},
        "class_distribution_test": {ID_TO_LABEL[int(k)]: int(v) for k, v in y_test.value_counts().sort_index().items()},
    }
    if training_context:
        for key in ["model_type", "used_multimodal_files", "used_ravdess", "ravdess_mode"]:
            if key in training_context:
                metrics[key] = training_context[key]
    save_json(metrics, metrics_dir / "metrics.json")

    split_report = pd.DataFrame(
        {
            "split": ["train"] * len(X_train) + ["test"] * len(X_test),
            "prospect_id": list(X_train.get("prospect_id", pd.Series(index=X_train.index, dtype="string"))) + list(X_test.get("prospect_id", pd.Series(index=X_test.index, dtype="string"))),
            "video_id": list(X_train.get("video_id", pd.Series(index=X_train.index, dtype="string"))) + list(X_test.get("video_id", pd.Series(index=X_test.index, dtype="string"))),
            "label": list(X_train.get("label", pd.Series(index=X_train.index, dtype="string"))) + list(X_test.get("label", pd.Series(index=X_test.index, dtype="string"))),
        }
    )
    dataframe_to_csv(split_report, metrics_dir / "split_report.csv")
    save_json(
        {
            "train_ids": X_train.get("video_id", pd.Series(dtype="string")).astype(str).tolist(),
            "test_ids": X_test.get("video_id", pd.Series(dtype="string")).astype(str).tolist(),
            "train_count": int(len(X_train)),
            "test_count": int(len(X_test)),
        },
        metrics_dir / "split_report.json",
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID_TO_LABEL[i] for i in range(3)],
        yticklabels=[ID_TO_LABEL[i] for i in range(3)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(metrics_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    logger.info("Training complete: accuracy=%.4f f1=%.4f", accuracy, macro_f1)
    return {
        "model_path": str(model_path),
        "encoders_path": str(encoders_path),
        "feature_schema_path": str(feature_schema_path),
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
