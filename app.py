from __future__ import annotations

import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import av
except Exception:  # pragma: no cover
    av = None

try:
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
except Exception:  # pragma: no cover
    WebRtcMode = None
    RTCConfiguration = None

    class VideoProcessorBase:  # type: ignore
        pass

    def webrtc_streamer(*args, **kwargs):  # type: ignore
        return None

from prop_score_ai.config import ensure_runtime_dirs, load_config
from prop_score_ai.constants import ID_TO_LABEL, LABEL_TO_ID, LABELS
from prop_score_ai.inference_pipeline import load_bundle, predict_from_raw_row, predict_visual_frame, process_uploaded_video, _default_raw_row
from prop_score_ai.online_video_model import DEFAULT_VIDEO_MODEL_NAME, DEFAULT_VIDEO_PROMPTS, score_video_against_prompts
from prop_score_ai.audio_pipeline import WhisperTranscriber, extract_audio_from_video, extract_librosa_features
from prop_score_ai.nlp_pipeline import CamembertSemanticAnalyzer
from prop_score_ai.video_pipeline import analyze_video_visual_signals
from prop_score_ai.io_utils import prospect_id_from_video_path, video_id_from_path
from utils.feature_schema import clean_multimodal_dataframe, validate_multimodal_schema

st.set_page_config(page_title="PropScore_AI", page_icon="🏠", layout="wide")

COLOR_MAP = {
    "hot": "#EF4444",   # Crimson Rose
    "warm": "#F59E0B",  # Sunset Amber
    "cold": "#3B82F6",  # Muted Frost
}

DISPLAY_LABEL_MAP = {"cold": "Cold", "warm": "Warm", "hot": "Hot"}


def _class_scale() -> alt.Scale:
    return alt.Scale(domain=["hot", "warm", "cold"], range=[COLOR_MAP["hot"], COLOR_MAP["warm"], COLOR_MAP["cold"]])


def _chart_title(title: str | None) -> dict[str, Any]:
    return {"title": title} if title else {}


def _altair_bar(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None, sort=None) -> alt.Chart:
    chart = alt.Chart(df).mark_bar()
    enc = {
        "x": alt.X(f"{x}:N", sort=sort, title=x.replace("_", " ").title()) if df[x].dtype == object or df[x].dtype.name.startswith("string") else alt.X(f"{x}:Q", title=x.replace("_", " ").title()),
        "y": alt.Y(f"{y}:Q", title=y.replace("_", " ").title()),
        "tooltip": [x, y],
    }
    if color and color in df.columns:
        enc["color"] = alt.Color(f"{color}:N", scale=_class_scale(), title=color.replace("_", " ").title())
    chart = chart.encode(**enc).properties(**_chart_title(title))
    return chart


def _altair_grouped_bar(df: pd.DataFrame, x: str, y: str, group: str, title: str | None = None) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", title=x.replace("_", " ").title(), axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=y.replace("_", " ").title()),
            xOffset=alt.XOffset(f"{group}:N"),
            color=alt.Color(f"{group}:N", scale=_class_scale(), title=group.replace("_", " ").title()),
            tooltip=[x, group, y],
        )
        .properties(**_chart_title(title))
    )


def _altair_hist(df: pd.DataFrame, col: str, color: str | None = None, title: str | None = None) -> alt.Chart:
    chart = alt.Chart(df).mark_bar()
    enc = {
        "x": alt.X(f"{col}:Q", bin=alt.Bin(maxbins=24), title=col.replace("_", " ").title()),
        "y": alt.Y("count()", title="Count"),
    }
    if color and color in df.columns:
        enc["color"] = alt.Color(f"{color}:N", scale=_class_scale(), title="Class")
    chart = chart.encode(**enc).properties(**_chart_title(title))
    return chart


def _altair_box(df: pd.DataFrame, x: str, y: str, title: str | None = None) -> alt.Chart:
    chart = alt.Chart(df).mark_boxplot(extent="min-max").encode(
        x=alt.X(f"{x}:N", title=x.replace("_", " ").title()),
        y=alt.Y(f"{y}:Q", title=y.replace("_", " ").title()),
        color=alt.Color(f"{x}:N", scale=_class_scale(), legend=None),
    )
    return chart.properties(**_chart_title(title))


def _altair_pie(df: pd.DataFrame, names: str, values: str, title: str | None = None) -> alt.Chart:
    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(f"{values}:Q"),
        color=alt.Color(f"{names}:N", scale=alt.Scale(domain=["Hot", "Warm", "Cold"], range=[COLOR_MAP["hot"], COLOR_MAP["warm"], COLOR_MAP["cold"]]), title=names.replace("_", " ").title()),
        tooltip=[names, values],
    )
    return chart.properties(**_chart_title(title))


def _altair_probabilities(df: pd.DataFrame, title: str | None = None) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("class:N", title="Class", sort=["Hot", "Warm", "Cold"]),
            y=alt.Y("probability:Q", title="Probability"),
            color=alt.Color("class:N", scale=alt.Scale(domain=["Hot", "Warm", "Cold"], range=[COLOR_MAP["hot"], COLOR_MAP["warm"], COLOR_MAP["cold"]]), legend=None),
            tooltip=["class", alt.Tooltip("probability:Q", format=".4f")],
        )
        .properties(height=280, **_chart_title(title))
    )


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_csv_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_json_safe(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_status(path: str | Path) -> str:
    path = Path(path)
    return "Available" if path.exists() and path.stat().st_size > 0 else "Missing"


def _multimodal_file_summary(path: str | Path, dataset_name: str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {
            "path": path,
            "available": False,
            "rows": 0,
            "columns": 0,
            "missing_values": 0,
            "label_distribution": {},
            "schema_valid": False,
            "exact_schema_match": False,
            "df": pd.DataFrame(),
        }
    raw_df = load_csv_safe(path)
    cleaned_df = clean_multimodal_dataframe(raw_df.copy(), dataset_name=dataset_name, add_prospect_id=False)
    validation = validate_multimodal_schema(cleaned_df)
    return {
        "path": path,
        "available": True,
        "rows": int(len(raw_df)),
        "columns": int(raw_df.shape[1]),
        "missing_values": int(raw_df.isna().sum().sum()),
        "label_distribution": cleaned_df["label"].value_counts().to_dict() if "label" in cleaned_df.columns else {},
        "schema_valid": bool(validation["is_valid"]),
        "exact_schema_match": bool(validation["exact_schema_match"]),
        "df": cleaned_df,
    }


def normalize_label_display(label: str | None) -> str:
    if not label:
        return "Unknown"
    label = str(label).strip().lower()
    return DISPLAY_LABEL_MAP.get(label, label.title())


def get_class_color(label: str | None) -> str:
    if not label:
        return "#7a7a7a"
    return COLOR_MAP.get(str(label).strip().lower(), "#7a7a7a")


def show_metric_card(title: str, value: Any, help_text: str | None = None) -> None:
    st.metric(title, value, help=help_text)


def align_features_to_schema(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    feature_cols = schema.get("feature_columns") or schema.get("feature_schema") or []
    defaults = schema.get("feature_defaults", {})
    aligned = pd.DataFrame([{col: defaults.get(col) for col in feature_cols}])
    if not df.empty:
        row = df.iloc[0].to_dict()
        for col in feature_cols:
            if col in row:
                aligned.at[0, col] = row[col]
    return aligned


def _load_runtime_objects(prefix: str = ""):
    cfg = load_config()
    ensure_runtime_dirs(cfg)
    prefix = f"{prefix}_" if prefix else ""
    model_path = cfg["paths"]["models_root"] / f"{prefix}xgboost_model.pkl"
    encoders_path = cfg["paths"]["models_root"] / f"{prefix}encoders.pkl"
    feature_schema_path = cfg["paths"]["models_root"] / f"{prefix}feature_schema.json"
    if model_path.exists() and encoders_path.exists():
        model, bundle = load_bundle(model_path, encoders_path)
    else:
        model, bundle = None, None
    schema = load_json_safe(feature_schema_path)
    if bundle is not None and schema:
        bundle["feature_schema_details"] = schema
        bundle.setdefault("model_mode", schema.get("model_mode"))
        bundle.setdefault("feature_defaults", schema.get("feature_defaults", bundle.get("feature_defaults", {})))
        bundle.setdefault("feature_schema", schema.get("feature_columns", bundle.get("feature_schema", [])))
    return cfg, model, bundle, schema


@st.cache_resource(show_spinner=False)
def load_model_bundle(prefix: str = ""):
    return _load_runtime_objects(prefix)


def model_mode(bundle: dict | None, metrics: dict | None = None) -> str:
    if metrics and metrics.get("model_mode"):
        return str(metrics["model_mode"])
    if bundle and bundle.get("model_mode"):
        return str(bundle["model_mode"])
    if bundle and bundle.get("feature_schema_details", {}).get("model_mode"):
        return str(bundle["feature_schema_details"]["model_mode"])
    return "tabular_only"

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        /* Global font and background override */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0B0F19;
            color: #F3F4F6;
        }

        /* Header style override */
        [data-testid="stHeader"] {
            background-color: rgba(11, 15, 25, 0.8) !important;
            backdrop-filter: blur(10px) !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        /* Styled containers/cards */
        .dashboard-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(8px);
        }

        .hot-card {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%) !important;
            border: 1px solid rgba(239, 68, 68, 0.4) !important;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.1);
            backdrop-filter: blur(8px);
        }

        .warm-card {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%) !important;
            border: 1px solid rgba(245, 158, 11, 0.4) !important;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.1);
            backdrop-filter: blur(8px);
        }

        .cold-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.05) 100%) !important;
            border: 1px solid rgba(59, 130, 246, 0.4) !important;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
            backdrop-filter: blur(8px);
        }

        /* Custom fonts & highlights */
        .gradient-text {
            background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #F3F4F6;
            margin-bottom: 0.75rem;
        }

        /* Video container constraints */
        .video-container {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            margin-bottom: 1.5rem;
            background: #000;
        }

        /* Override default streamlit metric card padding & background */
        div[data-testid="metric-container"] {
            background-color: rgba(30, 41, 59, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def page_dataset_dashboard(cfg: dict[str, Any]) -> None:
    st.header("Dataset Dashboard")
    prospects_path = cfg["paths"]["processed_root"] / "prospects.csv"
    df = load_csv_safe(prospects_path)
    if df.empty:
        st.warning("No processed tabular dataset found. Run scripts/00_prepare_tabular_dataset.py first.")
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    categorical_cols = [c for c in categorical_cols if c not in {"prospect_id", "label", "label_raw", "binary_target"}]
    label_counts = df["label"].astype(str).str.lower().value_counts().reindex(["hot", "warm", "cold"], fill_value=0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Columns", f"{len(df.columns):,}")
    k3.metric("Numerical Features", f"{len(numeric_cols):,}")
    k4.metric("Categorical Features", f"{len(categorical_cols):,}")

    st.metric("Missing Values", int(df.isna().sum().sum()))

    chart_df = label_counts.reset_index()
    chart_df.columns = ["label", "count"]
    chart_df["label_display"] = chart_df["label"].map(normalize_label_display)
    c1, c2 = st.columns(2)
    c1.altair_chart(_altair_bar(chart_df, "label_display", "count", color="label", title="Hot / Warm / Cold Distribution", sort=["Hot", "Warm", "Cold"]), use_container_width=True)
    c2.altair_chart(_altair_pie(chart_df, "label_display", "count", title="Class Distribution"), use_container_width=True)

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Numeric Summary Statistics")
    st.dataframe(df[numeric_cols].describe().T if numeric_cols else pd.DataFrame(), use_container_width=True)

    st.subheader("Categorical Value Counts")
    for col in ["profession", "property_type", "city", "source_lead", "lead_origin", "last_activity", "last_notable_activity"]:
        if col in df.columns:
            with st.expander(col, expanded=False):
                counts = df[col].astype(str).value_counts().head(20)
                st.dataframe(counts.to_frame("count"), use_container_width=True)


def page_tabular_analytics(cfg: dict[str, Any]) -> None:
    st.header("Tabular Analytics")
    df = load_csv_safe(cfg["paths"]["processed_root"] / "prospects.csv")
    if df.empty:
        st.warning("No processed tabular dataset found. Run scripts/00_prepare_tabular_dataset.py first.")
        return

    labels = ["All", "Hot", "Warm", "Cold"]
    cities = ["All"] + sorted(df["city"].dropna().astype(str).unique().tolist()) if "city" in df.columns else ["All"]
    property_types = ["All"] + sorted(df["property_type"].dropna().astype(str).unique().tolist()) if "property_type" in df.columns else ["All"]
    professions = ["All"] + sorted(df["profession"].dropna().astype(str).unique().tolist()) if "profession" in df.columns else ["All"]

    c1, c2, c3, c4 = st.columns(4)
    selected_label = c1.selectbox("Class filter", labels, index=0)
    selected_city = c2.selectbox("City filter", cities, index=0)
    selected_property = c3.selectbox("Property type filter", property_types, index=0)
    selected_profession = c4.selectbox("Profession filter", professions, index=0)

    filtered = df.copy()
    if selected_label != "All" and "label" in filtered.columns:
        filtered = filtered[filtered["label"].astype(str).str.lower() == selected_label.lower()]
    if selected_city != "All" and "city" in filtered.columns:
        filtered = filtered[filtered["city"].astype(str) == selected_city]
    if selected_property != "All" and "property_type" in filtered.columns:
        filtered = filtered[filtered["property_type"].astype(str) == selected_property]
    if selected_profession != "All" and "profession" in filtered.columns:
        filtered = filtered[filtered["profession"].astype(str) == selected_profession]

    numeric_targets = [
        "age",
        "income",
        "budget",
        "num_visits",
        "total_time_spent_website",
        "page_views_per_visit",
        "ratio_budget_income",
        "interaction_score",
    ]

    cols = st.columns(2)
    with cols[0]:
        for col in numeric_targets[:4]:
            if col in filtered.columns:
                st.altair_chart(_altair_hist(filtered, col, color="label", title=f"{col.replace('_', ' ').title()} Distribution"), use_container_width=True)
    with cols[1]:
        for col in numeric_targets[4:]:
            if col in filtered.columns:
                st.altair_chart(_altair_hist(filtered, col, color="label", title=f"{col.replace('_', ' ').title()} Distribution"), use_container_width=True)

    st.subheader("Class Aggregates")
    agg_cols = [c for c in ["income", "budget", "interaction_score", "num_visits"] if c in filtered.columns]
    if agg_cols and "label" in filtered.columns:
        summary = filtered.groupby("label")[agg_cols].mean(numeric_only=True).reset_index()
        summary["label_display"] = summary["label"].map(normalize_label_display)
        long = summary.melt(id_vars=["label", "label_display"], var_name="metric", value_name="value")
        st.altair_chart(_altair_grouped_bar(long, "metric", "value", "label_display", title="Average Feature Values by Class"), use_container_width=True)
        st.dataframe(summary, use_container_width=True)


def page_multimodal_features(cfg: dict[str, Any]) -> None:
    st.header("Multimodal Features")
    include_ravdess = st.checkbox("Include RAVDESS in analysis", value=False)

    file_specs = [
        ("hot", cfg["paths"]["features_root"] / "hot.csv"),
        ("warm", cfg["paths"]["features_root"] / "warm.csv"),
        ("cold", cfg["paths"]["features_root"] / "cold.csv"),
        ("ravdess", cfg["paths"]["features_root"] / "ravdess.csv"),
    ]
    summaries = {name: _multimodal_file_summary(path, name) for name, path in file_specs}

    st.subheader("Feature File Status")
    status_rows = []
    for name, summary in summaries.items():
        status_rows.append(
            {
                "file": f"{name}.csv",
                "status": "Available" if summary["available"] else "Missing",
                "rows": summary["rows"],
                "columns": summary["columns"],
                "missing_values": summary["missing_values"],
                "label_distribution": json.dumps(summary["label_distribution"], ensure_ascii=False),
                "schema_valid": "Valid" if summary["schema_valid"] else "Invalid",
                "exact_schema": "Exact" if summary["exact_schema_match"] else "Non-exact",
            }
        )
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    warm_summary = summaries["warm"]
    if warm_summary["available"]:
        st.subheader("Warm Feature Preview")
        st.metric("Warm Rows", f"{warm_summary['rows']:,}")
        st.metric("Warm Columns", f"{warm_summary['columns']:,}")
        st.caption("Updated warm features are loaded from features/warm.csv.")
        st.dataframe(warm_summary["df"].head(25), use_container_width=True)

    prospect_frames = [
        summaries["hot"]["df"] if summaries["hot"]["available"] else pd.DataFrame(),
        summaries["warm"]["df"] if summaries["warm"]["available"] else pd.DataFrame(),
        summaries["cold"]["df"] if summaries["cold"]["available"] else pd.DataFrame(),
    ]
    if include_ravdess and summaries["ravdess"]["available"]:
        prospect_frames.append(summaries["ravdess"]["df"])
    prospect_frames = [frame for frame in prospect_frames if not frame.empty]

    if prospect_frames:
        df = pd.concat(prospect_frames, ignore_index=True, sort=False)
        st.subheader("Prospect Multimodal Analysis")
        st.metric("Processed Rows", f"{len(df):,}")

        if "label" in df.columns:
            label_counts = df["label"].astype(str).str.lower().value_counts().reindex(["hot", "warm", "cold"], fill_value=0).reset_index()
            label_counts.columns = ["label", "count"]
            label_counts["label_display"] = label_counts["label"].map(normalize_label_display)
            st.altair_chart(
                _altair_bar(label_counts, "label_display", "count", color="label", title="Processed Rows per Class", sort=["Hot", "Warm", "Cold"]),
                use_container_width=True,
            )

        st.subheader("Feature Preview")
        preview_cols = [c for c in ["video_path", "label", "transcript", "dominant_emotion", "confidence_emotion", "vision_score", "audio_score", "semantic_score", "final_score"] if c in df.columns]
        st.dataframe(df[preview_cols].head(25), use_container_width=True)

        if "dominant_emotion" in df.columns:
            emotion_counts = df["dominant_emotion"].astype(str).value_counts().reset_index()
            emotion_counts.columns = ["emotion", "count"]
            st.altair_chart(_altair_bar(emotion_counts, "emotion", "count", title="Dominant Emotion Distribution"), use_container_width=True)

        for col in ["vision_score", "audio_score", "semantic_score", "pitch_mean", "energy_mean"]:
            if col in df.columns and "label" in df.columns:
                st.altair_chart(_altair_box(df, "label", col, title=f"{col.replace('_', ' ').title()} by Class"), use_container_width=True)
    else:
        st.warning("No prospect multimodal feature files found. Import warm.csv or run the video pipeline first.")

    ravdess_summary = summaries["ravdess"]
    st.subheader("RAVDESS Auxiliary Dataset")
    st.caption(
        "RAVDESS is an external emotion dataset. It is displayed for analysis and is not used for prospect model training unless explicitly enabled."
    )
    if ravdess_summary["available"]:
        ravdess_df = ravdess_summary["df"].copy()
        st.metric("Total Rows", f"{ravdess_summary['rows']:,}")
        st.metric("Column Count", f"{ravdess_summary['columns']:,}")
        st.metric("Schema Status", "Valid" if ravdess_summary["schema_valid"] else "Invalid")
        st.metric("Label Distribution Classes", f"{len(ravdess_summary['label_distribution']):,}")

        st.subheader("RAVDESS Preview")
        ravdess_preview_cols = [c for c in ["video_path", "label", "dominant_emotion", "confidence_emotion", "audio_score", "semantic_score", "pitch_mean", "energy_mean", "face_detected", "landmark_count"] if c in ravdess_df.columns]
        st.dataframe(ravdess_df[ravdess_preview_cols].head(25), use_container_width=True)

        if "label" in ravdess_df.columns:
            label_counts = ravdess_df["label"].astype(str).str.lower().value_counts().reset_index()
            label_counts.columns = ["label", "count"]
            label_counts["label_display"] = label_counts["label"].map(normalize_label_display)
            st.altair_chart(
                _altair_bar(label_counts, "label_display", "count", color="label", title="RAVDESS Label Distribution", sort=["Hot", "Warm", "Cold"]),
                use_container_width=True,
            )

        if "dominant_emotion" in ravdess_df.columns:
            emotion_counts = ravdess_df["dominant_emotion"].astype(str).value_counts().reset_index()
            emotion_counts.columns = ["emotion", "count"]
            st.altair_chart(_altair_bar(emotion_counts, "emotion", "count", title="RAVDESS Dominant Emotion Distribution"), use_container_width=True)

        for col in ["audio_score", "semantic_score", "pitch_mean", "energy_mean"]:
            if col in ravdess_df.columns:
                st.altair_chart(_altair_box(ravdess_df, "label", col, title=f"RAVDESS {col.replace('_', ' ').title()} by Label"), use_container_width=True)
    else:
        st.warning("RAVDESS auxiliary CSV is missing.")


def page_model_performance(cfg: dict[str, Any]) -> None:
    st.header("Model Performance")
    metrics = load_json_safe(cfg["paths"]["metrics_root"] / "metrics.json")
    report = (cfg["paths"]["metrics_root"] / "classification_report.txt")
    confusion = cfg["paths"]["metrics_root"] / "confusion_matrix.png"
    leakage = cfg["paths"]["metrics_root"] / "leakage_report.txt"
    tabular_report = load_json_safe(cfg["paths"]["metrics_root"] / "tabular_dataset_report.json")

    if not metrics:
        st.warning("No metrics found. Train the model first.")
    if not (cfg["paths"]["models_root"] / "xgboost_model.pkl").exists():
        st.warning("No trained model found. Run scripts/run_full_pipeline.py first.")

    model_type = metrics.get("model_type") if metrics else None
    if not model_type and metrics:
        model_type = metrics.get("model_mode")
    if not model_type and (cfg["paths"]["models_root"] / "feature_schema.json").exists():
        schema = load_json_safe(cfg["paths"]["models_root"] / "feature_schema.json")
        model_type = schema.get("model_type") or schema.get("model_mode")
    model_type = model_type or "tabular_only"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Type", model_type)
    c2.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.4f}" if metrics else "N/A")
    c3.metric("Macro F1", f"{metrics.get('macro_f1', 0.0):.4f}" if metrics else "N/A")
    c4.metric("Weighted F1", f"{metrics.get('f1_weighted', 0.0):.4f}" if metrics else "N/A")

    c5, c6, c7 = st.columns(3)
    c5.metric("Macro Precision", f"{metrics.get('macro_precision', 0.0):.4f}" if metrics else "N/A")
    c6.metric("Macro Recall", f"{metrics.get('macro_recall', 0.0):.4f}" if metrics else "N/A")
    c7.metric("Training Samples", f"{metrics.get('train_rows', 0):,}" if metrics else "N/A")

    c8, c9, c10 = st.columns(3)
    c8.metric("Test Samples", f"{metrics.get('test_rows', 0):,}" if metrics else "N/A")
    c9.metric("Feature Count", f"{metrics.get('feature_count_before_encoding', 0):,}" if metrics else "N/A")
    c10.metric("Training Date", metrics.get("training_date", datetime.fromtimestamp((cfg["paths"]["metrics_root"] / "metrics.json").stat().st_mtime).isoformat()) if (cfg["paths"]["metrics_root"] / "metrics.json").exists() else "N/A")

    if report.exists():
        st.subheader("Classification Report")
        st.text(report.read_text(encoding="utf-8"))
    else:
        st.warning("No classification report found. Train the model first.")

    if confusion.exists():
        st.subheader("Confusion Matrix")
        st.image(str(confusion), use_container_width=True)
    else:
        st.warning("No confusion matrix found. Train the model first.")

    if leakage.exists():
        st.subheader("Leakage Report")
        leakage_text = leakage.read_text(encoding="utf-8")
        if "WARNING: score_lead" in leakage_text or "score_lead appears" in leakage_text:
            st.error("Leakage warning detected. score_lead may be target leakage and is excluded by default.")
        st.text(leakage_text)
    else:
        st.warning("No leakage report found. Run scripts/check_dataset_leakage.py first.")

    if tabular_report:
        st.subheader("Tabular Dataset Report")
        st.json(tabular_report)


def _load_prediction_defaults(cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    prospects = load_csv_safe(cfg["paths"]["processed_root"] / "prospects.csv")
    if prospects.empty:
        return {}, prospects
    return {
        "profession": sorted(prospects["profession"].dropna().astype(str).unique().tolist()) if "profession" in prospects.columns else ["commercial", "engineer", "teacher"],
        "property_type": sorted(prospects["property_type"].dropna().astype(str).unique().tolist()) if "property_type" in prospects.columns else ["apartment", "house", "villa"],
        "source_lead": sorted(prospects["source_lead"].dropna().astype(str).unique().tolist()) if "source_lead" in prospects.columns else ["website", "call", "whatsapp"],
        "city": sorted(prospects["city"].dropna().astype(str).unique().tolist()) if "city" in prospects.columns else ["Casablanca", "Rabat", "Marrakech"],
        "lead_origin": sorted(prospects["lead_origin"].dropna().astype(str).unique().tolist()) if "lead_origin" in prospects.columns else ["website", "ads", "social"],
        "last_activity": sorted(prospects["last_activity"].dropna().astype(str).unique().tolist()) if "last_activity" in prospects.columns else ["viewed_property", "called", "messaged"],
        "last_notable_activity": sorted(prospects["last_notable_activity"].dropna().astype(str).unique().tolist()) if "last_notable_activity" in prospects.columns else ["site_visit", "meeting", "chat"],
    }, prospects


def _prediction_row_from_inputs(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "prospect_id": "manual_tabular",
        "age": values["age"],
        "income": values["income"],
        "profession": values["profession"],
        "property_type": values["property_type"],
        "budget": values["budget"],
        "num_visits": values["num_visits"],
        "source_lead": values["source_lead"],
        "total_time_spent_website": values["total_time_spent_website"],
        "page_views_per_visit": values["page_views_per_visit"],
        "city": values["city"],
        "lead_origin": values["lead_origin"],
        "last_activity": values["last_activity"],
        "last_notable_activity": values["last_notable_activity"],
        "ratio_budget_income": values["ratio_budget_income"],
        "interaction_score": values["interaction_score"],
        "label": "",
    }


def page_prediction(cfg: dict[str, Any], model, bundle: dict | None) -> None:
    st.header("Prediction")
    if model is None or bundle is None:
        with st.spinner("Loading trained model..."):
            _, model, bundle, _ = load_model_bundle()
    if model is None or bundle is None:
        st.warning("No trained model found. Train the model first.")
        return

    schema = bundle.get("feature_schema_details", load_json_safe(cfg["paths"]["models_root"] / "feature_schema.json"))
    options, prospects = _load_prediction_defaults(cfg)
    fallback_text = {
        "profession": ["commercial", "engineer", "teacher", "doctor", "self_employed"],
        "property_type": ["apartment", "house", "villa", "plot", "studio"],
        "source_lead": ["website", "call", "whatsapp", "facebook", "referral"],
        "city": ["Casablanca", "Rabat", "Marrakech", "Tanger", "Fes"],
        "lead_origin": ["website", "ads", "social", "organic", "referral"],
        "last_activity": ["viewed_property", "called", "messaged", "follow_up", "no_response"],
        "last_notable_activity": ["site_visit", "meeting", "chat", "callback", "email_open"],
    }
    for key, fallback in fallback_text.items():
        options.setdefault(key, fallback)

    with st.form("tabular_prediction_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("age", min_value=0, max_value=120, value=42, step=1)
        income = c2.number_input("income", min_value=0.0, value=35000.0, step=1000.0)
        budget = c3.number_input("budget", min_value=0.0, value=250000.0, step=1000.0)

        c4, c5, c6 = st.columns(3)
        num_visits = c4.number_input("num_visits", min_value=0, value=2, step=1)
        total_time_spent_website = c5.number_input("total_time_spent_website", min_value=0.0, value=180.0, step=10.0)
        page_views_per_visit = c6.number_input("page_views_per_visit", min_value=0.0, value=4.5, step=0.1)

        c7, c8, c9 = st.columns(3)
        ratio_budget_income = c7.number_input("ratio_budget_income", min_value=0.0, value=6.0, step=0.1)
        interaction_score = c8.number_input("interaction_score", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
        profession = c9.selectbox("profession", options["profession"])

        c10, c11, c12 = st.columns(3)
        property_type = c10.selectbox("property_type", options["property_type"])
        source_lead = c11.selectbox("source_lead", options["source_lead"])
        city = c12.selectbox("city", options["city"])

        c13, c14, c15 = st.columns(3)
        lead_origin = c13.selectbox("lead_origin", options["lead_origin"])
        last_activity = c14.selectbox("last_activity", options["last_activity"])
        last_notable_activity = c15.selectbox("last_notable_activity", options["last_notable_activity"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = _prediction_row_from_inputs(
            {
                "age": age,
                "income": income,
                "profession": profession,
                "property_type": property_type,
                "budget": budget,
                "num_visits": num_visits,
                "source_lead": source_lead,
                "total_time_spent_website": total_time_spent_website,
                "page_views_per_visit": page_views_per_visit,
                "city": city,
                "lead_origin": lead_origin,
                "last_activity": last_activity,
                "last_notable_activity": last_notable_activity,
                "ratio_budget_income": ratio_budget_income,
                "interaction_score": interaction_score,
            }
        )
        prediction = predict_from_raw_row(model, bundle, row)
        probs = prediction["probabilities"]
        pred_label = prediction["predicted_label"]
        messages = {
            "hot": "High priority. Contact immediately.",
            "warm": "Medium priority. Follow up with personalized offer.",
            "cold": "Low priority. Keep in nurture campaign.",
        }
        cols = st.columns(3)
        cols[0].metric("Predicted Class", normalize_label_display(pred_label))
        cols[1].metric("Hot Probability", f"{probs.get('hot', 0.0):.3f}")
        cols[2].metric("Warm Probability", f"{probs.get('warm', 0.0):.3f}")
        if float(prediction.get("max_probability", 0.0)) < 0.45 or float(prediction.get("probability_gap", 0.0)) < 0.12:
            st.warning(
                f"Low-confidence prediction ({float(prediction.get('max_probability', 0.0)):.1%} top probability, "
                f"{float(prediction.get('probability_gap', 0.0)):.1%} gap)."
            )
        st.info(messages.get(pred_label, "No recommendation available."))
        prob_df = pd.DataFrame({"class": [normalize_label_display(k) for k in ["hot", "warm", "cold"]], "probability": [probs.get(k, 0.0) for k in ["hot", "warm", "cold"]]})
        st.altair_chart(_altair_probabilities(prob_df, title="Prediction Probabilities"), use_container_width=True)


def page_video_upload(cfg: dict[str, Any], model, bundle: dict | None) -> None:
    st.markdown(
        "<h1 class='gradient-text'>Video Upload Prediction</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#9CA3AF; font-size:1.1rem; margin-top:-0.5rem;'>Upload customer behavior videos for automated hybrid prospect scoring (Hot/Warm/Cold).</p>",
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"], key="video_upload_prediction", label_visibility="collapsed")
    
    if uploaded is None:
        st.markdown(
            """
            <div class="dashboard-card" style="text-align: center; padding: 3rem 1.5rem; border: 2px dashed rgba(255, 255, 255, 0.1);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
                <h3 style="margin: 0; color: #F3F4F6;">Select or drop a video file above</h3>
                <p style="color: #9CA3AF; margin-top: 0.5rem; margin-bottom: 0;">Supported formats: MP4, MOV, AVI, MKV, WEBM</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        temp_path = Path(tmp.name)

    # Two column layout: Left for Video, Right for Prediction Output
    col1, col2 = st.columns([1.1, 1.0], gap="large")

    with col1:
        st.markdown("<div class='card-title'>📹 Uploaded Frame Preview</div>", unsafe_allow_html=True)
        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
        st.video(str(temp_path))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display basic metadata
        st.markdown(
            f"""
            <div class="dashboard-card" style="padding: 1rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #9CA3AF;">File Name</span>
                    <span style="color: #FFFFFF; font-weight: 500;">{uploaded.name}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span style="color: #9CA3AF;">File Size</span>
                    <span style="color: #FFFFFF; font-weight: 500;">{uploaded.size / (1024*1024):.2f} MB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        legacy_model = None
        legacy_bundle = None
        
        with st.spinner("Running multimodal inference..."):
            result = process_uploaded_video(temp_path, legacy_model, legacy_bundle, cfg)

        prediction = result["prediction"]
        online_error = prediction.get("online_error")
        final_source = prediction.get("prediction_source", "online_video_model")
        transcript_text = (result["transcript"].get("transcript", "") or "").strip()
        sentiment_label = result["semantic"].get("sentiment_label", "neutral") if transcript_text else "unavailable"
        
        st.subheader("Final Predicted Class")
        st.markdown(f"**{normalize_label_display(prediction['predicted_label'])}**")
        st.caption(
            f"Prediction source: {final_source} • "
            f"Backend: {prediction.get('scoring_backend', 'unknown')} • "
            f"Frames analyzed: {int(prediction.get('num_frames', 0))}"
        )

        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Dominant Emotion", result["visual"].get("dominant_emotion", "unknown"))
        vc2.metric("Vision Score", f"{float(result['visual'].get('vision_score', 0.0)):.2f}")
        vc3.metric("Audio Score", f"{float(result['audio'].get('audio_score', 0.0)):.2f}")
        vc4.metric("Online Top Prob.", f"{float(prediction.get('max_probability', 0.0)):.2f}")
        
        st.write("Transcript")
        st.code(result["transcript"].get("transcript", "") or "No transcript available", language="text")
        st.metric("Semantic Score", f"{float(result['semantic'].get('semantic_score', 0.0)):.2f}")
        
        if float(prediction.get("max_probability", 0.0)) < 0.45 or float(prediction.get("probability_gap", 0.0)) < 0.12:
            st.warning(
                f"Low-confidence video prediction ({float(prediction.get('max_probability', 0.0)):.1%} top probability, "
                f"{float(prediction.get('probability_gap', 0.0)):.1%} gap)."
            )
            
        with st.expander("Prediction diagnostics", expanded=False):
            st.json(
                {
                    "sentiment_label": sentiment_label,
                    "final_label": prediction.get("predicted_label"),
                    "prediction_source": prediction.get("prediction_source", "online_video_model"),
                    "scoring_backend": prediction.get("scoring_backend", "unknown"),
                    "online_model_name": prediction.get("model_name"),
                    "online_error": online_error,
                    "online_probabilities": prediction.get("probabilities", {}),
                    "legacy_model_label": prediction.get("legacy_predicted_label", "unavailable"),
                    "legacy_model_probabilities": prediction.get("legacy_probabilities", {}),
                    "legacy_model_max_probability": prediction.get("legacy_max_probability", 0.0),
                    "legacy_model_gap": prediction.get("legacy_probability_gap", 0.0),
                }
            )
            
        prob_df = pd.DataFrame(
            {
                "class": [normalize_label_display(k) for k in ["hot", "warm", "cold"]],
                "probability": [prediction["probabilities"].get(k, 0.0) for k in ["hot", "warm", "cold"]],
            }
        )
        st.subheader("Class Probabilities")
        st.altair_chart(_altair_probabilities(prob_df, title="Class Probabilities"), use_container_width=True)


class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self, model, bundle, frame_stride: int = 5):
        self.model = model
        self.bundle = bundle
        self.frame_stride = max(1, frame_stride)
        self.frame_count = 0
        self.last_state = {
            "predicted_label": "cold",
            "probabilities": {"cold": 1.0, "warm": 0.0, "hot": 0.0},
            "visual": {"dominant_emotion": "unknown", "emotion_confidence_mean": 0.0, "face_detected_ratio": 0.0, "landmark_quality_score": 0.0, "vision_score": 0.0},
            "note": "Live mode is currently using visual features only. Audio and NLP require short audio recording windows.",
            "face_detected": False,
        }
        from prop_score_ai.video_pipeline import VisualSignalAnalyzer

        self.analyzer = VisualSignalAnalyzer()

    def _visual_only_state(self, visual: dict[str, Any]) -> dict[str, Any]:
        face_detected = bool(float(visual.get("face_detected_ratio", 0.0)) > 0.0)
        dominant_emotion = str(visual.get("dominant_emotion", "unknown")).lower()
        confidence = float(visual.get("emotion_confidence_mean", 0.0) or 0.0)
        vision_score = float(visual.get("vision_score", 0.0) or 0.0)

        if not face_detected:
            predicted_label = "cold"
            probabilities = {"hot": 0.10, "warm": 0.20, "cold": 0.70}
        elif dominant_emotion in {"happy", "surprise"} and confidence >= 0.65 and vision_score >= 0.35:
            predicted_label = "hot"
            probabilities = {"hot": 0.58, "warm": 0.30, "cold": 0.12}
        elif dominant_emotion in {"sad", "angry", "disgust", "fear"} and confidence >= 0.35:
            predicted_label = "cold"
            probabilities = {"hot": 0.10, "warm": 0.25, "cold": 0.65}
        elif face_detected and vision_score >= 0.22:
            predicted_label = "warm"
            probabilities = {"hot": 0.25, "warm": 0.50, "cold": 0.25}
        else:
            predicted_label = "warm"
            probabilities = {"hot": 0.25, "warm": 0.50, "cold": 0.25}

        sorted_probs = sorted(probabilities.values(), reverse=True)
        return {
            "predicted_label": predicted_label,
            "probabilities": probabilities,
            "visual": visual,
            "note": "Live mode is currently using visual features only. Audio and NLP require short audio recording windows.",
            "face_detected": face_detected,
            "max_probability": float(sorted_probs[0]),
            "probability_gap": float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0]),
        }

    def recv(self, frame):
        from prop_score_ai.video_pipeline import get_live_visual_features

        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        if self.frame_count % self.frame_stride == 0:
            try:
                visual = get_live_visual_features(image, analyzer=self.analyzer)
                if self.model is not None and self.bundle is not None:
                    try:
                        pred = predict_visual_frame(self.model, self.bundle, image, visual=visual)
                        self.last_state = {
                            "predicted_label": pred["predicted_label"],
                            "probabilities": pred["probabilities"],
                            "visual": visual,
                            "note": "Live mode is currently using visual features only. Audio and NLP require short audio recording windows.",
                            "face_detected": bool(visual.get("face_detected_ratio", 0.0) > 0.0),
                            "max_probability": pred.get("max_probability", 0.0),
                            "probability_gap": pred.get("probability_gap", 0.0),
                        }
                    except Exception:
                        self.last_state = self._visual_only_state(visual)
                else:
                    self.last_state = self._visual_only_state(visual)
            except Exception:
                pass

        label = self.last_state["predicted_label"].upper()
        probs = self.last_state["probabilities"]
        color = {"HOT": (40, 180, 70), "WARM": (0, 180, 255), "COLD": (50, 80, 220)}.get(label, (255, 255, 255))
        overlay = image.copy()
        cv2.rectangle(overlay, (12, 12), (560, 132), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
        cv2.putText(image, f"STATE: {label}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(image, f"Emotion {self.last_state['visual'].get('dominant_emotion', 'unknown')}", (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(image, f"Confidence {float(self.last_state['visual'].get('emotion_confidence_mean', 0.0)):.2f}", (24, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(image, f"Face {self.last_state['face_detected']}", (24, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
        if av is not None:
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        return frame


def page_live_camera(cfg: dict[str, Any], model, bundle: dict | None) -> None:
    st.header("Live Camera")
    model = None
    bundle = None
    st.info("Live mode is currently using visual features only. Audio and NLP require short audio recording windows.")
    use_webrtc = av is not None and WebRtcMode is not None and RTCConfiguration is not None

    if use_webrtc:
        frame_stride = int(cfg["live_camera"]["frame_stride"])
        ctx = webrtc_streamer(
            key="propscore_live_camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: LiveVideoProcessor(model, bundle, frame_stride=frame_stride),
            async_processing=True,
        )

        if ctx and ctx.video_processor:
            state = ctx.video_processor.last_state
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current State", normalize_label_display(state["predicted_label"]))
            c2.metric("Face Status", "Detected" if state.get("face_detected") else "Not detected")
            c3.metric("Emotion", state["visual"].get("dominant_emotion", "unknown"))
            c4.metric("Emotion Confidence", f"{float(state['visual'].get('emotion_confidence_mean', 0.0)):.2f}")
            st.metric("Visual Score", f"{float(state['visual'].get('vision_score', 0.0)):.2f}")
            if float(state.get("max_probability", 0.0)) < 0.45 or float(state.get("probability_gap", 0.0)) < 0.12:
                st.warning(
                    f"Low-confidence live prediction ({float(state.get('max_probability', 0.0)):.1%} top probability, "
                    f"{float(state.get('probability_gap', 0.0)):.1%} gap)."
                )
            st.caption(state["note"])
            return

    st.warning("Live streaming is unavailable in this session. Use snapshot camera mode below.")
    snapshot = st.camera_input("Capture a live frame")

    if snapshot is None:
        st.info("Take a photo above to run the visual check.")
        return

    try:
        if hasattr(snapshot, "getvalue"):
            raw_bytes = snapshot.getvalue()
        elif hasattr(snapshot, "read"):
            raw_bytes = snapshot.read()
        elif isinstance(snapshot, (bytes, bytearray)):
            raw_bytes = bytes(snapshot)
        else:
            st.error(f"Unsupported camera snapshot type: {type(snapshot).__name__}")
            return

        image_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not read the captured camera frame.")
            return

        from prop_score_ai.video_pipeline import VisualSignalAnalyzer, get_live_visual_features

        analyzer = VisualSignalAnalyzer()
        visual = get_live_visual_features(image, analyzer=analyzer)
        processor = LiveVideoProcessor(None, None, frame_stride=1)
        state = processor._visual_only_state(visual)
        state["note"] = "Snapshot mode is using visual features only. Audio and NLP require short audio recording windows."

        st.success("Snapshot analyzed.")
        st.caption("Camera snapshots are analyzed immediately after capture.")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Captured camera frame", use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current State", normalize_label_display(state["predicted_label"]))
        c2.metric("Face Status", "Detected" if state.get("face_detected") else "Not detected")
        c3.metric("Emotion", state["visual"].get("dominant_emotion", "unknown"))
        c4.metric("Emotion Confidence", f"{float(state['visual'].get('emotion_confidence_mean', 0.0)):.2f}")
        st.metric("Visual Score", f"{float(state['visual'].get('vision_score', 0.0)):.2f}")
        if float(state.get("max_probability", 0.0)) < 0.45 or float(state.get("probability_gap", 0.0)) < 0.12:
            st.warning(
                f"Low-confidence snapshot prediction ({float(state.get('max_probability', 0.0)):.1%} top probability, "
                f"{float(state.get('probability_gap', 0.0)):.1%} gap)."
            )
        st.caption(state["note"])
    except Exception as exc:
        st.exception(exc)
        st.error("Snapshot analysis failed. The app stayed alive, but this frame could not be processed.")


def page_pipeline_status(cfg: dict[str, Any]) -> None:
    st.header("Pipeline Status")
    sections = {
        "Data": [
            cfg["paths"]["processed_root"] / "prospects.csv",
            cfg["paths"]["processed_root"] / "fused_dataset.csv",
            cfg["paths"]["processed_root"] / "multimodal_features.csv",
        ],
        "Features": [
            cfg["paths"]["features_root"] / "hot.csv",
            cfg["paths"]["features_root"] / "warm.csv",
            cfg["paths"]["features_root"] / "cold.csv",
            cfg["paths"]["features_root"] / "ravdess.csv",
        ],
        "Models": [
            cfg["paths"]["models_root"] / "xgboost_model.pkl",
            cfg["paths"]["models_root"] / "encoders.pkl",
            cfg["paths"]["models_root"] / "feature_schema.json",
            cfg["paths"]["models_root"] / "video_xgboost_model.pkl",
            cfg["paths"]["models_root"] / "video_encoders.pkl",
            cfg["paths"]["models_root"] / "video_feature_schema.json",
        ],
        "Metrics": [
            cfg["paths"]["metrics_root"] / "metrics.json",
            cfg["paths"]["metrics_root"] / "video_metrics.json",
            cfg["paths"]["metrics_root"] / "import_feature_csvs_report.json",
            cfg["paths"]["metrics_root"] / "classification_report.txt",
            cfg["paths"]["metrics_root"] / "video_classification_report.txt",
            cfg["paths"]["metrics_root"] / "confusion_matrix.png",
            cfg["paths"]["metrics_root"] / "video_confusion_matrix.png",
            cfg["paths"]["metrics_root"] / "leakage_report.txt",
        ],
    }

    for section, paths in sections.items():
        st.subheader(section)
        for path in paths:
            status = file_status(path)
            color = "#1f8a4c" if status == "Available" else "#b23b3b"
            st.markdown(f"**{path}**: <span style='color:{color};font-weight:700'>{status}</span>", unsafe_allow_html=True)

    st.subheader("Command Hints")
    st.code("python scripts/00_prepare_tabular_dataset.py --input dataset_final_catboost.csv", language="bash")
    st.code("python scripts/import_feature_csvs.py --warm warm.csv --ravdess ravdess.csv", language="bash")
    st.code("python scripts/check_dataset_leakage.py --input data/processed/prospects.csv", language="bash")
    st.code("python scripts/run_full_pipeline.py --tabular-csv dataset_final_catboost.csv", language="bash")
    st.code("python scripts/run_full_pipeline.py --tabular-csv dataset_final_catboost.csv --video-root hayat-ai-videos", language="bash")


def main() -> None:
    inject_custom_css()
    cfg = load_config()
    cfg["paths"] = {key: Path(value) if not isinstance(value, Path) else value for key, value in cfg.get("paths", {}).items()}
    ensure_runtime_dirs(cfg)
    model = None
    bundle = None
    metrics_root = Path(cfg["paths"]["metrics_root"])
    metrics = load_json_safe(metrics_root / "metrics.json")
    pages = [
        "Video Upload Prediction",
        "Dataset Dashboard",
        "Tabular Analytics",
        "Multimodal Features",
        "Live Camera",
        "Pipeline Status",
    ]

    with st.sidebar:
        st.title("PropScore_AI")
        selected_page = st.radio("Navigation", pages, index=0)
        st.caption(f"Model mode: {model_mode(bundle, metrics)}")
        st.caption("Hybrid Real Estate Prospect Qualification System")

    if selected_page == "Video Upload Prediction":
        page_video_upload(cfg, model, bundle)
    elif selected_page == "Dataset Dashboard":
        page_dataset_dashboard(cfg)
    elif selected_page == "Tabular Analytics":
        page_tabular_analytics(cfg)
    elif selected_page == "Multimodal Features":
        page_multimodal_features(cfg)
    elif selected_page == "Live Camera":
        page_live_camera(cfg, model, bundle)
    elif selected_page == "Pipeline Status":
        page_pipeline_status(cfg)


if __name__ == "__main__":
    main()
