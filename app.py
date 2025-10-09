"""Gradio interface for the pruefomat Veri pipeline builder."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, fbeta_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config_loader import AppConfig, ConfigError, load_config, normalize_config_columns
from build_pipeline_veri import (
    NULL_LIKE_DEFAULT,
    DataFramePreparer,
    DaysUntilDueAdder,
    FeaturePlan,
    build_preprocessor,
    infer_feature_plan,
    normalize_column_name,
    normalize_columns,
)
from logging_utils import configure_logging, get_logger, mask_sensitive_data


configure_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration setup
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")


def _load_app_config(path: Optional[str | Path]) -> AppConfig:
    return normalize_config_columns(load_config(path))


try:
    DEFAULT_CONFIG = _load_app_config(DEFAULT_CONFIG_PATH)
except ConfigError:
    DEFAULT_CONFIG = AppConfig()
    DEFAULT_CONFIG = normalize_config_columns(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Gradio schema monkey-patch (4.44.0 bug fix)
# ---------------------------------------------------------------------------
try:
    from gradio_client import utils as grc_utils

    _ORIG_JSON_SCHEMA_TO_PYTHON_TYPE = grc_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):  # type: ignore[override]
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        return _ORIG_JSON_SCHEMA_TO_PYTHON_TYPE(schema, defs)

    grc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
except (ImportError, AttributeError):
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sheet(sheet_text: str) -> int | str:
    text = (sheet_text or "").strip()
    return int(text) if text.isdigit() else text or 0


def _ensure_config(state: Dict[str, Any]) -> AppConfig:
    config = state.get("config")
    if config is None:
        config = DEFAULT_CONFIG.model_copy(deep=True)
        state["config"] = config
        state["config_path"] = str(DEFAULT_CONFIG_PATH)
    return config


def _format_schema(df: pd.DataFrame) -> Dict[str, Any]:
    details = []
    total_rows = len(df)
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        details.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null": non_null,
                "null_pct": float(100 * (1 - non_null / total_rows)) if total_rows else 0.0,
            }
        )
    return {"rows": total_rows, "columns": len(df.columns), "fields": details}


def _reset_pipeline_state(state: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("preprocessor", "feature_plan", "prep_path", "baseline_model", "baseline_path", "metrics_path"):
        state.pop(key, None)
    return state


def load_dataset(upload, config_upload, sheet_text: str, target_text: str, state: Optional[Dict[str, Any]]):
    state = state or {}
    if upload is None:
        logger.warning("ui_no_file")
        return "Bitte zuerst eine Excel-Datei hochladen.", None, None, state

    config_path: Optional[str]
    if config_upload is not None:
        config_path = config_upload.name
    else:
        config_path = state.get("config_path", str(DEFAULT_CONFIG_PATH))

    try:
        config = _load_app_config(config_path)
    except ConfigError as exc:
        logger.error("ui_config_error", message=str(exc), config_path=config_path)
        return f"Konfigurationsfehler: {exc}", None, None, state

    if target_text:
        config.data.target_col = normalize_column_name(target_text)

    logger.info(
        "ui_config_loaded",
        config_path=config_path,
        target_col=config.data.target_col,
    )

    state["config"] = config
    state["config_path"] = config_path

    sheet = _parse_sheet(sheet_text)
    try:
        df_raw = pd.read_excel(upload.name, sheet_name=sheet, dtype="object")
    except Exception as exc:  # pragma: no cover
        logger.error(
            "ui_excel_error",
            message=str(exc),
            file=upload.name,
            sheet=sheet,
        )
        return f"Fehler beim Laden: {exc}", None, None, state

    df_norm, column_mapping = normalize_columns(df_raw)
    logger.info(
        "ui_dataset_loaded",
        file=upload.name,
        sheet=sheet,
        rows=len(df_norm),
        columns=df_norm.shape[1],
    )

    target_norm = config.data.target_col
    target_series = None
    df_features = df_norm
    target_msg = "Keine Zielspalte gesetzt."
    if target_norm and target_norm in df_norm.columns:
        target_series = df_norm[target_norm]
        df_features = df_norm.drop(columns=[target_norm])
        target_msg = f"Zielspalte erkannt: {target_norm}"
        logger.info("ui_target_detected", target=target_norm)
    elif target_norm:
        target_msg = f"Warnung: Zielspalte '{target_norm}' nicht gefunden."
        logger.warning("ui_target_missing", target=target_norm)

    state.update(
        {
            "excel_path": upload.name,
            "sheet": sheet,
            "column_mapping": column_mapping,
            "df_features": df_features,
            "target": target_series,
            "target_name": target_norm,
        }
    )
    state = _reset_pipeline_state(state)

    preview = df_norm.head(8)
    schema = _format_schema(df_norm)
    schema["column_mapping"] = mask_sensitive_data(column_mapping)
    schema["config"] = {
        "config_path": config_path,
        "target_col": target_norm,
    }

    status = (
        f"Geladen: {Path(upload.name).name} (Sheet {sheet}) | {len(df_norm)} Zeilen, {df_norm.shape[1]} Spalten. {target_msg}"
    )
    return status, preview, schema, state


def build_pipeline_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    if df_features is None or df_features.empty:
        logger.warning("ui_build_without_data")
        return "Keine Daten geladen.", None, None, state

    config = _ensure_config(state)
    logger.info("ui_build_pipeline_start", rows=len(df_features))

    preparer_preview = DataFramePreparer(
        amount_col=config.data.amount_col or "",
        issue_col=config.data.issue_col or "",
        due_col=config.data.due_col or "",
        date_columns=config.data.additional_date_columns,
        null_like=list(set(NULL_LIKE_DEFAULT).union(set(config.data.null_like))),
    )
    prepared = preparer_preview.fit_transform(df_features)
    with_due = DaysUntilDueAdder(
        issue_col=config.data.issue_col or "",
        due_col=config.data.due_col or "",
    ).fit_transform(prepared)

    feature_plan = infer_feature_plan(with_due, config)
    if not feature_plan.numeric and not feature_plan.categorical and not feature_plan.text:
        logger.warning("ui_feature_plan_empty")
        return "Keine nutzbaren Features gefunden.", None, None, state

    preprocessor = build_preprocessor(feature_plan, config)
    preprocessor.fit(df_features)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_prep_"))
    prep_path = tmp_dir / "prep_pipeline.joblib"
    joblib.dump(preprocessor, prep_path)

    state.update(
        {
            "preprocessor": preprocessor,
            "feature_plan": feature_plan,
            "prep_path": str(prep_path),
        }
    )

    plan_json = {
        "numeric": feature_plan.numeric,
        "categorical": feature_plan.categorical,
        "text": feature_plan.text,
        "dropped": getattr(preprocessor.named_steps["prep_df"], "columns_to_drop_", []),
        "config_path": state.get("config_path", str(DEFAULT_CONFIG_PATH)),
    }

    logger.info(
        "ui_build_pipeline_success",
        **mask_sensitive_data({
            "prep_path": str(prep_path),
            "numeric": feature_plan.numeric,
            "categorical": feature_plan.categorical,
            "text": feature_plan.text,
        }),
    )

    return "Preprocessor trainiert und gespeichert.", plan_json, str(prep_path), state


def preview_features_action(state: Optional[Dict[str, Any]], sample_size: int = 10):
    state = state or {}
    df_features = state.get("df_features")
    preprocessor = state.get("preprocessor")
    if df_features is None or preprocessor is None:
        logger.warning("ui_preview_without_pipeline")
        return "Bitte zuerst Pipeline bauen.", None

    sample = df_features.head(sample_size)
    matrix = preprocessor.transform(sample)
    encoder = preprocessor.named_steps["encode"]
    feature_names = list(encoder.get_feature_names_out())
    if not feature_names:
        return "Keine transformierten Merkmale vorhanden.", None

    keep = min(10, len(feature_names))
    if hasattr(matrix, "toarray"):
        arr = matrix[:, :keep].toarray()
    else:
        arr = np.asarray(matrix)[:, :keep]
    preview_df = pd.DataFrame(arr, columns=feature_names[:keep])
    logger.info("ui_preview_features", rows=len(sample), features=keep)
    return f"Vorschau fuer {len(sample)} Zeilen (erste {keep} Features).", preview_df


def train_baseline_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    target = state.get("target")
    feature_plan: FeaturePlan | None = state.get("feature_plan")

    if df_features is None or feature_plan is None:
        logger.warning("ui_baseline_without_pipeline")
        return "Bitte zuerst Pipeline bauen.", None, None, None, None, state
    if target is None:
        logger.warning("ui_baseline_without_target")
        return "Keine Zielspalte verfuegbar.", None, None, None, None, state

    config = _ensure_config(state)
    rf_kwargs = config.model.random_forest.model_dump(exclude_none=True)

    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_plan, config)),
            (
                "classifier",
                RandomForestClassifier(**rf_kwargs),
            ),
        ]
    )

    stratify = target if target.dropna().nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df_features,
        target,
        test_size=0.2,
        random_state=config.model.random_forest.random_state,
        stratify=stratify,
    )

    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    metrics = {
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "f2_score": float(fbeta_score(y_test, y_pred, average="macro", beta=2.0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    confusion = confusion_matrix(y_test, y_pred)
    confusion_df = pd.DataFrame(confusion)

    encoder = baseline.named_steps["preprocessor"].named_steps.get("encode")
    if encoder is not None and hasattr(encoder, "get_feature_names_out"):
        feature_names = encoder.get_feature_names_out()
    else:
        feature_names = np.array([f"f_{idx}" for idx in range(len(baseline.named_steps["classifier"].feature_importances_))])

    importance = baseline.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df.sort_values("importance", ascending=False, inplace=True)
    top20 = importance_df.head(20).reset_index(drop=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_model_"))
    model_path = tmp_dir / "baseline_model.joblib"
    metrics_path = tmp_dir / "metrics.json"
    importance_path = tmp_dir / "feature_importance.csv"
    plot_path = tmp_dir / "feature_importance.png"

    joblib.dump(baseline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance_df.to_csv(importance_path, index=False)

    plot_image = None
    if not top20.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(top20["feature"][::-1], top20["importance"][::-1], color="#2b8a3e")
        plt.xlabel("Importance")
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plot_image = str(plot_path)

    state.update(
        {
            "baseline_model": baseline,
            "baseline_path": str(model_path),
            "metrics_path": str(metrics_path),
            "importance_path": str(importance_path),
            "importance_plot": plot_image,
        }
    )

    metric_summary = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "importance_path": str(importance_path),
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f2_score": metrics["f2_score"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }

    logger.info(
        "ui_baseline_trained",
        **mask_sensitive_data(metric_summary),
    )

    return (
        "Baseline trainiert.",
        metrics,
        confusion_df,
        str(model_path),
        str(metrics_path),
        top20,
        plot_image,
        str(importance_path),
        state,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="pruefomat") as demo:
        gr.Markdown(
            """
            # pruefomat – Fraud/Veri-Selector
            1. Excel laden und Schema pruefen.
            2. Pipeline bauen (Preprocessing + Feature-Encoding).
            3. Optional Baseline trainieren und Ergebnisse inspizieren.
            """
        )

        state = gr.State({
            "config": DEFAULT_CONFIG.model_copy(deep=True),
            "config_path": str(DEFAULT_CONFIG_PATH),
        })

        with gr.Row():
            file_input = gr.File(label="Excel-Datei", file_types=[".xlsx", ".xls"])  # type: ignore[arg-type]
            config_input = gr.File(label="Config (optional)", file_types=[".yaml", ".yml", ".json"])  # type: ignore[arg-type]
            sheet_input = gr.Textbox(value="0", label="Sheet (Index oder Name)")
            target_input = gr.Textbox(value=DEFAULT_CONFIG.data.target_col or "", label="Zielspalte (optional)")
            load_btn = gr.Button("Daten laden")

        load_status = gr.Textbox(label="Status", interactive=False)
        data_preview = gr.Dataframe(label="Daten (erste Zeilen)", interactive=False)
        schema_json = gr.JSON(label="Schema / Mapping")

        gr.Markdown("## Pipeline")
        build_btn = gr.Button("Pipeline bauen")
        build_status = gr.Textbox(label="Pipeline-Status", interactive=False)
        plan_json = gr.JSON(label="Feature-Plan")
        prep_download = gr.File(label="Preprocessor Download", interactive=False)

        preview_btn = gr.Button("Features Vorschau")
        preview_status = gr.Textbox(label="Preview-Status", interactive=False)
        preview_table = gr.Dataframe(label="Transformierte Features", interactive=False)

        gr.Markdown("## Baseline Modell")
        baseline_btn = gr.Button("Baseline trainieren")
        baseline_status = gr.Textbox(label="Baseline-Status", interactive=False)
        metrics_json = gr.JSON(label="Metriken")
        confusion_df = gr.Dataframe(label="Confusion Matrix", interactive=False)
        model_download = gr.File(label="Baseline Modell", interactive=False)
        metrics_download = gr.File(label="Metrics JSON", interactive=False)
        importance_table = gr.Dataframe(label="Feature Importanzen (Top 20)", interactive=False)
        importance_plot = gr.Image(label="Feature Importance Chart", interactive=False)
        importance_download = gr.File(label="Feature Importances CSV", interactive=False)

        load_btn.click(
            load_dataset,
            inputs=[file_input, config_input, sheet_input, target_input, state],
            outputs=[load_status, data_preview, schema_json, state],
        )

        build_btn.click(
            build_pipeline_action,
            inputs=[state],
            outputs=[build_status, plan_json, prep_download, state],
        )

        preview_btn.click(
            preview_features_action,
            inputs=[state],
            outputs=[preview_status, preview_table],
        )

        baseline_btn.click(
            train_baseline_action,
            inputs=[state],
            outputs=[
                baseline_status,
                metrics_json,
                confusion_df,
                model_download,
                metrics_download,
                importance_table,
                importance_plot,
                importance_download,
                state,
            ],
        )

    return demo


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
