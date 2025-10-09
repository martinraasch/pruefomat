"""Gradio interface for the pruefomat Veri pipeline builder."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
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


os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_DO_NOT_TRACK", "True")
os.environ.setdefault("GRADIO_DISABLE_USAGE_STATS", "True")

configure_logging()
logger = get_logger(__name__)

FEEDBACK_DB_PATH = Path(json.loads(os.environ.get("PRUEFOMAT_SETTINGS", "{}")).get("feedback_db", "feedback.db"))


def ensure_feedback_db():
    FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                beleg_index INTEGER,
                beleg_id TEXT,
                timestamp TEXT,
                user TEXT,
                score REAL,
                prediction INTEGER,
                feedback TEXT,
                comment TEXT
            )
            """
        )
        conn.commit()

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
        target_numeric = pd.to_numeric(df_norm[target_norm], errors="coerce")
        mask = target_numeric.notna()
        if mask.any():
            target_series = target_numeric.loc[mask].astype(int)
            df_features = df_norm.loc[mask].drop(columns=[target_norm])
            target_msg = f"Zielspalte erkannt: {target_norm}"
            logger.info("ui_target_detected", target=target_norm)
        else:
            target_msg = f"Zielspalte '{target_norm}' enthält keine nutzbaren Werte."
            df_features = df_norm.drop(columns=[target_norm])
            logger.warning("ui_target_empty", target=target_norm)
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

    proba = baseline.predict_proba(X_test)
    if proba.ndim == 1:
        pos_scores = proba
    elif proba.shape[1] == 1:
        pos_scores = proba[:, 0]
    else:
        classes = list(baseline.named_steps["classifier"].classes_)
        positive_index = 1 if len(classes) > 1 else 0
        for candidate in ("1", 1, "fraud", "Fraud", True):
            if candidate in classes:
                positive_index = classes.index(candidate)
                break
        if len(classes) == 2:
            pos_scores = proba[:, positive_index]
        else:
            pos_scores = proba.max(axis=1)

    fraud_scores = pd.Series(pos_scores * 100.0, name="fraud_score")
    predictions_raw = pd.Series(baseline.predict(X_test))
    predictions_numeric = pd.to_numeric(predictions_raw, errors="coerce")
    if predictions_numeric.notna().all():
        predictions_col = predictions_numeric.round().astype(int)
    else:
        predictions_col = predictions_raw

    actual_series = pd.Series(y_test).reset_index(drop=True)
    predictions_col = predictions_col.reset_index(drop=True)
    fraud_scores = fraud_scores.reset_index(drop=True)

    predictions_df = pd.DataFrame(
        {
            "row_index": np.arange(len(fraud_scores)),
            "fraud_score": fraud_scores,
            "prediction": predictions_col,
            "actual": actual_series,
        }
    )
    predictions_df.sort_values("fraud_score", ascending=False, inplace=True)
    predictions_display = predictions_df.head(50).reset_index(drop=True)
    predictions_display.columns = ["Index", "Score", "Prediction", "Actual"]

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_model_"))
    model_path = tmp_dir / "baseline_model.joblib"
    metrics_path = tmp_dir / "metrics.json"
    importance_path = tmp_dir / "feature_importance.csv"
    plot_path = tmp_dir / "feature_importance.png"
    predictions_path = tmp_dir / "predictions.csv"

    joblib.dump(baseline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance_df.to_csv(importance_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

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
            "predictions_path": str(predictions_path),
            "shap_feature_names": list(feature_names),
            "feature_importances": importance_df.to_dict(orient="records"),
            "predictions_full": predictions_df,
        }
    )

    try:
        state["feature_importance_df"] = importance_df
        state["notes_text_vectorizer"] = baseline.named_steps["preprocessor"].named_steps["encode"].named_transformers_["text"].named_steps["tfidf"]
    except Exception as exc:
        logger.warning("ui_importance_store_failed", message=str(exc))

    try:
        background_size = min(200, len(df_features))
        background = baseline.named_steps["preprocessor"].transform(df_features.head(background_size))
        background_dense = background.toarray() if hasattr(background, "toarray") else background
        state["shap_background"] = background_dense
    except Exception as exc:
        logger.warning("ui_shap_background_failed", message=str(exc))

    metric_summary = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "importance_path": str(importance_path),
        "predictions_path": str(predictions_path),
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
        predictions_display,
        str(predictions_path),
        state,
    )


def explain_prediction_action(state: Optional[Dict[str, Any]], row_index):
    state = state or {}
    baseline = state.get("baseline_model")
    df_features = state.get("df_features")
    if baseline is None or df_features is None:
        return "Bitte zuerst Baseline trainieren.", None, None, state

    try:
        idx = int(row_index)
    except (TypeError, ValueError):
        return "Ungültiger Index.", None, None, state

    if idx < 0 or idx >= len(df_features):
        return f"Index muss zwischen 0 und {len(df_features)-1} liegen.", None, None, state

    preprocessor = baseline.named_steps.get("preprocessor")
    classifier = baseline.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        return "Pipeline unvollständig.", None, None, state

    if not hasattr(classifier, "estimators_"):
        return "Erklärungen unterstützen derzeit nur Baum-Modelle (z. B. RandomForest).", None, None, state

    feature_names = state.get("shap_feature_names")
    if feature_names is None:
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            transformed = preprocessor.transform(df_features.head(1))
            feature_names = [f"f_{i}" for i in range(transformed.shape[1])]
        state["shap_feature_names"] = feature_names

    sample = df_features.iloc[[idx]]
    transformed = preprocessor.transform(sample)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    background = state.get("shap_background")
    if background is None:
        bg = preprocessor.transform(df_features.head(min(200, len(df_features))))
        background = bg.toarray() if hasattr(bg, "toarray") else bg
        state["shap_background"] = background

    try:
        explainer = shap.TreeExplainer(classifier, background)
        shap_values = explainer.shap_values(transformed_dense)
        if isinstance(shap_values, list):
            classes = list(getattr(classifier, "classes_", [0, 1]))
            pos_idx = 1 if len(classes) > 1 else 0
            for candidate in ("1", 1, True, "fraud", "Fraud"):
                if candidate in classes:
                    pos_idx = classes.index(candidate)
                    break
            shap_row = shap_values[pos_idx][0]
        else:
            shap_row = shap_values[0]
    except Exception as exc:
        logger.error("ui_shap_failed", message=str(exc))
        return f"SHAP konnte nicht berechnet werden: {exc}", None, None, state

    feature_array = np.array(feature_names)
    values_array = transformed_dense[0]
    order = np.argsort(np.abs(shap_row))[::-1][:5]
    explanation = []
    for i in order:
        explanation.append(
            {
                "feature": feature_array[i],
                "shap_value": float(shap_row[i]),
                "feature_value": float(values_array[i]),
                "impact": "erhöht Risiko" if shap_row[i] >= 0 else "senkt Risiko",
            }
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_shap_"))
    json_path = tmp_dir / "explanation.json"
    json_path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")

    return f"Top-Features für Zeile {idx}", explanation, str(json_path), state


def generate_pattern_report_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    target = state.get("target")
    feature_importance = state.get("feature_importance_df")
    vectorizer = state.get("notes_text_vectorizer")

    if df_features is None or target is None or feature_importance is None:
        return "Bitte zuerst Baseline trainieren.", None, state

    report_lines = ["# Fraud Pattern Report", ""]
    top_features = feature_importance.head(10)
    report_lines.append("## Top Risiko-Features")
    for _, row in top_features.iterrows():
        report_lines.append(f"- **{row['feature']}**: Importance {row['importance']:.4f}")
    report_lines.append("")

    df_local = df_features.copy()
    y = target.astype(int)
    df_local["is_fraud"] = y

    report_lines.append("## Fraud-Rate nach Land")
    if "Land" in df_local.columns:
        table_land = df_local.groupby("Land")["is_fraud"].mean().sort_values(ascending=False)
        report_lines.append("| Land | Fraud-Rate |")
        report_lines.append("| --- | --- |")
        for land, rate in table_land.items():
            report_lines.append(f"| {land} | {rate:.2%} |")
    else:
        report_lines.append("Keine Spalte 'Land' vorhanden.")
    report_lines.append("")

    report_lines.append("## Fraud-Rate nach BUK")
    if "BUK" in df_local.columns:
        table_buk = df_local.groupby("BUK")["is_fraud"].mean().sort_values(ascending=False)
        report_lines.append("| BUK | Fraud-Rate |")
        report_lines.append("| --- | --- |")
        for buk, rate in table_buk.items():
            report_lines.append(f"| {buk} | {rate:.2%} |")
    else:
        report_lines.append("Keine Spalte 'BUK' vorhanden.")
    report_lines.append("")

    report_lines.append("## Betragsanalyse")
    if "Betrag_parsed" in df_local.columns:
        fraud_amount = df_local.loc[df_local["is_fraud"] == 1, "Betrag_parsed"].mean()
        non_amount = df_local.loc[df_local["is_fraud"] == 0, "Betrag_parsed"].mean()
        report_lines.append(f"- Durchschnittlicher Betrag (Fraud): {fraud_amount:.2f}")
        report_lines.append(f"- Durchschnittlicher Betrag (Normal): {non_amount:.2f}")
    else:
        report_lines.append("Betragsspalte nicht vorhanden.")
    report_lines.append("")

    report_lines.append("## Text-Keywords")
    if vectorizer is not None:
        feature_names = vectorizer.get_feature_names_out()
        fraud_texts = df_features.loc[y == 1, "notes_text"].fillna("")
        if not fraud_texts.empty:
            transformed = vectorizer.transform(fraud_texts)
            summed = np.asarray(transformed.sum(axis=0)).ravel()
            top_idx = np.argsort(summed)[::-1][:10]
            for idx in top_idx:
                report_lines.append(f"- {feature_names[idx]} (Gewicht {summed[idx]:.2f})")
        else:
            report_lines.append("Keine Fraud-Texte vorhanden.")
    else:
        report_lines.append("Text-Vektorisierer nicht verfügbar.")

    markdown_text = "\n".join(report_lines)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_report_"))
    md_path = tmp_dir / "pattern_report.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    return "Report erstellt.", markdown_text, str(md_path), state


def batch_predict_action(upload, state: Optional[Dict[str, Any]]):
    state = state or {}
    model = state.get("baseline_model")
    if model is None:
        return "Bitte zuerst Baseline trainieren.", None

    if upload is None:
        return "Bitte eine Excel-Datei hochladen.", None

    with gr.Progress(track_tqdm=False) as progress:
        progress(0.05, desc="Lade Datei")
        df_input = pd.read_excel(upload.name)
        progress(0.3, desc="Berechne Scores")
        scores = model.predict_proba(df_input)[:, -1]
        predictions = (scores >= 0.5).astype(int)
        df_out = df_input.copy()
        df_out["fraud_score"] = scores * 100.0
        df_out["prediction"] = predictions
        progress(0.8, desc="Schreibe Ergebnis")
        tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_batch_"))
        out_path = tmp_dir / "batch_predictions.xlsx"
        df_out.to_excel(out_path, index=False)
        progress(1.0, desc="Fertig")

    logger.info(
        "ui_batch_prediction_completed",
        rows=len(df_input),
        output=str(out_path),
    )

    return "Batch abgeschlossen.", str(out_path)


def record_feedback(state: Dict[str, Any], row_index: int, feedback: str, user: str, comment: str) -> str:
    ensure_feedback_db()
    predictions = state.get("predictions_full")
    df_features = state.get("df_features")
    if predictions is None or df_features is None:
        return "Keine Predictions verfügbar."

    if row_index < 0 or row_index >= len(predictions):
        return "Index außerhalb gültiger Grenzen."

    row = predictions.iloc[row_index]
    df_row = df_features.iloc[row_index]
    beleg_id = df_row.get("Rechnungsnummer") if isinstance(df_row, pd.Series) else None

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO feedback (beleg_index, beleg_id, timestamp, user, score, prediction, feedback, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row_index),
                None if pd.isna(beleg_id) else str(beleg_id),
                datetime.utcnow().isoformat(),
                user or "unknown",
                float(row["fraud_score"]),
                int(row["prediction"]),
                feedback,
                comment or "",
            ),
        )
        conn.commit()

    return "Feedback gespeichert."


def feedback_action(state: Optional[Dict[str, Any]], row_index, user, comment, label):
    state = state or {}
    try:
        idx = int(row_index)
    except (TypeError, ValueError):
        return "Ungültiger Index.", state

    message = record_feedback(state, idx, label, user or "unknown", comment)
    return message, state


def feedback_report_action(state: Optional[Dict[str, Any]]):
    ensure_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM feedback", conn, parse_dates=["timestamp"]) if conn else pd.DataFrame()

    if df.empty:
        return "Noch kein Feedback vorhanden.", "", state

    now = datetime.utcnow()
    last_week = now - timedelta(days=7)
    df_week = df[df["timestamp"] >= last_week.isoformat()]
    if df_week.empty:
        summary = "Keine Feedback-Daten in den letzten 7 Tagen."
    else:
        tp = (df_week["feedback"] == "TP").sum()
        fp = (df_week["feedback"] == "FP").sum()
        total = tp + fp
        precision = tp / total if total else float("nan")
        false_positive_rate = fp / total if total else float("nan")
        summary = (
            f"Letzte Woche: Präzision {precision:.2%}, False Positive Rate {false_positive_rate:.2%}"
            if total
            else "Keine validen Feedback-Daten für letzte Woche."
        )

    overall_tp = (df["feedback"] == "TP").sum()
    overall_fp = (df["feedback"] == "FP").sum()
    overall_total = overall_tp + overall_fp
    overall_precision = overall_tp / overall_total if overall_total else float("nan")

    report_lines = ["# Feedback-Report", ""]
    report_lines.append(summary)
    report_lines.append("")
    report_lines.append(f"Gesamt-Precision: {overall_precision:.2%}" if overall_total else "Gesamt-Precision: n/a")
    report_lines.append(f"Anzahl Feedbacks: {overall_total}")

    md_text = "\n".join(report_lines)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_feedback_"))
    md_path = tmp_dir / "feedback_report.md"
    md_path.write_text(md_text, encoding="utf-8")

    return summary, md_text, str(md_path), state


def feedback_tp_action(state, row_index, user, comment):
    return feedback_action(state, row_index, user, comment, "TP")


def feedback_fp_action(state, row_index, user, comment):
    return feedback_action(state, row_index, user, comment, "FP")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="pruefomat") as demo:
        gr.Markdown(
            """
            # pruefomat – Fraud/Veri-Selector
            """
        )

        state = gr.State({
            "config": DEFAULT_CONFIG.model_copy(deep=True),
            "config_path": str(DEFAULT_CONFIG_PATH),
        })

        with gr.Tabs():
            with gr.Tab("Training & Analyse"):
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
                predictions_table = gr.Dataframe(label="Predictions (Test Set, sortiert nach Risiko)", interactive=False)
                predictions_download = gr.File(label="Predictions CSV", interactive=False)
                explain_index = gr.Number(label="Zeilenindex für Erklärung", value=0, precision=0)
                explain_btn = gr.Button("Erklärung anzeigen")
                explain_status = gr.Textbox(label="Erklärungs-Status", interactive=False)
                explain_json = gr.JSON(label="Top-Features (SHAP)")
                explain_download = gr.File(label="Erklärungs-JSON", interactive=False)
                report_btn = gr.Button("Pattern Report generieren")
                report_status = gr.Textbox(label="Report-Status", interactive=False)
                report_preview = gr.Markdown(label="Report Vorschau")
                report_download = gr.File(label="Report Download", interactive=False)
                feedback_user = gr.Textbox(label="Prüfer:in", value="")
                feedback_comment = gr.Textbox(label="Feedback-Kommentar", lines=2)
                feedback_index = gr.Number(label="Zeilenindex (Feedback)", value=0, precision=0)
                feedback_tp_btn = gr.Button("✅ True Positive")
                feedback_fp_btn = gr.Button("❌ False Positive")
                feedback_status = gr.Textbox(label="Feedback-Status", interactive=False)
                feedback_report_btn = gr.Button("Feedback-Report")
                feedback_report_status = gr.Textbox(label="Feedback-Report-Status", interactive=False)
                feedback_report_preview = gr.Markdown(label="Feedback Report Vorschau")
                feedback_report_download = gr.File(label="Feedback Report Download", interactive=False)

            with gr.Tab("Batch Prediction"):
                batch_file = gr.File(label="Excel-Datei", file_types=[".xlsx", ".xls"])  # type: ignore[arg-type]
                batch_button = gr.Button("Belege prüfen")
                batch_status = gr.Textbox(label="Batch-Status", interactive=False)
                batch_download = gr.File(label="Batch Download", interactive=False)

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
                predictions_table,
                predictions_download,
                state,
            ],
        )

        explain_btn.click(
            explain_prediction_action,
            inputs=[state, explain_index],
            outputs=[explain_status, explain_json, explain_download, state],
        )

        report_btn.click(
            generate_pattern_report_action,
            inputs=[state],
            outputs=[report_status, report_preview, report_download, state],
        )

        feedback_tp_btn.click(
            feedback_tp_action,
            inputs=[state, feedback_index, feedback_user, feedback_comment],
            outputs=[feedback_status, state],
        )

        feedback_fp_btn.click(
            feedback_fp_action,
            inputs=[state, feedback_index, feedback_user, feedback_comment],
            outputs=[feedback_status, state],
        )

        feedback_report_btn.click(
            feedback_report_action,
            inputs=[state],
            outputs=[feedback_report_status, feedback_report_preview, feedback_report_download, state],
        )

        batch_button.click(
            batch_predict_action,
            inputs=[batch_file, state],
            outputs=[batch_status, batch_download],
        )

    return demo


def main():
    demo = build_interface()
    demo.launch(share=False, show_error=True)


if __name__ == "__main__":
    main()
