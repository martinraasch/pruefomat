"""Gradio interface for the pruefomat Veri pipeline builder."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from build_pipeline_veri import (
    DataFramePreparer,
    DaysUntilDueAdder,
    FeaturePlan,
    build_preprocessor,
    infer_feature_plan,
    normalize_column_name,
    normalize_columns,
)

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


def load_dataset(upload, sheet_text: str, target_text: str, state: Optional[Dict[str, Any]]):
    state = state or {}
    if upload is None:
        return "Bitte zuerst eine Excel-Datei hochladen.", None, None, state

    sheet = _parse_sheet(sheet_text)
    try:
        df_raw = pd.read_excel(upload.name, sheet_name=sheet, dtype="object")
    except Exception as exc:  # pragma: no cover
        return f"Fehler beim Laden: {exc}", None, None, state

    df_norm, column_mapping = normalize_columns(df_raw)

    target_norm = normalize_column_name(target_text) if target_text else None
    target_series = None
    df_features = df_norm
    target_msg = "Keine Zielspalte gesetzt."
    if target_norm and target_norm in df_norm.columns:
        target_series = df_norm[target_norm]
        df_features = df_norm.drop(columns=[target_norm])
        target_msg = f"Zielspalte erkannt: {target_norm}"
    elif target_norm:
        target_msg = f"Warnung: '{target_text}' (normalisiert '{target_norm}') nicht gefunden."

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
    schema["column_mapping"] = column_mapping

    status = f"Geladen: {Path(upload.name).name} (Sheet {sheet}) | {len(df_norm)} Zeilen, {df_norm.shape[1]} Spalten. {target_msg}"
    return status, preview, schema, state


def build_pipeline_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    if df_features is None or df_features.empty:
        return "Keine Daten geladen.", None, None, state

    preparer_preview = DataFramePreparer(amount_col="Betrag", issue_col="Belegdatum", due_col="Faellig", date_columns=["Datum"])
    prepared = preparer_preview.fit_transform(df_features)
    with_due = DaysUntilDueAdder(issue_col="Belegdatum", due_col="Faellig").fit_transform(prepared)

    feature_plan = infer_feature_plan(with_due)
    if not feature_plan.numeric and not feature_plan.categorical and not feature_plan.text:
        return "Keine nutzbaren Features gefunden.", None, None, state

    preprocessor = build_preprocessor(feature_plan)
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
    }

    return "Preprocessor trainiert und gespeichert.", plan_json, str(prep_path), state


def preview_features_action(state: Optional[Dict[str, Any]], sample_size: int = 10):
    state = state or {}
    df_features = state.get("df_features")
    preprocessor = state.get("preprocessor")
    if df_features is None or preprocessor is None:
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
    return f"Vorschau fuer {len(sample)} Zeilen (erste {keep} Features).", preview_df


def train_baseline_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    target = state.get("target")
    feature_plan: FeaturePlan | None = state.get("feature_plan")

    if df_features is None or feature_plan is None:
        return "Bitte zuerst Pipeline bauen.", None, None, None, None, state
    if target is None:
        return "Keine Zielspalte verfuegbar.", None, None, None, None, state

    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_plan)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    stratify = target if target.dropna().nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df_features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    confusion = confusion_matrix(y_test, y_pred)
    confusion_df = pd.DataFrame(confusion)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_model_"))
    model_path = tmp_dir / "baseline_model.joblib"
    metrics_path = tmp_dir / "metrics.json"
    joblib.dump(baseline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    state.update(
        {
            "baseline_model": baseline,
            "baseline_path": str(model_path),
            "metrics_path": str(metrics_path),
        }
    )

    return "Baseline trainiert.", metrics, confusion_df, str(model_path), str(metrics_path), state


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

        state = gr.State({})

        with gr.Row():
            file_input = gr.File(label="Excel-Datei", file_types=[".xlsx", ".xls"])  # type: ignore[arg-type]
            sheet_input = gr.Textbox(value="0", label="Sheet (Index oder Name)")
            target_input = gr.Textbox(value="Ampel", label="Zielspalte (optional)")
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

        load_btn.click(
            load_dataset,
            inputs=[file_input, sheet_input, target_input, state],
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
            outputs=[baseline_status, metrics_json, confusion_df, model_download, metrics_download, state],
        )

    return demo


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
