"""Training helpers and CLI for multi-class Maßnahme prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config_loader import load_config, normalize_config_columns
from build_pipeline_veri import (
    DataFramePreparer,
    DaysUntilDueAdder,
    build_preprocessor,
    infer_feature_plan,
    normalize_columns,
)
from logging_utils import get_logger, mask_sensitive_data

from .train_binary import Schema, parse_flag, parse_money


logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Return engineered features and encoded Maßnahme labels."""

    df = df.copy()
    label_col = "Massnahme_2025"
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col} missing")

    label_series = df[label_col].fillna("Unbekannt").astype(str)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_series)

    flag_cols = [col for col, meta in schema.columns.items() if meta.get("type") == "flag"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = parse_flag(df[col]).astype(int)

    amount_col = "Betrag_parsed" if "Betrag_parsed" in df.columns else None
    if amount_col is None and "Betrag" in df.columns:
        df["Betrag_parsed"] = parse_money(df["Betrag"], schema.options.get("currency_symbols", []))
        amount_col = "Betrag_parsed"
    df["betrag_log"] = np.log1p(df[amount_col].clip(lower=0)) if amount_col else 0.0

    if "Belegdatum" in df.columns and "Fällig" in df.columns:
        beleg = pd.to_datetime(df["Belegdatum"], errors="coerce")
        due = pd.to_datetime(df["Fällig"], errors="coerce")
        df["tage_bis_faellig"] = (due - beleg).dt.days
    else:
        df["tage_bis_faellig"] = np.nan
    df["ist_ueberfaellig"] = (df["tage_bis_faellig"].fillna(0) < 0).astype(int)

    text_cols = [col for col, meta in schema.columns.items() if meta.get("type") == "text" and col in df.columns]
    if text_cols:
        df["notes_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        df["notes_text"] = ""

    feature_df = df.drop(columns=[label_col], errors="ignore")
    return feature_df, pd.Series(y_encoded), label_encoder


def evaluate_multiclass(y_true, y_pred, classes):
    """Return multi-class metrics dictionary for Maßnahme predictions."""

    labels = np.arange(len(classes))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


def _prepare_training_data(
    df_norm: pd.DataFrame,
    config,
) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = config.data.target_col or "Massnahme_2025"
    if target_col not in df_norm.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    target_series = df_norm[target_col].astype("string").str.strip().replace({"": pd.NA})
    mask = target_series.notna()
    if not mask.any():
        raise ValueError(f"Target column '{target_col}' contains no valid values")

    df_clean = df_norm.loc[mask].reset_index(drop=True)
    target_clean = target_series.loc[mask].reset_index(drop=True)
    features = df_clean.drop(columns=[target_col], errors="ignore")
    return features, target_clean


def train_massnahmen_cli(args: argparse.Namespace) -> int:
    excel_path = Path(args.excel)
    if not excel_path.exists():
        logger.error("cli_excel_missing", file=str(excel_path))
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    config = load_config(args.config)
    config = normalize_config_columns(config)

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df_raw = pd.read_excel(excel_path, sheet_name=sheet, dtype="object")
    df_norm, column_mapping = normalize_columns(df_raw)
    features, target_series = _prepare_training_data(df_norm, config)

    preparer = DataFramePreparer(
        amount_col=config.data.amount_col or "",
        issue_col=config.data.issue_col or "",
        due_col=config.data.due_col or "",
        date_columns=config.data.additional_date_columns,
        null_like=list({*config.data.null_like}),
    )
    prepared = preparer.fit_transform(features)
    with_due = DaysUntilDueAdder(
        issue_col=config.data.issue_col or "",
        due_col=config.data.due_col or "",
    ).fit_transform(prepared)

    feature_plan = infer_feature_plan(with_due, config)
    preprocessor = build_preprocessor(feature_plan, config)
    rf_kwargs = config.model.random_forest.model_dump(exclude_none=True)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_kwargs)),
        ]
    )

    stratify_series = target_series if target_series.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target_series,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_series,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(target_series)
    y_test_encoded = label_encoder.transform(y_test)
    y_pred_encoded = label_encoder.transform(y_pred)

    metrics = evaluate_multiclass(y_test_encoded, y_pred_encoded, list(label_encoder.classes_))
    confusion = metrics["confusion_matrix"]
    confusion_df = pd.DataFrame(confusion, index=label_encoder.classes_, columns=label_encoder.classes_)

    encoder = model.named_steps["preprocessor"].named_steps.get("encode")
    if encoder is not None and hasattr(encoder, "get_feature_names_out"):
        feature_names = encoder.get_feature_names_out()
    else:
        feature_names = np.array([f"f_{idx}" for idx in range(len(model.named_steps["classifier"].feature_importances_))])

    importance = model.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df.sort_values("importance", ascending=False, inplace=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "massnahmen_model.joblib"
    encoder_path = out_dir / "label_encoder.joblib"
    metrics_path = out_dir / "metrics.json"
    confusion_path = out_dir / "confusion_matrix.csv"
    importance_path = out_dir / "feature_importance.csv"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    importance_df.to_csv(importance_path, index=False)
    confusion_df.to_csv(confusion_path)

    metrics_serializable = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "classes": list(label_encoder.classes_),
        "confusion_matrix": confusion.tolist(),
        "classification_report": metrics["classification_report"],
        "column_mapping": column_mapping,
    }
    metrics_path.write_text(json.dumps(metrics_serializable, indent=2), encoding="utf-8")

    logger.info(
        "cli_train_completed",
        **mask_sensitive_data(
            {
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "accuracy": metrics_serializable["accuracy"],
            }
        ),
    )

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-class Maßnahme predictor")
    parser.add_argument("--excel", required=True, help="Pfad zur Trainingsdatei (Excel)")
    parser.add_argument("--sheet", default=0, help="Sheet-Name oder Index")
    parser.add_argument("--config", default="config/default_config.yaml", help="Pfad zur App-Konfiguration")
    parser.add_argument("--output-dir", required=True, help="Zielordner für Artefakte")
    parser.add_argument("--test-size", type=float, default=0.2, help="Testset-Anteil (0-1)")
    parser.add_argument("--random-state", type=int, default=42, help="Zufallsseed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_massnahmen_cli(args)


if __name__ == "__main__":  # pragma: no cover
    main()
