"""Training helpers for multi-class Maßnahme prediction."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from .train_binary import Schema, parse_flag, parse_money


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

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, output_dict=True
        ),
    }
