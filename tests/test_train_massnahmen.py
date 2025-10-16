import numpy as np
import pandas as pd

from src.train_binary import Schema
from src.train_massnahmen import engineer_features, evaluate_multiclass


def test_engineer_features_label_encoding():
    schema = Schema(
        columns={
            "Ampel": {"type": "categorical"},
            "negativ": {"type": "flag"},
            "Hinweise": {"type": "text"},
        },
        options={"currency_symbols": []},
        label_name="Massnahme_2025",
        label_positive_rule="",
    )
    df = pd.DataFrame(
        {
            "Massnahme_2025": ["Rechnungsprüfung", "Gutschrift"],
            "Ampel": [1, 2],
            "negativ": ["ja", "nein"],
            "Betrag": [10000, 75000],
            "Belegdatum": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Fällig": pd.to_datetime(["2024-01-10", "2024-01-05"]),
            "Hinweise": ["Test", "Noch ein Test"],
        }
    )

    features, y, encoder = engineer_features(df, schema)

    assert list(encoder.classes_) == ["Gutschrift", "Rechnungsprüfung"]
    assert set(features.columns) >= {"Ampel", "negativ", "betrag_log", "tage_bis_faellig"}
    assert np.array_equal(y.values, np.array([1, 0]))


def test_evaluate_multiclass_metrics_structure():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])
    metrics = evaluate_multiclass(y_true, y_pred, ["A", "B"])

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert set(metrics["classification_report"].keys()) >= {"A", "B"}
