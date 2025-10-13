import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from types import SimpleNamespace

import shap

from app import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    FEEDBACK_DB_PATH,
    build_pipeline_action,
    batch_predict_action,
    explain_prediction_action,
    feedback_fp_action,
    feedback_report_action,
    feedback_tp_action,
    train_baseline_action,
)


@pytest.fixture(autouse=True)
def patch_shap(monkeypatch):
    class DummyTreeExplainer:
        def __init__(self, model, background=None):
            self.model = model

        def shap_values(self, X):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr[None, :]
            zeros = np.zeros_like(arr, dtype=float)
            return [zeros, zeros]

    class DummyLinearExplainer:
        def __init__(self, model, data=None):
            pass

        def __call__(self, X):
            arr = np.asarray(X)
            zeros = np.zeros_like(arr, dtype=float)
            return SimpleNamespace(values=zeros)

    monkeypatch.setattr(shap, "TreeExplainer", DummyTreeExplainer)
    monkeypatch.setattr(shap, "LinearExplainer", DummyLinearExplainer)
    monkeypatch.setattr(shap, "summary_plot", lambda *args, **kwargs: None)


@pytest.fixture
def baseline_state():
    data = {
        "Betrag": [
            "1000,00",
            "2500,00",
            "500,00",
            "1500,00",
            "750,00",
            "1800,00",
            "2200,00",
            "1950,00",
            "640,00",
            "3100,00",
        ],
        "Belegdatum": [
            "01.01.2025",
            "05.01.2025",
            "06.01.2025",
            "07.01.2025",
            "08.01.2025",
            "09.01.2025",
            "10.01.2025",
            "11.01.2025",
            "12.01.2025",
            "13.01.2025",
        ],
        "Fällig": [
            "03.01.2025",
            "10.01.2025",
            "06.02.2025",
            "07.02.2025",
            "08.02.2025",
            "10.02.2025",
            "11.02.2025",
            "12.02.2025",
            "13.02.2025",
            "14.02.2025",
        ],
        "Land": ["DE", "DE", "AT", "DE", "AT", "DE", "DE", "AT", "DE", "DE"],
        "BUK": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "Debitor": ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109"],
        "DEB Name": [
            "Alpha",
            "Beta",
            "Gamma",
            "Delta",
            "Epsilon",
            "Zeta",
            "Eta",
            "Theta",
            "Iota",
            "Kappa",
        ],
        "Maßnahme 2025": [
            "Pruefen",
            "Rueckruf",
            "",
            "Nachfassen",
            "Klärung",
            "Pruefen",
            "Überprüfung",
            "Monitoring",
            "Reminder",
            "Erinnerung",
        ],
        "Hinweise": [
            "Verdacht",
            "ok",
            "Warnung",
            "Bitte prüfen",
            "offen",
            "Verdächtig",
            "Rückfrage",
            "Nachhaken",
            "Zahlung angekündigt",
            "Telefonat ausstehend",
        ],
        "Ampel": [1, 2, 1, 3, 1, 2, 3, 1, 2, 3],
    }
    df = pd.DataFrame(data)
    features = df.drop(columns=["Ampel"])
    target = df["Ampel"]
    config = DEFAULT_CONFIG.model_copy(deep=True)
    config.preprocessing.tfidf_min_df = 1

    state = {
        "config": config,
        "config_path": str(DEFAULT_CONFIG_PATH),
        "df_features": features,
        "target": target,
    }
    pipeline_result = build_pipeline_action(state)
    state = pipeline_result[-1]
    result = train_baseline_action(state)
    metrics = result[1]
    predictions = result[8]
    updated_state = result[-1]
    return metrics, predictions, updated_state


@pytest.fixture
def feedback_db(monkeypatch, tmp_path):
    path = tmp_path / "feedback.db"
    monkeypatch.setattr("app.FEEDBACK_DB_PATH", path)
    return path


def test_recall_metrics(baseline_state):
    metrics, _, _ = baseline_state
    assert "recall" in metrics
    assert "precision" in metrics
    assert "f2_score" in metrics


def test_confidence_scores(baseline_state):
    _, predictions, _ = baseline_state
    assert (predictions["fraud_score"] >= 0).all()
    assert (predictions["fraud_score"] <= 100).all()


def test_threshold_application(baseline_state):
    _, predictions, _ = baseline_state
    expected = (predictions["fraud_score"] / 100 >= 0.5).astype(int)
    assert predictions["prediction"].astype(int).tolist() == expected.tolist()


def test_shap_explanations(baseline_state):
    _, _, state = baseline_state
    status, explanation, download_path, _ = explain_prediction_action(state, 0)
    assert "Top-Features" in status
    assert isinstance(explanation, list)
    assert len(explanation) == 5
    assert Path(download_path).exists()


def test_batch_prediction(tmp_path, baseline_state):
    _, _, state = baseline_state
    df = state["df_features"].copy()
    batch_file = tmp_path / "batch.xlsx"
    df.to_excel(batch_file, index=False)
    upload = SimpleNamespace(name=str(batch_file))
    status, download_path = batch_predict_action(upload, state)
    assert "Batch abgeschlossen" in status
    out_df = pd.read_excel(download_path)
    assert "fraud_score" in out_df.columns
    assert "prediction" in out_df.columns


def test_feedback_flow(baseline_state, feedback_db):
    _, predictions, state = baseline_state
    status, state = feedback_tp_action(state, predictions.iloc[0]["row_index"], "alice", "korrekt")
    assert "gespeichert" in status
    status_fp, state = feedback_fp_action(state, predictions.iloc[1]["row_index"], "alice", "falsch")
    assert "gespeichert" in status_fp
    report_status, report_text, report_path, _ = feedback_report_action(state)
    assert isinstance(report_status, str)
    assert Path(report_path).exists()
