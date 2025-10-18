import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from types import SimpleNamespace

import shap

from app import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    CANONICAL_MASSNAHMEN,
    FEEDBACK_DB_PATH,
    batch_predict_action,
    build_pipeline_action,
    export_explanation_action,
    explain_massnahme_action,
    feedback_fp_action,
    feedback_report_action,
    feedback_tp_action,
    generate_pattern_report_action,
    train_baseline_action,
)


@pytest.fixture(autouse=True)
def patch_shap(monkeypatch):
    class DummyTreeExplainer:
        def __init__(self, model, background=None):
            self.model = model

        def shap_values(self, X, **kwargs):
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
        "negativ": [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ],
    }
    df = pd.DataFrame(data)
    df = df.rename(columns={"Maßnahme 2025": "Massnahme_2025"})
    features = df.drop(columns=["Ampel", "Massnahme_2025"])
    target = df["Massnahme_2025"]
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
    (
        status,
        metrics,
        confusion_df,
        confusion_plot,
        model_path,
        metrics_path,
        importance_table,
        importance_plot_path,
        importance_csv_path,
        predictions,
        predictions_path,
        rule_cov_display,
        ml_fallback_display,
        distribution,
        updated_state,
    ) = train_baseline_action(state)
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
    assert "samples" in metrics
    assert metrics["samples"]["validation"] > 0


def test_confidence_scores(baseline_state):
    _, predictions, state = baseline_state
    assert "final_confidence" in predictions
    assert predictions["final_confidence"].between(0, 1).all()
    predictions_full = state["predictions_full"]
    assert "review_score" in predictions_full
    assert predictions_full["review_score"].between(0, 1).all()
    assert "explanation" in predictions


def test_threshold_application(baseline_state):
    _, predictions, state = baseline_state
    assert "final_prediction" in predictions
    assert predictions["final_prediction"].notna().all()
    full_predictions = state["predictions_full"]
    assert "is_correct" in full_predictions
    assert set(full_predictions["is_correct"].unique()).issubset({True, False})


def test_shap_explanations(baseline_state):
    _, _, state = baseline_state
    status, explanation, download_path, _ = explain_massnahme_action(state, 0)

    assert "Erklärung" in status
    assert isinstance(explanation, str)
    assert explanation.startswith("# Erklärung für Rechnung")
    assert "## Begründung" in explanation
    assert Path(download_path).exists()

    # Ensure export uses stored explanation
    export_status, export_path, _ = export_explanation_action(state)
    assert "Markdown exportiert" in export_status
    assert Path(export_path).exists()


def test_batch_prediction(tmp_path, baseline_state):
    _, _, state = baseline_state
    df = state["df_features"].copy()
    batch_file = tmp_path / "batch.xlsx"
    df.to_excel(batch_file, index=False)
    upload = SimpleNamespace(name=str(batch_file))
    status, download_path = batch_predict_action(upload, state)
    assert "Batch abgeschlossen" in status
    out_df = pd.read_excel(download_path)
    assert "final_prediction" in out_df.columns
    assert "final_confidence" in out_df.columns


def test_negativ_feature_is_available(baseline_state):
    _, _, state = baseline_state
    feature_plan = state.get("feature_plan")
    assert feature_plan is not None
    assert "negativ" in feature_plan.categorical


def test_feedback_flow(baseline_state, feedback_db):
    _, predictions, state = baseline_state
    status, state = feedback_tp_action(state, predictions.iloc[0]["row_index"], "alice", "korrekt")
    assert "gespeichert" in status
    status_fp, state = feedback_fp_action(state, predictions.iloc[1]["row_index"], "alice", "falsch")
    assert "gespeichert" in status_fp
    report_status, report_text, report_path, _ = feedback_report_action(state)
    assert isinstance(report_status, str)
    assert Path(report_path).exists()


def test_top_k_metrics_and_cost_simulation(baseline_state):
    metrics, predictions, _ = baseline_state
    top_k = metrics.get("top_k_hit_rate")
    assert top_k is not None
    assert 0 < top_k["k"] <= len(metrics["classes"])
    assert 0.0 <= top_k["hit_rate"] <= 1.0

    cost_sim = metrics.get("cost_simulation")
    assert cost_sim is not None
    for key in ("review_share", "cost_review", "cost_miss", "net_benefit_top_share", "net_benefit_full_review"):
        assert key in cost_sim


def test_confusion_heatmap_is_created(baseline_state):
    _, _, state = baseline_state
    heatmap_path = state.get("confusion_heatmap")
    assert heatmap_path is not None
    assert Path(heatmap_path).exists()


def test_status_message_contains_accuracy(baseline_state):
    _, _, state = baseline_state
    status, *_ = train_baseline_action(state)
    assert "Genauigkeit" in status


def test_pattern_report_multiclass(baseline_state):
    _, _, state = baseline_state
    status, markdown, report_path, _ = generate_pattern_report_action(state)

    assert "Report" in status
    assert "Klassenverteilung" in markdown
    assert "Pattern Report" in markdown
    assert Path(report_path).exists()


def test_batch_prediction_rule_columns_preserved(tmp_path, baseline_state):
    _, _, state = baseline_state
    state["selected_columns"] = ["Hinweise"]

    df_min = pd.DataFrame(
        {
            "Hinweise": ["ok"],
            "Ampel": [1],
            "Betrag": ["1.000,00"],
            "BUK": ["A"],
            "Debitor": ["100"],
        }
    )
    batch_file = tmp_path / "batch_minimal.xlsx"
    df_min.to_excel(batch_file, index=False)
    upload = SimpleNamespace(name=str(batch_file))

    status, download_path = batch_predict_action(upload, state)

    assert "Batch abgeschlossen" in status
    result_df = pd.read_excel(download_path)
    match = result_df.loc[result_df["prediction_source"] == "niedrig_betrag_gruene_ampel"]
    assert not match.empty
    assert (match["final_prediction"] == "Rechnungsprüfung").all()


def test_batch_predictions_are_canonical(tmp_path, baseline_state):
    _, _, state = baseline_state
    df = state["df_features"].copy()
    df = pd.concat([df] * 20, ignore_index=True)
    batch_file = tmp_path / "batch_canonical.xlsx"
    df.to_excel(batch_file, index=False)
    upload = SimpleNamespace(name=str(batch_file))

    status, download_path = batch_predict_action(upload, state)
    assert "Batch abgeschlossen" in status
    out_df = pd.read_excel(download_path)

    allowed = set(CANONICAL_MASSNAHMEN) | {"Unbekannt"}
    assert set(out_df["final_prediction"].unique()).issubset(allowed)


def test_batch_prediction_without_negativ_column(tmp_path, baseline_state):
    _, _, state = baseline_state
    df = state["df_features"].copy().drop(columns=["negativ"], errors="ignore")
    batch_file = tmp_path / "batch_no_negativ.xlsx"
    df.to_excel(batch_file, index=False)
    upload = SimpleNamespace(name=str(batch_file))

    status, download_path = batch_predict_action(upload, state)
    assert "Batch abgeschlossen" in status
    out_df = pd.read_excel(download_path)
    assert len(out_df) == len(df)


def test_historical_gutschrift_rule_triggers():
    data = {
        "Betrag": ["500,00", "750,00", "600,00", "900,00"],
        "Ampel": [1, 2, 1, 2],
        "BUK": ["A", "A", "A", "B"],
        "Debitor": ["100", "100", "100", "200"],
        "Hinweise": ["", "", "", ""],
        "Massnahme_2025": [
            "Gutschriftsverfahren",
            "Gutschriftsverfahren",
            "Gutschriftsverfahren",
            "Rechnungsprüfung",
        ],
    }

    df = pd.DataFrame(data)
    features_full = df.drop(columns=["Massnahme_2025"])
    features_selected = features_full[["Betrag", "Ampel"]]
    target = df["Massnahme_2025"]

    config = DEFAULT_CONFIG.model_copy(deep=True)
    config.preprocessing.tfidf_min_df = 1

    state = {
        "config": config,
        "config_path": str(DEFAULT_CONFIG_PATH),
        "df_features": features_selected.copy(),
        "df_features_full": features_full.copy(),
        "target": target,
        "selected_columns": list(features_selected.columns),
    }

    _, _, _, state = build_pipeline_action(state)
    # Ensure full features remain available for historical lookup
    state["df_features_full"] = features_full.copy()

    result = train_baseline_action(state)
    state = result[-1]

    predictor = state["hybrid_predictor"]
    assert predictor is not None

    new_invoice = pd.DataFrame(
        {
            "Betrag": ["1.200,00"],
            "Ampel": [2],
            "BUK": ["A"],
            "Debitor": ["100"],
        }
    )

    prediction = predictor.predict(new_invoice)
    assert prediction.loc[0, "prediction"] == "Gutschriftsverfahren"
    assert prediction.loc[0, "source"] == "historische_gutschrift_muster"


def test_label_encoder_uses_canonical_massnahmen():
    data = {
        "Betrag": [
            "1.000,00",
            "2.500,00",
            "1.200,00",
            "3.000,00",
            "4.100,00",
            "900,00",
        ],
        "Ampel": [1, 2, 2, 3, 1, 3],
        "BUK": ["A", "B", "C", "A", "B", "C"],
        "Debitor": ["100", "101", "102", "103", "104", "105"],
        "Hinweise": ["", "", "", "", "", ""],
        "Massnahme_2025": [
            "Rechnungsprüfung",
            "telefonische rechnungsbestätigung(vorgelagert)",
            "Gutschriftsverfahren",
            "schriftliche Saldenbestätigung (nachgelagert)",
            "nur zur Belegerfassung",
            "Freigabe gemäß Kompetenzkatalog",
        ],
    }

    df = pd.DataFrame(data)
    features = df.drop(columns=["Massnahme_2025"])
    target = df["Massnahme_2025"]

    config = DEFAULT_CONFIG.model_copy(deep=True)
    config.preprocessing.tfidf_min_df = 1

    state = {
        "config": config,
        "config_path": str(DEFAULT_CONFIG_PATH),
        "df_features": features.copy(),
        "df_features_full": features.copy(),
        "target": target.copy(),
        "selected_columns": list(features.columns),
    }

    _, _, _, state = build_pipeline_action(state)
    result = train_baseline_action(state)
    updated_state = result[-1]

    classes = updated_state.get("label_classes", [])
    assert "telefonische Rechnungsbestätigung (vorgelagert)" in classes
    assert "telefonische rechnungsbestätigung(vorgelagert)" not in classes
    allowed = set(CANONICAL_MASSNAHMEN) | {"Unbekannt"}
    assert set(classes).issubset(allowed)
