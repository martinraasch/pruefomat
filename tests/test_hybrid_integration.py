from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app import (
    _initial_state,
    batch_predict_action,
    build_pipeline_action,
    load_dataset,
    train_baseline_action,
)
from src.business_rules import BusinessRule, RuleOperator, SimpleCondition
from src.rule_engine import RuleEngine
from src.hybrid_predictor import HybridMassnahmenPredictor


def _run_training(tmp_path):
    train_df = pd.DataFrame(
        {
            "Betrag": [1000.0, 80000.0, 2000.0],
            "Ampel": [1, 2, 1],
            "BUK": ["A", "B", "A"],
            "Debitor": ["100", "200", "100"],
            "negativ": [False, False, False],
            "Belegdatum": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "Fällig": ["2025-02-01", "2025-02-10", "2025-02-05"],
            "Hinweise": ["verdacht", "kontrolle", "verdacht"],
            "Massnahme_2025": ["Rechnungsprüfung", "Prüfung", "Rechnungsprüfung"],
        }
    )

    train_path = tmp_path / "train.xlsx"
    train_df.to_excel(train_path, index=False)

    upload = SimpleNamespace(name=str(train_path))
    state = _initial_state()
    load_result = load_dataset(upload, None, "0", "Massnahme_2025", None, state)
    state = load_result[-1]
    state["config"].preprocessing.tfidf_min_df = 1

    build_result = build_pipeline_action(state)
    state = build_result[-1]

    train_result = train_baseline_action(state)
    state = train_result[-1]

    return state


def test_hybrid_pipeline_end_to_end(tmp_path):
    state = _run_training(tmp_path)
    predictor = state["hybrid_predictor"]
    assert predictor is not None

    feature_columns = state["df_features"].columns
    base_row = {col: state["df_features"].iloc[0].get(col, np.nan) for col in feature_columns}
    base_row["Betrag"] = 3000.0
    base_row["Ampel"] = 1
    base_row["negativ"] = False

    new_df = pd.DataFrame([base_row])
    result = predictor.predict(new_df)

    assert result.loc[0, "prediction"] == "Rechnungsprüfung"
    assert result.loc[0, "source"] != "ml"


def test_rule_vs_ml_conflict_resolution():
    rule = BusinessRule(
        name="force_rule",
        priority=1,
        condition_type="always",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="RuleAction",
        confidence=0.95,
    )
    fallback = BusinessRule(
        name="ml_fallback",
        priority=999,
        condition_type="always",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="ML_PREDICTION",
        confidence=None,
    )
    engine = RuleEngine([rule, fallback])

    class DummyModel:
        classes_ = np.array(["MLAction"])  # type: ignore[assignment]

        def predict_proba(self, X):  # noqa: N802
            return np.array([[0.1]])

        def predict(self, X):  # noqa: N802
            return np.array(["MLAction" for _ in range(len(X))])

    predictor = HybridMassnahmenPredictor(DummyModel(), engine)

    df = pd.DataFrame([{"Ampel": 2}])
    result = predictor.predict(df)

    assert result.loc[0, "prediction"] == "RuleAction"
    assert result.loc[0, "source"] == "force_rule"

    # Rule returns None → ML fallback
    engine_no_rule = RuleEngine([fallback])
    predictor_fallback = HybridMassnahmenPredictor(DummyModel(), engine_no_rule)
    result_fallback = predictor_fallback.predict(df)
    assert result_fallback.loc[0, "source"] == "ml"
    assert result_fallback.loc[0, "prediction"] == "MLAction"


def test_negativ_filter_in_batch(tmp_path, monkeypatch):
    class DummyProgress:
        def __call__(self, *args, **kwargs):
            return None

    class DummyModel:
        def predict_proba(self, X):  # noqa: N802
            return np.tile(np.array([0.2, 0.8]), (len(X), 1))

        def predict(self, X):  # noqa: N802
            return np.array(["Rechnungsprüfung"] * len(X))

    monkeypatch.setattr("app.gr.Progress", lambda track_tqdm=False: DummyProgress())

    df = pd.DataFrame(
        {
            "Betrag": ["1000,00", "2000,00", "3000,00"],
            "Ampel": [1, 2, 1],
            "negativ": [False, True, False],
            "BUK": ["A", "B", "C"],
            "Debitor": ["100", "200", "300"],
        }
    )
    path = tmp_path / "batch.xlsx"
    df.to_excel(path, index=False)

    state = _initial_state()
    state.update(
        {
            "baseline_model": DummyModel(),
            "selected_columns": ["Betrag", "Ampel", "BUK", "Debitor"],
            "hybrid_predictor": None,
            "rule_engine": None,
        }
    )

    upload = SimpleNamespace(name=str(path))
    status, output_path = batch_predict_action(upload, state)

    assert "Batch abgeschlossen" in status
    out_df = pd.read_excel(output_path)

    negativ_rows = out_df[out_df["Debitor"] == "200"]
    assert len(negativ_rows) == 1
    negativ_row = negativ_rows.iloc[0]
    assert "Bereits abgelehnt" in negativ_row["Massnahme_2025"]
    assert pytest.approx(negativ_row["fraud_score"], rel=1e-6) == 100.0
    assert negativ_row["prediction_source"] == "negativ_flag"

    normal_rows = out_df[out_df["Debitor"].isin(["100", "300"])]
    assert len(normal_rows) == 2
    assert all(normal_rows["prediction_source"] != "negativ_flag")
