from types import SimpleNamespace

import pandas as pd
import numpy as np

from app import (
    _initial_state,
    load_dataset,
    build_pipeline_action,
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
