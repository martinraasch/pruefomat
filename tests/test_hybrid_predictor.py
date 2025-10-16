import numpy as np
import pandas as pd

from src.business_rules import BusinessRule, RuleOperator, SimpleCondition
from src.hybrid_predictor import HybridMassnahmenPredictor
from src.rule_engine import RuleEngine


class DummyModel:
    classes_ = np.array(["Rechnungsprüfung", "Gutschrift"])

    def predict_proba(self, X):  # noqa: N802 - sklearn style
        return np.array([[0.2, 0.8] for _ in range(len(X))])

    def predict(self, X):  # pragma: no cover - fallback path
        return np.array(["Gutschrift" for _ in range(len(X))])


def build_rule_engine():
    direct_rule = BusinessRule(
        name="direct_rule",
        priority=1,
        condition_type="simple",
        conditions=[SimpleCondition("Ampel", RuleOperator.EQUALS, 1)],
        action_field="Massnahme_2025",
        action_value="Rechnungsprüfung",
        confidence=1.0,
    )
    fallback_rule = BusinessRule(
        name="ml_fallback",
        priority=2,
        condition_type="always",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="ML_PREDICTION",
        confidence=None,
    )
    return RuleEngine([direct_rule, fallback_rule])


def test_hybrid_predictor_rules_first():
    engine = build_rule_engine()
    model = DummyModel()
    predictor = HybridMassnahmenPredictor(model, engine)
    df = pd.DataFrame([{"Ampel": 1}, {"Ampel": 0}])

    result = predictor.predict(df)

    assert list(result["prediction"]) == ["Rechnungsprüfung", "Gutschrift"]
    assert list(result["source"]) == ["direct_rule", "ml"]
    assert result.loc[0, "confidence"] == 1.0
    assert result.loc[1, "confidence"] == 0.8


def test_hybrid_predictor_explain_rule_path():
    engine = build_rule_engine()
    model = DummyModel()
    predictor = HybridMassnahmenPredictor(model, engine)
    row = pd.Series({"Ampel": 1})

    explanation = predictor.explain(row)

    assert explanation["prediction"] == "Rechnungsprüfung"
    assert explanation["source"] == "direct_rule"
    matched = explanation["details"]["matched_conditions"]
    assert matched[0]["matched"] is True


def test_hybrid_predictor_explain_ml_path():
    engine = build_rule_engine()
    model = DummyModel()
    predictor = HybridMassnahmenPredictor(model, engine)
    row = pd.Series({"Ampel": 0})

    explanation = predictor.explain(row)

    assert explanation["prediction"] == "Gutschrift"
    assert explanation["source"] == "ml"
    assert "shap_top5" in explanation["details"]
