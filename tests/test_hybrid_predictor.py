import sys

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
    probs = explanation["details"].get("probabilities")
    assert probs is not None
    assert set(probs.keys()) == {"Rechnungsprüfung", "Gutschrift"}


def test_hybrid_predictor_explain_ml_restricted():
    restricted_rule = BusinessRule(
        name="restricted_ml",
        priority=1,
        condition_type="simple",
        conditions=[SimpleCondition("Ampel", RuleOperator.EQUALS, 2)],
        action_field="Massnahme_2025",
        action_value="ML_PREDICTION",
        confidence=None,
        ml_allowed_classes=["Rechnungsprüfung"],
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
    engine = RuleEngine([restricted_rule, fallback_rule])
    model = DummyModel()
    predictor = HybridMassnahmenPredictor(model, engine)
    row = pd.Series({"Ampel": 2})

    explanation = predictor.explain(row)

    assert explanation["prediction"] == "Rechnungsprüfung"
    assert explanation["source"] == "ml_restricted"
    context = explanation["details"].get("ml_context")
    assert context is not None
    assert context["rule"] == "restricted_ml"
    assert context["restricted"] is True
    assert context["allowed_classes"] == ["Rechnungsprüfung"]


class DummyMatrix:
    def __init__(self, data):
        self._data = np.asarray(data)

    def toarray(self):
        return self._data


class DummyPreprocessor:
    def transform(self, sample):
        return DummyMatrix([[1.0, -2.0]])


class DummyPipelineWithClassifier:
    def __init__(self):
        self.named_steps = {"classifier": object()}
        self.classes_ = np.array(["Rechnungsprüfung", "Gutschrift"])

    def predict_proba(self, sample):  # noqa: N802
        return np.array([[0.1, 0.9]])


def test_hybrid_predictor_shap_explanation(monkeypatch):
    engine = build_rule_engine()
    predictor = HybridMassnahmenPredictor(DummyPipelineWithClassifier(), engine)
    predictor.preprocessor_ = DummyPreprocessor()
    predictor.background_ = np.zeros((1, 2))
    predictor.feature_names_ = ["feat_a", "feat_b"]

    class DummyExplainer:
        def __init__(self, model, background):
            self.calls = [(model, background)]

        def shap_values(self, dense, check_additivity=False):  # noqa: D401
            return [
                np.array([[0.05, -0.1]]),
                np.array([[0.1, -0.4]]),
            ]

    class DummyShapModule:
        TreeExplainer = DummyExplainer

    monkeypatch.setitem(sys.modules, "shap", DummyShapModule)

    row = pd.Series({"Ampel": 0, "foo": 1})
    explanation = predictor._get_shap_explanation(row, top_n=2)

    assert explanation is not None
    assert explanation[0][0] == "feat_b"
    assert explanation[0][1] == -0.4
    assert explanation[1][0] == "feat_a"
    assert explanation[1][1] == 0.1
