import numpy as np
import pandas as pd

from src.business_rules import BusinessRule, RuleOperator, SimpleCondition
from src.rule_engine import RuleEngine
from src.hybrid_predictor import HybridMassnahmenPredictor


def _simple_rule(name: str, operator: RuleOperator, value, priority: int = 1):
    return BusinessRule(
        name=name,
        priority=priority,
        condition_type="simple",
        conditions=[SimpleCondition(field="Ampel", operator=operator, value=value)],
        action_field="Massnahme_2025",
        action_value="Rechnungsprüfung",
        confidence=1.0,
    )


def test_niedrig_betrag_gruene_ampel():
    rule = BusinessRule(
        name="niedrig_betrag",
        priority=1,
        condition_type="and",
        conditions=[
            SimpleCondition("Ampel", RuleOperator.EQUALS, 1),
            SimpleCondition("Betrag_parsed", RuleOperator.LESS_THAN, 50000),
        ],
        action_field="Massnahme_2025",
        action_value="Rechnungsprüfung",
        confidence=1.0,
    )
    engine = RuleEngine([rule])

    massnahme, confidence, _ = engine.evaluate(pd.Series({"Ampel": 1, "Betrag_parsed": 30000}))
    assert massnahme == "Rechnungsprüfung"
    assert confidence == 1.0

    massnahme_high, _, _ = engine.evaluate(pd.Series({"Ampel": 1, "Betrag_parsed": 60000}))
    assert massnahme_high is None


def test_rule_engine_priority_order():
    high_priority = _simple_rule("high", RuleOperator.EQUALS, 1, priority=1)
    low_priority = _simple_rule("low", RuleOperator.EQUALS, 1, priority=10)
    engine = RuleEngine([low_priority, high_priority])
    row = pd.Series({"Ampel": 1})

    prediction, _, source = engine.evaluate(row)

    assert source == "high"
    assert prediction == "Rechnungsprüfung"


def test_rule_operator_greater_equal():
    rule = BusinessRule(
        name="gte_rule",
        priority=1,
        condition_type="simple",
        conditions=[SimpleCondition("Betrag_parsed", RuleOperator.GREATER_THAN_OR_EQUAL, 50000)],
        action_field="Massnahme_2025",
        action_value="Freigabe gemäß Kompetenzkatalog",
        confidence=1.0,
    )
    engine = RuleEngine([rule])
    row = pd.Series({"Betrag_parsed": 75000})

    prediction, confidence, source = engine.evaluate(row)

    assert prediction == "Freigabe gemäß Kompetenzkatalog"
    assert confidence == 1.0
    assert source == "gte_rule"


def test_rule_engine_and_condition():
    rule = BusinessRule(
        name="and_rule",
        priority=1,
        condition_type="and",
        conditions=[
            SimpleCondition("Ampel", RuleOperator.EQUALS, 1),
            SimpleCondition("Betrag", RuleOperator.LESS_THAN, 50000),
        ],
        action_field="Massnahme_2025",
        action_value="Rechnungsprüfung",
        confidence=1.0,
    )
    engine = RuleEngine([rule])
    row = pd.Series({"Ampel": 1, "Betrag": 40000})

    prediction, _, source = engine.evaluate(row)

    assert prediction == "Rechnungsprüfung"
    assert source == "and_rule"


def test_rule_engine_or_condition():
    rule = BusinessRule(
        name="or_rule",
        priority=1,
        condition_type="or",
        conditions=[
            SimpleCondition("Ampel", RuleOperator.EQUALS, 2),
            SimpleCondition("Ampel", RuleOperator.EQUALS, 1),
        ],
        action_field="Massnahme_2025",
        action_value="Rechnungsprüfung",
        confidence=1.0,
    )
    engine = RuleEngine([rule])
    row = pd.Series({"Ampel": 1})

    prediction, _, source = engine.evaluate(row)

    assert prediction == "Rechnungsprüfung"
    assert source == "or_rule"


def test_historische_gutschrift_muster():
    historical_df = pd.DataFrame(
        {
            "BUK": ["A", "A", "A", "B"],
            "Debitor": ["100", "100", "100", "200"],
            "Massnahme_2025": ["Gutschrift", "Gutschrift", "Gutschrift", "Prüfung"],
        }
    )

    rule = BusinessRule(
        name="gutschrift_lookup",
        priority=2,
        condition_type="feature_lookup",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="Gutschrift",
        confidence=0.9,
        lookup_keys=["BUK", "Debitor"],
        historical_action="Gutschrift",
        min_occurrences=2,
    )

    engine = RuleEngine([rule], historical_data=historical_df)

    prediction_match, confidence_match, _ = engine.evaluate(pd.Series({"BUK": "A", "Debitor": "100"}))
    prediction_nomatch, _, _ = engine.evaluate(pd.Series({"BUK": "B", "Debitor": "200"}))

    assert prediction_match == "Gutschrift"
    assert confidence_match == 0.9
    assert prediction_nomatch is None


def test_rule_engine_ml_fallback_signal():
    fallback_rule = BusinessRule(
        name="ml_fallback",
        priority=1,
        condition_type="always",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="ML_PREDICTION",
        confidence=None,
    )
    engine = RuleEngine([fallback_rule])
    row = pd.Series({})

    prediction, confidence, source = engine.evaluate(row)

    assert prediction is None
    assert confidence is None
    assert source == "ml_fallback"


def test_rule_engine_handles_missing_values():
    rule = _simple_rule("rule_eq", RuleOperator.EQUALS, 1)
    engine = RuleEngine([rule])
    row = pd.Series({"Ampel": np.nan})

    prediction, confidence, source = engine.evaluate(row)

    assert prediction is None
    assert confidence is None
    assert source == "no_match"


def test_rule_engine_missing_lookup_columns():
    rule = BusinessRule(
        name="lookup_missing",
        priority=1,
        condition_type="feature_lookup",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="Gutschrift",
        confidence=0.9,
        lookup_keys=["BUK", "Debitor"],
        historical_action="Gutschrift",
        min_occurrences=1,
    )

    history = pd.DataFrame({"BUK": ["X"], "Debitor": ["Y"], "Massnahme_2025": ["Gutschrift"]})
    engine = RuleEngine([rule], historical_data=history)

    row = pd.Series({"BUK": "X"})  # missing Debitor
    prediction, _, source = engine.evaluate(row)

    assert prediction is None
    assert source == "no_match"
def test_hoher_betrag_gruene_ampel():
    rule = BusinessRule(
        name="hoher_betrag",
        priority=1,
        condition_type="and",
        conditions=[
            SimpleCondition("Ampel", RuleOperator.EQUALS, 1),
            SimpleCondition("Betrag_parsed", RuleOperator.GREATER_THAN_OR_EQUAL, 50000),
        ],
        action_field="Massnahme_2025",
        action_value="Freigabe gemäß Kompetenzkatalog",
        confidence=1.0,
    )
    engine = RuleEngine([rule])

    assert engine.evaluate(pd.Series({"Ampel": 1, "Betrag_parsed": 50000}))[0] == "Freigabe gemäß Kompetenzkatalog"
    assert engine.evaluate(pd.Series({"Ampel": 1, "Betrag_parsed": 50001}))[0] == "Freigabe gemäß Kompetenzkatalog"
    assert engine.evaluate(pd.Series({"Ampel": 1, "Betrag_parsed": 49999}))[0] is None
def test_ml_allowed_classes_restriction():
    class MockModel:
        classes_ = np.array(
            [
                "Rechnungsprüfung",
                "telefonische rechnungsbestätigung (vorgelagert)",
                "telefonische Lieferbestätigung (vorgelagert)",
            ]
        )

        def predict_proba(self, X):  # noqa: N802
            return np.array([[0.05, 0.15, 0.8]])

        def predict(self, X):  # noqa: N802
            return np.array(["Rechnungsprüfung"] * len(X))

    rule = BusinessRule(
        name="gelbe_ampel",
        priority=1,
        condition_type="simple",
        conditions=[SimpleCondition("Ampel", RuleOperator.EQUALS, 2)],
        action_field="Massnahme_2025",
        action_value="ML_PREDICTION",
        confidence=None,
        ml_allowed_classes=[
            "Telefonische Rechnungsbestätigung (Vorgelagert)",
            "Rechnungsprüfung",
        ],
    )
    engine = RuleEngine([rule])
    predictor = HybridMassnahmenPredictor(MockModel(), engine)

    result = predictor.predict(pd.DataFrame([{"Ampel": 2}]))
    chosen = result.loc[0, "prediction"]
    normalized = chosen.strip().lower()
    assert normalized == "telefonische rechnungsbestätigung (vorgelagert)"


def test_historical_lookup_requires_columns():
    history = pd.DataFrame({"BUK": ["A"]})
    engine = RuleEngine([], historical_data=history)
    assert engine.historical_lookup == {}


def test_check_condition_without_conditions_returns_false():
    rule = BusinessRule(
        name="empty_simple",
        priority=1,
        condition_type="simple",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="Rechnungsprüfung",
    )
    engine = RuleEngine([rule])
    assert engine._check_condition(rule, pd.Series()) is False


def test_check_condition_unknown_type_returns_false():
    rule = BusinessRule(
        name="unknown_type",
        priority=1,
        condition_type="unknown",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="Rechnungsprüfung",
    )
    engine = RuleEngine([rule])
    assert engine._check_condition(rule, pd.Series()) is False


def test_coerce_amount_handles_strings_and_invalid_values():
    engine = RuleEngine([])
    assert engine._coerce_amount("1.234,50") == 1234.5
    assert engine._coerce_amount("abc") is None
    assert engine._coerce_amount(np.nan) is None


def test_check_simple_condition_various_operators():
    engine = RuleEngine([])
    row = pd.Series({"Ampel": 2, "Betrag_parsed": 100})

    assert engine._check_simple_condition(
        SimpleCondition("Ampel", RuleOperator.NOT_EQUALS, 1),
        row,
    )
    assert engine._check_simple_condition(
        SimpleCondition("Betrag_parsed", RuleOperator.LESS_THAN_OR_EQUAL, 100),
        row,
    )
    assert engine._check_simple_condition(
        SimpleCondition("Betrag_parsed", RuleOperator.GREATER_THAN, 50),
        row,
    )
    assert engine._check_simple_condition(
        SimpleCondition("Ampel", RuleOperator.IN, [1, 2, 3]),
        row,
    )
    assert engine._check_simple_condition(
        SimpleCondition("Ampel", RuleOperator.NOT_IN, [3, 4]),
        row,
    )


def test_compare_numeric_invalid_inputs():
    assert RuleEngine._compare_numeric("a", "b", lambda a, b: a < b) is False


def test_compare_equality_branches():
    obj = object()
    assert RuleEngine._compare_equality(obj, obj)
    assert RuleEngine._compare_equality(np.nan, 1) is False
    assert RuleEngine._compare_equality("test", "test")
    assert RuleEngine._compare_equality("5", 5)
    assert RuleEngine._compare_equality("A", "a")


def test_in_collection_variants():
    assert RuleEngine._in_collection("x", None, True) is False
    assert RuleEngine._in_collection("x", None, False) is True
    assert RuleEngine._in_collection("x", "x", True) is True
    assert RuleEngine._in_collection("x", "x", False) is False


def test_check_lookup_condition_edge_cases():
    engine = RuleEngine([])
    rule = BusinessRule(
        name="lookup",
        priority=1,
        condition_type="feature_lookup",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="Gutschrift",
        lookup_keys=[],
        historical_action="Gutschrift",
        min_occurrences=1,
    )

    assert engine._check_lookup_condition(rule, pd.Series({})) is False

    rule.lookup_keys = ["BUK"]

    class BadRow:
        def get(self, _):
            raise KeyError("BUK")

    assert engine._check_lookup_condition(rule, BadRow()) is False

    assert engine._check_lookup_condition(rule, pd.Series({"BUK": None})) is False

    engine.historical_lookup = {("A",): {"Gutschrift": 2}}
    assert engine._check_lookup_condition(rule, pd.Series({"BUK": "A"})) is True
