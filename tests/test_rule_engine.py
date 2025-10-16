import pandas as pd

from src.business_rules import BusinessRule, RuleOperator, SimpleCondition
from src.rule_engine import RuleEngine


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


def test_rule_engine_simple_match():
    rule = _simple_rule("rule_eq", RuleOperator.EQUALS, 1)
    engine = RuleEngine([rule])
    row = pd.Series({"Ampel": 1})

    prediction, confidence, source = engine.evaluate(row)

    assert prediction == "Rechnungsprüfung"
    assert confidence == 1.0
    assert source == "rule_eq"


def test_rule_engine_priority_order():
    high_priority = _simple_rule("high", RuleOperator.EQUALS, 1, priority=1)
    low_priority = _simple_rule("low", RuleOperator.EQUALS, 1, priority=10)
    engine = RuleEngine([low_priority, high_priority])
    row = pd.Series({"Ampel": 1})

    prediction, _, source = engine.evaluate(row)

    assert source == "high"
    assert prediction == "Rechnungsprüfung"


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


def test_rule_engine_lookup_condition():
    lookup_rule = BusinessRule(
        name="lookup",
        priority=1,
        condition_type="feature_lookup",
        conditions=[],
        action_field="Massnahme_2025",
        action_value="Gutschrift",
        confidence=0.9,
        lookup_keys=["BUK", "Debitor"],
        historical_action="Gutschrift",
        min_occurrences=2,
    )

    history = pd.DataFrame(
        {
            "BUK": ["100", "100", "200"],
            "Debitor": ["A", "A", "B"],
            "Massnahme_2025": ["Gutschrift", "Gutschrift", "Rechnungsprüfung"],
        }
    )

    engine = RuleEngine([lookup_rule], historical_data=history)
    row = pd.Series({"BUK": "100", "Debitor": "A"})

    prediction, confidence, source = engine.evaluate(row)

    assert prediction == "Gutschrift"
    assert confidence == 0.9
    assert source == "lookup"


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
