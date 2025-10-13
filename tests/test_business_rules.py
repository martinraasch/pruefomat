import random
from pathlib import Path

import pandas as pd

from data_synthethizer.config import load_config
from app import _apply_bias_patch, _parse_business_rules_block
from data_synthethizer.constraints import (
    BoundsConstraint,
    ConditionalProbabilityConstraint,
    DependencyConstraint,
    is_round_amount,
    weekday,
    build_default_constraints,
    apply_dependency_rules,
)


def test_bounds_constraint_clamps_and_validates():
    rule = BoundsConstraint(
        column="Betrag",
        condition="value >= 100 and value <= 200",
        min_value=100.0,
        min_inclusive=True,
        max_value=200.0,
        max_inclusive=True,
        label="Range",
    )
    row = {"Betrag": 50}
    rule.apply(row)
    assert 100.0 <= row["Betrag"] <= 200.0
    assert rule.validate(row) is None


def test_conditional_probability_constraint_applies_choice():
    random.seed(0)
    constraint = ConditionalProbabilityConstraint(
        label="Ampel",
        condition="Betrag > 50000",
        target_column="Ampel",
        choices=[2, 3],
        probability=1.0,
    )
    row = {"Betrag": 60000, "Ampel": 1}
    constraint.apply(row)
    assert row["Ampel"] in {2, 3}


def test_dependency_constraint_evaluates_formula():
    constraint = DependencyConstraint(
        target_column="Fällig",
        depends_on=["Belegdatum"],
        formula="Belegdatum + timedelta(days=30)",
    )
    row = {"Belegdatum": pd.Timestamp("2024-01-01")}
    constraint.apply(row)
    assert row["Fällig"] == pd.Timestamp("2024-01-31")


def test_build_default_constraints_loads_rules():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "data_synthethizer" / "configs" / "default.yaml"
    config = load_config(config_path)
    engine = build_default_constraints(config.business_rules)
    assert any(c.name == "bounds" for c in engine.rules)
    assert any(c.name == "dependency" for c in engine.rules)


def test_apply_dependency_rules_converts_ampel():
    frame = pd.DataFrame({'Ampel': ['Grün', 'Gelb', 'Rot']})
    dependencies = {
        'Ampel': {
            'depends_on': ['Ampel'],
            'formula': "1 if Ampel == 'Grün' else 2 if Ampel == 'Gelb' else 3",
        }
    }
    out = apply_dependency_rules(frame, dependencies)
    assert out['Ampel'].tolist() == [1, 2, 3]


def test_weekday_helper_handles_timestamp():
    result = weekday(pd.Timestamp("2024-01-01"))  # Monday
    assert result == 0


def test_is_round_amount_helper():
    assert is_round_amount(120.0)
    assert is_round_amount("10.50", decimals=2)
    assert not is_round_amount(12.345, decimals=2)


def test_apply_bias_patch_merges_without_duplicates():
    base_yaml = """
business_rules:
  rules:
    - name: Baseline
      columns: ["Ampel"]
      condition: "value in [1, 2, 3]"
"""
    data, block = _parse_business_rules_block(base_yaml)
    patch = {
        "rules": [
            {
                "name": "Bias Ampel Montag",
                "columns": ["Belegdatum", "Ampel"],
                "condition": "weekday(Belegdatum) == 0",
            },
            {
                "name": "Bias Ampel Montag",
                "columns": ["Belegdatum", "Ampel"],
                "condition": "weekday(Belegdatum) == 1",
            },
        ],
        "dependencies": {
            "Flag": {
                "depends_on": ["Ampel"],
                "formula": "1",
            }
        },
    }
    added_rules, added_dependencies = _apply_bias_patch(data, block, patch)
    assert len(added_rules) == 2
    assert added_rules[0]["name"] == "Bias Ampel Montag"
    # duplicate name should receive suffix
    assert added_rules[1]["name"].startswith("Bias Ampel Montag (")
    assert "Flag" in added_dependencies
    all_names = [rule["name"] for rule in block["rules"]]
    assert len(all_names) == 3
    assert len(set(all_names)) == 3
