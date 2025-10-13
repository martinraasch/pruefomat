import random
from pathlib import Path

import pandas as pd

from data_synthethizer.config import load_config
from data_synthethizer.constraints import (
    BoundsConstraint,
    ConditionalProbabilityConstraint,
    DependencyConstraint,
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
