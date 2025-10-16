"""Rule engine for Maßnahme-Business-Logic."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .business_rules import BusinessRule, RuleOperator, SimpleCondition


class RuleEngine:
    """Evaluate ordered business rules for Maßnahme predictions."""

    def __init__(
        self,
        rules: Iterable[BusinessRule],
        historical_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.rules: List[BusinessRule] = sorted(list(rules), key=lambda rule: rule.priority)
        self.historical_lookup = self._build_historical_lookup(historical_data)
        self.current_allowed_classes: Optional[List[str]] = None

    @staticmethod
    def _build_historical_lookup(
        df: Optional[pd.DataFrame],
    ) -> Dict[Tuple, Dict[str, int]]:
        """Return lookup of (BUK, Debitor) → Maßnahme counts."""

        if df is None or df.empty:
            return {}

        required_columns = {"BUK", "Debitor", "Massnahme_2025"}
        if not required_columns.issubset(df.columns):
            return {}

        lookup: Dict[Tuple, Dict[str, int]] = {}
        grouped = (
            df.groupby(["BUK", "Debitor", "Massnahme_2025"], dropna=False)
            .size()
            .reset_index(name="count")
        )

        for _, row in grouped.iterrows():
            key = (row["BUK"], row["Debitor"])
            lookup.setdefault(key, {})[str(row["Massnahme_2025"])] = int(row["count"])

        return lookup

    def evaluate(self, row: pd.Series) -> Tuple[Optional[str], Optional[float], str]:
        """Evaluate ordered rules for a single row."""

        self.current_allowed_classes = None
        for rule in self.rules:
            if self._check_condition(rule, row):
                if rule.action_value == "ML_PREDICTION":
                    self.current_allowed_classes = rule.ml_allowed_classes
                    return None, None, rule.name
                return rule.action_value, rule.confidence, rule.name

        return None, None, "no_match"

    def _check_condition(self, rule: BusinessRule, row: pd.Series) -> bool:
        condition_type = rule.condition_type

        if condition_type == "always":
            return True

        if condition_type == "simple":
            if not rule.conditions:
                return False
            return self._check_simple_condition(rule.conditions[0], row)

        if condition_type == "and":
            return all(self._check_simple_condition(cond, row) for cond in rule.conditions)

        if condition_type == "or":
            return any(self._check_simple_condition(cond, row) for cond in rule.conditions)

        if condition_type == "feature_lookup":
            return self._check_lookup_condition(rule, row)

        return False

    def _check_simple_condition(self, condition: SimpleCondition, row: pd.Series) -> bool:
        field_value = row.get(condition.field)

        if pd.isna(field_value):
            return False

        operator = condition.operator
        target_value = condition.value

        if operator == RuleOperator.EQUALS:
            return bool(field_value == target_value)
        if operator == RuleOperator.NOT_EQUALS:
            return bool(field_value != target_value)
        if operator == RuleOperator.LESS_THAN:
            return self._compare_numeric(field_value, target_value, lambda a, b: a < b)
        if operator == RuleOperator.LESS_THAN_OR_EQUAL:
            return self._compare_numeric(field_value, target_value, lambda a, b: a <= b)
        if operator == RuleOperator.GREATER_THAN:
            return self._compare_numeric(field_value, target_value, lambda a, b: a > b)
        if operator == RuleOperator.GREATER_THAN_OR_EQUAL:
            return self._compare_numeric(field_value, target_value, lambda a, b: a >= b)
        if operator == RuleOperator.IN:
            return self._in_collection(field_value, target_value, True)
        if operator == RuleOperator.NOT_IN:
            return self._in_collection(field_value, target_value, False)

        return False

    @staticmethod
    def _compare_numeric(left: object, right: object, comparer) -> bool:
        try:
            left_val = float(left)
            right_val = float(right)
        except (TypeError, ValueError):
            return False
        return comparer(left_val, right_val)

    @staticmethod
    def _in_collection(value: object, container: object, positive: bool) -> bool:
        if container is None:
            return not positive
        if not isinstance(container, (list, tuple, set, frozenset)):
            container = [container]
        result = value in container
        return result if positive else not result

    def _check_lookup_condition(self, rule: BusinessRule, row: pd.Series) -> bool:
        lookup_keys = rule.lookup_keys or []
        if not lookup_keys:
            return False

        try:
            key = tuple(row.get(field) for field in lookup_keys)
        except KeyError:
            return False

        if None in key or any(pd.isna(part) for part in key):
            return False

        action_counts = self.historical_lookup.get(key)
        if not action_counts:
            return False

        occurrences = action_counts.get(str(rule.historical_action), 0)
        threshold = rule.min_occurrences or 0
        return occurrences >= threshold
