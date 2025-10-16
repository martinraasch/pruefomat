"""Data structures for Maßnahme-Business-Rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import yaml


class RuleOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN = "in"
    NOT_IN = "not_in"


@dataclass
class SimpleCondition:
    field: str
    operator: RuleOperator
    value: Any


@dataclass
class BusinessRule:
    name: str
    priority: int
    condition_type: Literal["simple", "and", "or", "feature_lookup", "always"]
    conditions: List[SimpleCondition] = field(default_factory=list)
    action_field: str = ""
    action_value: str = ""
    confidence: Optional[float] = None
    lookup_keys: Optional[List[str]] = None
    historical_action: Optional[str] = None
    min_occurrences: Optional[int] = None


def _parse_operator(value: str) -> RuleOperator:
    try:
        return RuleOperator(value.lower())
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported rule operator: {value}") from exc


def _build_simple_conditions(nodes: Iterable[dict]) -> List[SimpleCondition]:
    conditions: List[SimpleCondition] = []
    for entry in nodes or []:
        field = str(entry.get("field", "")).strip()
        if not field:
            continue
        operator_raw = str(entry.get("operator", "equals")).strip()
        operator = _parse_operator(operator_raw)
        value = entry.get("value")
        conditions.append(SimpleCondition(field=field, operator=operator, value=value))
    return conditions


def _load_business_rules_structure(data: Dict[str, Any]) -> List[BusinessRule]:
    rules_section = data.get("business_rules", {})
    if isinstance(rules_section, list):
        rules_raw = rules_section
    else:
        rules_raw = rules_section.get("rules", []) if isinstance(rules_section, dict) else []

    rules: List[BusinessRule] = []
    for entry in rules_raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "rule")).strip() or "rule"
        priority = int(entry.get("priority", 100))
        condition_block = entry.get("condition", {}) or {}
        condition_type = str(condition_block.get("type", "always")).strip().lower()
        conditions = _build_simple_conditions(condition_block.get("rules", []))
        action = entry.get("action", {}) or {}
        set_field = str(action.get("set_field", "")).strip()
        action_value = action.get("value")
        confidence = entry.get("confidence")
        lookup_keys = condition_block.get("lookup_key")
        if isinstance(lookup_keys, str):
            lookup_keys = [lookup_keys]
        historical_action = condition_block.get("historical_action")
        min_occurrences = condition_block.get("min_occurrences")

        rules.append(
            BusinessRule(
                name=name,
                priority=priority,
                condition_type=condition_type,
                conditions=conditions,
                action_field=set_field,
                action_value="" if action_value is None else str(action_value),
                confidence=None if confidence is None else float(confidence),
                lookup_keys=list(lookup_keys) if lookup_keys else None,
                historical_action=str(historical_action) if historical_action is not None else None,
                min_occurrences=int(min_occurrences) if min_occurrences is not None else None,
            )
        )

    return sorted(rules, key=lambda rule: rule.priority)


def load_business_rules_from_file(path: str | Path) -> List[BusinessRule]:
    rule_path = Path(path)
    if not rule_path.exists():
        raise FileNotFoundError(f"Business rule configuration not found: {rule_path}")
    data = yaml.safe_load(rule_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Business rules YAML must define a mapping at the root level")
    return _load_business_rules_structure(data)
