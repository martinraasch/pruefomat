"""Data structures for Maßnahme-Business-Rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Literal, Optional


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
