"""Natural language formatting for discovered pattern insights."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .pattern_analyzer import PatternInsight


class InsightFormatter:
    """Render pattern insights into human readable statements."""

    def __init__(
        self,
        target_name: Optional[str] = None,
        template: str = "{change:+.1f} %-Punkte ({lift:.2f}x) wahrscheinlicher, dass {outcome}, wenn {condition} (Support: {support_ratio:.1%}, p={p_value:.3f})",
    ) -> None:
        self.target_name = target_name
        self.template = template

    def format_insight(self, insight: PatternInsight) -> str:
        condition = f"{insight.feature_description} ({insight.feature_value_label})"
        if self.target_name:
            outcome = f"{self.target_name} = {insight.target_value}"
        else:
            outcome = insight.target_value
        change = (insight.delta * 100.0)
        support_ratio = insight.support_ratio
        return self.template.format(
            change=change,
            lift=insight.lift,
            outcome=outcome,
            condition=condition,
            support_ratio=support_ratio,
            p_value=insight.p_value,
        )

    def format_many(self, insights: Iterable[PatternInsight]) -> List[str]:
        return [self.format_insight(insight) for insight in insights]


__all__ = ["InsightFormatter"]
