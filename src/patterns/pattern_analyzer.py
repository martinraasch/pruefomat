"""Generic conditional probability pattern analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

from config_loader import AppConfig
from .feature_transformations import GeneratedFeature


@dataclass
class PatternInsight:
    feature_name: str
    feature_description: str
    feature_value: str
    feature_value_label: str
    target_value: str
    probability: float
    baseline_probability: float
    lift: float
    delta: float
    support: int
    population: int
    p_value: float
    chi2: float

    @property
    def support_ratio(self) -> float:
        return self.support / self.population if self.population else 0.0


class ConditionalProbabilityAnalyzer:
    """Analyze conditional probabilities across generated features."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config
        pattern_cfg = getattr(config, "pattern_analysis", None)
        self.min_lift = getattr(pattern_cfg, "min_lift", 1.15) if pattern_cfg else 1.15
        self.min_samples = getattr(pattern_cfg, "min_samples", 20) if pattern_cfg else 20
        self.max_p_value = getattr(pattern_cfg, "max_p_value", 0.05) if pattern_cfg else 0.05
        self.max_feature_values = getattr(pattern_cfg, "max_feature_values", 30) if pattern_cfg else 30
        self.top_n = getattr(pattern_cfg, "top_n", 40) if pattern_cfg else 40

    def analyze(
        self,
        features: Iterable[GeneratedFeature],
        target: pd.Series,
        target_mapping: Optional[Dict[int, str]] = None,
    ) -> List[PatternInsight]:
        target_series = target.astype(int).reset_index(drop=True)
        population = int(target_series.shape[0])
        if population == 0:
            return []

        value_counts = target_series.value_counts()
        baseline_probs = value_counts / population

        insights: List[PatternInsight] = []
        for generated in features:
            series = generated.series.reset_index(drop=True)
            if series.shape[0] != population:
                continue
            series = series.fillna("unbekannt").astype(str)
            if series.nunique(dropna=False) > self.max_feature_values:
                continue

            crosstab = pd.crosstab(series, target_series, dropna=False)
            for feature_value, row in crosstab.iterrows():
                support = int(row.sum())
                if support < self.min_samples:
                    continue
                for target_value, count in row.items():
                    probability = count / support if support else 0.0
                    baseline = baseline_probs.get(target_value, 0.0)
                    if baseline <= 0 and probability <= 0:
                        continue
                    lift = (probability / baseline) if baseline > 0 else float("inf")
                    delta = probability - baseline

                    if not self._passes_lift_threshold(lift, delta):
                        continue

                    chi2, p_value = self._chi_square_test(row.values, value_counts.values, target_value, count)
                    if not np.isfinite(p_value):
                        p_value = 1.0

                    if self.max_p_value and p_value > self.max_p_value:
                        continue

                    formatted_target = self._map_target(target_value, target_mapping)
                    value_label = generated.value_describer(feature_value)
                    insight = PatternInsight(
                        feature_name=generated.name,
                        feature_description=generated.description,
                        feature_value=str(feature_value),
                        feature_value_label=str(value_label),
                        target_value=formatted_target,
                        probability=probability,
                        baseline_probability=baseline,
                        lift=lift,
                        delta=delta,
                        support=support,
                        population=population,
                        p_value=float(p_value),
                        chi2=float(chi2),
                    )
                    insights.append(insight)

        insights.sort(key=lambda i: (abs(i.delta), i.support), reverse=True)
        if self.top_n:
            insights = insights[: self.top_n]
        return insights

    def _passes_lift_threshold(self, lift: float, delta: float) -> bool:
        threshold = max(self.min_lift, 1.0)
        if lift >= threshold:
            return True
        if lift <= (1.0 / threshold) and not isclose(lift, 0.0):
            return True
        if abs(delta) >= 0.05:  # 5 percentage points fallback
            return True
        return False

    @staticmethod
    def _chi_square_test(row_counts: np.ndarray, global_counts: np.ndarray, target_value, count) -> tuple[float, float]:
        """Chi-square for selected value vs rest across target categories."""

        observed = np.vstack([row_counts, global_counts - row_counts])
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            total = observed.sum()
            expected = row_sums @ col_sums / total if total else np.zeros_like(observed)
            mask = expected > 0
        chi2_stat = np.where(mask, (observed - expected) ** 2 / expected, 0.0).sum()
        dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
        if dof <= 0:
            return float(chi2_stat), 1.0
        from scipy.stats import chi2 as chi2_dist

        p_value = chi2_dist.sf(chi2_stat, dof)
        return float(chi2_stat), float(p_value)

    @staticmethod
    def _map_target(value: int, mapping: Optional[Dict[int, str]]) -> str:
        if mapping and value in mapping:
            return mapping[value]
        return str(value)


__all__ = ["PatternInsight", "ConditionalProbabilityAnalyzer"]
