"""Hybrid predictor combining business rules with ML model."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from .rule_engine import RuleEngine


class HybridMassnahmenPredictor:
    """Predict Maßnahme values by combining deterministic rules with ML output."""

    def __init__(
        self,
        ml_model: Pipeline,
        rule_engine: RuleEngine,
        strategy: Literal["rules_first", "ml_first", "ensemble"] = "rules_first",
    ) -> None:
        self.ml_model = ml_model
        self.rule_engine = rule_engine
        self.strategy = strategy

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame containing prediction, confidence, and source."""

        if self.strategy != "rules_first":
            raise NotImplementedError("Currently only strategy='rules_first' is implemented")

        records = []

        for _, row in X.iterrows():
            prediction, confidence, source = self.rule_engine.evaluate(row)

            if prediction is None:
                prediction, confidence = self._run_ml(row)
                source = "ml"

            records.append(
                {
                    "prediction": prediction,
                    "confidence": confidence,
                    "source": source,
                }
            )

        return pd.DataFrame.from_records(records)

    def explain(self, row: pd.Series) -> Dict[str, Any]:
        """Provide per-row explanation for hybrid prediction."""

        prediction, confidence, source = self.rule_engine.evaluate(row)

        explanation: Dict[str, Any] = {
            "prediction": prediction,
            "confidence": confidence,
            "source": source,
            "details": {},
        }

        if prediction is None:
            prediction, confidence = self._run_ml(row)
            explanation["prediction"] = prediction
            explanation["confidence"] = confidence
            explanation["source"] = "ml"
            explanation["details"]["shap_top5"] = self._get_shap_explanation(row)
        elif source:
            explanation["details"]["matched_conditions"] = self._get_rule_conditions(source, row)

        return explanation

    def _run_ml(self, row: pd.Series) -> tuple[str, Optional[float]]:
        proba = self.ml_model.predict_proba(row.to_frame().T)[0]
        classes = getattr(self.ml_model, "classes_", None)
        if classes is None:
            classes = self.ml_model.predict(row.to_frame().T)
            return str(classes[0]), None
        best_idx = int(proba.argmax())
        return str(classes[best_idx]), float(proba[best_idx])

    def _get_shap_explanation(self, row: pd.Series, top_n: int = 5) -> Optional[list[tuple[str, float]]]:
        explainer = getattr(self.ml_model, "explainer_", None)
        if explainer is None:
            return None
        shap_values = explainer.shap_values(row.to_frame().T)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        feature_names = getattr(self.ml_model, "feature_names_in_", row.index.tolist())
        values = shap_values[0]
        pairs = list(zip(feature_names, values))
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        return pairs[:top_n]

    def _get_rule_conditions(self, rule_name: str, row: pd.Series) -> Optional[list[Dict[str, Any]]]:
        rule = next((rule for rule in self.rule_engine.rules if rule.name == rule_name), None)
        if rule is None:
            return None

        matched = []
        for condition in rule.conditions:
            result = self.rule_engine._check_simple_condition(condition, row)  # pylint: disable=protected-access
            matched.append(
                {
                    "field": condition.field,
                    "operator": condition.operator.value,
                    "target": condition.value,
                    "matched": result,
                    "value": row.get(condition.field),
                }
            )
        return matched

