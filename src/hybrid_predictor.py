"""Hybrid predictor combining business rules with ML model."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
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
        self.preprocessor_: Optional[Pipeline] = None
        self.background_: Optional[np.ndarray] = None
        self.feature_names_: Optional[Iterable[str]] = None
        self.label_encoder_ = None
        self._shap_explainer = None
        self._logger = logging.getLogger(__name__)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame containing prediction, confidence, and source."""

        if self.strategy != "rules_first":
            raise NotImplementedError("Currently only strategy='rules_first' is implemented")

        records = []

        for _, row in X.iterrows():
            prediction, confidence, source = self.rule_engine.evaluate(row)
            allowed_classes = getattr(self.rule_engine, "current_allowed_classes", None)

            if prediction is None:
                prediction, confidence, restricted = self._run_ml(row, allowed_classes=allowed_classes)
                source = (
                    f"ml_restricted_{row.get('Ampel', 'unknown')}" if restricted else "ml"
                )

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
        allowed_classes = getattr(self.rule_engine, "current_allowed_classes", None)

        explanation: Dict[str, Any] = {
            "prediction": prediction,
            "confidence": confidence,
            "source": source,
            "details": {},
        }

        if prediction is None:
            prediction, confidence, restricted = self._run_ml(row, allowed_classes=allowed_classes)
            explanation["prediction"] = prediction
            explanation["confidence"] = confidence
            explanation["source"] = "ml_restricted" if restricted else "ml"
            explanation["details"]["shap_top5"] = self._get_shap_explanation(row)
        elif source:
            explanation["details"]["matched_conditions"] = self._get_rule_conditions(source, row)

        return explanation

    def _run_ml(
        self, row: pd.Series, allowed_classes: Optional[List[str]] = None
    ) -> tuple[str, Optional[float], bool]:
        proba = self.ml_model.predict_proba(row.to_frame().T)[0]
        classes = getattr(self.ml_model, "classes_", None)
        if classes is None:
            classes = self.ml_model.predict(row.to_frame().T)
            return str(classes[0]), None, False
        if allowed_classes:
            allowed_set = {str(name) for name in allowed_classes}
            class_to_index = {str(cls): idx for idx, cls in enumerate(classes)}
            valid_indices = [class_to_index[name] for name in allowed_set if name in class_to_index]
            if valid_indices:
                restricted = proba[valid_indices]
                if restricted.sum() == 0:
                    restricted = np.full(len(valid_indices), 1.0 / len(valid_indices), dtype=float)
                else:
                    restricted = restricted / restricted.sum()
                best_local = valid_indices[int(restricted.argmax())]
                chosen_class = str(classes[best_local])
                chosen_conf = float(restricted.max())
                self._logger.info(
                    "hybrid_ml_restricted ampelfarbe=%s allowed=%s chosen=%s confidence=%.3f",
                    row.get("Ampel"),
                    list(allowed_set),
                    chosen_class,
                    chosen_conf,
                )
                return chosen_class, chosen_conf, True
            else:
                self._logger.warning(
                    "hybrid_ml_restriction_empty ampelfarbe=%s allowed=%s",
                    row.get("Ampel"),
                    list(allowed_set),
                )
        best_idx = int(proba.argmax())
        return str(classes[best_idx]), float(proba[best_idx]), False

    def _get_shap_explanation(self, row: pd.Series, top_n: int = 5) -> Optional[list[tuple[str, float]]]:
        if self.preprocessor_ is None or self.background_ is None:
            return None

        sample = row.to_frame().T
        transformed = self.preprocessor_.transform(sample)
        dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

        try:
            import shap  # noqa: WPS433 - optional heavy import inside function
        except ImportError:  # pragma: no cover - optional dependency
            return None

        if self._shap_explainer is None:
            classifier = getattr(self.ml_model, "named_steps", {}).get("classifier", None)
            if classifier is None:
                return None
            self._shap_explainer = shap.TreeExplainer(classifier, self.background_)

        shap_values = self._shap_explainer.shap_values(dense)
        if isinstance(shap_values, list):
            try:
                proba = self.ml_model.predict_proba(sample)
                class_index = int(np.argmax(proba[0])) if proba.ndim == 2 else 0
            except Exception:  # pragma: no cover - defensive
                class_index = 0
            class_index = max(0, min(class_index, len(shap_values) - 1))
            shap_values = shap_values[class_index]

        values = shap_values[0]
        feature_names = list(self.feature_names_) if self.feature_names_ is not None else list(row.index)
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
