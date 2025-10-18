"""Hybrid predictor combining business rules with ML model."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .rule_engine import RuleEngine
from .utils import align_by_length, ensure_1d


class HybridMassnahmenPredictor:
    """Predict Maßnahme values by combining deterministic rules with ML output."""

    @staticmethod
    def _normalise_class_name(name: object) -> str:
        if name is None:
            return ""
        text = str(name).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("( ", "(")
        return text

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
        self._last_predict_proba: Optional[np.ndarray] = None
        self._last_predict_classes: Optional[np.ndarray] = None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame containing prediction, confidence, and source."""

        if self.strategy != "rules_first":
            raise NotImplementedError("Currently only strategy='rules_first' is implemented")

        records = []

        for index, row in X.iterrows():
            prediction, confidence, source = self.rule_engine.evaluate(row)
            allowed_classes = getattr(self.rule_engine, "current_allowed_classes", None)
            restricted = False

            if prediction is None:
                prediction, confidence, restricted = self._run_ml(
                    row,
                    allowed_classes=allowed_classes,
                )
                source = (
                    f"ml_restricted_{row.get('Ampel', 'unknown')}"
                    if restricted
                    else "ml"
                )

            records.append(
                {
                    "row_index": index,
                    "prediction": prediction,
                    "confidence": confidence,
                    "source": source,
                }
            )

        if not records:
            return pd.DataFrame(columns=["prediction", "confidence", "source"])

        frame = pd.DataFrame.from_records(records).set_index("row_index")
        return frame[["prediction", "confidence", "source"]]

    def explain(self, row: pd.Series) -> Dict[str, Any]:
        """Provide per-row explanation for hybrid prediction."""

        prediction, confidence, source = self.rule_engine.evaluate(row)
        allowed_classes_raw = getattr(self.rule_engine, "current_allowed_classes", None)
        allowed_classes = [str(item) for item in allowed_classes_raw or []]

        explanation: Dict[str, Any] = {
            "prediction": prediction,
            "confidence": confidence,
            "source": source,
            "details": {},
        }

        if prediction is None:
            prediction, confidence, restricted = self._run_ml(
                row,
                allowed_classes=allowed_classes,
            )
            explanation["prediction"] = prediction
            explanation["confidence"] = confidence
            explanation["source"] = "ml_restricted" if restricted else "ml"
            explanation["details"]["ml_context"] = {
                "rule": source,
                "restricted": bool(restricted and allowed_classes),
                "allowed_classes": allowed_classes,
            }
            explanation["details"]["shap_top5"] = self._get_shap_explanation(row)
            if self._last_predict_proba is not None and self._last_predict_classes is not None:
                probas = self._last_predict_proba.astype(float)
                classes = [str(cls) for cls in self._last_predict_classes]
                explanation["details"]["probabilities"] = dict(zip(classes, probas))
        elif source:
            explanation["details"]["matched_conditions"] = self._get_rule_conditions(source, row)

        return explanation

    def _run_ml(
        self, row: pd.Series, allowed_classes: Optional[List[str]] = None
    ) -> tuple[str, Optional[float], bool]:
        proba = self.ml_model.predict_proba(row.to_frame().T)[0]
        classes = getattr(self.ml_model, "classes_", None)
        self._last_predict_proba = np.array(proba, copy=True)
        self._last_predict_classes = np.array(classes, copy=True) if classes is not None else None
        if classes is None:
            classes = self.ml_model.predict(row.to_frame().T)
            return str(classes[0]), None, False
        if allowed_classes:
            allowed_set = {str(name) for name in allowed_classes}
            allowed_normalized = {self._normalise_class_name(name): str(name) for name in allowed_set}
            class_to_index = {
                self._normalise_class_name(cls): idx for idx, cls in enumerate(classes)
            }
            valid_indices = [
                class_to_index[norm]
                for norm in allowed_normalized
                if norm and norm in class_to_index
            ]
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
                    "hybrid_ml_restriction_empty ampelfarbe=%s allowed=%s available_normalized=%s",
                    row.get("Ampel"),
                    list(allowed_set),
                    list(class_to_index.keys()),
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

        shap_values = self._compute_shap_values(shap, dense, sample)
        if shap_values is None:
            return None

        values = ensure_1d(shap_values)
        feature_names = self._feature_names(row)
        feature_names, values = align_by_length(feature_names, values)

        pairs = [(name, float(val)) for name, val in zip(feature_names, values)]
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        return pairs[:top_n]

    def _feature_names(self, row: pd.Series) -> List[str]:
        if self.feature_names_ is not None:
            return list(self.feature_names_)
        return list(row.index)

    def _compute_shap_values(self, shap_module, dense: Any, sample: pd.DataFrame):
        explainer = self._ensure_shap_explainer(shap_module)
        if explainer is None:
            return None

        try:
            shap_values = explainer.shap_values(dense, check_additivity=False)
        except Exception as exc:  # pragma: no cover - graceful fallback for unsupported estimators
            self._logger.warning("shap_values_failed", exc_info=exc)
            return None
        if isinstance(shap_values, list):
            class_index = self._dominant_class_index(sample)
            class_index = max(0, min(class_index, len(shap_values) - 1))
            shap_values = shap_values[class_index]
        return shap_values

    def _dominant_class_index(self, sample: pd.DataFrame) -> int:
        try:
            proba = self.ml_model.predict_proba(sample)
            if getattr(proba, "ndim", 1) >= 2:
                return int(np.argmax(proba[0]))
        except Exception:  # pragma: no cover - defensive
            pass
        return 0

    def _ensure_shap_explainer(self, shap_module):
        if self._shap_explainer is not None:
            return self._shap_explainer

        classifier = getattr(self.ml_model, "named_steps", {}).get("classifier", None)
        if classifier is None:
            return None

        try:
            self._shap_explainer = shap_module.TreeExplainer(classifier, self.background_)
        except Exception as exc:  # pragma: no cover - defensive guard for unsupported models
            self._logger.warning("shap_tree_explainer_unavailable", exc_info=exc)
            self._shap_explainer = None
        return self._shap_explainer

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
