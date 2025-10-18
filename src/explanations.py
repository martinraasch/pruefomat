"""Utilities for human-readable explanations of rule and ML predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.hybrid_predictor import HybridMassnahmenPredictor

CONFIDENCE_LABELS = [
    (0.95, "Sehr hoch"),
    (0.8, "Hoch"),
    (0.6, "Mittel"),
    (0.0, "Niedrig"),
]


def format_confidence(probability: Optional[float]) -> str:
    """Return textual label and percentage for a probability."""

    if probability is None or np.isnan(probability):
        probability = 1.0
    probability = max(0.0, min(1.0, float(probability)))
    for threshold, label in CONFIDENCE_LABELS:
        if probability >= threshold:
            percent = f"{probability * 100:.0f}%"
            return f"{label} ({percent})"
    return f"Niedrig ({probability * 100:.0f}%)"


def _format_currency(value: Optional[float]) -> str:
    try:
        return f"{float(value):,.2f}€".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "n/a"


def humanize_shap_feature(
    feature_name: str,
    feature_value: object,
    shap_value: float,
    row_data: pd.Series,
) -> str:
    """Translate a SHAP feature contribution into human-readable text."""

    feature_name_lower = feature_name.lower()
    base_name = feature_name_lower.split("__")[-1]
    positive = shap_value >= 0

    if "betrag" in feature_name_lower or "betrag" in base_name:
        amount_value = row_data.get("Betrag_parsed", feature_value)
        try:
            amount_value = float(amount_value)
        except (TypeError, ValueError):
            amount_value = None
        amount_text = _format_currency(amount_value)
        if positive:
            return f"Betrag von {amount_text} → Erhöht die Prüfintensität"
        return f"Betrag von {amount_text} → Erlaubt einfachere Prüfung"

    if "ampel" in feature_name_lower or "ampel" in base_name:
        ampel_map = {1: "Grün", 2: "Gelb", 3: "Rot"}
        try:
            color = ampel_map.get(int(feature_value), str(feature_value))
        except (TypeError, ValueError):
            color = str(feature_value)
        return f"Ampel-Farbe: {color} → Steuert erlaubte Maßnahmen-Kategorien"

    if feature_name_lower.startswith("hist_") or base_name.startswith("hist_"):
        return "Historie für diese BUK/Debitor-Kombination beeinflusst die Wahl"

    if "success_rate" in feature_name_lower or "success_rate" in base_name:
        try:
            percent = float(feature_value) * 100
        except (TypeError, ValueError):
            percent = 0.0
        return f"Erfolgsquote bei diesem Lieferanten: {percent:.0f}%"

    if "notes_text_tfidf" in feature_name_lower or "notes_text_tfidf" in base_name:
        token = feature_name.split("__")[-1].replace("_", " ")
        return f"Schlüsselwörter in Hinweisen: '{token}'"

    return f"{feature_name}: {feature_value}".strip()


def _format_condition_line(condition: Dict[str, object]) -> str:
    matched = bool(condition.get("matched"))
    status = "✅" if matched else "❌"
    field = condition.get("field", "")
    operator = condition.get("operator", "=")
    target = condition.get("target")
    value = condition.get("value")
    return f"• {status} {field} {operator} (Soll: {target} | Ist: {value})"


def format_rule_explanation(data: Dict[str, object]) -> str:
    """Return markdown block for a rule-based explanation."""

    lines = [
        f"✅ Empfehlung: {data['prediction']}",
        f"🎯 Sicherheit: {data['confidence_text']}",
        "",
        f"⚙️ Angewendete Regel: \"{data['rule_name']}\"",
        "",
        "Erfüllte Bedingungen:",
    ]
    conditions = data.get("conditions", [])
    if conditions:
        lines.extend(_format_condition_line(cond) for cond in conditions)
    else:
        lines.append("• (keine Bedingungen ausgewertet)")

    support = data.get("rule_support")
    if support is not None:
        lines.extend([
            "",
            f"Diese Regel traf in {support:.0%} der historischen Fälle zu.",
        ])

    return "\n".join(lines)


def format_ml_explanation(data: Dict[str, object]) -> str:
    """Return markdown block for an ML-based explanation."""

    lines = [
        f"✅ Empfehlung: {data['prediction']}",
        f"🎯 Sicherheit: {data['confidence_text']}",
        "",
        "💡 Die 3 wichtigsten Faktoren:",
        "",
    ]

    factors = data.get("factors", [])
    if factors:
        for idx, factor in enumerate(factors[:3], start=1):
            lines.append(f"{idx}. {factor}")
    else:
        lines.append("(Keine SHAP-Faktoren verfügbar)")

    restriction_info = data.get("restriction_info")
    if restriction_info:
        lines.extend(["", *restriction_info])

    alternatives = data.get("alternatives", [])
    if alternatives:
        lines.extend(["", "📊 Alternative Maßnahmen:"])
        for name, probability in alternatives:
            lines.append(f"• {name}: {probability:.0%}")

    return "\n".join(lines)


def format_explanation_markdown(payload: Dict[str, object]) -> str:
    """Create a full markdown document for an explanation payload."""

    template = (
        "# Erklärung für Rechnung #{beleg_index}\n\n"
        "## Empfehlung\n"
        "**{massnahme}**\n\n"
        "**Konfidenz:** {confidence}\n\n"
        "## Begründung\n\n"
        "{reason_block}\n\n"
        "---\n"
        "*Generiert am {timestamp}*\n"
    )

    return template.format(
        beleg_index=payload["row_index"],
        massnahme=payload["prediction"],
        confidence=payload["confidence_text"],
        reason_block=payload["reason_block"],
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


@dataclass
class ExplanationComponents:
    payload: Dict[str, object]
    markdown: str


def build_explanation_payload(
    row_index: int,
    prediction_row: pd.Series,
    row_features: pd.Series,
    explanation: Dict[str, object],
    predictions_df: pd.DataFrame,
    shap_descriptions: Sequence[str],
    alternatives: Sequence[tuple[str, float]],
    restriction_info: Optional[List[str]] = None,
) -> ExplanationComponents:
    """Aggregate all pieces of information for markdown generation."""

    source = explanation.get("source", "ml") or "ml"
    confidence_value = float(prediction_row.get("final_confidence", np.nan))
    payload: Dict[str, object] = {
        "row_index": row_index,
        "prediction": prediction_row.get("final_prediction"),
        "confidence_value": confidence_value,
        "confidence_text": format_confidence(confidence_value),
        "source": "rule" if source != "ml" and not str(source).startswith("ml_restricted") else "ml",
    }

    if payload["source"] == "rule":
        rule_name = str(source)
        rule_confidence = prediction_row.get("rule_confidence")
        payload["confidence_text"] = format_confidence(rule_confidence if rule_confidence == rule_confidence else 1.0)
        conditions = explanation.get("details", {}).get("matched_conditions", [])
        if not conditions and "details" in explanation:
            conditions = explanation["details"].get("matched_conditions", [])
        support = None
        if "is_correct" in predictions_df.columns:
            mask = predictions_df.get("rule_source").fillna("") == rule_name
            if mask.any():
                support = float(predictions_df.loc[mask, "is_correct"].mean())
        payload["reason_block"] = format_rule_explanation(
            {
                "prediction": payload["prediction"],
                "confidence_text": payload["confidence_text"],
                "rule_name": rule_name,
                "conditions": conditions,
                "rule_support": support,
            }
        )
    else:
        payload["confidence_text"] = format_confidence(confidence_value)
        payload["reason_block"] = format_ml_explanation(
            {
                "prediction": payload["prediction"],
                "confidence_text": payload["confidence_text"],
                "factors": list(shap_descriptions),
                "alternatives": list(alternatives),
                "restriction_info": restriction_info or [],
            }
        )

    markdown = format_explanation_markdown(payload)
    payload["reason_block"] = payload["reason_block"]
    return ExplanationComponents(payload=payload, markdown=markdown)


__all__ = [
    "create_explanation_components",
    "ExplanationComponents",
    "build_explanation_payload",
    "format_confidence",
    "format_explanation_markdown",
    "format_ml_explanation",
    "format_rule_explanation",
    "humanize_shap_feature",
]


def create_explanation_components(
    predictor: HybridMassnahmenPredictor,
    row_index: int,
    prediction_row: pd.Series,
    feature_columns: Sequence[str],
    predictions_df: pd.DataFrame,
) -> ExplanationComponents:
    """Build explanation components for UI/CLI consumers."""

    feature_series = prediction_row[feature_columns].copy()
    row_series = pd.Series(feature_series)
    explanation = predictor.explain(row_series)
    details = explanation.get("details", {})

    shap_pairs: Iterable[tuple[str, float]] = details.get("shap_top5") or []
    lower_index_map = {col.lower(): col for col in row_series.index}

    def _lookup_value(name: str) -> object:
        if name in row_series.index:
            return row_series[name]
        lowered = name.lower()
        if lowered in lower_index_map:
            return row_series[lower_index_map[lowered]]
        if "__" in name:
            parts = name.split("__")
            for part in reversed(parts):
                part_lower = part.lower()
                if part in row_series.index:
                    return row_series[part]
                if part_lower in lower_index_map:
                    return row_series[lower_index_map[part_lower]]
        return prediction_row.get(name)

    shap_descriptions: List[str] = []
    for feature_name, shap_value in list(shap_pairs)[:3]:
        value = _lookup_value(feature_name)
        shap_descriptions.append(
            humanize_shap_feature(feature_name, value, shap_value, row_series)
        )

    probabilities = details.get("probabilities") or {}
    final_prediction = prediction_row.get("final_prediction")
    alternatives: List[tuple[str, float]] = []
    if probabilities:
        sorted_prob = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        for name, prob in sorted_prob:
            if str(name) == str(final_prediction):
                continue
            alternatives.append((str(name), float(prob)))
        alternatives = alternatives[:3]

    restriction_lines: List[str] = []
    ml_context = details.get("ml_context") or {}
    if ml_context.get("restricted"):
        allowed = [str(item) for item in ml_context.get("allowed_classes", []) if item]
        if allowed:
            restriction_lines.append(
                "🚦 Ampel-Einschränkung: Auswahl begrenzt auf " + ", ".join(allowed)
            )

    return build_explanation_payload(
        row_index=row_index,
        prediction_row=prediction_row,
        row_features=row_series,
        explanation=explanation,
        predictions_df=predictions_df,
        shap_descriptions=shap_descriptions,
        alternatives=alternatives,
        restriction_info=restriction_lines,
    )
