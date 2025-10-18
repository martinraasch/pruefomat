import pandas as pd

from src.explanations import (
    ExplanationComponents,
    build_explanation_payload,
    create_explanation_components,
    format_confidence,
    format_explanation_markdown,
    format_ml_explanation,
    format_rule_explanation,
    humanize_shap_feature,
)


def test_humanize_shap_feature_handles_numeric_prefix():
    row = pd.Series({"Betrag_parsed": 18500})
    text = humanize_shap_feature("numeric__Betrag_parsed", 18500, 0.2, row)
    assert "Betrag" in text
    assert "Erhöht" in text

    ampel_row = pd.Series({"Ampel": 2})
    ampel_text = humanize_shap_feature("categorical__Ampel_2", 2, -0.1, ampel_row)
    assert "Ampel-Farbe" in ampel_text


def test_format_confidence_labels():
    assert format_confidence(0.97).startswith("Sehr hoch")
    assert format_confidence(0.72).startswith("Mittel")
    assert format_confidence(None).startswith("Sehr hoch")


def test_build_rule_explanation_block():
    payload = {
        "prediction": "Rechnungsprüfung",
        "confidence_text": "Sehr hoch (100%)",
        "rule_name": "niedrig_betrag",
        "conditions": [
            {"matched": True, "field": "Ampel", "operator": "=", "target": 1, "value": 1}
        ],
        "rule_support": 0.95,
    }
    text = format_rule_explanation(payload)
    assert "Angewendete Regel" in text
    assert "95%" in text


def test_build_explanation_payload_for_ml():
    prediction_row = pd.Series(
        {
            "final_prediction": "Telefonische Bestätigung",
            "final_confidence": 0.87,
            "rule_source": None,
        }
    )
    predictions_df = pd.DataFrame([prediction_row])
    explanation = {
        "prediction": "Telefonische Bestätigung",
        "source": "ml",
        "details": {},
    }
    components = build_explanation_payload(
        row_index=5,
        prediction_row=prediction_row,
        row_features=pd.Series({}),
        explanation=explanation,
        predictions_df=predictions_df,
        shap_descriptions=["Faktor A"],
        alternatives=[("Alternative", 0.23)],
        restriction_info=["🚦 Hinweis"],
    )
    assert isinstance(components, ExplanationComponents)
    assert "Faktor A" in components.payload["reason_block"]
    markdown = format_explanation_markdown(components.payload)
    assert "## Empfehlung" in markdown


def test_humanize_shap_feature_additional_branches():
    base_row = pd.Series({"Betrag_parsed": "unbekannt", "Ampel": "Gelb"})

    negative_text = humanize_shap_feature(
        "numeric__Betrag_parsed",
        "n/a",
        -0.3,
        base_row,
    )
    assert "einfachere Prüfung" in negative_text

    ampel_text = humanize_shap_feature("categorical__Ampel", "Gelb", 0.2, base_row)
    assert "Ampel-Farbe" in ampel_text

    history_text = humanize_shap_feature("hist_action_diversity", 0.4, 0.1, base_row)
    assert "Historie" in history_text

    success_text = humanize_shap_feature(
        "massnahme_success_rate_buk_debitor",
        "0.8",
        0.1,
        base_row,
    )
    assert "80%" in success_text

    notes_text = humanize_shap_feature("notes_text_tfidf__mahnen", 0.3, 0.1, base_row)
    assert "Schlüsselwörter" in notes_text

    fallback_text = humanize_shap_feature("other_feature", "wert", 0.1, base_row)
    assert "other_feature" in fallback_text


def test_format_rule_explanation_without_conditions():
    block = format_rule_explanation(
        {
            "prediction": "Freigabe",
            "confidence_text": "Hoch (85%)",
            "rule_name": "ohne_bedingungen",
            "conditions": [],
            "rule_support": None,
        }
    )
    assert "keine Bedingungen" in block


def test_build_explanation_payload_for_rule_support():
    prediction_row = pd.Series(
        {
            "final_prediction": "Gutschrift",
            "final_confidence": 0.66,
            "rule_confidence": 0.85,
            "rule_source": "historical_rule",
        }
    )
    predictions_df = pd.DataFrame(
        [
            {"rule_source": "historical_rule", "is_correct": True},
            {"rule_source": "historical_rule", "is_correct": False},
            {"rule_source": "andere_regel", "is_correct": True},
        ]
    )
    explanation = {
        "prediction": "Gutschrift",
        "source": "historical_rule",
        "details": {
            "matched_conditions": [
                {
                    "field": "Ampel",
                    "operator": "=",
                    "target": 2,
                    "value": 2,
                    "matched": True,
                }
            ]
        },
    }

    components = build_explanation_payload(
        row_index=7,
        prediction_row=prediction_row,
        row_features=pd.Series({"Ampel": 2}),
        explanation=explanation,
        predictions_df=predictions_df,
        shap_descriptions=[],
        alternatives=[],
    )

    assert components.payload["source"] == "rule"
    assert "historischen Fälle" in components.payload["reason_block"]
    assert components.payload["confidence_text"].startswith("Hoch")


def test_create_explanation_components_handles_restrictions():
    class DummyPredictor:
        def explain(self, _row):
            return {
                "prediction": "Telefonische Rechnungsprüfung",
                "confidence": 0.72,
                "source": "ml_restricted_gelb",
                "details": {
                    "shap_top5": [
                        ("numeric__Betrag_parsed", 0.2),
                        ("vectorizer__notes_text_tfidf__urgent", -0.1),
                    ],
                    "probabilities": {
                        "Telefonische Rechnungsprüfung": 0.55,
                        "Alternative 1": 0.3,
                        "Alternative 2": 0.15,
                    },
                    "ml_context": {
                        "restricted": True,
                        "allowed_classes": ["Alternative 1", "Alternative 2", None],
                    },
                },
            }

    predictor = DummyPredictor()
    prediction_row = pd.Series(
        {
            "final_prediction": "Telefonische Rechnungsprüfung",
            "final_confidence": 0.72,
            "Betrag_parsed": 12345.0,
            "Ampel": 2,
            "notes_text_tfidf__urgent": 0.35,
            "vectorizer__notes_text_tfidf__urgent": 0.4,
        }
    )
    predictions_df = pd.DataFrame([prediction_row])

    components = create_explanation_components(
        predictor=predictor,
        row_index=3,
        prediction_row=prediction_row,
        feature_columns=["Betrag_parsed", "Ampel", "notes_text_tfidf__urgent"],
        predictions_df=predictions_df,
    )

    reason_block = components.payload["reason_block"]
    assert "Alternative Maßnahmen" in reason_block
    assert "🚦 Ampel-Einschränkung" in reason_block
    assert "Schlüsselwörter" in reason_block
