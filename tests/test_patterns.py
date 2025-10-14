import pandas as pd

from config_loader import AppConfig
from app import analyze_patterns_action
from src.patterns import (
    ConditionalProbabilityAnalyzer,
    FeatureTypeDetector,
    InsightFormatter,
    InterpretableFeatureGenerator,
)


def test_pattern_pipeline_basic():
    config = AppConfig()
    config.pattern_analysis.min_samples = 1
    config.pattern_analysis.min_lift = 1.0
    config.pattern_analysis.max_p_value = 1.0
    df = pd.DataFrame(
        {
            "Belegdatum": ["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-06"],
            "Betrag": [100.0, 100.0, 2500.0, 2600.0],
            "Kategorie": ["A", "B", "A", "A"],
        }
    )
    target = pd.Series([1, 2, 3, 3])

    detector = FeatureTypeDetector(config)
    detected = detector.detect(df)
    assert detected, "No features detected"

    generator = InterpretableFeatureGenerator(config)
    generated = generator.generate(df, detected)
    assert generated, "No features generated"

    analyzer = ConditionalProbabilityAnalyzer(config)
    insights = analyzer.analyze(generated, target)
    assert insights, "No insights generated"

    formatter = InsightFormatter(target_name="Ampel")
    text = formatter.format_insight(insights[0])
    assert "Ampel" in text


def test_analyze_patterns_action(tmp_path):
    config = AppConfig()
    config.pattern_analysis.min_samples = 1
    config.pattern_analysis.min_lift = 1.0
    config.pattern_analysis.max_p_value = 1.0

    state = {
        "df_features": pd.DataFrame(
            {
                "Belegdatum": ["2024-01-01", "2024-01-06", "2024-01-06", "2024-01-06"],
                "Betrag": [100.0, 2500.0, 2600.0, 2700.0],
            }
        ),
        "target": pd.Series([1, 3, 3, 2]),
        "target_name": "Ampel",
        "config": config,
    }
    status, markdown, csv_path, new_state = analyze_patterns_action(state)
    assert "Muster" in status
    assert markdown is not None
    assert csv_path is not None
    assert new_state.get("pattern_insights_df") is not None
