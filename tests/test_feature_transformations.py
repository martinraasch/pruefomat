import pandas as pd

from config_loader import AppConfig
from src.patterns.feature_transformations import GeneratedFeature, InterpretableFeatureGenerator
from src.patterns.feature_type_detector import FeatureInfo, FeatureType


def build_feature(name, feature_type):
    return FeatureInfo(
        name=name,
        feature_type=feature_type,
        dtype="object",
        nullable=False,
        distinct=3,
    )


def test_interpretable_feature_generator_covers_all_types():
    df = pd.DataFrame(
        {
            "booking_date": ["2024-01-01", "2024-01-06", "2024-01-07"],
            "amount": [0.0, 10.5, 20.25],
            "status": ["offen", "bezahlt", "offen"],
            "flag": [True, False, True],
            "notes": ["kurz", "ein etwas längerer text", ""]
        }
    )

    config = AppConfig()
    config.pattern_analysis.date_features = ["weekday", "is_weekend", "month", "quarter", "hour_bucket"]
    config.pattern_analysis.numeric_features.is_round = [0, 2]
    config.pattern_analysis.numeric_features.quantiles = [0.25, 0.75]
    config.pattern_analysis.numeric_features.extreme_percentile = 0.9
    config.pattern_analysis.categorical_features.top_k = 2
    config.pattern_analysis.categorical_features.include_other = True
    config.pattern_analysis.text_features.length_buckets = [5, 10]
    config.pattern_analysis.text_features.non_empty = True

    generator = InterpretableFeatureGenerator(config)
    features = [
        build_feature("booking_date", FeatureType.DATETIME),
        build_feature("amount", FeatureType.NUMERIC),
        build_feature("status", FeatureType.CATEGORICAL),
        build_feature("flag", FeatureType.BOOLEAN),
        build_feature("notes", FeatureType.TEXT),
    ]

    generated = generator.generate(df, features)
    names = {feature.name for feature in generated}

    expected_names = {
        "booking_date_weekday",
        "booking_date_is_weekend",
        "booking_date_month",
        "booking_date_quarter",
        "booking_date_daypart",
        "amount_round_0",
        "amount_round_2",
        "amount_zero_check",
        "amount_quantile_bucket",
        "amount_extreme_high",
        "status_topk",
        "flag_bool",
        "notes_non_empty",
        "notes_length_bucket",
    }

    assert expected_names.issubset(names)

    for feature in generated:
        assert isinstance(feature, GeneratedFeature)
        assert feature.series.isna().sum() == 0


def test_interpretable_feature_generator_skips_unknown_type():
    df = pd.DataFrame({"misc": [1, 2, 3]})
    generator = InterpretableFeatureGenerator()
    features = [build_feature("misc", FeatureType.UNKNOWN)]

    assert generator.generate(df, features) == []


def test_numeric_quantiles_with_invalid_values_are_ignored():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    config = AppConfig()
    config.pattern_analysis.numeric_features.quantiles = ["bad", 0.9]
    generator = InterpretableFeatureGenerator(config)
    feature = build_feature("amount", FeatureType.NUMERIC)

    results = generator._transform_numeric(feature, df["amount"])
    names = {item.name for item in results}

    assert "amount_quantile_bucket" not in names
    assert "amount_zero_check" in names
