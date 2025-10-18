import numpy as np
import pandas as pd

from config_loader import AppConfig
from src.patterns.feature_type_detector import FeatureType, FeatureTypeDetector


def test_feature_type_detector_with_overrides():
    config = AppConfig()
    config.data.numeric_columns = ["forced_numeric"]
    config.data.additional_date_columns = ["issued"]
    config.data.categorical_columns = ["category"]
    config.data.text_columns = ["notes"]
    df = pd.DataFrame(
        {
            "forced_numeric": ["1", "2", None],
            "issued": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "category": ["A", "B", "A"],
            "notes": ["lorem", "ipsum", "dolor"],
            "boolean_str": ["true", "false", "true"],
            "scores": [0.1, 0.5, 0.2],
            "mixed": ["2020-01-02", "2020-01-03", "foo"],
            "long_text": ["x" * 200, "y" * 210, "z" * 220],
            "unknown": [np.nan, np.nan, np.nan],
        }
    )

    detector = FeatureTypeDetector(config)
    detected = {info.name: info for info in detector.detect(df)}

    assert detected["forced_numeric"].feature_type == FeatureType.NUMERIC
    assert detected["issued"].feature_type == FeatureType.DATETIME
    assert detected["category"].feature_type == FeatureType.CATEGORICAL
    assert detected["notes"].feature_type == FeatureType.TEXT
    assert detected["scores"].feature_type == FeatureType.NUMERIC
    assert detected["unknown"].feature_type == FeatureType.UNKNOWN

    inferred_numeric = FeatureTypeDetector._infer_object_type(df["boolean_str"])
    assert inferred_numeric == FeatureType.BOOLEAN

    inferred_mixed = FeatureTypeDetector._infer_object_type(df["mixed"])
    assert inferred_mixed == FeatureType.CATEGORICAL

    mostly_dates = pd.Series([f"2020-01-{day:02d}" for day in range(1, 11)] + ["invalid"])
    assert FeatureTypeDetector._infer_object_type(mostly_dates) == FeatureType.DATETIME

    inferred_long = FeatureTypeDetector._infer_object_type(df["long_text"])
    assert inferred_long == FeatureType.TEXT

    boolean_info = detected["category"]
    assert boolean_info.is_categorical_like() is True

    sample_value = FeatureTypeDetector._sample_value(df["mixed"])
    assert sample_value == "2020-01-02"
