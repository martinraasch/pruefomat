import pandas as pd

from build_pipeline_veri import HistoricalPatternFeatures


def test_historical_pattern_features_basic():
    X = pd.DataFrame(
        {
            "BUK": ["100", "100", "200"],
            "Debitor": ["A", "A", "B"],
        }
    )
    y = pd.Series(["Gutschrift", "Rechnungsprüfung", "Gutschrift"])

    transformer = HistoricalPatternFeatures()
    transformer.fit(X, y)

    transformed = transformer.transform(pd.DataFrame({"BUK": ["100", "300"], "Debitor": ["A", "Z"]}))

    assert transformed.loc[0, "hist_most_frequent_action"] == "Gutschrift"
    assert transformed.loc[0, "hist_action_diversity"] == 2
    assert transformed.loc[0, "hist_gutschrift_count"] == 1
    assert transformed.loc[1, "hist_most_frequent_action"] == "unknown"
    assert transformed.loc[1, "hist_action_diversity"] == 0
    assert transformed.loc[1, "hist_gutschrift_count"] == 0


def test_historical_pattern_features_no_fit_data():
    X = pd.DataFrame({"BUK": ["100"], "Debitor": ["A"]})
    transformer = HistoricalPatternFeatures()
    transformer.fit(X, y=None)
    transformed = transformer.transform(pd.DataFrame({"BUK": ["100"], "Debitor": ["A"]}))

    assert transformed.loc[0, "hist_most_frequent_action"] == "unknown"
    assert transformed.loc[0, "hist_action_diversity"] == 0
    assert transformed.loc[0, "hist_gutschrift_count"] == 0
