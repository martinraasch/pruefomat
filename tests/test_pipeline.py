import numpy as np
import pandas as pd

from config_loader import AppConfig, normalize_config_columns
from build_pipeline_veri import (
    DataFramePreparer,
    DaysUntilDueAdder,
    MassnahmenSuccessFeatures,
    build_preprocessor,
    infer_feature_plan,
)


def _make_config() -> AppConfig:
    config = AppConfig.model_validate(
        {
            "data": {
                "amount_col": "Betrag",
                "issue_col": "Belegdatum",
                "due_col": "Faellig",
                "categorical_columns": ["Land", "BUK", "Debitor"],
                "text_columns": ["DEB_Name", "Massnahme_2025", "Hinweise"],
                "target_col": "Ampel",
            },
            "preprocessing": {"tfidf_max_features": 50, "tfidf_min_df": 1, "tfidf_ngram_max": 1},
        }
    )
    return normalize_config_columns(config)


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Betrag": ["1.000,00", "250,00", "3.100,50"],
            "Belegdatum": ["01.01.2025", "05.01.2025", "10.01.2025"],
            "Faellig": ["05.01.2025", "07.01.2025", "15.01.2025"],
            "Land": ["DE", "AT", "DE"],
            "BUK": ["X", "Y", "X"],
            "Debitor": ["100", "200", "300"],
            "DEB_Name": ["Alpha GmbH", "Beta AG", "Gamma SE"],
            "Massnahme_2025": ["Pruefen", "", "Rueckfrage"],
            "Hinweise": ["Testfall", "Weitere pruefung", "Vorgemerkt"],
            "Ruckmeldung_erhalten": ["x", "", "x"],
            "Ampel": ["1", "2", "3"],
        }
    )


def test_preprocessor_produces_feature_matrix():
    config = _make_config()
    df = _sample_dataframe()

    features_df = df.drop(columns=[config.data.target_col])
    preparer = DataFramePreparer(
        amount_col=config.data.amount_col,
        issue_col=config.data.issue_col,
        due_col=config.data.due_col,
        date_columns=config.data.additional_date_columns,
        null_like=list({"", " ", "-"}),
    )
    prepared = preparer.fit_transform(features_df)
    enriched = DaysUntilDueAdder(
        issue_col=config.data.issue_col,
        due_col=config.data.due_col,
    ).fit_transform(prepared)

    feature_plan = infer_feature_plan(enriched, config)
    preprocessor = build_preprocessor(feature_plan, config)
    matrix = preprocessor.fit_transform(features_df)

    assert matrix.shape[0] == len(df)
    # matrix can be sparse or dense
    if hasattr(matrix, "toarray"):
        arr = matrix.toarray()
    else:
        arr = np.asarray(matrix)
    assert arr.shape[1] > 0
    assert np.isfinite(arr).all()
    select_columns = preprocessor.named_steps["select"].columns
    assert "massnahme_success_rate_buk_debitor" in select_columns
    assert "overall_response_rate" in select_columns
    assert "massnahme_success_rate_ampel" in select_columns


def test_preprocessor_handles_pandas_na_values():
    config = _make_config()
    df = _sample_dataframe()

    features_df = df.drop(columns=[config.data.target_col]).copy()
    features_df.loc[0, "Land"] = pd.NA
    features_df.loc[1, "DEB_Name"] = pd.NA
    features_df.loc[2, "Hinweise"] = pd.NA

    preparer = DataFramePreparer(
        amount_col=config.data.amount_col,
        issue_col=config.data.issue_col,
        due_col=config.data.due_col,
        date_columns=config.data.additional_date_columns,
        null_like=list({"", " ", "-"}),
    )
    prepared = preparer.fit_transform(features_df)
    enriched = DaysUntilDueAdder(
        issue_col=config.data.issue_col,
        due_col=config.data.due_col,
    ).fit_transform(prepared)

    feature_plan = infer_feature_plan(enriched, config)
    preprocessor = build_preprocessor(feature_plan, config)

    matrix = preprocessor.fit_transform(features_df)
    assert matrix.shape[0] == len(df)


def test_massnahmen_success_features_computes_rates(capsys):
    df = pd.DataFrame(
        {
            "BUK": ["A", "A", "A", "B", "B", "Z"],
            "Debitor": ["100", "100", "100", "200", "200", "999"],
            "Ampel": ["gelb", "gelb", "rot", "gelb", "rot", "gelb"],
            "Ruckmeldung_erhalten": ["x", "x", "", "", "x", ""],
        }
    )

    transformer = MassnahmenSuccessFeatures(min_samples=3, default_success=0.5)
    transformer.fit(df)
    logs = capsys.readouterr().out
    transformed = transformer.transform(df.iloc[[0, 3, 5]])

    assert np.isclose(transformed.loc[0, "massnahme_success_rate_buk_debitor"], 2 / 3)
    assert np.isclose(transformed.loc[0, "overall_response_rate"], 2 / 3)
    assert np.isclose(
        transformed.loc[0, "massnahme_success_rate_ampel"],
        transformer._lookup_ampel_rate("gelb"),
    )

    # Combination B/200 has only two samples (< min_samples) -> global average fallback
    fallback_value = transformer.global_success_rate_
    assert np.isclose(transformed.loc[3, "massnahme_success_rate_buk_debitor"], fallback_value)
    assert np.isclose(transformed.loc[5, "massnahme_success_rate_buk_debitor"], fallback_value)

    decoded_logs = logs.encode("utf-8").decode("unicode_escape")
    assert "Historische Success-Rates berechnet" in decoded_logs


def test_massnahmen_success_features_missing_column_defaults(capsys):
    df = pd.DataFrame({"BUK": ["A", "B"], "Debitor": ["100", "200"]})

    transformer = MassnahmenSuccessFeatures(default_success=0.5)
    transformer.fit(df)
    logs = capsys.readouterr().out
    transformed = transformer.transform(df)

    assert np.allclose(transformed["massnahme_success_rate_buk_debitor"], 0.5)
    assert np.allclose(transformed["overall_response_rate"], 0.5)
    assert np.allclose(transformed["massnahme_success_rate_ampel"], 0.5)
    decoded_logs = logs.encode("utf-8").decode("unicode_escape")
    assert "Rücklauf-Spalte nicht gefunden" in decoded_logs
