import pandas as pd
import pytest

from build_pipeline_veri import safe_amount, parse_date_series, DataFramePreparer


def test_safe_amount_parsing_handles_locales():
    series = pd.Series(["1.234,56", "2,50", "3 000,75", "-", None])
    parsed = safe_amount(series)
    assert pytest.approx(parsed.iloc[0], rel=1e-6) == 1234.56
    assert pytest.approx(parsed.iloc[1], rel=1e-6) == 2.50
    assert pytest.approx(parsed.iloc[2], rel=1e-6) == 3000.75
    assert parsed.iloc[3:].isna().all()


def test_parse_date_series_dayfirst():
    series = pd.Series(["01.02.2025", "2025-03-04", "not a date"])
    parsed = parse_date_series(series)
    assert parsed.iloc[0].day == 1 and parsed.iloc[0].month == 2
    assert parsed.iloc[1].month == 3 and parsed.iloc[1].day == 4
    assert pd.isna(parsed.iloc[2])


def test_dataframe_preparer_drops_empty_columns():
    df = pd.DataFrame(
        {
            "Betrag": ["1,00", "2,00"],
            "Belegdatum": ["01.01.2025", "02.01.2025"],
            "Faellig": ["03.01.2025", "04.01.2025"],
            "EmptyCol": ["-", "-"],
        }
    )

    preparer = DataFramePreparer(amount_col="Betrag", issue_col="Belegdatum", due_col="Faellig")
    prepared = preparer.fit_transform(df)

    assert "EmptyCol" not in prepared.columns
    assert "Betrag_parsed" in prepared.columns
    assert "tage_bis_faellig" not in prepared.columns
