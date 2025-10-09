import numpy as np
import pandas as pd
import pytest

from build_pipeline_veri import safe_amount, parse_date_series, DataFramePreparer
from eval_utils import precision_recall_at_k, compute_cost_simulation


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


def test_precision_recall_at_k_small_sample():
    y_true = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    prec, rec = precision_recall_at_k(y_true, scores, 0.4)
    assert pytest.approx(prec, rel=1e-6) == 1.0
    assert pytest.approx(rec, rel=1e-6) == 2 / 3


def test_cost_simulation_behaviour():
    y_true = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.2, 0.8, 0.3, 0.1])
    benefit = compute_cost_simulation(y_true, scores, 0.4, cost_review=10, cost_miss=100)
    assert benefit > 0
