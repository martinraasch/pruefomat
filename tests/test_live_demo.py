import pandas as pd
import pytest

from datetime import date

from app import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    balance_classes_dataframe,
    build_pipeline_action,
    parse_ampel_choice,
    prepare_live_invoice,
    predict_live_action,
    train_baseline_action,
)


def test_parse_ampel_choice_variants():
    assert parse_ampel_choice("Grün (1)") == 1
    assert parse_ampel_choice("Gelb (2)") == 2
    assert parse_ampel_choice("rot") == 3
    assert parse_ampel_choice(1) == 1
    assert parse_ampel_choice(None) is None


def test_prepare_live_invoice_includes_amount():
    state = {
        "df_features_full": pd.DataFrame(columns=["Betrag", "Ampel", "BUK", "Debitor", "Betrag_parsed"]),
        "selected_columns": ["Betrag", "Ampel", "BUK", "Debitor"],
    }
    df = prepare_live_invoice(
        state,
        {
            "betrag": 12345.0,
            "ampel": "Gelb (2)",
            "buk": "A",
            "debitor": "100",
            "belegdatum": date(2025, 1, 1),
            "faelligkeit": date(2025, 1, 15),
            "deb_name": "Test GmbH",
            "hinweise": "",
            "negativ": False,
        },
    )
    assert df.loc[0, "Betrag"].startswith("12.345")
    assert df.loc[0, "Ampel"] == 2
    assert "Betrag_parsed" in df.columns


def test_balance_classes_dataframe_balances():
    df = pd.DataFrame(
        {
            "Massnahme_2025": ["A"] * 2 + ["B"] * 5,
            "Betrag": list(range(7)),
        }
    )
    balanced = balance_classes_dataframe(df, "Massnahme_2025", seed=0)
    counts = balanced["Massnahme_2025"].value_counts()
    assert counts.nunique() == 1
    assert counts.iloc[0] == counts.iloc[-1]


@pytest.fixture
def live_demo_state():
    data = {
        "Betrag": ["15.000,00", "95.000,00", "35.000,00", "60.000,00"],
        "Ampel": [1, 1, 2, 3],
        "BUK": ["A", "B", "A", "C"],
        "Debitor": ["100", "200", "300", "400"],
        "Belegdatum": ["01.01.2025", "02.01.2025", "03.01.2025", "04.01.2025"],
        "Faellig": ["31.01.2025", "02.02.2025", "03.02.2025", "05.02.2025"],
        "DEB_Name": ["Bestandskunde", "Top Lieferant", "Neue Firma", "Kritisch"],
        "Hinweise": ["", "", "", ""],
        "negativ": [False, False, False, True],
        "Massnahme_2025": [
            "Rechnungsprüfung",
            "Freigabe gemäß Kompetenzkatalog",
            "Beibringung Liefer-/Leistungsnachweis (vorgelagert)",
            "telefonische Lieferbestätigung (nachgelagert)",
        ],
    }
    df = pd.DataFrame(data)
    features = df.drop(columns=["Massnahme_2025"])
    target = df["Massnahme_2025"]

    config = DEFAULT_CONFIG.model_copy(deep=True)
    config.preprocessing.tfidf_min_df = 1

    state = {
        "config": config,
        "config_path": str(DEFAULT_CONFIG_PATH),
        "df_features": features.copy(),
        "df_features_full": features.copy(),
        "target": target,
        "selected_columns": list(features.columns),
    }

    pipeline_result = build_pipeline_action(state)
    state = pipeline_result[-1]
    train_result = train_baseline_action(state)
    state = train_result[-1]
    return state


def test_predict_live_action_rule_trigger(live_demo_state):
    result_md, explanation_md, export_status, _ = predict_live_action(
        15000.0,
        "Grün (1)",
        "A",
        "100",
        date(2025, 1, 1),
        date(2025, 1, 31),
        "",
        "",
        False,
        live_demo_state,
    )
    assert "Rechnungsprüfung" in result_md
    assert explanation_md.startswith("# Erklärung")
    assert export_status == ""


def test_predict_live_action_high_amount_warning(live_demo_state):
    result_md, _, _, _ = predict_live_action(
        1_500_000.0,
        "Gelb (2)",
        "A",
        "300",
        date(2025, 1, 1),
        date(2025, 1, 31),
        "",
        "",
        False,
        live_demo_state,
    )
    assert "⚠️ Sehr hoher Betrag" in result_md
