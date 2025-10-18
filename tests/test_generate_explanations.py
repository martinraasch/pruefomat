import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import generate_explanations


def test_generate_explanations_creates_markdown(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    model = Pipeline(
        steps=[
            ("preprocessor", FunctionTransformer(validate=False)),
            ("classifier", DummyClassifier(strategy="stratified", random_state=0)),
        ]
    )
    feature_data = pd.DataFrame(
        {
            "Betrag_parsed": [1000, 2000, 3000, 1500],
            "Ampel": [1, 2, 3, 2],
        }
    )
    labels = ["A", "B", "A", "C"]
    model.fit(feature_data, labels)

    model_path = artifacts_dir / "baseline_model.joblib"
    predictions_path = artifacts_dir / "predictions.csv"
    joblib.dump(model, model_path)

    predictions_df = pd.DataFrame(
        [
            {
                "row_index": 0,
                "ml_prediction": "A",
                "ml_confidence": 0.6,
                "rule_source": "rule_a",
                "rule_prediction": "A",
                "rule_confidence": 1.0,
                "final_prediction": "A",
                "final_confidence": 1.0,
                "actual": "A",
                "is_correct": True,
                "review_score": 0.0,
                "confidence_percent": 100.0,
                "explanation": "🔍 Erklärung anzeigen",
                "Betrag_parsed": 1000,
                "Ampel": 1,
            },
            {
                "row_index": 1,
                "ml_prediction": "B",
                "ml_confidence": 0.75,
                "rule_source": float("nan"),
                "rule_prediction": None,
                "rule_confidence": float("nan"),
                "final_prediction": "B",
                "final_confidence": 0.75,
                "actual": "B",
                "is_correct": True,
                "review_score": 0.25,
                "confidence_percent": 75.0,
                "explanation": "🔍 Erklärung anzeigen",
                "Betrag_parsed": 2000,
                "Ampel": 2,
            },
            {
                "row_index": 2,
                "ml_prediction": "C",
                "ml_confidence": 0.65,
                "rule_source": "ml_restricted_gelb",
                "rule_prediction": None,
                "rule_confidence": float("nan"),
                "final_prediction": "C",
                "final_confidence": 0.65,
                "actual": "A",
                "is_correct": False,
                "review_score": 0.35,
                "confidence_percent": 65.0,
                "explanation": "🔍 Erklärung anzeigen",
                "Betrag_parsed": 3000,
                "Ampel": 2,
            },
            {
                "row_index": 3,
                "ml_prediction": "A",
                "ml_confidence": 0.45,
                "rule_source": float("nan"),
                "rule_prediction": None,
                "rule_confidence": float("nan"),
                "final_prediction": "A",
                "final_confidence": 0.45,
                "actual": "C",
                "is_correct": False,
                "review_score": 0.55,
                "confidence_percent": 45.0,
                "explanation": "🔍 Erklärung anzeigen",
                "Betrag_parsed": 1500,
                "Ampel": 3,
            },
        ]
    )
    predictions_df.to_csv(predictions_path, index=False)

    output_dir = tmp_path / "workshop_examples"
    original_output_dir = generate_explanations.OUTPUT_DIR
    generate_explanations.OUTPUT_DIR = output_dir

    args = SimpleNamespace(
        top_n=4,
        artifacts_dir=str(artifacts_dir),
        rules=str(generate_explanations.DEFAULT_RULE_PATH),
    )

    try:
        generate_explanations.generate_explanations(args)
    finally:
        generate_explanations.OUTPUT_DIR = original_output_dir

    assert output_dir.exists()
    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary) <= 4
    for entry in summary:
        assert Path(entry["path"]).exists()
