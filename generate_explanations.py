#!/usr/bin/env python3
"""Generate workshop-ready explanations from stored artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

from src.business_rules import load_business_rules_from_file
from src.explanations import create_explanation_components
from src.hybrid_predictor import HybridMassnahmenPredictor
from src.rule_engine import RuleEngine

DEFAULT_RULE_PATH = Path("config/business_rules_massnahmen.yaml")
DEFAULT_ARTIFACT_DIR = Path("outputs")
OUTPUT_DIR = Path("workshop_examples")

EXCLUDED_PREDICTION_COLUMNS = {
    "row_index",
    "ml_prediction",
    "ml_confidence",
    "rule_source",
    "rule_prediction",
    "rule_confidence",
    "final_prediction",
    "final_confidence",
    "actual",
    "is_correct",
    "review_score",
    "confidence_percent",
    "explanation",
}


def _slugify_label(text: object) -> str:
    value = str(text)
    normalized = value.encode("ascii", "ignore").decode("ascii")
    safe = "".join(ch if ch.isalnum() else "_" for ch in normalized.lower())
    return safe.strip("_") or "massnahme"


def _load_predictor(model_path: Path, rules_path: Path) -> HybridMassnahmenPredictor:
    business_rules = load_business_rules_from_file(rules_path)
    baseline_model = joblib.load(model_path)
    rule_engine = RuleEngine(business_rules)
    predictor = HybridMassnahmenPredictor(baseline_model, rule_engine)
    preprocessor = baseline_model.named_steps.get("preprocessor")
    predictor.preprocessor_ = preprocessor
    if preprocessor is not None:
        encoder = preprocessor.named_steps.get("encode") if hasattr(preprocessor, "named_steps") else None
        if encoder is not None and hasattr(encoder, "get_feature_names_out"):
            predictor.feature_names_ = list(encoder.get_feature_names_out())
    return predictor


def _prepare_background(predictor: HybridMassnahmenPredictor, features: pd.DataFrame) -> None:
    preprocessor = predictor.preprocessor_
    if preprocessor is None or features.empty:
        return
    background = preprocessor.transform(features.head(min(len(features), 200)))
    predictor.background_ = background.toarray() if hasattr(background, "toarray") else background
    if predictor.feature_names_ is None and hasattr(background, "shape"):
        predictor.feature_names_ = [f"f_{idx}" for idx in range(background.shape[1])]


def _candidate_rows(df: pd.DataFrame, top_n: int) -> List[int]:
    selected: List[int] = []
    selected_set = set()

    def _add(rows: Sequence[int], limit: int | None = None) -> None:
        count = 0
        for row in rows:
            if row in selected_set:
                continue
            selected.append(row)
            selected_set.add(row)
            count += 1
            if limit is not None and count >= limit:
                break
            if len(selected) >= top_n:
                break

    rule_mask = df["rule_source"].notna() & ~df["rule_source"].fillna("").str.startswith("ml_restricted")
    rule_candidates: List[int] = []
    for _, group in df.loc[rule_mask].groupby("rule_source"):
        top_row = group.sort_values("final_confidence", ascending=False).iloc[0]
        rule_candidates.append(int(top_row["row_index"]))
    _add(rule_candidates, limit=2)

    ml_plain = df[df["rule_source"].isna()].sort_values("final_confidence", ascending=False)
    _add([int(val) for val in ml_plain["row_index"]], limit=3)

    ml_restricted_mask = df["rule_source"].fillna("").str.startswith("ml_restricted")
    ml_restricted = df.loc[ml_restricted_mask].sort_values("final_confidence", ascending=False)
    _add([int(val) for val in ml_restricted["row_index"]], limit=3)

    edge_cases = df.sort_values("final_confidence", ascending=True)
    _add([int(val) for val in edge_cases["row_index"]], limit=2)

    if len(selected) < top_n:
        remaining = df.sort_values("final_confidence", ascending=False)
        _add([int(val) for val in remaining["row_index"]])

    return selected[:top_n]


def generate_explanations(args: argparse.Namespace) -> None:
    artifacts_dir = Path(args.artifacts_dir)
    predictions_path = artifacts_dir / "predictions.csv"
    model_path = artifacts_dir / "baseline_model.joblib"
    rules_path = Path(args.rules)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    predictions_df = pd.read_csv(predictions_path)
    feature_columns = [col for col in predictions_df.columns if col not in EXCLUDED_PREDICTION_COLUMNS]
    feature_frame = predictions_df[feature_columns].copy()

    predictor = _load_predictor(model_path, rules_path)
    _prepare_background(predictor, feature_frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_indices = _candidate_rows(predictions_df, args.top_n)
    summaries: List[Dict[str, object]] = []

    for row_index in selected_indices:
        row_df = predictions_df.loc[predictions_df["row_index"] == row_index]
        if row_df.empty:
            continue
        prediction_row = row_df.iloc[0]
        components = create_explanation_components(
            predictor=predictor,
            row_index=row_index,
            prediction_row=prediction_row,
            feature_columns=feature_columns,
            predictions_df=predictions_df,
        )
        prediction = components.payload.get("prediction", "massnahme")
        slug = _slugify_label(prediction)
        path = OUTPUT_DIR / f"erklaerung_beleg_{row_index}_{slug}.md"
        path.write_text(components.markdown, encoding="utf-8")

        summaries.append(
            {
                "row_index": row_index,
                "prediction": prediction,
                "confidence": components.payload.get("confidence_value"),
                "path": str(path),
            }
        )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Gespeichert {len(summaries)} Erklärungen in {OUTPUT_DIR.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erklärungen für Workshop vorbereiten")
    parser.add_argument("--top-n", type=int, default=10, help="Anzahl der zu generierenden Beispiele")
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACT_DIR,
        help="Verzeichnis mit Baseline-Artefakten (predictions.csv, baseline_model.joblib)",
    )
    parser.add_argument(
        "--rules",
        default=DEFAULT_RULE_PATH,
        help="Pfad zu den Business Rules",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generate_explanations(args)


if __name__ == "__main__":
    main()
