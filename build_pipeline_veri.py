#!/usr/bin/env python3
"""Build preprocessing pipeline and optional baseline model for Veri data."""

import argparse
import json
import re
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# Helper functions
# =============================================================================

NULL_LIKE_DEFAULT = {"", " ", "-", "_", "n/a", "na", "nan", "None"}


def to_str(values: Iterable) -> pd.Series:
    """Return values as pandas Series of strings."""
    return pd.Series(list(values), dtype="string").fillna("")


def strip_lower(series: pd.Series) -> pd.Series:
    """Strip whitespace and lowercase string entries while keeping missing as NaN."""
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA})
    return s.str.lower()


def normalize_column_name(name: str) -> str:
    """Convert column name to snake_case ASCII for stable references."""
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^0-9A-Za-z_]+", "", text)
    return text


def parse_date_series(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Parse date strings using pandas with heuristic dayfirst handling."""
    text = series.astype("string").str.strip()
    text = text.replace({"": pd.NA, "-": pd.NA})

    options = [dayfirst, not dayfirst] if dayfirst in (True, False) else [True, False]
    best = None
    best_valid = -1
    for flag in options:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=flag)
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best_valid = valid
            best = parsed
        if valid == len(text):
            break
    if best is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            best = pd.to_datetime(text, errors="coerce")
    return best


def safe_amount(series: pd.Series) -> pd.Series:
    """Parse localized amount strings into floats."""
    s = series.astype("string").str.replace(r"\s", "", regex=True)
    s = s.str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def days_until_due(issue: pd.Series, due: pd.Series) -> pd.Series:
    """Compute days between issue and due dates."""
    return (due - issue).dt.days


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Custom transformers
# =============================================================================


class DataFramePreparer(BaseEstimator, TransformerMixin):
    """Normalize raw Veri dataframe: clean strings, parse amounts and dates."""

    def __init__(
        self,
        amount_col: str = "Betrag",
        issue_col: str = "Belegdatum",
        due_col: str = "Faellig",
        date_columns: Optional[Sequence[str]] = None,
        null_like: Optional[Sequence[str]] = None,
    ) -> None:
        self.amount_col = amount_col
        self.issue_col = issue_col
        self.due_col = due_col
        self.date_columns = list(date_columns) if date_columns is not None else []
        self.null_like = set(null_like) if null_like is not None else set(NULL_LIKE_DEFAULT)

    def fit(self, X, y=None):  # noqa: D401
        df = self._prepare_dataframe(pd.DataFrame(X).copy())
        self.columns_to_drop_ = [col for col in df.columns if df[col].isna().all()]
        self.feature_names_in_ = list(df.columns)
        self.feature_names_out_ = [c for c in df.columns if c not in self.columns_to_drop_]
        return self

    def transform(self, X):
        df = self._prepare_dataframe(pd.DataFrame(X).copy())
        for col in getattr(self, "columns_to_drop_", []):
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        replacements = {val: np.nan for val in self.null_like}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df = df.replace(replacements)
        df = df.infer_objects(copy=False)

        object_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in object_cols:
            df[col] = strip_lower(df[col])

        if self.amount_col and self.amount_col in df.columns:
            df["Betrag_parsed"] = safe_amount(df[self.amount_col]).astype(float)

        date_cols = list(self.date_columns)
        for candidate in (self.issue_col, self.due_col):
            if candidate and candidate not in date_cols:
                date_cols.append(candidate)
        for col in date_cols:
            if col in df.columns:
                df[col] = parse_date_series(df[col])
        return df


class DaysUntilDueAdder(BaseEstimator, TransformerMixin):
    """Add tage_bis_faellig numerical feature from issue and due date columns."""

    def __init__(self, issue_col: str = "Belegdatum", due_col: str = "Faellig", out_col: str = "tage_bis_faellig") -> None:
        self.issue_col = issue_col
        self.due_col = due_col
        self.out_col = out_col

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.issue_col in df.columns and self.due_col in df.columns:
            issue = df[self.issue_col]
            due = df[self.due_col]
            issue_dt = issue if np.issubdtype(issue.dtype, np.datetime64) else parse_date_series(issue)
            due_dt = due if np.issubdtype(due.dtype, np.datetime64) else parse_date_series(due)
            df[self.out_col] = pd.to_numeric(days_until_due(issue_dt, due_dt), errors="coerce")
        else:
            df[self.out_col] = np.nan
        return df


class SelectColumns(BaseEstimator, TransformerMixin):
    """Select a subset of columns, dropping missing ones gracefully."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)

    def fit(self, X, y=None):  # noqa: D401
        df = pd.DataFrame(X)
        self.columns_ = [col for col in self.columns if col in df.columns]
        self.missing_ = [col for col in self.columns if col not in df.columns]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        available = [col for col in getattr(self, "columns_", []) if col in df.columns]
        return df[available]


class TextConcatenator(BaseEstimator, TransformerMixin):
    """Concatenate multiple text columns into a single cleaned string per row."""

    def __init__(self, separator: str = " ") -> None:
        self.separator = separator

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        combined: List[str] = []
        for _, row in df.iterrows():
            parts: List[str] = []
            for value in row.tolist():
                if pd.isna(value):
                    continue
                text = str(value).strip()
                if text:
                    parts.append(text)
            combined.append(self.separator.join(parts))
        return np.asarray(combined, dtype=object)


# =============================================================================
# Core pipeline builder
# =============================================================================


@dataclass
class FeaturePlan:
    numeric: List[str]
    categorical: List[str]
    text: List[str]

    @property
    def all(self) -> List[str]:
        ordered: List[str] = []
        for group in (self.numeric, self.categorical, self.text):
            for col in group:
                if col not in ordered:
                    ordered.append(col)
        return ordered


def infer_feature_plan(df: pd.DataFrame) -> FeaturePlan:
    numeric_candidates = ["Betrag_parsed", "tage_bis_faellig"]
    categorical_candidates = ["Land", "BUK", "Debitor"]
    text_candidates = ["DEB_Name", "Massnahme_2025", "Hinweise"]

    numeric = [col for col in numeric_candidates if col in df.columns and df[col].notna().any()]
    categorical = [col for col in categorical_candidates if col in df.columns]
    text = [col for col in text_candidates if col in df.columns]

    return FeaturePlan(numeric=numeric, categorical=categorical, text=text)


def build_preprocessor(feature_plan: FeaturePlan) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    text_pipeline = Pipeline(
        steps=[
            ("concat", TextConcatenator()),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=4000,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, feature_plan.numeric),
            ("cat", categorical_pipeline, feature_plan.categorical),
            ("text", text_pipeline, feature_plan.text),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            (
                "prep_df",
                DataFramePreparer(
                    amount_col="Betrag",
                    issue_col="Belegdatum",
                    due_col="Faellig",
                    date_columns=["Datum"],
                ),
            ),
            ("add_due_days", DaysUntilDueAdder(issue_col="Belegdatum", due_col="Faellig")),
            ("select", SelectColumns(feature_plan.all)),
            ("encode", column_transformer),
        ]
    )
    return pipeline


# =============================================================================
# CLI logic
# =============================================================================


def read_excel(path: Path, sheet: str | int) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, dtype="object")


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mapping = {col: normalize_column_name(col) for col in df.columns}
    renamed = df.rename(columns=mapping)
    return renamed, mapping


def run_pipeline_builder(args: argparse.Namespace) -> int:
    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df_raw = read_excel(excel_path, sheet)
    df_norm, column_mapping = normalize_columns(df_raw)

    target_name = None
    y = None
    df_features = df_norm.copy()

    if args.target:
        normalized_target = normalize_column_name(args.target)
        if normalized_target in df_norm.columns:
            raw_target = pd.Series(df_norm[normalized_target], dtype="object")
            target_clean = raw_target.astype("string").str.strip()
            target_clean = target_clean.replace({"": pd.NA})
            mask = target_clean.notna()
            if mask.any():
                target_name = normalized_target
                y = target_clean.loc[mask].astype("string")
                df_features = df_features.loc[mask]
            else:
                print(
                    f"Warning: target column '{args.target}' exists but contains no labels after cleaning; baseline will be skipped.",
                    file=sys.stderr,
                )
            df_features = df_features.drop(columns=[normalized_target])
        else:
            print(
                f"Warning: target column '{args.target}' not found after normalization (expected '{normalized_target}').",
                file=sys.stderr,
            )
    df_features = df_features.reset_index(drop=True)
    if y is not None:
        y = y.reset_index(drop=True)

    preparer_preview = DataFramePreparer(
        amount_col="Betrag",
        issue_col="Belegdatum",
        due_col="Faellig",
        date_columns=["Datum"],
    )
    prepared = preparer_preview.fit_transform(df_features)
    with_due = DaysUntilDueAdder(issue_col="Belegdatum", due_col="Faellig").fit_transform(prepared)

    feature_plan = infer_feature_plan(with_due)

    preprocessor = build_preprocessor(feature_plan)
    preprocessor.fit(df_features)

    prep_out = Path(args.prep_out)
    ensure_directory(prep_out)
    joblib.dump(preprocessor, prep_out)

    profile = {
        "input_excel": str(excel_path),
        "sheet": sheet,
        "column_mapping": column_mapping,
        "feature_plan": {
            "numeric": feature_plan.numeric,
            "categorical": feature_plan.categorical,
            "text": feature_plan.text,
        },
        "dropped_columns": getattr(preprocessor.named_steps["prep_df"], "columns_to_drop_", []),
    }

    metrics = None
    model_path = None
    metrics_path = None

    if target_name and y is not None and y.nunique(dropna=True) > 1:
        stratify = y if y.nunique(dropna=True) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            df_features,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        baseline_model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_plan)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        baseline_model.fit(X_train, y_train)
        y_pred = baseline_model.predict(X_test)

        metrics = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        model_path = Path(args.model_out)
        ensure_directory(model_path)
        joblib.dump(baseline_model, model_path)

        metrics_path = Path(args.metrics_out)
        ensure_directory(metrics_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        profile["baseline_model"] = str(model_path)
        profile["metrics_file"] = str(metrics_path)

    elif target_name:
        print(
            "Warning: Target available but only a single class after cleaning; baseline model skipped.",
            file=sys.stderr,
        )

    profile_path = Path(args.report_out)
    ensure_directory(profile_path)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"Saved preprocessor to {prep_out}")
    if model_path:
        print(f"Saved baseline model to {model_path}")
    if metrics_path:
        print(f"Saved metrics to {metrics_path}")
    print(f"Profile written to {profile_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Veri preprocessing pipeline")
    parser.add_argument("--excel", required=True, help="Path to Excel file with Veri data")
    parser.add_argument("--sheet", default="0", help="Sheet name or index")
    parser.add_argument("--target", default=None, help="Target column for baseline training")
    parser.add_argument("--prep-out", default="outputs/prep_pipeline.joblib", help="Path to save preprocessing pipeline")
    parser.add_argument("--model-out", default="outputs/model.joblib", help="Path to save baseline model")
    parser.add_argument("--metrics-out", default="outputs/metrics.json", help="Path to save baseline metrics")
    parser.add_argument("--report-out", default="outputs/profile.json", help="Path to save profile report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_pipeline_builder(args)


if __name__ == "__main__":
    sys.exit(main())
