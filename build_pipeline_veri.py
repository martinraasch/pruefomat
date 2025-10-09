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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from config_loader import (
    AppConfig,
    ConfigError,
    RandomForestConfig,
    load_config,
    normalize_config_columns,
)
from logging_utils import configure_logging, get_logger, mask_sensitive_data


logger = get_logger(__name__)

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

    def __init__(self, separator: str = " ", output_feature: str = "text_concat") -> None:
        self.separator = separator
        self.output_feature = output_feature

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

    def get_feature_names_out(self, input_features=None):
        return np.asarray([self.output_feature])


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


def infer_feature_plan(df: pd.DataFrame, config: AppConfig) -> FeaturePlan:
    numeric: List[str] = []
    if config.data.amount_col and "Betrag_parsed" in df.columns and df["Betrag_parsed"].notna().any():
        numeric.append("Betrag_parsed")
    if (
        config.data.issue_col
        and config.data.due_col
        and "tage_bis_faellig" in df.columns
        and df["tage_bis_faellig"].notna().any()
    ):
        numeric.append("tage_bis_faellig")

    for col in config.data.numeric_columns:
        if col in df.columns and df[col].notna().any():
            numeric.append(col)

    categorical = [col for col in config.data.categorical_columns if col in df.columns]
    text = [col for col in config.data.text_columns if col in df.columns]

    return FeaturePlan(numeric=numeric, categorical=categorical, text=text)


def build_preprocessor(feature_plan: FeaturePlan, config: AppConfig) -> Pipeline:
    transformers: List[tuple] = []

    if feature_plan.numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num", numeric_pipeline, feature_plan.numeric))

    if feature_plan.categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, feature_plan.categorical))

    if feature_plan.text:
        text_pipeline = Pipeline(
            steps=[
                ("concat", TextConcatenator()),
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, config.preprocessing.tfidf_ngram_max),
                        min_df=config.preprocessing.tfidf_min_df,
                        max_features=config.preprocessing.tfidf_max_features,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        transformers.append(("text", text_pipeline, feature_plan.text))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )

    date_columns = config.data.additional_date_columns
    null_like = list({*NULL_LIKE_DEFAULT, *config.data.null_like})

    pipeline = Pipeline(
        steps=[
            (
                "prep_df",
                DataFramePreparer(
                    amount_col=config.data.amount_col or "",
                    issue_col=config.data.issue_col or "",
                    due_col=config.data.due_col or "",
                    date_columns=date_columns,
                    null_like=null_like,
                ),
            ),
            (
                "add_due_days",
                DaysUntilDueAdder(
                    issue_col=config.data.issue_col or "",
                    due_col=config.data.due_col or "",
                ),
            ),
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
    config_path = Path(args.config) if args.config else None
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        logger.error("config_error", message=str(exc), config_path=str(config_path) if config_path else "default")
        return 1

    config = normalize_config_columns(config)

    if args.target:
        config.data.target_col = normalize_column_name(args.target)

    logger.info(
        "config_loaded",
        config_path=str(config_path or Path("config/default_config.yaml")),
        target_col=config.data.target_col,
    )

    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df_raw = read_excel(excel_path, sheet)
    df_norm, column_mapping = normalize_columns(df_raw)

    logger.info(
        "excel_loaded",
        file=str(excel_path),
        sheet=sheet,
        rows=len(df_norm),
        columns=df_norm.shape[1],
    )

    df_features = df_norm.copy()
    y = None
    target_name = config.data.target_col

    if target_name:
        if target_name in df_norm.columns:
            raw_target = df_norm[target_name].astype("string").str.strip()
            raw_target = raw_target.replace({"": pd.NA})
            mask = raw_target.notna()
            if mask.any():
                y = raw_target.loc[mask].reset_index(drop=True)
                df_features = df_features.loc[mask].reset_index(drop=True)
            else:
                logger.warning("target_no_labels", target=target_name)
            df_features = df_features.drop(columns=[target_name])
        else:
            logger.warning("target_missing", target=target_name)
            target_name = None

    preparer = DataFramePreparer(
        amount_col=config.data.amount_col or "",
        issue_col=config.data.issue_col or "",
        due_col=config.data.due_col or "",
        date_columns=config.data.additional_date_columns,
        null_like=list({*NULL_LIKE_DEFAULT, *config.data.null_like}),
    )
    prepared = preparer.fit_transform(df_features)
    with_due = DaysUntilDueAdder(
        issue_col=config.data.issue_col or "",
        due_col=config.data.due_col or "",
    ).fit_transform(prepared)

    feature_plan = infer_feature_plan(with_due, config)

    logger.info(
        "feature_plan_created",
        numeric=len(feature_plan.numeric),
        categorical=len(feature_plan.categorical),
        text=len(feature_plan.text),
    )

    preprocessor = build_preprocessor(feature_plan, config)
    preprocessor.fit(df_features)

    prep_out = Path(args.prep_out)
    ensure_directory(prep_out)
    joblib.dump(preprocessor, prep_out)

    profile: Dict[str, Any] = {
        "input_excel": str(excel_path),
        "sheet": sheet,
        "config_file": str(config_path or Path("config/default_config.yaml")),
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
            random_state=config.model.random_forest.random_state,
            stratify=stratify,
        )

        rf_kwargs = config.model.random_forest.model_dump(exclude_none=True)

        baseline_model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_plan, config)),
                ("classifier", RandomForestClassifier(**rf_kwargs)),
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

        encoder = baseline_model.named_steps["preprocessor"].named_steps.get("encode")
        if encoder is not None and hasattr(encoder, "get_feature_names_out"):
            feature_names = encoder.get_feature_names_out()
        else:
            feature_names = np.array([f"f_{i}" for i in range(len(baseline_model.named_steps["classifier"].feature_importances_))])

        importance = baseline_model.named_steps["classifier"].feature_importances_
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
        importance_df.sort_values("importance", ascending=False, inplace=True)

        importance_csv = Path(args.feature_importance_csv)
        ensure_directory(importance_csv)
        importance_df.to_csv(importance_csv, index=False)

        top20 = importance_df.head(20)
        if not top20.empty:
            plt.figure(figsize=(10, 6))
            plt.barh(top20["feature"][::-1], top20["importance"][::-1], color="#2b8a3e")
            plt.xlabel("Importance")
            plt.title("Top 20 Feature Importances")
            plt.tight_layout()
            plot_path = Path(args.feature_importance_plot)
            ensure_directory(plot_path)
            plt.savefig(plot_path, dpi=150)
            plt.close()
        else:
            plot_path = None

        model_path = Path(args.model_out)
        ensure_directory(model_path)
        joblib.dump(baseline_model, model_path)

        metrics_path = Path(args.metrics_out)
        ensure_directory(metrics_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        profile["baseline_model"] = str(model_path)
        profile["metrics_file"] = str(metrics_path)
        profile["feature_importance_csv"] = str(importance_csv)
        if plot_path:
            profile["feature_importance_plot"] = str(plot_path)

        logger.info(
            "baseline_trained",
            metrics=mask_sensitive_data({"balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"]}),
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            feature_importance_csv=str(importance_csv),
            feature_importance_plot=str(plot_path) if plot_path else None,
        )
    elif target_name:
        logger.warning("baseline_skipped_single_class", target=target_name)

    profile_path = Path(args.report_out)
    ensure_directory(profile_path)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    logger.info(
        "artifacts_written",
        **mask_sensitive_data(
            {
                "prep_path": str(prep_out),
                "model_path": str(model_path) if model_path else None,
                "metrics_path": str(metrics_path) if metrics_path else None,
                "profile_path": str(profile_path),
            }
        ),
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Veri preprocessing pipeline")
    parser.add_argument("--excel", required=True, help="Path to Excel file with Veri data")
    parser.add_argument("--sheet", default="0", help="Sheet name or index")
    parser.add_argument("--config", default=None, help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--target", default=None, help="Override target column for baseline training")
    parser.add_argument("--prep-out", default="outputs/prep_pipeline.joblib", help="Path to save preprocessing pipeline")
    parser.add_argument("--model-out", default="outputs/model.joblib", help="Path to save baseline model")
    parser.add_argument("--metrics-out", default="outputs/metrics.json", help="Path to save baseline metrics")
    parser.add_argument("--report-out", default="outputs/profile.json", help="Path to save profile report")
    parser.add_argument("--feature-importance-csv", default="outputs/feature_importance.csv", help="Path to save feature importances (CSV)")
    parser.add_argument("--feature-importance-plot", default="outputs/feature_importance.png", help="Path to save feature importance chart (PNG)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return run_pipeline_builder(args)


if __name__ == "__main__":
    sys.exit(main())
