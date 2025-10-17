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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
                             confusion_matrix, f1_score, precision_score,
                             recall_score, fbeta_score)
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
    lowered = s.str.lower()
    # Convert pandas <NA> markers to numpy nan so downstream sklearn
    # imputers operating on object arrays do not trip over ambiguous NA.
    lowered = lowered.astype(object)
    mask = pd.isna(lowered)
    if mask.any():
        lowered[mask] = np.nan
    return lowered


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
    else:
        # Fill any remaining NaT entries with a dayfirst-agnostic parse so
        # ISO-style dates (yyyy-mm-dd) stay intact regardless of the initial flag.
        na_mask = best.isna()
        if na_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fallback = pd.to_datetime(text[na_mask], errors="coerce")
            best = best.copy()
            best.loc[na_mask] = fallback
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


class HistoricalPatternFeatures(BaseEstimator, TransformerMixin):
    """Aggregate historical Maßnahme patterns per BUK/Debitor combination."""

    def __init__(self, group_cols: List[str] | None = None) -> None:
        self.group_cols = group_cols or ["BUK", "Debitor"]

    def fit(self, X, y=None):  # noqa: D401
        df = pd.DataFrame(X).copy()
        self.group_cols_ = [col for col in self.group_cols if col in df.columns]
        self.most_frequent_: Dict[Tuple, str] = {}
        self.action_counts_: Dict[Tuple, Dict[str, int]] = {}

        if not self.group_cols_ or y is None:
            return self

        target_series = pd.Series(y)
        df["_target"] = target_series.values

        grouped_mode = (
            df.groupby(self.group_cols_, dropna=False)["_target"]
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else None)
        )
        for key, value in grouped_mode.items():
            self.most_frequent_[self._ensure_tuple(key)] = self._coerce_label(value)

        count_frame = (
            df.groupby(self.group_cols_ + ["_target"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        for _, row in count_frame.iterrows():
            key = self._ensure_tuple(tuple(row[col] for col in self.group_cols_))
            label = self._coerce_label(row["_target"])
            self.action_counts_.setdefault(key, {})[label] = int(row["count"])

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if not getattr(self, "group_cols_", None):
            df["hist_most_frequent_action"] = "unknown"
            df["hist_action_diversity"] = 0
            df["hist_gutschrift_count"] = 0
            return df

        keys = df[self.group_cols_].apply(lambda row: self._ensure_tuple(tuple(row)), axis=1)

        df["hist_most_frequent_action"] = keys.apply(
            lambda key: self.most_frequent_.get(key, "unknown")
        )
        df["hist_action_diversity"] = keys.apply(
            lambda key: len(self.action_counts_.get(key, {}))
        )
        df["hist_gutschrift_count"] = keys.apply(
            lambda key: self.action_counts_.get(key, {}).get("Gutschrift", 0)
        )
        return df

    @staticmethod
    def _ensure_tuple(value: Tuple | Any) -> Tuple:
        if isinstance(value, tuple):
            return value
        if isinstance(value, pd.Series):
            return tuple(value.tolist())
        return (value,)

    @staticmethod
    def _coerce_label(value: Any) -> str:
        if pd.isna(value):
            return "unknown"
        return str(value)


class MassnahmenSuccessFeatures(BaseEstimator, TransformerMixin):
    """Derive historical response rates for BUK/Debitor combinations."""

    def __init__(
        self,
        buk_col: str = "BUK",
        debitor_col: str = "Debitor",
        response_col: str = "Ruckmeldung_erhalten",
        ampel_col: str = "Ampel",
        min_samples: int = 3,
        default_success: float = 0.5,
    ) -> None:
        self.buk_col = buk_col
        self.debitor_col = debitor_col
        self.response_col = response_col
        self.ampel_col = ampel_col
        self.min_samples = int(min_samples)
        self.default_success = float(default_success)

    def fit(self, X, y=None):  # noqa: D401
        df = pd.DataFrame(X).copy()

        if self.response_col not in df.columns:
            logger.warning("Rücklauf-Spalte nicht gefunden, verwende neutrale Success-Rates")
            return self._set_defaults(global_rate=self.default_success)

        success_flags = df[self.response_col].apply(self._to_success_flag)
        if success_flags.isna().all():
            logger.warning("Rücklauf-Spalte leer, verwende neutrale Success-Rates")
            return self._set_defaults(global_rate=self.default_success)

        df["_success_flag"] = success_flags.fillna(False).astype(float)

        group_cols = [col for col in (self.buk_col, self.debitor_col) if col in df.columns]
        if len(group_cols) < 2:
            logger.warning("BUK/Debitor-Spalten nicht vollständig, verwende neutrale Success-Rates")
            return self._set_defaults(global_rate=float(df["_success_flag"].mean()))

        grouped = (
            df.groupby(group_cols, dropna=False)["_success_flag"]
            .agg(success_count="sum", total_count="count")
            .reset_index()
        )
        grouped["success_rate"] = grouped["success_count"] / grouped["total_count"].clip(lower=1)

        self.success_lookup_: Dict[Tuple[Any, Any], Dict[str, float]] = {}
        below_threshold = 0
        for _, row in grouped.iterrows():
            key = self._ensure_tuple(row[group_cols])
            total = float(row["total_count"])
            if total < self.min_samples:
                below_threshold += 1
            self.success_lookup_[key] = {
                "rate": float(row["success_rate"]),
                "count": total,
                "success": float(row["success_count"]),
            }

        global_success = float(df["_success_flag"].mean()) if len(df) else self.default_success
        if not np.isfinite(global_success):
            global_success = self.default_success

        ampel_lookup: Dict[Any, float] = {}
        if self.ampel_col in df.columns:
            grouped_ampel = (
                df.groupby(self.ampel_col, dropna=False)["_success_flag"].mean().fillna(global_success)
            )
            ampel_lookup = {key: float(val) for key, val in grouped_ampel.items()}
        self.ampel_success_lookup_ = ampel_lookup

        self.global_success_rate_ = global_success if np.isfinite(global_success) else self.default_success
        self.success_counts_ = {key: meta["count"] for key, meta in self.success_lookup_.items()}
        self.history_count_ = len(self.success_lookup_)
        self.below_threshold_ = below_threshold

        logger.info(
            "Historische Success-Rates berechnet: %d unique BUK/Debitor-Kombinationen gefunden",
            self.history_count_,
        )
        logger.info(
            "Globale Durchschnitts-Success-Rate: %.1f%%",
            self.global_success_rate_ * 100,
        )
        logger.info(
            "Kombinationen mit <3 Datenpunkten: %d (verwenden globalen Durchschnitt)",
            self.below_threshold_,
        )

        return self

    def transform(self, X):  # noqa: D401
        df = pd.DataFrame(X).copy()
        rate_col = "massnahme_success_rate_buk_debitor"
        overall_col = "overall_response_rate"
        ampel_col = "massnahme_success_rate_ampel"

        rates = []
        overall_rates = []
        ampel_rates = []

        index = df.index
        buk_series = df[self.buk_col] if self.buk_col in df.columns else pd.Series(pd.NA, index=index)
        debitor_series = df[self.debitor_col] if self.debitor_col in df.columns else pd.Series(pd.NA, index=index)
        ampel_series = df[self.ampel_col] if self.ampel_col in df.columns else pd.Series(pd.NA, index=index)

        for idx in range(len(df)):
            buk = buk_series.iloc[idx] if idx < len(buk_series) else pd.NA
            debitor = debitor_series.iloc[idx] if idx < len(debitor_series) else pd.NA
            key = self._ensure_tuple([buk, debitor]) if not (pd.isna(buk) and pd.isna(debitor)) else None

            rate, used_global, count = self._lookup_rate(key)
            rates.append(rate)
            overall_rates.append(rate)

            ampel_val = ampel_series.iloc[idx] if idx < len(ampel_series) else pd.NA
            ampel_rate = self._lookup_ampel_rate(ampel_val)
            ampel_rates.append(ampel_rate)

            source = "globaler Durchschnitt, keine Historie" if used_global else f"aus {int(count)} historischen Fällen"
            logger.debug(
                "success_rate_lookup",
                buk=buk,
                debitor=debitor,
                rate=float(rate),
                source=source,
            )

        df[rate_col] = np.asarray(rates, dtype=float)
        df[overall_col] = np.asarray(overall_rates, dtype=float)
        df[ampel_col] = np.asarray(ampel_rates, dtype=float)
        return df

    def _lookup_rate(self, key: Optional[Tuple[Any, Any]]) -> Tuple[float, bool, float]:
        if key is None or key not in getattr(self, "success_lookup_", {}):
            return self.global_success_rate_, True, 0.0
        meta = self.success_lookup_[key]
        if meta["count"] < self.min_samples:
            return self.global_success_rate_, True, meta["count"]
        return meta["rate"], False, meta["count"]

    def _lookup_ampel_rate(self, ampel_value: Any) -> float:
        if pd.isna(ampel_value):
            return self.global_success_rate_
        return self.ampel_success_lookup_.get(ampel_value, self.global_success_rate_)

    def _set_defaults(self, global_rate: float) -> "MassnahmenSuccessFeatures":
        rate = self.default_success if not np.isfinite(global_rate) else float(global_rate)
        self.global_success_rate_ = rate
        self.success_lookup_ = {}
        self.success_counts_ = {}
        self.ampel_success_lookup_ = {}
        self.history_count_ = 0
        self.below_threshold_ = 0
        return self

    @staticmethod
    def _ensure_tuple(values: Iterable[Any]) -> Tuple[Any, Any]:
        seq = list(values)
        if len(seq) == 1:
            seq.append(None)
        return tuple(seq[:2])

    @staticmethod
    def _to_success_flag(value: Any) -> Optional[bool]:
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(value)
        text = str(value).strip().lower()
        if not text:
            return False
        return text in {"x", "true", "1", "ja", "yes"}


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
    historical_numeric = ["hist_action_diversity", "hist_gutschrift_count"]
    historical_categorical = ["hist_most_frequent_action"]
    success_numeric = [
        "massnahme_success_rate_buk_debitor",
        "overall_response_rate",
        "massnahme_success_rate_ampel",
    ]

    augmented_numeric = list(feature_plan.numeric)
    for col in historical_numeric:
        if col not in augmented_numeric:
            augmented_numeric.append(col)
    for col in success_numeric:
        if col not in augmented_numeric:
            augmented_numeric.append(col)

    augmented_categorical = list(feature_plan.categorical)
    for col in historical_categorical:
        if col not in augmented_categorical:
            augmented_categorical.append(col)

    if augmented_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num", numeric_pipeline, augmented_numeric))

    if augmented_categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, augmented_categorical))

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

    selected_columns = feature_plan.all
    for col in (*historical_categorical, *historical_numeric):
        if col not in selected_columns:
            selected_columns.append(col)
    for col in success_numeric:
        if col not in selected_columns:
            selected_columns.append(col)

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
            ("historical", HistoricalPatternFeatures()),
            ("success", MassnahmenSuccessFeatures()),
            ("select", SelectColumns(selected_columns)),
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
            "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "f2_score": float(fbeta_score(y_test, y_pred, average="macro", beta=2.0, zero_division=0)),
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

        proba = baseline_model.predict_proba(X_test)
        if proba.ndim == 1:
            pos_scores = proba
        elif proba.shape[1] == 1:
            pos_scores = proba[:, 0]
        else:
            classes = list(baseline_model.named_steps["classifier"].classes_)
            positive_index = 1 if len(classes) > 1 else 0
            for candidate in ("1", 1, "fraud", "Fraud", True):
                if candidate in classes:
                    positive_index = classes.index(candidate)
                    break
            if len(classes) == 2:
                pos_scores = proba[:, positive_index]
            else:
                pos_scores = proba.max(axis=1)

        fraud_scores = pd.Series(pos_scores * 100.0, name="fraud_score")
        predictions_raw = pd.Series(baseline_model.predict(X_test))
        predictions_numeric = pd.to_numeric(predictions_raw, errors="coerce")
        if predictions_numeric.notna().all():
            predictions_col = predictions_numeric.round().astype(int)
        else:
            predictions_col = predictions_raw

        actual_series = pd.Series(y_test).reset_index(drop=True)
        predictions_col = predictions_col.reset_index(drop=True)
        fraud_scores = fraud_scores.reset_index(drop=True)

        predictions_df = pd.DataFrame(
            {
                "fraud_score": fraud_scores,
                "prediction": predictions_col,
                "actual": actual_series,
            }
        )
        predictions_df.sort_values("fraud_score", ascending=False, inplace=True)

        predictions_path = Path(args.predictions_out)
        ensure_directory(predictions_path)
        predictions_df.to_csv(predictions_path, index=False)

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
        profile["predictions_file"] = str(predictions_path)

        metric_summary = {
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f2_score": metrics["f2_score"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

        logger.info(
            "baseline_trained",
            metrics=mask_sensitive_data(metric_summary),
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            feature_importance_csv=str(importance_csv),
            feature_importance_plot=str(plot_path) if plot_path else None,
            predictions_path=str(predictions_path),
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
    parser.add_argument("--predictions-out", default="outputs/predictions.csv", help="Path to save predictions for the test set")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return run_pipeline_builder(args)


if __name__ == "__main__":
    sys.exit(main())
