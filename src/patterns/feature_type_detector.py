"""Automated feature type detection for pattern analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config_loader import AppConfig


class FeatureType(str, Enum):
    """Supported canonical feature types."""

    NUMERIC = "numeric"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class FeatureInfo:
    """Metadata about a detected feature."""

    name: str
    feature_type: FeatureType
    dtype: str
    nullable: bool
    distinct: int
    sample: Optional[str] = None

    def is_categorical_like(self) -> bool:
        return self.feature_type in {FeatureType.CATEGORICAL, FeatureType.BOOLEAN}


class FeatureTypeDetector:
    """Infer feature types from a pandas dataframe and optional configuration."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config
        self._config_overrides = self._build_overrides(config)

    @staticmethod
    def _build_overrides(config: Optional[AppConfig]) -> Dict[str, FeatureType]:
        overrides: Dict[str, FeatureType] = {}
        if config is None:
            return overrides

        data_cfg = config.data
        for col in data_cfg.numeric_columns:
            overrides[col] = FeatureType.NUMERIC
        for col in data_cfg.additional_date_columns:
            overrides[col] = FeatureType.DATETIME
        for col in data_cfg.categorical_columns:
            overrides[col] = FeatureType.CATEGORICAL
        for col in data_cfg.text_columns:
            overrides[col] = FeatureType.TEXT
        # explicit amount/issue/due columns if set
        if data_cfg.amount_col:
            overrides[data_cfg.amount_col] = FeatureType.NUMERIC
        if data_cfg.issue_col:
            overrides[data_cfg.issue_col] = FeatureType.DATETIME
        if data_cfg.due_col:
            overrides[data_cfg.due_col] = FeatureType.DATETIME
        if data_cfg.target_col:
            overrides[data_cfg.target_col] = FeatureType.CATEGORICAL
        return {k: v for k, v in overrides.items() if k}

    def detect(self, df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> List[FeatureInfo]:
        """Detect feature types for columns in the dataframe."""

        if columns is None:
            columns = df.columns
        detected: List[FeatureInfo] = []
        for column in columns:
            if column not in df.columns:
                continue
            series = df[column]
            feature_type = self._detect_series_type(column, series)
            nullable = series.isna().any()
            distinct = int(series.nunique(dropna=True))
            sample_value = self._sample_value(series)
            detected.append(
                FeatureInfo(
                    name=column,
                    feature_type=feature_type,
                    dtype=str(series.dtype),
                    nullable=nullable,
                    distinct=distinct,
                    sample=sample_value,
                )
            )
        return detected

    def _detect_series_type(self, name: str, series: pd.Series) -> FeatureType:
        # Config override takes precedence
        override = self._config_overrides.get(name)
        if override:
            return override

        series_notna = series.dropna()
        if series_notna.empty:
            return FeatureType.UNKNOWN

        if pd.api.types.is_bool_dtype(series_notna):
            return FeatureType.BOOLEAN

        if pd.api.types.is_datetime64_any_dtype(series_notna):
            return FeatureType.DATETIME

        if pd.api.types.is_numeric_dtype(series_notna):
            return FeatureType.NUMERIC

        # object / string like
        if pd.api.types.is_string_dtype(series_notna) or pd.api.types.is_object_dtype(series_notna):
            inferred = self._infer_object_type(series_notna)
            if inferred is not None:
                return inferred

        return FeatureType.UNKNOWN

    @staticmethod
    def _infer_object_type(series: pd.Series) -> Optional[FeatureType]:
        str_values = series.astype(str)
        lengths = str_values.str.len()
        avg_length = lengths.replace({np.nan: 0}).mean()
        distinct = series.nunique(dropna=True)
        total = len(series)
        if total == 0:
            return FeatureType.UNKNOWN

        distinct_ratio = distinct / total
        # numeric-looking? try convert
        numeric_candidate = pd.to_numeric(series, errors="coerce")
        if numeric_candidate.notna().mean() > 0.95:
            return FeatureType.NUMERIC

        # date-like? try parse
        parsed_dates = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed_dates.notna().mean() > 0.9:
            return FeatureType.DATETIME

        if distinct <= 2:
            return FeatureType.BOOLEAN

        if avg_length > 60 and distinct_ratio > 0.3:
            return FeatureType.TEXT

        if distinct_ratio <= 0.2 or distinct <= 50:
            return FeatureType.CATEGORICAL

        if avg_length > 120:
            return FeatureType.TEXT

        # default fallback
        return FeatureType.CATEGORICAL if distinct_ratio < 0.6 else FeatureType.TEXT

    @staticmethod
    def _sample_value(series: pd.Series) -> Optional[str]:
        for value in series:
            if pd.notna(value):
                return str(value)[:80]
        return None


__all__ = ["FeatureType", "FeatureInfo", "FeatureTypeDetector"]
