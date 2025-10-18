"""Configuration loading and validation utilities for pruefomat."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class RandomForestConfig(BaseModel):
    n_estimators: int = 300
    class_weight: Optional[str] = "balanced"
    max_depth: Optional[int] = None
    n_jobs: int = -1
    random_state: int = 42


class ModelSection(BaseModel):
    type: str = Field(default="random_forest", description="Model type identifier.")
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)


class PreprocessingSection(BaseModel):
    tfidf_max_features: int = 4000
    tfidf_min_df: int = 2
    tfidf_ngram_max: int = 2

    @field_validator("tfidf_max_features")
    @classmethod
    def _positive_max_features(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("tfidf_max_features must be positive")
        return value

    @field_validator("tfidf_min_df", "tfidf_ngram_max")
    @classmethod
    def _positive_integers(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("preprocessing parameters must be positive integers")
        return value


class DataSection(BaseModel):
    amount_col: Optional[str] = None
    issue_col: Optional[str] = None
    due_col: Optional[str] = None
    additional_date_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    flag_columns: list[str] = Field(default_factory=list)
    target_col: Optional[str] = None
    null_like: list[str] = Field(default_factory=list)

    @field_validator(
        "additional_date_columns",
        "numeric_columns",
        "categorical_columns",
        "text_columns",
        "flag_columns",
        "null_like",
    )
    @classmethod
    def _normalize_list(cls, value: list[str]) -> list[str]:
        return [str(item) for item in value]


class AppConfig(BaseModel):
    data: DataSection = Field(default_factory=DataSection)
    preprocessing: PreprocessingSection = Field(default_factory=PreprocessingSection)
    model: ModelSection = Field(default_factory=ModelSection)
    pattern_analysis: "PatternAnalysisSection" = Field(default_factory=lambda: PatternAnalysisSection())
    evaluation: "EvaluationSection" = Field(default_factory=lambda: EvaluationSection())


class NumericFeatureOptions(BaseModel):
    is_round: List[int] = Field(default_factory=lambda: [0, 2])
    quantiles: List[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    zero_check: bool = True
    extreme_percentile: float = 0.98


class CategoricalFeatureOptions(BaseModel):
    top_k: int = 10
    include_other: bool = True


class TextFeatureOptions(BaseModel):
    length_buckets: List[int] = Field(default_factory=lambda: [50, 200])
    non_empty: bool = True


class PatternAnalysisSection(BaseModel):
    date_features: List[str] = Field(
        default_factory=lambda: ["weekday", "is_weekend", "month", "quarter"]
    )
    numeric_features: NumericFeatureOptions = Field(default_factory=NumericFeatureOptions)
    categorical_features: CategoricalFeatureOptions = Field(default_factory=CategoricalFeatureOptions)
    text_features: TextFeatureOptions = Field(default_factory=TextFeatureOptions)
    min_lift: float = 1.15
    min_samples: int = 20
    max_p_value: float = 0.05
    max_feature_values: int = 30
    top_n: int = 40


class EvaluationSection(BaseModel):
    validation_size: float = Field(default=0.25, description="Fraction of samples used for validation.")
    top_k: int = Field(default=3, description="Number of top predictions considered for hit-rate metrics.")
    review_share: float = Field(default=0.2, description="Fraction of predictions to review in cost simulation.")
    cost_review: float = Field(default=50.0, description="Cost per manual review in simulation.")
    cost_miss: float = Field(default=250.0, description="Cost for missing an incorrect prediction in simulation.")

    @field_validator("validation_size", "review_share")
    @classmethod
    def _fraction_range(cls, value: float) -> float:
        if not 0 < value <= 0.9:
            raise ValueError("fractions must be in (0, 0.9]")
        return value

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k must be >= 1")
        return value

    @field_validator("cost_review", "cost_miss")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("cost values must be non-negative")
        return value


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


def load_config(path: Optional[str | Path]) -> AppConfig:
    """Load configuration from YAML or JSON file.

    If ``path`` is ``None`` the default configuration file under ``config/default_config.yaml``
    is used.
    """

    config_path: Path
    if path is None:
        config_path = Path("config/default_config.yaml")
    else:
        config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        raise ConfigError(f"Failed to read config: {exc}") from exc

    try:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(content) or {}
        else:
            data = json.loads(content)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Invalid configuration syntax: {exc}") from exc

    if not isinstance(data, Dict):
        raise ConfigError("Configuration root must be a mapping/dict")

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Configuration validation failed: {exc}") from exc



def normalize_config_columns(config: AppConfig) -> AppConfig:
    """Normalize column identifiers within the configuration in-place."""

    def _normalize_name(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        text = unicodedata.normalize("NFKD", str(value))
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", "_", text.strip())
        text = re.sub(r"[^0-9A-Za-z_]+", "", text)
        return text

    data_cfg = config.data
    data_cfg.amount_col = _normalize_name(data_cfg.amount_col)
    data_cfg.issue_col = _normalize_name(data_cfg.issue_col)
    data_cfg.due_col = _normalize_name(data_cfg.due_col)
    data_cfg.target_col = _normalize_name(data_cfg.target_col)
    data_cfg.additional_date_columns = [_normalize_name(col) for col in data_cfg.additional_date_columns]
    data_cfg.numeric_columns = [_normalize_name(col) for col in data_cfg.numeric_columns]
    data_cfg.categorical_columns = [_normalize_name(col) for col in data_cfg.categorical_columns]
    data_cfg.text_columns = [_normalize_name(col) for col in data_cfg.text_columns]
    data_cfg.flag_columns = [_normalize_name(col) for col in data_cfg.flag_columns]
    data_cfg.null_like = [str(item) for item in data_cfg.null_like]
    return config


__all__ = [
    "AppConfig",
    "ConfigError",
    "RandomForestConfig",
    "load_config",
    "normalize_config_columns",
    "PatternAnalysisSection",
    "EvaluationSection",
]
