"""Interpretable feature transformations based on detected feature types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config_loader import AppConfig
from .feature_type_detector import FeatureInfo, FeatureType


@dataclass
class GeneratedFeature:
    """Represents a derived feature ready for pattern analysis."""

    name: str
    series: pd.Series
    base: FeatureInfo
    description: str
    value_describer: Callable[[object], str]


class InterpretableFeatureGenerator:
    """Generate human interpretable features according to configuration."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config
        self.pattern_cfg = getattr(config, "pattern_analysis", None)

    def generate(self, df: pd.DataFrame, features: Iterable[FeatureInfo]) -> List[GeneratedFeature]:
        results: List[GeneratedFeature] = []
        for feature in features:
            if feature.feature_type == FeatureType.UNKNOWN:
                continue
            series = df[feature.name]
            if feature.feature_type == FeatureType.DATETIME:
                results.extend(self._transform_datetime(feature, series))
            elif feature.feature_type == FeatureType.NUMERIC:
                results.extend(self._transform_numeric(feature, series))
            elif feature.feature_type == FeatureType.CATEGORICAL:
                results.extend(self._transform_categorical(feature, series))
            elif feature.feature_type == FeatureType.BOOLEAN:
                results.extend(self._transform_boolean(feature, series))
            elif feature.feature_type == FeatureType.TEXT:
                results.extend(self._transform_text(feature, series))
        return results

    # ------------------------------------------------------------------
    # Datetime
    # ------------------------------------------------------------------

    def _datetime_options(self) -> Iterable[str]:
        if self.pattern_cfg and getattr(self.pattern_cfg, "date_features", None):
            return self.pattern_cfg.date_features
        return ("weekday", "is_weekend", "month", "quarter")

    def _transform_datetime(self, feature: FeatureInfo, series: pd.Series) -> List[GeneratedFeature]:
        dt_series = pd.to_datetime(series, errors="coerce")
        results: List[GeneratedFeature] = []
        options = set(self._datetime_options())

        weekday_names = [
            "Montag",
            "Dienstag",
            "Mittwoch",
            "Donnerstag",
            "Freitag",
            "Samstag",
            "Sonntag",
        ]
        month_names = [
            "Januar",
            "Februar",
            "März",
            "April",
            "Mai",
            "Juni",
            "Juli",
            "August",
            "September",
            "Oktober",
            "November",
            "Dezember",
        ]

        weekday_raw = dt_series.dt.weekday

        if "weekday" in options:
            weekday_series = weekday_raw.map(
                lambda x: weekday_names[int(x)] if pd.notna(x) and 0 <= int(x) < len(weekday_names) else np.nan
            )
            results.append(
                GeneratedFeature(
                    name=f"{feature.name}_weekday",
                    series=self._finalize_series(weekday_series, suffix="Unbekannt"),
                    base=feature,
                    description=f"Wochentag von {feature.name}",
                    value_describer=lambda v: str(v),
                )
            )

        if "is_weekend" in options:
            weekend_series = weekday_raw.map(
                lambda x: "Wochenende" if pd.notna(x) and int(x) in {5, 6} else ("Werktag" if pd.notna(x) else np.nan)
            )
            results.append(
                GeneratedFeature(
                    name=f"{feature.name}_is_weekend",
                    series=self._finalize_series(weekend_series, suffix="Unbekannt"),
                    base=feature,
                    description=f"{feature.name} fällt auf",
                    value_describer=lambda v: str(v),
                )
            )

        if "month" in options:
            month_series = dt_series.dt.month.map(
                lambda x: month_names[int(x) - 1] if pd.notna(x) and 1 <= int(x) <= 12 else np.nan
            )
            results.append(
                GeneratedFeature(
                    name=f"{feature.name}_month",
                    series=self._finalize_series(month_series, suffix="Unbekannt"),
                    base=feature,
                    description=f"Monat von {feature.name}",
                    value_describer=lambda v: str(v),
                )
            )

        if "quarter" in options:
            quarter_series = dt_series.dt.quarter.map(lambda x: f"Q{int(x)}" if pd.notna(x) else np.nan)
            results.append(
                GeneratedFeature(
                    name=f"{feature.name}_quarter",
                    series=self._finalize_series(quarter_series, suffix="Unbekannt"),
                    base=feature,
                    description=f"Quartal von {feature.name}",
                    value_describer=lambda v: str(v),
                )
            )

        if "hour_bucket" in options:
            hour_series = dt_series.dt.hour
            bucketed = pd.cut(
                hour_series,
                bins=[-1, 6, 11, 15, 20, 24],
                labels=["Nacht", "Vormittag", "Nachmittag", "Abend", "Spät"],
            )
            results.append(
                GeneratedFeature(
                    name=f"{feature.name}_daypart",
                    series=self._finalize_series(bucketed.astype(str), suffix="Unbekannt"),
                    base=feature,
                    description=f"Tageszeit von {feature.name}",
                    value_describer=lambda v: str(v),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Numeric
    # ------------------------------------------------------------------

    def _numeric_options(self) -> Dict[str, object]:
        if self.pattern_cfg and getattr(self.pattern_cfg, "numeric_features", None):
            cfg = self.pattern_cfg.numeric_features
            return {
                "is_round": cfg.is_round,
                "quantiles": cfg.quantiles,
                "zero_check": cfg.zero_check,
                "extreme_percentile": cfg.extreme_percentile,
            }
        return {
            "is_round": [0, 2],
            "quantiles": [0.25, 0.5, 0.75],
            "zero_check": True,
            "extreme_percentile": 0.98,
        }

    def _transform_numeric(self, feature: FeatureInfo, series: pd.Series) -> List[GeneratedFeature]:
        numeric_series = pd.to_numeric(series, errors="coerce")
        results: List[GeneratedFeature] = []
        options = self._numeric_options()

        if numeric_series.notna().sum() == 0:
            return results

        if options.get("is_round"):
            decimals_list = options["is_round"]
            for decimals in decimals_list:
                decimals_int = int(decimals)
                rounded = numeric_series.round(decimals_int)
                is_round_mask = numeric_series.notna() & (np.isclose(numeric_series, rounded, atol=10 ** (-(decimals_int + 1))))
                mapped = is_round_mask.map(lambda x: "ja" if x else "nein")
                results.append(
                    GeneratedFeature(
                        name=f"{feature.name}_round_{decimals_int}",
                        series=self._finalize_series(mapped.astype(object), suffix="unbekannt"),
                        base=feature,
                        description=f"{feature.name} rund (auf {decimals_int} Dezimalstellen)",
                        value_describer=lambda v, d=decimals_int: "ist rund" if v == "ja" else "ist nicht rund",
                    )
                )

        if options.get("zero_check"):
            zero_mapped = numeric_series.fillna(np.nan).map(lambda x: "gleich 0" if pd.notna(x) and abs(x) < 1e-9 else "ungleich 0")
            results.append(
                GeneratedFeature(
                    name=f"{feature.name}_zero_check",
                    series=self._finalize_series(zero_mapped.astype(object), suffix="unbekannt"),
                    base=feature,
                    description=f"{feature.name} ist",
                    value_describer=lambda v: str(v),
                )
            )

        quantiles = options.get("quantiles")
        if quantiles:
            try:
                percentiles = sorted({float(q) for q in quantiles if 0 < float(q) < 1})
            except (TypeError, ValueError):
                percentiles = []
            if percentiles:
                bins = [numeric_series.min() - 1]
                labels = []
                q_values = numeric_series.quantile(percentiles).to_list()
                for p, value in zip(percentiles, q_values):
                    bins.append(value)
                    labels.append(f"<= P{int(p*100)}")
                bins.append(numeric_series.max() + 1)
                labels.append(f"> P{int(percentiles[-1]*100)}")
                try:
                    bucketed = pd.cut(numeric_series, bins=bins, labels=labels, include_lowest=True)
                    results.append(
                        GeneratedFeature(
                            name=f"{feature.name}_quantile_bucket",
                            series=self._finalize_series(bucketed.astype(str), suffix="unbekannt"),
                            base=feature,
                            description=f"Quantil-Bucket von {feature.name}",
                            value_describer=lambda v: str(v),
                        )
                    )
                except ValueError:
                    pass

        extreme_p = options.get("extreme_percentile")
        if extreme_p:
            try:
                extreme_p = float(extreme_p)
            except (TypeError, ValueError):
                extreme_p = None
            if extreme_p and 0.5 < extreme_p < 1:
                threshold = numeric_series.quantile(extreme_p)
                if pd.notna(threshold):
                    def describe(v):
                        return "im oberen Extrem" if v == "ja" else "nicht im oberen Extrem"

                    mapped = numeric_series.map(
                        lambda x: "ja" if pd.notna(x) and x >= threshold else "nein"
                    )
                    results.append(
                        GeneratedFeature(
                            name=f"{feature.name}_extreme_high",
                            series=self._finalize_series(mapped.astype(object), suffix="unbekannt"),
                            base=feature,
                            description=f"{feature.name} im oberen {int((1-extreme_p)*100)}%",  # type: ignore[arg-type]
                            value_describer=describe,
                        )
                    )

        return results

    # ------------------------------------------------------------------
    # Categorical
    # ------------------------------------------------------------------

    def _categorical_options(self) -> Dict[str, object]:
        if self.pattern_cfg and getattr(self.pattern_cfg, "categorical_features", None):
            cfg = self.pattern_cfg.categorical_features
            return {"top_k": cfg.top_k, "include_other": cfg.include_other}
        return {"top_k": 10, "include_other": True}

    def _transform_categorical(self, feature: FeatureInfo, series: pd.Series) -> List[GeneratedFeature]:
        series = series.astype(str)
        options = self._categorical_options()
        top_k = int(options.get("top_k", 10))
        include_other = bool(options.get("include_other", True))
        value_counts = series.value_counts(dropna=False)
        top_values = value_counts.nlargest(top_k).index.tolist()
        mapped = series.map(lambda x: x if x in top_values else "Other") if include_other else series
        results = [
            GeneratedFeature(
                name=f"{feature.name}_topk",
                series=self._finalize_series(mapped, suffix="unbekannt"),
                base=feature,
                description=f"Ausprägung von {feature.name}",
                value_describer=lambda v: str(v),
            )
        ]
        return results

    # ------------------------------------------------------------------
    # Boolean
    # ------------------------------------------------------------------

    def _transform_boolean(self, feature: FeatureInfo, series: pd.Series) -> List[GeneratedFeature]:
        mapped = series.map(lambda x: "wahr" if bool(x) else "falsch")
        return [
            GeneratedFeature(
                name=f"{feature.name}_bool",
                series=self._finalize_series(mapped.astype(object), suffix="unbekannt"),
                base=feature,
                description=f"{feature.name}",
                value_describer=lambda v: "trifft zu" if v == "wahr" else "trifft nicht zu",
            )
        ]

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def _text_options(self) -> Dict[str, object]:
        if self.pattern_cfg and getattr(self.pattern_cfg, "text_features", None):
            cfg = self.pattern_cfg.text_features
            return {"length_buckets": cfg.length_buckets, "non_empty": cfg.non_empty}
        return {"length_buckets": [50, 200], "non_empty": True}

    def _transform_text(self, feature: FeatureInfo, series: pd.Series) -> List[GeneratedFeature]:
        str_series = series.fillna("").astype(str)
        options = self._text_options()
        results: List[GeneratedFeature] = []

        if options.get("non_empty"):
            mapped = str_series.map(lambda x: "leer" if len(x.strip()) == 0 else "gefüllt")
            results.append(
                GeneratedFeature(
                    name=f"{feature.name}_non_empty",
                    series=self._finalize_series(mapped.astype(object), suffix="unbekannt"),
                    base=feature,
                    description=f"{feature.name} ist",
                    value_describer=lambda v: str(v),
                )
            )

        length_buckets = options.get("length_buckets") or []
        if length_buckets:
            try:
                breakpoints = sorted({int(x) for x in length_buckets if int(x) > 0})
            except (TypeError, ValueError):
                breakpoints = []
            if breakpoints:
                lengths = str_series.str.len()
                bins = [-1] + breakpoints + [np.inf]
                labels = []
                previous = 0
                for bp in breakpoints:
                    labels.append(f"<= {bp} Zeichen")
                    previous = bp
                labels.append(f"> {breakpoints[-1]} Zeichen")
                bucketed = pd.cut(lengths, bins=bins, labels=labels)
                results.append(
                    GeneratedFeature(
                        name=f"{feature.name}_length_bucket",
                        series=self._finalize_series(bucketed.astype(str), suffix="unbekannt"),
                        base=feature,
                        description=f"Länge von {feature.name}",
                        value_describer=lambda v: str(v),
                    )
                )

        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _finalize_series(series: pd.Series, suffix: str = "Unbekannt") -> pd.Series:
        finalized = series.fillna(suffix).astype(str)
        finalized = finalized.replace({"nan": suffix, "NaT": suffix})
        return finalized


__all__ = ["GeneratedFeature", "InterpretableFeatureGenerator"]
