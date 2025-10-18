"""Gradio interface for the pruefomat Veri pipeline builder."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import textwrap
import unicodedata
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys
import logging
import re

SKIP_GRADIO_IMPORT = os.environ.get("PRUEFOMAT_DISABLE_GRADIO") == "1"

try:
    if SKIP_GRADIO_IMPORT:
        raise ModuleNotFoundError("gradio import disabled via PRUEFOMAT_DISABLE_GRADIO")
    import gradio as gr
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    class _SilentProgress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __call__(self, *args, **kwargs):
            return None

    class _GradioFallback:
        """Minimal fallback to keep non-UI helpers usable when gradio is missing."""

        @staticmethod
        def update(**kwargs):
            return kwargs

        Progress = _SilentProgress

        def __getattr__(self, item):  # pragma: no cover - defensive
            raise ModuleNotFoundError(
                "gradio is required for the interactive UI. "
                "Install it via 'pip install gradio' to enable app.launch()."
            )

    gr = _GradioFallback()
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, fbeta_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config_loader import AppConfig, ConfigError, EvaluationSection, load_config, normalize_config_columns
from build_pipeline_veri import (
    NULL_LIKE_DEFAULT,
    DataFramePreparer,
    DaysUntilDueAdder,
    FeaturePlan,
    build_preprocessor,
    infer_feature_plan,
    normalize_column_name,
    normalize_columns,
)
from logging_utils import configure_logging, get_logger, mask_sensitive_data
from src.patterns import (
    ConditionalProbabilityAnalyzer,
    FeatureTypeDetector,
    InsightFormatter,
    InterpretableFeatureGenerator,
)
from src.business_rules import BusinessRule, load_business_rules_from_file
from src.rule_engine import RuleEngine
from src.explanations import create_explanation_components
from src.hybrid_predictor import HybridMassnahmenPredictor
from src.train_massnahmen import evaluate_multiclass
from src.eval_utils import compute_cost_simulation
from src.train_binary import parse_money


os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_DO_NOT_TRACK", "True")
os.environ.setdefault("GRADIO_DISABLE_USAGE_STATS", "True")

configure_logging(os.environ.get("PRUEFOMAT_LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SYNTH_ROOT = PROJECT_ROOT.parent / "data_synthethizer"
DATA_SYNTH_SRC = DATA_SYNTH_ROOT / "src"
if DATA_SYNTH_SRC.exists() and str(DATA_SYNTH_SRC) not in sys.path:
    sys.path.insert(0, str(DATA_SYNTH_SRC))
try:
    from data_synthethizer.synthetic_data_generator import run_cli as synth_run_cli
except Exception:  # pragma: no cover - optional dependency
    synth_run_cli = None

DEFAULT_SYNTH_CONFIG = DATA_SYNTH_ROOT / "configs" / "default.yaml"
DEFAULT_BUSINESS_RULES_PATH = DATA_SYNTH_ROOT / "configs" / "business_rules.yaml"
DEFAULT_PROFILE_PATH = DATA_SYNTH_ROOT / "configs" / "invoice_profile.yaml"
MASSNAHMEN_RULES_PATH = Path("config/business_rules_massnahmen.yaml")

AUTO_EXCLUDE_FEATURES = {"Manahme", "source_file", "Ticketnummer", "2025"}

CANONICAL_MASSNAHMEN = [
    "Rechnungsprüfung",
    "Beibringung Liefer-/Leistungsnachweis (vorgelagert)",
    "Beibringung Liefer-/Leistungsnachweis (nachgelagert)",
    "Beibringung Auftrag/Bestellung/Vertrag (vorgelagert)",
    "Beibringung Auftrag/Bestellung/Vertrag (nachgelagert)",
    "telefonische Rechnungsbestätigung (vorgelagert)",
    "telefonische Rechnungsbestätigung (nachgelagert)",
    "telefonische Lieferbestätigung (vorgelagert)",
    "telefonische Lieferbestätigung (nachgelagert)",
    "schriftliche Saldenbestätigung (vorgelagert)",
    "schriftliche Saldenbestätigung (nachgelagert)",
    "telefonische Saldenbestätigung (nachgelagert)",
    "schriftliche Rechnungsbestätigung beim DEB (vorgelagert)",
    "schriftliche Rechnungsbestätigung beim DEB (nachgelagert)",
    "Freigabe gemäß Kompetenzkatalog",
    "nur zur Belegerfassung",
    "Gutschriftsverfahren",
]

_MASSNAHME_CANONICAL_LOOKUP = {
    re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()): name
    for name in CANONICAL_MASSNAHMEN
}
_UNKNOWN_MASSNAHMEN_LOGGED: set[str] = set()

COLUMN_CANONICAL_MAP = {
    "Manahme_2025": "Massnahme_2025",
}

FEEDBACK_DB_PATH = Path(json.loads(os.environ.get("PRUEFOMAT_SETTINGS", "{}")).get("feedback_db", "feedback.db"))

LIVE_SCENARIO_DEFAULT = "Keins"

LIVE_SCENARIOS = {
    "Szenario A: Niedrig-Betrag-Grün": {
        "betrag": 15000.0,
        "ampel": "Grün (1)",
        "buk": "A",
        "debitor": "90001",
        "belegdatum": datetime(2025, 1, 1).date(),
        "faelligkeit": datetime(2025, 1, 31).date(),
        "deb_name": "AutoParts GmbH",
        "hinweise": "Routineprüfung",
        "negativ": False,
    },
    "Szenario B: Hoch-Betrag-Grün": {
        "betrag": 95000.0,
        "ampel": "Grün (1)",
        "buk": "B",
        "debitor": "90002",
        "belegdatum": datetime(2025, 2, 10).date(),
        "faelligkeit": datetime(2025, 3, 12).date(),
        "deb_name": "Industriebedarf AG",
        "hinweise": "Großauftrag",
        "negativ": False,
    },
    "Szenario C: Gelbe Ampel": {
        "betrag": 35000.0,
        "ampel": "Gelb (2)",
        "buk": "A",
        "debitor": "90003",
        "belegdatum": datetime(2025, 3, 1).date(),
        "faelligkeit": datetime(2025, 3, 31).date(),
        "deb_name": "Handelspartner KG",
        "hinweise": "Lieferverzug",
        "negativ": False,
    },
    "Szenario D: Rote Ampel": {
        "betrag": 60000.0,
        "ampel": "Rot (3)",
        "buk": "C",
        "debitor": "90004",
        "belegdatum": datetime(2025, 4, 5).date(),
        "faelligkeit": datetime(2025, 5, 5).date(),
        "deb_name": "Risky Supplies Ltd",
        "hinweise": "Vorherige Mahnungen",
        "negativ": True,
    },
    "Szenario E: Gutschrifts-Historie": {
        "betrag": 42000.0,
        "ampel": "Gelb (2)",
        "buk": "A",
        "debitor": "100",
        "belegdatum": datetime(2025, 5, 15).date(),
        "faelligkeit": datetime(2025, 6, 14).date(),
        "deb_name": "Bestandskunde",
        "hinweise": "Historische Gutschriften",
        "negativ": False,
    },
}


def _format_amount_input(value: float) -> str:
    return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def parse_ampel_choice(choice: object) -> Optional[int]:
    if choice is None:
        return None
    if isinstance(choice, (int, float)):
        return int(choice)
    text = str(choice)
    match = re.search(r"(\d+)", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    normalized = text.strip().lower()
    mapping = {
        "grün": 1,
        "gruen": 1,
        "gelb": 2,
        "rot": 3,
    }
    return mapping.get(normalized)


def balance_classes_dataframe(df: pd.DataFrame, target_col: str, seed: Optional[int] = None) -> pd.DataFrame:
    """Return a DataFrame where all classes in ``target_col`` have equal counts."""

    if target_col not in df.columns:
        return df

    counts = df[target_col].value_counts(dropna=False)
    if counts.empty:
        return df

    max_count = int(counts.max())
    if max_count <= 1:
        return df

    rng = np.random.default_rng(seed if seed is not None else 42)
    balanced_parts = []
    for _, group in df.groupby(target_col, dropna=False, sort=False):
        size = len(group)
        if size == max_count:
            balanced_parts.append(group)
            continue
        random_state = int(rng.integers(0, 1_000_000))
        sampled = group.sample(max_count, replace=True, random_state=random_state)
        balanced_parts.append(sampled)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    shuffled = balanced_df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
    return shuffled.reset_index(drop=True)


def _default_live_form() -> Dict[str, Any]:
    today = datetime.utcnow().date()
    return {
        "betrag": 25000.0,
        "ampel": "Gelb (2)",
        "buk": "A",
        "debitor": "12345",
        "belegdatum": today,
        "faelligkeit": today + timedelta(days=30),
        "deb_name": "",
        "hinweise": "",
        "negativ": False,
    }


def ensure_feedback_db():
    FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                beleg_index INTEGER,
                beleg_id TEXT,
                timestamp TEXT,
                user TEXT,
                score REAL,
                prediction INTEGER,
                feedback TEXT,
                comment TEXT,
                actual_label TEXT,
                predicted_label TEXT,
                is_correct INTEGER
            )
            """
        )
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(feedback)")
        }
        optional_columns = {
            "actual_label": "TEXT",
            "predicted_label": "TEXT",
            "is_correct": "INTEGER",
            "score": "REAL",
            "comment": "TEXT",
            "feedback": "TEXT",
        }
        for column, definition in optional_columns.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE feedback ADD COLUMN {column} {definition}")
        conn.commit()


def save_feedback_to_db(entry: Dict[str, Any]) -> None:
    ensure_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                beleg_index,
                beleg_id,
                timestamp,
                user,
                score,
                prediction,
                feedback,
                comment,
                actual_label,
                predicted_label,
                is_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("beleg_index"),
                entry.get("beleg_id"),
                entry.get("timestamp"),
                entry.get("user"),
                entry.get("score"),
                entry.get("prediction"),
                entry.get("feedback"),
                entry.get("comment"),
                entry.get("actual_label"),
                entry.get("predicted_label"),
                int(bool(entry.get("is_correct"))),
            ),
        )
        conn.commit()


def load_feedback_from_db() -> pd.DataFrame:
    ensure_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM feedback", conn, parse_dates=["timestamp"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# Configuration setup
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")
BIAS_PROMPT_MODEL = "gpt-4.1-mini"


def _load_app_config(path: Optional[str | Path]) -> AppConfig:
    return normalize_config_columns(load_config(path))


try:
    DEFAULT_CONFIG = _load_app_config(DEFAULT_CONFIG_PATH)
except ConfigError:
    DEFAULT_CONFIG = AppConfig()
    DEFAULT_CONFIG = normalize_config_columns(DEFAULT_CONFIG)


def _initial_state() -> Dict[str, Any]:
    return {
        "config": DEFAULT_CONFIG.model_copy(deep=True),
        "config_path": str(DEFAULT_CONFIG_PATH),
        "balance_classes": False,
        "bias_rules_yaml": "",
        "hybrid_predictor": None,
        "rule_engine": None,
        "rule_metrics": {},
        "massnahmen_distribution": None,
        "label_encoder": None,
        "label_classes": [],
    }


TRAINING_STATE_KEYS = {
    "excel_paths",
    "excel_path",
    "sheet",
    "column_mapping",
    "df_features",
    "df_features_full",
    "target",
    "target_name",
    "target_mapping",
    "selected_columns",
    "preprocessor",
    "feature_plan",
    "prep_path",
    "baseline_model",
    "baseline_path",
    "metrics_path",
    "feature_importance_df",
    "notes_text_vectorizer",
    "shap_background",
    "shap_feature_names",
    "pattern_insights_df",
    "pattern_insights_path",
    "hybrid_predictor",
    "rule_engine",
    "rule_metrics",
    "massnahmen_distribution",
    "label_encoder",
    "label_classes",
    "hybrid_results",
    "business_rules",
    "historical_training_data",
}

SYNTH_STATE_KEYS = {
    "last_generated_path",
    "last_quality_path",
    "bias_rules_yaml",
    "synthetic_balanced",
    "synthetic_balance_requested",
}


# ---------------------------------------------------------------------------
# Gradio schema monkey-patch (4.44.0 bug fix)
# ---------------------------------------------------------------------------
try:
    from gradio_client import utils as grc_utils

    _ORIG_JSON_SCHEMA_TO_PYTHON_TYPE = grc_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):  # type: ignore[override]
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        return _ORIG_JSON_SCHEMA_TO_PYTHON_TYPE(schema, defs)

    grc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
except (ImportError, AttributeError):
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sheet(sheet_text: str) -> int | str:
    text = (sheet_text or "").strip()
    return int(text) if text.isdigit() else text or 0


def _ensure_config(state: Dict[str, Any]) -> AppConfig:
    config = state.get("config")
    if config is None:
        config = DEFAULT_CONFIG.model_copy(deep=True)
        state["config"] = config
        state["config_path"] = str(DEFAULT_CONFIG_PATH)
    return config


def _format_schema(df: pd.DataFrame) -> Dict[str, Any]:
    details = []
    total_rows = len(df)
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        details.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null": non_null,
                "null_pct": float(100 * (1 - non_null / total_rows)) if total_rows else 0.0,
            }
        )
    return {"rows": total_rows, "columns": len(df.columns), "fields": details}


def _canonicalize_column_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    return COLUMN_CANONICAL_MAP.get(name, name)


def normalize_massnahme(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()) or (not isinstance(value, str) and pd.isna(value)):
        return "Unbekannt"

    text = str(value).strip()

    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalized_key = re.sub(r"[^a-z0-9]", "", ascii_text.lower())
    match = _MASSNAHME_CANONICAL_LOOKUP.get(normalized_key)
    if match:
        return match

    if text not in _UNKNOWN_MASSNAHMEN_LOGGED:
        _UNKNOWN_MASSNAHMEN_LOGGED.add(text)
        logger.warning("massnahme_unknown", value=text)
    return "Unbekannt"


def canonicalize_massnahmen(series: pd.Series) -> pd.Series:
    return series.apply(normalize_massnahme)


def _reset_pipeline_state(state: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("preprocessor", "feature_plan", "prep_path", "baseline_model", "baseline_path", "metrics_path"):
        state.pop(key, None)
    return state


def _clear_state_keys(state: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    for key in keys:
        state.pop(key, None)
    return state


def _serialize_component_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return {
            "columns": list(value.columns),
            "rows": int(len(value)),
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialize_component_value(item) for item in value]
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, val in value.items():
            if key in {"data", "file", "binary", "base64"}:
                continue
            sanitized[key] = _serialize_component_value(val)
        return sanitized
    if hasattr(value, "name") and hasattr(value, "size"):
        return {
            "name": getattr(value, "name", ""),
            "size": getattr(value, "size", None),
        }
    try:
        return value.__dict__
    except AttributeError:
        return str(value)


def _build_inputs_payload(state: Dict[str, Any], sections: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    serialized_sections = {
        section: {name: _serialize_component_value(val) for name, val in values.items()}
        for section, values in sections.items()
    }
    snapshot = {
        "created_at": timestamp,
        "config_path": state.get("config_path"),
        "target": state.get("target_name"),
        "balance_classes": state.get("balance_classes", False),
        "sections": serialized_sections,
    }
    return snapshot


def _write_inputs_payload(payload: Dict[str, Any]) -> str:
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_inputs_"))
    file_path = tmp_dir / "inputs_snapshot.json"
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(file_path)


def _format_shap_table(entries: Optional[list[tuple[str, float]]]) -> str:
    if not entries:
        return "- Keine SHAP-Werte verfügbar."
    lines = ["| Feature | SHAP-Wert |", "| --- | --- |"]
    for feature, value in entries:
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            value_float = 0.0
        lines.append(f"| {feature} | {value_float:.4f} |")
    return "\n".join(lines)


def _format_conditions(conditions: Optional[list[Dict[str, Any]]]) -> str:
    if not conditions:
        return "- Keine Bedingungen ausgewertet."
    lines = []
    for entry in conditions:
        matched = bool(entry.get("matched"))
        status = "✅" if matched else "❌"
        operator = entry.get("operator", "=")
        target = entry.get("target")
        value = entry.get("value")
        field = entry.get("field", "")
        lines.append(
            f"{status} `{field} {operator}` (Soll: {target} | Ist: {value})"
        )
    return "\n".join(lines)


def _load_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _tooltip_label(text: str, tip: str) -> str:
    return (
        "<div class='pf-field-label'>"
        f"{text}"
        "<span class='pf-tooltip'>ℹ️"
        f"<span class='pf-tooltiptext'>{tip}</span>"
        "</span>"
        "</div>"
    )


def _slugify_label(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    safe = "".join(ch if ch.isalnum() else "_" for ch in ascii_value.lower())
    return safe.strip("_") or "massnahme"


def _compute_split_params(target: pd.Series, default_test_size: float = 0.2) -> tuple[int, Optional[pd.Series]]:
    """Return a safe test size and optional stratify series for tiny datasets."""

    clean_target = target.dropna()
    n_samples = len(clean_target)
    if n_samples < 2:
        return 1, None

    test_size_count = max(1, int(round(n_samples * default_test_size)))
    if test_size_count >= n_samples:
        test_size_count = max(1, n_samples - 1)

    class_counts = clean_target.value_counts()
    if class_counts.empty:
        return test_size_count, None

    n_classes = len(class_counts)
    min_class = class_counts.min()
    if min_class < 2 or test_size_count < n_classes:
        return test_size_count, None

    return test_size_count, target


class BiasPromptError(RuntimeError):
    """Raised when a natural-language bias prompt cannot be transformed."""


def _extract_response_text(response: Any) -> Optional[str]:
    """Best-effort extraction for both Responses and ChatCompletion payloads."""

    if response is None:
        return None
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    if hasattr(response, "output"):
        chunks: List[str] = []
        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) in {"text", "output_text"}:
                        chunks.append(getattr(content, "text", ""))
            elif item_type in {"output_text", "text"}:
                chunks.append(getattr(item, "text", ""))
        combined = " ".join(chunk.strip() for chunk in chunks if chunk)
        if combined.strip():
            return combined.strip()

    data_attr = getattr(response, "data", None)
    if data_attr:
        chunks: List[str] = []
        for item in data_attr:
            item_dict = item
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif hasattr(item, "dict"):
                item_dict = item.dict()
            if isinstance(item_dict, dict):
                content = item_dict.get("content")
                if isinstance(content, list):
                    for entry in content:
                        if isinstance(entry, dict) and entry.get("type") in {"text", "output_text"}:
                            chunks.append(str(entry.get("text", "")))
                elif isinstance(content, dict) and content.get("type") in {"text", "output_text"}:
                    chunks.append(str(content.get("text", "")))
        combined = " ".join(chunk.strip() for chunk in chunks if chunk)
        if combined.strip():
            return combined.strip()

    choices = getattr(response, "choices", None)
    if choices:
        choice0 = choices[0]
        message = getattr(choice0, "message", None)
        if message and getattr(message, "content", None):
            return str(message.content).strip()
        text = getattr(choice0, "text", None)
        if text:
            return str(text).strip()

    if hasattr(response, "model_dump"):  # pragma: no cover - defensive
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            output_text = dumped.get("output_text")
            if output_text:
                return str(output_text).strip()
            data = dumped.get("data")
            if isinstance(data, list):
                chunks = []
                for entry in data:
                    if isinstance(entry, dict):
                        content = entry.get("content")
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                                    chunks.append(str(part.get("text", "")))
                combined = " ".join(chunk.strip() for chunk in chunks if chunk)
                if combined.strip():
                    return combined.strip()

    return None


def _extract_json_block(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    sanitized = text.strip()
    if sanitized.startswith("```"):
        parts = sanitized.split("```")
        if len(parts) >= 2:
            sanitized = parts[1].strip()
    if sanitized.lower().startswith("json"):
        sanitized = sanitized[4:].lstrip()
    if sanitized.startswith("{") and sanitized.endswith("}"):
        return sanitized
    start_idx = sanitized.find("{")
    end_idx = sanitized.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return sanitized[start_idx : end_idx + 1]
    return None


def _preview_text(value: Any, limit: int = 800) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, (dict, list)):
            snippet = json.dumps(value, ensure_ascii=False)
        else:
            snippet = str(value)
    except Exception:  # pragma: no cover - defensive fallback
        snippet = repr(value)
    snippet = snippet.strip()
    if len(snippet) > limit:
        return snippet[: limit - 3] + "..."
    return snippet


def _collect_excel_columns(path: Path, max_sheets: int = 3, sample_rows: int = 5) -> List[str]:
    columns: List[str] = []
    ext = path.suffix.lower()
    try:
        if ext in {".csv"}:
            frame = pd.read_csv(path, nrows=sample_rows)
            columns.extend(str(col) for col in frame.columns)
        elif ext in {".xls", ".xlsx", ".xlsm", ".xlsb"}:
            workbook = pd.ExcelFile(path)
            for sheet_name in workbook.sheet_names[:max_sheets]:
                try:
                    frame = workbook.parse(sheet_name, nrows=sample_rows)
                except Exception:
                    continue
                columns.extend(str(col) for col in frame.columns)
        else:
            frame = pd.read_excel(path, nrows=sample_rows)
            columns.extend(str(col) for col in frame.columns)
    except Exception:
        columns = []

    if not columns:
        return []
    deduped: List[str] = []
    seen = set()
    for col in columns:
        if col not in seen:
            deduped.append(col)
            seen.add(col)
    return deduped


def _call_bias_llm(
    *,
    prompt: str,
    columns: List[str],
    existing_rule_names: List[str],
    model: str,
    api_key: str,
) -> Dict[str, Any]:
    model = BIAS_PROMPT_MODEL
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BiasPromptError(
            "Für Bias-Prompts wird das 'openai'-Paket benötigt. Bitte installiere es oder deaktiviere den Bias-Prompt."
        ) from exc

    client = OpenAI(api_key=api_key)
    column_text = "\n".join(f"- {name}" for name in columns) if columns else "- (keine Spalten erkannt)"
    existing_text = ", ".join(existing_rule_names) if existing_rule_names else "(keine vorhanden)"

    helper_doc = (
        "Hilfsfunktionen: weekday(Belegdatum) -> 0..6 (Montag=0), "
        "isoweekday(Belegdatum) -> 1..7 (Montag=1), "
        "is_round_amount(Betrag, decimals=2) -> bool."
    )

    system_prompt = textwrap.dedent(
        """
        Du wandelst natürliche Sprache in strukturierte Regeln für einen Syntheseprozessor um.
        Erzeuge JSON mit den Feldern "rules" und optional "dependencies".
        Jede Regel: {"name": str, "columns": [Spaltennamen], "condition": str}.
        Form von Wahrscheinlichkeiten: "if <Bedingung> then <Spalte> in [Werte] with probability <0.x>".
        Verwende ausschließlich die aufgeführten Spaltennamen und Hilfsfunktionen.
        Prozentangaben wie 5% → 0.05, 30 Prozent → 0.3.
        Keine freien Texte, nur valides JSON.
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        Verfügbare Spalten:
        {column_text}

        Vorhandene Regel-Namen: {existing_text}
        {helper_doc}

        Anweisung:
        {prompt.strip()}

        Gib ausschließlich JSON zurück.
        """
    ).strip()

    if not hasattr(client, "responses"):
        raise BiasPromptError(
            "Das konfigurierte Modell unterstützt die Responses API nicht – Bias-Prompts stehen daher nicht zur Verfügung."
        )

    json_schema = {
        "name": "bias_rules",
        "schema": {
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                            "condition": {"type": "string"},
                        },
                        "required": ["name", "columns", "condition"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                },
                "dependencies": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "formula": {"type": "string"},
                        },
                        "required": ["formula"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["rules"],
            "additionalProperties": False,
        },
    }

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            response_format={"type": "json_schema", "json_schema": json_schema},
            max_output_tokens=256,
            parallel_tool_calls=False,
            tool_choice="none",
        )
    except TypeError as exc:
        if "response_format" in str(exc):
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                max_output_tokens=256,
                parallel_tool_calls=False,
                tool_choice="none",
            )
        else:
            raise BiasPromptError(f"OpenAI-Anfrage fehlgeschlagen: {exc}") from exc
    except Exception as exc:  # pragma: no cover - runtime variability
        raise BiasPromptError(f"OpenAI-Anfrage fehlgeschlagen: {exc}") from exc

    text = _extract_response_text(response)
    json_text = _extract_json_block(text) if text else None
    candidate = json_text or text
    if not candidate and hasattr(client, "responses"):
        try:
            retry_resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                response_format={"type": "json_object"},
                max_output_tokens=512,
            )
            retry_text = _extract_response_text(retry_resp)
            json_text = _extract_json_block(retry_text) if retry_text else None
            candidate = json_text or retry_text
            if candidate:
                response = retry_resp
                text = retry_text
        except Exception:
            pass

    if not candidate:
        raw_dump = None
        if hasattr(response, "model_dump"):
            try:
                raw_dump = response.model_dump()
            except Exception:  # pragma: no cover - defensive
                raw_dump = None
        raw_excerpt = _preview_text(text) or _preview_text(raw_dump) or _preview_text(
            getattr(response, "output", None)
        ) or _preview_text(getattr(response, "data", None)) or _preview_text(response)
        raise BiasPromptError(
            "LLM lieferte kein gültiges JSON zurück. Prompt:\n"
            f"{prompt.strip()}\nAntwort-Auszug:\n{raw_excerpt}"
        )
    candidate = candidate.strip()
    candidate = _repair_json_candidate(candidate)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        try:
            payload = yaml.safe_load(candidate)
        except yaml.YAMLError as exc:
            raise BiasPromptError(
                "JSON konnte nicht geparst werden: "
                f"{exc}. Prompt:\n{prompt.strip()}\nAntwort-Auszug:\n{candidate.strip()}"
            ) from exc
        if not isinstance(payload, (dict, list)):
            raise BiasPromptError(
                "LLM lieferte kein gültiges JSON zurück. Prompt:\n"
                f"{prompt.strip()}\nAntwort-Auszug:\n{candidate.strip()}"
            ) from exc
        json_text = json_text or yaml.safe_dump(payload)

    if not json_text and isinstance(payload, dict):
        json_text = json.dumps(payload, ensure_ascii=False)

    if isinstance(payload, list):
        payload = {"rules": payload}

    if not isinstance(payload, dict):
        raise BiasPromptError(
            "Ergebnis muss ein JSON-Objekt sein. Prompt:\n"
            f"{prompt.strip()}\nAntwort-Auszug:\n{candidate.strip()}"
        )
    return payload


def _repair_json_candidate(candidate: str) -> str:
    trimmed = candidate.strip()
    if not trimmed:
        return trimmed
    # drop trailing unmatched quote
    if trimmed.endswith('"') and trimmed.count('"') % 2 == 1:
        trimmed = trimmed[:-1]
    # balance brackets
    diff_curly = trimmed.count('{') - trimmed.count('}')
    if diff_curly > 0:
        trimmed += '}' * diff_curly
    diff_bracket = trimmed.count('[') - trimmed.count(']')
    if diff_bracket > 0:
        trimmed += ']' * diff_bracket
    return trimmed


def _heuristic_bias_from_prompt(
    prompt: str,
    columns: List[str],
    df_features: Optional[pd.DataFrame],
    target_name: str,
) -> Dict[str, Any]:
    text_lower = prompt.lower()
    prob_match = re.search(r"(\d+[\.,]?\d*)\s*%", text_lower)
    probability = 0.5
    if prob_match:
        value = prob_match.group(1).replace(",", ".")
        try:
            probability = float(value)
            if probability > 1.0:
                probability /= 100.0
        except ValueError:
            probability = 0.5
    else:
        fraction_match = re.search(r"(\d+[\.,]?\d*)\s*(?:fach|mal)", text_lower)
        if fraction_match:
            try:
                probability = float(fraction_match.group(1))
                if probability > 1.0:
                    probability = min(probability / 2.0, 1.0)
            except ValueError:
                probability = 0.5

    probability = max(0.01, min(probability, 1.0))

    target_map = {
        "rot": "3",
        "gelb": "2",
        "grün": "1",
        "gruen": "1",
    }
    target_values: List[str] = []
    for word, mapped in target_map.items():
        if word in text_lower:
            target_values.append(mapped)
    if not target_values:
        target_values = ["3"]

    condition_text = prompt
    match_wenn = re.search(r"(?:wenn|if)\s+(.+)", prompt, flags=re.IGNORECASE)
    if match_wenn:
        condition_text = match_wenn.group(1).strip().rstrip(".")

    preferred_keywords = [
        "name",
        "firma",
        "company",
        "deb",
        "kunde",
        "customer",
        "type",
    ]

    candidate_columns: List[str] = []

    def _add_candidate(raw_name: str) -> None:
        if not raw_name:
            return
        normalized = normalize_column_name(raw_name) or raw_name
        if normalized not in candidate_columns:
            candidate_columns.append(normalized)

    for col in columns:
        lower_col = col.lower()
        if any(keyword in lower_col for keyword in preferred_keywords):
            _add_candidate(col)
    if df_features is not None:
        for col in df_features.columns:
            lower_col = str(col).lower()
            if any(keyword in lower_col for keyword in preferred_keywords):
                _add_candidate(str(col))
    if not candidate_columns and columns:
        _add_candidate(columns[0])

    condition_parts = []
    negation_regex = re.compile(r"(kein|keine|keinen|ohne|nicht)\s+[^.,;]*gmbh")
    has_gmbh = "gmbh" in text_lower
    is_negated = bool(negation_regex.search(text_lower))

    if has_gmbh:
        token = '"GmbH"'
        operator = "not in" if is_negated else "in"
        for col in candidate_columns[:1]:
            condition_parts.append(f"{token} {operator} {col}")
    elif candidate_columns:
        condition_parts.append(f"{candidate_columns[0]} != ''")
    else:
        condition_parts.append("True")

    condition_expr = " and ".join(condition_parts)
    rule_name = prompt.strip().split("\n")[0][:60] or "Bias Regel"

    target_column_normalized = normalize_column_name(target_name) or target_name
    columns_used = candidate_columns[:1] + [target_column_normalized]
    deduped_columns = list(dict.fromkeys(columns_used))

    rule = {
        "name": rule_name,
        "columns": deduped_columns,
        "condition": f"if {condition_expr} then {target_name} in [{', '.join(target_values)}] with probability {probability}",
    }

    return {"rules": [rule], "dependencies": {}}


def _normalise_bias_patch(data: Dict[str, Any]) -> Dict[str, Any]:
    rules_raw = data.get("rules", []) if isinstance(data, dict) else []
    dependencies_raw = data.get("dependencies", {}) if isinstance(data, dict) else {}

    rules: List[Dict[str, Any]] = []
    for entry in rules_raw or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "Bias-Regel")
        columns = entry.get("columns", [])
        if isinstance(columns, (list, tuple)):
            column_list = [str(col) for col in columns if str(col).strip()]
        else:
            column_list = [str(columns)] if columns else []
        condition = str(entry.get("condition", "")).strip()
        if not condition:
            continue
        rule = {
            "name": name,
            "columns": column_list,
            "condition": condition,
        }
        rules.append(rule)

    dependencies: Dict[str, Dict[str, Any]] = {}
    if isinstance(dependencies_raw, dict):
        for key, value in dependencies_raw.items():
            if not isinstance(value, dict):
                continue
            formula = value.get("formula")
            if not formula:
                continue
            depends_on = value.get("depends_on", [])
            if isinstance(depends_on, (list, tuple)):
                depends_list = [str(dep) for dep in depends_on if str(dep).strip()]
            else:
                depends_list = [str(depends_on)] if depends_on else []
            dependencies[str(key)] = {
                "depends_on": depends_list,
                "formula": str(formula),
            }

    return {"rules": rules, "dependencies": dependencies}


def _parse_business_rules_block(content: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not content.strip():
        base: Dict[str, Any] = {"business_rules": {}}
        return base, base["business_rules"]
    try:
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if "business_rules" in data and isinstance(data["business_rules"], dict):
        return data, data["business_rules"]
    business_block = data
    wrapper = {"business_rules": business_block}
    return wrapper, business_block


def _apply_bias_patch(
    data: Dict[str, Any],
    business_block: Dict[str, Any],
    patch: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rules_list = business_block.setdefault("rules", [])
    if not isinstance(rules_list, list):
        rules_list = business_block["rules"] = []
    dependencies_map = business_block.setdefault("dependencies", {})
    if not isinstance(dependencies_map, dict):
        dependencies_map = business_block["dependencies"] = {}

    existing_names = {
        str(rule.get("name"))
        for rule in rules_list
        if isinstance(rule, dict) and rule.get("name")
    }

    added_rules: List[Dict[str, Any]] = []
    for entry in patch.get("rules", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "Bias-Regel")
        original_name = name
        suffix = 2
        while name in existing_names:
            name = f"{original_name} ({suffix})"
            suffix += 1
        columns = entry.get("columns", [])
        if isinstance(columns, (list, tuple)):
            column_list = [str(col) for col in columns if str(col).strip()]
        else:
            column_list = [str(columns)] if columns else []
        condition = str(entry.get("condition", "")).strip()
        if not condition:
            continue
        rule_copy = {
            "name": name,
            "columns": column_list,
            "condition": condition,
        }
        rules_list.append(rule_copy)
        added_rules.append(rule_copy)
        existing_names.add(name)

    added_dependencies: Dict[str, Dict[str, Any]] = {}
    for key, value in patch.get("dependencies", {}).items():
        if not isinstance(value, dict):
            continue
        formula = value.get("formula")
        if not formula:
            continue
        depends_on = value.get("depends_on", [])
        if isinstance(depends_on, (list, tuple)):
            depends_list = [str(dep) for dep in depends_on if str(dep).strip()]
        else:
            depends_list = [str(depends_on)] if depends_on else []
        normalized = {"depends_on": depends_list, "formula": str(formula)}
        dependencies_map[str(key)] = normalized
        added_dependencies[str(key)] = normalized

    return added_rules, added_dependencies


def _build_config_overview(state: Dict[str, Any]) -> Dict[str, Any]:
    config_path = state.get("config_path")
    target_name = state.get("target_name")
    selected = state.get("selected_columns") or []
    overview: Dict[str, Any] = {
        "config_path": config_path,
        "target_col": target_name,
        "selected_columns": selected,
        "balance_classes": bool(state.get("balance_classes")),
    }
    mapping = state.get("target_mapping")
    if mapping:
        overview["target_mapping"] = mapping
    return overview


def _extract_upload_path(item) -> Optional[Path]:
    """Return a filesystem path for a Gradio file-like upload.

    Adds detailed debug logging so UI upload issues can be diagnosed from the
    CLI logs.
    """

    if item is None:
        logger.debug("upload_item_none")
        return None

    cached = getattr(item, "_resolved_path", None)
    if cached:
        cached_path = Path(cached)
        if cached_path.exists():
            logger.debug("upload_item_cached", path=str(cached_path))
            return cached_path

    if isinstance(item, (str, Path)):
        path = Path(item)
        logger.debug("upload_item_pathlike", path=str(path), exists=path.exists())
        return path if path.exists() else None

    logger.debug("upload_item_type", item_type=type(item).__name__)

    candidate_attrs = (
        "name",
        "path",
        "temp_file_path",
        "tmp_path",
        "temporary_file_path",
    )

    for attr in candidate_attrs:
        value = getattr(item, attr, None)
        logger.debug(
            "upload_item_attr",
            attr=attr,
            present=value is not None,
            value=str(value) if isinstance(value, str) else None,
        )
        if isinstance(value, str):
            path = Path(value)
            if path.exists():
                logger.debug("upload_path_resolved", attr=attr, path=str(path))
                try:
                    setattr(item, "_resolved_path", str(path))
                except Exception:  # pragma: no cover - cache best-effort
                    pass
                return path

    # Gradio >=4 exposes `data` (bytes) + `orig_name` when running without a temp file.
    data = getattr(item, "data", None)
    if isinstance(data, bytes):
        suffix = (
            Path(getattr(item, "orig_name", "")).suffix
            or Path(getattr(item, "name", "")).suffix
            or ".xlsx"
        )
        logger.debug(
            "upload_item_bytes",
            orig_name=getattr(item, "orig_name", None),
            name=getattr(item, "name", None),
            size=len(data),
            suffix=suffix,
        )
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.flush()
            tmp.close()
            logger.debug("upload_tempfile_created", path=tmp.name)
            try:
                setattr(item, "_resolved_path", tmp.name)
            except Exception:  # pragma: no cover - cache best-effort
                pass
            return Path(tmp.name)
        except Exception as exc:  # pragma: no cover - defensive fallback creation
            logger.error("upload_tempfile_failed", message=str(exc))
            return None

    logger.debug("upload_item_unresolved", item_repr=repr(item))
    return None


def _collect_excel_files(upload, folder_text: str | None) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    warnings_list: list[str] = []

    if upload is not None:
        uploads = upload if isinstance(upload, list) else [upload]
        for item in uploads:
            logger.debug(
                "ui_upload_inspect",
                item_type=type(item).__name__,
                has_name=hasattr(item, "name"),
                has_path=hasattr(item, "path"),
            )
            path = _extract_upload_path(item)
            if path is None or not path.exists():
                display_name = getattr(item, "orig_name", None) or getattr(item, "name", "unbekannt")
                warnings_list.append(f"Datei nicht gefunden: {display_name}")
                continue
            if path.suffix.lower() not in {".xlsx", ".xls"}:
                warnings_list.append(f"Überspringe nicht unterstützte Datei: {path.name}")
                continue
            paths.append(path)

    if folder_text:
        folder = Path(folder_text).expanduser()
        if folder.exists() and folder.is_file() and folder.suffix.lower() in {".xlsx", ".xls"}:
            paths.append(folder)
        elif folder.exists() and folder.is_dir():
            excel_files = sorted(
                {p for ext in ("*.xlsx", "*.xls") for p in folder.rglob(ext)}
            )
            if not excel_files:
                warnings_list.append(f"Im Ordner wurden keine Excel-Dateien gefunden: {folder}")
            else:
                paths.extend(excel_files)
        else:
            warnings_list.append(f"Ordner/Datei nicht gefunden: {folder}")

    deduped: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped, warnings_list


def _balance_training_set(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if X.empty or y.empty:
        return X, y
    df = X.copy()
    df["__target__"] = y.reset_index(drop=True)
    counts = df["__target__"].value_counts()
    max_count = counts.max()
    rng = np.random.default_rng(random_state or 42)
    balanced_parts = []
    for value, count in counts.items():
        subset = df[df["__target__"] == value]
        if count < max_count:
            sample_indices = rng.choice(subset.index, size=max_count - count, replace=True)
            extra = subset.loc[sample_indices].copy()
            balanced_parts.extend([subset, extra])
        else:
            balanced_parts.append(subset)
    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    y_balanced = balanced_df.pop("__target__")
    return balanced_df.reset_index(drop=True), y_balanced.reset_index(drop=True)


def load_dataset(upload, config_upload, sheet_text: str, target_text: str, folder_text: str, state: Optional[Dict[str, Any]]):
    state = state or {}

    def _empty_result(message: str) -> tuple[Any, ...]:
        return (
            message,
            None,
            None,
            gr.update(value={}),
            gr.update(value=[], choices=[]),
            "",
            gr.update(value=False),
            state,
        )

    print("=" * 80, flush=True)
    print("CALLBACK WURDE AUFGERUFEN!", flush=True)
    print(f"Received upload: {upload}", flush=True)
    print(f"Type: {type(upload)}", flush=True)
    print("=" * 80, flush=True)
    print("DEBUG load_dataset invoked", flush=True)
    logger.debug(
        "ui_load_clicked",
        upload_type=type(upload).__name__ if upload is not None else None,
        folder=folder_text,
        config_upload_type=type(config_upload).__name__ if config_upload is not None else None,
    )

    excel_paths, warnings_list = _collect_excel_files(upload, folder_text)
    if not excel_paths:
        logger.warning("ui_no_file", folder=folder_text or None)
        warn = warnings_list[0] if warnings_list else "Bitte zuerst eine Excel-Datei oder einen Ordner auswählen."
        return _empty_result(warn)

    config_path: Optional[str]
    if config_upload is not None:
        config_path_obj = _extract_upload_path(config_upload)
        if config_path_obj is None:
            logger.warning("ui_config_upload_missing")
            return _empty_result("Konfigurationsdatei nicht gefunden.")
        config_path = str(config_path_obj)
    else:
        config_path = state.get("config_path", str(DEFAULT_CONFIG_PATH))

    try:
        config = _load_app_config(config_path)
    except ConfigError as exc:
        logger.error("ui_config_error", message=str(exc), config_path=config_path)
        return _empty_result(f"Konfigurationsfehler: {exc}")

    config.data.target_col = _canonicalize_column_name(config.data.target_col)

    if target_text:
        normalised_target = normalize_column_name(target_text)
        config.data.target_col = _canonicalize_column_name(normalised_target)

    logger.info(
        "ui_config_loaded",
        config_path=config_path,
        target_col=config.data.target_col,
    )

    state["config"] = config
    state["config_path"] = config_path

    sheet = _parse_sheet(sheet_text)
    frames: list[pd.DataFrame] = []
    column_mappings: dict[str, dict[str, str]] = {}
    total_rows = 0
    errors: list[str] = []

    for idx, path in enumerate(excel_paths, start=1):
        try:
            df_raw = pd.read_excel(path, sheet_name=sheet, dtype="object")
        except Exception as exc:  # pragma: no cover
            message = f"Fehler beim Laden {path.name}: {exc}"
            logger.error(
                "ui_excel_error",
                message=str(exc),
                file=str(path),
                sheet=sheet,
            )
            errors.append(message)
            continue

        df_norm, column_mapping = normalize_columns(df_raw)
        for alias, canonical in COLUMN_CANONICAL_MAP.items():
            if alias in df_norm.columns and canonical not in df_norm.columns:
                df_norm = df_norm.rename(columns={alias: canonical})
                column_mapping = {orig: (canonical if mapped == alias else mapped) for orig, mapped in column_mapping.items()}
        df_norm["source_file"] = path.name
        frames.append(df_norm)
        column_mappings[path.name] = column_mapping
        total_rows += len(df_norm)
        logger.info(
            "ui_dataset_loaded",
            file=str(path),
            sheet=sheet,
            rows=len(df_norm),
            columns=df_norm.shape[1],
        )

    if not frames:
        first_error = errors[0] if errors else "Keine gültigen Excel-Dateien gefunden."
        if warnings_list:
            errors = warnings_list + errors
        return _empty_result(first_error)

    df_norm_all = pd.concat(frames, ignore_index=True, sort=False)
    if "source_file" in df_norm_all.columns:
        df_norm_all["source_file"] = df_norm_all["source_file"].fillna("unbekannt")

    target_norm = config.data.target_col
    target_series = None
    df_features = df_norm_all
    target_mapping: Dict[str, int] | None = None
    state.pop("target_mapping", None)
    target_msg = "Keine Zielspalte gesetzt."
    mass_target_norm = normalize_column_name("Massnahme_2025")

    if target_norm and target_norm in df_norm_all.columns:
        if target_norm == mass_target_norm:
            raw_target = df_norm_all[target_norm].astype("string").str.strip()
            raw_target = raw_target.replace({"": pd.NA})
            mask_mass = raw_target.notna()
            if mask_mass.any():
                target_series = raw_target.loc[mask_mass]
                df_features = df_norm_all.loc[mask_mass].drop(columns=[target_norm])
                target_msg = f"Zielspalte erkannt: {target_norm} (mehrklassig)"
                logger.info("ui_target_detected", target=target_norm)
                unique_values = list(dict.fromkeys(target_series.dropna().astype(str)))
                target_mapping = {str(val): idx for idx, val in enumerate(unique_values, start=1)}
                if target_mapping:
                    state["target_mapping"] = target_mapping
            else:
                target_msg = f"Zielspalte '{target_norm}' enthält keine nutzbaren Werte."
                df_features = df_norm_all.drop(columns=[target_norm])
                logger.warning("ui_target_empty", target=target_norm)
        else:
            target_numeric = pd.to_numeric(df_norm_all[target_norm], errors="coerce")
            mask = target_numeric.notna()
            if mask.any():
                target_series = target_numeric.loc[mask].astype(int)
                df_features = df_norm_all.loc[mask].drop(columns=[target_norm])
                target_msg = f"Zielspalte erkannt: {target_norm}"
                logger.info("ui_target_detected", target=target_norm)
            else:
                raw_strings = df_norm_all[target_norm].astype("string").str.strip()
                raw_strings = raw_strings.replace({"": pd.NA})

                def _encode_target_strings(series: pd.Series) -> tuple[pd.Series, Dict[str, int]]:
                    series = series.astype("string")
                    series = series.replace({"": pd.NA})
                    if series.notna().sum() == 0:
                        return series.astype("Int64"), {}
                    known_map = {
                        "grün": 1,
                        "gruen": 1,
                        "gelb": 2,
                        "rot": 3,
                    }
                    lower = series.str.lower()
                    if lower.dropna().isin(known_map).all():
                        mapped = lower.map(known_map).astype("Int64")
                        mapping: Dict[str, int] = {}
                        for value in series.dropna().unique():
                            mapping[str(value)] = known_map[str(value).lower()]
                        return mapped, mapping
                    valid = series.dropna()
                    codes, uniques = pd.factorize(valid, sort=True)
                    mapping = {str(cat): int(idx + 1) for idx, cat in enumerate(uniques)}
                    mapped = pd.Series(pd.NA, index=series.index, dtype="Int64")
                    mapped.loc[valid.index] = codes + 1
                    return mapped, mapping

                encoded_series, mapping = _encode_target_strings(raw_strings)
                mask_encoded = encoded_series.notna()
                if mask_encoded.any():
                    target_series = encoded_series.loc[mask_encoded].astype(int)
                    df_features = df_norm_all.loc[mask_encoded].drop(columns=[target_norm])
                    target_msg = f"Zielspalte erkannt: {target_norm} (kategorisch)"
                    logger.info("ui_target_detected", target=target_norm)
                    if mapping:
                        target_mapping = mapping
                else:
                    target_msg = f"Zielspalte '{target_norm}' enthält keine nutzbaren Werte."
                    df_features = df_norm_all.drop(columns=[target_norm])
                    logger.warning("ui_target_empty", target=target_norm)
    elif target_norm:
        target_msg = f"Warnung: Zielspalte '{target_norm}' nicht gefunden."
        logger.warning("ui_target_missing", target=target_norm)

    state.update(
        {
            "excel_paths": [str(path) for path in excel_paths],
            "excel_path": str(excel_paths[0]) if excel_paths else None,
            "sheet": sheet,
            "column_mapping": column_mappings if len(column_mappings) > 1 else next(iter(column_mappings.values())),
            "df_features": df_features.copy(),
            "df_features_full": df_features.copy(),
            "target": target_series,
            "target_name": target_norm,
        }
    )
    state = _reset_pipeline_state(state)

    if target_mapping:
        state["target_mapping"] = target_mapping

    df_full = state.get("df_features_full", df_features.copy())
    available_columns = list(df_full.columns)
    default_columns = [col for col in available_columns if col not in AUTO_EXCLUDE_FEATURES]
    if not default_columns:
        default_columns = available_columns
    state["selected_columns"] = default_columns
    state["df_features"] = df_full[default_columns].copy()

    preview = df_norm_all.head(8)
    schema = _format_schema(df_norm_all)
    schema["column_mapping"] = (
        {name: mask_sensitive_data(mapping) for name, mapping in column_mappings.items()}
        if len(column_mappings) > 1
        else mask_sensitive_data(next(iter(column_mappings.values())))
    )
    schema["sources"] = [path.name for path in excel_paths]
    if warnings_list:
        schema["warnings"] = warnings_list
    if errors:
        schema["errors"] = errors
    schema["config"] = {
        "config_path": config_path,
        "target_col": target_norm,
    }

    status_parts = [
        f"Geladen: {len(excel_paths)} Datei(en) (Sheet {sheet}) | {len(df_norm_all)} Zeilen, {df_norm_all.shape[1]} Spalten.",
        target_msg,
    ]
    if warnings_list:
        status_parts.append("Hinweise: " + "; ".join(warnings_list))
    if errors:
        status_parts.append("Fehler: " + "; ".join(errors))
    status = " ".join(status_parts)

    selected_columns = state.get("selected_columns", [])
    column_selector_update = gr.update(choices=available_columns, value=selected_columns)
    config_overview = _build_config_overview(state)
    column_status = (
        f"{len(selected_columns)} Spalten automatisch ausgewählt."
        if selected_columns
        else "Keine Spalten ausgewählt."
    )

    balance_update = gr.update(value=bool(state.get("balance_classes", False)))

    return (
        status,
        preview,
        schema,
        config_overview,
        column_selector_update,
        column_status,
        balance_update,
        state,
    )


def build_pipeline_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    if df_features is None or df_features.empty:
        logger.warning("ui_build_without_data")
        return "Bitte Daten laden und Spalten auswählen.", None, None, state

    config = _ensure_config(state)
    logger.info("ui_build_pipeline_start", rows=len(df_features))

    preparer_preview = DataFramePreparer(
        amount_col=config.data.amount_col or "",
        issue_col=config.data.issue_col or "",
        due_col=config.data.due_col or "",
        date_columns=config.data.additional_date_columns,
        null_like=list(set(NULL_LIKE_DEFAULT).union(set(config.data.null_like))),
    )
    prepared = preparer_preview.fit_transform(df_features)
    with_due = DaysUntilDueAdder(
        issue_col=config.data.issue_col or "",
        due_col=config.data.due_col or "",
    ).fit_transform(prepared)

    feature_plan = infer_feature_plan(with_due, config)
    if not feature_plan.numeric and not feature_plan.categorical and not feature_plan.text:
        logger.warning("ui_feature_plan_empty")
        return "Keine nutzbaren Features gefunden.", None, None, state

    preprocessor = build_preprocessor(feature_plan, config)
    preprocessor.fit(df_features)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_prep_"))
    prep_path = tmp_dir / "prep_pipeline.joblib"
    joblib.dump(preprocessor, prep_path)

    state.update(
        {
            "preprocessor": preprocessor,
            "feature_plan": feature_plan,
            "prep_path": str(prep_path),
        }
    )

    plan_json = {
        "numeric": feature_plan.numeric,
        "categorical": feature_plan.categorical,
        "text": feature_plan.text,
        "dropped": getattr(preprocessor.named_steps["prep_df"], "columns_to_drop_", []),
        "config_path": state.get("config_path", str(DEFAULT_CONFIG_PATH)),
    }

    logger.info(
        "ui_build_pipeline_success",
        **mask_sensitive_data({
            "prep_path": str(prep_path),
            "numeric": feature_plan.numeric,
            "categorical": feature_plan.categorical,
            "text": feature_plan.text,
        }),
    )

    return "Preprocessor trainiert und gespeichert.", plan_json, str(prep_path), state


def update_selected_columns_action(selected_columns: list[str], state: Optional[Dict[str, Any]]):
    state = state or {}
    df_full = state.get("df_features_full")
    if df_full is None:
        return "Bitte zuerst Daten laden.", _build_config_overview(state), state

    if not selected_columns:
        state["df_features"] = df_full.iloc[:, 0:0].copy()
        state["selected_columns"] = []
        state = _reset_pipeline_state(state)
        return "Keine Spalten ausgewählt – Pipeline zurückgesetzt.", _build_config_overview(state), state

    missing = [col for col in selected_columns if col not in df_full.columns]
    valid_columns = [col for col in selected_columns if col in df_full.columns]
    if not valid_columns:
        state["df_features"] = df_full.iloc[:, 0:0].copy()
        state["selected_columns"] = []
        state = _reset_pipeline_state(state)
        return "Alle ausgewählten Spalten sind unbekannt – Pipeline zurückgesetzt.", _build_config_overview(state), state

    df_subset = df_full[valid_columns].copy()
    state["df_features"] = df_subset
    state["selected_columns"] = valid_columns
    state = _reset_pipeline_state(state)

    if missing:
        message = f"{len(valid_columns)} Spalten übernommen, unbekannte Spalten ignoriert: {', '.join(missing)}"
    else:
        message = f"{len(valid_columns)} Spalten für das Training ausgewählt."
    return message, _build_config_overview(state), state


def update_balance_action(balance_flag: bool, state: Optional[Dict[str, Any]]):
    state = state or {}
    state["balance_classes"] = bool(balance_flag)
    message = "Balancierung aktiviert." if state["balance_classes"] else "Balancierung deaktiviert."
    return message, _build_config_overview(state), state


def generate_bias_rules_action(
    state: Optional[Dict[str, Any]],
    prompt: str,
    business_rules_text,
    base_file,
    gpt_enabled,
    gpt_model,
    gpt_key,
):
    state = state or {}
    prompt_clean = (prompt or "").strip()
    existing_yaml = state.get("bias_rules_yaml", "")

    if not prompt_clean:
        return "Bitte zuerst einen Bias-Prompt eingeben.", existing_yaml, state

    base_path = _extract_upload_path(base_file) if base_file is not None else None
    if base_path is None:
        return "Bitte zuerst eine Ausgangsdatei hochladen.", existing_yaml, state

    if not gpt_enabled or not gpt_key:
        return (
            "Bias-Prompts benötigen GPT (API-Key + Aktivierung). Bitte Key hinterlegen und GPT aktivieren.",
            existing_yaml,
            state,
        )

    if not base_path.exists():
        return f"Ausgangsdatei nicht gefunden: {base_path}", existing_yaml, state

    columns = _collect_excel_columns(base_path)
    config_data, business_section = _parse_business_rules_block(business_rules_text or "")
    rules_section = business_section.get("rules") if isinstance(business_section, dict) else None
    existing_rule_names = (
        [str(rule.get("name")) for rule in rules_section if isinstance(rule, dict) and rule.get("name")]
        if isinstance(rules_section, list)
        else []
    )

    df_features = state.get("df_features")

    try:
        payload = _call_bias_llm(
            prompt=prompt_clean,
            columns=columns,
            existing_rule_names=existing_rule_names,
            model=BIAS_PROMPT_MODEL,
            api_key=gpt_key or "",
        )
        failure_hint = ""
    except BiasPromptError as exc:
        failure_hint = str(exc)
        target_name = state.get("target_name") or "Ampel"
        payload = _heuristic_bias_from_prompt(
            prompt_clean,
            columns,
            df_features,
            target_name,
        )

    normalized_patch = _normalise_bias_patch(payload)
    added_rules, added_dependencies = _apply_bias_patch(config_data, business_section, normalized_patch)

    rule_count = len(added_rules)
    dep_count = len(added_dependencies)
    if not added_rules and not added_dependencies:
        return "Bias-Prompt lieferte keine zusätzlichen Regeln.", existing_yaml, state

    bias_yaml = yaml.safe_dump(
        {"business_rules": {"rules": added_rules, "dependencies": added_dependencies}},
        sort_keys=False,
        allow_unicode=True,
    ).strip()

    state.update(
        {
            "bias_rules_yaml": bias_yaml,
            "bias_prompt": prompt_clean,
        }
    )

    status = f"Bias-Regeln generiert ({rule_count} Regeln, {dep_count} Abhängigkeiten)."
    if failure_hint:
        status += " Hinweis: " + failure_hint

    return status, bias_yaml, state


def generate_synthetic_data_action(
    base_file,
    config_file,
    business_rules_file,
    profile_file,
    business_rules_text,
    profile_text,
    bias_prompt,
    bias_rules_yaml,
    variation,
    lines,
    ratio,
    seed,
    gpt_model,
    gpt_key,
    gpt_enabled,
    balance_classes,
    debug_enabled,
    state: Optional[Dict[str, Any]],
):
    state = state or {}
    balance_classes = bool(balance_classes)
    log_capture: list[str] = []

    class _SynthLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = self.format(record)
            log_capture.append(message)

    if synth_run_cli is None:
        return (
            "Data Synthesizer nicht verfügbar – bitte Modul installieren.",
            None,
            None,
            None,
            "",
            state,
        )
    base_path = _extract_upload_path(base_file) if base_file is not None else None
    if base_path is None:
        return "Bitte eine Ausgangsdatei auswählen.", None, None, None, "", state

    if not base_path.exists():
        return "Ausgangsdatei nicht gefunden.", None, None, None, "", state

    try:
        variation = float(variation)
    except (TypeError, ValueError):
        variation = 0.35
    if variation < 0:
        variation = 0.0
    if variation > 1:
        variation = 1.0

    try:
        lines_int = int(lines) if lines not in (None, "") else None
    except (TypeError, ValueError):
        lines_int = None
    try:
        ratio_val = float(ratio) if ratio not in (None, "") else None
    except (TypeError, ValueError):
        ratio_val = None

    try:
        seed_val = int(seed) if seed not in (None, "") else None
    except (TypeError, ValueError):
        seed_val = None

    config_path = None
    if config_file is not None:
        config_path = _extract_upload_path(config_file)
    elif DEFAULT_SYNTH_CONFIG.exists():
        config_path = DEFAULT_SYNTH_CONFIG

    business_rules_content = (business_rules_text or "").strip()
    profile_content = (profile_text or "").strip()
    bias_prompt_content = (bias_prompt or "").strip()
    bias_rules_input = bias_rules_yaml if bias_rules_yaml is not None else state.get("bias_rules_yaml", "")
    bias_rules_content = (bias_rules_input or "").strip()
    use_gpt = bool(gpt_enabled) and bool(gpt_key)

    business_rules_path = _extract_upload_path(business_rules_file) if business_rules_file is not None else None
    profile_path = _extract_upload_path(profile_file) if profile_file is not None else None

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_synth_"))
    output_path = tmp_dir / f"{base_path.stem}_synthetic.xlsx"
    quality_path = tmp_dir / "quality_report.json"

    log_handler = _SynthLogHandler()
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    synth_logger = logging.getLogger("data_synthethizer")
    synth_logger.addHandler(log_handler)
    if synth_logger.level > logging.INFO or synth_logger.level == 0:
        synth_logger.setLevel(logging.INFO)

    config_data, business_section = _parse_business_rules_block(business_rules_content or "")

    if bias_rules_content:
        _, bias_section = _parse_business_rules_block(bias_rules_content)
        normalized_patch = _normalise_bias_patch(bias_section)
        added_rules, added_dependencies = _apply_bias_patch(config_data, business_section, normalized_patch)
        business_rules_content = yaml.safe_dump(
            config_data,
            sort_keys=False,
            allow_unicode=True,
        )
        state["bias_rules_yaml"] = yaml.safe_dump(
            {"business_rules": {"rules": added_rules, "dependencies": added_dependencies}},
            sort_keys=False,
            allow_unicode=True,
        ).strip()
        log_capture.append(
            "INFO Bias-Regeln angewendet (Regeln: %s, Dependencies: %s)"
            % (len(added_rules), len(added_dependencies))
        )
    elif bias_prompt_content:
        warn_msg = (
            "Bias-Prompt gesetzt, aber keine Bias-Regeln generiert. Bitte zuerst den Schritt 'Bias-Regeln generieren' ausführen."
        )
        log_capture.append("WARN " + warn_msg)
        return warn_msg, None, None, None, "\n".join(log_capture), state

    if business_rules_content:
        business_rules_path = tmp_dir / "business_rules_override.yaml"
        business_rules_path.write_text(business_rules_content, encoding="utf-8")
    if profile_content:
        profile_path = tmp_dir / "profile_override.yaml"
        profile_path.write_text(profile_content, encoding="utf-8")

    args = [
        "--input",
        str(base_path),
        "--output",
        str(output_path),
        "--variation",
        f"{variation}",
    ]
    if lines_int and lines_int > 0:
        args.extend(["--lines", str(lines_int)])
    elif ratio_val and ratio_val > 0:
        args.extend(["--ratio", f"{ratio_val}"])
    if seed_val is not None:
        args.extend(["--seed", str(seed_val)])
    if config_path:
        args.extend(["--column-config", str(config_path)])
    if business_rules_path:
        args.extend(["--business-rules", str(business_rules_path)])
    if profile_path:
        args.extend(["--profile", str(profile_path)])
    args.extend(["--quality-report", str(quality_path)])

    if use_gpt:
        args.extend(["--gpt-model", gpt_model or "gpt-5-mini"])
        args.extend(["--openai-api-key", gpt_key])
    else:
        args.append("--disable-gpt")

    if debug_enabled:
        args.append("--debug")

    log_capture.append(f"INFO Ausgangsdatei: {base_path}")
    if config_path:
        log_capture.append(f"INFO Konfiguration: {config_path}")
    if business_rules_path:
        log_capture.append(f"INFO Business Rules: {business_rules_path}")
    if profile_path:
        log_capture.append(f"INFO Profil: {profile_path}")
    log_capture.append("INFO Aufruf: " + " ".join(args))

    prev_key = os.environ.get("OPENAI_API_KEY")
    if use_gpt:
        os.environ["OPENAI_API_KEY"] = gpt_key

    progress_factory = getattr(gr, "Progress", None)
    progress_manager = None
    progress_callback = None
    if callable(progress_factory):
        try:
            progress_manager = progress_factory(track_tqdm=False)
        except TypeError:
            progress_manager = progress_factory()
        if hasattr(progress_manager, "__enter__") and hasattr(progress_manager, "__exit__"):
            progress_callback = progress_manager.__enter__()
        else:
            progress_callback = progress_manager

    try:
        if progress_callback is not None:
            progress_callback(0.05, desc="Starte Generator – Log wird aufgebaut")
            progress_callback(0.2, desc="Parameter übergeben – Warte auf Synthesizer")
        log_capture.append("INFO Generatorlauf gestartet")
        synth_run_cli(args)
        if progress_callback is not None:
            progress_callback(0.9, desc="Generator abgeschlossen – verarbeite Ergebnisse")
        log_capture.append("INFO Generatorlauf abgeschlossen")
    except Exception as exc:  # pragma: no cover - runtime errors
        logger.exception("synthetic_generation_failed", exc_info=exc)
        if prev_key is not None:
            os.environ["OPENAI_API_KEY"] = prev_key
        elif use_gpt:
            os.environ.pop("OPENAI_API_KEY", None)
        log_capture.append(f"ERROR {exc}")
        return f"Fehler beim Erzeugen: {exc}", None, None, None, "\n".join(log_capture), state
    finally:
        if progress_manager is not None and hasattr(progress_manager, "__exit__"):
            progress_manager.__exit__(None, None, None)
        if prev_key is not None:
            os.environ["OPENAI_API_KEY"] = prev_key
        elif use_gpt:
            os.environ.pop("OPENAI_API_KEY", None)
        synth_logger.removeHandler(log_handler)

    if not output_path.exists():
        log_capture.append("WARN Kein Ausgabepfad erzeugt")
        return "Generatorlauf ohne Ergebnis – bitte prüfen.", None, None, None, "\n".join(log_capture), state

    df_generated: Optional[pd.DataFrame] = None
    try:
        df_generated = pd.read_excel(output_path)
        preview_df = df_generated.head(20)
    except Exception:
        preview_df = None

    balance_applied = False
    balance_warning: Optional[str] = None
    state["synthetic_balance_requested"] = bool(balance_classes)

    if balance_classes:
        config_obj = _ensure_config(state)
        target_column = getattr(config_obj.data, "target_col", None) or "Massnahme_2025"
        if df_generated is None:
            try:
                df_generated = pd.read_excel(output_path)
            except Exception as exc:
                balance_warning = f"Ausgabe konnte nicht geladen werden: {exc}"
        if df_generated is not None:
            if target_column in df_generated.columns:
                df_balanced = balance_classes_dataframe(df_generated, target_column, seed_val)
                try:
                    df_balanced.to_excel(output_path, index=False)
                    preview_df = df_balanced.head(20)
                    df_generated = df_balanced
                    balance_applied = True
                except Exception as exc:  # pragma: no cover - unexpected I/O issue
                    balance_warning = f"Balancing konnte nicht gespeichert werden: {exc}"
            else:
                balance_warning = f"Zielspalte '{target_column}' nicht im Output gefunden – Balancing übersprungen."

    status = f"Synthetische Daten erzeugt: {output_path.name}"
    if balance_applied:
        status += " | Klassen balanciert"
    if quality_path.exists():
        status += f" | Quality Report: {quality_path.name}"

    if balance_warning:
        log_capture.append("WARN " + balance_warning)

    state["last_generated_path"] = str(output_path)
    if quality_path.exists():
        state["last_quality_path"] = str(quality_path)
    state["synthetic_balanced"] = balance_applied

    synth_file = str(output_path)
    quality_file = str(quality_path) if quality_path.exists() else None

    log_text = "\n".join(log_capture)
    return status, preview_df, synth_file, quality_file, log_text, state
def preview_features_action(state: Optional[Dict[str, Any]], sample_size: int = 10):
    state = state or {}
    df_features = state.get("df_features")
    preprocessor = state.get("preprocessor")
    if df_features is None or preprocessor is None:
        logger.warning("ui_preview_without_pipeline")
        return "Bitte zuerst Pipeline bauen.", None

    sample = df_features.head(sample_size)
    matrix = preprocessor.transform(sample)
    encoder = preprocessor.named_steps["encode"]
    feature_names = list(encoder.get_feature_names_out())
    if not feature_names:
        return "Keine transformierten Merkmale vorhanden.", None

    keep = min(10, len(feature_names))
    if hasattr(matrix, "toarray"):
        arr = matrix[:, :keep].toarray()
    else:
        arr = np.asarray(matrix)[:, :keep]
    preview_df = pd.DataFrame(arr, columns=feature_names[:keep])
    logger.info("ui_preview_features", rows=len(sample), features=keep)
    return f"Vorschau fuer {len(sample)} Zeilen (erste {keep} Features).", preview_df


def train_baseline_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    target = state.get("target")
    feature_plan: FeaturePlan | None = state.get("feature_plan")

    def _baseline_empty_result(message: str) -> tuple[Any, ...]:
        empty_update = gr.update(value=None)
        empty_text = gr.update(value="")
        return (
            message,
            empty_update,
            empty_update,
            empty_update,
            empty_update,
            empty_update,
            empty_update,
            empty_update,
            empty_update,
            empty_update,
            empty_update,
            empty_text,
            empty_text,
            empty_update,
            state,
        )

    if df_features is None or feature_plan is None:
        logger.warning("ui_baseline_without_pipeline")
        return _baseline_empty_result("Bitte zuerst Pipeline bauen.")
    if target is None:
        logger.warning("ui_baseline_without_target")
        return _baseline_empty_result("Keine Zielspalte verfuegbar.")

    target_series = pd.Series(target).reset_index(drop=True)
    target_series = target_series.apply(normalize_massnahme)

    config = _ensure_config(state)
    evaluation_cfg: EvaluationSection = getattr(config, "evaluation", EvaluationSection())
    validation_fraction = float(getattr(evaluation_cfg, "validation_size", 0.25))
    top_k_eval = int(getattr(evaluation_cfg, "top_k", 3))
    review_share = float(getattr(evaluation_cfg, "review_share", 0.2))
    cost_review = float(getattr(evaluation_cfg, "cost_review", 50.0))
    cost_miss = float(getattr(evaluation_cfg, "cost_miss", 250.0))
    rf_kwargs = config.model.random_forest.model_dump(exclude_none=True)

    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_plan, config)),
            ("classifier", RandomForestClassifier(**rf_kwargs)),
        ]
    )

    test_size_count, stratify = _compute_split_params(target_series, default_test_size=validation_fraction)
    X_train, X_test, y_train, y_test = train_test_split(
        df_features,
        target_series,
        test_size=test_size_count,
        random_state=config.model.random_forest.random_state,
        stratify=stratify,
    )

    train_count = len(X_train)
    validation_count = len(X_test)
    total_count = max(train_count + validation_count, 1)

    train_indices = X_train.index
    base_training_source = state.get("df_features_full")
    if isinstance(base_training_source, pd.DataFrame) and len(base_training_source) == len(df_features):
        historical_df = base_training_source.loc[train_indices].copy()
    else:
        historical_df = df_features.loc[train_indices].copy()

    target_column_name = config.data.target_col or "Massnahme_2025"
    historical_df = historical_df.reset_index(drop=True)
    historical_df[target_column_name] = y_train.reset_index(drop=True)

    if state.get("balance_classes"):
        X_train, y_train = _balance_training_set(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            config.model.random_forest.random_state,
        )
        logger.info(
            "ui_training_balanced",
            classes=int(y_train.nunique()),
            samples=len(y_train),
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(target_series)
    classes = list(label_encoder.classes_)

    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    y_test_encoded = label_encoder.transform(y_test)
    y_pred_encoded = label_encoder.transform(y_pred)
    metrics_summary = evaluate_multiclass(y_test_encoded, y_pred_encoded, classes)

    metrics = {
        "accuracy": float(metrics_summary["accuracy"]),
        "macro_f1": float(metrics_summary["macro_f1"]),
        "weighted_f1": float(metrics_summary["weighted_f1"]),
        "classes": classes,
        "classification_report": metrics_summary["classification_report"],
    }
    metrics["recall"] = float(recall_score(y_test_encoded, y_pred_encoded, average="macro", zero_division=0))
    metrics["precision"] = float(precision_score(y_test_encoded, y_pred_encoded, average="macro", zero_division=0))
    metrics["f2_score"] = float(fbeta_score(y_test_encoded, y_pred_encoded, beta=2.0, average="macro", zero_division=0))
    metrics["samples"] = {
        "train": int(train_count),
        "validation": int(validation_count),
        "validation_fraction_actual": float(validation_count / total_count),
        "validation_fraction_config": float(validation_fraction),
    }
    metrics["accuracy_display"] = f"In {metrics['accuracy']:.0%} der Fälle richtig"

    confusion = np.asarray(metrics_summary["confusion_matrix"], dtype=float)
    label_lookup = {idx: cls for idx, cls in enumerate(classes)}
    confusion_df = pd.DataFrame(
        confusion,
        index=[label_lookup.get(idx, str(idx)) for idx in range(confusion.shape[0])],
        columns=[label_lookup.get(idx, str(idx)) for idx in range(confusion.shape[1])],
    )

    encoder = baseline.named_steps["preprocessor"].named_steps.get("encode")
    if encoder is not None and hasattr(encoder, "get_feature_names_out"):
        feature_names = encoder.get_feature_names_out()
    else:
        feature_names = np.array([f"f_{idx}" for idx in range(len(baseline.named_steps["classifier"].feature_importances_))])

    importance = baseline.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df.sort_values("importance", ascending=False, inplace=True)
    top20 = importance_df.head(20).reset_index(drop=True)

    proba = baseline.predict_proba(X_test)
    if proba.size:
        effective_top_k = min(max(top_k_eval, 1), proba.shape[1])
        sorted_indices = np.argsort(proba, axis=1)[:, ::-1][:, :effective_top_k]
        hits = (sorted_indices == y_test_encoded.reshape(-1, 1)).any(axis=1)
        top_k_hit_rate = float(hits.mean()) if len(hits) else 0.0
    else:  # pragma: no cover - defensive
        effective_top_k = min(top_k_eval, len(classes)) if classes else top_k_eval
        top_k_hit_rate = 0.0
    metrics["top_k_hit_rate"] = {
        "k": int(effective_top_k),
        "hit_rate": top_k_hit_rate,
        "display": f"Top-{effective_top_k}: {top_k_hit_rate:.1%}",
    }

    ml_confidence = proba.max(axis=1)
    ml_prediction = canonicalize_massnahmen(pd.Series(y_pred, name="ml_prediction")).reset_index(drop=True)
    ml_confidence_series = pd.Series(ml_confidence, name="ml_confidence").reset_index(drop=True)

    business_rules: List[BusinessRule] = []
    try:
        business_rules = load_business_rules_from_file(MASSNAHMEN_RULES_PATH)
    except FileNotFoundError:
        logger.warning("ui_rules_missing", path=str(MASSNAHMEN_RULES_PATH))
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("ui_rules_load_failed", message=str(exc))

    rule_engine = RuleEngine(business_rules, historical_data=historical_df)
    hybrid_predictor = HybridMassnahmenPredictor(baseline, rule_engine)
    hybrid_predictor.preprocessor_ = baseline.named_steps["preprocessor"]
    hybrid_predictor.feature_names_ = list(feature_names)
    hybrid_predictor.label_encoder_ = label_encoder
    try:
        shap_background = state.get("shap_background")
    except KeyError:
        shap_background = None
    if shap_background is not None:
        hybrid_predictor.background_ = shap_background

    hybrid_results = hybrid_predictor.predict(X_test.reset_index(drop=True))
    source_series = hybrid_results["source"].astype("string").fillna("ml")
    rule_source_series = source_series.where(source_series != "ml", None)
    rule_prediction_series = hybrid_results["prediction"].where(rule_source_series.notna(), None)
    rule_confidence_series = hybrid_results["confidence"].where(rule_source_series.notna(), None)

    final_prediction = canonicalize_massnahmen(ml_prediction.copy())
    final_confidence = ml_confidence_series.copy()
    non_ml_mask = rule_source_series.notna()
    if non_ml_mask.any():
        final_prediction.loc[non_ml_mask] = hybrid_results.loc[non_ml_mask, "prediction"].to_list()
        updated_conf = hybrid_results.loc[non_ml_mask, "confidence"].astype(float)
        final_confidence.loc[non_ml_mask] = updated_conf.values

    result_index = pd.RangeIndex(len(final_prediction))
    predictions_df = pd.DataFrame(index=result_index)
    predictions_df["row_index"] = result_index
    predictions_df["ml_prediction"] = ml_prediction.reset_index(drop=True)
    predictions_df["ml_confidence"] = ml_confidence_series.reset_index(drop=True)
    predictions_df["rule_source"] = rule_source_series.reset_index(drop=True)
    predictions_df["rule_prediction"] = rule_prediction_series.reset_index(drop=True)
    predictions_df["rule_confidence"] = pd.to_numeric(
        rule_confidence_series.reset_index(drop=True), errors="coerce"
    )
    predictions_df["final_prediction"] = final_prediction.reset_index(drop=True)
    predictions_df["final_confidence"] = final_confidence.reset_index(drop=True)
    predictions_df["actual"] = y_test.reset_index(drop=True)
    predictions_df["is_correct"] = predictions_df["final_prediction"].astype(str) == predictions_df["actual"].astype(str)
    final_conf_numeric = pd.to_numeric(predictions_df["final_confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    error_indicator = (~predictions_df["is_correct"]).astype(int).to_numpy(dtype=float)
    review_scores = (1.0 - final_conf_numeric).to_numpy(dtype=float)
    predictions_df["review_score"] = review_scores

    feature_frame = X_test.reset_index(drop=True).copy()
    feature_columns = list(feature_frame.columns)
    predictions_df = pd.concat([predictions_df, feature_frame], axis=1)

    predictions_df["explanation"] = "🔍 Erklärung anzeigen"

    if len(predictions_df) > 0:
        benefit_top_share = compute_cost_simulation(error_indicator, review_scores, review_share, cost_review, cost_miss)
        benefit_full_review = compute_cost_simulation(error_indicator, review_scores, 1.0, cost_review, cost_miss)
    else:
        benefit_top_share = 0.0
        benefit_full_review = 0.0

    metrics["cost_simulation"] = {
        "review_share": review_share,
        "cost_review": cost_review,
        "cost_miss": cost_miss,
        "net_benefit_top_share": float(benefit_top_share),
        "net_benefit_full_review": float(benefit_full_review),
        "delta_vs_full_review": float(benefit_top_share - benefit_full_review),
    }

    predictions_df["confidence_percent"] = predictions_df["final_confidence"].astype(float, errors="ignore") * 100.0
    predictions_df.sort_values("final_confidence", ascending=False, inplace=True)
    display_columns = [
        "row_index",
        "final_prediction",
        "final_confidence",
        "ml_prediction",
        "ml_confidence",
        "rule_source",
        "rule_prediction",
        "actual",
        "explanation",
    ]
    predictions_display = predictions_df[display_columns].head(50).reset_index(drop=True)

    rule_source_full = predictions_df["rule_source"].astype("object")
    rule_coverage = float(rule_source_full.notna().mean()) if not rule_source_full.empty else 0.0
    ml_fallback_rate = float(rule_source_full.isna().mean()) if not rule_source_full.empty else 0.0
    metrics["rule_coverage"] = rule_coverage
    metrics["ml_fallback_rate"] = ml_fallback_rate
    metrics["rule_coverage_display"] = f"{rule_coverage:.1%}"
    metrics["ml_fallback_display"] = f"{ml_fallback_rate:.1%}"

    distribution_df = (
        predictions_df["final_prediction"].astype("string").value_counts().rename_axis("Massnahme").reset_index(name="Anzahl")
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_model_"))
    model_path = tmp_dir / "baseline_model.joblib"
    metrics_path = tmp_dir / "metrics.json"
    importance_path = tmp_dir / "feature_importance.csv"
    plot_path = tmp_dir / "feature_importance.png"
    predictions_path = tmp_dir / "predictions.csv"
    heatmap_path = tmp_dir / "confusion_heatmap.png"

    joblib.dump(baseline, model_path)
    importance_df.to_csv(importance_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    confusion_plot_image = None
    if confusion.size:
        fig_width = max(6.0, len(classes) * 0.75)
        fig_height = max(5.0, len(classes) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Vorhergesagt")
        ax.set_ylabel("Wahr")
        ax.set_title("Konfusionsmatrix")
        vmax = confusion.max() if confusion.size else 1
        for (i, j), val in np.ndenumerate(confusion):
            color = "white" if val > vmax / 2 else "black"
            ax.text(j, i, int(val), ha="center", va="center", color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(heatmap_path, dpi=160)
        plt.close(fig)
        confusion_plot_image = str(heatmap_path)

    plot_image = None
    if not top20.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(top20["feature"][::-1], top20["importance"][::-1], color="#2b8a3e")
        plt.xlabel("Importance")
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plot_image = str(plot_path)

    metrics_serializable = {
        **metrics,
        "confusion_matrix": confusion.tolist(),
        "massnahmen_distribution": distribution_df.to_dict(orient="records"),
        "confusion_heatmap_path": confusion_plot_image,
    }
    metrics_path.write_text(json.dumps(metrics_serializable, indent=2), encoding="utf-8")

    outputs_dir = Path("outputs")
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, outputs_dir / "baseline_model.joblib")
        joblib.dump(label_encoder, outputs_dir / "label_encoder.joblib")
        predictions_df.to_csv(outputs_dir / "predictions.csv", index=False)
        importance_df.to_csv(outputs_dir / "feature_importance.csv", index=False)
        (outputs_dir / "metrics.json").write_text(
            json.dumps(metrics_serializable, indent=2),
            encoding="utf-8",
        )
        if confusion_plot_image:
            try:
                shutil.copy(confusion_plot_image, outputs_dir / "confusion_heatmap.png")
            except (OSError, shutil.SameFileError):  # pragma: no cover
                pass
    except Exception as exc:  # pragma: no cover - best-effort export
        logger.warning("ui_outputs_export_failed", message=str(exc))

    state.update(
        {
            "baseline_model": baseline,
            "baseline_path": str(model_path),
            "metrics_path": str(metrics_path),
            "importance_path": str(importance_path),
            "importance_plot": plot_image,
            "predictions_path": str(predictions_path),
            "shap_feature_names": list(feature_names),
            "feature_importances": importance_df.to_dict(orient="records"),
            "predictions_full": predictions_df,
            "label_encoder": label_encoder,
            "label_classes": classes,
            "rule_engine": rule_engine,
            "business_rules": business_rules,
            "historical_training_data": historical_df,
            "hybrid_results": hybrid_results,
            "hybrid_predictor": hybrid_predictor,
            "rule_metrics": {"rule_coverage": rule_coverage, "ml_fallback_rate": ml_fallback_rate},
            "massnahmen_distribution": distribution_df,
            "confusion_heatmap": confusion_plot_image,
            "feature_columns": feature_columns,
            "validation_features": X_test.reset_index(drop=True),
            "validation_labels": y_test.reset_index(drop=True),
        }
    )

    try:
        state["feature_importance_df"] = importance_df
        state["notes_text_vectorizer"] = baseline.named_steps["preprocessor"].named_steps["encode"].named_transformers_["text"].named_steps["tfidf"]
    except Exception as exc:
        logger.warning("ui_importance_store_failed", message=str(exc))

    try:
        background_size = min(200, len(df_features))
        background = baseline.named_steps["preprocessor"].transform(df_features.head(background_size))
        background_dense = background.toarray() if hasattr(background, "toarray") else background
        state["shap_background"] = background_dense
        hybrid_predictor.background_ = background_dense
    except Exception as exc:
        logger.warning("ui_shap_background_failed", message=str(exc))

    metric_summary = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "importance_path": str(importance_path),
        "predictions_path": str(predictions_path),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }

    logger.info("ui_baseline_trained", **mask_sensitive_data(metric_summary))

    rule_coverage_display = f"{rule_coverage:.1%}"
    ml_fallback_display = f"{ml_fallback_rate:.1%}"
    distribution_display = distribution_df.copy()

    status_message = (
        f"Baseline trainiert – Genauigkeit {metrics['accuracy']:.1%} auf {validation_count} Validierungs-Belegen."
    )

    feedback_summary_text, feedback_trend_df, feedback_per_class_df, feedback_critical_text, state = feedback_dashboard_action(state)
    retrain_status_value = ""
    retrain_comparison_value = pd.DataFrame()

    return (
        status_message,
        metrics,
        confusion_df,
        confusion_plot_image,
        str(model_path),
        str(metrics_path),
        top20,
        plot_image,
        str(importance_path),
        predictions_display,
        str(predictions_path),
        rule_coverage_display,
        ml_fallback_display,
        distribution_display,
        feedback_summary_text,
        feedback_trend_df,
        feedback_per_class_df,
        feedback_critical_text,
        retrain_status_value,
        retrain_comparison_value,
        state,
    )


def explain_massnahme_action(state: Optional[Dict[str, Any]], row_index):
    state = state or {}
    hybrid_predictor: Optional[HybridMassnahmenPredictor] = state.get("hybrid_predictor")
    predictions_df = state.get("predictions_full")
    feature_columns: Optional[List[str]] = state.get("feature_columns")

    if hybrid_predictor is None or predictions_df is None or predictions_df.empty:
        return "Bitte zuerst Baseline trainieren.", None, None, state

    try:
        idx = int(row_index)
    except (TypeError, ValueError):
        return "Ungültiger Index.", None, None, state

    row_match = predictions_df.loc[predictions_df["row_index"] == idx]
    if row_match.empty:
        return f"Kein Eintrag für Zeile {idx} gefunden.", None, None, state

    prediction_row = row_match.iloc[0]

    if not feature_columns:
        excluded = {
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
        feature_columns = [col for col in predictions_df.columns if col not in excluded]

    components = create_explanation_components(
        predictor=hybrid_predictor,
        row_index=idx,
        prediction_row=prediction_row,
        feature_columns=list(feature_columns),
        predictions_df=predictions_df,
    )

    markdown = components.markdown
    final_prediction = components.payload.get("prediction")
    massnahme_slug = _slugify_label(final_prediction or "massnahme")
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_erklaerung_"))
    md_path = tmp_dir / f"erklaerung_beleg_{idx}_{massnahme_slug}.md"
    md_path.write_text(markdown, encoding="utf-8")

    state["last_explanation_payload"] = components.payload
    state["last_explanation_markdown"] = markdown
    state["last_explanation_row_index"] = idx

    status_message = f"Erklärung für Beleg #{idx} generiert."
    return status_message, markdown, str(md_path), state


def export_explanation_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    payload = state.get("last_explanation_payload")
    markdown = state.get("last_explanation_markdown")

    if not payload or not markdown:
        return "Keine Erklärung vorhanden. Bitte zuerst eine Erklärung anzeigen.", None, state

    row_index = payload.get("row_index", "unbekannt")
    prediction = payload.get("prediction", "massnahme")
    massnahme_slug = _slugify_label(prediction)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_export_"))
    md_path = tmp_dir / f"erklaerung_beleg_{row_index}_{massnahme_slug}.md"
    md_path.write_text(markdown, encoding="utf-8")

    status_message = f"Markdown exportiert ({md_path.name})."
    return status_message, str(md_path), state


def generate_pattern_report_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    target = state.get("target")
    feature_importance = state.get("feature_importance_df")
    vectorizer = state.get("notes_text_vectorizer")

    def _report_empty_result(message: str) -> tuple[Any, ...]:
        return message, gr.update(value=""), None, state

    if df_features is None or target is None or feature_importance is None:
        return _report_empty_result("Bitte zuerst Baseline trainieren.")

    target_series = pd.Series(target).reset_index(drop=True)
    df_local = df_features.reset_index(drop=True).copy()
    min_len = min(len(df_local), len(target_series))
    if min_len == 0:
        return _report_empty_result("Keine Daten für den Pattern-Report vorhanden.")

    if len(df_local) != len(target_series):
        df_local = df_local.iloc[:min_len].copy()
        target_series = target_series.iloc[:min_len].copy()

    target_label = state.get("target_name") or "Ziel"
    internal_target_col = "__target_value__"
    df_local[internal_target_col] = target_series
    class_counts = target_series.value_counts(dropna=False)
    n_classes = class_counts.shape[0]

    def _format_cell(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "(NaN)"
        return str(value)

    report_lines = [f"# Pattern Report für {target_label}", ""]
    top_features = feature_importance.head(10)
    report_lines.append("## Wichtigste Features")
    for _, row in top_features.iterrows():
        report_lines.append(f"- **{row['feature']}**: Importance {row['importance']:.4f}")
    report_lines.append("")

    report_lines.append("## Klassenverteilung")
    report_lines.append("| Klasse | Anteil | Anzahl |")
    report_lines.append("| --- | --- | --- |")
    total_count = float(len(target_series))
    for klass, count in class_counts.items():
        report_lines.append(
            f"| {_format_cell(klass)} | {count / total_count:.2%} | {int(count)} |"
        )
    report_lines.append("")

    for column, label in (("Land", "Land"), ("BUK", "BUK")):
        report_lines.append(f"## Verteilung nach {label}")
        if column in df_local.columns:
            group = (
                df_local.groupby([column, internal_target_col])
                .size()
                .reset_index(name="count")
            )
            if not group.empty:
                group["share"] = group["count"] / group.groupby(column)["count"].transform("sum")
                top_rows = group.sort_values(["share", "count"], ascending=[False, False]).head(10)
                report_lines.append(f"| {label} | {target_label} | Anteil | Anzahl |")
                report_lines.append("| --- | --- | --- | --- |")
                for _, row in top_rows.iterrows():
                    report_lines.append(
                        f"| {_format_cell(row[column])} | {_format_cell(row[internal_target_col])} | "
                        f"{row['share']:.2%} | {int(row['count'])} |"
                    )
            else:
                report_lines.append("Keine Daten verfügbar.")
        else:
            report_lines.append(f"Keine Spalte '{column}' vorhanden.")
        report_lines.append("")

    report_lines.append("## Betrag je Klasse")
    amount_column = "Betrag_parsed"
    if amount_column in df_local.columns:
        amount_series = pd.to_numeric(df_local[amount_column], errors="coerce")
    else:
        amount_series = pd.Series(dtype="float64")
    if isinstance(amount_series, pd.Series) and amount_series.notna().any():
        df_local["__amount_numeric__"] = amount_series
        stats = (
            df_local.groupby(internal_target_col)["__amount_numeric__"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        report_lines.append("| Klasse | Mittelwert | Median | Anzahl |")
        report_lines.append("| --- | --- | --- | --- |")
        for _, row in stats.iterrows():
            report_lines.append(
                f"| {_format_cell(row[internal_target_col])} | {row['mean']:.2f} | {row['median']:.2f} | {int(row['count'])} |"
            )
    else:
        report_lines.append("Betragsspalte nicht vorhanden oder ohne numerische Werte.")
    report_lines.append("")

    report_lines.append("## Text-Keywords")
    if vectorizer is not None:
        feature_names = vectorizer.get_feature_names_out()

        text_series = None
        if "notes_text" in df_features.columns:
            text_series = df_features["notes_text"].astype(str)
        else:
            feature_plan = state.get("feature_plan")
            candidate_cols: list[str] = []
            if feature_plan is not None:
                candidate_cols.extend(
                    [col for col in getattr(feature_plan, "text", []) if col in df_features.columns]
                )
            if not candidate_cols:
                config = _ensure_config(state)
                candidate_cols.extend([col for col in config.data.text_columns if col in df_features.columns])

            if candidate_cols:
                text_source = df_features[candidate_cols].fillna("")
                text_series = text_source.agg(" ".join, axis=1).astype(str)

        if text_series is not None and not text_series.empty:
            top_classes = class_counts.head(min(3, n_classes)).index
            for klass in top_classes:
                class_mask = target_series == klass
                class_texts = text_series.loc[class_mask].fillna("")
                if class_texts.empty:
                    continue
                transformed = vectorizer.transform(class_texts)
                summed = np.asarray(transformed.sum(axis=0)).ravel()
                if not np.any(summed):
                    continue
                top_idx = np.argsort(summed)[::-1][:10]
                report_lines.append(f"### {_format_cell(klass)}")
                for idx in top_idx:
                    weight = summed[idx]
                    if weight <= 0:
                        continue
                    report_lines.append(f"- {feature_names[idx]} (Gewicht {weight:.2f})")
            if len(report_lines) > 0 and report_lines[-1] == "## Text-Keywords":
                report_lines.append("Keine Textmuster gefunden.")
        else:
            report_lines.append("Keine Texte vorhanden.")
    else:
        report_lines.append("Text-Vektorisierer nicht verfügbar.")
    report_lines.append("")

    markdown_text = "\n".join(report_lines)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_report_"))
    md_path = tmp_dir / "pattern_report.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    return "Report erstellt.", markdown_text, str(md_path), state


def batch_predict_action(upload, state: Optional[Dict[str, Any]]):
    state = state or {}
    model = state.get("baseline_model")
    if model is None:
        return "Bitte zuerst Baseline trainieren.", None

    upload_path = _extract_upload_path(upload) if upload is not None else None
    if upload_path is None:
        return "Bitte eine Excel-Datei hochladen.", None

    progress = gr.Progress(track_tqdm=False)
    progress(0.05, desc="Lade Datei")
    df_input_raw = pd.read_excel(upload_path)
    df_input, _ = normalize_columns(df_input_raw)
    rule_context = df_input.copy()


    config = _ensure_config(state)
    target_col = config.data.target_col
    if target_col and target_col in df_input.columns:
        df_input = df_input.drop(columns=[target_col])

    selected_columns = state.get("selected_columns") or list(df_input.columns)
    missing = [col for col in selected_columns if col not in df_input.columns]
    for col in missing:
        df_input[col] = np.nan
    df_input = df_input[selected_columns]

    required_columns = {
        "Ampel",
        config.data.amount_col or "Betrag",
        "Betrag_parsed",
        "BUK",
        "Debitor",
        "Land",
        "negativ",
    }

    df_augmented = df_input.copy()
    rule_lookup = rule_context.reindex(df_augmented.index)

    amount_source_column = config.data.amount_col or "Betrag"
    currency_symbols = ["€", "EUR", " "]

    for column in required_columns:
        if column in df_augmented.columns:
            continue
        if column == "Betrag_parsed":
            if column in rule_lookup.columns:
                series = rule_lookup[column]
            elif amount_source_column in rule_lookup.columns:
                series = parse_money(rule_lookup[amount_source_column], currency_symbols)
            else:
                series = pd.Series(np.nan, index=df_augmented.index)
            df_augmented[column] = series
            continue

        if column in rule_lookup.columns:
            df_augmented[column] = rule_lookup[column]
        else:
            fill_value = False if column == "negativ" else np.nan
            df_augmented[column] = pd.Series(fill_value, index=df_augmented.index)

    df_input = df_augmented

    progress(0.3, desc="Berechne Maßnahmen")

    hybrid_predictor = state.get("hybrid_predictor")
    rule_engine = state.get("rule_engine")
    if hybrid_predictor is None and rule_engine is not None:
        hybrid_predictor = HybridMassnahmenPredictor(model, rule_engine)
        hybrid_predictor.preprocessor_ = model.named_steps.get("preprocessor")
        hybrid_predictor.feature_names_ = state.get("shap_feature_names")
        shap_background = state.get("shap_background")
        if shap_background is not None:
            hybrid_predictor.background_ = shap_background
        state["hybrid_predictor"] = hybrid_predictor

    if hybrid_predictor is not None:
        results = hybrid_predictor.predict(df_input.reset_index(drop=True))
        if not isinstance(results, pd.DataFrame) or "prediction" not in results.columns:
            fallback_predictions: List[str] = []
            fallback_confidences: List[Optional[float]] = []
            fallback_sources: List[str] = []
            for _, row in df_input.reset_index(drop=True).iterrows():
                explanation = hybrid_predictor.explain(row)
                fallback_predictions.append(explanation.get("prediction"))
                fallback_confidences.append(explanation.get("confidence"))
                source_value = explanation.get("source", "ml") or "ml"
                fallback_sources.append(str(source_value))
            results = pd.DataFrame(
                {
                    "prediction": canonicalize_massnahmen(pd.Series(fallback_predictions)).reset_index(drop=True),
                    "confidence": pd.Series(fallback_confidences, dtype="float"),
                    "source": pd.Series(fallback_sources, dtype="string"),
                }
            )
        predicted_series = canonicalize_massnahmen(results["prediction"]).reset_index(drop=True)
        fill_value = predicted_series.iloc[0] if not predicted_series.empty else "Unbekannt"
        predicted_series = predicted_series.reindex(range(len(df_input)), fill_value=fill_value)
        df_input["Massnahme_2025"] = predicted_series.to_numpy()
        df_input["final_confidence"] = results["confidence"].astype(float, errors="ignore")
        df_input["prediction_source"] = results["source"]
    else:
        proba = model.predict_proba(df_input)
        raw_predictions = pd.Series(model.predict(df_input), index=df_input.index)
        ml_prediction = canonicalize_massnahmen(raw_predictions).reset_index(drop=True)
        ml_confidence = proba.max(axis=1)
        ml_pred_series = ml_prediction.reset_index(drop=True)
        fill_value = ml_pred_series.iloc[0] if not ml_pred_series.empty else "Unbekannt"
        ml_pred_series = ml_pred_series.reindex(range(len(df_input)), fill_value=fill_value)
        df_input["Massnahme_2025"] = ml_pred_series.to_numpy()
        df_input["final_confidence"] = ml_confidence
        df_input["prediction_source"] = "ml"

    df_input["final_confidence"] = pd.to_numeric(df_input["final_confidence"], errors="coerce").fillna(0.0)
    df_input["fraud_score"] = df_input["final_confidence"] * 100.0
    final_series = canonicalize_massnahmen(df_input["Massnahme_2025"].astype("string")).reset_index(drop=True)
    fill_value = final_series.iloc[0] if not final_series.empty else "Unbekannt"
    final_series = final_series.reindex(range(len(df_input)), fill_value=fill_value)
    df_input["final_prediction"] = final_series.to_numpy()

    df_out = df_input

    progress(0.8, desc="Schreibe Ergebnis")
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_batch_"))
    out_path = tmp_dir / "batch_predictions.xlsx"
    df_out.to_excel(out_path, index=False)
    progress(1.0, desc="Fertig")

    logger.info(
        "ui_batch_prediction_completed",
        rows_predicted=len(df_input),
        output=str(out_path),
    )

    status = f"Batch abgeschlossen: {len(df_input)} Predictions"
    return status, str(out_path)


def record_feedback_entry(
    state: Dict[str, Any],
    row_index: int,
    actual_label: Optional[str],
    user: str,
    comment: str,
    feedback_flag: str,
) -> str:
    predictions = state.get("predictions_full")
    if predictions is None or predictions.empty:
        return "Keine Predictions verfügbar."

    row_match = predictions.loc[predictions["row_index"] == row_index]
    if row_match.empty:
        return f"Kein Eintrag für Zeile {row_index} gefunden."

    row = row_match.iloc[0]

    predicted_label = str(row.get("final_prediction", "Unbekannt"))
    actual_label = str(actual_label or predicted_label)
    is_correct = predicted_label == actual_label
    final_confidence = row.get("final_confidence")
    score = float(final_confidence) * 100.0 if final_confidence is not None else 0.0

    label_encoder = state.get("label_encoder")
    prediction_code = None
    if label_encoder is not None and predicted_label:
        try:
            prediction_code = int(label_encoder.transform([predicted_label])[0])
        except Exception:  # pragma: no cover - defensive
            prediction_code = None

    beleg_id = None
    for candidate in ("Rechnungsnummer", "Belegnummer", "belegnummer", "invoice_id"):
        if candidate in row.index and not pd.isna(row.get(candidate)):
            beleg_id = str(row.get(candidate))
            break

    entry = {
        "beleg_index": int(row_index),
        "beleg_id": beleg_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user": user or "Unbekannt",
        "score": score,
        "prediction": prediction_code,
        "feedback": feedback_flag,
        "comment": comment or "",
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "is_correct": is_correct,
    }

    save_feedback_to_db(entry)
    return "✅ Feedback gespeichert. Danke!"


def feedback_report_action(state: Optional[Dict[str, Any]]):
    ensure_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM feedback", conn, parse_dates=["timestamp"]) if conn else pd.DataFrame()

    if df.empty:
        empty_df = pd.DataFrame({"Feedback Report": []})
        return "Noch kein Feedback vorhanden.", empty_df, None, state

    now = datetime.utcnow()
    last_week = now - timedelta(days=7)
    df_week = df[df["timestamp"] >= last_week.isoformat()]
    if df_week.empty:
        summary = "Keine Feedback-Daten in den letzten 7 Tagen."
    else:
        tp = (df_week["feedback"] == "TP").sum()
        fp = (df_week["feedback"] == "FP").sum()
        total = tp + fp
        precision = tp / total if total else float("nan")
        false_positive_rate = fp / total if total else float("nan")
        summary = (
            f"Letzte Woche: Präzision {precision:.2%}, False Positive Rate {false_positive_rate:.2%}"
            if total
            else "Keine validen Feedback-Daten für letzte Woche."
        )

    overall_tp = (df["feedback"] == "TP").sum()
    overall_fp = (df["feedback"] == "FP").sum()
    overall_total = overall_tp + overall_fp
    overall_precision = overall_tp / overall_total if overall_total else float("nan")

    report_lines = ["# Feedback-Report", ""]
    report_lines.append(summary)
    report_lines.append("")
    report_lines.append(f"Gesamt-Precision: {overall_precision:.2%}" if overall_total else "Gesamt-Precision: n/a")
    report_lines.append(f"Anzahl Feedbacks: {overall_total}")

    md_text = "\n".join(report_lines)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_feedback_"))
    md_path = tmp_dir / "feedback_report.md"
    md_path.write_text(md_text, encoding="utf-8")

    preview_df = pd.DataFrame({"Feedback Report": report_lines})

    return summary, preview_df, str(md_path), state


def feedback_dashboard_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df = load_feedback_from_db()
    if df.empty:
        summary = "Noch kein Feedback erfasst."
        trend_df = pd.DataFrame({"week": [], "accuracy": [], "feedbacks": []})
        per_class_df = pd.DataFrame({"Maßnahme": [], "Anzahl": [], "Korrekt": []})
        critical = "Keine kritischen Maßnahmen erkannt."
    else:
        df["is_correct"] = df["is_correct"].fillna(0).astype(int)
        total = int(len(df))
        correct = int(df["is_correct"].sum())
        incorrect = total - correct
        correct_share = correct / total if total else 0
        incorrect_share = incorrect / total if total else 0
        summary = (
            "🎯 Gesamt-Zustimmung: {total} Feedbacks\n"
            "- {correct_share:.0%} Korrekt ({correct})\n"
            "- {incorrect_share:.0%} Korrigiert ({incorrect})"
        ).format(
            total=total,
            correct_share=correct_share,
            correct=correct,
            incorrect_share=incorrect_share,
            incorrect=incorrect,
        )

        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            weeks = (
                df["timestamp"].dt.to_period("W").dt.start_time.dt.date
            )
            df_weeks = pd.DataFrame({"week": weeks, "is_correct": df["is_correct"]})
            trend_df = (
                df_weeks.groupby("week")
                .agg(accuracy=("is_correct", "mean"), feedbacks=("is_correct", "count"))
                .reset_index()
            )
            trend_df["accuracy"] = trend_df["accuracy"].astype(float)
        else:
            trend_df = pd.DataFrame({"week": [], "accuracy": [], "feedbacks": []})

        group = (
            df.groupby("predicted_label")
            .agg(
                accuracy=("is_correct", "mean"),
                feedbacks=("is_correct", "count"),
                correct=("is_correct", "sum"),
            )
            .reset_index()
        )
        group["accuracy"] = group["accuracy"].astype(float)
        per_class_df = pd.DataFrame(
            {
                "Maßnahme": group["predicted_label"].fillna("Unbekannt"),
                "Anzahl": group["feedbacks"].astype(int),
                "Korrekt": (group["accuracy"] * 100).round(1),
            }
        )

        critical_rows = group[group["accuracy"] < 0.8]
        if critical_rows.empty:
            critical = "Keine kritischen Maßnahmen erkannt."
        else:
            critical_lines = ["🚨 Kritische Maßnahmen (< 80% Korrekt):"]
            for _, item in critical_rows.iterrows():
                correct_count = int(item["correct"])
                total_count = int(item["feedbacks"])
                critical_lines.append(
                    f"• {item['predicted_label']}: {item['accuracy']:.0%} ({correct_count}/{total_count})"
                )
            critical = "\n".join(critical_lines)

    state["feedback_cache"] = df
    return summary, trend_df, per_class_df, critical, state


def open_feedback_modal_action(state: Optional[Dict[str, Any]], row_index, mode: str):
    state = state or {}
    predictions = state.get("predictions_full")
    try:
        idx = int(row_index)
    except (TypeError, ValueError):
        return (
            gr.update(visible=False),
            "Bitte eine gültige Rechnungsnummer angeben.",
            gr.update(),
            gr.update(value=""),
            gr.update(value=state.get("last_feedback_user", "Workshop Gast")),
            "",
            state,
        )

    if predictions is None or predictions.empty:
        return (
            gr.update(visible=False),
            "Bitte zuerst das Modell trainieren und Predictions erzeugen.",
            gr.update(),
            gr.update(value=""),
            gr.update(value=state.get("last_feedback_user", "Workshop Gast")),
            "",
            state,
        )

    match = predictions.loc[predictions["row_index"] == idx]
    if match.empty:
        return (
            gr.update(visible=False),
            f"Kein Eintrag für Rechnung #{idx} vorhanden.",
            gr.update(),
            gr.update(value=""),
            gr.update(value=state.get("last_feedback_user", "Workshop Gast")),
            "",
            state,
        )

    row = match.iloc[0]
    predicted_label = str(row.get("final_prediction", "Unbekannt"))
    title = textwrap.dedent(
        f"""
        ### Feedback für Rechnung #{idx}

        Empfohlen: **{predicted_label}**
        """
    ).strip()

    default_actual = predicted_label
    state["feedback_dialog"] = {
        "row_index": idx,
        "mode": mode,
        "predicted": predicted_label,
    }

    status = "Bitte tatsächliche Maßnahme auswählen." if mode == "incorrect" else "Bestätige die vorgeschlagene Maßnahme."
    return (
        gr.update(visible=True),
        title,
        gr.update(value=default_actual),
        gr.update(value=""),
        gr.update(value=state.get("last_feedback_user", "Workshop Gast")),
        status,
        state,
    )


def cancel_feedback_modal_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    state.pop("feedback_dialog", None)
    return gr.update(visible=False), "Feedback verworfen.", state


def submit_feedback_action(user: str, actual_label: str, comment: str, state: Optional[Dict[str, Any]], progress=gr.Progress()):
    state = state or {}
    dialog = state.get("feedback_dialog")
    if not dialog:
        summary, trend_df, per_class_df, critical, state = feedback_dashboard_action(state)
        return (
            gr.update(visible=False),
            "Kein Feedback-Kontext aktiv.",
            summary,
            trend_df,
            per_class_df,
            critical,
            state,
        )

    progress(0.2, desc="Speichere Feedback")
    mode = dialog.get("mode", "correct")
    flag = "correct" if mode == "correct" else "incorrect"
    row_index = dialog.get("row_index")

    message = record_feedback_entry(
        state,
        row_index=row_index,
        actual_label=actual_label,
        user=user or "Workshop Gast",
        comment=comment,
        feedback_flag=flag,
    )
    state["last_feedback_user"] = user or "Workshop Gast"
    state.pop("feedback_dialog", None)

    progress(0.6, desc="Aktualisiere Dashboard")
    summary, trend_df, per_class_df, critical, state = feedback_dashboard_action(state)
    progress(1.0, desc="Fertig")

    return (
        gr.update(visible=False),
        message,
        summary,
        trend_df,
        per_class_df,
        critical,
        state,
    )


def prepare_feedback_features(state: Dict[str, Any], feedback_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    predictions = state.get("predictions_full")
    feature_columns = state.get("feature_columns")
    if predictions is None or feature_columns is None:
        return pd.DataFrame(), pd.Series(dtype="object")

    merged = feedback_df.merge(
        predictions,
        left_on="beleg_index",
        right_on="row_index",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), pd.Series(dtype="object")

    features = merged[feature_columns].copy()
    labels = merged["actual_label"].fillna(merged["predicted_label"]).astype(str)
    return features.reset_index(drop=True), labels.reset_index(drop=True)


def _evaluate_hybrid_predictor(
    predictor: HybridMassnahmenPredictor,
    X: pd.DataFrame,
    y_true: pd.Series,
    top_k: int = 3,
) -> Dict[str, float]:
    if predictor is None or X is None or y_true is None:
        return {"accuracy": float("nan"), "rule_coverage": float("nan"), "top_k": float("nan")}

    hybrid_results = predictor.predict(X)
    predictions = hybrid_results["prediction"].astype(str)
    truth = pd.Series(y_true).astype(str)
    accuracy = float((predictions == truth).mean()) if len(predictions) else float("nan")
    rule_coverage = float((hybrid_results["source"].astype(str) != "ml").mean()) if len(hybrid_results) else float("nan")

    try:
        proba = predictor.ml_model.predict_proba(X)
        classes = getattr(predictor.ml_model, "classes_", None)
        if classes is None:
            top_k_rate = float("nan")
        else:
            class_index = {cls: idx for idx, cls in enumerate(classes)}
            hits: list[bool] = []
            for row_idx, label in enumerate(truth):
                true_idx = class_index.get(label)
                if true_idx is None:
                    hits.append(False)
                    continue
                ranked = np.argsort(proba[row_idx])[::-1][:top_k]
                hits.append(true_idx in ranked)
            top_k_rate = float(np.mean(hits)) if hits else float("nan")
    except Exception:  # pragma: no cover - defensive
        top_k_rate = float("nan")

    return {
        "accuracy": accuracy,
        "rule_coverage": rule_coverage,
        "top_k": top_k_rate,
    }


def retrain_with_feedback_action(state: Optional[Dict[str, Any]], progress=gr.Progress()):
    state = state or {}
    feedback_df = load_feedback_from_db()
    if len(feedback_df) < 50:
        return (f"⚠️ Nur {len(feedback_df)} Feedbacks vorhanden. Mindestens 50 werden empfohlen.", pd.DataFrame(), state)

    original_features = state.get("df_features_full")
    original_target = state.get("target")
    feature_plan: FeaturePlan | None = state.get("feature_plan")
    if original_features is None or original_target is None or feature_plan is None:
        return ("⚠️ Trainingsdaten oder Pipeline-Konfiguration fehlen.", pd.DataFrame(), state)

    progress(0.1, desc="Bereite Trainingsdaten vor")
    config = _ensure_config(state)
    rf_kwargs = config.model.random_forest.model_dump(exclude_none=True)
    ml_classifier = RandomForestClassifier(**rf_kwargs)
    preprocessor = build_preprocessor(feature_plan, config)

    train_count = min(500, len(original_features))
    if train_count < 2:
        return ("⚠️ Zu wenige Trainingsdaten für den Demo-Modus.", pd.DataFrame(), state)

    X_train_small, _, y_train_small, _ = train_test_split(
        original_features,
        original_target,
        train_size=train_count,
        random_state=config.model.random_forest.random_state,
        stratify=original_target if original_target.nunique() > 1 else None,
    )

    X_feedback, y_feedback = prepare_feedback_features(state, feedback_df)
    if X_feedback.empty:
        return ("⚠️ Feedback konnte nicht den Predictions zugeordnet werden.", pd.DataFrame(), state)

    progress(0.35, desc="Kombiniere Trainingsdaten")
    X_combined = pd.concat([X_train_small.reset_index(drop=True), X_feedback], ignore_index=True)
    y_combined = pd.concat([y_train_small.reset_index(drop=True), y_feedback], ignore_index=True)

    progress(0.55, desc="Trainiere neues Modell")
    new_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", ml_classifier),
    ])
    new_model.fit(X_combined, y_combined)

    rule_engine: RuleEngine | None = state.get("rule_engine")
    if rule_engine is None:
        return ("⚠️ Regel-Engine nicht verfügbar.", pd.DataFrame(), state)

    new_predictor = HybridMassnahmenPredictor(new_model, rule_engine)
    new_predictor.preprocessor_ = new_model.named_steps["preprocessor"]
    new_predictor.feature_names_ = list(state.get("shap_feature_names", [])) or list(X_combined.columns)
    new_predictor.background_ = state.get("shap_background")
    new_predictor.label_encoder_ = state.get("label_encoder")

    old_predictor: Optional[HybridMassnahmenPredictor] = state.get("hybrid_predictor")
    validation_features = state.get("validation_features")
    validation_labels = state.get("validation_labels")
    if validation_features is None or validation_labels is None or old_predictor is None:
        return ("⚠️ Validierungsdaten fehlen für den Vergleich.", pd.DataFrame(), state)

    progress(0.75, desc="Evaluiere Modelle")
    old_metrics = _evaluate_hybrid_predictor(old_predictor, validation_features, validation_labels)
    new_metrics = _evaluate_hybrid_predictor(new_predictor, validation_features, validation_labels)

    comparison_df = pd.DataFrame(
        [
            {
                "Metrik": "Gesamt-Genauigkeit",
                "Vorher": f"{old_metrics['accuracy']:.1%}",
                "Nachher": f"{new_metrics['accuracy']:.1%}",
                "Änderung": f"{(new_metrics['accuracy'] - old_metrics['accuracy']):+.1%}",
            },
            {
                "Metrik": "Regel-Abdeckung",
                "Vorher": f"{old_metrics['rule_coverage']:.1%}",
                "Nachher": f"{new_metrics['rule_coverage']:.1%}",
                "Änderung": f"{(new_metrics['rule_coverage'] - old_metrics['rule_coverage']):+.1%}",
            },
            {
                "Metrik": "Top-3-Quote",
                "Vorher": f"{old_metrics['top_k']:.1%}",
                "Nachher": f"{new_metrics['top_k']:.1%}",
                "Änderung": f"{(new_metrics['top_k'] - old_metrics['top_k']):+.1%}",
            },
            {
                "Metrik": "Neue Labels",
                "Vorher": "-",
                "Nachher": str(len(y_feedback)),
                "Änderung": "-",
            },
        ]
    )

    progress(0.9, desc="Speichere Modell")
    state["hybrid_predictor"] = new_predictor
    state["baseline_model"] = new_model
    state["model_version"] = state.get("model_version", 1) + 1

    summary = textwrap.dedent(
        f"""
        ✅ Retraining abgeschlossen!

        📊 {len(feedback_df)} neue Labels integriert
        🎯 Genauigkeit: {old_metrics['accuracy']:.1%} → {new_metrics['accuracy']:.1%}
        """
    ).strip()

    progress(1.0, desc="Fertig")
    return summary, comparison_df, state


def _format_date_input(value: object) -> Optional[str]:
    if value in (None, "", pd.NA):
        return None
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    return str(value)


def prepare_live_invoice(state: Dict[str, Any], form_values: Dict[str, Any]) -> pd.DataFrame:
    amount_value = float(form_values.get("betrag", 0.0))
    amount_formatted = _format_amount_input(amount_value)
    ampel_value = parse_ampel_choice(form_values.get("ampel"))

    record: Dict[str, Any] = {
        "Betrag": amount_formatted,
        "Betrag_parsed": amount_value,
        "Ampel": ampel_value,
        "BUK": form_values.get("buk") or None,
        "Debitor": form_values.get("debitor") or None,
        "Belegdatum": _format_date_input(form_values.get("belegdatum")),
        "Faellig": _format_date_input(form_values.get("faelligkeit")),
        "DEB_Name": form_values.get("deb_name") or None,
        "Hinweise": form_values.get("hinweise") or None,
        "negativ": bool(form_values.get("negativ", False)),
    }

    df = pd.DataFrame([record])

    template = state.get("df_features_full")
    if isinstance(template, pd.DataFrame):
        for column in template.columns:
            if column not in df.columns:
                df[column] = pd.NA

    selected_columns = state.get("selected_columns") or []
    for column in selected_columns:
        if column not in df.columns:
            df[column] = pd.NA

    ordered_columns: List[str] = []
    if selected_columns:
        ordered_columns.extend(selected_columns)
    for column in df.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)

    return df[ordered_columns]


def _build_live_result_markdown(reason_payload: Dict[str, Any], warnings: List[str]) -> str:
    prediction = reason_payload.get("prediction", "Unbekannt")
    confidence_text = reason_payload.get("confidence_text", "n/a")
    reason_block = reason_payload.get("reason_block", "")

    lines = [
        "### ✅ Empfohlene Maßnahme",
        f"**{prediction}**",
        "",
        f"🎯 Konfidenz: {confidence_text}",
    ]

    if reason_block:
        lines.extend(["", "💡 Begründung:", "", reason_block])

    if warnings:
        lines.append("")
        lines.extend(warnings)

    return "\n".join(lines)


def load_scenario_action(scenario_name: str, state: Optional[Dict[str, Any]] = None):
    state = state or {}
    defaults = _default_live_form()
    scenario = LIVE_SCENARIOS.get(scenario_name)
    if scenario is None:
        scenario = defaults

    updates = []
    updates.append(gr.update(value=scenario.get("betrag", defaults["betrag"])))
    updates.append(gr.update(value=scenario.get("ampel", defaults["ampel"])))
    updates.append(gr.update(value=scenario.get("buk", defaults["buk"])))
    updates.append(gr.update(value=scenario.get("debitor", defaults["debitor"])))
    updates.append(gr.update(value=scenario.get("belegdatum", defaults["belegdatum"])))
    updates.append(gr.update(value=scenario.get("faelligkeit", defaults["faelligkeit"])))
    updates.append(gr.update(value=scenario.get("deb_name", defaults["deb_name"])))
    updates.append(gr.update(value=scenario.get("hinweise", defaults["hinweise"])))
    updates.append(gr.update(value=bool(scenario.get("negativ", defaults["negativ"]))))
    return tuple(updates)


def reset_live_form(state: Optional[Dict[str, Any]] = None):
    return load_scenario_action(LIVE_SCENARIO_DEFAULT, state)


def reset_live_form_action(state: Optional[Dict[str, Any]] = None):
    field_updates = reset_live_form(state)
    updates: List[Any] = [gr.update(value=LIVE_SCENARIO_DEFAULT)]
    updates.extend(field_updates)
    updates.append(gr.update(value=""))  # live_result_status
    updates.append(gr.update(value=""))  # live_result_explanation
    updates.append(gr.update(value=""))  # live_export_status
    updates.append(gr.update(value=None))  # live_export_download
    return tuple(updates)


def predict_live_action(
    betrag: float,
    ampel: object,
    buk: str,
    debitor: str,
    belegdatum,
    faelligkeit,
    deb_name: str,
    hinweise: str,
    negativ: bool,
    state: Optional[Dict[str, Any]],
    progress=gr.Progress(),
):
    state = state or {}
    predictor: Optional[HybridMassnahmenPredictor] = state.get("hybrid_predictor")
    if predictor is None:
        return "❌ Bitte zuerst ein Modell trainieren (Tab 'Training & Analyse').", "", "", state

    try:
        amount_value = float(betrag)
    except (TypeError, ValueError):
        return "❌ Fehler: Betrag muss angegeben werden.", "", "", state

    if amount_value <= 0:
        return "❌ Fehler: Betrag muss positiv sein.", "", "", state

    warnings: List[str] = []
    if amount_value > 1_000_000:
        warnings.append("⚠️ Sehr hoher Betrag, bitte prüfen.")

    ampel_value = parse_ampel_choice(ampel)
    if ampel_value is None:
        return "❌ Fehler: Ampel-Auswahl ist ungültig.", "", "", state

    if not buk or not str(buk).strip() or not debitor or not str(debitor).strip():
        return "❌ Fehler: BUK und Debitor sind Pflichtfelder.", "", "", state

    beleg_date_obj = belegdatum
    faellig_date_obj = faelligkeit
    if isinstance(beleg_date_obj, str) and beleg_date_obj:
        try:
            beleg_date_obj = datetime.strptime(beleg_date_obj, "%Y-%m-%d").date()
        except ValueError:
            try:
                beleg_date_obj = datetime.strptime(beleg_date_obj, "%d.%m.%Y").date()
            except ValueError:
                beleg_date_obj = None
    if isinstance(faellig_date_obj, str) and faellig_date_obj:
        try:
            faellig_date_obj = datetime.strptime(faellig_date_obj, "%Y-%m-%d").date()
        except ValueError:
            try:
                faellig_date_obj = datetime.strptime(faellig_date_obj, "%d.%m.%Y").date()
            except ValueError:
                faellig_date_obj = None

    if isinstance(beleg_date_obj, date) and isinstance(faellig_date_obj, date):
        if faellig_date_obj < beleg_date_obj:
            warnings.append("⚠️ Fälligkeitsdatum liegt vor dem Belegdatum.")

    progress(0.2, desc="Bereite Eingabe vor...")
    form_values = {
        "betrag": amount_value,
        "ampel": ampel,
        "buk": buk,
        "debitor": debitor,
        "belegdatum": beleg_date_obj,
        "faelligkeit": faellig_date_obj,
        "deb_name": deb_name,
        "hinweise": hinweise,
        "negativ": negativ,
    }
    invoice_df = prepare_live_invoice(state, form_values)

    progress(0.5, desc="Führe Prediction aus...")
    results = predictor.predict(invoice_df)
    if results.empty:
        return "❌ Prediction fehlgeschlagen – keine Ausgabe erhalten.", "", "", state

    row_prediction = results.iloc[0]
    explanation = predictor.explain(invoice_df.iloc[0])

    rule_source = row_prediction.get("source")
    rule_confidence = row_prediction.get("confidence") if rule_source and str(rule_source) != "ml" else None

    prediction_row = pd.Series(
        {
            "row_index": 0,
            "final_prediction": row_prediction.get("prediction"),
            "final_confidence": row_prediction.get("confidence"),
            "rule_source": rule_source,
            "rule_prediction": explanation.get("prediction") if rule_source else None,
            "rule_confidence": rule_confidence,
        }
    )
    predictions_df = pd.DataFrame([prediction_row])

    feature_columns = list(invoice_df.columns)
    components = create_explanation_components(
        predictor=predictor,
        row_index=0,
        prediction_row=prediction_row,
        feature_columns=feature_columns,
        predictions_df=predictions_df,
    )

    result_markdown = _build_live_result_markdown(components.payload, warnings)
    explanation_markdown = components.markdown

    state["last_explanation_payload"] = components.payload
    state["last_explanation_markdown"] = explanation_markdown
    state["last_explanation_row_index"] = 0

    progress(0.9, desc="Abgeschlossen")
    return result_markdown, explanation_markdown, "", state
def analyze_patterns_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    target = state.get("target")
    target_name = state.get("target_name")
    if df_features is None or target is None:
        logger.warning("ui_patterns_no_data")
        return "Bitte zuerst Baseline trainieren.", None, None, state

    if df_features.empty or len(target) == 0:
        return "Keine Daten für Musteranalyse vorhanden.", None, None, state

    config = _ensure_config(state)

    detector = FeatureTypeDetector(config)
    feature_infos = detector.detect(df_features)
    generator = InterpretableFeatureGenerator(config)
    generated = generator.generate(df_features, feature_infos)
    if not generated:
        return "Keine interpretierbaren Merkmale generiert.", None, None, state

    target_mapping_raw = state.get("target_mapping") or {}
    target_series_raw = pd.Series(target).reset_index(drop=True)
    target_strings = target_series_raw.astype("string").fillna("Unbekannt")
    label_to_code: dict[str, int] = {}
    for label, code in target_mapping_raw.items():
        try:
            label_to_code[str(label)] = int(code)
        except (TypeError, ValueError):
            continue
    if not label_to_code:
        unique_labels = list(dict.fromkeys(target_strings.tolist()))
        label_to_code = {str(val): idx for idx, val in enumerate(unique_labels, start=1)}
        state["target_mapping"] = label_to_code
    if "Unbekannt" not in label_to_code:
        label_to_code["Unbekannt"] = max(label_to_code.values(), default=0) + 1
    target_encoded = target_strings.map(label_to_code).astype(int)
    inverse_mapping = {code: label for label, code in label_to_code.items()}

    analyzer = ConditionalProbabilityAnalyzer(config)
    insights = analyzer.analyze(generated, target_encoded, inverse_mapping)
    if not insights:
        return "Keine signifikanten Muster gefunden.", None, None, state

    formatter = InsightFormatter(target_name=target_name)
    lines = formatter.format_many(insights)
    markdown = "\n".join(f"- {line}" for line in lines)
    state["pattern_summary_markdown"] = markdown

    df_rows = [
        {
            "feature": insight.feature_description,
            "value": insight.feature_value_label,
            "target": insight.target_value,
            "probability": insight.probability,
            "baseline": insight.baseline_probability,
            "lift": insight.lift,
            "delta": insight.delta,
            "support": insight.support,
            "population": insight.population,
            "support_ratio": insight.support_ratio,
            "p_value": insight.p_value,
            "chi2": insight.chi2,
        }
        for insight in insights
    ]
    df_out = pd.DataFrame(df_rows)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_patterns_"))
    csv_path = tmp_dir / "pattern_insights.csv"
    df_out.to_csv(csv_path, index=False)

    state["pattern_insights_df"] = df_out
    state["pattern_insights_path"] = str(csv_path)

    status = f"{len(insights)} Muster gefunden."
    return status, df_out, str(csv_path), state


def handle_menu_action(
    action: Optional[str],
    state: Optional[Dict[str, Any]],
    file_input,
    folder_input,
    config_input,
    sheet_input,
    target_input,
    explain_index,
    feedback_user,
    feedback_comment,
    feedback_index,
    column_selector,
    balance_checkbox,
    synth_base,
    synth_config,
    synth_business_rules,
    synth_profile,
    synth_business_rules_text,
    synth_profile_text,
    synth_gpt_enable,
    synth_gpt_model,
    synth_gpt_key,
    synth_bias_prompt,
    synth_bias_yaml,
    synth_variation,
    synth_lines,
    synth_ratio,
    synth_seed,
    synth_balance_checkbox=None,
    synth_debug,
    batch_file,
    rule_coverage_box,
    ml_fallback_box,
    massnahmen_distribution_plot,
    rule_explanation_component,
    *,
    defaults: Dict[str, Any],
):
    action_key = defaults["action_map"].get(action)
    state_copy = dict(state) if state else _initial_state()
    updated_state = state_copy

    outputs: Dict[str, Any] = {name: gr.update() for name in defaults["output_names"]}
    component_updates: Dict[str, Any] = {}
    menu_status = "Bitte eine Menü-Aktion auswählen."
    menu_download_value: Any = gr.update()

    if action_key == "save":
        sections = {
            "training": {
                "files": file_input,
                "folder": folder_input,
                "config_file": config_input,
                "sheet": sheet_input,
                "target": target_input,
                "explain_index": explain_index,
                "feedback_user": feedback_user,
                "feedback_comment": feedback_comment,
                "feedback_index": feedback_index,
            },
            "configuration": {
                "selected_columns": column_selector,
                "balance_classes": balance_checkbox,
            },
            "analysis": {
                "rule_coverage": rule_coverage_box,
                "ml_fallback": ml_fallback_box,
                "distribution": massnahmen_distribution_plot,
                "explanation": rule_explanation_component,
            },
            "synthetic": {
                "base_file": synth_base,
                "config_file": synth_config,
                "business_rules_file": synth_business_rules,
                "profile_file": synth_profile,
                "business_rules_text": synth_business_rules_text,
                "profile_text": synth_profile_text,
                "gpt_enable": synth_gpt_enable,
                "gpt_model": synth_gpt_model,
                "gpt_key": synth_gpt_key,
                "bias_prompt": synth_bias_prompt,
                "bias_yaml": synth_bias_yaml,
                "variation": synth_variation,
                "lines": synth_lines,
                "ratio": synth_ratio,
                "seed": synth_seed,
                "balance_classes": synth_balance_checkbox,
                "debug": synth_debug,
            },
            "batch_prediction": {
                "batch_file": batch_file,
            },
        }
        payload = _build_inputs_payload(state_copy, sections)
        menu_download_value = _write_inputs_payload(payload)
        menu_status = "Eingaben als JSON gespeichert."
    elif action_key == "reset_all":
        updated_state = _initial_state()
        component_updates.update(defaults["training_reset"])
        component_updates.update(defaults["synth_reset"])
        component_updates.update(defaults["batch_reset"])
        component_updates["config_info"] = gr.update(value={})
        component_updates["column_selector"] = gr.update(value=[], choices=[])
        component_updates["balance_checkbox"] = gr.update(value=False)
        menu_download_value = gr.update(value=None)
        menu_status = "Alle Tabs zurückgesetzt."
    elif action_key == "reset_training":
        updated_state = dict(state_copy)
        updated_state = _clear_state_keys(updated_state, TRAINING_STATE_KEYS)
        updated_state = _reset_pipeline_state(updated_state)
        updated_state["balance_classes"] = False
        component_updates.update(defaults["training_reset"])
        component_updates["config_info"] = gr.update(value={})
        component_updates["column_selector"] = gr.update(value=[], choices=[])
        component_updates["balance_checkbox"] = gr.update(value=False)
        menu_download_value = gr.update(value=None)
        menu_status = "Training & Analyse zurückgesetzt."
    elif action_key == "reset_config":
        updated_state = dict(state_copy)
        df_full = updated_state.get("df_features_full")
        component_updates["balance_checkbox"] = gr.update(value=False)
        updated_state["balance_classes"] = False
        if isinstance(df_full, pd.DataFrame) and not df_full.empty:
            available_columns = list(df_full.columns)
            default_columns = [c for c in available_columns if c not in AUTO_EXCLUDE_FEATURES]
            if not default_columns:
                default_columns = available_columns
            updated_state["selected_columns"] = default_columns
            if default_columns:
                updated_state["df_features"] = df_full[default_columns].copy()
            else:
                updated_state["df_features"] = df_full.iloc[:, 0:0].copy()
            component_updates["column_selector"] = gr.update(value=default_columns, choices=available_columns)
            component_updates["config_info"] = _build_config_overview(updated_state)
            status_text = (
                f"{len(default_columns)} Spalten automatisch ausgewählt."
                if default_columns
                else "Keine Spalten ausgewählt."
            )
            component_updates["column_status"] = status_text
            menu_status = "Konfiguration auf Standardauswahl zurückgesetzt."
        else:
            updated_state = _clear_state_keys(updated_state, {"selected_columns", "df_features"})
            component_updates["column_selector"] = gr.update(value=[], choices=[])
            component_updates["config_info"] = gr.update(value={})
            component_updates["column_status"] = ""
            menu_status = "Keine Daten geladen – Konfiguration geleert."
        menu_download_value = gr.update(value=None)
    elif action_key == "reset_synth":
        updated_state = dict(state_copy)
        updated_state = _clear_state_keys(updated_state, SYNTH_STATE_KEYS)
        updated_state["bias_rules_yaml"] = ""
        component_updates.update(defaults["synth_reset"])
        menu_download_value = gr.update(value=None)
        menu_status = "Synthetik-Tab zurückgesetzt."
    elif action_key == "reset_batch":
        component_updates.update(defaults["batch_reset"])
        menu_download_value = gr.update(value=None)
        menu_status = "Batch Prediction zurückgesetzt."

    outputs["menu_status"] = menu_status
    outputs["menu_download"] = menu_download_value
    outputs["state"] = updated_state
    for key, value in component_updates.items():
        outputs[key] = value

    return tuple(outputs[name] for name in defaults["output_names"])


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    default_business_rules_text = _load_text(DEFAULT_BUSINESS_RULES_PATH)
    default_profile_text = _load_text(DEFAULT_PROFILE_PATH)

    with gr.Blocks(title="pruefomat") as demo:
        gr.Markdown(
            """
            # pruefomat – Fraud/Veri-Selector
            """
        )
        gr.HTML(
            """
            <style>
            .pf-field-label {font-weight:600;margin-bottom:4px;display:flex;align-items:center;gap:6px;}
            .pf-tooltip {position:relative;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;background:#f8f9fa;color:#141414;font-size:12px;border:1px solid #d0d7de;cursor:help;}
            .pf-tooltiptext {visibility:hidden;opacity:0;transition:opacity 0.2s ease;position:absolute;z-index:20;background:#ffffff;color:#141414;padding:10px;border-radius:6px;width:280px;bottom:130%;left:50%;transform:translateX(-50%);box-shadow:0 4px 12px rgba(15,23,42,0.18);border:1px solid #d0d7de;font-weight:400;line-height:1.4;}
            .pf-tooltiptext::after {content:"";position:absolute;top:100%;left:50%;margin-left:-6px;border-width:6px;border-style:solid;border-color:#ffffff transparent transparent transparent;}
            .gradio-container .pf-tooltiptext,
            .gradio-container .pf-tooltiptext::after,
            .gradio-container.dark .pf-tooltiptext,
            .gradio-container.dark .pf-tooltiptext::after {background:#ffffff !important;color:#141414 !important;border-color:#d0d7de !important;}
            .gradio-container.dark .pf-tooltiptext::after {border-color:#ffffff transparent transparent transparent !important;}
            .pf-tooltip:hover .pf-tooltiptext {visibility:visible;opacity:1;}
            </style>
            """
        )

        state = gr.State(_initial_state())
        target_default = DEFAULT_CONFIG.data.target_col or ""
        sheet_default = "0"

        menu_action_map = {
            "💾 Eingaben speichern": "save",
            "♻️ Alles zurücksetzen": "reset_all",
            "🧹 Training & Analyse zurücksetzen": "reset_training",
            "🧭 Konfiguration zurücksetzen": "reset_config",
            "🧪 Synthetische Daten zurücksetzen": "reset_synth",
            "📦 Batch Prediction zurücksetzen": "reset_batch",
        }

        with gr.Column():
            gr.Markdown("### Aktionen-Menü")
            with gr.Row(equal_height=True):
                menu_action = gr.Dropdown(
                    choices=list(menu_action_map.keys()),
                    label="Aktion",
                    value=None,
                    interactive=True,
                )
                menu_button = gr.Button("Ausführen", variant="secondary")
                menu_status = gr.Textbox(label="Menü-Status", interactive=False)
            menu_download = gr.File(label="Menü-Download", interactive=False)

        with gr.Tabs():
            with gr.TabItem("Training & Analyse"):
                with gr.Column():
                    gr.Markdown("## Training & Analyse")
                    with gr.Row():
                        file_input = gr.File(label="Excel-Dateien", file_types=[".xlsx", ".xls"])  # type: ignore[arg-type]
                        folder_input = gr.Textbox(label="Ordner (optional)", placeholder="Pfad zu einem Ordner mit Excel-Dateien")
                        config_input = gr.File(label="Config (optional)", file_types=[".yaml", ".yml", ".json"])  # type: ignore[arg-type]
                        sheet_input = gr.Textbox(value=sheet_default, label="Sheet (Index oder Name)")
                        target_input = gr.Textbox(value=target_default, label="Zielspalte (optional)")
                        load_btn = gr.Button("Daten laden")

                    load_status = gr.Textbox(label="Status", interactive=False)
                    data_preview = gr.Dataframe(label="Daten (erste Zeilen)", interactive=False)
                    with gr.Accordion("Schema / Mapping", open=False):
                        schema_json = gr.JSON(label="Schema / Mapping")

                    gr.Markdown("## Pipeline")
                    build_btn = gr.Button("Pipeline bauen")
                    build_status = gr.Textbox(label="Pipeline-Status", interactive=False)
                    with gr.Accordion("Feature-Plan", open=False):
                        plan_json = gr.JSON(label="Feature-Plan")
                        prep_download = gr.File(label="Preprocessor Download", interactive=False)

                    preview_btn = gr.Button("Features Vorschau")
                    preview_status = gr.Textbox(label="Preview-Status", interactive=False)
                    with gr.Accordion("Transformierte Features", open=False):
                        preview_table = gr.Dataframe(label="Transformierte Features", interactive=False)

                    gr.Markdown("## Baseline Modell")
                    baseline_btn = gr.Button("Baseline trainieren")
                    baseline_status = gr.Textbox(label="Training-Status", interactive=False)
                    with gr.Accordion("Metriken", open=False):
                        metrics_json = gr.JSON(label="Metriken")
                        metrics_download = gr.File(label="Metrics JSON", interactive=False)
                    with gr.Accordion("Konfusionsmatrix", open=False):
                        confusion_df = gr.Dataframe(label="Konfusionsmatrix", interactive=False)
                        confusion_plot = gr.Image(label="Konfusionsmatrix (Heatmap)", interactive=False)
                    model_download = gr.File(label="Baseline Modell", interactive=False)
                    with gr.Accordion("Feature Importances", open=False):
                        importance_table = gr.Dataframe(label="Feature Importances", interactive=False)
                        importance_plot = gr.Image(label="Feature Importances Plot", interactive=False)
                        importance_download = gr.File(label="Feature Importances CSV", interactive=False)
                    predictions_table = gr.Dataframe(label="Predictions", interactive=False)
                    predictions_download = gr.File(label="Predictions CSV", interactive=False)

                    gr.Markdown("## Feedback")
                    with gr.Row():
                        feedback_index = gr.Number(label="Rechnung #", value=0, precision=0)
                        feedback_correct_btn = gr.Button("👍 Korrekt", variant="primary")
                        feedback_incorrect_btn = gr.Button("👎 Falsch", variant="secondary")
                    feedback_modal = gr.Column(visible=False)
                    with feedback_modal:
                        feedback_modal_title = gr.Markdown("Feedback")
                        feedback_actual_dropdown = gr.Dropdown(
                            label="Tatsächliche Maßnahme",
                            choices=CANONICAL_MASSNAHMEN,
                            value=CANONICAL_MASSNAHMEN[0],
                        )
                        feedback_comment = gr.Textbox(label="Kommentar (optional)", lines=3)
                        feedback_user = gr.Textbox(label="Ihr Name", value="Workshop Gast")
                        with gr.Row():
                            feedback_save_btn = gr.Button("Speichern", variant="primary")
                            feedback_cancel_btn = gr.Button("Abbrechen")
                    feedback_status = gr.Textbox(label="Feedback-Status", interactive=False)
                    feedback_report_btn = gr.Button("Feedback-Report erzeugen")
                    feedback_report_status = gr.Textbox(label="Report-Status", interactive=False)
                    feedback_report_preview = gr.Dataframe(label="Feedback Report", interactive=False)
                    feedback_report_download = gr.File(label="Feedback Report Download", interactive=False)

                    gr.Markdown("## Musteranalyse")
                    pattern_btn = gr.Button("Auffällige Muster analysieren")
                    pattern_status = gr.Textbox(label="Muster-Status", interactive=False)
                    pattern_preview = gr.Dataframe(label="Auffällige Muster", interactive=False)
                    pattern_download = gr.File(label="Muster Download", interactive=False)
                    report_btn = gr.Button("Report generieren")
                    report_status = gr.Textbox(label="Report-Status", interactive=False)
                    report_preview = gr.Markdown(label="Report Vorschau")
                    report_download = gr.File(label="Report Download", interactive=False)

                    gr.Markdown("## Maßnahmen-Analyse")
                    with gr.Row():
                        rule_coverage_box = gr.Textbox(label="Rule Coverage", interactive=False)
                        ml_fallback_box = gr.Textbox(label="ML Fallback Rate", interactive=False)
                    massnahmen_distribution_plot = gr.BarPlot(
                        label="Verteilung vorhergesagter Maßnahmen",
                        x="Massnahme",
                        y="Anzahl",
                    )

                    gr.Markdown("### Einzel-Erklärung")
                    with gr.Row():
                        explain_index = gr.Number(label="Zeilenindex", value=0, precision=0)
                        explain_btn = gr.Button("🔍 Erklärung anzeigen")
                        export_md_btn = gr.Button("📄 Als Markdown exportieren")
                    explain_status = gr.Textbox(label="Erklärungs-Status", interactive=False)
                    rule_explanation = gr.Markdown(label="Erklärung")
                    explain_download = gr.File(label="Erklärungs-Download", interactive=False)

            with gr.TabItem("Live-Demo"):
                live_defaults = _default_live_form()
                with gr.Column():
                    gr.Markdown("## Neue Rechnung eingeben")
                    scenario_dropdown = gr.Dropdown(
                        choices=[LIVE_SCENARIO_DEFAULT] + list(LIVE_SCENARIOS.keys()),
                        label="Beispiel-Szenario laden",
                        value=LIVE_SCENARIO_DEFAULT,
                    )

                    with gr.Row():
                        betrag_input = gr.Number(label="Betrag (€)", value=live_defaults["betrag"], precision=2)
                        ampel_input = gr.Dropdown(
                            label="Ampel",
                            choices=["Grün (1)", "Gelb (2)", "Rot (3)"],
                            value=live_defaults["ampel"],
                        )
                    with gr.Row():
                        buk_input = gr.Textbox(label="BUK", value=live_defaults["buk"])
                        debitor_input = gr.Textbox(label="Debitor", value=live_defaults["debitor"])

                    with gr.Row():
                        belegdatum_input = gr.Date(label="Belegdatum", value=live_defaults["belegdatum"])
                        faelligkeit_input = gr.Date(label="Fälligkeitsdatum", value=live_defaults["faelligkeit"])

                    deb_name_input = gr.Textbox(label="DEB-Name", value=live_defaults["deb_name"])
                    hinweise_input = gr.Textbox(
                        label="Hinweise",
                        value=live_defaults["hinweise"],
                        lines=3,
                    )
                    negativ_checkbox = gr.Checkbox(
                        label="Historisch abgelehnt (negativ-Flag)",
                        value=live_defaults["negativ"],
                    )

                    submit_live_btn = gr.Button("🚀 Empfehlung berechnen", variant="primary")

                    live_result_status = gr.Markdown(label="Ergebnis")
                    live_result_explanation = gr.Markdown(label="Begründung")

                    with gr.Row():
                        export_live_btn = gr.Button("📄 Erklärung exportieren")
                        reset_live_btn = gr.Button("🔄 Neue Rechnung")

                    live_export_status = gr.Textbox(label="Status", interactive=False)
                    live_export_download = gr.File(label="Markdown-Download", interactive=False)

            with gr.TabItem("Feedback & Lernen"):
                with gr.Column():
                    gr.Markdown("## Feedback-Dashboard")
                    feedback_summary_md = gr.Markdown(label="Übersicht", value="Noch kein Feedback erfasst.")
                    feedback_trend_plot = gr.LinePlot(
                        label="Trend über Zeit",
                        x="week",
                        y="accuracy",
                        overlay_point=True,
                        tooltip=["accuracy", "feedbacks"],
                    )
                    feedback_per_class_df = gr.Dataframe(label="Pro Maßnahme", interactive=False)
                    feedback_critical_md = gr.Markdown(label="Hinweise", value="Keine kritischen Maßnahmen erkannt.")

                    gr.Markdown("## Retraining")
                    retrain_btn = gr.Button("🔄 Modell nachtrainieren (Demo-Modus)")
                    retrain_status = gr.Textbox(label="Status", interactive=False)
                    retrain_comparison_df = gr.Dataframe(label="Vorher-Nachher-Vergleich", interactive=False)

            with gr.TabItem("Konfiguration"):
                with gr.Column():
                    gr.Markdown("## Konfiguration")
                    config_info = gr.JSON(label="Aktive Konfiguration", value={})
                    column_selector = gr.CheckboxGroup(label="Spalten für das Training", choices=[])
                    balance_checkbox = gr.Checkbox(label="Ampel-Klassenbalancierung (Oversampling)", value=False)
                    column_status = gr.Textbox(label="Konfigurations-Status", interactive=False)

            with gr.TabItem("Synthetische Daten"):
                with gr.Column():
                    gr.Markdown("## Synthetische Daten")
                    gr.HTML(_tooltip_label(
                        "Ausgangsdatei (Excel)",
                        "Referenzdaten (z. B. `Veri-Bsp.xlsx`). Struktur, Spalten & Wertebereich leiten daraus die Synthese ab."
                    ))
                    synth_base = gr.File(show_label=False, file_types=[".xlsx", ".xls"])  # type: ignore[arg-type]

                    gr.HTML(_tooltip_label(
                        "Config (optional)",
                        "Pipeline-Overrides für Spalten, Textfeatures etc. Leer lassen → Standard `configs/default.yaml` wird genutzt."
                    ))
                    synth_config = gr.File(show_label=False, file_types=[".yaml", ".yml", ".json"])  # type: ignore[arg-type]

                    with gr.Row():
                        with gr.Column():
                            gr.HTML(_tooltip_label(
                                "Business Rules (optional)",
                                "Eigene Geschäftslogik (MwSt, Fixwerte). Alternativ unten im YAML-Editor anpassen."
                            ))
                            synth_business_rules = gr.File(show_label=False, file_types=[".yaml", ".yml", ".json"])  # type: ignore[arg-type]
                        with gr.Column():
                            gr.HTML(_tooltip_label(
                                "Profil (optional)",
                                "Vordefinierte Variation/Textpools. Leer lassen → `invoice_profile.yaml`. Änderungen über den YAML-Editor möglich."
                            ))
                            synth_profile = gr.File(show_label=False, file_types=[".yaml", ".yml", ".json"])  # type: ignore[arg-type]
                    gr.HTML(_tooltip_label(
                        "Business Rules Override (YAML)",
                        "Vorbelegung aus `configs/business_rules.yaml`. Hier direkt anpassen statt Datei hochzuladen."
                    ))
                    synth_business_rules_text = gr.Code(
                        label="Business Rules (YAML)",
                        value=default_business_rules_text,
                        language="yaml",
                    )
                    gr.HTML(_tooltip_label(
                        "Profil Override (YAML)",
                        "Base-Profile (Lieferzeiten etc.). Leer lassen → Standardprofil."
                    ))
                    synth_profile_text = gr.Code(
                        label="Profil (YAML)",
                        value=default_profile_text,
                        language="yaml",
                    )
                    with gr.Row():
                        synth_gpt_enable = gr.Checkbox(label="GPT verwenden", value=True)
                        synth_gpt_model = gr.Textbox(label="GPT-Modell", value="gpt-5-mini")
                        synth_gpt_key = gr.Textbox(label="OpenAI API-Key", type="password")
                    gr.HTML(_tooltip_label(
                        "Bias-Prompt",
                        "Textbeschreibung geplanter Verzerrungen (z. B. höherer Fraud-Anteil an Wochenenden)."
                    ))
                    synth_bias_prompt = gr.Textbox(label="Bias-Prompt", placeholder="z. B. Erhöhe Fraud am Wochenende")
                    synth_bias_button = gr.Button("Bias-Regeln generieren")
                    synth_bias_status = gr.Textbox(label="Bias-Status", interactive=False)
                    synth_bias_yaml = gr.Textbox(
                        label="Bias-Regeln (YAML)",
                        placeholder="Hier erscheinen die generierten Bias-Regeln – optional editierbar",
                        lines=8,
                        value="",
                    )
                    gr.HTML(_tooltip_label(
                        "Variation",
                        "0 = minimale Mutationen (Original fast kopiert), 1 = starke Abweichungen (neue Texte/Beträge). Beispiel: 0.3 erzeugt leichte Textvarianten."
                    ))
                    synth_variation = gr.Slider(0.0, 1.0, value=0.35, step=0.05, show_label=False)
                    synth_balance = gr.Checkbox(label="Klassen-Ungleichgewicht beheben", value=False)
                    with gr.Row():
                        with gr.Column():
                            gr.HTML(_tooltip_label(
                                "Ziel-Zeilen (pro Sheet)",
                                "Absolute Anzahl zusätzlicher Zeilen je Blatt. Überschreibt das Ratio. Beispiel: 250 → jeweils 250 synthetische Zeilen."
                            ))
                            synth_lines = gr.Number(value=100, precision=0, show_label=False)
                        with gr.Column():
                            gr.HTML(_tooltip_label(
                                "Ratio (optional)",
                                "Relative Menge im Vergleich zur Vorlage (1.0 = gleich viele Zeilen). Wird ignoriert, wenn Ziel-Zeilen gesetzt sind."
                            ))
                            synth_ratio = gr.Number(value=None, show_label=False)
                        with gr.Column():
                            gr.HTML(_tooltip_label(
                                "Seed",
                                "Deterministischer Zufalls-Seed. Gleiche Eingaben + Seed → reproduzierbare Ergebnisse."
                            ))
                            synth_seed = gr.Number(value=1234, precision=0, show_label=False)

                    gr.HTML(_tooltip_label(
                        "Debug-Logging aktivieren",
                        "Schreibt detaillierte Synthesizer-Logs (z. B. pro Tabelle). Hilfreich beim Debugging, erzeugt aber mehr Output."
                    ))
                    synth_debug = gr.Checkbox(show_label=False, value=False)
                    synth_bias_prompt.info = "Verwendet GPT, um Wahrscheinlichkeiten z. B. anhand von Wochentagen oder Betragsmustern anzupassen."
                    synth_gpt_model.info = "Welches OpenAI-Modell für Textplausibilisierung genutzt wird."
                    synth_gpt_key.info = "API-Key nur nötig, wenn GPT aktiv ist. Wird temporär gesetzt und nach dem Lauf wieder entfernt."
                    synth_debug.info = "Schreibt detaillierte Fortschrittslogs (z. B. pro Tabelle). Gut zum Troubleshooting, verlängert ggf. den Output."
                    synth_button = gr.Button("Synthetische Daten erzeugen")
                    synth_status = gr.Textbox(label="Generator-Status", interactive=False)
                    synth_preview = gr.Dataframe(label="Vorschau der synthetischen Daten", interactive=False)
                    synth_download = gr.File(label="Download Synthetic Workbook", interactive=False)
                    synth_quality = gr.File(label="Download Quality Report", interactive=False)
                    synth_log = gr.Textbox(label="Generator-Log", lines=12, interactive=False)

            with gr.TabItem("Batch Prediction"):
                with gr.Column():
                    gr.Markdown("## Batch Prediction")
                    batch_file = gr.File(label="Excel-Datei", file_types=[".xlsx", ".xls"])  # type: ignore[arg-type]
                    batch_button = gr.Button("Belege prüfen")
                    batch_status = gr.Textbox(label="Batch-Status", interactive=False)
                    batch_download = gr.File(label="Batch Download", interactive=False)

        load_btn.click(
            load_dataset,
            inputs=[file_input, config_input, sheet_input, target_input, folder_input, state],
            outputs=[load_status, data_preview, schema_json, config_info, column_selector, column_status, balance_checkbox, state],
        )

        column_selector.change(
            update_selected_columns_action,
            inputs=[column_selector, state],
            outputs=[column_status, config_info, state],
        )

        balance_checkbox.change(
            update_balance_action,
            inputs=[balance_checkbox, state],
            outputs=[column_status, config_info, state],
        )

        synth_button.click(
            generate_synthetic_data_action,
            inputs=[
                synth_base,
                synth_config,
                synth_business_rules,
                synth_profile,
                synth_business_rules_text,
                synth_profile_text,
                synth_bias_prompt,
                synth_bias_yaml,
                synth_variation,
                synth_lines,
                synth_ratio,
                synth_seed,
                synth_gpt_model,
                synth_gpt_key,
                synth_gpt_enable,
                synth_balance,
                synth_debug,
                state,
            ],
            outputs=[synth_status, synth_preview, synth_download, synth_quality, synth_log, state],
        )

        synth_bias_button.click(
            generate_bias_rules_action,
            inputs=[
                state,
                synth_bias_prompt,
                synth_business_rules_text,
                synth_base,
                synth_gpt_enable,
                synth_gpt_model,
                synth_gpt_key,
            ],
            outputs=[synth_bias_status, synth_bias_yaml, state],
        )

        build_btn.click(
            build_pipeline_action,
            inputs=[state],
            outputs=[build_status, plan_json, prep_download, state],
        )

        preview_btn.click(
            preview_features_action,
            inputs=[state],
            outputs=[preview_status, preview_table],
        )

        baseline_btn.click(
            train_baseline_action,
            inputs=[state],
            outputs=[
                baseline_status,
                metrics_json,
                confusion_df,
                confusion_plot,
                model_download,
                metrics_download,
                importance_table,
                importance_plot,
                importance_download,
                predictions_table,
                predictions_download,
                rule_coverage_box,
                ml_fallback_box,
                massnahmen_distribution_plot,
                feedback_summary_md,
                feedback_trend_plot,
                feedback_per_class_df,
                feedback_critical_md,
                retrain_status,
                retrain_comparison_df,
                state,
            ],
        )

        explain_btn.click(
            explain_massnahme_action,
            inputs=[state, explain_index],
            outputs=[explain_status, rule_explanation, explain_download, state],
        )

        export_md_btn.click(
            export_explanation_action,
            inputs=[state],
            outputs=[explain_status, explain_download, state],
        )

        scenario_dropdown.change(
            load_scenario_action,
            inputs=[scenario_dropdown, state],
            outputs=[
                betrag_input,
                ampel_input,
                buk_input,
                debitor_input,
                belegdatum_input,
                faelligkeit_input,
                deb_name_input,
                hinweise_input,
                negativ_checkbox,
            ],
        )

        submit_live_btn.click(
            predict_live_action,
            inputs=[
                betrag_input,
                ampel_input,
                buk_input,
                debitor_input,
                belegdatum_input,
                faelligkeit_input,
                deb_name_input,
                hinweise_input,
                negativ_checkbox,
                state,
            ],
            outputs=[live_result_status, live_result_explanation, live_export_status, state],
        )

        export_live_btn.click(
            export_explanation_action,
            inputs=[state],
            outputs=[live_export_status, live_export_download, state],
        )

        reset_live_btn.click(
            reset_live_form_action,
            inputs=[state],
            outputs=[
                scenario_dropdown,
                betrag_input,
                ampel_input,
                buk_input,
                debitor_input,
                belegdatum_input,
                faelligkeit_input,
                deb_name_input,
                hinweise_input,
                negativ_checkbox,
                live_result_status,
                live_result_explanation,
                live_export_status,
                live_export_download,
            ],
        )

        report_btn.click(
            generate_pattern_report_action,
            inputs=[state],
            outputs=[report_status, report_preview, report_download, state],
        )

        pattern_btn.click(
            analyze_patterns_action,
            inputs=[state],
            outputs=[pattern_status, pattern_preview, pattern_download, state],
        )

        feedback_correct_btn.click(
            partial(open_feedback_modal_action, mode="correct"),
            inputs=[state, feedback_index],
            outputs=[
                feedback_modal,
                feedback_modal_title,
                feedback_actual_dropdown,
                feedback_comment,
                feedback_user,
                feedback_status,
                state,
            ],
        )

        feedback_incorrect_btn.click(
            partial(open_feedback_modal_action, mode="incorrect"),
            inputs=[state, feedback_index],
            outputs=[
                feedback_modal,
                feedback_modal_title,
                feedback_actual_dropdown,
                feedback_comment,
                feedback_user,
                feedback_status,
                state,
            ],
        )

        feedback_cancel_btn.click(
            cancel_feedback_modal_action,
            inputs=[state],
            outputs=[feedback_modal, feedback_status, state],
        )

        feedback_save_btn.click(
            submit_feedback_action,
            inputs=[feedback_user, feedback_actual_dropdown, feedback_comment, state],
            outputs=[
                feedback_modal,
                feedback_status,
                feedback_summary_md,
                feedback_trend_plot,
                feedback_per_class_df,
                feedback_critical_md,
                state,
            ],
        )

        feedback_report_btn.click(
            feedback_report_action,
            inputs=[state],
            outputs=[feedback_report_status, feedback_report_preview, feedback_report_download, state],
        )

        retrain_btn.click(
            retrain_with_feedback_action,
            inputs=[state],
            outputs=[retrain_status, retrain_comparison_df, state],
        )

        batch_button.click(
            batch_predict_action,
            inputs=[batch_file, state],
            outputs=[batch_status, batch_download],
        )

    return demo


def main():
    demo = build_interface()
    demo.launch(share=False, show_error=True)


if __name__ == "__main__":
    main()
