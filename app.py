"""Gradio interface for the pruefomat Veri pipeline builder."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import textwrap
from datetime import datetime, timedelta
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

from config_loader import AppConfig, ConfigError, load_config, normalize_config_columns
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
from src.hybrid_predictor import HybridMassnahmenPredictor
from src.train_massnahmen import evaluate_multiclass


os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_DO_NOT_TRACK", "True")
os.environ.setdefault("GRADIO_DISABLE_USAGE_STATS", "True")

configure_logging()
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

AUTO_EXCLUDE_FEATURES = {"Manahme", "source_file", "negativ", "Ruckmeldung_erhalten", "Ticketnummer", "2025"}

FEEDBACK_DB_PATH = Path(json.loads(os.environ.get("PRUEFOMAT_SETTINGS", "{}")).get("feedback_db", "feedback.db"))


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
                comment TEXT
            )
            """
        )
        conn.commit()

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

SYNTH_STATE_KEYS = {"last_generated_path", "last_quality_path", "bias_rules_yaml"}


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


def _collect_excel_files(upload, folder_text: str | None) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    warnings_list: list[str] = []

    if upload is not None:
        uploads = upload if isinstance(upload, list) else [upload]
        for item in uploads:
            try:
                path = Path(item.name)
            except AttributeError:  # pragma: no cover - defensive
                continue
            if not path.exists():
                warnings_list.append(f"Datei nicht gefunden: {path}")
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

    excel_paths, warnings_list = _collect_excel_files(upload, folder_text)
    if not excel_paths:
        logger.warning("ui_no_file", folder=folder_text or None)
        warn = warnings_list[0] if warnings_list else "Bitte zuerst eine Excel-Datei oder einen Ordner auswählen."
        return warn, None, None, state

    config_path: Optional[str]
    if config_upload is not None:
        config_path = config_upload.name
    else:
        config_path = state.get("config_path", str(DEFAULT_CONFIG_PATH))

    try:
        config = _load_app_config(config_path)
    except ConfigError as exc:
        logger.error("ui_config_error", message=str(exc), config_path=config_path)
        return f"Konfigurationsfehler: {exc}", None, None, state

    if target_text:
        config.data.target_col = normalize_column_name(target_text)

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
        return first_error, None, None, state

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
                state["target_mapping"] = {str(val): str(val) for val in target_series.dropna().unique()}
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

    if base_file is None:
        return "Bitte zuerst eine Ausgangsdatei hochladen.", existing_yaml, state

    if not gpt_enabled or not gpt_key:
        return (
            "Bias-Prompts benötigen GPT (API-Key + Aktivierung). Bitte Key hinterlegen und GPT aktivieren.",
            existing_yaml,
            state,
        )

    base_path = Path(base_file.name)
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
    debug_enabled,
    state: Optional[Dict[str, Any]],
):
    state = state or {}
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
    if base_file is None:
        return "Bitte eine Ausgangsdatei auswählen.", None, None, None, "", state

    base_path = Path(base_file.name)
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
        config_path = Path(config_file.name)
    elif DEFAULT_SYNTH_CONFIG.exists():
        config_path = DEFAULT_SYNTH_CONFIG

    business_rules_content = (business_rules_text or "").strip()
    profile_content = (profile_text or "").strip()
    bias_prompt_content = (bias_prompt or "").strip()
    bias_rules_input = bias_rules_yaml if bias_rules_yaml is not None else state.get("bias_rules_yaml", "")
    bias_rules_content = (bias_rules_input or "").strip()
    use_gpt = bool(gpt_enabled) and bool(gpt_key)

    business_rules_path = Path(business_rules_file.name) if business_rules_file is not None else None
    profile_path = Path(profile_file.name) if profile_file is not None else None

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

    try:
        preview_df = pd.read_excel(output_path).head(20)
    except Exception:
        preview_df = None

    status = f"Synthetische Daten erzeugt: {output_path.name}"
    if quality_path.exists():
        status += f" | Quality Report: {quality_path.name}"

    state["last_generated_path"] = str(output_path)
    if quality_path.exists():
        state["last_quality_path"] = str(quality_path)

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

    if df_features is None or feature_plan is None:
        logger.warning("ui_baseline_without_pipeline")
        return "Bitte zuerst Pipeline bauen.", None, None, None, None, state
    if target is None:
        logger.warning("ui_baseline_without_target")
        return "Keine Zielspalte verfuegbar.", None, None, None, None, state

    target_series = pd.Series(target).reset_index(drop=True)
    target_series = target_series.fillna("Unbekannt").astype(str)

    config = _ensure_config(state)
    rf_kwargs = config.model.random_forest.model_dump(exclude_none=True)

    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_plan, config)),
            ("classifier", RandomForestClassifier(**rf_kwargs)),
        ]
    )

    test_size_count, stratify = _compute_split_params(target_series)
    X_train, X_test, y_train, y_test = train_test_split(
        df_features,
        target_series,
        test_size=test_size_count,
        random_state=config.model.random_forest.random_state,
        stratify=stratify,
    )

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
    ml_confidence = proba.max(axis=1)
    ml_prediction = pd.Series(y_pred, name="ml_prediction").reset_index(drop=True)
    ml_confidence_series = pd.Series(ml_confidence, name="ml_confidence").reset_index(drop=True)

    historical_df = X_train.copy()
    historical_df[config.data.target_col or "Massnahme_2025"] = y_train.reset_index(drop=True)

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

    final_prediction = ml_prediction.copy()
    final_confidence = ml_confidence_series.copy()
    rule_prediction = []
    rule_confidence = []
    applied_rule = []

    for idx, row in hybrid_results.iterrows():
        source = row.get("source", "ml") or "ml"
        applied_rule.append(source if source != "ml" else None)
        rule_prediction.append(row.get("prediction") if source != "ml" else None)
        conf = row.get("confidence")
        rule_confidence.append(conf if source != "ml" else None)
        if source != "ml":
            final_prediction.iloc[idx] = row.get("prediction", final_prediction.iloc[idx])
            if conf is not None:
                try:
                    final_confidence.iloc[idx] = float(conf)
                except (TypeError, ValueError):
                    pass

    predictions_df = pd.DataFrame(
        {
            "row_index": np.arange(len(final_prediction)),
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence_series,
            "rule_source": applied_rule,
            "rule_prediction": rule_prediction,
            "rule_confidence": rule_confidence,
            "final_prediction": final_prediction,
            "final_confidence": final_confidence,
            "actual": y_test.reset_index(drop=True),
        }
    )
    predictions_df["confidence_percent"] = predictions_df["final_confidence"].astype(float, errors="ignore") * 100.0
    predictions_df.sort_values("final_confidence", ascending=False, inplace=True)
    predictions_display = predictions_df.head(50).reset_index(drop=True)

    rule_source_series = pd.Series(applied_rule, dtype="object")
    rule_coverage = float(rule_source_series.notna().mean()) if not rule_source_series.empty else 0.0
    ml_fallback_rate = float(rule_source_series.isna().mean()) if not rule_source_series.empty else 0.0
    metrics["rule_coverage"] = rule_coverage
    metrics["ml_fallback_rate"] = ml_fallback_rate

    distribution_df = (
        predictions_df["final_prediction"].astype("string").value_counts().rename_axis("Massnahme").reset_index(name="Anzahl")
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_model_"))
    model_path = tmp_dir / "baseline_model.joblib"
    metrics_path = tmp_dir / "metrics.json"
    importance_path = tmp_dir / "feature_importance.csv"
    plot_path = tmp_dir / "feature_importance.png"
    predictions_path = tmp_dir / "predictions.csv"

    joblib.dump(baseline, model_path)
    importance_df.to_csv(importance_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

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
    }
    metrics_path.write_text(json.dumps(metrics_serializable, indent=2), encoding="utf-8")

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

    return (
        "Baseline trainiert.",
        metrics,
        confusion_df,
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
        state,
    )


def explain_massnahme_action(state: Optional[Dict[str, Any]], row_index):
    state = state or {}
    hybrid_predictor: Optional[HybridMassnahmenPredictor] = state.get("hybrid_predictor")
    df_features = state.get("df_features")

    if hybrid_predictor is None or df_features is None:
        return "Bitte zuerst Baseline trainieren.", None, None, state

    try:
        idx = int(row_index)
    except (TypeError, ValueError):
        return "Ungültiger Index.", None, None, state

    if idx < 0 or idx >= len(df_features):
        return f"Index muss zwischen 0 und {len(df_features) - 1} liegen.", None, None, state

    row = df_features.iloc[idx]
    explanation = hybrid_predictor.explain(row)
    prediction_label = explanation.get("prediction")
    source = explanation.get("source", "ml") or "ml"
    confidence = explanation.get("confidence")
    confidence_text = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "n/a"

    if source == "ml":
        shap_table = _format_shap_table(explanation.get("details", {}).get("shap_top5"))
        formatted = textwrap.dedent(
            f"""
            ### ML-Prediction: {prediction_label}
            **Confidence:** {confidence_text}

            **Top SHAP-Features:**
            {shap_table}
            """
        ).strip()
    else:
        conditions_markdown = _format_conditions(explanation.get("details", {}).get("matched_conditions"))
        formatted = textwrap.dedent(
            f"""
            ### Rule-Based: {prediction_label}
            **Regel:** {source}
            **Confidence:** {confidence_text}

            **Erfüllte Bedingungen:**
            {conditions_markdown}
            """
        ).strip()

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_massnahme_"))
    md_path = tmp_dir / f"explanation_{idx}.md"
    md_path.write_text(formatted, encoding="utf-8")

    return "Erklärung generiert", formatted, str(md_path), state


def generate_pattern_report_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    target = state.get("target")
    feature_importance = state.get("feature_importance_df")
    vectorizer = state.get("notes_text_vectorizer")

    if df_features is None or target is None or feature_importance is None:
        return "Bitte zuerst Baseline trainieren.", None, state

    target_series = pd.Series(target)
    if target_series.nunique(dropna=True) > 2:
        return "Pattern-Report derzeit nur für binäre Ziele verfügbar.", None, state

    report_lines = ["# Fraud Pattern Report", ""]
    top_features = feature_importance.head(10)
    report_lines.append("## Top Risiko-Features")
    for _, row in top_features.iterrows():
        report_lines.append(f"- **{row['feature']}**: Importance {row['importance']:.4f}")
    report_lines.append("")

    df_local = df_features.copy()
    y = target_series.astype(int, errors="ignore")
    if not np.issubdtype(y.dtype, np.number):
        y = pd.Series(np.where(target_series == target_series.mode().iloc[0], 1, 0))
    df_local["is_fraud"] = y

    report_lines.append("## Fraud-Rate nach Land")
    if "Land" in df_local.columns:
        table_land = df_local.groupby("Land")["is_fraud"].mean().sort_values(ascending=False)
        report_lines.append("| Land | Fraud-Rate |")
        report_lines.append("| --- | --- |")
        for land, rate in table_land.items():
            report_lines.append(f"| {land} | {rate:.2%} |")
    else:
        report_lines.append("Keine Spalte 'Land' vorhanden.")
    report_lines.append("")

    report_lines.append("## Fraud-Rate nach BUK")
    if "BUK" in df_local.columns:
        table_buk = df_local.groupby("BUK")["is_fraud"].mean().sort_values(ascending=False)
        report_lines.append("| BUK | Fraud-Rate |")
        report_lines.append("| --- | --- |")
        for buk, rate in table_buk.items():
            report_lines.append(f"| {buk} | {rate:.2%} |")
    else:
        report_lines.append("Keine Spalte 'BUK' vorhanden.")
    report_lines.append("")

    report_lines.append("## Betragsanalyse")
    if "Betrag_parsed" in df_local.columns:
        fraud_amount = df_local.loc[df_local["is_fraud"] == 1, "Betrag_parsed"].mean()
        non_amount = df_local.loc[df_local["is_fraud"] == 0, "Betrag_parsed"].mean()
        report_lines.append(f"- Durchschnittlicher Betrag (Fraud): {fraud_amount:.2f}")
        report_lines.append(f"- Durchschnittlicher Betrag (Normal): {non_amount:.2f}")
    else:
        report_lines.append("Betragsspalte nicht vorhanden.")
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
                candidate_cols.extend([col for col in getattr(feature_plan, "text", []) if col in df_features.columns])
            if not candidate_cols:
                config = _ensure_config(state)
                candidate_cols.extend([col for col in config.data.text_columns if col in df_features.columns])

            if candidate_cols:
                text_source = df_features[candidate_cols].fillna("")
                text_series = text_source.agg(" ".join, axis=1).astype(str)

        if text_series is not None:
            fraud_mask = y == 1
            fraud_texts = text_series.loc[fraud_mask].fillna("")
        else:
            fraud_texts = pd.Series(dtype="string")

        if not fraud_texts.empty:
            transformed = vectorizer.transform(fraud_texts)
            summed = np.asarray(transformed.sum(axis=0)).ravel()
            top_idx = np.argsort(summed)[::-1][:10]
            for idx in top_idx:
                report_lines.append(f"- {feature_names[idx]} (Gewicht {summed[idx]:.2f})")
        else:
            report_lines.append("Keine Fraud-Texte vorhanden.")
    else:
        report_lines.append("Text-Vektorisierer nicht verfügbar.")

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

    if upload is None:
        return "Bitte eine Excel-Datei hochladen.", None

    progress = gr.Progress(track_tqdm=False)
    progress(0.05, desc="Lade Datei")
    df_input_raw = pd.read_excel(upload.name)
    df_input, _ = normalize_columns(df_input_raw)

    df_negativ = None
    if "negativ" in df_input.columns:
        negativ_mask = df_input["negativ"].fillna(False).astype(bool)
        n_negativ = int(negativ_mask.sum())
        if n_negativ > 0:
            logger.info(
                "batch_prediction_filtered_negativ",
                n_filtered=n_negativ,
                n_total=len(df_input),
            )
            df_negativ = df_input.loc[negativ_mask].copy()
            df_negativ["Massnahme_2025"] = "Bereits abgelehnt (negativ)"
            df_negativ["final_confidence"] = 1.0
            df_negativ["prediction_source"] = "negativ_flag"
            df_input = df_input.loc[~negativ_mask].copy()
        else:
            df_negativ = None
        df_input = df_input.drop(columns=["negativ"])
    else:
        df_negativ = None

    config = _ensure_config(state)
    target_col = config.data.target_col
    if target_col and target_col in df_input.columns:
        df_input = df_input.drop(columns=[target_col])

    selected_columns = state.get("selected_columns") or list(df_input.columns)
    missing = [col for col in selected_columns if col not in df_input.columns]
    for col in missing:
        df_input[col] = np.nan
    df_input = df_input[selected_columns]

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
        df_input["Massnahme_2025"] = results["prediction"]
        df_input["final_confidence"] = results["confidence"].astype(float, errors="ignore")
        df_input["prediction_source"] = results["source"]
    else:
        proba = model.predict_proba(df_input)
        ml_prediction = model.predict(df_input)
        ml_confidence = proba.max(axis=1)
        df_input["Massnahme_2025"] = ml_prediction
        df_input["final_confidence"] = ml_confidence
        df_input["prediction_source"] = "ml"

    df_input["final_confidence"] = pd.to_numeric(df_input["final_confidence"], errors="coerce").fillna(0.0)
    df_input["fraud_score"] = df_input["final_confidence"] * 100.0

    if df_negativ is not None:
        df_negativ["fraud_score"] = 100.0
        for column in df_input.columns:
            if column not in df_negativ.columns:
                df_negativ[column] = np.nan
        df_negativ = df_negativ[df_input.columns]
        df_out = pd.concat([df_input, df_negativ], ignore_index=True)
    else:
        df_out = df_input

    progress(0.8, desc="Schreibe Ergebnis")
    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_batch_"))
    out_path = tmp_dir / "batch_predictions.xlsx"
    df_out.to_excel(out_path, index=False)
    progress(1.0, desc="Fertig")

    logger.info(
        "ui_batch_prediction_completed",
        rows_predicted=len(df_input),
        rows_negativ=len(df_negativ) if df_negativ is not None else 0,
        output=str(out_path),
    )

    status = (
        f"Batch abgeschlossen: {len(df_input)} Predictions,"
        f" {len(df_negativ) if df_negativ is not None else 0} bereits abgelehnt"
    )
    return status, str(out_path)


def record_feedback(state: Dict[str, Any], row_index: int, feedback: str, user: str, comment: str) -> str:
    ensure_feedback_db()
    predictions = state.get("predictions_full")
    df_features = state.get("df_features")
    label_encoder = state.get("label_encoder")
    if predictions is None or df_features is None:
        return "Keine Predictions verfügbar."

    if row_index < 0 or row_index >= len(predictions):
        return "Index außerhalb gültiger Grenzen."

    row = predictions.iloc[row_index]
    df_row = df_features.iloc[row_index]
    beleg_id = df_row.get("Rechnungsnummer") if isinstance(df_row, pd.Series) else None

    final_prediction = row.get("final_prediction")
    final_confidence = float(row.get("final_confidence", 0.0)) if row.get("final_confidence") is not None else 0.0
    prediction_code: int
    if label_encoder is not None and final_prediction is not None:
        try:
            prediction_code = int(label_encoder.transform([str(final_prediction)])[0])
        except Exception:  # pragma: no cover - defensive
            prediction_code = 0
    else:
        prediction_code = 0

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO feedback (beleg_index, beleg_id, timestamp, user, score, prediction, feedback, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row_index),
                None if pd.isna(beleg_id) else str(beleg_id),
                datetime.utcnow().isoformat(),
                user or "unknown",
                float(final_confidence * 100.0),
                prediction_code,
                feedback,
                comment or "",
            ),
        )
        conn.commit()

    return "Feedback gespeichert."


def feedback_action(state: Optional[Dict[str, Any]], row_index, user, comment, label):
    state = state or {}
    try:
        idx = int(row_index)
    except (TypeError, ValueError):
        return "Ungültiger Index.", state

    message = record_feedback(state, idx, label, user or "unknown", comment)
    return message, state


def feedback_report_action(state: Optional[Dict[str, Any]]):
    ensure_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM feedback", conn, parse_dates=["timestamp"]) if conn else pd.DataFrame()

    if df.empty:
        return "Noch kein Feedback vorhanden.", "", None, state

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

    return summary, md_text, str(md_path), state


def feedback_tp_action(state, row_index, user, comment):
    return feedback_action(state, row_index, user, comment, "TP")


def feedback_fp_action(state, row_index, user, comment):
    return feedback_action(state, row_index, user, comment, "FP")


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

    target_mapping = state.get("target_mapping") or {}
    inverse_mapping = {int(v): str(k) for k, v in target_mapping.items()} if target_mapping else None

    analyzer = ConditionalProbabilityAnalyzer(config)
    insights = analyzer.analyze(generated, pd.Series(target).reset_index(drop=True), inverse_mapping)
    if not insights:
        return "Keine signifikanten Muster gefunden.", None, None, state

    formatter = InsightFormatter(target_name=target_name)
    lines = formatter.format_many(insights)
    markdown = "\n".join(f"- {line}" for line in lines)

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
    return status, markdown, str(csv_path), state


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
            with gr.Tab("Training & Analyse"):
                with gr.Row():
                    file_input = gr.File(label="Excel-Dateien", file_types=[".xlsx", ".xls"], file_count="multiple")  # type: ignore[arg-type]
                    folder_input = gr.Textbox(label="Ordner (optional)", placeholder="Pfad zu einem Ordner mit Excel-Dateien")
                    config_input = gr.File(label="Config (optional)", file_types=[".yaml", ".yml", ".json"])  # type: ignore[arg-type]
                    sheet_input = gr.Textbox(value=sheet_default, label="Sheet (Index oder Name)")
                    target_input = gr.Textbox(value=target_default, label="Zielspalte (optional)")
                    load_btn = gr.Button("Daten laden")

                load_status = gr.Textbox(label="Status", interactive=False)
                data_preview = gr.Dataframe(label="Daten (erste Zeilen)", interactive=False)
                schema_json = gr.JSON(label="Schema / Mapping")

                gr.Markdown("## Pipeline")
                build_btn = gr.Button("Pipeline bauen")
                build_status = gr.Textbox(label="Pipeline-Status", interactive=False)
                plan_json = gr.JSON(label="Feature-Plan")
                prep_download = gr.File(label="Preprocessor Download", interactive=False)

                preview_btn = gr.Button("Features Vorschau")
                preview_status = gr.Textbox(label="Preview-Status", interactive=False)
                preview_table = gr.Dataframe(label="Transformierte Features", interactive=False)

                gr.Markdown("## Baseline Modell")
                baseline_btn = gr.Button("Baseline trainieren")
                baseline_status = gr.Textbox(label="Baseline-Status", interactive=False)
                metrics_json = gr.JSON(label="Metriken")
                confusion_df = gr.Dataframe(label="Confusion Matrix", interactive=False)
                model_download = gr.File(label="Baseline Modell", interactive=False)
                metrics_download = gr.File(label="Metrics JSON", interactive=False)
                importance_table = gr.Dataframe(label="Feature Importanzen (Top 20)", interactive=False)
                importance_plot = gr.Image(label="Feature Importance Chart", interactive=False)
                importance_download = gr.File(label="Feature Importances CSV", interactive=False)
                predictions_table = gr.Dataframe(label="Predictions (Test Set, sortiert nach Vertrauen)", interactive=False)
                predictions_download = gr.File(label="Predictions CSV", interactive=False)
                report_btn = gr.Button("Pattern Report generieren")
                report_status = gr.Textbox(label="Report-Status", interactive=False)
                report_preview = gr.Markdown(label="Report Vorschau")
                report_download = gr.File(label="Report Download", interactive=False)
                pattern_btn = gr.Button("Musteranalyse starten")
                pattern_status = gr.Textbox(label="Muster-Status", interactive=False)
                pattern_preview = gr.Markdown(label="Automatisch erkannte Muster")
                pattern_download = gr.File(label="Muster Download", interactive=False)
                feedback_user = gr.Textbox(label="Prüfer:in", value="")
                feedback_comment = gr.Textbox(label="Feedback-Kommentar", lines=2)
                feedback_index = gr.Number(label="Zeilenindex (Feedback)", value=0, precision=0)
                feedback_tp_btn = gr.Button("✅ True Positive")
                feedback_fp_btn = gr.Button("❌ False Positive")
                feedback_status = gr.Textbox(label="Feedback-Status", interactive=False)
                feedback_report_btn = gr.Button("Feedback-Report")
                feedback_report_status = gr.Textbox(label="Feedback-Report-Status", interactive=False)
                feedback_report_preview = gr.Markdown(label="Feedback Report Vorschau")
                feedback_report_download = gr.File(label="Feedback Report Download", interactive=False)

            with gr.Tab("Maßnahmen-Analyse"):
                gr.Markdown("## Hybrid Prediction: Rules + ML")
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
                    explain_btn = gr.Button("Erklärung anzeigen")
                explain_status = gr.Textbox(label="Erklärungs-Status", interactive=False)
                rule_explanation = gr.Markdown(label="Erklärung")
                explain_download = gr.File(label="Erklärungs-Download", interactive=False)

            with gr.Tab("Konfiguration"):
                config_info = gr.JSON(label="Aktive Konfiguration", value={})
                column_selector = gr.CheckboxGroup(label="Spalten für das Training", choices=[])
                balance_checkbox = gr.Checkbox(label="Ampel-Klassenbalancierung (Oversampling)", value=False)
                column_status = gr.Textbox(label="Konfigurations-Status", interactive=False)

            with gr.Tab("Synthetische Daten"):
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
                synth_business_rules_text = gr.Textbox(
                    show_label=False,
                    placeholder="Optional: YAML direkt bearbeiten",
                    lines=10,
                    value=default_business_rules_text,
                )
                gr.HTML(_tooltip_label(
                    "Profil Override (YAML)",
                    "Vorbelegung aus `configs/invoice_profile.yaml`. Ergänze z. B. neue Value-Pools oder Variation-Skalen."
                ))
                synth_profile_text = gr.Textbox(
                    show_label=False,
                    placeholder="Optional: YAML direkt bearbeiten",
                    lines=10,
                    value=default_profile_text,
                )
                gr.HTML(_tooltip_label(
                    "GPT-Verfeinerung verwenden",
                    "Aktiviert sprachliche Verfeinerung (z. B. plausiblere Hinweise). Erhöht Laufzeit und API-Kosten."
                ))
                synth_gpt_enable = gr.Checkbox(show_label=False, value=True)
                gr.HTML(_tooltip_label(
                    "GPT-Modell",
                    "Welches OpenAI-Modell für Textplausibilisierung und Bias-Regeln genutzt wird."
                ))
                synth_gpt_model = gr.Dropdown(choices=["gpt-5-mini", "gpt-5", "gpt-4.1-mini"], value="gpt-5-mini", show_label=False)

                gr.HTML(_tooltip_label(
                    "OpenAI API Key",
                    "Nur notwendig, wenn GPT aktiviert ist. Der Key wird während des Laufs gesetzt und danach entfernt."
                ))
                synth_gpt_key = gr.Textbox(type="password", show_label=False)

                gr.HTML(_tooltip_label(
                    "Bias-Prompt (optional)",
                    "Natürliche Sprache → zusätzliche Regeln. Beispiel: 'Erhöhe Ampel Gelb um 5 %, wenn Belegdatum ein Montag ist.' Erfordert GPT."
                ))
                synth_bias_prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Beschreibe gewünschte Tendenzen für die synthetischen Daten",
                    lines=4,
                )
                with gr.Row():
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

            with gr.Tab("Batch Prediction"):
                batch_file = gr.File(label="Excel-Datei", file_types=[".xlsx", ".xls"])  # type: ignore[arg-type]
                batch_button = gr.Button("Belege prüfen")
                batch_status = gr.Textbox(label="Batch-Status", interactive=False)
                batch_download = gr.File(label="Batch Download", interactive=False)

        menu_output_pairs = [
            ("menu_status", menu_status),
            ("menu_download", menu_download),
            ("state", state),
            ("file_input", file_input),
            ("folder_input", folder_input),
            ("config_input", config_input),
            ("sheet_input", sheet_input),
            ("target_input", target_input),
            ("load_status", load_status),
            ("data_preview", data_preview),
            ("schema_json", schema_json),
            ("build_status", build_status),
            ("plan_json", plan_json),
            ("prep_download", prep_download),
            ("preview_status", preview_status),
            ("preview_table", preview_table),
            ("baseline_status", baseline_status),
            ("metrics_json", metrics_json),
            ("confusion_df", confusion_df),
            ("model_download", model_download),
            ("metrics_download", metrics_download),
            ("importance_table", importance_table),
            ("importance_plot", importance_plot),
            ("importance_download", importance_download),
            ("predictions_table", predictions_table),
            ("predictions_download", predictions_download),
            ("rule_coverage", rule_coverage_box),
            ("ml_fallback", ml_fallback_box),
            ("massnahmen_distribution", massnahmen_distribution_plot),
            ("explain_index", explain_index),
            ("explain_status", explain_status),
            ("rule_explanation", rule_explanation),
            ("explain_download", explain_download),
            ("report_status", report_status),
            ("report_preview", report_preview),
            ("report_download", report_download),
            ("pattern_status", pattern_status),
            ("pattern_preview", pattern_preview),
            ("pattern_download", pattern_download),
            ("feedback_user", feedback_user),
            ("feedback_comment", feedback_comment),
            ("feedback_index", feedback_index),
            ("feedback_status", feedback_status),
            ("feedback_report_status", feedback_report_status),
            ("feedback_report_preview", feedback_report_preview),
            ("feedback_report_download", feedback_report_download),
            ("config_info", config_info),
            ("column_status", column_status),
            ("column_selector", column_selector),
            ("balance_checkbox", balance_checkbox),
            ("synth_base", synth_base),
            ("synth_config", synth_config),
            ("synth_business_rules", synth_business_rules),
            ("synth_profile", synth_profile),
            ("synth_business_rules_text", synth_business_rules_text),
            ("synth_profile_text", synth_profile_text),
            ("synth_gpt_enable", synth_gpt_enable),
            ("synth_gpt_model", synth_gpt_model),
            ("synth_gpt_key", synth_gpt_key),
            ("synth_bias_prompt", synth_bias_prompt),
            ("synth_bias_status", synth_bias_status),
            ("synth_bias_yaml", synth_bias_yaml),
            ("synth_variation", synth_variation),
            ("synth_lines", synth_lines),
            ("synth_ratio", synth_ratio),
            ("synth_seed", synth_seed),
            ("synth_debug", synth_debug),
            ("synth_status", synth_status),
            ("synth_preview", synth_preview),
            ("synth_download", synth_download),
            ("synth_quality", synth_quality),
            ("synth_log", synth_log),
            ("batch_file", batch_file),
            ("batch_status", batch_status),
            ("batch_download", batch_download),
        ]
        menu_output_names = [name for name, _ in menu_output_pairs]
        menu_outputs = [component for _, component in menu_output_pairs]

        training_reset_updates = {
            "file_input": gr.update(value=None),
            "folder_input": "",
            "config_input": gr.update(value=None),
            "sheet_input": sheet_default,
            "target_input": target_default,
            "load_status": "",
            "data_preview": gr.update(value=None),
            "schema_json": gr.update(value=None),
            "build_status": "",
            "plan_json": gr.update(value=None),
            "prep_download": gr.update(value=None),
            "preview_status": "",
            "preview_table": gr.update(value=None),
            "baseline_status": "",
            "metrics_json": gr.update(value=None),
            "confusion_df": gr.update(value=None),
            "model_download": gr.update(value=None),
            "metrics_download": gr.update(value=None),
            "importance_table": gr.update(value=None),
            "importance_plot": gr.update(value=None),
            "importance_download": gr.update(value=None),
            "predictions_table": gr.update(value=None),
            "predictions_download": gr.update(value=None),
            "rule_coverage": "",
            "ml_fallback": "",
            "massnahmen_distribution": gr.update(value=None),
            "explain_index": 0,
            "explain_status": "",
            "rule_explanation": gr.update(value=""),
            "explain_download": gr.update(value=None),
            "report_status": "",
            "report_preview": gr.update(value=""),
            "report_download": gr.update(value=None),
            "pattern_status": "",
            "pattern_preview": gr.update(value=""),
            "pattern_download": gr.update(value=None),
            "column_status": "",
            "feedback_user": "",
            "feedback_comment": "",
            "feedback_index": 0,
            "feedback_status": "",
            "feedback_report_status": "",
            "feedback_report_preview": gr.update(value=""),
            "feedback_report_download": gr.update(value=None),
        }

        synth_reset_updates = {
            "synth_base": gr.update(value=None),
            "synth_config": gr.update(value=None),
            "synth_business_rules": gr.update(value=None),
            "synth_profile": gr.update(value=None),
            "synth_business_rules_text": default_business_rules_text,
            "synth_profile_text": default_profile_text,
            "synth_gpt_enable": True,
            "synth_gpt_model": "gpt-5-mini",
            "synth_gpt_key": "",
            "synth_bias_prompt": "",
            "synth_bias_status": "",
            "synth_bias_yaml": "",
            "synth_variation": 0.35,
            "synth_lines": 100,
            "synth_ratio": gr.update(value=None),
            "synth_seed": 1234,
            "synth_debug": False,
            "synth_status": "",
            "synth_preview": gr.update(value=None),
            "synth_download": gr.update(value=None),
            "synth_quality": gr.update(value=None),
            "synth_log": "",
        }

        batch_reset_updates = {
            "batch_file": gr.update(value=None),
            "batch_status": "",
            "batch_download": gr.update(value=None),
        }

        menu_defaults = {
            "action_map": menu_action_map,
            "output_names": menu_output_names,
            "training_reset": training_reset_updates,
            "synth_reset": synth_reset_updates,
            "batch_reset": batch_reset_updates,
        }

        menu_button.click(
            partial(
                handle_menu_action,
                defaults=menu_defaults,
            ),
            inputs=[
                menu_action,
                state,
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
                synth_debug,
                batch_file,
                rule_coverage_box,
                ml_fallback_box,
                massnahmen_distribution_plot,
                rule_explanation,
            ],
            outputs=menu_outputs,
        )

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
                state,
            ],
        )

        explain_btn.click(
            explain_massnahme_action,
            inputs=[state, explain_index],
            outputs=[explain_status, rule_explanation, explain_download, state],
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

        feedback_tp_btn.click(
            feedback_tp_action,
            inputs=[state, feedback_index, feedback_user, feedback_comment],
            outputs=[feedback_status, state],
        )

        feedback_fp_btn.click(
            feedback_fp_action,
            inputs=[state, feedback_index, feedback_user, feedback_comment],
            outputs=[feedback_status, state],
        )

        feedback_report_btn.click(
            feedback_report_action,
            inputs=[state],
            outputs=[feedback_report_status, feedback_report_preview, feedback_report_download, state],
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
