"""Gradio interface for the pruefomat Veri pipeline builder."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import sys
import logging

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
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, fbeta_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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


def _load_app_config(path: Optional[str | Path]) -> AppConfig:
    return normalize_config_columns(load_config(path))


try:
    DEFAULT_CONFIG = _load_app_config(DEFAULT_CONFIG_PATH)
except ConfigError:
    DEFAULT_CONFIG = AppConfig()
    DEFAULT_CONFIG = normalize_config_columns(DEFAULT_CONFIG)


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
    return balanced_df, y_balanced.astype(int)


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
    if target_norm and target_norm in df_norm_all.columns:
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


def generate_synthetic_data_action(
    base_file,
    config_file,
    business_rules_file,
    profile_file,
    business_rules_text,
    profile_text,
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

    use_gpt = bool(gpt_enabled) and bool(gpt_key)
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

    config = _ensure_config(state)
    rf_kwargs = config.model.random_forest.model_dump(exclude_none=True)

    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_plan, config)),
            (
                "classifier",
                RandomForestClassifier(**rf_kwargs),
            ),
        ]
    )

    test_size_count, stratify = _compute_split_params(target)
    X_train, X_test, y_train, y_test = train_test_split(
        df_features,
        target,
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

    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    metrics = {
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "f2_score": float(fbeta_score(y_test, y_pred, average="macro", beta=2.0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    confusion = confusion_matrix(y_test, y_pred)
    confusion_df = pd.DataFrame(confusion)
    confusion_df.columns = [f"Pred {c}" for c in confusion_df.columns]
    confusion_df.index = [f"Actual {i}" for i in confusion_df.index]

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
    if proba.ndim == 1:
        pos_scores = proba
    elif proba.shape[1] == 1:
        pos_scores = proba[:, 0]
    else:
        classes = list(baseline.named_steps["classifier"].classes_)
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
    predictions_raw = pd.Series(baseline.predict(X_test), name="prediction_raw")

    positive_label = None
    classifier = baseline.named_steps.get("classifier")
    classes = list(getattr(classifier, "classes_", []))
    n_classes = len(classes)

    if n_classes == 2:
        positive_index = 1
        for candidate in ("1", 1, "fraud", "Fraud", True):
            if candidate in classes:
                positive_index = classes.index(candidate)
                break
        positive_label = classes[positive_index]

    if positive_label is not None:
        predictions_binary = (predictions_raw == positive_label).astype(int)
    else:
        predictions_binary = pd.Series(
            np.where(fraud_scores >= 50.0, 1, 0),
            index=predictions_raw.index,
            name="prediction",
        )

    predictions_col = predictions_binary.rename("prediction")

    actual_series = pd.Series(y_test).reset_index(drop=True)
    predictions_col = predictions_col.reset_index(drop=True)
    fraud_scores = fraud_scores.reset_index(drop=True)

    predictions_df = pd.DataFrame(
        {
            "row_index": np.arange(len(fraud_scores)),
            "fraud_score": fraud_scores,
            "prediction": predictions_col,
            "prediction_raw": predictions_raw.reset_index(drop=True),
            "actual": actual_series,
        }
    )
    predictions_df.sort_values("fraud_score", ascending=False, inplace=True)
    predictions_display = predictions_df.head(50).reset_index(drop=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_model_"))
    model_path = tmp_dir / "baseline_model.joblib"
    metrics_path = tmp_dir / "metrics.json"
    importance_path = tmp_dir / "feature_importance.csv"
    plot_path = tmp_dir / "feature_importance.png"
    predictions_path = tmp_dir / "predictions.csv"

    joblib.dump(baseline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
    except Exception as exc:
        logger.warning("ui_shap_background_failed", message=str(exc))

    metric_summary = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "importance_path": str(importance_path),
        "predictions_path": str(predictions_path),
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f2_score": metrics["f2_score"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }

    logger.info(
        "ui_baseline_trained",
        **mask_sensitive_data(metric_summary),
    )

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
        state,
    )


def explain_prediction_action(state: Optional[Dict[str, Any]], row_index):
    state = state or {}
    baseline = state.get("baseline_model")
    df_features = state.get("df_features")
    if baseline is None or df_features is None:
        return "Bitte zuerst Baseline trainieren.", None, None, state

    try:
        idx = int(row_index)
    except (TypeError, ValueError):
        return "Ungültiger Index.", None, None, state

    if idx < 0 or idx >= len(df_features):
        return f"Index muss zwischen 0 und {len(df_features)-1} liegen.", None, None, state

    preprocessor = baseline.named_steps.get("preprocessor")
    classifier = baseline.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        return "Pipeline unvollständig.", None, None, state

    if not hasattr(classifier, "estimators_"):
        return "Erklärungen unterstützen derzeit nur Baum-Modelle (z. B. RandomForest).", None, None, state

    feature_names = state.get("shap_feature_names")
    if feature_names is None:
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            transformed = preprocessor.transform(df_features.head(1))
            feature_names = [f"f_{i}" for i in range(transformed.shape[1])]
        state["shap_feature_names"] = feature_names

    sample = df_features.iloc[[idx]]
    transformed = preprocessor.transform(sample)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    background = state.get("shap_background")
    if background is None:
        bg = preprocessor.transform(df_features.head(min(200, len(df_features))))
        background = bg.toarray() if hasattr(bg, "toarray") else bg
        state["shap_background"] = background

    try:
        explainer = shap.TreeExplainer(classifier, background)
        shap_values = explainer.shap_values(transformed_dense)
        if isinstance(shap_values, list):
            classes = list(getattr(classifier, "classes_", [0, 1]))
            pos_idx = 1 if len(classes) > 1 else 0
            for candidate in ("1", 1, True, "fraud", "Fraud"):
                if candidate in classes:
                    pos_idx = classes.index(candidate)
                    break
            shap_row = shap_values[pos_idx][0]
        else:
            shap_row = shap_values[0]
    except Exception as exc:
        logger.error("ui_shap_failed", message=str(exc))
        return f"SHAP konnte nicht berechnet werden: {exc}", None, None, state

    shap_row = np.asarray(shap_row)
    if shap_row.ndim > 1:
        shap_row = shap_row.reshape(-1)
    try:
        shap_row = shap_row.astype(float, copy=False)
    except (TypeError, ValueError):
        flattened = []
        for val in np.asarray(shap_row, dtype=object).ravel():
            arr = np.asarray(val)
            if arr.size == 0:
                continue
            flattened.extend(arr.astype(float, copy=False).ravel().tolist())
        shap_row = np.array(flattened, dtype=float)

    values_array = transformed_dense[0]
    values_array = np.asarray(values_array)
    if values_array.ndim > 1:
        values_array = values_array.reshape(-1)
    values_array = values_array.astype(float, copy=False)

    feature_array = np.asarray(feature_names, dtype=str)
    lengths = (len(feature_array), shap_row.size, values_array.size)
    if len(set(lengths)) != 1:
        min_len = min(lengths)
        logger.warning(
            "ui_shap_length_mismatch",
            feature_count=len(feature_array),
            shap_len=shap_row.size,
            value_len=values_array.size,
            truncated_to=min_len,
        )
        feature_array = feature_array[:min_len]
        shap_row = shap_row[:min_len]
        values_array = values_array[:min_len]

    if shap_row.size == 0:
        return "Keine SHAP-Werte für diese Zeile verfügbar.", None, None, state

    order = np.argsort(np.abs(shap_row))[::-1][:5]
    explanation = []
    for i in order:
        explanation.append(
            {
                "feature": feature_array[i],
                "shap_value": float(shap_row[i]),
                "feature_value": float(values_array[i]),
                "impact": "erhöht Risiko" if shap_row[i] >= 0 else "senkt Risiko",
            }
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_shap_"))
    json_path = tmp_dir / "explanation.json"
    json_path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")

    return f"Top-Features für Zeile {idx}", explanation, str(json_path), state


def generate_pattern_report_action(state: Optional[Dict[str, Any]]):
    state = state or {}
    df_features = state.get("df_features")
    target = state.get("target")
    feature_importance = state.get("feature_importance_df")
    vectorizer = state.get("notes_text_vectorizer")

    if df_features is None or target is None or feature_importance is None:
        return "Bitte zuerst Baseline trainieren.", None, state

    report_lines = ["# Fraud Pattern Report", ""]
    top_features = feature_importance.head(10)
    report_lines.append("## Top Risiko-Features")
    for _, row in top_features.iterrows():
        report_lines.append(f"- **{row['feature']}**: Importance {row['importance']:.4f}")
    report_lines.append("")

    df_local = df_features.copy()
    y = target.astype(int)
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

    with gr.Progress(track_tqdm=False) as progress:
        progress(0.05, desc="Lade Datei")
        df_input = pd.read_excel(upload.name)
        progress(0.3, desc="Berechne Scores")
        scores = model.predict_proba(df_input)[:, -1]
        predictions = (scores >= 0.5).astype(int)
        df_out = df_input.copy()
        df_out["fraud_score"] = scores * 100.0
        df_out["prediction"] = predictions
        progress(0.8, desc="Schreibe Ergebnis")
        tmp_dir = Path(tempfile.mkdtemp(prefix="pruefomat_batch_"))
        out_path = tmp_dir / "batch_predictions.xlsx"
        df_out.to_excel(out_path, index=False)
        progress(1.0, desc="Fertig")

    logger.info(
        "ui_batch_prediction_completed",
        rows=len(df_input),
        output=str(out_path),
    )

    return "Batch abgeschlossen.", str(out_path)


def record_feedback(state: Dict[str, Any], row_index: int, feedback: str, user: str, comment: str) -> str:
    ensure_feedback_db()
    predictions = state.get("predictions_full")
    df_features = state.get("df_features")
    if predictions is None or df_features is None:
        return "Keine Predictions verfügbar."

    if row_index < 0 or row_index >= len(predictions):
        return "Index außerhalb gültiger Grenzen."

    row = predictions.iloc[row_index]
    df_row = df_features.iloc[row_index]
    beleg_id = df_row.get("Rechnungsnummer") if isinstance(df_row, pd.Series) else None

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
                float(row["fraud_score"]),
                int(row["prediction"]),
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
            .pf-tooltip {position:relative;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;background:#e0e0e0;color:#333;font-size:12px;cursor:help;}
            .pf-tooltiptext {visibility:hidden;opacity:0;transition:opacity 0.2s ease;position:absolute;z-index:20;background:#1e1e1e;color:#fff;padding:10px;border-radius:6px;width:280px;bottom:130%;left:50%;transform:translateX(-50%);box-shadow:0 4px 12px rgba(0,0,0,0.25);font-weight:400;line-height:1.4;}
            .pf-tooltiptext::after {content:"";position:absolute;top:100%;left:50%;margin-left:-6px;border-width:6px;border-style:solid;border-color:#1e1e1e transparent transparent transparent;}
            .pf-tooltip:hover .pf-tooltiptext {visibility:visible;opacity:1;}
            </style>
            """
        )

        state = gr.State({
            "config": DEFAULT_CONFIG.model_copy(deep=True),
            "config_path": str(DEFAULT_CONFIG_PATH),
            "balance_classes": False,
        })

        with gr.Tabs():
            with gr.Tab("Training & Analyse"):
                with gr.Row():
                    file_input = gr.File(label="Excel-Dateien", file_types=[".xlsx", ".xls"], file_count="multiple")  # type: ignore[arg-type]
                    folder_input = gr.Textbox(label="Ordner (optional)", placeholder="Pfad zu einem Ordner mit Excel-Dateien")
                    config_input = gr.File(label="Config (optional)", file_types=[".yaml", ".yml", ".json"])  # type: ignore[arg-type]
                    sheet_input = gr.Textbox(value="0", label="Sheet (Index oder Name)")
                    target_input = gr.Textbox(value=DEFAULT_CONFIG.data.target_col or "", label="Zielspalte (optional)")
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
                predictions_table = gr.Dataframe(label="Predictions (Test Set, sortiert nach Risiko)", interactive=False)
                predictions_download = gr.File(label="Predictions CSV", interactive=False)
                explain_index = gr.Number(label="Zeilenindex für Erklärung", value=0, precision=0)
                explain_btn = gr.Button("Erklärung anzeigen")
                explain_status = gr.Textbox(label="Erklärungs-Status", interactive=False)
                explain_json = gr.JSON(label="Top-Features (SHAP)")
                explain_download = gr.File(label="Erklärungs-JSON", interactive=False)
                report_btn = gr.Button("Pattern Report generieren")
                report_status = gr.Textbox(label="Report-Status", interactive=False)
                report_preview = gr.Markdown(label="Report Vorschau")
                report_download = gr.File(label="Report Download", interactive=False)
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
                    "GPT-Verfeinerung verwenden",
                    "Aktiviert sprachliche Verfeinerung (z. B. plausiblere Hinweise). Erhöht Laufzeit und API-Kosten."
                ))
                synth_gpt_enable = gr.Checkbox(show_label=False, value=True)
                gr.HTML(_tooltip_label(
                    "GPT-Modell",
                    "Welches OpenAI-Modell für Textplausibilisierung genutzt wird."
                ))
                synth_gpt_model = gr.Dropdown(choices=["gpt-5-mini", "gpt-5", "gpt-4.1-mini"], value="gpt-5-mini", show_label=False)

                gr.HTML(_tooltip_label(
                    "OpenAI API Key",
                    "Nur notwendig, wenn GPT-Verfeinerung aktiv ist. Der Key wird während des Laufs gesetzt und danach entfernt."
                ))
                synth_gpt_key = gr.Textbox(type="password", show_label=False)

                gr.HTML(_tooltip_label(
                    "Debug-Logging aktivieren",
                    "Schreibt detaillierte Synthesizer-Logs (z. B. pro Tabelle). Hilfreich beim Debugging, erzeugt aber mehr Output."
                ))
                synth_debug = gr.Checkbox(show_label=False, value=False)
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
                state,
            ],
        )

        explain_btn.click(
            explain_prediction_action,
            inputs=[state, explain_index],
            outputs=[explain_status, explain_json, explain_download, state],
        )

        report_btn.click(
            generate_pattern_report_action,
            inputs=[state],
            outputs=[report_status, report_preview, report_download, state],
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
