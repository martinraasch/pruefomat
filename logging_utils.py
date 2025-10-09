"""Structured logging helpers for pruefomat."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

import structlog

MASK_TOKEN = "***MASKED***"
SENSITIVE_SUBSTRINGS = ("betrag", "amount", "name", "debitor")


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for JSON logging with the given level."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(level=numeric_level, format="%(message)s", force=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.typing.FilteringBoundLogger:
    """Return a structured logger instance."""

    return structlog.get_logger(name)


def _should_mask(key: str) -> bool:
    key_lower = key.lower()
    return any(substr in key_lower for substr in SENSITIVE_SUBSTRINGS)


def _mask_value(value: Any) -> Any:
    if isinstance(value, dict):
        return mask_sensitive_data(value)
    if isinstance(value, list):
        return [_mask_value(item) for item in value]
    if isinstance(value, (str, int, float)):
        return MASK_TOKEN
    return value


def mask_sensitive_data(data: Dict[str, Any], additional_keywords: Iterable[str] | None = None) -> Dict[str, Any]:
    """Return a copy of *data* with sensitive fields masked."""

    keywords = set(SENSITIVE_SUBSTRINGS)
    if additional_keywords:
        keywords.update(k.lower() for k in additional_keywords)

    masked: Dict[str, Any] = {}
    for key, value in data.items():
        if key is None:
            masked[key] = value
            continue
        if any(substr in key.lower() for substr in keywords):
            masked[key] = _mask_value(value)
        else:
            masked[key] = value
    return masked


__all__ = ["configure_logging", "get_logger", "mask_sensitive_data", "MASK_TOKEN"]
