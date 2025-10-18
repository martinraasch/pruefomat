"""Helpers to harmonise pandas/NumPy outputs with scalar-oriented logic."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


def ensure_scalar(value: object) -> object:
    """Return a plain scalar for pandas/NumPy containers.

    Empty containers yield ``None`` to make downstream null checks simple.
    """

    if isinstance(value, (np.generic,)):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.ndim == 0:
            return value.item()
        return ensure_scalar(value.flat[0])

    if isinstance(value, pd.Series):
        if value.empty:
            return None
        try:
            return value.item()
        except ValueError:
            return ensure_scalar(value.iloc[0])

    return value


def ensure_1d(values: object) -> np.ndarray:
    """Return a 1-D NumPy array, squeezing higher dimensions and keeping first sample."""

    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim > 1:
        # treat first axis as sample dimension and flatten features
        first_axis = array.shape[0]
        array = array.reshape(first_axis, -1)[0]
    return array.ravel()


def align_by_length(names: Sequence[str], values: Iterable[float]) -> Tuple[Sequence[str], np.ndarray]:
    """Truncate both sequences to their shared minimal length."""

    values_array = ensure_1d(list(values))
    min_len = min(len(names), len(values_array))
    return names[:min_len], values_array[:min_len]
