import numpy as np
import pandas as pd

from src.utils import align_by_length, ensure_1d, ensure_scalar


def test_ensure_scalar_with_numpy_array():
    assert ensure_scalar(np.array([42])) == 42
    assert ensure_scalar(np.array([])) is None
    assert ensure_scalar(np.array(7)) == 7


def test_ensure_scalar_with_series():
    series = pd.Series(["value"])
    assert ensure_scalar(series) == "value"

    multi = pd.Series([1, 2, 3])
    assert ensure_scalar(multi) == 1

    empty = pd.Series([], dtype=float)
    assert ensure_scalar(empty) is None


def test_ensure_1d_and_align_length():
    values = np.array([[0.1, -0.2, 0.3]])
    flattened = ensure_1d(values)
    assert flattened.ndim == 1
    assert list(flattened) == [0.1, -0.2, 0.3]

    names = ["f1", "f2"]
    aligned_names, aligned_values = align_by_length(names, flattened)
    assert aligned_names == ["f1", "f2"]
    assert list(aligned_values) == [0.1, -0.2]
