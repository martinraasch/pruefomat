import numpy as np
import pytest

from src.eval_utils import (
    compute_cost_simulation,
    lift_at_k,
    plot_lift_chart,
    plot_precision_at_k,
    plot_precision_recall,
    precision_at_k_curve,
    precision_recall_at_k,
    pr_curve,
)


def test_precision_recall_at_k_validates_inputs():
    y_true = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.4, 0.8, 0.1])

    precision, recall = precision_recall_at_k(y_true, scores, 0.5)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)

    precision_small, recall_small = precision_recall_at_k(y_true, scores, 0.25)
    assert precision_small == pytest.approx(1.0)
    assert recall_small == pytest.approx(0.5)

    with pytest.raises(ValueError):
        precision_recall_at_k(y_true, scores, 0)
    with pytest.raises(ValueError):
        precision_recall_at_k(y_true, scores, 1.5)
    with pytest.raises(ValueError):
        precision_recall_at_k(np.array([1, 0]), np.array([0.2]), 0.5)


def test_precision_at_k_curve_and_lift():
    y_true = np.array([1, 0, 1, 0, 1])
    scores = np.linspace(0.9, 0.1, 5)
    ks = [0.2, 0.4, 0.6]

    metrics = precision_at_k_curve(y_true, scores, ks)
    assert metrics["precision@20"] == pytest.approx(1.0)
    assert metrics["lift@40"] == pytest.approx(
        lift_at_k(metrics["precision@40"], base_positive_rate=0.6)
    )


def test_pr_curve_and_cost(tmp_path):
    y_true = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.5, 0.8, 0.4, 0.2, 0.7])

    curve = pr_curve(y_true, scores)
    assert len(curve.thresholds) == len(curve.precision) - 1

    benefit = compute_cost_simulation(y_true, scores, k=0.5, cost_review=10, cost_miss=100)
    assert benefit == pytest.approx(300.0)

    loss = compute_cost_simulation(y_true, scores, k=0.67, cost_review=15, cost_miss=50)
    assert loss == pytest.approx(120.0)

    pr_path = tmp_path / "pr.png"
    pk_path = tmp_path / "pk.png"
    lift_path = tmp_path / "lift.png"

    plot_precision_recall(curve, pr_auc=0.85, path=str(pr_path))
    plot_precision_at_k([0.2, 0.4, 0.6], [0.9, 0.8, 0.7], path=str(pk_path))
    plot_lift_chart([0.2, 0.4, 0.6], [1.5, 1.2, 1.0], path=str(lift_path))

    assert pr_path.exists()
    assert pk_path.exists()
    assert lift_path.exists()
