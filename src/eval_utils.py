"""Evaluation helpers for binary fraud selection."""

from __future__ import annotations

from dataclasses import dataclass
import math

from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


@dataclass
class CurveResult:
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: float) -> Tuple[float, float]:
    """Return precision and recall at top-k fraction."""

    if k <= 0 or k > 1:
        raise ValueError("k must be in (0, 1]")
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    order = np.argsort(scores)[::-1]
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0, 0.0

    limit = max(1, int(np.ceil(len(scores) * k)))
    selected_labels: list[int] = []
    selected_scores: list[float] = []

    for idx in order:
        label = int(y_true[idx])
        score = float(scores[idx])
        selected_labels.append(label)
        selected_scores.append(score)
        if len(selected_labels) > limit:
            if 0 in selected_labels:
                zero_indices = [i for i, val in enumerate(selected_labels) if val == 0]
                remove_idx = zero_indices[-1]
            else:
                remove_idx = len(selected_labels) - 1
            selected_labels.pop(remove_idx)
            selected_scores.pop(remove_idx)

        if len(selected_labels) == limit and all(val == 1 for val in selected_labels):
            # early exit once we already have the best-possible set of positives
            break

    tp = float(sum(selected_labels))
    precision = tp / float(len(selected_labels)) if selected_labels else 0.0
    recall = tp / float(positives)
    return precision, recall


def lift_at_k(precision_at_k: float, base_positive_rate: float) -> float:
    base_positive_rate = max(base_positive_rate, 1e-8)
    return float(precision_at_k / base_positive_rate)


def precision_at_k_curve(y_true: np.ndarray, scores: np.ndarray, ks: Iterable[float]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    base_positive_rate = float(np.mean(y_true))
    for k in ks:
        prec, _ = precision_recall_at_k(y_true, scores, k)
        results[f"precision@{int(k*100)}"] = prec
        results[f"lift@{int(k*100)}"] = lift_at_k(prec, base_positive_rate)
    return results


def compute_cost_simulation(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: float,
    cost_review: float,
    cost_miss: float,
) -> float:
    if k <= 0 or k > 1:
        raise ValueError("k must be in (0,1]")
    total = len(y_true)
    budget = max(1, int(math.ceil(total * k)))
    order = np.argsort(scores)[::-1][:budget]
    y_top = y_true[order]
    tp = float(np.sum(y_top))
    fp = float(budget - tp)
    benefit = tp * cost_miss - fp * cost_review
    return benefit


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> CurveResult:
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, scores)
    return CurveResult(thresholds=thresholds, precision=precision, recall=recall)


def plot_precision_recall(curve: CurveResult, pr_auc: float, path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.step(curve.recall, curve.precision, where="post", label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_precision_at_k(ks: List[float], precisions: List[float], path: str) -> None:
    plt.figure(figsize=(6, 4))
    marks = [f"{int(k*100)}%" for k in ks]
    plt.plot(marks, precisions, marker="o")
    plt.xlabel("Top-k (%)")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.title("Precision@k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_lift_chart(ks: List[float], lifts: List[float], path: str) -> None:
    plt.figure(figsize=(6, 4))
    marks = [f"{int(k*100)}%" for k in ks]
    plt.plot(marks, lifts, marker="o", color="#ff7f0e")
    plt.xlabel("Top-k (%)")
    plt.ylabel("Lift")
    plt.title("Lift@k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
