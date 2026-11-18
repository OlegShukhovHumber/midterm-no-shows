"""The cost rule, the decision rule, the threshold search and the metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CALL_COST = 8  # call cost
MISSED_APPOINTMENT_COST = 35  # now_show cost

THERESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)  # 0.01-0.95


def total_cost(y_true, flagged, call_cost=CALL_COST, miss_cost=MISSED_APPOINTMENT_COST):
    """Calculate the total cost of the predictions."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    calls = flagged.sum()
    misses = (y_true & ~flagged).sum()
    return float((calls * call_cost) + (misses * miss_cost))


def flag(probabilities, threshold) -> np.ndarray:
    """Flag the probabilities above or equal to the threshold."""
    return np.asarray(probabilities) >= threshold


def choose_threshold(
    y_true, probabilities, call_cost=CALL_COST, miss_cost=MISSED_APPOINTMENT_COST
) -> float:
    """Choose the threshold that minimizes the total cost."""
    costs = [
        total_cost(y_true, flag(probabilities, t), call_cost, miss_cost)
        for t in THERESHOLDS
    ]
    best_index = int(np.argmin(costs))
    return float(THERESHOLDS[best_index])


def policy_costs(y_true, probabilities, threshold) -> dict:
    """Calculate the costs of call nobody, call everybody, call flagged."""
    costs = {
        "call_nobody": total_cost(y_true, np.zeros(len(y_true), dtype=bool)),
        "call_everybody": total_cost(y_true, np.ones(len(y_true), dtype=bool)),
        "model": total_cost(y_true, flag(probabilities, threshold)),
    }
    return costs


def summary(y_true, probabilities, threshold) -> dict:
    """Return a summary of the metrics."""
    y_true = np.asarray(y_true)
    flagged = flag(probabilities, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[False, True]).ravel()
    return {
        "accuracy": accuracy_score(y_true, flagged),
        "precision": precision_score(y_true, flagged),
        "recall": recall_score(y_true, flagged),
        "f1": f1_score(y_true, flagged),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
