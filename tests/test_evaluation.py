import numpy as np
import pytest

from noshows.evaluation import (
    THERESHOLDS,
    choose_threshold,
    flag,
    policy_costs,
    summary,
    total_cost,
)


def test_total_cost_hand():
    y_true = np.array([1, 0, 1, 1, 0])
    flagged = np.array([1, 1, 0, 0, 0])  # 2 calls, 3 missed= 2*8 + 2*35 = 16 + 70 = 86
    assert total_cost(y_true, flagged, call_cost=8, miss_cost=35) == float(86)


@pytest.mark.parametrize(
    ("y_true", "cheaper", "nobody", "everybody"),
    [
        ([0, 0, 0, 0, 0], "nobody", 0, 40),  # nobody misses: 5 calls cost 5 x 8
        ([1, 1, 1, 1, 1], "everybody", 175, 40),  # all miss: 5 x 35 vs 5 x 8
    ],
)
def test_calling_nobody_or_everybody_can_be_cheapest(
    y_true, cheaper, nobody, everybody
):
    probabilities = np.full(5, 0.5)
    costs = policy_costs(y_true, probabilities, threshold=0.5)
    assert costs["call_nobody"] == nobody
    assert costs["call_everybody"] == everybody
    assert costs["call_" + cheaper] == min(nobody, everybody)


def test_flag_greater_equal():
    assert np.array_equal(flag([0.1, 0.5, 0.9], 0.5), [False, True, True])


def test_threshold_values():
    assert THERESHOLDS[0] == 0.05
    assert THERESHOLDS[-1] == 0.95
    assert len(THERESHOLDS) == 91


def test_summary():
    y_true = np.array([1, 0, 1, 1, 0])
    probabilities = np.array([0.9, 0.2, 0.8, 0.7, 0.1])
    threshold = 0.5
    result = summary(y_true, probabilities, threshold)
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 3}
    assert result["accuracy"] == 1.0
    assert result["roc_auc"] == 1.0


@pytest.mark.parametrize(
    "y_true, call_cost, miss_cost, expected_threshold",
    [
        # calls cost a lot and misses are cheap, so it is best to call nobody
        ([0, 1, 0, 1, 0, 1, 0, 1, 0], 1000, 1, 0.95),
        # misses cost a lot and calls are cheap, so it is best to call everybody
        ([1, 1, 1, 1, 1, 1, 1, 1, 1], 1, 1000, 0.05),
    ],
)
def test_threshold_at_the_extremes(y_true, call_cost, miss_cost, expected_threshold):
    probs = np.linspace(0.06, 0.94, 9)
    best = choose_threshold(y_true, probs, call_cost, miss_cost)
    assert best == expected_threshold
