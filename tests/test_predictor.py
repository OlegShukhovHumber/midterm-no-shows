"""Tests for NoShowPredictor: the probability, the decision and loading once."""

import json
from pathlib import Path

import joblib
import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def predictor():
    return NoShowPredictor()


def test_probability_is_between_0_and_1(predictor, valid_booking):
    probability = predictor.probability(Appointment(**valid_booking))
    assert 0.0 <= probability <= 1.0


def test_risky_booking_scores_higher_than_safe_booking(
    predictor, risky_booking, safe_booking
):
    risky = predictor.probability(Appointment(**risky_booking))
    safe = predictor.probability(Appointment(**safe_booking))
    assert risky > safe


@pytest.mark.parametrize(
    ("offset", "expected_call"),
    [(0.0, True), (-0.01, False), (0.01, True)],
    ids=["at_threshold", "just_below", "just_above"],
)
def test_decision_around_the_threshold(predictor, offset, expected_call):
    # Uses the saved threshold, never a typed-in number.
    assert predictor.decide(predictor.threshold + offset) is expected_call


def test_threshold_comes_from_model_info(predictor):
    info = json.loads((ROOT / "models" / "model_info.json").read_text("utf-8"))
    assert predictor.threshold == info["threshold"]


def test_model_file_is_read_once(monkeypatch, valid_booking):
    real_load = joblib.load
    calls = []

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)
    predictor = NoShowPredictor()
    appointment = Appointment(**valid_booking)
    for _ in range(3):
        predictor.predict(appointment)
    assert len(calls) == 1
