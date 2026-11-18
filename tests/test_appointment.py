"""Tests for Appointment: what the page may send and what it must refuse."""

import dataclasses
import math

import pytest

from noshows.appointment import Appointment
from noshows.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

def test_valid_booking_gives_one_row_with_training_columns(valid_booking):
    frame = Appointment(**valid_booking).to_frame()
    assert frame.shape == (1, len(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
    assert list(frame.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES

def test_empty_distance_becomes_nan(valid_booking):
    valid_booking["distance_km"] = None
    appointment = Appointment(**valid_booking)
    frame = appointment.to_frame()
    assert appointment.distance_km is None
    assert frame["distance_km"].dtype == float
    assert math.isnan(frame.loc[0, "distance_km"])

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("age", 11),
        ("age", 111),
        ("prior_no_shows", -1),
        ("prior_no_shows", 31),
        ("hour", 7),
        ("hour", 18),
        ("hour", 9.5),
        ("first_visit", 2),
        ("distance_km", 0.05),
        ("distance_km", 151),
        ("lead_days", None),
    ],
)
def test_impossible_numbers_are_refused(valid_booking, field, value):
    valid_booking[field] = value
    with pytest.raises(ValueError, match=field):
        Appointment(**valid_booking)

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weekday", "sat"),
        ("weekday", "Mon"),
        ("booking", "email"),
        ("insurance", "none"),
        ("treatment", "yoga"),
    ],
)
def test_unseen_categories_are_refused(valid_booking, field, value):
    valid_booking[field] = value
    with pytest.raises(ValueError, match=field):
        Appointment(**valid_booking)

def test_booking_cannot_change_after_creation(valid_booking):
    appointment = Appointment(**valid_booking)
    with pytest.raises(dataclasses.FrozenInstanceError):
        appointment.age = 30

@pytest.mark.parametrize(
    ("field", "value"),
    [("age", True), ("age", float("nan")), ("distance_km", float("inf"))],
)
def test_bool_nan_and_infinity_are_refused(valid_booking, field, value):
    valid_booking[field] = value
    with pytest.raises(ValueError, match=field):
        Appointment(**valid_booking)

def test_whole_number_sent_as_float_is_accepted(valid_booking):
    # The page's number boxes may send 5.0 instead of 5.
    valid_booking["lead_days"] = 5.0
    appointment = Appointment(**valid_booking)
    assert appointment.lead_days == 5
    assert isinstance(appointment.lead_days, int)
