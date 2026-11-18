"""Bookings shared by the tests. Each fixture returns a fresh copy."""

import pytest

VALID_BOOKING = dict(
    lead_days=7,
    age=46,
    prior_no_shows=0,
    hour=10,
    weekday="tue",
    booking="phone",
    insurance="private",
    treatment="physio",
    first_visit=0,
    distance_km=5.0,
)

SAFE_BOOKING = dict(
    lead_days=1,
    age=75,
    prior_no_shows=0,
    hour=12,
    weekday="wed",
    booking="front_desk",
    insurance="provincial",
    treatment="acupuncture",
    first_visit=0,
    distance_km=1.0,
)

RISKY_BOOKING = dict(
    lead_days=50,
    age=20,
    prior_no_shows=5,
    hour=8,
    weekday="mon",
    booking="online",
    insurance="self_pay",
    treatment="physio",
    first_visit=1,
    distance_km=40.0,
)

@pytest.fixture
def valid_booking():
    return dict(VALID_BOOKING)


@pytest.fixture
def safe_booking():
    return dict(SAFE_BOOKING)


@pytest.fixture
def risky_booking():
    return dict(RISKY_BOOKING)
