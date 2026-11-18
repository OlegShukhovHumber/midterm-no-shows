"""One booking as the front desk enters it, checked before the model sees it."""

import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd

from noshows.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

LIMITS = {
    "lead_days": (0, 120),  
    "age": (12, 110),  
    "prior_no_shows": (0, 30),  
    "hour": (8, 17),  
    "distance_km": (0.1, 150),  
}

CHOICES = {
    "weekday": ("mon", "tue", "wed", "thu", "fri"),
    "booking": ("online", "phone", "front_desk"),
    "insurance": ("private", "provincial", "self_pay"),
    "treatment": ("physio", "massage", "acupuncture"),
    "first_visit": (0, 1),
}

WHOLE_NUMBER_FIELDS = ("lead_days", "age", "prior_no_shows", "hour")
TEXT_FIELDS = ("weekday", "booking", "insurance", "treatment")

def _finite_number(field: str, value) -> float:
    """Return value as a float, or raise if it is missing, a bool, NaN or infinite."""
    if value is None:
        raise ValueError(f"{field} is required.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a number.")
    return number

def _whole_number(field: str, value) -> int:
    """Accept 5 or 5.0, refuse 5.5."""
    number = _finite_number(field, value)
    if not number.is_integer():
        raise ValueError(f"{field} must be a whole number.")
    return int(number)

def _check_range(field: str, number: float) -> None:
    low, high = LIMITS[field]
    if not low <= number <= high:
        raise ValueError(f"{field} must be between {low} and {high}.")

def _one_of(field: str, value, allowed: tuple) -> None:
    if value not in allowed:
        options = ", ".join(str(option) for option in allowed)
        raise ValueError(f"{field} must be one of: {options}.")

@dataclass(frozen=True)
class Appointment:
    
    lead_days: int
    age: int
    prior_no_shows: int
    hour: int
    weekday: str
    booking: str
    insurance: str
    treatment: str
    first_visit: int
    distance_km: float | None = None

    def __post_init__(self) -> None:
        for field in WHOLE_NUMBER_FIELDS:
            number = _whole_number(field, getattr(self, field))
            _check_range(field, number)
            object.__setattr__(self, field, number)

        for field in TEXT_FIELDS:
            value = getattr(self, field)
            if value is None:
                raise ValueError(f"{field} is required.")
            _one_of(field, value, CHOICES[field])

        first_visit = _whole_number("first_visit", self.first_visit)
        _one_of("first_visit", first_visit, CHOICES["first_visit"])
        object.__setattr__(self, "first_visit", first_visit)

        if self.distance_km is not None:
            distance = _finite_number("distance_km", self.distance_km)
            _check_range("distance_km", distance)
            object.__setattr__(self, "distance_km", distance)

    def to_frame(self) -> pd.DataFrame:
        """One row with the training columns; an empty distance becomes NaN."""
        row = {name: getattr(self, name) for name in NUMERIC_FEATURES}
        row.update({name: getattr(self, name) for name in CATEGORICAL_FEATURES})
        if row["distance_km"] is None:
            row["distance_km"] = np.nan
        frame = pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        frame["distance_km"] = frame["distance_km"].astype(float)
        return frame
