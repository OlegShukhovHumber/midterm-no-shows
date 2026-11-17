"""Turn one booking into a chance of a no-show and an action for the front desk."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from noshows.appointment import Appointment

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "no_show.joblib"
INFO_PATH = MODELS_DIR / "model_info.json"


@dataclass(frozen=True)
class Prediction:
    """The chance of a no-show, the threshold used and the decision taken."""

    probability: float
    threshold: float
    call: bool

    @property
    def action(self) -> str:
        """A plain instruction for the receptionist."""
        if self.call:
            return "Call the patient the day before to confirm."
        return "No call needed."


class NoShowPredictor:
    """Load the saved pipeline once and score bookings against its threshold."""

    def __init__(
        self, model_path: Path | str = MODEL_PATH, info_path: Path | str = INFO_PATH
    ) -> None:
        self._pipeline = joblib.load(model_path)
        self._info = json.loads(Path(info_path).read_text(encoding="utf-8"))
        if "threshold" not in self._info:
            raise ValueError(f"{info_path} has no 'threshold'.")
        threshold = float(self._info["threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")
        self._threshold = threshold
        self._positive_column = list(self._pipeline.classes_).index(1)

    @property
    def threshold(self) -> float:
        """The threshold read from model_info.json."""
        return self._threshold

    def probability(self, appointment: Appointment) -> float:
        """Return the chance, from 0 to 1, that this booking ends in a no-show."""
        if not isinstance(appointment, Appointment):
            raise TypeError(
                f"Expected an Appointment, got {type(appointment).__name__}."
            )
        row = appointment.to_frame()
        columns = getattr(self._pipeline, "feature_names_in_", None)
        if columns is not None:
            missing = [name for name in columns if name not in row.columns]
            if missing:
                raise ValueError(f"The booking is missing columns: {missing}.")
            row = row[list(columns)]
        return float(self._pipeline.predict_proba(row)[0, self._positive_column])

    def decide(self, probability: float) -> bool:
        """Call when the chance of a no-show reaches the threshold."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be between 0 and 1, got {probability}.")
        return probability >= self._threshold

    def predict(self, appointment: Appointment) -> Prediction:
        """Return the probability and the decision together."""
        probability = self.probability(appointment)
        return Prediction(
            probability=probability,
            threshold=self._threshold,
            call=self.decide(probability),
        )

    def __repr__(self) -> str:
        model = self._info.get("model", type(self._pipeline[-1]).__name__)
        version = self._info.get("sklearn_version", "unknown")
        return (
            f"NoShowPredictor(model={model!r}, threshold={self._threshold:.2f}, "
            f"sklearn={version!r})"
        )
