import gradio as gr

from noshows.appointment import CHOICES, LIMITS, Appointment
from noshows.predictor import NoShowPredictor

try:  # present only on Hugging Face ZeroGPU hardware
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU
    def keep_zerogpu_happy() -> None:
        """ZeroGPU will not start without one GPU function. This one does nothing."""


predictor = NoShowPredictor()


def number_label(field: str, text: str, optional: bool = False) -> str:
    """Write the allowed range into the label, straight from Appointment's limits."""
    low, high = LIMITS[field]
    empty = ", may be empty" if optional else ""
    return f"{text} ({low} to {high}{empty})"


def score(
    lead_days,
    age,
    prior_no_shows,
    hour,
    weekday,
    booking,
    insurance,
    treatment,
    first_visit,
    distance_km,
):
    try:
        appointment = Appointment(
            lead_days=lead_days,
            age=age,
            prior_no_shows=prior_no_shows,
            hour=hour,
            weekday=weekday,
            booking=booking,
            insurance=insurance,
            treatment=treatment,
            first_visit=first_visit,
            distance_km=distance_km,
        )
    except ValueError as error:
        raise gr.Error(str(error)) from error
    if appointment.distance_km is None:
        gr.Warning(
            "No distance given, so the model used a typical distance. "
            "The chance may be too low for a patient who lives far away."
        )
    result = predictor.predict(appointment)
    chance = result.probability
    return {"No-show": chance, "Comes": 1 - chance}, result.action


with gr.Blocks(title="Maple Physio No-Show Check") as demo:
    gr.Markdown(
        "# Maple Physio No-Show Check\n"
        "Enter the booking as it is made. The page shows the chance that the "
        "patient will not come and whether to phone them the day before. "
        f"The clinic calls when the chance reaches {predictor.threshold:.0%}."
    )
    with gr.Row():
        with gr.Column():
            lead_in = gr.Number(
                label=number_label("lead_days", "Days between booking and visit"),
                value=7,
                precision=0,
            )
            age_in = gr.Number(
                label=number_label("age", "Patient age"), value=46, precision=0
            )
            prior_in = gr.Number(
                label=number_label("prior_no_shows", "Earlier missed appointments"),
                value=0,
                precision=0,
            )
            hour_in = gr.Number(
                label=number_label("hour", "Hour the visit starts"),
                value=10,
                precision=0,
            )
            distance_in = gr.Number(
                label=number_label(
                    "distance_km", "Distance from home (km)", optional=True
                ),
                value=5.0,
            )
            weekday_in = gr.Dropdown(
                list(CHOICES["weekday"]), value="tue", label="Weekday"
            )
            booking_in = gr.Dropdown(
                list(CHOICES["booking"]), value="phone", label="Booked by"
            )
            insurance_in = gr.Dropdown(
                list(CHOICES["insurance"]), value="private", label="Insurance"
            )
            treatment_in = gr.Dropdown(
                list(CHOICES["treatment"]), value="physio", label="Treatment"
            )
            first_in = gr.Radio(
                [("Yes" if value else "No", value) for value in CHOICES["first_visit"]],
                value=0,
                label="First visit to the clinic",
            )
            button = gr.Button("Check booking", variant="primary")
        with gr.Column():
            chances_out = gr.Label(label="Chance of each outcome")
            action_out = gr.Textbox(label="What to do")
    inputs = [
        lead_in,
        age_in,
        prior_in,
        hour_in,
        weekday_in,
        booking_in,
        insurance_in,
        treatment_in,
        first_in,
        distance_in,
    ]
    gr.Examples(
        examples=[
            [2, 70, 0, 11, "wed", "front_desk", "provincial", "acupuncture", 0, 3.0],
            [35, 24, 3, 8, "mon", "online", "self_pay", "physio", 1, 25.0],
            [10, 45, 1, 13, "thu", "phone", "private", "massage", 0, None],
        ],
        inputs=inputs,
        label="Try a low-risk, a high-risk and a no-distance booking",
    )
    button.click(score, inputs=inputs, outputs=[chances_out, action_out])

if __name__ == "__main__":
    demo.launch()
