---
title: Maple Physio No-Show Check
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Will this patient miss the appointment?
---

# Maple Physio no-show check

## What it does

About one booked appointment in four at Maple Physio ends in a no-show. The front desk can phone a patient the day before for $8, and a no-show that nobody phoned about costs the clinic $35. This page lets a receptionist enter a booking, see the chance of a no-show, and get a plain answer: call or don't call.

Space: https://huggingface.co/spaces/Oleg04/midterm-no-shows

Demo recording: [watch the demo](https://humberital-my.sharepoint.com/:v:/g/personal/n10041441_humber_ca/IQBs7uJE1DqmRJoK16Hj7ZWOAUu7nepiC5MWuD6_lZwYcik?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=DTntlV)

## How to run it locally

```bash
conda env create -f environment.yml
conda activate midterm-no-shows
python -m noshows.train     # recreates models/ and reports/
gradio app.py               # or: python app.py
```

## Team

| Member | GitHub | Parts owned |
|---|---|---|
| Oleg Shukhov | OlegShukhovHumber | `noshows/data.py`, `noshows/predictor.py`, `app.py`; repository, merging PRs, Hugging Face Space, final check, submission |
| Ana Corsi | anamatiascorsi | `noshows/pipeline.py`, `noshows/evaluation.py`, `tests/test_pipeline.py`, `tests/test_evaluation.py` |
| Gerardo Ahuatzin | GerardoAhuatzin741 | `noshows/appointment.py`, `noshows/train.py`, `models/`, `reports/`, `tests/conftest.py`, `tests/test_appointment.py` |
| Aditya Sharma | adityalochan | `environment.yml`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.gitignore`, `tests/test_predictor.py`, `tests/test_deployment.py`, `README.md`, final cleanup |

## Decisions

### Columns

A column is only used if the front desk knows it when the appointment is booked.

| Column | Used? | Why |
|---|---|---|
| `lead_days` | Used (number) | Known at booking. Bookings made far ahead are missed more often. |
| `age` | Used (number) | Known at booking. |
| `prior_no_shows` | Used (number) | Known from the patient's history. The strongest single signal. |
| `hour` | Used (number) | Known at booking. Early and late visits are missed more often. |
| `first_visit` | Used (number, 0 or 1) | Known at booking. |
| `distance_km` | Used (number) | Known at booking, but sometimes not recorded (see the gap below). |
| `weekday` | Used (category) | Known at booking. Mondays and Fridays are riskier. |
| `booking` | Used (category) | Known at booking. Online bookings are missed more often. |
| `insurance` | Used (category) | Known at booking. |
| `treatment` | Used (category) | Known at booking. |
| `appointment_id` | **Excluded** | Only a label. It carries no information about the patient, so any pattern the model found in it would be noise. |
| `fee_charged` | **Excluded** | Recorded after the visit, and only ever 1 when the patient did not come. It gives the answer away and does not exist when the booking is made. |
| `no_show` | Target | What the model predicts. |

### Page limits

The page refuses values the model has no basis for judging. Each limit is a little wider than the training data, so ordinary bookings are never refused, but values far outside anything the model has seen are.

| Field | Training data | Page allows | Why |
|---|---|---|---|
| `lead_days` | 0 to 60 | 0 to 120, whole number | Up to about four months ahead. A negative lead time is impossible. |
| `age` | 16 to 92 | 12 to 110, whole number | Covers young teens to the oldest patients. |
| `prior_no_shows` | 0 to 4 | 0 to 30, whole number | Cannot be negative. Very high counts are far outside the data. |
| `hour` | 8 to 17 | 8 to 17, whole number | The clinic's opening hours. There is nothing to learn from outside them. |
| `first_visit` | 0 or 1 | 0 or 1 | A yes/no flag. |
| `distance_km` | 0.5 to 37.2 | 0.1 to 150, or empty | Covers patients travelling from outside the city. Empty is accepted (see below). |
| `weekday` | mon to fri | mon, tue, wed, thu, fri | The clinic is closed at weekends. |
| `booking` | 3 values | online, phone, front_desk | Only the values the model was trained on. |
| `insurance` | 3 values | private, provincial, self_pay | Only the values the model was trained on. |
| `treatment` | 3 values | physio, massage, acupuncture | Only the values the model was trained on. |

**Unseen categories.** The pipeline's one-hot encoder is set to ignore categories it has never seen, so a misspelt value such as `"Mon"` or `"sat"` would not crash. Instead it would quietly score the booking as if it had no weekday at all, and the page would show a confident but wrong answer. That is the worst kind of failure, because nothing looks broken. So `Appointment` refuses any value outside the allowed list, and names the field in the error message.

**The accepted gap.** `distance_km` may be left empty, because about 4% of the training rows have no distance recorded. The pipeline fills it in with the training median, and the page shows a warning that a typical distance was used.

## Model comparison

Five-fold stratified cross-validation on the training data only (3,000 bookings). Each cell is the mean ± the standard deviation across the five folds (from `reports/cv_results.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (most frequent) | 0.738 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.756 ± 0.012 | 0.595 ± 0.067 | 0.215 ± 0.024 | 0.316 ± 0.035 | **0.713 ± 0.022** |
| Logistic regression, `class_weight="balanced"` | 0.669 ± 0.016 | 0.415 ± 0.017 | 0.640 ± 0.027 | 0.503 ± 0.014 | 0.713 ± 0.022 |
| Random forest | 0.748 ± 0.009 | 0.581 ± 0.083 | 0.126 ± 0.029 | 0.207 ± 0.044 | 0.695 ± 0.020 |
| Histogram gradient boosting | 0.730 ± 0.005 | 0.470 ± 0.021 | 0.248 ± 0.020 | 0.325 ± 0.021 | 0.667 ± 0.015 |

**Why accuracy misleads.** About 26% of bookings are no-shows. The Dummy says "everyone comes" and is 73.8% accurate, almost as good as the best model, yet it never flags a single no-show and saves the clinic nothing. The balanced logistic regression has the lowest accuracy of the real models but finds the most no-shows. Recall, F1 and ROC-AUC show whether a model actually finds the no-shows.

**Chosen model: logistic regression.** It has the highest mean ROC-AUC (0.713), and ROC-AUC is the right measure here because the call/no-call decision is made later by the cost threshold, so what matters is how well the model ranks risky bookings above safe ones. The two logistic regressions rank equally well (same ROC-AUC); `class_weight` only shifts the probabilities, and the threshold takes care of that. The tree models did worse on this small, mostly linear table, and logistic regression is also the simplest to explain to the front desk.

**Tuning.** `GridSearchCV` over `C` (0.01, 0.1, 1, 10) and `class_weight` (None, balanced), 8 combinations, scored by ROC-AUC with the same five folds. The best setting was `C=0.1`, `class_weight=None`, with ROC-AUC 0.7131 against 0.7129 untuned. That gain of 0.0002 is far smaller than the fold-to-fold spread of 0.022, so tuning did not really beat the untuned model.

## Result

Test set of 1,000 bookings, scored once, with the threshold chosen from out-of-fold training probabilities (from `models/model_info.json`):

| Metric | Value |
|---|---|
| Threshold | 0.21 |
| Accuracy | 0.625 |
| Precision | 0.394 |
| Recall | 0.805 |
| F1 | 0.529 |
| ROC-AUC | 0.761 |

The threshold is low because a missed no-show ($35) costs over four times as much as a call ($8), so it pays to call anyone with more than about a one-in-five chance. The model flags 535 of the 1,000 bookings. It catches 211 of the 262 no-shows and misses 51, and 324 of the calls go to patients who would have come anyway.

| Policy | Total cost on the test set |
|---|---|
| Call nobody | $9,170 (262 no-shows × $35) |
| Call everybody | $8,000 (1,000 calls × $8) |
| Use the model | **$6,065** (535 calls × $8 + 51 missed × $35) |

The model beats both: it saves $1,935 against calling everybody and $3,105 against calling nobody, about $1.94 and $3.11 per booking. With only 1,000 test bookings these figures are rough. Resampling the test set (2,000 bootstrap samples) puts the saving against calling everybody between about $1,450 and $2,400 (95% range). So the model is clearly cheaper than both policies, but the exact dollar amount should not be quoted too precisely.

**Leakage check.** Honest cross-validated ROC-AUC: 0.713. With `fee_charged` added: 0.923. The larger number cannot be trusted, because `fee_charged` is only known after the visit, so the model is reading the answer rather than predicting it.

## Tests

There are 27 test functions (49 cases once the parametrized ones are expanded). They run without a network and cover:

- **Data and pipeline:** the table is reproducible, the excluded columns are not features, the split is stratified 75/25, and a missing `distance_km` is handled.
- **Cost rule:** the cost function against a hand-worked example, the cases where calling nobody or calling everybody is cheapest, and the threshold choice at both extremes.
- **Appointment:** valid bookings and the empty distance are accepted. Each impossible number and each unseen category is refused, with the field named.
- **Predictor:** the probability is between 0 and 1, a risky booking scores higher than a safe one, the decision is correct at, just below and just above the threshold, the threshold comes from `model_info.json`, and the model file is read once.
- **Deployment:** the installed scikit-learn matches `model_info.json` and `requirements.txt`, Gradio is not in `requirements.txt`, the README's `sdk_version` matches the installed Gradio, and `app_file` exists.

Run the three checks from the repository root with the environment active:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Limitations

- **The data is invented.** The bookings are simulated, so the numbers describe no real clinic or patients.
- **The model cannot see the day itself.** Illness, weather, transport problems or a family emergency are invisible to it.
- **It assumes a call always works.** The cost rule treats every phoned patient as someone who then comes. In reality some will still miss the appointment, so the real saving is smaller.
- **It gives a chance, not a certainty.** Many patients flagged as risky will come anyway, and some marked safe will not.
- **It must not be used to refuse or penalise patients.** It is only meant to decide who gets a reminder call. Using it to deny bookings or charge fees would be unfair, especially as it uses age and insurance type.
