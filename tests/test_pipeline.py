import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from noshows.data import make_appointments
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_pipeline,
    split_data,
)


def test_data_is_reproducible():
    """Check that the data split is reproducible."""
    pd.testing.assert_frame_equal(make_appointments(), make_appointments())


def test_excluded_features_are_not_included():
    """Check that excluded features are not in the training data."""
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)
    for feature in EXCLUDED_FEATURES:
        assert feature not in X_train.columns
        assert feature not in X_test.columns


def test_split_is_stratified_and_the_right_size():
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)
    assert len(X_train) == 3000
    assert len(X_test) == 1000
    assert list(X_train.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    overall = frame[TARGET].mean()
    assert abs(y_train.mean() - overall) < 0.01
    assert abs(y_test.mean() - overall) < 0.01


def test_pipeline_missing_distance_values():
    """Check that the pipeline can handle missing distance values."""
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)
    model = LogisticRegression(max_iter=1000)
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(y_test)
    row = X_test.iloc[[0]].copy()
    row["distance_km"] = np.nan
    prediction_with_missing_distance = pipeline.predict(row)
    assert prediction_with_missing_distance in [0, 1]
