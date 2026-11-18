"""Feature lists, the train/test split and the pipeline builder."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED=42
TARGET = 'no_show'

NUMERIC_FEATURES= [
    'lead_days',
    'age',
    'prior_no_shows',
    'hour',
    'first_visit',
    'distance_km',
]

CATEGORICAL_FEATURES=['weekday','booking','insurance','treatment']

EXCLUDED_FEATURES=['appointment_id','fee_charged','no_show']

def split_data(frame: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into train and test sets (75/25)."""
    X = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    return X_train, X_test, y_train, y_test

def build_pipeline(model, numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES) -> Pipeline:
    """Build a pipeline for preprocessing and modeling."""
    numeric_transformer = Pipeline(
        [
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, list(numeric_features)),
            ('cat', categorical_transformer, list(categorical_features))
        ]
    )

    pipeline = Pipeline([('preprocessor', preprocessor),
                               ('model', model)])
    return pipeline