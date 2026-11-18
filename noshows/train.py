"""Train, compare, tune and save the no-show model.

Run from the repository root, inside the midterm-no-shows environment:

    python -m noshows.train

It recreates models/no_show.joblib, models/model_info.json,
reports/cv_results.csv and reports/metrics.json. Never edit those by hand.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from noshows.data import make_appointments
from noshows.evaluation import choose_threshold, policy_costs, summary
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEED,
    build_pipeline,
    split_data,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SCORING = {
    "accuracy": "accuracy",
    # zero_division=0 so the Dummy (which never predicts a no-show) does not warn
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

BASELINE = "dummy"


def candidates() -> dict:
    """The models compared with cross-validation, by short name."""
    return {
        BASELINE: DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000),
        "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        ),
        "hist_gb": HistGradientBoostingClassifier(random_state=SEED),
    }


PARAM_GRIDS = {
    "logreg": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    },
    "logreg_balanced": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [1, 5, 20],
        "model__class_weight": [None, "balanced"],
    },
    "hist_gb": {
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3, None],
        "model__max_iter": [100, 300],
    },
}


def plain(value):
    """Turn numpy numbers (also inside dicts and lists) into plain Python for JSON."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare_models(X_train, y_train, folds) -> pd.DataFrame:
    """Step 2: mean and std of every metric for every candidate."""
    rows = []
    for name, model in candidates().items():
        scores = cross_validate(
            build_pipeline(model), X_train, y_train, cv=folds, scoring=SCORING
        )
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
        print(
            f"  {name:<16} ROC-AUC {row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Training rows: {len(X_train)}, test rows: {len(X_test)}")

    print("Comparing models with 5-fold cross-validation...")
    cv_results = compare_models(X_train, y_train, folds)
    cv_results.to_csv(REPORTS_DIR / "cv_results.csv", index=False, float_format="%.4f")
    real_models = cv_results[cv_results["model"] != BASELINE]
    best_row = real_models.loc[real_models["roc_auc_mean"].idxmax()]
    chosen = best_row["model"]
    print(f"Chosen model (highest mean CV ROC-AUC): {chosen}")

    search = GridSearchCV(
        build_pipeline(candidates()[chosen]),
        PARAM_GRIDS[chosen],
        scoring="roc_auc",
        cv=folds,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    untuned_mean = float(best_row["roc_auc_mean"])
    untuned_std = float(best_row["roc_auc_std"])
    tuned_score = float(search.best_score_)
    tuning_helped = tuned_score - untuned_mean > untuned_std
    print(
        f"Tuned ROC-AUC {tuned_score:.4f} vs untuned {untuned_mean:.4f} "
        f"± {untuned_std:.4f} -> "
        f"{'beats it by more than one std' if tuning_helped else 'within one std'}"
    )
    tuned = clone(search.best_estimator_)

    oof_probabilities = cross_val_predict(
        tuned, X_train, y_train, cv=folds, method="predict_proba"
    )[:, 1]
    threshold = round(float(choose_threshold(y_train, oof_probabilities)), 2)
    print(f"Chosen threshold: {threshold}")

    leaky_X_train = X_train.assign(fee_charged=frame.loc[X_train.index, "fee_charged"])
    leaky = build_pipeline(
        clone(tuned.named_steps["model"]),
        numeric_features=NUMERIC_FEATURES + ["fee_charged"],
        categorical_features=CATEGORICAL_FEATURES,
    )
    leaky_auc = cross_validate(
        leaky, leaky_X_train, y_train, cv=folds, scoring="roc_auc"
    )["test_score"].mean()
    leakage = {
        "honest_cv_roc_auc": round(tuned_score, 4),
        "with_fee_charged_cv_roc_auc": round(float(leaky_auc), 4),
    }
    print(f"Leakage check: {leakage}")

    tuned.fit(X_train, y_train)
    test_probabilities = tuned.predict_proba(X_test)[:, 1]
    test_results = plain(summary(y_test, test_probabilities, threshold))
    test_results["costs"] = plain(policy_costs(y_test, test_probabilities, threshold))
    print(f"Test results: {test_results}")

    joblib.dump(tuned, MODELS_DIR / "no_show.joblib")
    model_info = {
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "model": chosen,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_params": plain(search.best_params_),
        "test_results": test_results,
        "leakage": leakage,
    }
    metrics = {
        "chosen_model": chosen,
        "threshold": threshold,
        "tuning": {
            "untuned_cv_roc_auc_mean": round(untuned_mean, 4),
            "untuned_cv_roc_auc_std": round(untuned_std, 4),
            "tuned_cv_roc_auc": round(tuned_score, 4),
            "beats_untuned_by_more_than_std": bool(tuning_helped),
            "best_params": plain(search.best_params_),
        },
        "test_results": test_results,
        "leakage": leakage,
    }
    write_json(MODELS_DIR / "model_info.json", model_info)
    write_json(REPORTS_DIR / "metrics.json", metrics)

    costs = test_results["costs"]
    if not costs["model"] < min(costs["call_nobody"], costs["call_everybody"]):
        print("WARNING: the model does not beat both simple policies. Tell the team.")
    print("Saved models/ and reports/.")


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
