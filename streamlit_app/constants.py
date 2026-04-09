from __future__ import annotations

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_LEARNER = "T"
DEFAULT_BASE_MODEL = "random_forest"

LEARNER_OPTIONS = {
    "S": "S-Learner",
    "T": "T-Learner",
    "X": "X-Learner",
    "R": "R-Learner",
}

BASE_MODEL_OPTIONS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "linear_regression": "Linear Regression",
}

SESSION_DEFAULTS = {
    "raw_dataset": None,
    "dataset_source": None,
    "dataset_file_hash": None,
    "column_selection": None,
    "validation_report": None,
    "training_result": None,
}

PREFERRED_COLUMN_NAMES = {
    "treatment": ["treatment", "w", "t", "treatment_group", "treatment_flag"],
    "outcome": ["outcome", "y", "target", "response"],
    "true_cate": ["true_cate", "tau", "treatment_effect", "true_effect"],
    "propensity": ["propensity", "p", "ps", "p_score", "propensity_score"],
}
