from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from causalml.inference.meta.rlearner import BaseRRegressor
from causalml.inference.meta.slearner import BaseSRegressor
from causalml.inference.meta.tlearner import BaseTRegressor
from causalml.inference.meta.xlearner import BaseXRegressor

from ..constants import BASE_MODEL_OPTIONS, LEARNER_OPTIONS


def build_base_model(base_model_name: str, random_state: int):
    if base_model_name == "linear_regression":
        return LinearRegression()
    if base_model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        )
    if base_model_name == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported base model: {base_model_name}")


def build_supported_learner(learner_name: str, base_model_name: str, random_state: int):
    base_model = build_base_model(base_model_name, random_state=random_state)

    if learner_name == "S":
        return BaseSRegressor(learner=base_model)
    if learner_name == "T":
        return BaseTRegressor(learner=base_model)
    if learner_name == "X":
        return BaseXRegressor(learner=base_model)
    if learner_name == "R":
        return BaseRRegressor(learner=base_model, random_state=random_state)

    raise ValueError(f"Unsupported learner: {learner_name}")


def learner_label(learner_name: str) -> str:
    return LEARNER_OPTIONS[learner_name]


def base_model_label(base_model_name: str) -> str:
    return BASE_MODEL_OPTIONS[base_model_name]
