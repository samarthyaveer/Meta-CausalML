from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ValidationReport:
    is_valid: bool
    row_count: int
    column_count: int
    detected_column_types: dict[str, str]
    treatment_unique_values: list[str]
    mapped_treatment_labels: dict[int, str]
    candidate_feature_columns: list[str]
    rejected_columns: dict[str, str]
    missing_summary: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    treatment_col: str
    outcome_col: str
    feature_cols: list[str]
    learner_names: list[str]
    base_model_name: str
    true_cate_col: str | None = None
    propensity_col: str | None = None
    test_size: float = 0.2
    random_state: int = 42
    control_value: Any = 0
    treatment_value: Any = 1
    exclude_cols: list[str] = field(default_factory=list)


@dataclass
class PreparedData:
    raw_full_df: pd.DataFrame
    raw_train_df: pd.DataFrame
    raw_test_df: pd.DataFrame
    X_full: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_full: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    treatment_full: pd.Series
    treatment_train: pd.Series
    treatment_test: pd.Series
    true_cate_full: pd.Series | None
    true_cate_test: pd.Series | None
    propensity_full: pd.Series | None
    propensity_test: pd.Series | None
    processed_feature_names: list[str]
    processed_to_original: dict[str, str]
    preprocessor: Any


@dataclass
class LearnerArtifacts:
    learner_name: str
    learner_label: str
    estimator: Any
    fit_seconds: float
    full_cate: pd.Series
    test_cate: pd.Series
    metrics: dict[str, float]
    gain_curve: pd.DataFrame
    feature_importance: pd.DataFrame
    segment_summary: pd.DataFrame
    top_users: pd.DataFrame


@dataclass
class TrainingResult:
    config: RunConfig
    prepared_data: PreparedData
    validation_report: ValidationReport
    learner_outputs: dict[str, LearnerArtifacts]
    comparison_table: pd.DataFrame
    predictions_full: pd.DataFrame
    predictions_test: pd.DataFrame
    export_metadata: dict[str, Any]
