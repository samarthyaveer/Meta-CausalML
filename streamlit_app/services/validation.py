from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..constants import PREFERRED_COLUMN_NAMES
from ..models import ValidationReport


def guess_default_column(columns: Iterable[str], column_role: str) -> str | None:
    columns_lower = {column.lower(): column for column in columns}
    for candidate in PREFERRED_COLUMN_NAMES[column_role]:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]
    return None


def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    detected = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            detected[column] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            detected[column] = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            detected[column] = "boolean"
        elif pd.api.types.is_string_dtype(series) or isinstance(
            series.dtype, pd.CategoricalDtype
        ) or pd.api.types.is_object_dtype(series):
            sample_values = series.dropna().head(5).tolist()
            if any(isinstance(value, (dict, list, tuple, set)) for value in sample_values):
                detected[column] = "nested"
            else:
                detected[column] = "categorical"
        else:
            detected[column] = "unsupported"
    return detected


def validate_dataset_selection(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    true_cate_col: str | None = None,
    propensity_col: str | None = None,
    exclude_cols: list[str] | None = None,
    control_value=None,
    treatment_value=None,
) -> ValidationReport:
    exclude_cols = exclude_cols or []
    detected_types = detect_column_types(df)
    missing_summary = df.isna().sum().astype(int).to_dict()
    warnings: list[str] = []
    errors: list[str] = []

    for required_col in [treatment_col, outcome_col]:
        if required_col not in df.columns:
            errors.append(f"Required column '{required_col}' is missing from the dataset.")

    if errors:
        return ValidationReport(
            is_valid=False,
            row_count=len(df),
            column_count=len(df.columns),
            detected_column_types=detected_types,
            treatment_unique_values=[],
            mapped_treatment_labels={},
            candidate_feature_columns=[],
            rejected_columns={},
            missing_summary=missing_summary,
            warnings=warnings,
            errors=errors,
        )

    if detected_types.get(outcome_col) != "numeric":
        errors.append("Outcome column must be numeric.")

    treatment_values = df[treatment_col].dropna().unique().tolist()
    if len(treatment_values) != 2:
        errors.append(
            "Treatment column must contain exactly two unique values for v1 of the app."
        )

    if control_value is not None and treatment_value is not None and control_value == treatment_value:
        errors.append("Control label and treatment label must be different.")

    mapped_labels = {}
    if len(treatment_values) == 2:
        if control_value is None or treatment_value is None:
            ordered_values = sorted(treatment_values, key=lambda value: str(value))
            control_value = ordered_values[0]
            treatment_value = ordered_values[1]
        mapped_labels = {0: str(control_value), 1: str(treatment_value)}

    rejected_columns: dict[str, str] = {}
    feature_candidates: list[str] = []
    reserved = {treatment_col, outcome_col, *exclude_cols}
    if true_cate_col:
        reserved.add(true_cate_col)
    if propensity_col:
        reserved.add(propensity_col)

    for column in df.columns:
        if column in reserved:
            continue

        column_type = detected_types[column]
        if column_type == "numeric":
            feature_candidates.append(column)
        elif column_type == "categorical":
            unique_count = int(df[column].nunique(dropna=True))
            if unique_count <= 25:
                feature_candidates.append(column)
            else:
                rejected_columns[column] = (
                    "Too many unique categories for v1. Please pre-process this column outside the app."
                )
        elif column_type == "datetime":
            rejected_columns[column] = (
                "Datetime columns are not supported in v1. Convert them to numeric features first."
            )
        elif column_type == "nested":
            rejected_columns[column] = (
                "Nested data is not supported in v1. Flatten this column before upload."
            )
        else:
            rejected_columns[column] = (
                "Unsupported feature type for v1. Please pre-process this column outside the app."
            )

    if not feature_candidates:
        errors.append("No usable feature columns remain after validation.")

    if true_cate_col:
        if true_cate_col not in df.columns:
            errors.append("Selected true_cate column is missing.")
        elif detected_types.get(true_cate_col) != "numeric":
            errors.append("true_cate column must be numeric when provided.")

    if propensity_col:
        if propensity_col not in df.columns:
            errors.append("Selected propensity column is missing.")
        elif detected_types.get(propensity_col) != "numeric":
            errors.append("Propensity column must be numeric when provided.")
        else:
            propensity_series = pd.to_numeric(df[propensity_col], errors="coerce")
            if propensity_series.isna().any() or not propensity_series.between(0, 1).all():
                errors.append("Propensity values must be numeric and stay between 0 and 1.")

    if rejected_columns:
        warnings.append(
            "Some columns were excluded because they are too complex for the v1 educational app."
        )

    return ValidationReport(
        is_valid=not errors,
        row_count=len(df),
        column_count=len(df.columns),
        detected_column_types=detected_types,
        treatment_unique_values=[str(value) for value in treatment_values],
        mapped_treatment_labels=mapped_labels,
        candidate_feature_columns=feature_candidates,
        rejected_columns=rejected_columns,
        missing_summary=missing_summary,
        warnings=warnings,
        errors=errors,
    )
