from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..models import PreparedData, RunConfig


def prepare_data(df: pd.DataFrame, config: RunConfig) -> PreparedData:
    working_df = df.copy().reset_index(drop=True)
    working_df["row_id"] = working_df.index

    treatment_series = working_df[config.treatment_col]
    working_df["treatment_binary"] = (
        treatment_series == config.treatment_value
    ).astype(int)
    working_df["outcome_numeric"] = pd.to_numeric(
        working_df[config.outcome_col], errors="coerce"
    )

    if working_df["outcome_numeric"].isna().any():
        raise ValueError("Outcome column could not be converted cleanly to numeric values.")

    true_cate_series = None
    if config.true_cate_col:
        true_cate_series = pd.to_numeric(
            working_df[config.true_cate_col], errors="coerce"
        )
        if true_cate_series.isna().any():
            raise ValueError("true_cate column must be numeric when provided.")
        working_df["true_cate_numeric"] = true_cate_series

    propensity_series = None
    if config.propensity_col:
        propensity_series = pd.to_numeric(
            working_df[config.propensity_col], errors="coerce"
        )
        if propensity_series.isna().any() or not propensity_series.between(0, 1).all():
            raise ValueError("Propensity column must contain numeric values between 0 and 1.")
        working_df["propensity_numeric"] = propensity_series

    raw_feature_df = working_df[config.feature_cols].copy()
    numeric_features = [
        column for column in config.feature_cols if pd.api.types.is_numeric_dtype(raw_feature_df[column])
    ]
    categorical_features = [
        column for column in config.feature_cols if column not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    split_columns = ["row_id", "treatment_binary", "outcome_numeric", *config.feature_cols]
    if config.true_cate_col:
        split_columns.append("true_cate_numeric")
    if config.propensity_col:
        split_columns.append("propensity_numeric")

    split_df = working_df[split_columns].copy()

    train_df, test_df = train_test_split(
        split_df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=split_df["treatment_binary"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    preprocessor.fit(train_df[config.feature_cols])

    processed_feature_names = list(preprocessor.get_feature_names_out())
    processed_to_original = _build_processed_feature_map(
        processed_feature_names, config.feature_cols
    )

    X_full = _transform_to_dataframe(preprocessor, split_df[config.feature_cols], processed_feature_names)
    X_train = _transform_to_dataframe(preprocessor, train_df[config.feature_cols], processed_feature_names)
    X_test = _transform_to_dataframe(preprocessor, test_df[config.feature_cols], processed_feature_names)

    raw_full_df = working_df.copy()
    raw_train_df = working_df.loc[train_df["row_id"]].reset_index(drop=True)
    raw_test_df = working_df.loc[test_df["row_id"]].reset_index(drop=True)

    return PreparedData(
        raw_full_df=raw_full_df,
        raw_train_df=raw_train_df,
        raw_test_df=raw_test_df,
        X_full=X_full,
        X_train=X_train,
        X_test=X_test,
        y_full=split_df["outcome_numeric"].reset_index(drop=True),
        y_train=train_df["outcome_numeric"],
        y_test=test_df["outcome_numeric"],
        treatment_full=split_df["treatment_binary"].reset_index(drop=True),
        treatment_train=train_df["treatment_binary"],
        treatment_test=test_df["treatment_binary"],
        true_cate_full=split_df["true_cate_numeric"].reset_index(drop=True)
        if config.true_cate_col
        else None,
        true_cate_test=test_df["true_cate_numeric"] if config.true_cate_col else None,
        propensity_full=split_df["propensity_numeric"].reset_index(drop=True)
        if config.propensity_col
        else None,
        propensity_test=test_df["propensity_numeric"] if config.propensity_col else None,
        processed_feature_names=processed_feature_names,
        processed_to_original=processed_to_original,
        preprocessor=preprocessor,
    )


def _transform_to_dataframe(preprocessor, raw_features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    transformed = preprocessor.transform(raw_features)
    return pd.DataFrame(transformed, columns=feature_names, index=raw_features.index)


def _build_processed_feature_map(
    processed_feature_names: list[str], raw_feature_names: list[str]
) -> dict[str, str]:
    sorted_candidates = sorted(raw_feature_names, key=len, reverse=True)
    mapping: dict[str, str] = {}
    for processed_name in processed_feature_names:
        matched_name = processed_name
        for raw_name in sorted_candidates:
            if processed_name == raw_name or processed_name.startswith(f"{raw_name}_"):
                matched_name = raw_name
                break
        mapping[processed_name] = matched_name
    return mapping
