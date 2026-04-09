from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..models import PreparedData


def build_feature_importance(
    prepared_data: PreparedData,
    predicted_cate: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    surrogate = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    surrogate.fit(prepared_data.X_full, predicted_cate)

    feature_importance = pd.DataFrame(
        {
            "processed_feature": prepared_data.processed_feature_names,
            "original_feature": [
                prepared_data.processed_to_original[name]
                for name in prepared_data.processed_feature_names
            ],
            "importance": surrogate.feature_importances_,
        }
    )

    aggregated = (
        feature_importance.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return aggregated


def build_segment_summary(df: pd.DataFrame, predicted_cate: pd.Series) -> pd.DataFrame:
    result_df = df.copy().reset_index(drop=True)
    result_df["predicted_cate"] = np.asarray(predicted_cate, dtype=float)

    upper_cutoff = result_df["predicted_cate"].quantile(0.75)
    lower_cutoff = result_df["predicted_cate"].quantile(0.25)

    def assign_segment(value: float) -> str:
        if value >= upper_cutoff:
            return "Top 25%"
        if value <= lower_cutoff:
            return "Bottom 25%"
        return "Middle 50%"

    result_df["segment"] = result_df["predicted_cate"].apply(assign_segment)
    agg_map = {
        "predicted_cate": "mean",
        "outcome_numeric": "mean",
        "row_id": "count",
    }
    rename_map = {
        "predicted_cate": "average_predicted_cate",
        "outcome_numeric": "average_outcome",
        "row_id": "user_count",
    }
    if "true_cate_numeric" in result_df.columns:
        agg_map["true_cate_numeric"] = "mean"
        rename_map["true_cate_numeric"] = "average_true_cate"

    summary = (
        result_df.groupby("segment", as_index=False)
        .agg(agg_map)
        .rename(columns=rename_map)
    )

    order = pd.Categorical(summary["segment"], ["Top 25%", "Middle 50%", "Bottom 25%"], ordered=True)
    summary = summary.assign(segment_order=order).sort_values("segment_order").drop(columns=["segment_order"])
    return summary.reset_index(drop=True)


def build_top_users(df: pd.DataFrame, predicted_cate: pd.Series, top_n: int = 25) -> pd.DataFrame:
    top_df = df.copy().reset_index(drop=True)
    top_df["predicted_cate"] = np.asarray(predicted_cate, dtype=float)

    columns = ["row_id", "predicted_cate", "treatment_binary", "outcome_numeric"]
    if "true_cate_numeric" in top_df.columns:
        columns.append("true_cate_numeric")

    return (
        top_df[columns]
        .sort_values("predicted_cate", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def build_row_explanation(
    df: pd.DataFrame,
    predicted_cate: pd.Series,
    feature_importance: pd.DataFrame,
    feature_cols: list[str],
    row_id: int,
) -> dict[str, object]:
    view_df = df.copy().reset_index(drop=True)
    view_df["predicted_cate"] = np.asarray(predicted_cate, dtype=float)

    if row_id not in view_df["row_id"].values:
        raise ValueError("Selected row_id is not available in the current result set.")

    row = view_df.loc[view_df["row_id"] == row_id].iloc[0]
    upper_cutoff = view_df["predicted_cate"].quantile(0.75)
    lower_cutoff = view_df["predicted_cate"].quantile(0.25)

    if row["predicted_cate"] >= upper_cutoff:
        segment = "Top 25%"
    elif row["predicted_cate"] <= lower_cutoff:
        segment = "Bottom 25%"
    else:
        segment = "Middle 50%"

    top_features = feature_importance["original_feature"].head(3).tolist()
    reasons = []

    for feature in top_features:
        series = view_df[feature]
        value = row[feature]
        if pd.api.types.is_numeric_dtype(series):
            median = series.median()
            direction = "higher" if value >= median else "lower"
            reasons.append(
                f"{feature} is {direction} than the dataset median ({value:.3f} vs {median:.3f})."
            )
        else:
            mode_value = series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else "missing"
            reasons.append(
                f"{feature} is '{value}', while the most common value is '{mode_value}'."
            )

    explanation_text = (
        f"In simple words, this row is predicted to gain about {row['predicted_cate']:.3f} outcome points from treatment. "
        f"It falls into the {segment} uplift segment. "
        f"The strongest global signals behind this score are {', '.join(top_features)}. "
        f"{' '.join(reasons)} "
        f"This explanation uses a simple surrogate feature-importance model, so it is meant to teach the pattern rather than prove a local causal fact."
    )

    return {
        "row_id": int(row_id),
        "segment": segment,
        "predicted_cate": float(row["predicted_cate"]),
        "explanation_text": explanation_text,
        "top_features": top_features,
    }


def plot_feature_importance(feature_importance: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = feature_importance.sort_values("importance", ascending=True).tail(12)
    ax.barh(plot_df["original_feature"], plot_df["importance"], color="#7D3C98")
    ax.set_title(title)
    ax.set_xlabel("Aggregated importance")
    ax.set_ylabel("Original feature")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
