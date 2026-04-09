from __future__ import annotations

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_gain_curve(
    df: pd.DataFrame,
    predicted_cate: pd.Series,
    treatment_col: str = "treatment_binary",
    outcome_col: str = "outcome_numeric",
    true_cate_col: str | None = None,
) -> pd.DataFrame:
    ranked_df = df.copy().reset_index(drop=True)
    ranked_df["predicted_cate"] = np.asarray(predicted_cate, dtype=float)
    ranked_df = ranked_df.sort_values("predicted_cate", ascending=False).reset_index(drop=True)

    rows = []
    total_count = len(ranked_df)

    for end_index in range(1, total_count + 1):
        current = ranked_df.iloc[:end_index]
        if true_cate_col and true_cate_col in current.columns:
            gain = float(current[true_cate_col].sum())
            uplift = float(current[true_cate_col].mean())
        else:
            treated_outcomes = current.loc[current[treatment_col] == 1, outcome_col]
            control_outcomes = current.loc[current[treatment_col] == 0, outcome_col]
            if treated_outcomes.empty or control_outcomes.empty:
                uplift = 0.0
            else:
                uplift = float(treated_outcomes.mean() - control_outcomes.mean())
            gain = uplift * end_index

        rows.append(
            {
                "population_size": end_index,
                "population_fraction": end_index / total_count,
                "uplift": uplift,
                "gain": gain,
            }
        )

    curve_df = pd.DataFrame(rows)
    curve_df["random_gain"] = curve_df["population_fraction"] * curve_df["gain"].iloc[-1]
    return curve_df


def calculate_metrics(
    df: pd.DataFrame,
    predicted_cate: pd.Series,
    true_cate_col: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    curve_df = build_gain_curve(
        df=df,
        predicted_cate=predicted_cate,
        true_cate_col=true_cate_col,
    )

    metrics = {
        "average_predicted_cate": float(np.mean(predicted_cate)),
        "final_uplift": float(curve_df["uplift"].iloc[-1]),
        "final_gain": float(curve_df["gain"].iloc[-1]),
        "auuc_like": float(
            np.trapezoid(curve_df["gain"], curve_df["population_fraction"])
        ),
    }

    if true_cate_col and true_cate_col in df.columns:
        true_cate = df[true_cate_col].to_numpy(dtype=float)
        predicted_array = np.asarray(predicted_cate, dtype=float)
        metrics["cate_rmse"] = float(np.sqrt(np.mean((predicted_array - true_cate) ** 2)))
        gain_at_top = calculate_gain_at_top_fraction(predicted_array, true_cate, fraction=0.25)
        metrics.update(gain_at_top)
        metrics["normalized_auuc"] = calculate_normalized_auuc(
            df=df,
            predicted_cate=predicted_cate,
            true_cate_col=true_cate_col,
        )

    return metrics, curve_df


def calculate_gain_at_top_fraction(
    predicted_cate: np.ndarray,
    true_cate: np.ndarray,
    fraction: float = 0.25,
) -> dict[str, float]:
    top_count = max(1, int(len(predicted_cate) * fraction))
    top_indices = np.argsort(predicted_cate)[::-1][:top_count]
    top_gain = float(true_cate[top_indices].sum())
    random_gain = float(np.mean(true_cate) * top_count)
    return {
        f"gain_at_top_{int(fraction * 100)}pct": top_gain,
        "random_gain_top_segment": random_gain,
        "gain_vs_random_baseline": top_gain - random_gain,
        "average_true_cate_in_top_group": float(true_cate[top_indices].mean()),
    }


def calculate_normalized_auuc(
    df: pd.DataFrame,
    predicted_cate: pd.Series,
    true_cate_col: str,
) -> float:
    model_curve = build_gain_curve(df=df, predicted_cate=predicted_cate, true_cate_col=true_cate_col)
    perfect_curve = build_gain_curve(
        df=df.assign(predicted_cate=df[true_cate_col]),
        predicted_cate=df[true_cate_col],
        true_cate_col=true_cate_col,
    )
    model_area = np.trapezoid(model_curve["gain"], model_curve["population_fraction"])
    perfect_area = np.trapezoid(
        perfect_curve["gain"], perfect_curve["population_fraction"]
    )
    random_area = np.trapezoid(
        model_curve["random_gain"], model_curve["population_fraction"]
    )

    denominator = perfect_area - random_area
    if denominator == 0:
        return float("nan")
    return float((model_area - random_area) / denominator)


def plot_gain_curve(curve_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        curve_df["population_fraction"],
        curve_df["gain"],
        label="Model gain",
        linewidth=2.5,
        color="#145A32",
    )
    ax.plot(
        curve_df["population_fraction"],
        curve_df["random_gain"],
        label="Random baseline",
        linestyle="--",
        color="#B03A2E",
    )
    ax.set_title(title)
    ax.set_xlabel("Fraction of users targeted")
    ax.set_ylabel("Cumulative gain")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_cate_distribution(predicted_cate: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(predicted_cate, bins=30, color="#1F618D", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Predicted CATE")
    ax.set_ylabel("Number of users")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
